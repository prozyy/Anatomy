# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np

from common.arguments import parse_args
import torch
import pickle
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import os
import sys
import errno

from common.camera import *
from common.model import *
from common.loss import *
from common.bone import *
from common.generators import ChunkedGenerator, UnchunkedGenerator, randomaug_
from time import time, sleep
from common.utils import deterministic_random
import random
import os
import threading
from queue import Queue
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
#torch.backends.cudnn.benchmark=True
args = parse_args()
# args.boneindex = "16,15,15,14,13,12,12,11,10,9,9,8,8,7,8,11,8,14,7,0,3,2,2,1,6,5,5,4,1,0,4,0"
print(args)
device_list = [0]

try:
    # Create checkpoint directory if it does not exist
    os.makedirs(args.checkpoint)
except OSError as e:
    if e.errno != errno.EEXIST:
        raise RuntimeError('Unable to create checkpoint directory:', args.checkpoint)

#n joints, (n-1) bones, 2(n-1) indexs
print('Loading bone index...')
boneindextemp = args.boneindex.split(',')
boneindex = []
for i in range(0, len(boneindextemp), 2):
    boneindex.append([int(boneindextemp[i]), int(boneindextemp[i + 1])])

subjects_train = args.subjects_train.split(',')
subjects_test = args.subjects_test.split(',')

filter_widths = [int(x) for x in args.architecture.split(',')]
if not args.disable_optimizations and not args.dense and args.stride == 1:
    # Use optimized model for single-frame predictions
    model_pos_train = TemporalModelOptimized1f(args.num_joints_in, args.in_features, args.num_joints_out, boneindex, args.temperature, filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels)
else:
    # When incompatible settings are detected (stride > 1, dense filters, or disabled optimization) fall back to normal model
    model_pos_train = TemporalModel(args.num_joints_in,
                                    args.in_features,
                                    args.num_joints_out,
                                    boneindex,
                                    args.temperature,
                                    args.randnumtest,
                                    filter_widths=filter_widths,
                                    causal=args.causal,
                                    dropout=args.dropout,
                                    channels=args.channels,
                                    dense=args.dense)

model_pos = TemporalModel(args.num_joints_in, args.in_features, args.num_joints_out, boneindex, args.temperature, args.randnumtest, filter_widths=filter_widths, causal=args.causal, dropout=args.dropout, channels=args.channels, dense=args.dense)
# device_list = [i for i in range(torch.cuda.device_count())]
model_pos_train = nn.DataParallel(model_pos_train, device_ids=device_list)  # multi-GPU
model_pos = nn.DataParallel(model_pos, device_ids=device_list)  # multi-GPU

receptive_field = model_pos.module.receptive_field()
print('INFO: Receptive field: {} frames'.format(receptive_field))
pad = (receptive_field - 1) // 2  # Padding on each side
if args.causal:
    print('INFO: Using causal convolutions')
    causal_shift = pad
else:
    causal_shift = 0

model_params = 0
for parameter in model_pos.parameters():
    model_params += parameter.numel()
print('INFO: Trainable parameter count:', model_params)

if torch.cuda.is_available():
    model_pos = model_pos.cuda()
    model_pos_train = model_pos_train.cuda()

if args.resume or args.evaluate:
    chk_filename = os.path.join(args.checkpoint, args.resume if args.resume else args.evaluate)
    print('Loading checkpoint', chk_filename)
    checkpoint = torch.load(chk_filename, map_location=lambda storage, loc: storage)
    print('This model was trained for {} epochs'.format(checkpoint['epoch']))
    model_pos_train.load_state_dict(checkpoint['model_pos'])
    model_pos.load_state_dict(checkpoint['model_pos'])

if not args.render:
    print('Loading dataset...')
    dataset3d_path = 'data/data_3d_' + args.dataset + '.npz'
    dataset2d_path = 'data/data_2d_' + args.dataset + '_' + args.keypoints + '.npz'

    if args.dataset == 'h36m':
        from common.h36m_dataset import Human36mDataset
        dataset_train = Human36mDataset(dataset3d_path, dataset2d_path, chunk_length=args.stride, augment=args.data_augmentation, pad=pad, causal_shift=causal_shift, is_train=True)
        dataset_test = Human36mDataset(dataset3d_path, dataset2d_path, chunk_length=args.stride, augment=False, pad=pad, causal_shift=causal_shift, is_train=False)
    elif str(args.dataset).startswith("h36m_24"):
        from common.h36m_24_dataset import Human36m_24Dataset
        dataset_train = Human36m_24Dataset(dataset3d_path, dataset2d_path, chunk_length=args.stride, augment=args.data_augmentation, pad=pad, causal_shift=causal_shift, is_train=True)
        dataset_test = Human36m_24Dataset(dataset3d_path, dataset2d_path, chunk_length=args.stride, augment=False, pad=pad, causal_shift=causal_shift, is_train=False)
    else:
        raise KeyError('Invalid dataset')

if not args.evaluate:
    trainDataLoader = DataLoader(dataset_train, batch_size=args.batch_size * len(device_list), shuffle=True, num_workers=16, pin_memory=True)
    testDataLoader = DataLoader(dataset_test, batch_size=64 * len(device_list), shuffle=False, num_workers=16, pin_memory=True)

    lr = args.learning_rate

    optimizer = optim.Adam(model_pos_train.parameters(), lr=lr, amsgrad=True)

    lr_decay = args.lr_decay

    losses_3d_train = []
    losses_3d_train_eval = []
    losses_3d_valid = []

    epoch = 0
    initial_momentum = 0.1
    final_momentum = 0.001

    if args.resume:
        epoch = checkpoint['epoch']
        if 'optimizer' in checkpoint and checkpoint['optimizer'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer'])
        else:
            print('WARNING: this checkpoint does not contain an optimizer state. The optimizer will be reinitialized.')

        lr = checkpoint['lr']

    print('** Note: reported losses are averaged over all frames and test-time augmentation is not used here.')
    print('** The final evaluation will be carried out after the last training epoch.')

    batch_num = len(trainDataLoader)

    # Pos model only
    while epoch < args.epochs:
        start_time = time()
        epoch_loss_3d_train = 0
        N = 0
        model_pos_train.train()
        batchid = 0
        let = time()
        tqBar = tqdm(trainDataLoader, ncols=200)
        for label, cam, inputs_3d, inputs_2d, inputs_3d_rand, inputs_2d_rand in tqBar:
            if torch.cuda.is_available():
                cam = cam.cuda()
                inputs_3d = inputs_3d.cuda()
                inputs_2d = inputs_2d.cuda()
                inputs_3d_rand = inputs_3d_rand.cuda()
                inputs_2d_rand = inputs_2d_rand.cuda()

            data_time = time()

            inputs_3d[:, :, 0] = 0
            inputs_3d_rand[:, :, 0] = 0
            if args.dataset == 'h36m':
                batch_3d_randaug, inputs_3d_lengthnew = randomaug_cuda(inputs_3d_rand, boneindex, args.augdegree)
            elif str(args.dataset).startswith("h36m_24"):
                batch_3d_randaug, inputs_3d_lengthnew = randomaug_cuda_24kpt(inputs_3d_rand, boneindex, args.augdegree)

            #also re-sampled the trajectory
            randomtraj = torch.normal(0.0, 0.5, size=(inputs_3d.size(0), args.randnum, 1, 3)).cuda()
            randomtraj[:, :, :, 2] = randomtraj[:, :, :, 2] + 5
            randomtraj[:, :, :, 1] = randomtraj[:, :, :, 1] - 0.3
            # the new 3D + trajectory will enable the reconstruction of augmented 2D input (do it in the main function)
            inputs_3d_randaugtraj = batch_3d_randaug + randomtraj
            inputs_3d_randauggt = batch_3d_randaug
            aut_time = time()
            # print(" aug time:",aut_time - data_time)

            optimizer.zero_grad()

            projection_func = project_to_2d_linear if args.linear_projection else project_to_2d
            #reconstruct the augmented 2D input
            inputs_2d_randaug = projection_func(inputs_3d_randaugtraj, cam)
            #bone length prediction network doesn't need the visibility score feature as input
            inputs_2d_rand = inputs_2d_rand[:, :, :, :2].contiguous()
            #get the grounth-truth bone length (b * nb), directly average the frames since bone length is consistent across frame
            inputs_3d_length = getbonelength(inputs_3d, boneindex).mean(1)

            predicted_3d_pos, bonelength, predicted_3d_rand, bonelengthaug, predicted_3d_randaug, bonedirect_2, bonedirect_1, predicted_js_2, predicted_js_1 = model_pos_train(inputs_2d, inputs_2d_rand, inputs_2d_randaug)

            bonedirect_2 = bonedirect_2.view(bonedirect_2.size(0), -1, 3)
            bonedirect_1 = bonedirect_1.view(bonedirect_1.size(0), -1, 3)
            #get the gt 3D joint locations of the current frame
            inputs_3d = inputs_3d[:, int((inputs_3d.size(1) - 1) / 2):int((inputs_3d.size(1) - 1) / 2) + 1]
            #compute the bone direction loss of each sub-network (totally 2) of the bone direction prediction network
            inputs_3d_direct = getbonedirect(inputs_3d, boneindex)
            loss_direct = args.wd * torch.pow(inputs_3d_direct - bonedirect_2, 2).sum(2).mean() + args.wd * args.snd * torch.pow(inputs_3d_direct - bonedirect_1, 2).sum(2).mean()
            #compute the mpjpe loss of the final 3D joint location prediction
            loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
            #compute the relative joint shifts loss
            inputs_3d_js = getbonejs(inputs_3d, boneindex)
            loss_js = args.wjs * mpjpe(predicted_js_2, inputs_3d_js) + args.wjs * args.snd * mpjpe(predicted_js_1, inputs_3d_js)

            #randomly sample one frame to compute the mpjpe loss of the bone length prediction network
            randnum = random.randint(0, predicted_3d_rand.size(1) - 1)
            #compute the mpjpe loss of the bone length prediction network (original + augmented, you can also remove the mpjpe loss of the augmented data as described in the paper)
            loss_3d_pos_rand = mpjpe(predicted_3d_rand[:, randnum:randnum + 1], inputs_3d_rand[:, randnum:randnum + 1])
            loss_3d_pos_randaug = mpjpe(predicted_3d_randaug[:, randnum:randnum + 1], inputs_3d_randauggt[:, randnum:randnum + 1])
            #compute bone length loss (original + augmented)
            loss_length = args.wl * torch.pow(inputs_3d_length - bonelength, 2).mean()
            loss_lengthaug = args.wl * torch.pow(inputs_3d_lengthnew - bonelengthaug, 2).mean()
            #total loss of the bone length prediction network
            loss_len = loss_3d_pos_rand + loss_3d_pos_randaug + loss_length + loss_lengthaug
            epoch_loss_3d_train += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
            N += inputs_3d.shape[0] * inputs_3d.shape[1]
            #total loss of the model
            loss_total = loss_3d_pos + loss_len + loss_direct + loss_js
            loss_total.backward()
            optimizer.step()

            et = time()
            batchid += 1
            tqBar.set_description("epoch :{}/{} \t train batch :{}/{} \t data time:{:.5f} \t forward_backward time:{:.5f} \t loss_total:{:.5f}".format(epoch, args.epochs, batchid, batch_num, data_time - let, et - data_time, loss_total.item()))
            let = time()
        print("")
        losses_3d_train.append(epoch_loss_3d_train / N)
        torch.cuda.empty_cache()

        # End-of-epoch evaluation
        if (epoch + 1) % args.eva_frequency == 0:
            torch.cuda.empty_cache()
            with torch.no_grad():
                model_pos.load_state_dict(model_pos_train.state_dict())
                model_pos.eval()
                epoch_loss_3d_valid = 0
                N = 0
                if not args.no_eval:
                    # Evaluate on test set
                    for label, cam, batch, batch_2d in tqdm(testDataLoader):
                        if torch.cuda.is_available():
                            inputs_3d = batch.cuda()
                            inputs_2d = batch_2d.cuda()
                        inputs_3d[:, :, 0] = 0
                        # Predict 3D poses
                        predicted_3d_pos = model_pos(inputs_2d)
                        pad = int((inputs_3d.size(1) - predicted_3d_pos.size(1)) / 2)
                        inputs_3d = inputs_3d[:, pad:inputs_3d.size(1) - pad]
                        loss_3d_pos = mpjpe(predicted_3d_pos, inputs_3d)
                        epoch_loss_3d_valid += inputs_3d.shape[0] * inputs_3d.shape[1] * loss_3d_pos.item()
                        N += inputs_3d.shape[0] * inputs_3d.shape[1]

                    losses_3d_valid.append(epoch_loss_3d_valid / N)
                    torch.cuda.empty_cache()

        elapsed = (time() - start_time) / 60

        if args.no_eval or (epoch + 1) % args.eva_frequency != 0:
            print('[%d] time %.2f lr %f 3d_train %f' % (epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000))
        else:
            print('[%d] time %.2f lr %f 3d_train %f 3d_valid %f' % (epoch + 1, elapsed, lr, losses_3d_train[-1] * 1000, losses_3d_valid[-1] * 1000))

        # Decay learning rate exponentially
        lr *= lr_decay
        for param_group in optimizer.param_groups:
            param_group['lr'] *= lr_decay
        epoch += 1

        # Decay BatchNorm momentum
        momentum = initial_momentum * np.exp(-epoch / args.epochs * np.log(initial_momentum / final_momentum))
        model_pos_train.module.set_bn_momentum(momentum)

        # Save checkpoint if necessary
        if epoch % args.checkpoint_frequency == 0:
            chk_path = os.path.join(args.checkpoint, 'epochfinal_{}.bin'.format(epoch))
            print('Saving checkpoint to', chk_path)

            torch.save({
                'epoch': epoch,
                'lr': lr,
                'optimizer': optimizer.state_dict(),
                'model_pos': model_pos_train.state_dict(),
            }, chk_path)

from common.skeleton import Skeleton
from common.visualization import  render_animation
data_skeleton = Skeleton(parents=[-1,0,1,2,2,3,4,1,1,7,8,9,10,0,0,13,14,15,16,17,18,17,18,1],
                         joints_left=[3, 5, 7, 9, 11, 13,15,17,19,21],
                         joints_right=[4,6,8,10,12,14,16,18,20,22])

test_info = {
    "fps":30,
    "res_w":1200,
    "res_h":1780,
    "receptive_field":243,
    "num_joints":24,
    "azimuth":0,
    "rot":np.array([0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],dtype='float32')
}

def eval_data_prepare(inputs_2d,receptive_field = test_info["receptive_field"]):
    inputs_2d_p = torch.squeeze(inputs_2d)
    out_num = inputs_2d_p.shape[0] - receptive_field + 1
    eval_input_2d = torch.empty(out_num, receptive_field, inputs_2d_p.shape[1], inputs_2d_p.shape[2])
    for i in range(out_num):
        eval_input_2d[i,:,:,:] = inputs_2d_p[i:i+receptive_field, :, :]
    return eval_input_2d

# Evaluate
def evaluate(batch_2d):
    with torch.no_grad():
        model_pos.eval()
        inputs_2d = torch.from_numpy(batch_2d.astype('float32'))

        ##### convert size
        inputs_2d = eval_data_prepare(inputs_2d)

        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()

        predicted_3d_pos = model_pos(inputs_2d)

        return predicted_3d_pos.squeeze(0).cpu().numpy()

keypoints_metadata = {'layout_name': 'coco', 'num_joints': 24, 'keypoints_symmetry': [[1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 22], [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 23]]}
joints_left, joints_right = list(data_skeleton.joints_left()), list(data_skeleton.joints_right())        
if args.render:
    input_keypoints = np.load("data/testData/test.npy").transpose(0,2,1)
    input_keypoints[..., :2] = normalize_screen_coordinates(input_keypoints[..., :2], w=test_info['res_w'], h=test_info['res_h'])
    # normalize_screen_coordinates
    pad = (test_info["receptive_field"] -1) // 2 # Padding on each side
    batch_2d = np.expand_dims(np.pad(input_keypoints,((pad, pad), (0, 0), (0, 0)),'edge'), axis=0)
    print(batch_2d.shape)
    prediction = evaluate(batch_2d)
    
    if args.viz_output is not None:
        # Invert camera transformation
        prediction = camera_to_world(prediction, R=test_info["rot"], t=0)
        # We don't have the trajectory, but at least we can rebase the height
        prediction[:, :,:, 2] -= np.min(prediction[:, :,:, 2])
        anim_output = {'Reconstruction': prediction}
        input_keypoints = image_coordinates(input_keypoints[..., :2], w=test_info['res_w'], h=test_info['res_h'])
        render_animation(input_keypoints, keypoints_metadata, anim_output,
                         data_skeleton, test_info['fps'], args.viz_bitrate, test_info['azimuth'], args.viz_output,
                         limit=args.viz_limit, downsample=args.viz_downsample, size=args.viz_size,
                         input_video_path=args.viz_video, viewport=(test_info['res_w'], test_info['res_h']),
                         input_video_skip=args.viz_skip)
else:
    print('Evaluating...')
    testDataLoader = DataLoader(dataset_test, batch_size=1024 * len(device_list), shuffle=False, num_workers=16, pin_memory=True)
    with torch.no_grad():
        model_pos.eval()
        N = 0
        epoch_loss_3d_pos = {}
        for label, cam, batch, batch_2d in tqdm(testDataLoader):
            if torch.cuda.is_available():
                inputs_3d = batch.cuda()
                inputs_2d = batch_2d.cuda()
            inputs_3d[:, :, 0] = 0
            # Predict 3D poses
            predicted_3d_pos = model_pos(inputs_2d)
            pad = int((inputs_3d.size(1) - predicted_3d_pos.size(1)) / 2)
            inputs_3d = inputs_3d[:, pad:inputs_3d.size(1) - pad]
            mpjpe_error = mpjpe(predicted_3d_pos, inputs_3d, True)
            n_mpjpe_error = n_mpjpe(predicted_3d_pos, inputs_3d, True)

            inputs = inputs_3d.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            predicted_3d_pos = predicted_3d_pos.cpu().numpy().reshape(-1, inputs_3d.shape[-2], inputs_3d.shape[-1])
            p_mpjpe_error = p_mpjpe(predicted_3d_pos, inputs, True)
            # Compute velocity error
            # velocity_error = mean_velocity_error(predicted_3d_pos, inputs,True)
            for index, subject_action_camIndex in enumerate(label):
                key = subject_action_camIndex.split("_")[1].split(" ")[0] if not args.by_subject else subject_action_camIndex.split("_")[0]
                if key in epoch_loss_3d_pos:
                    epoch_loss_3d_pos[key]["mpjpe"] += mpjpe_error[index].cpu().numpy()
                    epoch_loss_3d_pos[key]["p-mpjpe"] += p_mpjpe_error[index]
                    epoch_loss_3d_pos[key]["n_mpjpe"] += n_mpjpe_error[index].cpu().numpy()
                    epoch_loss_3d_pos[key]["count"] += 1
                else:
                    epoch_loss_3d_pos[key] = {"mpjpe": mpjpe_error[index].cpu().numpy(), "p-mpjpe": p_mpjpe_error[index], "n_mpjpe": n_mpjpe_error[index].cpu().numpy(), "count": 1}

        e1_list, e2_list, e3_list = [], [], []

        import prettytable as pt
        tb = pt.PrettyTable()
        tb_names = [" "]
        tb_row1 = ["mpjpe"]
        tb_row2 = ["p-mpjpe"]
        tb_row3 = ["n_mpjpe"]

        for key, errorInfo in epoch_loss_3d_pos.items():
            e1 = errorInfo["mpjpe"] / errorInfo["count"] * 1000
            e2 = errorInfo["p-mpjpe"] / errorInfo["count"] * 1000
            e3 = errorInfo["n_mpjpe"] / errorInfo["count"] * 1000
            e1_list.append(e1)
            e2_list.append(e2)
            e3_list.append(e3)

            tb_names.append(key)
            tb_row1.append('%.2f' % e1)
            tb_row2.append('%.2f' % e2)
            tb_row3.append('%.2f' % e3)

        tb_names.append("AVG")
        tb_row1.append('%.2f' % np.mean(np.array(e1_list)))
        tb_row2.append('%.2f' % np.mean(np.array(e2_list)))
        tb_row3.append('%.2f' % np.mean(np.array(e3_list)))

        torch.cuda.empty_cache()
        tb.field_names = tb_names
        tb.add_row(tb_row1)
        tb.add_row(tb_row2)
        tb.add_row(tb_row3)
        print(tb)
