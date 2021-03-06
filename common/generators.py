# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

from itertools import zip_longest
import numpy as np
import random


def getbonedirect(seq, boneindex):
    bs = np.shape(seq)[0]
    ss = np.shape(seq)[1]
    seq = np.reshape(seq,(bs*ss,-1,3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[0]] - seq[:,index[1]])
    bonedirect = np.stack(bone,1)
    bonesum = np.expand_dims(np.power(np.power(bonedirect,2).sum(2),0.5),2)
    bonedirect = bonedirect/(bonesum + 1e-6)
    bonedirect = np.reshape(bonedirect, (bs,ss,np.shape(bonedirect)[1],np.shape(bonedirect)[2]))
    return bonedirect



def getbone(seq, boneindex):
    bs = np.shape(seq)[0]
    ss = np.shape(seq)[1]
    seq = np.reshape(seq,(bs*ss,-1,3))
    bone = []
    for index in boneindex:
        bone.append(seq[:,index[0]] - seq[:,index[1]])
    bone = np.stack(bone,1)
    bone = np.power(np.power(bone,2).sum(2),0.5)
    bone = np.reshape(bone, (bs,ss,np.shape(bone)[1]))
    return bone

'''
sample new bone lengths and adjust the 3D gt of each frame based on them,
'''
def randomaug_(batch_3D_rand_ori, boneindex, augdegree):
    bs = np.shape(batch_3D_rand_ori)[0]
    ss = np.shape(batch_3D_rand_ori)[1]
    bonelen = getbone(batch_3D_rand_ori, boneindex).mean(1)
    bonelenmean = bonelen.mean(0)
    #sample new bone lengths
    randadd = (np.random.rand(bs,16)-0.5) * (bonelenmean * augdegree)
    bonelennew = bonelen + randadd
    bonedirect = getbonedirect(batch_3D_rand_ori, boneindex)
    '''
    if you experiment with another dataset, temporally in this version you need to manually modify the below indexs to re-compute the gt "3D joint location" based on the re-sampled bone lengths,
    because the change of a specific bone length can lead to the changes of multiple joints' location  
    '''       
    b = randadd[:,0]
    batch_3D_rand_ori[:,:,16:17] = batch_3D_rand_ori[:,:,16:17] + np.expand_dims(bonedirect[:,:,0] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,1]
    batch_3D_rand_ori[:,:,15:17] = batch_3D_rand_ori[:,:,15:17] + np.expand_dims(bonedirect[:,:,1] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,2]
    batch_3D_rand_ori[:,:,13:14] = batch_3D_rand_ori[:,:,13:14] + np.expand_dims(bonedirect[:,:,2] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,3]
    batch_3D_rand_ori[:,:,12:14] = batch_3D_rand_ori[:,:,12:14] + np.expand_dims(bonedirect[:,:,3] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,4]
    batch_3D_rand_ori[:,:,10:11] = batch_3D_rand_ori[:,:,10:11] + np.expand_dims(bonedirect[:,:,4] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,5]
    batch_3D_rand_ori[:,:,9:11] = batch_3D_rand_ori[:,:,9:11] + np.expand_dims(bonedirect[:,:,5] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,6]
    batch_3D_rand_ori[:,:,8:17] = batch_3D_rand_ori[:,:,8:17] + np.expand_dims(bonedirect[:,:,6] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2) 
    b = randadd[:,7]
    batch_3D_rand_ori[:,:,11:14] = batch_3D_rand_ori[:,:,11:14] - np.expand_dims(bonedirect[:,:,7] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,8]
    batch_3D_rand_ori[:,:,14:17] = batch_3D_rand_ori[:,:,14:17] - np.expand_dims(bonedirect[:,:,8] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,9]
    batch_3D_rand_ori[:,:,7:] = batch_3D_rand_ori[:,:,7:] + np.expand_dims(bonedirect[:,:,9] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,10]
    batch_3D_rand_ori[:,:,3:4] = batch_3D_rand_ori[:,:,3:4] + np.expand_dims(bonedirect[:,:,10] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,11]
    batch_3D_rand_ori[:,:,2:4] = batch_3D_rand_ori[:,:,2:4] + np.expand_dims(bonedirect[:,:,11] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,12]
    batch_3D_rand_ori[:,:,6:7] = batch_3D_rand_ori[:,:,6:7] + np.expand_dims(bonedirect[:,:,12] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,13]
    batch_3D_rand_ori[:,:,5:7] = batch_3D_rand_ori[:,:,5:7] + np.expand_dims(bonedirect[:,:,13] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,14]
    batch_3D_rand_ori[:,:,1:4] = batch_3D_rand_ori[:,:,1:4] + np.expand_dims(bonedirect[:,:,14] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,15]
    batch_3D_rand_ori[:,:,4:7] = batch_3D_rand_ori[:,:,4:7] + np.expand_dims(bonedirect[:,:,15] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    return batch_3D_rand_ori, bonelennew


'''
sample new bone lengths and adjust the 3D gt of each frame based on them,
'''
def randomaug(batch_3D_rand_ori, boneindex, augdegree):
    bs = np.shape(batch_3D_rand_ori)[0]
    ss = np.shape(batch_3D_rand_ori)[1]
    bonelen = getbone(batch_3D_rand_ori, boneindex).mean(1)
    bonelenmean = bonelen.mean(0)
    #sample new bone lengths
    randadd = (np.random.rand(bs,23)-0.5) * (bonelenmean * augdegree)
    bonelennew = bonelen + randadd
    bonedirect = getbonedirect(batch_3D_rand_ori, boneindex)
    '''
    if you experiment with another dataset, temporally in this version you need to manually modify the below indexs to re-compute the gt "3D joint location" based on the re-sampled bone lengths,
    because the change of a specific bone length can lead to the changes of multiple joints' location  
    '''       
    b = randadd[:,0]
    batch_3D_rand_ori[:,:,12:13] = batch_3D_rand_ori[:,:,12:13] + np.expand_dims(bonedirect[:,:,0] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,1]
    batch_3D_rand_ori[:,:,[12,10]] = batch_3D_rand_ori[:,:,[12,10]] + np.expand_dims(bonedirect[:,:,1] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,2]
    batch_3D_rand_ori[:,:,11:12] = batch_3D_rand_ori[:,:,11:12] + np.expand_dims(bonedirect[:,:,2] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,3]
    batch_3D_rand_ori[:,:,[9,11]] = batch_3D_rand_ori[:,:,[9,11]] + np.expand_dims(bonedirect[:,:,3] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,4]
    batch_3D_rand_ori[:,:,5:6] = batch_3D_rand_ori[:,:,5:6] + np.expand_dims(bonedirect[:,:,4] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,5]
    batch_3D_rand_ori[:,:,[5,3]] = batch_3D_rand_ori[:,:,[5,3]] + np.expand_dims(bonedirect[:,:,5] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,6]
    batch_3D_rand_ori[:,:,6:7] = batch_3D_rand_ori[:,:,6:7] + np.expand_dims(bonedirect[:,:,6] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2) 
    b = randadd[:,7]
    batch_3D_rand_ori[:,:,[4,6]] = batch_3D_rand_ori[:,:,[4,6]] - np.expand_dims(bonedirect[:,:,7] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,8]
    batch_3D_rand_ori[:,:,2:7] = batch_3D_rand_ori[:,:,2:7] - np.expand_dims(bonedirect[:,:,8] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,9]
    batch_3D_rand_ori[:,:,23:24] = batch_3D_rand_ori[:,:,23:24] + np.expand_dims(bonedirect[:,:,9] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,10]
    batch_3D_rand_ori[:,:,[7,9,11]] = batch_3D_rand_ori[:,:,[7,9,11]] + np.expand_dims(bonedirect[:,:,10] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,11]
    batch_3D_rand_ori[:,:,[8,10,12]] = batch_3D_rand_ori[:,:,[8,10,12]] + np.expand_dims(bonedirect[:,:,11] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,12]
    batch_3D_rand_ori[:,:,1:12] = batch_3D_rand_ori[:,:,1:12] + np.expand_dims(bonedirect[:,:,12] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    batch_3D_rand_ori[:,:,23:24] = batch_3D_rand_ori[:,:,23:24] + np.expand_dims(bonedirect[:,:,12] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,13]
    batch_3D_rand_ori[:,:,20:21] = batch_3D_rand_ori[:,:,20:21] + np.expand_dims(bonedirect[:,:,13] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,14]
    batch_3D_rand_ori[:,:,22:23] = batch_3D_rand_ori[:,:,22:23] + np.expand_dims(bonedirect[:,:,14] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,15]
    batch_3D_rand_ori[:,:,[18,20,22]] = batch_3D_rand_ori[:,:,[18,20,22]] + np.expand_dims(bonedirect[:,:,15] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,16]
    batch_3D_rand_ori[:,:,[16,18,20,22]] = batch_3D_rand_ori[:,:,[16,18,20,22]] + np.expand_dims(bonedirect[:,:,16] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,17]
    batch_3D_rand_ori[:,:,19:20] = batch_3D_rand_ori[:,:,19:20] + np.expand_dims(bonedirect[:,:,17] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,18]
    batch_3D_rand_ori[:,:,21:22] = batch_3D_rand_ori[:,:,21:22] + np.expand_dims(bonedirect[:,:,18] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,19]
    batch_3D_rand_ori[:,:,[17,19,21]] = batch_3D_rand_ori[:,:,[17,19,21]] + np.expand_dims(bonedirect[:,:,19] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,20]
    batch_3D_rand_ori[:,:,[15,17,19,21]] = batch_3D_rand_ori[:,:,[15,17,19,21]] + np.expand_dims(bonedirect[:,:,20] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,21]
    batch_3D_rand_ori[:,:,[13,15,17,19,21]] = batch_3D_rand_ori[:,:,[13,15,17,19,21]] + np.expand_dims(bonedirect[:,:,21] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    b = randadd[:,22]
    batch_3D_rand_ori[:,:,[14,16,18,20,22]] = batch_3D_rand_ori[:,:,[14,16,18,20,22]] + np.expand_dims(bonedirect[:,:,22] * np.tile(np.expand_dims(np.expand_dims(b,1),2),(1,ss,3)),2)
    
    return batch_3D_rand_ori, bonelennew


class ChunkedGenerator:
    """
    Batched data generator, used for training.
    The sequences are split into equal-length chunks and padded as necessary.
    
    Arguments:
    batch_size -- the batch size to use for training
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    chunk_length -- number of output frames to predict for each training example (usually 1)
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    shuffle -- randomly shuffle the dataset before each epoch
    random_seed -- initial seed to use for the random generator
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    randnum -- number of randomly sampled frames for bone length prediction
    boneindex -- bone index (each two indexs correspond to the two joints a bone)
    """
    def __init__(self, batch_size, cameras, poses_3d, poses_2d, randnum, boneindex, augdegree,
                 chunk_length, pad=0, causal_shift=0,
                 shuffle=True, random_seed=1234,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None,
                 endless=False):
        assert poses_3d is None or len(poses_3d) == len(poses_2d), (len(poses_3d), len(poses_2d))
        assert cameras is None or len(cameras) == len(poses_2d)
    
        # Build lineage info
        pairs = [] # (seq_idx, start_frame, end_frame, flip) tuples
        for i in range(len(poses_2d)):
            assert poses_3d is None or poses_3d[i].shape[0] == poses_3d[i].shape[0]
            n_chunks = (poses_2d[i].shape[0] + chunk_length - 1) // chunk_length
            offset = (n_chunks * chunk_length - poses_2d[i].shape[0]) // 2
            bounds = np.arange(n_chunks+1)*chunk_length - offset
            augment_vector = np.full(len(bounds - 1), False, dtype=bool)
            pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], augment_vector)
            if augment:
                pairs += zip(np.repeat(i, len(bounds - 1)), bounds[:-1], bounds[1:], ~augment_vector)

        # Initialize buffers
        if cameras is not None:
            self.batch_cam = np.empty((batch_size, cameras[0].shape[-1]))
        if poses_3d is not None:
            self.batch_3d = np.empty((batch_size, chunk_length + 2*pad, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
        self.batch_2d = np.empty((batch_size, chunk_length + 2*pad, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
        #randomly sampled 2D input for bone length prediction network (typically b * 50 * 17 * 2) 
        self.batch_2d_rand = np.empty((batch_size, randnum, poses_2d[0].shape[-2], poses_2d[0].shape[-1]))
        #randomly sampled 3D ground truth for bone length prediction network (typically b * 50 * 17 * 3)
        self.batch_3d_rand = np.empty((batch_size, randnum, poses_3d[0].shape[-2], poses_3d[0].shape[-1]))
    
        self.num_batches = (len(pairs) + batch_size - 1) // batch_size
        self.batch_size = batch_size
        self.random = np.random.RandomState(random_seed)
        self.pairs = pairs
        self.shuffle = shuffle
        self.pad = pad
        self.causal_shift = causal_shift
        self.endless = endless
        self.state = None
        
        self.cameras = cameras
        self.poses_3d = poses_3d
        self.poses_2d = poses_2d
        
        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.randnum = randnum
        self.boneindex = boneindex
        self.augdegree = augdegree
    def num_frames(self):
        return self.num_batches * self.batch_size
    
    def random_state(self):
        return self.random
    
    def set_random_state(self, random):
        self.random = random
        
    def augment_enabled(self):
        return self.augment
    
    def next_pairs(self):
        if self.state is None:
            if self.shuffle:
                pairs = self.random.permutation(self.pairs)
            else:
                pairs = self.pairs
            return 0, pairs
        else:
            return self.state
    
    def next_epoch(self):
        enabled = True
        while enabled:
            start_idx, pairs = self.next_pairs()
            for b_i in range(start_idx, self.num_batches):
                chunks = pairs[b_i*self.batch_size : (b_i+1)*self.batch_size]
                for i, (seq_i, start_3d, end_3d, flip) in enumerate(chunks):
                    start_2d = start_3d - self.pad - self.causal_shift
                    end_2d = end_3d + self.pad - self.causal_shift
                    '''
                    First we need to obtain the consecutive frames' 2D input/3D gt for the bone direction prediction network
                    '''
                    # 2D poses
                    seq_2d = self.poses_2d[seq_i]
                    low_2d = max(start_2d, 0)
                    high_2d = min(end_2d, seq_2d.shape[0])
                    pad_left_2d = low_2d - start_2d
                    pad_right_2d = end_2d - high_2d
                    if pad_left_2d != 0 or pad_right_2d != 0:
                        self.batch_2d[i] = np.pad(seq_2d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                        
                    else:
                        self.batch_2d[i] = seq_2d[low_2d:high_2d]

                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d[i, :, :, 0] *= -1
                        self.batch_2d[i, :, self.kps_left + self.kps_right] = self.batch_2d[i, :, self.kps_right + self.kps_left]

                    # 3D poses
                    if self.poses_3d is not None:
                        seq_3d = self.poses_3d[seq_i]
                        low_2d = max(start_2d, 0)
                        high_2d = min(end_2d, seq_2d.shape[0])
                        pad_left_2d = low_2d - start_2d
                        pad_right_2d = end_2d - high_2d
                        if pad_left_2d != 0 or pad_right_2d != 0:
                            self.batch_3d[i] = np.pad(seq_3d[low_2d:high_2d], ((pad_left_2d, pad_right_2d), (0, 0), (0, 0)), 'edge')
                        else:
                            self.batch_3d[i] = seq_3d[low_2d:high_2d]

                        if flip:
                            # Flip 3D joints
                            self.batch_3d[i, :, :, 0] *= -1
                            self.batch_3d[i, :, self.joints_left + self.joints_right] = \
                                    self.batch_3d[i, :, self.joints_right + self.joints_left]

                    # Cameras
                    if self.cameras is not None:
                        self.batch_cam[i] = self.cameras[seq_i]
                        if flip:
                            # Flip horizontal distortion coefficients
                            self.batch_cam[i, 2] *= -1
                            self.batch_cam[i, 7] *= -1
                    '''
                    Now we need to obtain the randomly sampled frames' 2D input/3D gt for the bone length prediction network
                    '''
                    #randomly sample 'randnum' frames' 2D/3D of the video
                    rand_no = list(range(len(seq_2d)))
                    rand_no = random.sample(rand_no,self.randnum)
                    seq_2d_rand = seq_2d[rand_no]
                    seq_3d_rand = seq_3d[rand_no]
                    self.batch_2d_rand[i] = seq_2d_rand
                    self.batch_3d_rand[i] = seq_3d_rand
                    if flip:
                        # Flip 2D keypoints
                        self.batch_2d_rand[i, :, :, 0] *= -1
                        self.batch_2d_rand[i, :, self.kps_left + self.kps_right] = self.batch_2d_rand[i, :, self.kps_right + self.kps_left]
                        self.batch_3d_rand[i, :, :, 0] *= -1
                        self.batch_3d_rand[i, :, self.joints_left + self.joints_right] = \
                                    self.batch_3d_rand[i, :, self.joints_right + self.joints_left]
                    
                if self.endless:
                    self.state = (b_i + 1, pairs)
                if self.poses_3d is None and self.cameras is None:
                    yield None, None, self.batch_2d[:len(chunks)]

                elif self.poses_3d is None:
                    yield self.batch_cam[:len(chunks)], None, self.batch_2d[:len(chunks)]
                else:
                    '''
                    create augmented 2D/3D with different bone lengths for training
                    '''
                    self.batch_3d_rand[:, :, 0] = 0
                    #first adjust the 3D gt of each frame based on the new sampled bone lengths
                    self.batch_3d_randaug, self.bonelennew = randomaug(self.batch_3d_rand[:len(chunks)], self.boneindex, self.augdegree)
                    #also re-sampled the trajectory
                    randomtraj = np.random.normal(loc=0.0, scale=0.5, size=(len(chunks), self.randnum, 1, 3))
                    randomtraj[:,:,:,2] = randomtraj[:,:,:,2] + 5
                    randomtraj[:,:,:,1] = randomtraj[:,:,:,1] -0.3
                    # the new 3D + trajectory will enable the reconstruction of augmented 2D input (do it in the main function)
                    self.batch_3d_randaugtraj = self.batch_3d_randaug + randomtraj
                    if self.cameras is None:
                        yield None, self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)], self.batch_3d_rand[:len(chunks)], self.batch_2d_rand[:len(chunks)], self.batch_3d_randaugtraj, self.bonelennew, self.batch_3d_randaug
                    else:
                        yield self.batch_cam[:len(chunks)], self.batch_3d[:len(chunks)], self.batch_2d[:len(chunks)], self.batch_3d_rand[:len(chunks)], self.batch_2d_rand[:len(chunks)], self.batch_3d_randaugtraj, self.bonelennew, self.batch_3d_randaug
            
            if self.endless:
                self.state = None
            else:
                enabled = False
            

class UnchunkedGenerator:
    """
    Non-batched data generator, used for testing.
    Sequences are returned one at a time (i.e. batch size = 1), without chunking.
    
    If data augmentation is enabled, the batches contain two sequences (i.e. batch size = 2),
    the second of which is a mirrored version of the first.
    
    Arguments:
    cameras -- list of cameras, one element for each video (optional, used for semi-supervised training)
    poses_3d -- list of ground-truth 3D poses, one element for each video (optional, used for supervised training)
    poses_2d -- list of input 2D keypoints, one element for each video
    pad -- 2D input padding to compensate for valid convolutions, per side (depends on the receptive field)
    causal_shift -- asymmetric padding offset when causal convolutions are used (usually 0 or "pad")
    augment -- augment the dataset by flipping poses horizontally
    kps_left and kps_right -- list of left/right 2D keypoints if flipping is enabled
    joints_left and joints_right -- list of left/right 3D joints if flipping is enabled
    """
    
    def __init__(self, cameras, poses_3d, poses_2d, pad=0, causal_shift=0,
                 augment=False, kps_left=None, kps_right=None, joints_left=None, joints_right=None):
        assert poses_3d is None or len(poses_3d) == len(poses_2d)
        assert cameras is None or len(cameras) == len(poses_2d)

        self.augment = augment
        self.kps_left = kps_left
        self.kps_right = kps_right
        self.joints_left = joints_left
        self.joints_right = joints_right
        
        self.pad = pad
        self.causal_shift = causal_shift
        self.cameras = [] if cameras is None else cameras
        self.poses_3d = [] if poses_3d is None else poses_3d
        self.poses_2d = poses_2d
        
    def num_frames(self):
        count = 0
        for p in self.poses_2d:
            count += p.shape[0]
        return count
    
    def augment_enabled(self):
        return self.augment
    
    def set_augment(self, augment):
        self.augment = augment
    
    def next_epoch(self):
        for seq_cam, seq_3d, seq_2d in zip_longest(self.cameras, self.poses_3d, self.poses_2d):
            batch_cam = None if seq_cam is None else np.expand_dims(seq_cam, axis=0)
            batch_3d = None if seq_3d is None else np.expand_dims(np.pad(seq_3d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            batch_2d = np.expand_dims(np.pad(seq_2d,
                            ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0), (0, 0)),
                            'edge'), axis=0)
            if self.augment:
                # Append flipped version
                if batch_cam is not None:
                    batch_cam = np.concatenate((batch_cam, batch_cam), axis=0)
                    batch_cam[1, 2] *= -1
                    batch_cam[1, 7] *= -1
                
                if batch_3d is not None:
                    batch_3d = np.concatenate((batch_3d, batch_3d), axis=0)
                    batch_3d[1, :, :, 0] *= -1
                    batch_3d[1, :, self.joints_left + self.joints_right] = batch_3d[1, :, self.joints_right + self.joints_left]

                batch_2d = np.concatenate((batch_2d, batch_2d), axis=0)
                batch_2d[1, :, :, 0] *= -1
                batch_2d[1, :, self.kps_left + self.kps_right] = batch_2d[1, :, self.kps_right + self.kps_left]

            yield batch_cam, batch_3d, batch_2d
