# Copyright (c) 2018-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import numpy as np


def getbonejs(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1, seq.size(2), seq.size(3))
    bone = []
    for i in range(24):
        #for j in range(i+1,17) is also OK
        for j in range(i, 24):
            if not ([i, j] in boneindex or [j, i] in boneindex):
                bone.append(seq[:, j] - seq[:, i])
    bone = torch.stack(bone, 1)
    bone = bone.view(bs, ss, bone.size(1), 3)
    return bone


def getbonelength(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1, seq.size(2), seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:, index[0]] - seq[:, index[1]])
    bone = torch.stack(bone, 1)
    bone = torch.pow(torch.pow(bone, 2).sum(2), 0.5)
    bone = bone.view(bs, ss, bone.size(1))
    return bone


def getbonedirect(seq, boneindex):
    bs = seq.size(0)
    ss = seq.size(1)
    seq = seq.view(-1, seq.size(2), seq.size(3))
    bone = []
    for index in boneindex:
        bone.append(seq[:, index[0]] - seq[:, index[1]])
    bonedirect = torch.stack(bone, 1)
    bonesum = torch.pow(torch.pow(bonedirect, 2).sum(2), 0.5).unsqueeze(2)
    bonedirect = bonedirect / (bonesum + 1e-6)
    bonedirect = bonedirect.view(bs, -1, 3)
    return bonedirect


'''
sample new bone lengths and adjust the 3D gt of each frame based on them,
'''
def randomaug_cuda(batch_3D_rand_ori, boneindex, augdegree):
    bs = batch_3D_rand_ori.size(0)
    ss = batch_3D_rand_ori.size(1)
    bonelen = getbonelength(batch_3D_rand_ori, boneindex).mean(1)
    bonelenmean = bonelen.mean(0)
    #sample new bone lengths
    randadd = (torch.rand(bs, 16).cuda() - 0.5) * (bonelenmean * augdegree)
    bonelennew = bonelen + randadd
    bonedirect = getbonedirect(batch_3D_rand_ori, boneindex).view(bs, ss, -1, 3)
    '''
    if you experiment with another dataset, temporally in this version you need to manually modify the below indexs to re-compute the gt "3D joint location" based on the re-sampled bone lengths,
    because the change of a specific bone length can lead to the changes of multiple joints' location  
    '''
    b = randadd[:, 0]
    batch_3D_rand_ori[:, :, 16:17] = batch_3D_rand_ori[:, :, 16:17] + torch.unsqueeze(bonedirect[:, :, 0] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 1]
    batch_3D_rand_ori[:, :, 15:17] = batch_3D_rand_ori[:, :, 15:17] + torch.unsqueeze(bonedirect[:, :, 1] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 2]
    batch_3D_rand_ori[:, :, 13:14] = batch_3D_rand_ori[:, :, 13:14] + torch.unsqueeze(bonedirect[:, :, 2] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 3]
    batch_3D_rand_ori[:, :, 12:14] = batch_3D_rand_ori[:, :, 12:14] + torch.unsqueeze(bonedirect[:, :, 3] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 4]
    batch_3D_rand_ori[:, :, 10:11] = batch_3D_rand_ori[:, :, 10:11] + torch.unsqueeze(bonedirect[:, :, 4] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 5]
    batch_3D_rand_ori[:, :, 9:11] = batch_3D_rand_ori[:, :, 9:11] + torch.unsqueeze(bonedirect[:, :, 5] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 6]
    batch_3D_rand_ori[:, :, 8:17] = batch_3D_rand_ori[:, :, 8:17] + torch.unsqueeze(bonedirect[:, :, 6] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 7]
    batch_3D_rand_ori[:, :, 11:14] = batch_3D_rand_ori[:, :, 11:14] - torch.unsqueeze(bonedirect[:, :, 7] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 8]
    batch_3D_rand_ori[:, :, 14:17] = batch_3D_rand_ori[:, :, 14:17] - torch.unsqueeze(bonedirect[:, :, 8] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 9]
    batch_3D_rand_ori[:, :, 7:] = batch_3D_rand_ori[:, :, 7:] + torch.unsqueeze(bonedirect[:, :, 9] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 10]
    batch_3D_rand_ori[:, :, 3:4] = batch_3D_rand_ori[:, :, 3:4] + torch.unsqueeze(bonedirect[:, :, 10] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 11]
    batch_3D_rand_ori[:, :, 2:4] = batch_3D_rand_ori[:, :, 2:4] + torch.unsqueeze(bonedirect[:, :, 11] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 12]
    batch_3D_rand_ori[:, :, 6:7] = batch_3D_rand_ori[:, :, 6:7] + torch.unsqueeze(bonedirect[:, :, 12] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 13]
    batch_3D_rand_ori[:, :, 5:7] = batch_3D_rand_ori[:, :, 5:7] + torch.unsqueeze(bonedirect[:, :, 13] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 14]
    batch_3D_rand_ori[:, :, 1:4] = batch_3D_rand_ori[:, :, 1:4] + torch.unsqueeze(bonedirect[:, :, 14] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 15]
    batch_3D_rand_ori[:, :, 4:7] = batch_3D_rand_ori[:, :, 4:7] + torch.unsqueeze(bonedirect[:, :, 15] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    return batch_3D_rand_ori, bonelennew


'''
sample new bone lengths and adjust the 3D gt of each frame based on them,
'''

def randomaug_cuda_24kpt(batch_3D_rand_ori, boneindex, augdegree):
    bs = batch_3D_rand_ori.size(0)
    ss = batch_3D_rand_ori.size(1)
    bonelen = getbonelength(batch_3D_rand_ori, boneindex).mean(1)
    bonelenmean = bonelen.mean(0)
    #sample new bone lengths
    randadd = (torch.rand(bs, 23).cuda() - 0.5) * (bonelenmean * augdegree)
    bonelennew = bonelen + randadd
    bonedirect = getbonedirect(batch_3D_rand_ori, boneindex).view(bs, ss, -1, 3)
    '''
    if you experiment with another dataset, temporally in this version you need to manually modify the below indexs to re-compute the gt "3D joint location" based on the re-sampled bone lengths,
    because the change of a specific bone length can lead to the changes of multiple joints' location  
    '''
    b = randadd[:, 0]
    batch_3D_rand_ori[:, :, 12:13] = batch_3D_rand_ori[:, :, 12:13] + torch.unsqueeze(bonedirect[:, :, 0] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 1]
    batch_3D_rand_ori[:, :, [12, 10]] = batch_3D_rand_ori[:, :, [12, 10]] + torch.unsqueeze(bonedirect[:, :, 1] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 2]
    batch_3D_rand_ori[:, :, 11:12] = batch_3D_rand_ori[:, :, 11:12] + torch.unsqueeze(bonedirect[:, :, 2] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 3]
    batch_3D_rand_ori[:, :, [9, 11]] = batch_3D_rand_ori[:, :, [9, 11]] + torch.unsqueeze(bonedirect[:, :, 3] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 4]
    batch_3D_rand_ori[:, :, 5:6] = batch_3D_rand_ori[:, :, 5:6] + torch.unsqueeze(bonedirect[:, :, 4] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 5]
    batch_3D_rand_ori[:, :, [5, 3]] = batch_3D_rand_ori[:, :, [5, 3]] + torch.unsqueeze(bonedirect[:, :, 5] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 6]
    batch_3D_rand_ori[:, :, 6:7] = batch_3D_rand_ori[:, :, 6:7] + torch.unsqueeze(bonedirect[:, :, 6] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 7]
    batch_3D_rand_ori[:, :, [4, 6]] = batch_3D_rand_ori[:, :, [4, 6]] + torch.unsqueeze(bonedirect[:, :, 7] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 8]
    batch_3D_rand_ori[:, :, 2:7] = batch_3D_rand_ori[:, :, 2:7] + torch.unsqueeze(bonedirect[:, :, 8] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 9]
    batch_3D_rand_ori[:, :, 23:24] = batch_3D_rand_ori[:, :, 23:24] + torch.unsqueeze(bonedirect[:, :, 9] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 10]
    batch_3D_rand_ori[:, :, [7, 9, 11]] = batch_3D_rand_ori[:, :, [7, 9, 11]] + torch.unsqueeze(bonedirect[:, :, 10] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 11]
    batch_3D_rand_ori[:, :, [8, 10, 12]] = batch_3D_rand_ori[:, :, [8, 10, 12]] + torch.unsqueeze(bonedirect[:, :, 11] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 12]
    batch_3D_rand_ori[:, :, 1:12] = batch_3D_rand_ori[:, :, 1:12] + torch.unsqueeze(bonedirect[:, :, 12] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    batch_3D_rand_ori[:, :, 23:24] = batch_3D_rand_ori[:, :, 23:24] + torch.unsqueeze(bonedirect[:, :, 12] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 13]
    batch_3D_rand_ori[:, :, 20:21] = batch_3D_rand_ori[:, :, 20:21] + torch.unsqueeze(bonedirect[:, :, 13] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 14]
    batch_3D_rand_ori[:, :, 22:23] = batch_3D_rand_ori[:, :, 22:23] + torch.unsqueeze(bonedirect[:, :, 14] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 15]
    batch_3D_rand_ori[:, :, [18, 20, 22]] = batch_3D_rand_ori[:, :, [18, 20, 22]] + torch.unsqueeze(bonedirect[:, :, 15] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 16]
    batch_3D_rand_ori[:, :, [16, 18, 20, 22]] = batch_3D_rand_ori[:, :, [16, 18, 20, 22]] + torch.unsqueeze(bonedirect[:, :, 16] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 17]
    batch_3D_rand_ori[:, :, 19:20] = batch_3D_rand_ori[:, :, 19:20] + torch.unsqueeze(bonedirect[:, :, 17] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 18]
    batch_3D_rand_ori[:, :, 21:22] = batch_3D_rand_ori[:, :, 21:22] + torch.unsqueeze(bonedirect[:, :, 18] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 19]
    batch_3D_rand_ori[:, :, [17, 19, 21]] = batch_3D_rand_ori[:, :, [17, 19, 21]] + torch.unsqueeze(bonedirect[:, :, 19] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 20]
    batch_3D_rand_ori[:, :, [15, 17, 19, 21]] = batch_3D_rand_ori[:, :, [15, 17, 19, 21]] + torch.unsqueeze(bonedirect[:, :, 20] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 21]
    batch_3D_rand_ori[:, :, [13, 15, 17, 19, 21]] = batch_3D_rand_ori[:, :, [13, 15, 17, 19, 21]] + torch.unsqueeze(bonedirect[:, :, 21] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)
    b = randadd[:, 22]
    batch_3D_rand_ori[:, :, [14, 16, 18, 20, 22]] = batch_3D_rand_ori[:, :, [14, 16, 18, 20, 22]] + torch.unsqueeze(bonedirect[:, :, 22] * torch.tile(torch.unsqueeze(torch.unsqueeze(b, 1), 2), (1, ss, 3)), 2)

    return batch_3D_rand_ori, bonelennew
