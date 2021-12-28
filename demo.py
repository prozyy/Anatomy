from common.model import TemporalModel
import torch
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
# matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def plot_pose3d(pose):
    import mpl_toolkits.mplot3d.axes3d as p3
    _CONNECTION = [[0,1],[1,2],[1,23],[2,3],[2,4],[3,5],[4,6],[1,7],[1,8],[7,9],[9,11],[8,10],[10,12],
                   [0,13],[0,14],[13,15],[14,16],[15,17],[16,18],[17,19],[17,21],[18,22],[18,20]]
    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure(figsize=(8,8))
    ax = fig.gca ( projection='3d' )
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % (255,0,0)
        ax.plot ( [pose[0, c[0]], pose[0, c[1]]],
                    [pose[2, c[0]], pose[2, c[1]]],
                    [-pose[1, c[0]], -pose[1, c[1]]], c=col )

    for j in range ( pose.shape[1] ):
        col = '#%02x%02x%02x' % (0,255,0)
        ax.scatter ( pose[0, j],pose[2, j], -pose[1, j],
                        c=col, marker='o', edgecolor=col )

    ax.set_xlim3d ( -1.5, 1.5)
    ax.set_ylim3d ( -1.5, 1.5)
    ax.set_zlim3d ( -1.5, 1.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")

    # ax.view_init(elev=0,azim=90)  ## 正向

    return fig


def normalize_screen_coordinates(X, w, h):
    assert X.shape[-1] == 2

    # Normalize so that [0, w] is mapped to [-1, 1], while preserving the aspect ratio
    return X / w * 2 - [1, h / w]


class Anatomy3DPose:
    def __init__(self, modelPath,imageSize):
        self.boneindex = [[12, 10], [10, 8], [11, 9], [9, 7], [5, 3], [3, 2], [6, 4], [4, 2], [2, 1], [23, 1], [7, 1], [8, 1], [1, 0], [20, 18], [22, 18], [18, 16], [16, 14], [19, 17], [21, 17], [17, 15], [15, 13], [13, 0], [14, 0]]
        self.model = TemporalModel(24, 2, 24, self.boneindex, 10.0, 50, filter_widths=[3, 3, 3, 3, 3], causal=False, dropout=0.25, channels=1024, dense=False)
        self.model = torch.nn.DataParallel(self.model)
        if torch.cuda.is_available():
            self.model = self.model.cuda()

        self.loadWeights(modelPath)
        self.model.eval()

        self.receptive_field = self.model.module.receptive_field()
        self.pad = (self.receptive_field - 1) // 2  # Padding on each side
        self.res_w = imageSize[0]
        self.res_h = imageSize[1]
        self.poseData = {}

    def loadWeights(self, weightPath):
        checkpoint = torch.load(weightPath, map_location=lambda storage, loc: storage)
        self.model.load_state_dict(checkpoint['model_pos'])

    def inference(self, poseDic, frameIndex):
        result_trackId = []
        input_keypoints_all = None
        for k, v in poseDic.items():
            cur_keypoints = v
            cur_keypoints[..., :2] = normalize_screen_coordinates(v[..., :2], w=self.res_w, h=self.res_h)
            if k in self.poseData:
                pose2darray = self.poseData[k]["point2darr"]
                pose2darray = np.concatenate((pose2darray, cur_keypoints), axis=0)
                self.poseData[k]["point2darr"] = pose2darray[-121:]
                self.poseData[k]["curFrameIndex"] = frameIndex
            else:
                pose2darray = cur_keypoints
                self.poseData[k] = {"point2darr": pose2darray, "curFrameIndex": frameIndex}

            result_trackId.append(k)

            input_keypoints = pose2darray.copy()
            pad_r = 121
            pad_l = 122 - pose2darray.shape[0]
            input_keypoints = np.expand_dims(np.pad(input_keypoints, ((pad_l, pad_r), (0, 0), (0, 0)), 'edge'), axis=0)

            if input_keypoints_all is None:
                input_keypoints_all = input_keypoints
            else:
                input_keypoints_all = np.concatenate((input_keypoints_all, input_keypoints), axis=0)

        inputs_2d = torch.from_numpy(input_keypoints_all.astype('float32'))
        if torch.cuda.is_available():
            inputs_2d = inputs_2d.cuda()

        predicted_3d_pos = self.model(inputs_2d)

        return result_trackId, predicted_3d_pos.squeeze(0).squeeze(0).detach().cpu().numpy()

if __name__ == "__main__":
    anatomyModel = Anatomy3DPose("checkpoint/epochfinal_60.bin",(1200,1780))
    input_keypoints = np.load("data/testData/test.npy").transpose(0,2,1)
    if not os.path.exists("temp"):
        os.mkdir("temp")
    for i in range(input_keypoints.shape[0]):
        trackIdList, pose3d = anatomyModel.inference({"1":input_keypoints[i:(i+1),:,:]}, i)

        print(pose3d.shape)
        fig = plot_pose3d(pose3d.transpose())

        if i % 10 == 0:
            fig.savefig("temp/"+"frame_"+ str(i) + ".png")
