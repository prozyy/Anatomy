import numpy as np
import json
import copy
from common.camera import normalize_screen_coordinates, image_coordinates, world_to_camera
import cv2
import matplotlib.pyplot as plt
import math
import sys
from tqdm import tqdm

skeleton = ((0, 21), (0, 1), (0, 2), (1, 3), (2, 4), (5, 6), (5, 7), (7, 9), (9, 22), (6, 8), (8, 10), (10, 23), (5, 11), (11, 12), (11, 13), (13, 15), (15, 17), (15, 19), (6, 12), (12, 14), (14, 16), (16, 18), (16, 20))

def process3DPose(positions, subject):
    frames, kptNum, _ = positions.shape
    positions_update = np.zeros((frames, 24, 3), dtype=np.float32)
    positions_update[:, 2:, :] = positions[:, :-2, :]
    positions_update[:, 0, :] = np.mean(positions_update[:, [13, 14]], axis=1)
    positions_update[:, 1, :] = np.mean(positions_update[:, [7, 8]], axis=1)

    positions_3d = []
    for cam in cameras_all[subject]:
        pos_3d = world_to_camera(positions_update, R=cam['orientation'], t=cam['translation'])
        pos_3d[:, 1:] -= pos_3d[:, :1]  # Remove global offset, but keep trajectory in first position
        positions_3d.append(pos_3d)
    return positions_3d

def vis_keypoints(kps, kps_lines=skeleton, kp_thresh=0.3, alpha=1):
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]
    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.ones((1080, 1920, 3), dtype=np.uint8) * 128

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(kp_mask, p1, p2, color=colors[l], thickness=3, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(kp_mask, p1, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(kp_mask, p2, radius=3, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    return kp_mask

def plot_pose3d(pose):
    import mpl_toolkits.mplot3d.axes3d as p3
    _CONNECTION = [[0, 1], [1, 2], [1, 23], [2, 3], [2, 4], [3, 5], [4, 6], [1, 7], [1, 8], [7, 9], [9, 11], [8, 10], [10, 12], [0, 13], [0, 14], [13, 15], [14, 16], [15, 17], [16, 18], [17, 19], [17, 21], [18, 22], [18, 20]]
    assert (pose.ndim == 2)
    assert (pose.shape[0] == 3)
    fig = plt.figure(figsize=(8, 8))
    ax = fig.gca(projection='3d')
    for c in _CONNECTION:
        col = '#%02x%02x%02x' % (255, 0, 0)
        ax.plot([pose[0, c[0]], pose[0, c[1]]], [pose[2, c[0]], pose[2, c[1]]], [-pose[1, c[0]], -pose[1, c[1]]], c=col)

    for j in range(pose.shape[1]):
        col = '#%02x%02x%02x' % (0, 255, 0)
        ax.scatter(pose[0, j], pose[2, j], -pose[1, j], c=col, marker='o', edgecolor=col)

    ax.set_xlim3d(-1.5, 1.5)
    ax.set_ylim3d(-1.5, 1.5)
    ax.set_zlim3d(-1.5, 1.5)

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_zlabel("-Y")

    ax.view_init(elev=0,azim=-90)  ## 正向

    return fig

def renderPose(subject, action, pose2dL, pose3dL):
    canNum = len(pose2dL)
    frameNum = pose2dL[0].shape[0]
    for frameIndnx in [0,frameNum//2,frameNum-1]:
        image_2d_list = []
        image_3d_list = []
        for camIndex in range(canNum):
            image_2d = vis_keypoints(pose2dL[camIndex][frameIndnx].transpose())
            image_2d_list.append(image_2d)
            pose3d = pose3dL[camIndex][frameIndnx].transpose()
            pose3d[:, 0] = pose3d[:, 0] * 0
            fig = plot_pose3d(pose3d)
            fig.canvas.draw()
            image_3d = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
            image_3d = image_3d.reshape(fig.canvas.get_width_height()[::-1] + (3, ))
            image_3d_list.append(image_3d)
            plt.close(fig)
            
        if canNum == 4:
            image_2d_all = np.array(image_2d_list).reshape(2,2,1080,1920,3).transpose(0,2,1,3,4).reshape(2160, 3840, 3)
            image_3d_all = np.array(image_3d_list).reshape(2,2,800,800,3).transpose(0,2,1,3,4).reshape(1600, 1600, 3)
            image_3d_all = np.pad(image_3d_all,((280, 280),(0,0),(0,0)),"edge")
            image_concat = np.concatenate((image_2d_all, image_3d_all),axis=1)
        if canNum == 9:
            image_2d_all = np.array(image_2d_list).reshape(3,3,1080,1920,3).transpose(0,2,1,3,4).reshape(1080 * 3, 1920 * 3, 3)
            image_3d_all = np.array(image_3d_list).reshape(3,3,800,800,3).transpose(0,2,1,3,4).reshape(800 * 3, 800 * 3, 3)
            image_3d_all = np.pad(image_3d_all,((140 * 3, 140 * 3),(0,0),(0,0)),"edge")
            image_concat = np.concatenate((image_2d_all, image_3d_all),axis=1)
        if canNum == 12:
            image_2d_all = np.array(image_2d_list).reshape(4,3,1080,1920,3).transpose(0,2,1,3,4).reshape(1080 * 4, 1920 * 3, 3)
            image_3d_all = np.array(image_3d_list).reshape(4,3,800,800,3).transpose(0,2,1,3,4).reshape(800 * 4, 800 * 3, 3)
            image_3d_all = np.pad(image_3d_all,((140 * 4, 140 * 4),(0,0),(0,0)),"edge")
            image_concat = np.concatenate((image_2d_all, image_3d_all),axis=1)
            
        
        cv2.imwrite("delete/{:s}_{:s}_{:d}.jpg".format(subject,action,frameIndnx),image_concat)
        # cv2.imwrite("{:s}_{:s}_{:d}.jpg".format(subject,action,frameIndnx),image_concat)

if __name__ == "__main__":
    cam_params = json.load(open("data/cam_all_z++.json"))
    cameras_all = copy.deepcopy(cam_params)
    for cameras in cameras_all.values():
        for i, cam in enumerate(cameras):
            for k, v in cam.items():
                if k not in ['id', 'res_w', 'res_h']:
                    cam[k] = np.array(v, dtype='float32')
            # Normalize camera frame
            cam['center'] = normalize_screen_coordinates(cam['center'], w=cam['res_w'], h=cam['res_h']).astype('float32')
            cam['focal_length'] = cam['focal_length'] / cam['res_w'] * 2
            if 'translation' in cam:
                cam['translation'] = cam['translation'] / 1000  # mm to meters
            # Add intrinsic parameters vector
            cam['intrinsic'] = np.concatenate((cam['focal_length'], cam['center'], cam['radial_distortion'], cam['tangential_distortion']))

    data2d_file = np.load("data/data_2d_h36m_24_aist_megapose_gt.npz", allow_pickle=True)
    data_2d = data2d_file['positions_2d'].item()
    data_3d = np.load("data/data_3d_h36m_24_aist_megapose.npz", allow_pickle=True)['positions_3d'].item()
    # ['S11', 'S1', 'S7', 'S9', 'S6', 'S5', 'S8', 
    #  'setting1', 'setting101', 'setting102', 'setting2', 'setting3', 'setting4', 'setting5', 'setting6', 'setting71', 'setting72', 'setting81', 'setting82', 'setting91', 'setting92', 
    #  'aerobicgymnastics', 'badminton', 'martialarts', 'rhythmicgymnastics', 'rumba', 'sanda']
    for subject in data_2d.keys():
        print(subject)
        for action in tqdm(data_2d[subject].keys()):
            keypoint2d_list = data_2d[subject][action]
            keypoint3d = data_3d[subject][action]
            keypoint3d_list = process3DPose(keypoint3d, subject)
            renderPose(subject, action, keypoint2d_list, keypoint3d_list)
