import pickle
import time

import tqdm
from pygame.time import Clock

import articulate as art
import torch
from scipy.spatial.transform import Rotation
from vmc_stream_client import MeocapVMC_Client
import config
from config import paths

class Rokoko_Recv:
    def __init__(self):
        self.client = MeocapVMC_Client(39539,"rkk")


    def start_forever(self):
        self.client.start_listen()
        clock = Clock()
        tick = 0

        while True:
            clock.tick(60)
            tick += 1
            if self.client.pose_data is not None:



def read_frames(file_name):
    frames = []
    with open(file_name, 'rb') as file:
        while True:
            try:
                frame_data = pickle.load(file)
                frames.append(frame_data)
            except EOFError:
                break
    return frames


def preprocess(file_paths: list[str]):
    vi_mask = torch.tensor(config.joint_set.VERTEX_IDS_10_Joint)
    ji_mask = torch.tensor(config.joint_set.sensor_full)
    body_model = art.ParametricModel(paths.smpl_file)

    def _syn_acc(v, smooth_n=4):
        r"""
        Synthesize accelerations from vertex positions.
        """
        mid = smooth_n // 2
        acc = torch.stack([(v[i] + v[i + 2] - 2 * v[i + 1]) * 3600 for i in range(0, v.shape[0] - 2)])
        acc = torch.cat((torch.zeros_like(acc[:1]), acc, torch.zeros_like(acc[:1])))
        if mid != 0:
            acc[smooth_n:-smooth_n] = torch.stack(
                [(v[i] + v[i + smooth_n * 2] - 2 * v[i + smooth_n]) * 3600 / smooth_n ** 2
                 for i in range(0, v.shape[0] - smooth_n * 2)])
        return acc

    for file_path in file_paths:
        recordings = read_frames(file_path)
        recordings = recordings[int(len(recordings) * 0.1):int(len(recordings) * 0.9)]
        poses = torch.stack([torch.tensor(r['pose_gt']) for r in recordings]).to(
                torch.float32)
        grot, joint, vert = body_model.forward_kinematics(poses, calc_mesh=True)
        v_rots = grot[:, ji_mask].view(-1, 10, 3, 3)
        v_accs = _syn_acc(vert[:, vi_mask]).view(-1, 10, 3)

