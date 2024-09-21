import dataclasses
import json
import pickle

from pygame.time import Clock
from scipy.spatial.transform import Rotation
import torch
from utils import smpl_to_rbdl,set_pose
import articulate.math
import utils
from dynamics import PhysicsOptimizer
# from net import MeoCapNet
from net_onnx import ONNX_PIP
import tqdm
@dataclasses.dataclass
class PlainTrackerReport:
    rotation: list[float]  # length 4
    acc: list[float]  # length 3


@dataclasses.dataclass
class MeoCapFrameResult:
    pose: list[list[list[float]]]
    tran: list[float]
    net_j_v: list[float]
    net_pose: list[list[list[float]]]
    imus: list[PlainTrackerReport]
    calibrated_imus: list[PlainTrackerReport]


@torch.no_grad()
def test():
    file_path = "test_motion.json"
    with open(file_path, "rb") as file:
        data = json.load(file)
        recordings: list[MeoCapFrameResult] = data["recordings"]

        net = ONNX_PIP("models/MeoCapNetV6.onnx")
        poses = []
        trans = []
        print("Start Inference")
        for i in tqdm.tqdm(range(len(recordings))):
            acc = torch.stack([torch.tensor(imu['acc']) for imu in recordings[i]['calibrated_imus']]).view(11,3)
            rot = torch.stack([torch.tensor(Rotation.from_quat(imu['rotation']).as_matrix()) for imu in recordings[i]['calibrated_imus']]).view(11, 3,3)
            pose,tran = net.forward_frame(acc,rot)
            poses.append(pose)
            trans.append(tran)
            #pose_opt, tran_opt = optimizer.optimize_frame(poses[i], torch.tensor(recordings[i]['joint_velocity']).view(24, 3), torch.tensor(recordings[i]['contact']).view(2))

        clock = Clock()
        index = 0

        while True:
            index += 1
            clock.tick(60)
            net.set_pose(poses[index % len(poses)],trans[index % len(trans)])






test()
