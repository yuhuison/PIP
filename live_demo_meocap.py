import asyncio
import dataclasses
import threading

from scipy.spatial.transform import Rotation
import pymeocap
import socket
import torch
from pygame.time import Clock
from net import PIP
import articulate as art
import win32api
import os
from config import *


@dataclasses.dataclass
class meocap_imu:
    acc: tuple[float]  # x,y,z
    rot: tuple[float]  # i,j,k,w
    accuracy: int


class meocap_imu_set:
    imus: list[meocap_imu]

    def __init__(self):
        self.imus = []

    def get(self):
        R = torch.tensor([Rotation.from_quat(imu.rot).as_matrix() for imu in self.imus]).to(torch.float).view(6, 3, 3)
        a = torch.tensor([imu.acc for imu in self.imus]).view(-1, 3).view(6, 3)
        a = R.bmm(a.unsqueeze(-1)).squeeze(-1)
        return None, R, a


imu_set = meocap_imu_set()
data_loaded = False


async def recv_imu():
    clock = Clock()
    m = pymeocap.Meocap(6)
    await m.connect()
    m.start()
    frame_count = 0
    imu_names = ["左手腕", "右手腕", "左小腿", "右小腿", "头部", "腰部"]
    acc_texts = ["不可信", "低准确度", "中准确度", "高准确度"]
    while True:
        clock.tick(60)
        rpt = m.poll()
        if rpt is not None and len(rpt) == 6:
            global data_loaded
            data_loaded = True
            imu_set.imus = rpt
            frame_count += 1
            if frame_count % 60 == 0:
                text = ""
                has_low_accuracy = False
                for index, imu in enumerate(imu_set.imus):
                    if imu.accuracy != 15:
                        acc = imu.accuracy
                        binary_str = format(acc, '04b')
                        high_two_bits = binary_str[:2]
                        high_two_bits_decimal = int(high_two_bits, 2)
                        if high_two_bits_decimal < 2:
                            has_low_accuracy = True
                            text += "{0}({1}) ,".format(imu_names[index], acc_texts[high_two_bits_decimal])
                if has_low_accuracy:
                    print("Warning! These IMU is not full-calibrated : " + text)


def tpose_calibration():
    clock_sec = Clock()
    t_remain = 5
    while True:
        clock_sec.tick(1)
        print("Waiting T-Pose Calibration after {0} second".format(t_remain))
        t_remain = t_remain - 1
        if t_remain == 0:
            break
    print("T-Pose Calibration Start")
    M_C = torch.tensor([0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0]).view(3, 3)
    rot = imu_set.get()[1]
    root_rot = rot[5]
    euler = Rotation.from_matrix(torch.transpose(root_rot.view(3, 3),0,1)).as_euler("xyz", degrees=True)
    euler[0] = -90.0 if euler[0] < 0 else 90.0
    euler[2] = -90.0 if euler[2] < 0 else 90.0
    a_pose_mat = torch.transpose(torch.tensor(Rotation.from_euler("xyz", euler, degrees=True).as_matrix()).view(3, 3),0,1)
    RMI = M_C.double().mm(a_pose_mat.t())
    RSB = torch.stack([(RMI.matmul(r.double())).t() for r in rot]).view(6, 3, 3)
    print("T-Pose Calibration End")
    return RMI, RSB


def run_recv_thread():
    pymeocap.enable_log("info")
    asyncio.run(recv_imu())


if __name__ == '__main__':
    t = threading.Thread(target=run_recv_thread)
    t.start()

    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)
    print('Server start. Waiting for unity3d to connect.')
    if paths.unity_file != '' and os.path.exists(paths.unity_file):
        win32api.ShellExecute(0, 'open', os.path.abspath(paths.unity_file), '', '', 1)
        is_executable = True
    conn, addr = server_for_unity.accept()
    RMI, RSB = tpose_calibration()
    net = PIP()
    clock = Clock()

    while True:
        clock.tick(63)
        tframe, RIS, aI = imu_set.get()
        RMB = RMI.matmul(RIS.double()).matmul(RSB)
        aM = aI.double().mm(RMI.t())
        pose, tran, cj, grf = net.forward_frame(aM.float().view(1, 6, 3), RMB.float().view(1, 6, 3, 3), return_grf=True)
        pose = art.math.rotation_matrix_to_axis_angle(pose).view(-1, 72)
        tran = tran.view(-1, 3)
        # send motion to Unity
        s = ','.join(['%g' % v for v in pose.view(-1)]) + '#' + \
            ','.join(['%g' % v for v in tran.view(-1)]) + '#' + \
            ','.join(['%d' % v for v in cj]) + '#' + \
            (','.join(['%g' % v for v in grf.view(-1)]) if grf is not None else '') + '$'

        try:
            conn.send(s.encode('utf8'))
        except:
            break
