import asyncio
import random
import threading
import time

import numpy
import numpy as np
import pymeocap
from pygame.time import Clock
from pythonosc import udp_client
from scipy.spatial.transform import Rotation

FIX_MAT_RECORD = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
FIX_MAT_M_C = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
NOW_RACKET_ROTATION = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=numpy.float32)
has_calibrated = False

def rotation_matrix_to_axis_angle(r):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    import cv2
    result = [cv2.Rodrigues(r[_].reshape(3,3))[0] for _ in range(len(r))]
    result = np.concatenate(result, axis=0)
    return result


async def start_listen_racket():
    cap = pymeocap.Meocap(1)
    await cap.connect()
    cap.start()

    clock = Clock()
    while True:
        clock.tick(50)
        rpt = cap.poll()
        if rpt is not None:
            if len(rpt) == 1:
                now_rot = Rotation.from_quat(rpt[0].rot).as_matrix()  # 得到球拍所采集的四元数并转化成旋转矩阵
                global has_calibrated
                if not has_calibrated:
                    has_calibrated = True
                    global FIX_MAT_RECORD
                    FIX_MAT_RECORD = now_rot
                    print("The Racket's Sensor Connected")
                global NOW_RACKET_ROTATION
                now_rot = np.dot(np.dot(np.dot(FIX_MAT_M_C, np.transpose(FIX_MAT_RECORD)), now_rot),
                                 np.transpose(FIX_MAT_M_C))
                NOW_RACKET_ROTATION = now_rot


def start_listen_racket_thread():
    asyncio.run(start_listen_racket())


if __name__ == "__main__":
    t = threading.Thread(target=start_listen_racket_thread)
    t.start()
    client = udp_client.SimpleUDPClient("127.0.0.1", 34567)
    clock = Clock()
    while True:
        clock.tick(30)
        rot = NOW_RACKET_ROTATION.copy()

        eulers = rotation_matrix_to_axis_angle(rot.reshape((1,3,3))).flatten().tolist()
        client.send_message("/Meo/Ext/Tracker/Rot",[eulers[0],eulers[1],eulers[2]])


