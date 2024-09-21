import json
import socket
import numpy as np
import torch
from pygame.time import Clock


def rotation_matrix_to_axis_angle(r: torch.Tensor):
    r"""
    Turn rotation matrices into axis-angles. (torch, batch)

    :param r: Rotation matrix tensor that can reshape to [batch_size, 3, 3].
    :return: Axis-angle tensor of shape [batch_size, 3].
    """
    import cv2
    result = [cv2.Rodrigues(_)[0] for _ in r.clone().detach().cpu().view(-1, 3, 3).numpy()]
    result = torch.from_numpy(np.stack(result)).float().squeeze(-1).to(r.device)
    return result


def start_unity_server(output_pose, output_tran):
    """

    """
    clock = Clock()
    print('Server start. Waiting for unity3d to connect.')
    server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_for_unity.bind(('127.0.0.1', 8888))
    server_for_unity.listen(1)

    i = 0
    while True:
        clock.tick(60)
        idx = i % output_pose.shape[0]
        print("send frame " + str(idx))
        i += 1
        pose = output_pose[idx]
        tran = output_tran[idx]
        pose = rotation_matrix_to_axis_angle(pose.reshape(1, 216)).view(72)
        s = ','.join(['%g' % v for v in pose]) + '#' + \
            ','.join(['%g' % v for v in tran])
        server_for_unity.send(s.encode('utf8'))


if __name__ == '__main__':
    file_path = "test_motion.json"
    with open(file_path, "rb") as file:
        data = json.load(file)
        recordings = data["recordings"]
        poses = []
        trans = []

        for i in range(len(recordings)):
            poses.append(recordings[i]["optimized_pose"])
            trans.append(recordings[i]["translation"])

        poses = torch.tensor(poses).view(-1,24,3,3)
        trans = torch.tensor(trans).view(-1,3)

        start_unity_server(poses,trans)


