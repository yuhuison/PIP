r"""
    Evaluate the pose estimation.
"""
import argparse

import torch
import tqdm
from pygame.time import Clock

from net import PIP
from config import *
import articulate as art
import socket

class PoseEvaluator:
    def __init__(self):
        self._eval_fn = art.FullMotionEvaluator(paths.smpl_file, joint_mask=torch.tensor([1, 2, 16, 17]))
        self.powered_errs = []
        self.powered_shape = 0

    def eval(self, pose_p, pose_t):
        pose_p = pose_p.clone().view(-1, 24, 3, 3)
        pose_t = pose_t.clone().view(-1, 24, 3, 3)
        pose_p[:, joint_set.ignored] = torch.eye(3, device=pose_p.device)
        pose_t[:, joint_set.ignored] = torch.eye(3, device=pose_t.device)
        errs = self._eval_fn(pose_p, pose_t).numpy()
        ret = [errs[9], errs[3], errs[0] * 100, errs[1] * 100, errs[4] / 100]
        return ret

    def get_avg(self):
        sum_error = torch.zeros(self.powered_errs[0].shape)
        for i in self.powered_errs:
            sum_error += i
        ret = sum_error / self.powered_shape
        return ret

    @staticmethod
    def print(errors):
        for i, name in enumerate(['SIP Error (deg)', 'Angular Error (deg)', 'Positional Error (cm)',
                                  'Mesh Error (cm)', 'Jitter Error (100m/s^3)']):
            print('%s: %.2f (+/- %.2f)' % (name, errors[i][0], errors[i][1]))


@torch.no_grad()
def evaluate_pose(send_to_unity = False):
    evaluator = PoseEvaluator()
    accs, rots, poses, _ = torch.load("test.pt").values()
    accs, rots, poses = accs[5], rots[5], poses[5]
    net: PIP = PIP().cuda()
    pose_results = []
    pose_gts = []
    tran_results = []
    for i in tqdm.tqdm(range(accs.shape[0])):
        acc, rot, pose_gt = accs[i], rots[i], poses[i]
        pose_online, tran_online = net.forward_frame(acc.view(1, 6, 3).cuda(), rot.view(1, 6, 3, 3).cuda())
        pose_results.append(pose_online)
        pose_gts.append(art.math.axis_angle_to_rotation_matrix(pose_gt))
        tran_results.append(tran_online)
    pose_results = torch.stack(pose_results)
    pose_gts = torch.stack(pose_gts)
    tran_results = torch.stack(tran_results)

    err = evaluator.eval(pose_results, pose_gts)
    evaluator.print(err)
    err = [_[0] for _ in err]
    if send_to_unity:
        pose_results = pose_results.view(-1,24,3,3)
        tran_results = tran_results.view(-1, 3)
        server_for_unity = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_for_unity.bind(('127.0.0.1', 8888))
        server_for_unity.listen(1)
        print('Server start. Waiting for unity3d to connect.')
        clock = Clock()
        idx = 0
        count_frames = pose_results.shape[0]
        conn, addr = server_for_unity.accept()
        while True:
            clock.tick(63)
            idx += 1
            pose = art.math.rotation_matrix_to_axis_angle(pose_results[idx % count_frames]).view(-1, 72)
            tran = tran_results[idx % count_frames].view(-1, 3)
            # send motion to Unity
            s = ','.join(['%g' % v for v in pose.view(-1)]) + '#' + \
                ','.join(['%g' % v for v in tran.view(-1)]) + '$'
            conn.send(s.encode('utf8'))


    return err



if __name__ == '__main__':
    evaluate_pose(True)
