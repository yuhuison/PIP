import os
import torch.nn
from torch.nn.functional import relu
import utils
from config import *
from dynamics import PhysicsOptimizer
from utils import normalize_and_concat
import articulate as art
from torch.nn.utils.rnn import *


class RNN(torch.nn.Module):
    r"""
    An RNN net including a linear input layer, an RNN, and a linear output layer.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None, name=""):
        r"""
        Init an RNN.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        super().__init__()
        self.rnn = getattr(torch.nn, rnn_type.upper())(hidden_size, hidden_size, num_rnn_layer,
                                                       bidirectional=bidirectional, dropout=dropout)
        self.linear1 = torch.nn.Linear(input_size, hidden_size)
        self.linear2 = torch.nn.Linear(hidden_size * (2 if bidirectional else 1), output_size)
        self.dropout = torch.nn.Dropout(dropout) if dropout > 0 else torch.nn.Identity()

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, init=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains tensors in shape [num_frames, input_size].
        :param init: Initial hidden states.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        length = [_.shape[0] for _ in x]
        x = self.dropout(relu(self.linear1(pad_sequence(x))))
        x = self.rnn(pack_padded_sequence(x, length, enforce_sorted=False), init)[0]
        x = self.linear2(pad_packed_sequence(x)[0])
        return [x[:l, i].clone() for i, l in enumerate(length)]


class RNNWithInit(RNN):
    r"""
    RNN with the initial hidden states regressed from the first output.
    """

    def __init__(self, input_size: int, output_size: int, hidden_size: int, num_rnn_layer: int,
                 rnn_type='lstm', bidirectional=False, dropout=0., load_weight_file: str = None, name=""):
        r"""
        Init an RNNWithInit net.

        :param input_size: Input size.
        :param output_size: Output size.
        :param hidden_size: Hidden size for RNN.
        :param num_rnn_layer: Number of RNN layers.
        :param rnn_type: Select from 'rnn', 'lstm', 'gru'.
        :param bidirectional: Whether if the RNN is bidirectional.
        :param dropout: Dropout after the input linear layer and in the rnn.
        :param load_weight_file: If not None and exists, weights will be loaded.
        """
        assert rnn_type == 'lstm' and bidirectional is False
        super().__init__(input_size, output_size, hidden_size, num_rnn_layer, rnn_type, bidirectional, dropout)

        self.init_net = torch.nn.Sequential(
            torch.nn.Linear(output_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size, hidden_size * num_rnn_layer),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_size * num_rnn_layer, 2 * (2 if bidirectional else 1) * num_rnn_layer * hidden_size)
        )

        if load_weight_file and os.path.exists(load_weight_file):
            self.load_state_dict(torch.load(load_weight_file))
            self.eval()

    def forward(self, x, _=None):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 2-tuple
                  (Tensor[num_frames, input_size], Tensor[output_size]).
        :param _: Not used.
        :return: A list in length [batch_size] which contains tensors in shape [num_frames, output_size].
        """
        x, x_init = list(zip(*x))
        nd, nh = self.rnn.num_layers * (2 if self.rnn.bidirectional else 1), self.rnn.hidden_size
        h, c = self.init_net(torch.stack(x_init)).view(-1, 2, nd, nh).permute(1, 2, 0, 3)
        return super(RNNWithInit, self).forward(x, (h, c))


def normalize_tensor(x: torch.Tensor, dim=-1, return_norm=False):
    r"""
    Normalize a tensor in a specific dimension to unit norm. (torch)

    :param x: Tensor in any shape.
    :param dim: The dimension to be normalized.
    :param return_norm: If True, norm(length) tensor will also be returned.
    :return: Tensor in the same shape. If return_norm is True, norm tensor in shape [*, 1, *] (1 at dim)
             will also be returned (keepdim=True).
    """
    norm = x.norm(dim=dim, keepdim=True)
    normalized_x = x / norm
    return normalized_x if not return_norm else (normalized_x, norm)


def r6d_to_rotation_matrix(r6d: torch.Tensor):
    r"""
    Turn 6D vectors into rotation matrices. (torch, batch)

    **Warning:** The two 3D vectors of any 6D vector must be linearly independent.

    :param r6d: 6D vector tensor that can reshape to [batch_size, 6].
    :return: Rotation matrix tensor of shape [batch_size, 3, 3].
    """
    r6d = r6d.view(15, 6)
    column0 = normalize_tensor(r6d[:, 0:3])
    column1 = normalize_tensor(r6d[:, 3:6] - (column0 * r6d[:, 3:6]).sum(dim=1, keepdim=True) * column0)
    column2 = column0.cross(column1, dim=1)
    r = torch.stack((column0, column1, column2), dim=-1)
    r[torch.isnan(r)] = 0
    return r


def _inverse_tree(x_global: torch.Tensor):
    r"""
    Inversely multiply/add matrices along the tree branches. x_global [N, J, *]. parent [J].
    """
    x_local = [x_global[:, 0]]
    for index, i in enumerate([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]):
        x_local.append(torch.bmm(torch.transpose(x_global[:, i], dim0=1, dim1=2), x_global[:, index + 1]))
    x_local = torch.stack(x_local, dim=1)
    return x_local


def inverse_kinematics_R(R_global: torch.Tensor):
    r"""
    :math:`R_local = IK(R_global)`

    Inverse kinematics that computes the local rotation of each joint from global rotations. (torch, batch)

    Notes
    -----
    A joint's *local* rotation is expressed in its parent's frame.

    A joint's *global* rotation is expressed in the base (root's parent) frame.

    R_global[:, i], parent[i] should be the global rotation and parent joint id of
    joint i. parent[i] must be smaller than i for any i > 0.

    Args
    -----
    :param R_global: Joint global rotation tensor in shape [batch_size, *] that can reshape to
                     [batch_size, num_joint, 3, 3] (rotation matrices).
    :param parent: Parent joint id list in shape [num_joint]. Use -1 or None for base id (parent[0]).
    :return: Joint local rotation, in shape [batch_size, num_joint, 3, 3].
    """
    R_global = R_global.view(1, 24, 3, 3)
    R_local = _inverse_tree(R_global)
    return R_local


class PIP_FULL(torch.nn.Module):
    name = 'MeoCapNetFullJoint'
    n_hidden = 256

    def __init__(self, load_weight_file="", load_checkpoint_file=""):
        super(PIP_FULL, self).__init__()
        self.jvel_init = None
        self.lj_init = None
        # 9 * 10 + 3 * 10 = 90

        self.rnn_leaf = RNNWithInit(input_size=132,
                                    output_size=joint_set.n_leaf * 3,
                                    hidden_size=256,
                                    num_rnn_layer=2,
                                    dropout=0.4, name="rnn_leaf")

        self.rnn_full = RNN(input_size=132 + joint_set.n_leaf * 3,
                            output_size=joint_set.n_full * 3,
                            hidden_size=256,
                            num_rnn_layer=2,
                            dropout=0.4, name="rnn_full")

        self.rnn_pose = RNN(input_size=132 + joint_set.n_full * 3,
                            output_size=15 * 6,
                            hidden_size=256,
                            num_rnn_layer=2,
                            dropout=0.4, name="rnn_pose")

        self.rnn_vel = RNNWithInit(input_size=132 + joint_set.n_full * 3,
                            output_size=72,
                            hidden_size=256,
                            num_rnn_layer=2,
                            dropout=0.4, name="rnn_vel")


        self.rnn_stables = RNN(input_size=132 + joint_set.n_full * 3,
                               output_size=2,
                               hidden_size=128,
                               num_rnn_layer=2,
                               dropout=0.4, name="rnn_stables")

        body_model = art.ParametricModel(paths.smpl_file)
        self.inverse_kinematics_R = body_model.inverse_kinematics_R
        self.forward_kinematics = body_model.forward_kinematics
        self.dynamics_optimizer = PhysicsOptimizer(debug=False)
        self.rnn_states = [None for _ in range(5)]
        if load_weight_file != "":
            self.load_state_dict(torch.load(load_weight_file))
        if load_checkpoint_file != "":
            state_dict: dict = torch.load(load_checkpoint_file)["state_dict"]
            rnn1_states = dict()
            rnn2_states = dict()
            rnn3_states = dict()
            for key in state_dict.keys():
                k: str = key
                if k.startswith("rnn_leaf"):
                    rnn1_states[k.replace("rnn_leaf.", "")] = state_dict[k]
                elif k.startswith("rnn_full"):
                    rnn2_states[k.replace("rnn_full.", "")] = state_dict[k]
                elif k.startswith("rnn_pose"):
                    rnn3_states[k.replace("rnn_pose.", "")] = state_dict[k]

            self.rnn_leaf.load_state_dict(rnn1_states)
            self.rnn_full.load_state_dict(rnn2_states)
            self.rnn_pose.load_state_dict(rnn3_states)

    @torch.no_grad()
    def forward_frame_rnn1_and_rnn2(self, glb_acc, glb_rot):
        r"""
        Forward. Currently only support 1 subject.

        :param glb_acc: A tensor in [num_subjects, 6, 3].
        :param glb_rot: A tensor in [num_subjects, 6, 3, 3].
        """
        imu = utils.normalize_and_concat_11(glb_acc.view(-1, 11, 3), glb_rot.view(-1, 11, 3, 3))
        x, state2 = self.rnn2.rnn(relu(self.rnn2.linear1(imu), inplace=True).unsqueeze(0))
        full_joint = self.rnn2.linear2(x[0])
        return full_joint

    def _reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def reduced_glb_6d_to_full_local_mat(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = art.math.r6d_to_rotation_matrix(glb_reduced_pose).view(-1, joint_set.n_reduced, 3, 3)
        global_full_pose = torch.eye(3, device=glb_reduced_pose.device).repeat(glb_reduced_pose.shape[0], 24, 1, 1)
        global_full_pose[:, joint_set.reduced] = glb_reduced_pose
        pose = self.inverse_kinematics_R(global_full_pose).view(-1, 24, 3, 3)
        pose[:, joint_set.ignored] = torch.eye(3, device=pose.device)
        pose[:, 0] = root_rotation.view(-1, 3, 3)
        return pose

    def normalize_and_concat(self, glb_acc, glb_rot):
        glb_acc = glb_acc.view(1, 6, 3)
        glb_rot = glb_rot.view(1, 6, 3, 3)
        acc = torch.cat((glb_acc[:, :5] - glb_acc[:, 5:], glb_acc[:, 5:]), dim=1).bmm(glb_rot[:, -1])
        ori = torch.cat((glb_rot[:, 5:].transpose(2, 3).matmul(glb_rot[:, :5]), glb_rot[:, 5:]), dim=1)
        data = torch.cat((acc.flatten(1), ori.flatten(1)), dim=1)
        return data, acc, ori

    def reduced_glb_6d_to_full_local_mat_onnx(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = r6d_to_rotation_matrix(glb_reduced_pose).view(1, 15, 3, 3)
        global_full_pose = torch.eye(3).repeat(1, 24, 1, 1)
        for index, i in enumerate([1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]):
            global_full_pose[:, i] = glb_reduced_pose[:, index]

        x_local = [global_full_pose[:, 0]]
        for index, i in enumerate([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]):
            x_local.append(
                torch.bmm(torch.transpose(global_full_pose[:, i], dim0=1, dim1=2), global_full_pose[:, index + 1]))
        x_local = torch.stack(x_local, dim=1)
        pose = x_local.view(1, 24, 3, 3)
        for index, i in enumerate([0, 7, 8, 10, 11, 20, 21, 22, 23]):
            pose[:, i] = torch.eye(3)
        pose[:, 0] = root_rotation.view(1, 3, 3)
        return pose

    def reduced_glb_6d_to_full_local_mat_onnx_g2l(self, root_rotation, glb_reduced_pose):
        glb_reduced_pose = r6d_to_rotation_matrix(glb_reduced_pose).view(1, 15, 3, 3)
        global_full_pose = torch.eye(3).repeat(1, 24, 1, 1)
        for index, i in enumerate([1, 2, 3, 4, 5, 6, 9, 12, 13, 14, 15, 16, 17, 18, 19]):
            global_full_pose[:, i] = glb_reduced_pose[:, index]

        global_full_pose[:,0] = root_rotation

        x_local = [global_full_pose[:, 0]]

        for index, i in enumerate([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]):
            x_local.append(
                torch.bmm(torch.transpose(global_full_pose[:, i], dim0=1, dim1=2), global_full_pose[:, index + 1]))
        x_local = torch.stack(x_local, dim=1)
        pose = x_local.view(1, 24, 3, 3)
        for index, i in enumerate([0, 7, 8, 10, 11, 20, 21, 22, 23]):
            pose[:, i] = torch.eye(3)
        pose[:, 0] = root_rotation.view(1, 3, 3)
        return pose

    def reduced_glb_6d_to_full_local_mat_onnx_full(self,root_rotation, glb_full_pose):
        glb_full_pose = glb_full_pose.view(-1,24,3,3)
        x_local = [root_rotation.view(1,3,3)]
        for index, i in enumerate([0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19, 20, 21]):
            x_local.append(
                torch.bmm(torch.transpose(glb_full_pose[:, i], dim0=1, dim1=2), glb_full_pose[:, index + 1]).view(1,3,3))
        x_local = torch.stack(x_local, dim=1)
        pose = x_local.view(1, 24, 3, 3)
        return pose

    def forward_x(self, x, _, __):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 3-tuple
                  (tensor [num_frames, 72], tensor [15], tensor [72]).
        """
        x, lj_init, vel_init = list(zip(*x))
        leaf_joint = self.rnn_leaf(list(zip(x, lj_init)))
        leaf_joint_no_grad = [t.detach() for t in leaf_joint]
        full_joint = self.rnn_full([torch.cat(_, dim=1) for _ in zip(leaf_joint_no_grad, x)])
        full_joint_no_grad = [t.detach() for t in full_joint]
        global_6d_pose = self.rnn_pose([torch.cat(_, dim=1) for _ in zip(full_joint_no_grad, x)])
        stables = self.rnn_stables([torch.cat(_, dim=-1) for _ in zip(full_joint_no_grad, x)])
        vels = self.rnn_vel(list(zip([torch.cat(_, dim=-1) for _ in zip(full_joint_no_grad, x)], vel_init)))
        return leaf_joint, full_joint, global_6d_pose, stables, vels

    def forward(self, glb_acc, glb_rot, root_rot_y, state1_h, state1_c, state2_h, state2_c, state3_h, state3_c,
                state4_h, state4_c,
                state5_h, state5_c):
        imu = utils.normalize_and_concat_11_onnx(glb_acc.view(-1, 11, 3), glb_rot.view(-1, 11, 3, 3), root_rot_y)
        x, state1 = self.rnn_leaf.rnn(relu(self.rnn_leaf.linear1(imu), inplace=True).unsqueeze(0),
                                      (state1_h, state1_c))
        x = self.rnn_leaf.linear2(x[0])
        x = torch.cat([x, imu], dim=1)
        x, state2 = self.rnn_full.rnn(relu(self.rnn_full.linear1(x), inplace=True).unsqueeze(0), (state2_h, state2_c))
        full_joint = self.rnn_full.linear2(x[0])
        x = torch.cat([full_joint, imu], dim=1)

        x1, state3 = self.rnn_pose.rnn(relu(self.rnn_pose.linear1(x), inplace=True).unsqueeze(0),
                                       (state3_h, state3_c))

        global_6d_pose = self.rnn_pose.linear2(x1[0])

        imu_reduced = utils.normalize_and_concat(glb_acc[:, :6].view(-1, 6, 3), glb_rot[:, :6].view(-1, 6, 3, 3)).view(
            -1, 72)
        x = torch.cat([full_joint, imu_reduced], dim=1)

        x1, state4 = self.rnn_v.rnn(relu(self.rnn_v.linear1(x), inplace=True).unsqueeze(0),
                                    (state4_h, state4_c))
        joint_velocity = self.rnn_v.linear2(x1[0])
        x1, state5 = self.rnn_contact.rnn(relu(self.rnn_contact.linear1(x), inplace=True).unsqueeze(0),
                                          (state5_h, state5_c))
        contact = self.rnn_contact.linear2(x1[0])
        root_rot = glb_rot[:, 5].transpose(1, 2)
        joint_velocity_with_root_rotation = joint_velocity.view(1, 24, 3).bmm(root_rot) * vel_scale
        full_local_pose = self.reduced_glb_6d_to_full_local_mat_onnx(glb_rot.view(-1, 11, 3, 3)[:, 5], global_6d_pose)
        return global_6d_pose, joint_velocity, contact, state1[0], state1[1], state2[0], state2[1], state3[0], state3[
            1], state4[0], state4[1], state5[0], state5[
            1], joint_velocity_with_root_rotation, full_local_pose

    def forward_frame_6d(self, glb_acc,glb_rot):
        r"""
        Forward. Currently only support 1 subject.

        :param glb_acc: A tensor in [num_subjects, 11, 3].
        :param glb_rot: A tensor in [num_subjects, 11, 3, 3].
        :return: return (pose).
        """
        imu = utils.normalize_and_concat_11(glb_acc.view(-1, 11, 3), glb_rot.view(-1, 11, 3, 3))
        x, self.rnn_states[1] = self.rnn_leaf.rnn(relu(self.rnn_leaf.linear1(imu), inplace=True).unsqueeze(0),
                                                  self.rnn_states[1])

        leaf = self.rnn_leaf.linear2(x[0])
        x = torch.cat([leaf, imu], dim=1)

        x1, self.rnn_states[2] = self.rnn_full.rnn(relu(self.rnn_full.linear1(x), inplace=True).unsqueeze(0),
                                                   self.rnn_states[2])

        full = self.rnn_full.linear2(x1[0])
        x = torch.cat([full, imu], dim=1)

        x1, self.rnn_states[3] = self.rnn_pose.rnn(relu(self.rnn_pose.linear1(x), inplace=True).unsqueeze(0),
                                                   self.rnn_states[3])
        global_6d_pose = self.rnn_pose.linear2(x1[0])

        x1, self.rnn_states[3] = self.rnn4.rnn(relu(self.rnn_vel.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[3])
        joint_velocity = self.rnn4.linear2(x1[0])

        x1, self.rnn_states[4] = self.rnn5.rnn(relu(self.rnn_stables.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[4])
        contact = self.rnn5.linear2(x1[0])

        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot[:, -1].cpu(), global_6d_pose.cpu())
        joint_velocity = (joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2)) * vel_scale).cpu()
        return pose,joint_velocity,contact

    def forward_without_states(self, x):
        r"""
        Forward.

        :param x: A list in length [batch_size] which contains 3-tuple
                  (tensor [num_frames, 72], tensor [15], tensor [72]).
        """
        x, lj_init, jvel_init = list(zip(*x))

        leaf_joint = self.rnn1(list(zip(x, lj_init)))
        full_joint = self.rnn2([torch.cat(_, dim=-1) for _ in zip(leaf_joint, x)])
        global_6d_pose = self.rnn3([torch.cat(_, dim=-1) for _ in zip(full_joint, x)])
        joint_velocity = self.rnn4(list(zip([torch.cat(_, dim=-1) for _ in zip(full_joint, x)], jvel_init)))
        contact = self.rnn5([torch.cat(_, dim=-1) for _ in zip(full_joint, x)])

        return leaf_joint, full_joint, global_6d_pose, joint_velocity, contact

    def predict(self, glb_acc, glb_rot, init_pose, use_cuda=False):
        r"""
        Predict the results for evaluation.

        :param glb_acc: A tensor that can reshape to [num_frames, 6, 3].
        :param glb_rot: A tensor that can reshape to [num_frames, 6, 3, 3].
        :param init_pose: A tensor that can reshape to [1, 24, 3, 3].
        :return: Pose tensor in shape [num_frames, 24, 3, 3] and
                 translation tensor in shape [num_frames, 3].
        """
        # self.dynamics_optimizer.reset_states()
        init_pose = init_pose.view(1, 24, 3, 3).cpu()
        init_pose[0, 0] = torch.eye(3)
        lj_init = self.forward_kinematics(init_pose)[1][0, joint_set.leaf].view(-1)
        jvel_init = torch.zeros(24 * 3)
        if (use_cuda):
            lj_init = lj_init.cuda()
            jvel_init = jvel_init.cuda()
        x = (normalize_and_concat(glb_acc, glb_rot), lj_init, jvel_init)
        leaf_joint, full_joint, global_6d_pose, joint_velocity, contact = [_[0] for _ in
                                                                           self.forward_without_states([x])]
        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot.view(-1, 6, 3, 3)[:, -1], global_6d_pose)
        joint_velocity = joint_velocity.view(-1, 24, 3).bmm(glb_rot[:, -1].transpose(1, 2)) * vel_scale
        pose_opt, tran_opt = [], []

        for p, v, c, a in zip(pose, joint_velocity, contact, glb_acc):
            p, t = self.dynamics_optimizer.optimize_frame(p, v, c, a)
            pose_opt.append(p)
            tran_opt.append(t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)
        return pose_opt, tran_opt

    def forward_frame(self, glb_acc, glb_rot):
        r"""
        Forward. Currently only support 1 subject.

        :param glb_acc: A tensor in [num_subjects, 6, 3].
        :param glb_rot: A tensor in [num_subjects, 6, 3, 3].
        """
        imu = utils.normalize_and_concat_11(glb_acc, glb_rot)

        x, self.rnn_states[1] = self.rnn2.rnn(relu(self.rnn2.linear1(imu), inplace=True).unsqueeze(0),
                                              self.rnn_states[1])
        full_joint = self.rnn2.linear2(x[0])
        x = torch.cat([full_joint, imu], dim=1)

        x1, self.rnn_states[2] = self.rnn3.rnn(relu(self.rnn3.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[2])
        global_6d_pose = self.rnn3.linear2(x1[0])
        imu_reduced = utils.normalize_and_concat(glb_acc[:, :6].view(-1, 6, 3), glb_rot[:, :6].view(-1, 6, 3, 3)).view(
            -1, 72)

        x = torch.cat([full_joint, imu_reduced], dim=1)
        x1, self.rnn_states[3] = self.rnn4.rnn(relu(self.rnn4.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[3])
        joint_velocity = self.rnn4.linear2(x1[0])
        x = torch.cat([full_joint, imu_reduced], dim=1)
        x1, self.rnn_states[4] = self.rnn5.rnn(relu(self.rnn5.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[4])
        contact = self.rnn5.linear2(x1[0])

        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot[:, 5].cpu(), global_6d_pose.cpu())
        joint_velocity = (joint_velocity.float().view(-1, 24, 3).bmm(
            (glb_rot[:, -1].transpose(1, 2)).float() * vel_scale).float()).cpu()
        pose_opt, tran_opt = [], []
        for p, v, c, a in zip(pose.cpu(), joint_velocity.cpu(), contact.cpu(), glb_acc):
            p, t = self.dynamics_optimizer.optimize_frame(p, v, c)
            pose_opt.append(p)
            tran_opt.append(t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)
        return pose_opt, tran_opt, contact

    def forward_frame_wo_optimize(self, glb_acc, glb_rot):
        r"""
        Forward. Currently only support 1 subject.

        :param glb_acc: A tensor in [num_subjects, 6, 3].
        :param glb_rot: A tensor in [num_subjects, 6, 3, 3].
        """
        imu = normalize_and_concat(glb_acc, glb_rot)

        x, self.rnn_states[0] = self.rnn1.rnn(relu(self.rnn1.linear1(imu), inplace=True).unsqueeze(0),
                                              self.rnn_states[0])
        x = self.rnn1.linear2(x[0])
        x = torch.cat([x, imu], dim=1)

        x, self.rnn_states[1] = self.rnn2.rnn(relu(self.rnn2.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[1])
        x = self.rnn2.linear2(x[0])
        x = torch.cat([x, imu], dim=1)

        x1, self.rnn_states[2] = self.rnn3.rnn(relu(self.rnn3.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[2])
        global_6d_pose = self.rnn3.linear2(x1[0])

        x1, self.rnn_states[3] = self.rnn4.rnn(relu(self.rnn4.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[3])
        joint_velocity = self.rnn4.linear2(x1[0])

        x1, self.rnn_states[4] = self.rnn5.rnn(relu(self.rnn5.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[4])
        contact = self.rnn5.linear2(x1[0])

        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot[:, -1].cpu(), global_6d_pose.cpu())
        joint_velocity = (joint_velocity.float().view(-1, 24, 3).bmm(
            (glb_rot[:, -1].transpose(1, 2)).float() * vel_scale).float()).cpu()

        return pose, joint_velocity, contact

    def forward_frame_full_result(self, glb_acc, glb_rot):
        r"""
        Forward. Currently only support 1 subject.

        :param glb_acc: A tensor in [num_subjects, 6, 3].
        :param glb_rot: A tensor in [num_subjects, 6, 3, 3].
        """
        imu = normalize_and_concat(glb_acc, glb_rot)

        x, self.rnn_states[0] = self.rnn1.rnn(relu(self.rnn1.linear1(imu), inplace=True).unsqueeze(0),
                                              self.rnn_states[0])
        x = self.rnn1.linear2(x[0])
        x = torch.cat([x, imu], dim=1)

        x, self.rnn_states[1] = self.rnn2.rnn(relu(self.rnn2.linear1(x), inplace=True).unsqueeze(0), self.rnn_states[1])
        x = self.rnn2.linear2(x[0])
        x = torch.cat([x, imu], dim=1)

        x1, self.rnn_states[2] = self.rnn3.rnn(relu(self.rnn3.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[2])
        global_6d_pose = self.rnn3.linear2(x1[0])

        x1, self.rnn_states[3] = self.rnn4.rnn(relu(self.rnn4.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[3])
        joint_velocity = self.rnn4.linear2(x1[0])

        x1, self.rnn_states[4] = self.rnn5.rnn(relu(self.rnn5.linear1(x), inplace=True).unsqueeze(0),
                                               self.rnn_states[4])
        contact = self.rnn5.linear2(x1[0])

        pose = self._reduced_glb_6d_to_full_local_mat(glb_rot[:, -1].cpu(), global_6d_pose.cpu())
        joint_velocity = (joint_velocity.float().view(-1, 24, 3).bmm(
            (glb_rot[:, -1].transpose(1, 2)).float() * vel_scale).float()).cpu()
        pose_opt, tran_opt = [], []
        for p, v, c, a in zip(pose, joint_velocity, contact, glb_acc):
            p, t = self.dynamics_optimizer.optimize_frame(p, v, c)
            pose_opt.append(p)
            tran_opt.append(t)
        pose_opt, tran_opt = torch.stack(pose_opt), torch.stack(tran_opt)
        return pose_opt, tran_opt, contact, pose, joint_velocity



