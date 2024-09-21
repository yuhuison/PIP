import tqdm
from torch.utils.data import DataLoader, Dataset
import torch
import config
import utils
from articulate.math import axis_angle_to_rotation_matrix
from config import paths, joint_set
import os
import articulate as art


class MeoCapDatasetsV3(Dataset):
    def __init__(self, filepath="", full=False,version=''):
        super(MeoCapDatasetsV3, self).__init__()
        self.datas = []
        if filepath != "":
            self.datas = torch.load(filepath)
            if not full:
                if len(self.datas[0]) != 3:
                    if version == "v6":
                        self.datas = [(data[0], data[1], data[2], data[4], data[7], data[5][1].view(200, 24, 3),data[5][0].view(200,3)) for
                                      data in
                                      self.datas]
                    else:
                        self.datas = [(data[0], data[1], data[2], data[4], data[7], data[5][1].view(200, 24, 3)) for
                                      data in
                                      self.datas]


    def append_from_dataset(self, filepath=""):
        self.datas = self.datas + [(data[0], data[2], data[4], data[5][1]) for data in torch.load(filepath)]

    def load_from_pure_dataset(self, filepath, need_convert_axis_angle=True, need_mirrored=True, need_virtual=False):
        self.data = torch.load(filepath)
        self.model = art.ParametricModel(paths.male_smpl_file)
        self.datas = []
        self.len_no_mirror = len(self.data["acc"])
        print("DataSet Init")
        for idx_d in tqdm.tqdm(range(self.len_no_mirror)):
            idx = idx_d
            if need_virtual:
                ori = self.data["v_rot"][idx]
                acc = self.data["v_acc"][idx]
            else:
                ori = self.data["ori"][idx]
                acc = self.data["acc"][idx]
            if "tran" in self.data.keys():
                tran = self.data["tran"][idx]
            else:
                tran = torch.zeros(ori.shape[0] * 3).view(-1, 3)

            if need_convert_axis_angle:
                pose_mtx = axis_angle_to_rotation_matrix(self.data["pose"][idx].view(-1, 3)).view(-1, 24, 3, 3)
            else:
                pose_mtx = self.data["pose"][idx].view(-1, 24, 3, 3).to(torch.float32)

            pose_global, joint_global = self.model.forward_kinematics(pose_mtx.view(-1, 24, 3, 3), None,
                                                                      None)
            _, joint_global_with_tran = self.model.forward_kinematics(pose_mtx.view(-1, 24, 3, 3), None,
                                                                      tran.view(-1, 3))

            pose_mtx_no_root_rotation = torch.einsum("nij,nkjm->nkim", pose_global[:, 0].transpose(1, 2),
                                                     pose_global).contiguous()

            pose_g6d = art.math.rotation_matrix_to_r6d(pose_mtx_no_root_rotation.view(-1, 3, 3)).reshape(-1, 24, 6)[:,
                       joint_set.reduced]

            pose_local = self.model.inverse_kinematics_R(pose_mtx_no_root_rotation)

            pose_global, joint_global = self.model.forward_kinematics(pose_local.view(-1, 24, 3, 3), None,
                                                                      None)

            nn_jtr = joint_global - joint_global[:, :1]
            leaf_jtr = nn_jtr[:, joint_set.leaf]
            full_jtr = nn_jtr[:, joint_set.full]
            imu = utils.normalize_and_concat_11(acc.view(-1, 11, 3), ori.view(-1, 11, 3, 3)).view(-1, 132)
            velocity = tran
            root_ori = ori[:, 5]  # 最后一组为胯部
            velocity_local = root_ori.transpose(1, 2).bmm(
                torch.cat([velocity[1:] - velocity[:-1], (velocity[-1] + velocity[-2] - velocity[
                    -3]).view(-1, 3)]).unsqueeze(-1)) / 0.016667 / 3

            acc_last_frame = (joint_global_with_tran[-1] - joint_global_with_tran[
                -2]) - (joint_global_with_tran[-3] - joint_global_with_tran[
                -4])

            joint_v = torch.cat([joint_global_with_tran[1:] - joint_global_with_tran[0:-1],
                                 ((joint_global_with_tran[-1] - joint_global_with_tran[
                                     -2]) + acc_last_frame).view(-1, 24, 3)], dim=0)

            joint_v = joint_v.view(-1, 24, 3).bmm(root_ori)

            stable_threshold = 0.008
            stable = (joint_v[:, :].norm(dim=2) < stable_threshold).float()
            joint_v = (joint_v / 0.016667 / 3)
            data = imu, leaf_jtr, full_jtr, pose_mtx, pose_g6d, (tran, joint_v, velocity_local.view(-1, 1, 3)), (
                acc, ori), stable
            self.datas.append(data)
        print("Init Finish")

    def __getitem__(self, idx):
        return self.datas[idx]

    def __len__(self):
        return len(self.datas)


class MeocapDatasetV3Mixed(Dataset):
    def __init__(self, datasets: list[str],version=""):
        self.datasets = [MeoCapDatasetsV3(fn,version=version) for fn in datasets]
        self.lens = [len(d) for d in self.datasets]

    def __len__(self):
        return sum(self.lens)

    def __getitem__(self, idx):
        for i, l in enumerate(self.lens):
            if idx < l:
                return self.datasets[i][idx]
            idx -= l
        raise IndexError('Index out of range')


if __name__ == "__main__":
    print("Loading Dataset")
    dataset = MeoCapDatasetsV3()

    dataset.load_from_pure_dataset(os.path.join(config.paths.moyo_dir, 'train_split_11.pt'), True, False, False)

    torch.save(dataset.datas, "G:\Projects\Meocap-net\moyo_train_spilt_no_mirror_full_11joint.dataset")
    # torch.save(dataset.datas, "dip_train_spilt_no_mirror_full_11joint.dataset")
    # dataset.append_from_dataset("amass_train_spilt_no_mirror_full_11joint.dataset")
    # dataset.append_from_dataset("amass_train_spilt_mirrored_full_11joint.dataset")
    # dataset.load_from_pure_dataset("freestyle_11_11_11joint.pure_dataset", False, False ,True)
    # torch.save(dataset.datas, "amass_train_spilt_mixed_full_11joint.dataset")
    print("Loaded Dataset")
