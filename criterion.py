import torch


class MeoCapNetLossStageV4FullJoint(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.l2lLoss = torch.nn.SmoothL1Loss()
        self.bceloss = torch.nn.BCEWithLogitsLoss()

    def forward(self, x, y, refine):
        l_j, f_j, pose, jv, stables = x
        l_j_y, f_j_y, pose_y, jv_y, stables_y = y
        if True:
            loss_dict = {
                "pose": self.l2lLoss(pose.view(-1, 90), pose_y.view(-1, 90)),
                "vel": self.l2lLoss(jv.view(-1, 72), jv_y.view(-1, 72)),
                "full_joint": self.l2lLoss(f_j.view(-1, 69), f_j_y.view(-1, 69)),
                "leaf_joint": self.l2lLoss(l_j.view(-1, 15), l_j_y.view(-1, 15)),
                "contact": self.bceloss(stables.view(-1, 2), stables_y.view(-1, 2)),
            }
        return loss_dict

    def backward(self, loss_dict):
        [v.backward() for k, v in loss_dict.items()]
