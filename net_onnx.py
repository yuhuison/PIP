import numpy
import onnxruntime
import torch
from dynamics import PhysicsOptimizer, smpl_to_rbdl
from utils import set_pose
from onnx_opcounter import calculate_params

class ONNX_PIP:
    def __init__(self, model_filename):
        #sess_options = onnxruntime.SessionOptions()
        # sess_options.enable_profiling = True
        self.net = onnxruntime.InferenceSession(model_filename)
        self.state1_h = numpy.zeros((2, 1, 256)).astype(numpy.float32)
        self.state2_h = numpy.zeros((2, 1, 256)).astype(numpy.float32)
        self.state3_h = numpy.zeros((2, 1, 256)).astype(numpy.float32)
        self.state4_h = numpy.zeros((2, 1, 256)).astype(numpy.float32)
        self.state5_h = numpy.zeros((2, 1, 64)).astype(numpy.float32)
        self.state1_c = numpy.zeros((2, 1, 256)).astype(numpy.float32)
        self.state2_c = numpy.zeros((2, 1, 256)).astype(numpy.float32)
        self.state3_c = numpy.zeros((2, 1, 256)).astype(numpy.float32)
        self.state4_c = numpy.zeros((2, 1, 256)).astype(numpy.float32)
        self.state5_c = numpy.zeros((2, 1, 64)).astype(numpy.float32)
        self.opt = PhysicsOptimizer(False)

    def forward_frame(self, glb_acc, glb_rot):
        inputs = {
            "glb_acc": glb_acc.view(1, 11, 3).numpy().astype(numpy.float32),
            "glb_rot": glb_rot.view(1, 11, 3, 3).numpy().astype(numpy.float32),
            "state1_h.1": self.state1_h,
            "state2_h.1": self.state2_h,
            "state3_h.1": self.state3_h,
            "state4_h.1": self.state4_h,
            "state5_h.1": self.state5_h,
            "state1_c.1": self.state1_c,
            "state2_c.1": self.state2_c,
            "state3_c.1": self.state3_c,
            "state4_c.1": self.state4_c,
            "state5_c.1": self.state5_c,
        }
        output_names = ["pose", "joint_velocity", "contact", "state1_h", "state1_c", "state2_h", "state2_c",
                        "state3_h", "state3_c", "state4_h", "state4_c", "state5_h", "state5_c",
                        "joint_velocity_with_root_rotation", "full_local_pose"]
        output = self.net.run(output_names, inputs)
        result_dict = dict(zip(output_names, output))
        self.state1_h = result_dict["state1_h"]
        self.state2_h = result_dict["state2_h"]
        self.state3_h = result_dict["state3_h"]
        self.state4_h = result_dict["state4_h"]
        self.state5_h = result_dict["state5_h"]
        self.state1_c = result_dict["state1_c"]
        self.state2_c = result_dict["state2_c"]
        self.state3_c = result_dict["state3_c"]
        self.state4_c = result_dict["state4_c"]
        self.state5_c = result_dict["state5_c"]

        full_local_pose = torch.tensor(result_dict["full_local_pose"])

        joint_velocity_with_root_rotation = torch.tensor(result_dict["joint_velocity_with_root_rotation"])
        contact = torch.tensor(result_dict["contact"])

        pose, tran = self.opt.optimize_frame(full_local_pose[0], joint_velocity_with_root_rotation[0], contact[0].cpu(),
                                             glb_acc.cpu())


        return pose,tran

    def set_pose(self, pose,trans = None):
        set_pose(self.opt.id_robot, smpl_to_rbdl(pose, torch.tensor([0, 0, 0]) if trans is None else trans)[0])


@torch.no_grad()
def test():
    net = ONNX_PIP("models/MeoCapNetV6.onnx")
    r = net.forward_frame(torch.zeros(33), torch.stack([torch.eye(3) for _ in range(11)]))
    print(r)


if __name__ == "__main__":
    test()
