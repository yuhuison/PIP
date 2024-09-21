import numpy as np
from scipy.spatial.transform import Rotation
import torch
from articulate.math import rotation_matrix_to_axis_angle

r1 = Rotation.from_quat([0.321581, -0.6444437, 0.14940792, 0.6774624])
r2 = Rotation.from_quat([-0.52314484, 0.8521933, -0.009216529, 0.0010986591])
R = r1.as_matrix().transpose() @ r2.as_matrix()
print(rotation_matrix_to_axis_angle(torch.tensor(R)))
print(Rotation.from_matrix(R).as_euler("xyz",degrees=True))