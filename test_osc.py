import numpy as np
from scipy.spatial.transform import Rotation

from vmc_stream_client import MeocapVMC_Client
from pygame.time import Clock
client = MeocapVMC_Client(39539,"")
client.start_listen()
clock = Clock()
frames = []


def multiply_rotation_matrices(A, B):
  """
  Multiply each rotation matrix in group A by the inverse of each corresponding
  rotation matrix in group B.

  Args:
  A (np.ndarray): A group of 24 3x3 rotation matrices.
  B (np.ndarray): Another group of 24 3x3 rotation matrices.

  Returns:
  np.ndarray: A new group of 24 3x3 rotation matrices, each being the product
              of a matrix from A and the inverse of the corresponding matrix from B.
  """
  # Initialize an array to hold the results
  result = np.zeros_like(A)

  # For each pair of matrices
  for i in range(A.shape[0]):
    # Multiply matrix from A with the inverse of the corresponding matrix from B
    result[i] = np.dot(A[i], np.transpose(B[i]))
  return result


def on_data_update():
  if client.pose_data is not None:
    # pose-data is ndarray (24,3,3)

    if len(frames) == 2:
      frames.pop(0)

    frames.append(client.pose_data)

    if len(frames) == 2:
      diff_rotation = multiply_rotation_matrices(frames[0],frames[1])
      euler = Rotation.from_matrix(diff_rotation).as_euler("xyz")
      print(euler)

client.on_update_func = on_data_update

