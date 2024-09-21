import numpy as np
from scipy.spatial.transform import Rotation as R

class RotationFilter3D:
    def __init__(self, p, d):
        # 初始化旋转为单位旋转
        self.rotation_last = R.from_euler('xyz', [0, 0, 0], degrees=True)
        self.rotation_diff_last = R.from_euler('xyz', [0, 0, 0], degrees=True)
        self.p_weight = p
        self.d_weight = d

    def update(self, rotation_now):
        # 计算旋转差和差的差
        rotation_diff = self.rotation_last.inv() * rotation_now
        rotation_diff_diff = self.rotation_diff_last.inv() * rotation_diff

        # 获取欧拉角并应用PD权重
        euler_diff = rotation_diff.as_euler('xyz', degrees=True)
        euler_diff_diff = rotation_diff_diff.as_euler('xyz', degrees=True)

        euler_filtered = (np.array(euler_diff) * self.p_weight +
                          np.array(euler_diff_diff) * self.d_weight)

        # 更新滤波后的旋转
        filtered_diff = R.from_euler('xyz', euler_filtered, degrees=True)
        filtered_rotation = self.rotation_last * filtered_diff

        # 更新状态
        self.rotation_last = filtered_rotation
        self.rotation_diff_last = rotation_diff
        #print(filtered_rotation.as_euler("xyz",degrees=True))

        return filtered_rotation




