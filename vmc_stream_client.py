import threading

import numpy as np
from pythonosc.dispatcher import Dispatcher
from pythonosc import osc_server
from scipy.spatial.transform import Rotation

smpl_node_to_vrm_node = ['Hips', 'LeftUpperLeg', 'RightUpperLeg', 'Spine', 'LeftLowerLeg', 'RightLowerLeg',
                         'Chest', 'LeftFoot', 'RightFoot', '-', '-', '-', 'Neck', 'LeftShoulder',
                         'RightShoulder', 'Head', 'LeftUpperArm', 'RightUpperArm', 'LeftLowerArm',
                         'RightLowerArm', 'LeftHand', 'RightHand', '-', '-']


class MeocapVMC_Client:
    def __init__(self, port: int, name: str):
        self.port = port
        dispatcher = Dispatcher()
        dispatcher.map("/VMC/Ext/Root/Pos", self.root_data_handler)
        dispatcher.map("/VMC/Ext/Bone/Pos", self.bone_data_handler)
        dispatcher.map("/VMC/Ext/OK", self.update_handler)
        dispatcher.map("/Meocap/Tracker", self.tracker_data_handler)
        self.osc_socket = osc_server.ThreadingOSCUDPServer(
            ("127.0.0.1", port), dispatcher)
        self.bone_data = dict()
        self.tracker_data = dict()
        self.root_rotation = None
        self.root_position = None
        self.receive_thread = None
        self.pose_data = None
        self.count = 0
        self.name = name
        self.on_update_func = None

    def tracker_data_handler(self, *args):
        self.tracker_data[args[1]] = args[2:9]

    def bone_data_handler(self, *args):
        if args[1] in smpl_node_to_vrm_node:
            euler = Rotation.from_quat(args[5:9]).as_euler("xyz")
            euler[2] = -euler[2]
            euler[1] = -euler[1]
            self.bone_data[args[1]] = Rotation.from_euler("xyz", euler).as_matrix()

    def root_data_handler(self, *args):
        pass
        # self.root_rotation = Rotation.from_quat(args[5:9]).as_matrix()
        # self.root_position = args[2:5]

    def on_data_update(self, func):
        self.on_update_func = func

    def update_handler(self, *arg):
        arr = np.empty((24, 3, 3))
        for i, name in enumerate(smpl_node_to_vrm_node):
            if name in self.bone_data.keys():
                arr[i] = self.bone_data[name]
            else:
                arr[i] = np.eye(3)
        self.pose_data = arr
        self.count = self.count + 1
        if self.on_update_func is not None:
            self.on_update_func()

    def start_listen(self):
        def receive_data():
            self.osc_socket.serve_forever()

        receive_thread = threading.Thread(target=receive_data)
        receive_thread.start()
