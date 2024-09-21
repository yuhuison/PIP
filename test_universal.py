from dataclasses import dataclass
from typing import List, Tuple
import json
import zmq
from pygame.time import Clock


@dataclass
class PlainTrackerReport:
    rotation: Tuple[float, float, float, float]
    timestamp: int
    acc: Tuple[float, float, float]


@dataclass
class UniversalFrame:
    raw_imus: List[PlainTrackerReport]
    calibrated_imus: List[PlainTrackerReport]
    raw_pose: List[float]
    optimized_pose: List[float]
    translation: List[float]
    joint_positions: List[float]
    joint_velocity: List[float]
    contact: List[float]


def subscribe_meocap_universal(url: str = "tcp://127.0.0.1:14999"):
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.connect(url)
    socket.subscribe('')  # 订阅所有消息
    print("Subscribing meocap")
    clock = Clock()
    while True:
        clock.tick(60)
        message = socket.recv_string()
        data = json.loads(message)
        frame = UniversalFrame(
            raw_imus=[PlainTrackerReport(imu['rotation'], imu['timestamp'], imu['acc']) for imu in data['raw_imus']],
            calibrated_imus=[PlainTrackerReport(imu['rotation'], imu['timestamp'], imu['acc']) for imu in data['calibrated_imus']],
            raw_pose=data['raw_pose'],
            optimized_pose=data['optimized_pose'],
            translation=data['translation'],
            joint_positions=data['joint_positions'],
            joint_velocity=data['joint_velocity'],
            contact=data['contact']
        )
        print(frame.optimized_pose)


if __name__ == "__main__":
    subscribe_meocap_universal()
