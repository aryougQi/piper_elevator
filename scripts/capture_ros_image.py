#!/usr/bin/env python3
"""Save one ROS Image message for repeatable headless simulation debugging."""

import argparse
from pathlib import Path
import time

import cv2
from cv_bridge import CvBridge
import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy
from rclpy.qos import HistoryPolicy
from rclpy.qos import QoSProfile
from rclpy.qos import ReliabilityPolicy
from sensor_msgs.msg import Image


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    parser.add_argument(
        '--topic',
        default='/camera/color/image_raw',
    )
    parser.add_argument('--timeout', type=float, default=10.0)
    arguments = parser.parse_args()

    rclpy.init()
    node = Node('capture_ros_image_once')
    bridge = CvBridge()
    captured = False

    def callback(message):
        nonlocal captured
        image = bridge.imgmsg_to_cv2(message, desired_encoding='bgr8')
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(arguments.output), image):
            raise RuntimeError(f'Could not write {arguments.output}')
        captured = True

    qos = QoSProfile(
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.VOLATILE,
    )
    subscription = node.create_subscription(
        Image,
        arguments.topic,
        callback,
        qos,
    )
    deadline = time.monotonic() + arguments.timeout
    while not captured and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.1)
    node.destroy_node()
    del subscription
    rclpy.shutdown()
    if not captured:
        raise RuntimeError(
            f'No image received from {arguments.topic} in '
            f'{arguments.timeout:.1f}s'
        )
    print(arguments.output)


if __name__ == '__main__':
    main()
