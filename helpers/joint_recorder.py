#!/usr/bin/env python3
# 41069 Robotics Studio 2
# Authors: Jayden Hazell (24953364), Lachlan Partridge (24964557), Tomas Klimes (24773804)
# Interactive ROS2 node that saves the current UR3e joint positions to a CSV file on keypress.
"""
joint_recorder.py

Subscribes to /joint_states and appends the most recently received joint
positions to joints.csv when the operator presses 'q'. Used for manually
recording waypoints for the movement node.
"""

import csv
import select
import sys
import termios
import tty
from datetime import datetime

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState


class JointRecorder(Node):
    def __init__(self):
        super().__init__('joint_recorder')

        self.latest_joint_state = None

        self.sub = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_callback,
            10
        )

        self.file_path = 'joints.csv'

        self.get_logger().info("Joint recorder running.")
        self.get_logger().info("Move the robot, then press 'q' to save current joint angles.")
        self.get_logger().info("Press Ctrl+C to exit.")

    def joint_callback(self, msg):
        self.latest_joint_state = msg

    def save_joint_state(self):
        if self.latest_joint_state is None:
            self.get_logger().warn("No joint state received yet.")
            return

        with open(self.file_path, 'a', newline='') as file:
            writer = csv.writer(file)

            row = [
                datetime.now().isoformat(),
                *self.latest_joint_state.name,
                *self.latest_joint_state.position
            ]

            writer.writerow(row)

        self.get_logger().info(f"Saved joint state to {self.file_path}")
        self.get_logger().info(str(self.latest_joint_state.position))


def key_pressed():
    dr, _, _ = select.select([sys.stdin], [], [], 0.1)
    if dr:
        return sys.stdin.read(1)
    return None


def main(args=None):
    rclpy.init(args=args)
    node = JointRecorder()

    old_settings = termios.tcgetattr(sys.stdin)
    tty.setcbreak(sys.stdin.fileno())

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.1)

            key = key_pressed()
            if key == 'q':
                node.save_joint_state()

    except KeyboardInterrupt:
        pass

    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old_settings)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()