#!/usr/bin/env python3
"""
trajectory_tracking.py
----------------------
Use a P controller to track a sequence of 3D waypoints.

Usage:
    ros2 run nsysu_drone_control trajectory_tracking
    ros2 run nsysu_drone_control trajectory_tracking --trajectory-file /ros2_ws/RL/trajectory1.txt

Assumptions:
    1. Gazebo simulation is running.
    2. The drone has already taken off.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import rclpy
from rclpy import utilities
from rclpy.node import Node

from geometry_msgs.msg import Pose, Twist


class TrajectoryTracking(Node):
    """Track a waypoint trajectory with a simple P controller."""

    def __init__(self, trajectory_file: Path, kp: float, max_speed: float,
                 tolerance: float, timer_period: float, loop: bool):
        super().__init__('trajectory_tracking_controller')

        self.trajectory_file = trajectory_file
        self.kp = kp
        self.max_speed = max_speed
        self.tolerance = tolerance
        self.loop = loop

        self.waypoints = self.load_waypoints(self.trajectory_file)
        if not self.waypoints:
            raise ValueError(f'No waypoints found in {self.trajectory_file}')

        self.current_x = None
        self.current_y = None
        self.current_z = None

        self.target_index = 0
        self.finished = False

        self.cmd_vel_pub = self.create_publisher(Twist, '/simple_drone/cmd_vel', 10)
        self.pose_sub = self.create_subscription(Pose, '/simple_drone/gt_pose', self.pose_callback, 10)
        self.timer = self.create_timer(timer_period, self.control_loop)

        first_waypoint = self.waypoints[0]
        last_waypoint = self.waypoints[-1]
        self.get_logger().info(
            f'Trajectory loaded from {self.trajectory_file} with {len(self.waypoints)} waypoints.'
        )
        self.get_logger().info(
            f'Start target: ({first_waypoint[0]:.2f}, {first_waypoint[1]:.2f}, {first_waypoint[2]:.2f})'
        )
        self.get_logger().info(
            f'Final target: ({last_waypoint[0]:.2f}, {last_waypoint[1]:.2f}, {last_waypoint[2]:.2f})'
        )

    def load_waypoints(self, file_path: Path) -> list[tuple[float, float, float]]:
        """Load the trajectory file and keep only the first three columns."""
        waypoints: list[tuple[float, float, float]] = []
        with file_path.open('r', encoding='utf-8') as file_handle:
            for line_number, raw_line in enumerate(file_handle, start=1):
                stripped_line = raw_line.strip()
                if not stripped_line or stripped_line.startswith('#'):
                    continue

                parts = stripped_line.split()
                if len(parts) < 3:
                    raise ValueError(
                        f'Expected at least 3 columns in {file_path} on line {line_number}, '
                        f'got {len(parts)}'
                    )

                waypoints.append((float(parts[0]), float(parts[1]), float(parts[2])))

        return waypoints

    def pose_callback(self, msg: Pose):
        self.current_x = msg.position.x
        self.current_y = msg.position.y
        self.current_z = msg.position.z

    def control_loop(self):
        if self.current_x is None:
            return

        if self.finished:
            self.publish_velocity(0.0, 0.0, 0.0)
            return

        while True:
            target_x, target_y, target_z = self.waypoints[self.target_index]
            error_x = target_x - self.current_x
            error_y = target_y - self.current_y
            error_z = target_z - self.current_z
            distance = math.sqrt(error_x**2 + error_y**2 + error_z**2)

            if distance >= self.tolerance:
                break

            if self.target_index >= len(self.waypoints) - 1:
                if self.loop:
                    self.get_logger().info('Loop enabled, restarting trajectory from the first waypoint.')
                    self.target_index = 0
                    continue

                self.finished = True
                self.get_logger().info(
                    f'✅ Trajectory completed at ({self.current_x:.2f}, '
                    f'{self.current_y:.2f}, {self.current_z:.2f})'
                )
                self.publish_velocity(0.0, 0.0, 0.0)
                return

            self.target_index += 1
            self.get_logger().info(
                f'Waypoint reached, switching to {self.target_index + 1}/{len(self.waypoints)}'
            )

        vx = self.kp * error_x
        vy = self.kp * error_y
        vz = self.kp * error_z
        vx, vy, vz = self.clamp_velocity(vx, vy, vz)
        self.publish_velocity(vx, vy, vz)

        self.get_logger().info(
            f'Target {self.target_index + 1}/{len(self.waypoints)} | '
            f'Pose: ({self.current_x:.2f}, {self.current_y:.2f}, {self.current_z:.2f}) | '
            f'Error: ({error_x:.2f}, {error_y:.2f}, {error_z:.2f}) | '
            f'Speed: ({vx:.2f}, {vy:.2f}, {vz:.2f})',
            throttle_duration_sec=1.0,
        )

    def publish_velocity(self, vx: float, vy: float, vz: float):
        msg = Twist()
        msg.linear.x = vx
        msg.linear.y = vy
        msg.linear.z = vz
        msg.angular.x = 0.0
        msg.angular.y = 0.0
        msg.angular.z = 0.0
        self.cmd_vel_pub.publish(msg)

    def clamp_velocity(self, vx: float, vy: float, vz: float):
        speed = math.sqrt(vx**2 + vy**2 + vz**2)
        if speed > self.max_speed:
            scale = self.max_speed / speed
            vx *= scale
            vy *= scale
            vz *= scale
        return vx, vy, vz


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Trajectory tracker using P control')
    parser.add_argument(
        '--trajectory-file',
        type=Path,
        default=Path(__file__).resolve().parents[3] / 'RL' / 'trajectory1.txt',
        help='Path to a text file containing trajectory waypoints',
    )
    parser.add_argument('--kp', type=float, default=0.5, help='Proportional gain')
    parser.add_argument('--max-speed', type=float, default=1.0, help='Maximum speed in m/s')
    parser.add_argument('--tolerance', type=float, default=0.25, help='Waypoint tolerance in m')
    parser.add_argument('--timer-period', type=float, default=0.1, help='Control loop period in seconds')
    parser.add_argument('--loop', action='store_true', help='Restart from the first waypoint when finished')
    return parser.parse_args()


def main(args=None):
    rclpy.init(args=args)
    parsed_args = parse_args(utilities.remove_ros_args(sys.argv[1:]))

    node = TrajectoryTracking(
        trajectory_file=parsed_args.trajectory_file.expanduser().resolve(),
        kp=parsed_args.kp,
        max_speed=parsed_args.max_speed,
        tolerance=parsed_args.tolerance,
        timer_period=parsed_args.timer_period,
        loop=parsed_args.loop,
    )

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Stopping trajectory tracker.')
    finally:
        node.publish_velocity(0.0, 0.0, 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()