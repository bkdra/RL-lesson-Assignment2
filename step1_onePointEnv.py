import math
import os
import gc

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from geometry_msgs.msg import Twist, Pose
from std_msgs.msg import Empty
from rclpy.node import Node
import rclpy
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import time
import yaml
import xacro
from ament_index_python.packages import get_package_share_directory
from gazebo_msgs.srv import SpawnEntity, DeleteEntity
from std_srvs.srv import Empty as EmptySrv

DT = 0.025  # 每一步的時間間隔 (秒)
UPDATE_DISTANCE = 0.1  # 無人機距離當前目標點多近時切換到下一個目標點

class DroneROSInterface(Node):
    """把 ROS 2 的 pub/sub 包裝成簡單的 get/set 介面。"""

    def __init__(self):
        super().__init__('rl_drone_interface')

        self.current_pose = np.zeros(3)  # [x, y, z]
        self.current_vel = np.zeros(3)   # [vx, vy, vz]
        self.current_orientation = np.zeros(4)  # [x, y, z, w]
        self.current_orientation[3] = 1.0  # 初始為單位四元數
        self.last_pose_time = None

        self.last_obs = np.zeros(9)  # 上一個 observation
        self.curr_obs = np.zeros(9)  # 當前 observation

        self.cmd_vel_pub = self.create_publisher(
            Twist, '/simple_drone/cmd_vel', 10
        )
        self.takeoff_pub = self.create_publisher(
            Empty, '/simple_drone/takeoff', 10
        )
        self.reset_pub = self.create_publisher(
            Empty, '/simple_drone/reset', 10
        )
        self.pose_sub = self.create_subscription(
            Pose, '/simple_drone/gt_pose', self._pose_cb, 10
        )

        self.spawn_cli = self.create_client(SpawnEntity, '/spawn_entity')
        self.delete_cli = self.create_client(DeleteEntity, '/delete_entity')
        self.reset_world_cli = self.create_client(EmptySrv, '/reset_world')

        self._robot_xml = None
        self._entity_name = None
        

    def _pose_cb(self, msg: Pose):
        new_pose = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.current_orientation = np.array([
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ])
        self.last_pose_time = time.monotonic()
        # 簡易速度估計 (差分法)
        self.current_vel = (new_pose - self.current_pose) * 40  # 約 40Hz
        self.current_pose = new_pose

    def send_velocity(self, vx, vy, vz, x, y, z):
        if not rclpy.ok():
            return
        msg = Twist()
        msg.linear.x, msg.linear.y, msg.linear.z = float(vx), float(vy), float(vz)
        msg.angular.x, msg.angular.y, msg.angular.z = float(x), float(y), float(z)
        self.cmd_vel_pub.publish(msg)

    def reset_drone(self):
        self.reset_pub.publish(Empty())
        # 稍等一下後起飛
        rclpy.spin_once(self, timeout_sec=0.5)
        self.takeoff_pub.publish(Empty())
        rclpy.spin_once(self, timeout_sec=1.0)

    def wait_for_pose(self, timeout_sec=1.0):
        start = time.monotonic()
        last_seen = self.last_pose_time
        while time.monotonic() - start < timeout_sec:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.last_pose_time is not None and self.last_pose_time != last_seen:
                return True
        return False
    

    def reset_world(self):
        req = EmptySrv.Request()

        if self.reset_world_cli.wait_for_service(timeout_sec=2.0):
            future = self.reset_world_cli.call_async(req)
            rclpy.spin_until_future_complete(self, future, timeout_sec=3.0)

            if future.result() is None:
                self.get_logger().error("reset_world failed")
                return False
            return True
        else:
            self.get_logger().error("/reset_world not available")
            return False
        
    def rise_to_height(self, target_z=10.5, speed=3.0):
        rate = 0.02  # 50Hz
        max_time = 20.0
        start = time.monotonic()

        while time.monotonic() - start < max_time:
            z = self.current_pose[2]

            if z >= target_z:
                break

            self.send_velocity(0.0, 0.0, speed, 0, 0, 0)
            rclpy.spin_once(self, timeout_sec=rate)

        # stop drone
        self.send_velocity(0.0, 0.0, 0.0, 0, 0, 0)


# ================================================================
# Gym Environment: 把 RL 標準介面包起來
# ================================================================
class DroneGymEnv(gym.Env):
    """讓無人機模擬變成 Gym 相容的環境。"""

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(3,), dtype=np.float32
        )

        # ---------- 定義 Observation Space ----------
        # [drone_x, drone_y, drone_z,
        #  target_x, target_y, target_z,
        #  vx, vy, vz]
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(15,), dtype=np.float32
        )

        base_dir = Path(__file__).resolve().parent
        with open(base_dir / 'trajectory1_noFace.txt', 'r') as f:
            self.trajectory1 = np.array([[float(num) for num in line.split()] for line in f])
        
        with open(base_dir / 'trajectory2_noFace.txt', 'r') as f:
            self.trajectory2 = np.array([[float(num) for num in line.split()] for line in f])

        self.randTraj = None 
        self.resetTimes = 0
        self.episode_count = 0
        
        self.max_steps = 500
        self.step_count = 0
        self.last_action = np.zeros(3)  # 上一次的 action (用於 observation)

        self.current_target_idx = 0  # 當前目標點在軌跡中的 index
        self.x_bound = None
        self.y_bound = None
        self.z_bound = None
        self.prev_pose = None

        self.target_point = None
        self.next_target_point = None
        self.next_next_target_point = None
        self.info = None

        self.total_steps = 0


    def _collect_garbage(self):
        gc.collect()
        if self.episode_count % 100 == 0:
            self.ros.get_logger().info(f"Episode {self.episode_count}: garbage collection completed")

    # ------------------------------------------------------------
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.episode_count += 1

        # randomly select a trajectory for this episode
        # if self.resetTimes < 3:
        #     self.randTraj = self.trajectory1
        #     self.resetTimes += 1
        # else:
        #     randnum = np.random.choice([0, 1])
        #     if randnum == 0:
        #         self.randTraj = self.trajectory1
        #     else:
        #         self.randTraj = self.trajectory2
        self.randTraj = self.trajectory1

        start_pos = self.randTraj[0][:3]
        self.target_point = self.randTraj[3][:3]
        self.next_target_point = np.zeros(3)  # step1 is only one point
        self.next_next_target_point = np.zeros(3)  # step1 is only one point

        # self.x_bound = (np.min(traj_positions[:, 0]) - 5, np.max(traj_positions[:, 0]) + 5)
        # self.y_bound = (np.min(traj_positions[:, 1]) - 5, np.max(traj_positions[:, 1]) + 5)
        # self.z_bound = (np.min(traj_positions[:, 2]) - 5, np.max(traj_positions[:, 2]) + 5)
        self.x_bound = self.target_point[0] - 5, self.target_point[0] + 5
        self.y_bound = self.target_point[1] - 5, self.target_point[1] + 5
        self.z_bound = self.target_point[2] - 5, self.target_point[2] + 5

        self.ros.reset_world()
        time.sleep(0.3)  # 等 Gazebo 穩定一下
        self.ros.reset_drone()
        self.ros.wait_for_pose(timeout_sec=2.0)
        self.ros.rise_to_height(target_z=self.target_point[2] - 1.0)


        self.step_count = 0
        self.current_target_idx = 0
        self.last_action = np.zeros(3)

        self.info = {"Success": False}  # episode 結束時是否成功完成任務的標記
        obs = self._get_obs()  # 初始 observation (動作全為 0)
        self._collect_garbage()
        return obs, {"trajectory": self.randTraj}

    # ------------------------------------------------------------
    def step(self, action):
        # set some value
        reward = 0.0
        self.prev_pose = self.ros.current_pose.copy()

        # 1. 執行動作
        
        vx, vy, vz = action[:3] # 線速度
        start = time.monotonic()
        self.ros.send_velocity(vx, vy, vz, 0, 0, 0)  # 這裡不控制角速度，讓 drone 自己穩定朝向

        while time.monotonic() - start < DT:
            rclpy.spin_once(self.ros, timeout_sec=0.01)
        # 這裡用 spin_once 讓 ROS callback 更新狀態，並

        self.step_count += 1
        self.total_steps += 1

        if np.linalg.norm(self.ros.current_pose - self.randTraj[self.current_target_idx][:3]) < UPDATE_DISTANCE:
            reward +=(20 * self.current_target_idx + 1)
            self.current_target_idx = self.current_target_idx + 1

        # 2. 取得新狀態
        obs = self._get_obs()

        # 3. 計算 reward
        reward += self._compute_reward(action)

        # 4. 判斷結束條件
        terminated = False
        # if self.ros.current_pose.
        if self.ros.current_pose[0] < self.x_bound[0] or self.ros.current_pose[0] > self.x_bound[1] or \
           self.ros.current_pose[1] < self.y_bound[0] or self.ros.current_pose[1] > self.y_bound[1] or \
           self.ros.current_pose[2] < self.z_bound[0] or self.ros.current_pose[2] > self.z_bound[1] :
            terminated = True
            reward -= 10.0  # 飛出邊界的懲罰
            print("Drone out of bounds!")
        # elif self.current_target_idx >= len(self.randTraj):
        #     terminated = True
        #     reward += 10.0  # 完成軌跡的獎勵
        elif np.linalg.norm(self.ros.current_pose - self.target_point) < UPDATE_DISTANCE:
            terminated = True
            reward += 10.0  # 接近最後目標點的獎勵
            self.info["Success"] = True
            print("Target reached!")
            
        truncated = self.step_count >= self.max_steps
        if truncated:
            print("Max steps reached, truncating episode.")
        self.last_action = action.copy()
        if self.step_count % 50 == 0 or terminated or truncated:
            self.printstate(reward)
        return obs, reward, terminated, truncated, self.info

    # ------------------------------------------------------------
    def _get_obs(self):
        """把當前狀態包成 observation vector。"""

        target_vector = self.target_point - self.ros.current_pose
        # next_target_vector = self.next_target_point - self.ros.current_pose
        # next_next_target_vector = self.next_next_target_point - self.ros.current_pose
        next_target_vector = np.zeros(3)  # step1 is only one point
        next_next_target_vector = np.zeros(3)  # step1 is only one point

        observation = np.concatenate([
            self.ros.current_pose,
            self.ros.current_vel,
            target_vector, 
            next_target_vector,
            next_next_target_vector,
        ]).astype(np.float32)
        return observation
    
    def _compute_reward(self, action):
        """根據當前狀態和動作計算 reward。"""
        
        position_error = np.linalg.norm(self.ros.current_pose - self.target_point)  # 無人機與目標的距離

        # if action make the drone move to the target position, then give a reward
        current_pose = self.ros.current_pose
        prev_dist = np.linalg.norm(self.prev_pose - self.target_point)
        curr_dist = np.linalg.norm(current_pose - self.target_point)
        progress_reward = prev_dist - curr_dist

        smoothness_penalty = np.linalg.norm(self.last_action - action)  # 動作的平滑度懲罰 (速度越大懲罰越多)

        reward = 100.0 * progress_reward - 0.5 * position_error - 0.1 * smoothness_penalty + 0.5 * self.current_target_idx
        
        return reward

    def printstate(self, reward):
        print(f"Step {self.step_count}, total steps {self.total_steps}")
        print(f"Current pose: {self.ros.current_pose}, Target point: {self.target_point}")
        print(f"Reward: {reward:.2f}")