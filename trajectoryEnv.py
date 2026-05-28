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

DT = 0.1  # 每一步的時間間隔 (秒)
UPDATE_DISTANCE = 1.0  # 無人機距離當前目標點多近時切換到下一個目標點

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
        self._robot_xml = None
        self._entity_name = None
        

    def _pose_cb(self, msg: Pose):
        new_pose = np.array([msg.position.x, msg.position.y, msg.position.z])
        self.current_orientation = np.array([
            msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w
        ])
        self.last_pose_time = time.monotonic()
        # 簡易速度估計 (差分法)
        self.current_vel = (new_pose - self.current_pose) * 10  # 約 10Hz
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

    def takeoff_with_stabilization(self, duration_sec=1.5, publish_period=0.2):
        end_time = time.monotonic() + duration_sec
        next_pub = time.monotonic()
        while time.monotonic() < end_time:
            now = time.monotonic()
            if now >= next_pub:
                self.takeoff_pub.publish(Empty())
                next_pub = now + publish_period
            rclpy.spin_once(self, timeout_sec=0.05)

    def _load_spawn_description(self):
        if self._robot_xml is not None and self._entity_name is not None:
            return

        xacro_file = os.path.join(
            get_package_share_directory("nsysu_drone_description"),
            "urdf", "nsysu_drone.urdf.xacro"
        )
        yaml_file_path = os.path.join(
            get_package_share_directory("nsysu_drone_bringup"),
            "config", "drone.yaml"
        )

        robot_description_config = xacro.process_file(
            xacro_file, mappings={"params_path": yaml_file_path}
        )
        self._robot_xml = robot_description_config.toxml()

        with open(yaml_file_path, 'r') as f:
            yaml_dict = yaml.load(f, Loader=yaml.FullLoader)
        self._entity_name = yaml_dict.get("namespace", "drone")

    def teleport_to_pose(self, position, face_position):
        self._load_spawn_description()

        delete_req = DeleteEntity.Request()
        delete_req.name = self._entity_name

        if self.delete_cli.wait_for_service(timeout_sec=1.0):
            delete_future = self.delete_cli.call_async(delete_req)
            rclpy.spin_until_future_complete(self, delete_future, timeout_sec=2.0)

        forward = face_position - position
        if np.linalg.norm(forward) < 1e-6:
            yaw = 0.0
            pitch = 0.0
        else:
            forward /= np.linalg.norm(forward)
            yaw = np.arctan2(forward[1], forward[0])
            pitch = np.arctan2(-forward[2], np.sqrt(forward[0] ** 2 + forward[1] ** 2))

        quat = R.from_euler('zyx', [yaw, pitch, 0.0]).as_quat()
        pose = Pose()
        pose.position.x = float(position[0])
        pose.position.y = float(position[1])
        pose.position.z = float(position[2])
        pose.orientation.x = float(quat[0])
        pose.orientation.y = float(quat[1])
        pose.orientation.z = float(quat[2])
        pose.orientation.w = float(quat[3])

        spawn_req = SpawnEntity.Request()
        spawn_req.name = self._entity_name
        spawn_req.xml = self._robot_xml
        spawn_req.robot_namespace = self._entity_name
        spawn_req.reference_frame = "world"
        spawn_req.initial_pose = pose

        if not self.spawn_cli.wait_for_service(timeout_sec=2.0):
            self.get_logger().error("/spawn_entity service not available")
            return False

        spawn_future = self.spawn_cli.call_async(spawn_req)
        rclpy.spin_until_future_complete(self, spawn_future, timeout_sec=5.0)
        if not spawn_future.done():
            self.get_logger().error("/spawn_entity timeout")
            return False
        if spawn_future.result() is None:
            self.get_logger().error("/spawn_entity call failed")
            return False

        return bool(spawn_future.result().success)


# ================================================================
# Gym Environment: 把 RL 標準介面包起來
# ================================================================
class DroneGymEnv(gym.Env):
    """讓無人機模擬變成 Gym 相容的環境。"""

    def __init__(self, ros_interface: DroneROSInterface):
        super().__init__()
        self.ros = ros_interface

        
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(6,), dtype=np.float32
        )

        # ---------- 定義 Observation Space ----------
        # [drone_x, drone_y, drone_z,
        #  target_x, target_y, target_z,
        #  vx, vy, vz]
        self.observation_space = spaces.Box(
            low=-50.0, high=50.0, shape=(16,), dtype=np.float32
        )

        base_dir = Path(__file__).resolve().parent
        with open(base_dir / 'trajectory1.txt', 'r') as f:
            self.trajectory1 = np.array([[float(num) for num in line.split()] for line in f])
        
        with open(base_dir / 'trajectory2.txt', 'r') as f:
            self.trajectory2 = np.array([[float(num) for num in line.split()] for line in f])

        self.randTraj = None 
        self.resetTimes = 0
        self.episode_count = 0
        
        self.max_steps = 1000
        self.step_count = 0
        self.last_action = np.zeros(6)  # 上一次的 action (用於 observation)
        self.last_obs = np.zeros(6)  # 上一個 part of observation, 
                                      # only include 1. current pose
                                                   # 2. target position
                                                   # 3. the position drone should face to 
        self.curr_obs = np.zeros(6)  # 當前 part of observation
        self.current_target_idx = 0  # 當前目標點在軌跡中的 index
        self.x_bound = None
        self.y_bound = None
        self.z_bound = None
        self.prev_pose = None

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

        # find x, y, z bound of the trajectory
        traj_positions = self.randTraj[:, :3]
        self.x_bound = (np.min(traj_positions[:, 0]) - 5, np.max(traj_positions[:, 0]) + 5)
        self.y_bound = (np.min(traj_positions[:, 1]) - 5, np.max(traj_positions[:, 1]) + 5)
        self.z_bound = (np.min(traj_positions[:, 2]) - 5, np.max(traj_positions[:, 2]) + 5)

        start_pos = self.randTraj[0][:3]
        start_face = self.randTraj[0][3:]
        if not self.ros.teleport_to_pose(start_pos, start_face):
            self.ros.reset_drone()
        else:
            self.ros.wait_for_pose(timeout_sec=1.0)
            self.ros.takeoff_with_stabilization(duration_sec=1.5, publish_period=0.2)
        self.step_count = 0
        self.current_target_idx = 0
        self.last_action = np.zeros(6)
        self.last_obs = np.zeros(6)  # 上一個 part of observation, 
                                      # only include 1. current pose
                                                   # 2. target position
                                                   # 3. the position drone should face to 
        self.curr_obs = np.zeros(6)  # 當前 part of observation
        self.prev_pose = self.randTraj[0][:3]  # 用於計算初始速度

        obs = self._get_obs(np.zeros(6))  # 初始 observation (動作全為 0)
        self._collect_garbage()
        return obs, {"trajectory": self.randTraj}

    # ------------------------------------------------------------
    def step(self, action):
        # set some value
        reward = 0.0
        self.prev_pose = self.ros.current_pose.copy()

        # 1. 執行動作
        print(f"Target position: {self.randTraj[self.current_target_idx][:3]}")
        
        vx, vy, vz = action[:3] # 線速度
        x, y, z = action[3:] # 角速度
        start = time.monotonic()
        self.ros.send_velocity(vx, vy, vz, x, y, z)

        while time.monotonic() - start < DT:
            rclpy.spin_once(self.ros, timeout_sec=0.01)
        # 這裡用 spin_once 讓 ROS callback 更新狀態，並

        self.step_count += 1
        if np.linalg.norm(self.ros.current_pose - self.randTraj[self.current_target_idx][:3]) < UPDATE_DISTANCE:
            reward += self.current_target_idx + 1
            self.current_target_idx = self.current_target_idx + 1

        # 2. 取得新狀態
        obs = self._get_obs(action)

        # 3. 計算 reward
        reward += self._compute_reward(obs, action)

        # 4. 判斷結束條件
        terminated = False
        # if self.ros.current_pose.
        if self.ros.current_pose[0] < self.x_bound[0] or self.ros.current_pose[0] > self.x_bound[1] or \
           self.ros.current_pose[1] < self.y_bound[0] or self.ros.current_pose[1] > self.y_bound[1] or \
           self.ros.current_pose[2] < self.z_bound[0] or self.ros.current_pose[2] > self.z_bound[1]:
            terminated = True
            reward -= 10.0  # 飛出邊界的懲罰
        elif self.current_target_idx >= len(self.randTraj):
            terminated = True
            reward += 100.0  # 完成軌跡的額外獎勵
        truncated = self.step_count >= self.max_steps
        print(f"Reward: {reward:.2f}")
        self.last_action = action.copy()
        return obs, reward, terminated, truncated, {}

    # ------------------------------------------------------------
    def _get_obs(self, action):
        """把當前狀態包成 observation vector。"""

        # traj_positions = self.randTraj[:, :3]
        # distances = np.linalg.norm(traj_positions - self.ros.current_pose, axis=1)
        # nearest_idx = int(np.argmin(distances))
        # next_idx = min(nearest_idx + 1, len(self.randTraj) - 1)
        # =================================


        # self.next_obs = np.concatenate([
        #     self.randTraj[self.current_target_idx][:3],  # 目標位置 (closest point + 1)
        #     self.randTraj[self.current_target_idx][3:],  # the position drone should face to
        # ]).astype(np.float32)
        
        # observation = np.concatenate(
        #     [self.ros.current_pose, 
        #      self.ros.current_orientation, 
        #      self.last_obs, self.curr_obs, self.next_obs, 
        #      self.ros.current_vel, action[3:]]
        # ).astype(np.float32)
        target_vector = self.randTraj[self.current_target_idx][:3] - self.ros.current_pose
        orientation_error = self._compute_orientation_error(
            self.randTraj[self.current_target_idx][:3],
            self.randTraj[self.current_target_idx][3:],
            isValue=False
        )
        observation = np.concatenate([
            self.ros.current_pose,
            self.ros.current_orientation,
            self.ros.current_vel,
            target_vector, 
            orientation_error,
        ]).astype(np.float32)

        # self.last_obs = self.curr_obs
        # self.curr_obs = self.next_obs
        return observation
    
    def _compute_reward(self, obs, action):
        """根據當前狀態和動作計算 reward。"""
        

        target_pos = self.randTraj[self.current_target_idx][:3]  # 目標位置(agent see the last time's target position and give an action)
                                  # we sholud use the last time's target position and current position (this position is affected by the action) to compute the reward
        target_face = self.randTraj[self.current_target_idx][3:]  # the position drone should face to
        position_error = np.linalg.norm(self.ros.current_pose - target_pos)  # 無人機與目標的距離

        # if action make the drone move to the target position, then give a reward
        current_pose = self.ros.current_pose
        prev_dist = np.linalg.norm(self.prev_pose - target_pos)
        curr_dist = np.linalg.norm(current_pose - target_pos)
        progress_reward = prev_dist - curr_dist
        print(f"current_pose: {current_pose}")

        # 計算朝向誤差 (無人機當前朝向與目標朝向的誤差)
        orientation_error = self._compute_orientation_error(target_pos, target_face)

        smoothness_penalty = np.linalg.norm(self.last_action - action)  # 動作的平滑度懲罰 (速度越大懲罰越多)
        
        position_reward = 0.0
        if position_error < 0.1:
            position_reward = 6.0

        # orientation_reward = 0.0
        # if orientation_error < np.deg2rad(30):  # 朝向誤差小於 30 度
        #     orientation_reward = 3.0

        reward = 20.0 * progress_reward - position_error - 0.5 * orientation_error - 0.1 * smoothness_penalty + position_reward 
        
        return reward

    def _compute_orientation_error(self, target_pos, target_face, isValue=True):
        """計算無人機當前朝向與目標朝向的誤差。"""
        forward_traj = target_face - target_pos
        if np.linalg.norm(forward_traj) < 1e-6:
            return 0.0  # 如果目標朝向和位置幾乎重合，則誤差為 0
        # print("----------------------------------1. Target Face:", target_face)
        forward_traj /= np.linalg.norm(forward_traj)

        # forward_drone = target_face - self.ros.current_pose
        # forward_drone /= np.linalg.norm(forward_drone)

        yaw_traj = np.arctan2(forward_traj[1], forward_traj[0])
        pitch_traj = np.arctan2(-forward_traj[2], np.sqrt(forward_traj[0]**2 + forward_traj[1]**2))
        roll_traj = 0.0
        # print("----------------------------------2. Target Face:", target_face)
        # print("----------------------------------Forward Trajectory:", forward_traj)
        # print("----------------------------------Yaw Trajectory:", yaw_traj)
        # print("----------------------------------Pitch Trajectory:", pitch_traj)

        # yaw_drone = np.arctan2(forward_drone[1], forward_drone[0])
        # pitch_drone = np.arctan2(-forward_drone[2], np.sqrt(forward_drone[0]**2 + forward_drone[1]**2))
        # roll_drone = 0.0

        quat_traj = R.from_euler('zyx', [yaw_traj, pitch_traj, roll_traj]).as_quat()
        quat_drone = R.from_quat(self.ros.current_orientation).as_quat() ############# may be wrong

        # print("----------------------------------Current Orientation (quaternion):", self.ros.current_orientation)
        # print("----------------------------------quat_traj:", quat_traj)
        r_error = R.from_quat(quat_traj).inv() * R.from_quat(quat_drone)
        if isValue:
            return r_error.magnitude()
        else:
            return r_error.as_euler('zyx', degrees=True)