from step4_fullTrajectoryEnv import DroneGymEnv, DroneROSInterface
import csv
import numpy as np
import time
from stable_baselines3 import PPO, TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
import argparse
import rclpy
from pathlib import Path
import matplotlib.pyplot as plt

class EpisodeRewardLogger(BaseCallback):
    def __init__(self, csv_path: str, plot_path: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.plot_path = Path(plot_path)
        self.episode_records = []
        self.current_reward = 0.0
        self.current_length = 0
        self.episode_index = 0

    def _on_training_start(self) -> None:
        self.current_reward = 0.0
        self.current_length = 0
        self.episode_index = 0
        self.episode_records = []

    def _on_step(self) -> bool:
        rewards = np.asarray(self.locals.get('rewards'))
        dones = np.asarray(self.locals.get('dones'))

        if rewards.size == 0 or dones.size == 0:
            return True

        step_reward = float(rewards.reshape(-1)[0])
        step_done = bool(dones.reshape(-1)[0])

        self.current_reward += step_reward
        self.current_length += 1

        if step_done:
            self.episode_index += 1
            self.episode_records.append({
                'episode': self.episode_index,
                'reward': self.current_reward,
                'length': self.current_length,
            })
            self.current_reward = 0.0
            self.current_length = 0

        return True

    def save(self) -> None:
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        with self.csv_path.open('w', newline='') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=['episode', 'reward', 'length', 'reward_per_step'])
            writer.writeheader()
            for record in self.episode_records:
                reward_per_step = record['reward'] / record['length'] if record['length'] > 0 else 0.0
                writer.writerow({
                    'episode': record['episode'],
                    'reward': record['reward'],
                    'length': record['length'],
                    'reward_per_step': reward_per_step,
                })

        if not self.episode_records:
            return

        episodes = [record['episode'] for record in self.episode_records]
        rewards = [record['reward'] for record in self.episode_records]
        lengths = [record['length'] for record in self.episode_records]
        reward_per_step = [reward / length if length > 0 else 0.0 for reward, length in zip(rewards, lengths)]

        plt.figure(figsize=(10, 5))
        plt.plot(episodes, rewards, label='Episode Reward', linewidth=1.5)
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        plt.title('Training Curve: Reward vs Episode')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        self.plot_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(self.plot_path, dpi=200)
        plt.close()

        reward_per_step_plot_path = self.plot_path.with_name(f'{self.plot_path.stem}_reward_per_step{self.plot_path.suffix}')
        plt.figure(figsize=(10, 5))
        plt.plot(episodes, reward_per_step, label='Reward / Episode Length', linewidth=1.5, color='tab:orange')
        plt.xlabel('Episode')
        plt.ylabel('Reward per Step')
        plt.title('Training Curve: Reward per Step vs Episode')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(reward_per_step_plot_path, dpi=200)
        plt.close()

def train(env, resume_from=None, total_timesteps=50_000):
    """用 PPO 演算法訓練。"""
    print('🎓 開始訓練...')
    if resume_from is not None:
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists() and checkpoint_path.suffix != '.zip':
            checkpoint_path = checkpoint_path.with_suffix('.zip')
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'找不到要載入的模型檔: {resume_from}')
        model = PPO.load(str(checkpoint_path), env=env)
        reset_num_timesteps = False
        print(f'📦 已載入預訓練模型: {checkpoint_path}')
    else:
        model = PPO(
            'MlpPolicy', env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=128,
            tensorboard_log='./ppo_drone_logs/',
        )
        reset_num_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path='./ppo_drone_checkpoints_step4/',
        name_prefix='ppo_drone',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    curve_callback = EpisodeRewardLogger(
        csv_path='ppo_training_curve.csv',
        plot_path='ppo_training_curve.png',
    )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
        callback=CallbackList([checkpoint_callback, curve_callback]),
    )
    
    curve_callback.save()
    model.save('ppo_drone')
    print('✅ 訓練完成,模型已存至 ppo_drone.zip')
    print('📈 訓練曲線已存至 ppo_training_curve.csv')
    print('🖼️ 訓練曲線圖已存至 ppo_training_curve.png')
    print('🖼️ 每步平均 reward 圖已存至 ppo_training_curve_reward_per_step.png')


def test(env, resume_from=None, test_times=10):
    """載入訓練好的模型並測試。"""
    print('🚀 載入模型測試...')
    
    info = None
    
    if resume_from is not None:
        model = PPO.load(resume_from)
    else:
        model = PPO.load('ppo_drone')
    # print(f'🔍 測試第 {i+1} 次...')
    obs, _ = env.reset()
    total_reward = 0
    for step in range(20000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f'Step {step}: pos={obs[:3]}, reward={reward:.2f}')
        if terminated or truncated:
            print(f'Episode ended at step {step} ')
            break
        
    print(f'✅ 測試結束,總 reward = {total_reward:.2f}')
    if info and info.get("Success"):
        print("Success")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--resume-from', default=None, help='載入既有 PPO 模型並接續訓練，例如 ppo_drone 或 ppo_drone.zip')
    parser.add_argument('--timesteps', type=int, default=50_000, help='本次要額外訓練的 timesteps')
    parser.add_argument('--test-times', type=int, default=10, help='測試的次數')
    parser.add_argument('--traj-num', type=int, default=None, help='訓練或測試時要使用的軌跡編號，1 或 2，預設為 None 表示隨機選擇')
    args = parser.parse_args()

    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface, traj_num=args.traj_num)

    try:
        if args.mode == 'train':
            train(env, resume_from=args.resume_from, total_timesteps=args.timesteps)
        else:
            test(env, resume_from=args.resume_from, test_times=args.test_times)
    finally:
        ros_interface.send_velocity(0, 0, 0, 0, 0, 0)
        ros_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
