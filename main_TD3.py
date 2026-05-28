from step2_oneRandomPointEnv_td3 import DroneGymEnv, DroneROSInterface
import numpy as np
import csv
import time
from stable_baselines3 import TD3
from stable_baselines3.common.callbacks import BaseCallback, CallbackList, CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
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
            writer = csv.DictWriter(csv_file, fieldnames=['episode', 'reward', 'length'])
            writer.writeheader()
            writer.writerows(self.episode_records)

        if not self.episode_records:
            return

        episodes = [record['episode'] for record in self.episode_records]
        rewards = [record['reward'] for record in self.episode_records]

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

def train(env, resume_from=None, total_timesteps=50_000):
    """用 TD3 演算法訓練。"""
    print('🎓 開始訓練...')
    if resume_from is not None:
        checkpoint_path = Path(resume_from)
        if not checkpoint_path.exists() and checkpoint_path.suffix != '.zip':
            checkpoint_path = checkpoint_path.with_suffix('.zip')
        if not checkpoint_path.exists():
            raise FileNotFoundError(f'找不到要載入的模型檔: {resume_from}')
        model = TD3.load(str(checkpoint_path), env=env)
        reset_num_timesteps = False
        print(f'📦 已載入預訓練模型: {checkpoint_path}')
    else:
        model = TD3(
            'MlpPolicy', env,
            verbose=1,
            learning_rate=3e-4,
            batch_size=128,
            buffer_size=30000,
            learning_starts=3000,
            tau = 0.005,
            action_noise = NormalActionNoise(
                mean=np.zeros(3),
                sigma=0.1*np.ones(3)
            ),
            tensorboard_log='./td3_drone_logs/',
        )
        reset_num_timesteps = True

    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path='./td3_drone_checkpoints_step2/',
        name_prefix='td3_drone',
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    curve_callback = EpisodeRewardLogger(
        csv_path='td3_training_curve.csv',
        plot_path='td3_training_curve.png',
    )

    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=reset_num_timesteps,
        callback=CallbackList([checkpoint_callback, curve_callback]),
    )

    curve_callback.save()
    model.save('td3_drone')
    print('✅ 訓練完成,模型已存至 td3_drone.zip')
    print('📈 訓練曲線已存至 td3_training_curve.csv')
    print('🖼️ 訓練曲線圖已存至 td3_training_curve.png')


def test(env, resume_from=None):
    """載入訓練好的模型並測試。"""
    print('🚀 載入模型測試...')
    if resume_from is not None:
        model = TD3.load(resume_from)
    else:
        model = TD3.load('td3_drone')
    obs, _ = env.reset()
    total_reward = 0
    for step in range(20000):
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        if step % 10 == 0:
            print(f'Step {step}: pos={obs[:3]}, reward={reward:.2f}')
        if terminated or truncated:
            print(f'Episode ended at step {step} ')
            break
    print(f'✅ 測試結束,總 reward = {total_reward:.2f}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['train', 'test'], default='train')
    parser.add_argument('--resume-from', default=None, help='載入既有 TD3 模型並接續訓練，例如 td3_drone 或 td3_drone.zip')
    parser.add_argument('--timesteps', type=int, default=50_000, help='本次要額外訓練的 timesteps')
    args = parser.parse_args()

    rclpy.init()
    ros_interface = DroneROSInterface()
    env = DroneGymEnv(ros_interface)

    try:
        if args.mode == 'train':
            train(env, resume_from=args.resume_from, total_timesteps=args.timesteps)
        else:
            test(env, resume_from=args.resume_from)
    finally:
        ros_interface.send_velocity(0, 0, 0, 0, 0, 0)
        ros_interface.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
