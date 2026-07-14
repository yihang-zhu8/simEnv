"""
SimEnv 训练配置（比赛合规版）

只使用比赛允许的话题，不使用 /Odometry_gazebo 和 /ground_truth/*。
"""
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List


@dataclass
class EnvConfig:
    """ROS-Gazebo 环境配置"""
    num_envs: int = 1
    episode_length_s: float = 600.0       # 10分钟探索
    dt: float = 0.05                       # 20 Hz 控制周期
    num_obs: int = 48                      # 观测维度
    num_actions: int = 3                   # vx, vy, yaw_rate
    log_rewards: bool = True


@dataclass
class SimpleEnvConfig:
    """2D 快速预训练环境配置"""
    grid_size: int = 100
    measure_resolution: float = 0.1
    max_episode_steps: int = 800
    theta_start: float = -2 * np.pi / 3
    theta_end: float = 2 * np.pi / 3
    theta_step: float = np.pi / 20          # 更密的射线
    num_rays: int = 41
    min_obstacles: int = 4
    max_obstacles: int = 10
    max_vx: float = 1.5
    max_vy: float = 0.8
    max_vyaw: float = 1.0
    reward_goal: float = 10.0
    reward_collision: float = -10.0
    reward_progress: float = 1.0
    safety_radius: float = 0.15
    goal_threshold: float = 0.3


@dataclass
class CBFConfig:
    """CBF 安全层配置"""
    num_rays: int = 41
    fov_deg: float = 180.0
    safe_radius: float = 0.15
    safety_margin: float = 0.05
    kappa: float = 10.0
    damping_factor: float = 1.0

    @property
    def d_safe(self) -> float:
        return self.safe_radius + self.safety_margin


@dataclass
class NetworkConfig:
    """网络结构配置"""
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 1.5


@dataclass
class PPOConfig:
    """PPO 训练配置"""
    learning_rate: float = 1e-3
    clip_param: float = 0.2
    value_loss_coef: float = 1.0
    entropy_coef: float = 0.003
    num_learning_epochs: int = 5
    num_mini_batches: int = 4
    gamma: float = 0.99
    lam: float = 0.95
    desired_kl: float = 0.01
    schedule: str = "adaptive"
    max_grad_norm: float = 1.0
    use_clipped_value_loss: bool = True
    num_steps_per_env: int = 48
    max_iterations: int = 2000
    save_interval: int = 200
    experiment_name: str = "simenv_nav"


@dataclass
class DetectConfig:
    """危险源检测配置"""
    red_threshold_h: tuple = (0, 20)       # 红色 HSV 范围 H
    red_threshold_s: tuple = (100, 255)
    red_threshold_v: tuple = (50, 255)
    min_contour_area: int = 50
    confirm_count: int = 3                  # 连续检测帧数确认
    detection_range: float = 5.0            # 有效检测距离 (m)


@dataclass
class SimEnvTrainConfig:
    """总配置"""
    env = EnvConfig()
    simple_env = SimpleEnvConfig()
    cbf = CBFConfig()
    network = NetworkConfig()
    ppo = PPOConfig()
    detect = DetectConfig()
    seed: int = 42
    device: str = "cpu"

    @property
    def model_dir(self) -> str:
        _ws = os.environ.get('SIMENV_WS', os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        return os.path.join(_ws, "models")

    @property
    def log_dir(self) -> str:
        _ws = os.environ.get('SIMENV_WS', os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
        return os.path.join(_ws, "logs")


cfg = SimEnvTrainConfig()
