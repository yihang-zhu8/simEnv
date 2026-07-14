"""
SimEnv 训练配置
从 SEA-Nav-Code 的 Go2PosRoughCfg / LeggedRobotPosCfg 移植

路径约定:
  模型保存在 SimEnv/models/ 下
  日志保存在 SimEnv/logs/ 下
"""
import os
import numpy as np
from dataclasses import dataclass, field
from typing import List


# SimEnv workspace 根目录（auto.sh 所在目录）
def _get_workspace_root():
    """获取 SimEnv workspace 根目录"""
    pkg_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.dirname(pkg_path)


_WS_ROOT = _get_workspace_root()


@dataclass
class EnvConfig:
    num_envs: int = 1             # ROS 模式下固定为 1
    episode_length_s: float = 40.0
    dt: float = 0.05              # 20 Hz 控制周期

    num_props: int = 12
    num_rays: int = 40
    num_goal_obs: int = 2
    his_len: int = 10

    @property
    def num_obs_one_step(self) -> int:
        return self.num_props + self.num_rays + self.num_goal_obs

    @property
    def num_observations(self) -> int:
        return self.num_obs_one_step * self.his_len

    num_actions: int = 3
    log_rewards: bool = True


@dataclass
class SimpleEnvConfig:
    grid_size: int = 100
    measure_resolution: float = 0.1
    max_episode_steps: int = 800

    theta_start: float = -2 * np.pi / 3
    theta_end: float = 2 * np.pi / 3
    theta_step: float = np.pi / 30

    min_obstacles: int = 4
    max_obstacles: int = 10

    max_vx: float = 1.5
    max_vy: float = 0.8
    max_vyaw: float = 1.0

    reward_goal: float = 10.0
    reward_collision: float = -10.0
    reward_progress: float = 1.0
    reward_smoothness: float = -0.01
    reward_safe_vel: float = 1.0

    safety_radius: float = 0.15
    goal_threshold: float = 0.3


@dataclass
class CBFConfig:
    num_rays: int = 40
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
    actor_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    critic_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    encoder_hidden_dims: List[int] = field(default_factory=lambda: [512, 256, 128])
    activation: str = "elu"
    init_noise_std: float = 1.5
    num_latent: int = 16
    nav_head_dims: List[int] = field(default_factory=lambda: [128])
    alpha_head_dims: List[int] = field(default_factory=lambda: [64])


@dataclass
class PPOConfig:
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
    max_iterations: int = 500
    save_interval: int = 50
    experiment_name: str = "simenv_cbf_nav"


class SimEnvTrainConfig:
    env = EnvConfig()
    simple_env = SimpleEnvConfig()
    cbf = CBFConfig()
    network = NetworkConfig()
    ppo = PPOConfig()
    seed: int = 42
    device: str = "cpu"

    @property
    def log_dir(self) -> str:
        return os.path.join(_WS_ROOT, "logs")

    @property
    def model_dir(self) -> str:
        return os.path.join(_WS_ROOT, "src", "simenv_cbf_train", "models")


cfg = SimEnvTrainConfig()
