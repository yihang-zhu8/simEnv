"""
神经网络模型
从 SEA-Nav-Code 的 DifferentiableSafeActorCritic 移植
"""
import torch
import torch.nn as nn
from torch.distributions import Normal

from simenv_cbf_train.cbf_layer import CBFLayer
from simenv_cbf_train.config import cfg


def get_activation(name: str):
    activations = {
        "elu": nn.ELU(), "relu": nn.ReLU(), "tanh": nn.Tanh(),
        "selu": nn.SELU(), "sigmoid": nn.Sigmoid(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}")
    return activations[name]


def build_mlp(input_dim: int, hidden_dims: list, activation: nn.Module,
              output_dim: int = None) -> nn.Sequential:
    layers = []
    prev_dim = input_dim
    for h in hidden_dims:
        layers.append(nn.Linear(prev_dim, h))
        layers.append(activation)
        prev_dim = h
    if output_dim is not None:
        layers.append(nn.Linear(prev_dim, output_dim))
    return nn.Sequential(*layers)


class CBFNavigationActorCritic(nn.Module):
    """CBF 导航策略网络

    观测（单步）: props(12) + rays(41) + goal(2) = 55
    历史窗口: 10 → 550 维
    动作: 3 维 (vx, vy, yaw_rate)
    """

    is_recurrent = False

    def __init__(self):
        super().__init__()
        ncfg = cfg.network
        ecfg = cfg.env
        ccfg = cfg.cbf

        self.num_obs_one_step = ecfg.num_obs_one_step
        self.num_props = ecfg.num_props
        self.num_rays = ecfg.num_rays
        self.num_actions = ecfg.num_actions
        self.num_latent = ncfg.num_latent

        # Encoder: 历史观测 → 隐向量
        mlp_input_dim_e = ecfg.num_observations
        self.encoder = build_mlp(mlp_input_dim_e, ncfg.encoder_hidden_dims,
                                 get_activation(ncfg.activation), self.num_latent)

        # Backbone: 单步观测 + 隐向量 → 共享特征
        mlp_input_dim_a = self.num_obs_one_step + self.num_latent
        backbone_layers = []
        backbone_layers.append(nn.Linear(mlp_input_dim_a, ncfg.actor_hidden_dims[0]))
        backbone_layers.append(get_activation(ncfg.activation))
        for l in range(len(ncfg.actor_hidden_dims) - 1):
            backbone_layers.append(nn.Linear(ncfg.actor_hidden_dims[l],
                                             ncfg.actor_hidden_dims[l + 1]))
            backbone_layers.append(get_activation(ncfg.activation))
        self.backbone = nn.Sequential(*backbone_layers)

        backbone_out_dim = ncfg.actor_hidden_dims[-1]

        # NavHead: 候选动作
        self.nav_head = build_mlp(backbone_out_dim, ncfg.nav_head_dims,
                                  get_activation(ncfg.activation), self.num_actions)

        # AlphaHead: 自适应安全参数
        self.alpha_head = build_mlp(backbone_out_dim, ncfg.alpha_head_dims,
                                    get_activation(ncfg.activation), 1)

        # CBF 安全层
        self.cbf_layer = CBFLayer(
            num_rays=ecfg.num_rays, fov_deg=ccfg.fov_deg,
            d_safe=ccfg.d_safe, kappa=ccfg.kappa,
            damping_factor=ccfg.damping_factor,
        )

        # Critic
        self.critic = build_mlp(mlp_input_dim_a, ncfg.critic_hidden_dims,
                                get_activation(ncfg.activation), 1)

        # 动作噪声
        self.std = nn.Parameter(ncfg.init_noise_std * torch.ones(self.num_actions))
        self.distribution = None

        # 中间变量（干预损失用）
        self.u_bar = None
        self.u_s = None
        self.alpha = None

    def extract(self, observations: torch.Tensor):
        obs_hist = observations
        obs_buf = observations[:, -self.num_obs_one_step:]
        props = obs_buf[:, :self.num_props]
        rays = obs_buf[:, self.num_props:self.num_props + self.num_rays]
        goals = obs_buf[:, -2:]
        return obs_buf, obs_hist, props, rays, goals

    def forward_safe(self, observations: torch.Tensor) -> torch.Tensor:
        obs_buf, obs_hist, props, rays, goals = self.extract(observations)

        latent = self.encoder(obs_hist)
        obs_cat = torch.cat([obs_buf, latent.detach()], dim=-1)
        shared_features = self.backbone(obs_cat)

        u_bar = self.nav_head(shared_features)
        alpha_raw = self.alpha_head(shared_features)
        alpha = torch.nn.functional.softplus(alpha_raw)

        # 还原射线距离（如果经过 log2 压缩）
        if rays.max() < 5.0 and rays.min() >= 0:
            rays_real = torch.exp2(rays.clamp(min=0.1, max=5.0))
        else:
            rays_real = rays

        u_s = self.cbf_layer(u_bar, rays_real, alpha)

        self.u_bar = u_bar
        self.u_s = u_s
        self.alpha = alpha

        return u_s

    def update_distribution(self, observations: torch.Tensor):
        mean = self.forward_safe(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)

    def act(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        self.update_distribution(observations)
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor) -> torch.Tensor:
        return self.forward_safe(observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations: torch.Tensor, **kwargs) -> torch.Tensor:
        obs_buf, obs_hist, props, rays, goals = self.extract(observations)
        latent = self.encoder(obs_hist)
        obs_cat = torch.cat([obs_buf, latent], dim=-1)
        return self.critic(obs_cat)

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass


class StandardActorCritic(nn.Module):
    """标准 ActorCritic（无 CBF，消融实验用）"""

    is_recurrent = False

    def __init__(self):
        super().__init__()
        ncfg = cfg.network
        ecfg = cfg.env

        self.num_obs_one_step = ecfg.num_obs_one_step
        self.num_props = ecfg.num_props
        self.num_rays = ecfg.num_rays
        self.num_actions = ecfg.num_actions
        self.num_latent = ncfg.num_latent

        mlp_input_dim_a = self.num_obs_one_step + self.num_latent
        mlp_input_dim_e = ecfg.num_observations

        self.encoder = build_mlp(mlp_input_dim_e, ncfg.encoder_hidden_dims,
                                 get_activation(ncfg.activation), self.num_latent)
        self.actor = build_mlp(mlp_input_dim_a, ncfg.actor_hidden_dims,
                               get_activation(ncfg.activation), self.num_actions)
        self.critic = build_mlp(mlp_input_dim_a, ncfg.critic_hidden_dims,
                                get_activation(ncfg.activation), 1)

        self.std = nn.Parameter(ncfg.init_noise_std * torch.ones(self.num_actions))
        self.distribution = None

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        obs_buf = observations[:, -self.num_obs_one_step:]
        latent = self.encoder(observations)
        obs_cat = torch.cat([obs_buf, latent], dim=-1)
        return self.actor(obs_cat)

    def act(self, observations, **kwargs):
        mean = self.forward(observations)
        self.distribution = Normal(mean, mean * 0. + self.std)
        return self.distribution.sample()

    def act_inference(self, observations):
        return self.forward(observations)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def evaluate(self, observations, **kwargs):
        obs_buf = observations[:, -self.num_obs_one_step:]
        latent = self.encoder(observations)
        obs_cat = torch.cat([obs_buf, latent], dim=-1)
        return self.critic(obs_cat)

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass


def make_actor_critic(with_cbf: bool = True) -> nn.Module:
    if with_cbf:
        model = CBFNavigationActorCritic()
        print(f"  [模型] CBFNavigationActorCritic（带 CBF 安全层）")
    else:
        model = StandardActorCritic()
        print(f"  [模型] StandardActorCritic（无 CBF）")
    print(f"  [参数] 总计: {sum(p.numel() for p in model.parameters()):,}")
    return model
