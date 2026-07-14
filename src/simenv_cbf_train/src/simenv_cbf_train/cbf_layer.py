"""
CBF（控制障碍函数）安全层
从 SEA-Nav-Code 的 ExactLSECBFLayer 移植

使用 LSE（Log-Sum-Exp）光滑近似组合 CBF
"""
import torch
import torch.nn as nn
import numpy as np


def get_ray_unit_vectors(num_rays: int = 41, fov_deg: float = 180.0, device=None):
    start_angle = -np.deg2rad(fov_deg) / 2
    end_angle = np.deg2rad(fov_deg) / 2
    angles = torch.linspace(start_angle, end_angle, num_rays, device=device)
    return torch.stack([torch.cos(angles), torch.sin(angles)], dim=1)


class CBFLayer(nn.Module):
    """闭式 CBF 安全层

    输入:
        u_bar: [B, 3]  候选动作 (vx, vy, yaw_rate)
        lidar_dists: [B, num_rays]  激光测距值（米）
        alpha: [B, 1]  自适应安全参数（>0）

    输出:
        u_s: [B, 3]  安全修正后的动作
    """

    def __init__(self,
                 num_rays: int = 41,
                 fov_deg: float = 180.0,
                 d_safe: float = 0.20,
                 kappa: float = 10.0,
                 damping_factor: float = 1.0):
        super().__init__()

        self.d_safe = d_safe
        self.kappa = kappa
        self.damping_factor = damping_factor

        ray_uv = get_ray_unit_vectors(num_rays, fov_deg)
        self.register_buffer('ray_unit_vectors', ray_uv)

    def forward(self, u_bar: torch.Tensor,
                lidar_dists: torch.Tensor,
                alpha: torch.Tensor) -> torch.Tensor:
        B = u_bar.shape[0]
        u_2d = u_bar[:, :2]
        yaw_rate = u_bar[:, 2:]

        # 个体 CBF: h_i(x) = d_i - d_safe
        h_i = lidar_dists - self.d_safe

        # LSE 光滑组合 CBF
        min_h, _ = torch.min(h_i, dim=1, keepdim=True)
        h_comp = min_h - (1.0 / self.kappa) * torch.log(
            torch.sum(torch.exp(-self.kappa * (h_i - min_h)), dim=1, keepdim=True)
        )

        # 权重 λ_i
        lambda_i = torch.exp(-self.kappa * (h_i - h_comp)).unsqueeze(-1)

        # L_g h(x)
        n_vecs = self.ray_unit_vectors.unsqueeze(0)
        Lg_h = -torch.sum(lambda_i * n_vecs, dim=1)

        # η(x) — CBF-QP 闭式解
        Lgh_u = torch.sum(Lg_h * u_2d, dim=1, keepdim=True)
        Lgh_norm_sq = torch.sum(Lg_h ** 2, dim=1, keepdim=True)
        eta = -(Lgh_u + alpha * h_comp) / (Lgh_norm_sq + self.damping_factor)

        # 安全动作
        u_s_2d = u_2d + torch.relu(eta) * Lg_h
        u_s = torch.cat((u_s_2d, yaw_rate), dim=-1)

        return u_s
