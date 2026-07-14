"""
探索策略模块

不依赖真值里程计，用传感器信息驱动探索：
1. 点云前沿检测
2. 房间级探索
3. 门/电梯交互
"""
import numpy as np
from typing import Tuple, Optional
import rospy


class RoomExplorer:
    """房间级探索器"""

    def __init__(self, env):
        self.env = env
        self.visited_frontiers: list = []
        self.current_room: Optional[str] = None

    def find_best_direction(self) -> Tuple[float, float, float]:
        """找到最佳探索方向

        基于 Livox 点云：找最远的空旷方向

        Returns:
            (vx, vy, yaw_rate) 目标速度
        """
        rays = self.env.get_livox_rays(num_rays=41)

        # 前方 180 度的射线
        half = len(rays) // 2
        front_rays = rays[:half]  # 左半边
        back_rays = rays[half:]   # 右半边

        # 找最远的方向
        if np.max(front_rays) > np.max(back_rays):
            best_idx = np.argmax(front_rays)
            best_dist = front_rays[best_idx]
        else:
            best_idx = np.argmax(back_rays) + half
            best_dist = back_rays[best_idx - half]

        # 角度映射（-120° 到 120°）
        angles = np.linspace(-2 * np.pi / 3, 2 * np.pi / 3, len(rays))
        target_angle = angles[best_idx]

        # 如果前方足够开阔，直行；否则转向空旷方向
        min_front = np.min(front_rays[:10])  # 最前方 1/4

        if min_front > 1.0:
            # 前方开阔，直行
            vx = min(1.0, best_dist / 5.0)
            vy = 0.0
            yaw_rate = 0.0
        elif best_dist > 1.0:
            # 转向空旷方向
            vx = 0.3
            vy = 0.0
            yaw_rate = 0.5 * np.sign(target_angle)
        else:
            # 原地旋转找方向
            vx = 0.0
            vy = 0.0
            yaw_rate = 1.0

        return vx, vy, yaw_rate


def build_exploration_obs(env) -> np.ndarray:
    """构建探索专用的观测

    包括：
    - 基础观测（IMU + 射线）
    - 当前 room 信息
    - 是否在走廊
    - 是否在楼梯附近
    """
    base_obs = env._build_observation()
    # 额外的探索特征
    ray_dists = env.get_livox_rays(num_rays=41)
    extra = np.array([
        np.mean(ray_dists),           # 平均距离
        np.min(ray_dists),            # 最近障碍物
        np.max(ray_dists),            # 最远可见距离
        np.std(ray_dists),            # 距离标准差（走廊 vs 房间特征）
    ])
    return np.concatenate([base_obs, extra])
