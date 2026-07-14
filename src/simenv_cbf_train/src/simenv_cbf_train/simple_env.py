"""
快速 2D 训练环境（纯 NumPy，用于预训练）
从 SEA-Nav-Code 的 grid2ray + legged_robot_pos 端口
"""
import numpy as np
import torch
from typing import List, Tuple, Optional

from simenv_cbf_train.config import cfg


class SimpleNavEnv:
    """简单 2D 导航环境 — 纯 NumPy，用于快速策略预训练"""

    def __init__(self, num_envs: int = None, seed: int = None):
        self.num_envs = num_envs or cfg.env.num_envs
        self.scfg = cfg.simple_env
        self.dt = cfg.env.dt
        self.max_steps = self.scfg.max_episode_steps

        if seed is not None:
            np.random.seed(seed)

        self.device = cfg.device

        self.ray_angles = np.arange(
            self.scfg.theta_start, self.scfg.theta_end, self.scfg.theta_step
        )
        self.num_rays = len(self.ray_angles)
        self.max_ray_dist = 5.0

        self._init_buffers()
        self.reset()

    def _init_buffers(self):
        num = self.num_envs
        self.robot_x = np.zeros(num)
        self.robot_y = np.zeros(num)
        self.robot_yaw = np.zeros(num)
        self.goal_x = np.zeros(num)
        self.goal_y = np.zeros(num)
        self.vx = np.zeros(num)
        self.vy = np.zeros(num)
        self.vyaw = np.zeros(num)
        self.collided = np.zeros(num, dtype=bool)
        self.goal_reached = np.zeros(num, dtype=bool)
        self.step_count = np.zeros(num, dtype=int)
        self.maps = []
        self.ray_dists = np.ones((num, self.num_rays)) * self.max_ray_dist

        his_len = cfg.env.his_len
        obs_dim = cfg.env.num_obs_one_step
        self.obs_history = np.zeros((num, his_len, obs_dim))
        self.last_dist = np.zeros(num)

    def _generate_map(self) -> np.ndarray:
        gs = self.scfg.grid_size
        room = np.zeros((gs, gs), dtype=float)
        room[0, :] = 1.0
        room[-1, :] = 1.0
        room[:, 0] = 1.0
        room[:, -1] = 1.0

        num_obs = np.random.randint(self.scfg.min_obstacles, self.scfg.max_obstacles + 1)
        for _ in range(num_obs):
            ow = np.random.randint(3, 15)
            oh = np.random.randint(3, 15)
            ox = np.random.randint(5, gs - ow - 5)
            oy = np.random.randint(5, gs - oh - 5)
            room[oy:oy + oh, ox:ox + ow] = np.random.uniform(0.3, 1.0)

        return room

    def _find_free_position(self, room: np.ndarray) -> Tuple[int, int]:
        gs = room.shape[0]
        while True:
            x = np.random.randint(5, gs - 5)
            y = np.random.randint(5, gs - 5)
            if room[y, x] < 0.01:
                surround = room[max(0, y - 3):min(gs, y + 4),
                                max(0, x - 3):min(gs, x + 4)]
                if np.all(surround < 0.01):
                    return x, y

    def _cast_rays(self, room: np.ndarray, x: float, y: float, yaw: float) -> np.ndarray:
        gs = room.shape[0]
        res = self.scfg.measure_resolution
        max_cells = int(self.max_ray_dist / res)
        dists = np.ones(self.num_rays) * self.max_ray_dist

        for i, angle in enumerate(self.ray_angles):
            theta = yaw + angle
            dx = np.cos(theta) * res
            dy = np.sin(theta) * res
            cx, cy = x / res, y / res
            for step in range(1, max_cells + 1):
                cx += dx / res
                cy += dy / res
                ix, iy = int(round(cx)), int(round(cy))
                if ix < 0 or ix >= gs or iy < 0 or iy >= gs:
                    break
                if room[iy, ix] > 0.01:
                    dists[i] = step * res
                    break

        return dists

    def _get_goal_local(self, robot_x, robot_y, robot_yaw, goal_x, goal_y) -> np.ndarray:
        dx = goal_x - robot_x
        dy = goal_y - robot_y
        local_x = dx * np.cos(-robot_yaw) - dy * np.sin(-robot_yaw)
        local_y = dx * np.sin(-robot_yaw) + dy * np.cos(-robot_yaw)
        return np.column_stack([local_x, local_y])

    def reset(self, env_ids: Optional[np.ndarray] = None) -> torch.Tensor:
        if env_ids is None:
            env_ids = np.arange(self.num_envs)

        for idx in env_ids:
            room = self._generate_map()
            if idx >= len(self.maps):
                self.maps.append(room)
            else:
                self.maps[idx] = room

            rx, ry = self._find_free_position(room)
            rx_m = rx * self.scfg.measure_resolution
            ry_m = ry * self.scfg.measure_resolution

            self.robot_x[idx] = rx_m
            self.robot_y[idx] = ry_m
            self.robot_yaw[idx] = np.random.uniform(-np.pi, np.pi)

            gx, gy = self._find_free_position(room)
            while np.hypot(gx - rx, gy - ry) < 30:
                gx, gy = self._find_free_position(room)
            self.goal_x[idx] = gx * self.scfg.measure_resolution
            self.goal_y[idx] = gy * self.scfg.measure_resolution

            self.vx[idx] = 0.0
            self.vy[idx] = 0.0
            self.vyaw[idx] = 0.0
            self.collided[idx] = False
            self.goal_reached[idx] = False
            self.step_count[idx] = 0

            self.last_dist[idx] = np.hypot(
                self.goal_x[idx] - self.robot_x[idx],
                self.goal_y[idx] - self.robot_y[idx],
            )

            self.ray_dists[idx] = self._cast_rays(
                room, self.robot_x[idx], self.robot_y[idx], self.robot_yaw[idx]
            )

        obs = self._get_observations()
        return obs

    def step(self, actions: np.ndarray):
        vx = np.clip(actions[:, 0], -self.scfg.max_vx, self.scfg.max_vx)
        vy = np.clip(actions[:, 1], -self.scfg.max_vy, self.scfg.max_vy)
        vyaw = np.clip(actions[:, 2], -self.scfg.max_vyaw, self.scfg.max_vyaw)

        self.vx = vx
        self.vy = vy
        self.vyaw = vyaw

        self.robot_yaw += vyaw * self.dt
        self.robot_x += (vx * np.cos(self.robot_yaw) - vy * np.sin(self.robot_yaw)) * self.dt
        self.robot_y += (vx * np.sin(self.robot_yaw) + vy * np.cos(self.robot_yaw)) * self.dt

        self.step_count += 1

        for i in range(self.num_envs):
            if self.goal_reached[i] or self.collided[i]:
                continue
            room = self.maps[i] if i < len(self.maps) else self._generate_map()
            gs = room.shape[0]
            res = self.scfg.measure_resolution

            ix = int(round(self.robot_x[i] / res))
            iy = int(round(self.robot_y[i] / res))
            if 0 <= ix < gs and 0 <= iy < gs:
                if room[iy, ix] > 0.01:
                    self.collided[i] = True
            else:
                self.collided[i] = True

            self.ray_dists[i] = self._cast_rays(
                room, self.robot_x[i], self.robot_y[i], self.robot_yaw[i]
            )

            dist = np.hypot(self.goal_x[i] - self.robot_x[i],
                            self.goal_y[i] - self.robot_y[i])
            if dist < self.scfg.goal_threshold:
                self.goal_reached[i] = True

        rewards = self._compute_rewards()

        dones = np.zeros(self.num_envs, dtype=bool)
        time_outs = self.step_count >= self.max_steps
        dones = self.collided | self.goal_reached | time_outs

        infos = {
            'time_outs': time_outs,
            'goal_reached': self.goal_reached,
            'collided': self.collided,
        }

        obs = self._get_observations()

        reset_ids = np.where(dones)[0]
        if len(reset_ids) > 0:
            reset_obs = self.reset(reset_ids)
            obs[reset_ids] = reset_obs[reset_ids]

        return obs, rewards, dones, infos

    def _compute_rewards(self) -> np.ndarray:
        rewards = np.zeros(self.num_envs)
        for i in range(self.num_envs):
            if self.collided[i]:
                rewards[i] = self.scfg.reward_collision
                continue
            if self.goal_reached[i]:
                rewards[i] = self.scfg.reward_goal
                continue
            dist = np.hypot(self.goal_x[i] - self.robot_x[i],
                            self.goal_y[i] - self.robot_y[i])
            progress = self.last_dist[i] - dist
            self.last_dist[i] = dist
            rewards[i] = self.scfg.reward_progress * np.clip(progress, -0.5, 0.5)
            rewards[i] += self.scfg.reward_smoothness * (
                abs(self.vx[i]) + abs(self.vy[i]) + abs(self.vyaw[i]))
            min_ray = np.min(self.ray_dists[i])
            if min_ray < 0.5:
                safe_vel = min_ray / 0.5
                actual_vel = np.hypot(self.vx[i], self.vy[i])
                if actual_vel > safe_vel * self.scfg.max_vx:
                    rewards[i] -= 0.5
        return rewards

    def _get_observations(self) -> torch.Tensor:
        num = self.num_envs
        onestep = cfg.env.num_obs_one_step

        base_ang_vel = np.zeros((num, 3))
        base_ang_vel[:, 2] = self.vyaw
        projected_gravity = np.tile([0.0, 0.0, -1.0], (num, 1))
        commands = np.column_stack([self.vx, self.vy, self.vyaw])
        base_lin_vel = np.column_stack([self.vx, self.vy, np.zeros(num)])
        props = np.concatenate([base_ang_vel, projected_gravity, commands, base_lin_vel], axis=1)

        rays = np.log2(np.clip(self.ray_dists, 0.1, self.max_ray_dist))

        goals = self._get_goal_local(self.robot_x, self.robot_y, self.robot_yaw,
                                      self.goal_x, self.goal_y)
        obs_step = np.concatenate([props, rays, goals], axis=1)

        self.obs_history = np.roll(self.obs_history, shift=-1, axis=1)
        self.obs_history[:, -1, :] = obs_step

        obs_flat = self.obs_history.reshape(num, -1)
        return torch.from_numpy(obs_flat).float().to(self.device)

    def get_observations(self) -> torch.Tensor:
        return self._get_observations()

    def get_num_envs(self) -> int:
        return self.num_envs
