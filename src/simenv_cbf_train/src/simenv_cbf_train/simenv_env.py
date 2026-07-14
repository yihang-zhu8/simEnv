"""
SimEnv ROS-Gazebo 环境封装
将比赛仿真环境封装为标准 gym 式接口

ROS 接口:
  发布: /cmd_vel (geometry_msgs/Twist)
  订阅: /Odometry_gazebo, /trunk_imu, /scan, /camera/image_raw

前提:
  auto.sh 已启动 Gazebo + A1 + junior_ctrl
  junior_ctrl 已进入 RL 模式（按键 2→站立, 6→RL）
"""
import time
import numpy as np
import torch
from typing import Tuple, Optional

import rospy
from sensor_msgs.msg import PointCloud2, Image, Imu
from nav_msgs.msg import Odometry
from geometry_msgs.msg import Twist

from simenv_cbf_train.config import cfg


class SimEnvROSEnv:
    """SimEnv ROS-Gazebo 环境封装

    将 ROS 话题封装为标准的 step/reset/get_observations 接口，
    供 PPO 训练循环直接调用。
    """

    def __init__(self, num_envs: int = 1, namespace: str = ""):
        self.num_envs = num_envs
        self.ns = namespace

        if not rospy.get_node_uri():
            rospy.init_node('simenv_train_env', anonymous=True, disable_signals=True)

        # ── 发布 ──
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # ── 订阅缓存 ──
        self._latest_odom = None
        self._latest_imu = None
        self._latest_scan = None
        self._latest_image = None

        rospy.Subscriber('/Odometry_gazebo', Odometry, self._odom_cb, queue_size=1)
        rospy.Subscriber('/trunk_imu', Imu, self._imu_cb, queue_size=1)
        rospy.Subscriber('/scan', PointCloud2, self._scan_cb, queue_size=1)
        rospy.Subscriber('/camera/image_raw', Image, self._image_cb, queue_size=1)

        rospy.loginfo("等待传感器数据...")
        self._wait_for_sensors()

        self.his_len = cfg.env.his_len
        self.obs_one_step = cfg.env.num_obs_one_step
        self.obs_history = np.zeros((self.num_envs, self.his_len, self.obs_one_step))

        self.goal_position = None
        self.step_count = 0
        self.max_steps = 800

        # 上一个距离（进度奖励用）
        self.last_dist = None

        rospy.loginfo("SimEnvROSEnv 初始化完成")

    def _wait_for_sensors(self, timeout: float = 15.0):
        start = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._latest_odom is not None and self._latest_imu is not None:
                rospy.loginfo("传感器就绪")
                return
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn(f"传感器等待超时 ({timeout}s)")
                return
            rate.sleep()

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg

    def _imu_cb(self, msg: Imu):
        self._latest_imu = msg

    def _scan_cb(self, msg: PointCloud2):
        self._latest_scan = msg

    def _image_cb(self, msg: Image):
        self._latest_image = msg

    def get_robot_pose(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._latest_odom is None:
            return np.zeros(3), np.array([0, 0, 0, 1])
        p = self._latest_odom.pose.pose.position
        o = self._latest_odom.pose.pose.orientation
        return np.array([p.x, p.y, p.z]), np.array([o.x, o.y, o.z, o.w])

    def get_twist(self) -> np.ndarray:
        if self._latest_odom is None:
            return np.zeros(6)
        lin = self._latest_odom.twist.twist.linear
        ang = self._latest_odom.twist.twist.angular
        return np.array([lin.x, lin.y, lin.z, ang.x, ang.y, ang.z])

    def get_imu(self) -> Tuple[np.ndarray, np.ndarray]:
        if self._latest_imu is None:
            return np.zeros(3), np.zeros(3)
        ang = self._latest_imu.angular_velocity
        acc = self._latest_imu.linear_acceleration
        return np.array([ang.x, ang.y, ang.z]), np.array([acc.x, acc.y, acc.z])

    def process_pointcloud(self, pc_msg: PointCloud2) -> np.ndarray:
        """PointCloud2 → 2D 射线距离"""
        if pc_msg is None:
            return np.ones(cfg.env.num_rays) * 5.0

        from sensor_msgs import point_cloud2
        points = list(point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"),
                                                skip_nans=True))
        if len(points) == 0:
            return np.ones(cfg.env.num_rays) * 5.0

        ray_angles = np.arange(cfg.simple_env.theta_start, cfg.simple_env.theta_end,
                               cfg.simple_env.theta_step)
        num_rays = len(ray_angles)
        ray_dists = np.ones(num_rays) * 5.0
        points_np = np.array(points)

        for i, angle in enumerate(ray_angles):
            point_angles = np.arctan2(points_np[:, 1], points_np[:, 0])
            angle_diff = np.abs(point_angles - angle)
            angle_diff = np.minimum(angle_diff, 2 * np.pi - angle_diff)
            mask = angle_diff < np.deg2rad(3)
            if mask.any():
                dists = np.linalg.norm(points_np[mask, :2], axis=1)
                ray_dists[i] = np.min(dists)

        return ray_dists

    def get_observations(self) -> torch.Tensor:
        ang_vel, _ = self.get_imu()
        twist = self.get_twist()
        pos, ori = self.get_robot_pose()

        # Props (12 维)
        props = np.concatenate([
            ang_vel[:3],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 0.0],
            twist[:3],
        ])

        # 2D 射线
        ray_dists = self.process_pointcloud(self._latest_scan)
        rays = np.log2(np.clip(ray_dists, 0.1, 5.0))

        # 目标（局部坐标）
        if self.goal_position is not None:
            dx = self.goal_position[0] - pos[0]
            dy = self.goal_position[1] - pos[1]
            # 提取 yaw 并旋转
            qx, qy, qz, qw = ori
            yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
            goal_local_x = dx * np.cos(-yaw) - dy * np.sin(-yaw)
            goal_local_y = dx * np.sin(-yaw) + dy * np.cos(-yaw)
            goal_local = np.array([goal_local_x, goal_local_y])
        else:
            goal_local = np.zeros(2)

        obs_step = np.concatenate([props, rays, goal_local])

        self.obs_history = np.roll(self.obs_history, shift=-1, axis=1)
        self.obs_history[0, -1, :] = obs_step

        obs_flat = self.obs_history.reshape(1, -1)
        return torch.from_numpy(obs_flat).float().to(cfg.device)

    def step(self, actions: np.ndarray, auto_reward: bool = True) -> Tuple:
        """执行一步

        Args:
            actions: [1, 3] (vx, vy, yaw_rate)，归一化值
            auto_reward: 是否自动计算奖励
        Returns:
            next_obs, rewards, dones, infos
        """
        # 发布速度指令
        cmd = Twist()
        cmd.linear.x = float(actions[0, 0])
        cmd.linear.y = float(actions[0, 1])
        cmd.angular.z = float(actions[0, 2])
        self.cmd_pub.publish(cmd)

        # 等待控制周期
        rospy.sleep(cfg.env.dt)

        self.step_count += 1

        next_obs = self.get_observations()
        rewards = np.zeros(self.num_envs)
        dones = np.zeros(self.num_envs, dtype=bool)

        if auto_reward:
            rewards[0] = self._compute_reward()

        if self.step_count >= self.max_steps:
            dones[0] = True

        time_outs = np.array([self.step_count >= self.max_steps])

        return next_obs, rewards, dones, {'time_outs': time_outs}

    def _compute_reward(self) -> float:
        """计算奖励：进度 + 安全"""
        reward = 0.0

        # 前进奖励
        twist = self.get_twist()
        vx = twist[0]
        reward += 0.1 * vx  # 鼓励前进

        # 安全惩罚（离障碍物太近）
        ray_dists = self.process_pointcloud(self._latest_scan)
        min_dist = np.min(ray_dists)
        if min_dist < 0.3:
            reward -= 1.0
        elif min_dist < 0.5:
            reward -= 0.3

        # 偏航惩罚（鼓励直行）
        vy = twist[1]
        reward -= 0.1 * abs(vy)

        # 碰撞终止惩罚
        if min_dist < 0.15:
            reward -= 5.0

        return float(reward)

    def reset(self, env_ids=None):
        """重置环境

        注意：在 Gazebo 中无法简单"重置"，只能重置内部状态。
        实际训练中由 auto.sh 重新生成场景。
        """
        self.step_count = 0
        self.last_dist = None
        self.obs_history = np.zeros((self.num_envs, self.his_len, self.obs_one_step))
        return self.get_observations()

    def set_goal(self, x: float, y: float):
        """设置导航目标点（世界坐标）"""
        self.goal_position = (x, y)

    def get_num_envs(self) -> int:
        return self.num_envs

    def close(self):
        pass
