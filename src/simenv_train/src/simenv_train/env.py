"""
SimEnv 比赛合规环境

只使用比赛允许的话题：
  /trunk_imu, /scan, /livox/Pointcloud2,
  /real_sense/rgb/image_raw, /real_sense/depth/points
  /cmd_vel (输出)

禁止使用（比赛规则明确禁止）：
  /Odometry_gazebo, /ground_truth/*
"""
import time
import json
import numpy as np
from typing import Tuple, List, Optional

import rospy
from sensor_msgs.msg import PointCloud2, Image, Imu, PointCloud
from geometry_msgs.msg import Twist


class SimEnv:
    """比赛合规的 SimEnv 环境封装

    关键设计：
    - 不订阅 /Odometry_gazebo 和 /ground_truth
    - 用 IMU 积分估计姿态（配合点云做简单的定位）
    - 用 RealSense RGB 检测危险源（红色球体）
    - 用 Livox 点云感知障碍物
    """

    # 危险源颜色：红色 (RGB)
    DANGER_RED_LOW = np.array([0, 100, 50])      # HSV lower
    DANGER_RED_HIGH = np.array([20, 255, 255])   # HSV upper
    DANGER_RED2_LOW = np.array([160, 100, 50])   # 红色在 HSV 有两段
    DANGER_RED2_HIGH = np.array([180, 255, 255])

    def __init__(self, num_envs: int = 1):
        self.num_envs = num_envs

        if not rospy.get_node_uri():
            rospy.init_node('simenv_env', anonymous=True, disable_signals=True)

        # ── 输出 ──
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)

        # ── 订阅（仅允许的话题） ──
        self._imu: Optional[Imu] = None
        self._scan: Optional[PointCloud] = None
        self._livox_pc2: Optional[PointCloud2] = None
        self._rgb: Optional[Image] = None
        self._depth_pts: Optional[PointCloud2] = None

        # ── 内部状态（必须在订阅之前初始化！） ──
        self.step_count = 0
        self.max_steps = 12000
        self.goal_position: Optional[Tuple[float, float]] = None
        self.visited_rooms: set = set()
        self.detected_dangers: List[dict] = []
        self._pose_x = 0.0
        self._pose_y = 0.0
        self._pose_yaw = 0.0
        self._last_imu_time: Optional[rospy.Time] = None

        # ── 订阅（仅允许的话题） ──
        rospy.Subscriber('/trunk_imu', Imu, self._imu_cb, queue_size=1)
        rospy.Subscriber('/scan', PointCloud, self._scan_cb, queue_size=1)
        rospy.Subscriber('/livox/Pointcloud2', PointCloud2, self._livox_cb, queue_size=1)
        rospy.Subscriber('/real_sense/rgb/image_raw', Image, self._rgb_cb, queue_size=1)
        rospy.Subscriber('/real_sense/depth/points', PointCloud2, self._depth_cb, queue_size=1)

        rospy.loginfo("SimEnv: 等待传感器数据...")
        self._wait_sensors(timeout=20.0)

        rospy.loginfo("SimEnv 初始化完成（合规模式）")

    # ── 传感器回调 ──
    def _imu_cb(self, msg: Imu):
        self._imu = msg
        self._update_imu_odom(msg)

    def _scan_cb(self, msg: PointCloud):
        self._scan = msg

    def _livox_cb(self, msg: PointCloud2):
        self._livox_pc2 = msg

    def _rgb_cb(self, msg: Image):
        self._rgb = msg

    def _depth_cb(self, msg: PointCloud2):
        self._depth_pts = msg

    def _wait_sensors(self, timeout: float):
        start = rospy.Time.now()
        rate = rospy.Rate(10)
        while not rospy.is_shutdown():
            if self._imu is not None and self._scan is not None:
                return
            if (rospy.Time.now() - start).to_sec() > timeout:
                rospy.logwarn(f"传感器等待超时 ({timeout}s)")
                return
            rate.sleep()

    # ── 简单 IMU 里程计（无真值依赖） ──
    def _update_imu_odom(self, msg: Imu):
        if self._last_imu_time is None:
            self._last_imu_time = msg.header.stamp
            return
        dt = (msg.header.stamp - self._last_imu_time).to_sec()
        if dt <= 0 or dt > 0.1:
            self._last_imu_time = msg.header.stamp
            return

        self._pose_yaw += msg.angular_velocity.z * dt
        # 只靠 IMU 积分做简单位置估计（会有漂移，训练用）
        self._last_imu_time = msg.header.stamp

    def get_imu_pose(self) -> Tuple[float, float, float]:
        """返回 IMU 估计的姿态 (x, y, yaw)，仅用于本地训练"""
        return self._pose_x, self._pose_y, self._pose_yaw

    def get_imu_data(self) -> np.ndarray:
        """获取 IMU 角速度和加速度"""
        if self._imu is None:
            return np.zeros(6)
        ang = self._imu.angular_velocity
        acc = self._imu.linear_acceleration
        return np.array([ang.x, ang.y, ang.z, acc.x, acc.y, acc.z])

    # ── 点云处理 ──
    def get_livox_rays(self, num_rays: int = 41,
                       fov_start: float = -2.0 * np.pi / 3,
                       fov_end: float = 2.0 * np.pi / 3) -> np.ndarray:
        """处理 Livox PointCloud → 2D 射线距离"""
        if self._scan is None:
            return np.ones(num_rays) * 5.0

        pts = np.array([[p.x, p.y, p.z] for p in self._scan.points])
        if len(pts) == 0:
            return np.ones(num_rays) * 5.0

        # 保留机器人高度附近的点
        height_mask = (pts[:, 2] > -0.3) & (pts[:, 2] < 0.5)
        pts = pts[height_mask]
        if len(pts) == 0:
            return np.ones(num_rays) * 5.0

        angles = np.linspace(fov_start, fov_end, num_rays)
        dists = np.ones(num_rays) * 5.0

        for i, angle in enumerate(angles):
            pt_angles = np.arctan2(pts[:, 1], pts[:, 0])
            diff = np.abs(pt_angles - angle)
            diff = np.minimum(diff, 2 * np.pi - diff)
            mask = diff < np.deg2rad(4)
            if mask.any():
                dists[i] = np.min(np.linalg.norm(pts[mask, :2], axis=1))

        return dists

    # ── 危险源检测 ──
    def detect_danger_in_rgb(self) -> Optional[np.ndarray]:
        """从 RGB 图像检测红色球体（危险源）"""
        if self._rgb is None:
            return None

        try:
            import cv2
            from cv_bridge import CvBridge
            bridge = CvBridge()
            img = bridge.imgmsg_to_cv2(self._rgb, "bgr8")
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

            mask1 = cv2.inRange(hsv, self.DANGER_RED_LOW, self.DANGER_RED_HIGH)
            mask2 = cv2.inRange(hsv, self.DANGER_RED2_LOW, self.DANGER_RED2_HIGH)
            mask = cv2.bitwise_or(mask1, mask2)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                return None

            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            if area < 50:
                return None

            M = cv2.moments(largest)
            if M["m00"] == 0:
                return None
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            return np.array([cx, cy, area])
        except ImportError:
            return None

    def estimate_danger_position(self, pixel_center: np.ndarray) -> Optional[np.ndarray]:
        """结合深度图估计危险源 3D 位置"""
        if self._depth_pts is None:
            return None

        try:
            from sensor_msgs import point_cloud2
            points = list(point_cloud2.read_points(
                self._depth_pts, field_names=("x", "y", "z"), skip_nans=True))
            if len(points) == 0:
                return None
            # 取深度图中心区域的平均距离
            pts_arr = np.array(points)
            center_dists = np.linalg.norm(pts_arr[:, :2].reshape(-1, 2), axis=1)
            median_mask = (center_dists > np.percentile(center_dists, 40)) & \
                          (center_dists < np.percentile(center_dists, 60))
            if median_mask.any():
                return np.mean(pts_arr[median_mask], axis=0)
            return np.mean(pts_arr, axis=0)
        except Exception:
            return None

    # ── 环境接口 ──
    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """执行一步控制

        Args:
            actions: [env, 3] (vx, vy, yaw_rate), 归一化值 [-1, 1]
        Returns:
            obs, rewards, dones, infos
        """
        cmd = Twist()
        cmd.linear.x = float(actions[0, 0])
        cmd.linear.y = float(actions[0, 1])
        cmd.angular.z = float(actions[0, 2])
        self.cmd_pub.publish(cmd)

        rospy.sleep(0.05)  # 20 Hz
        self.step_count += 1

        obs = self._build_observation()
        rewards = self._compute_rewards()
        dones = np.array([self.step_count >= self.max_steps])
        infos = {
            'time_outs': dones.copy(),
            'step': self.step_count,
        }

        # 检测危险源
        danger_pixel = self.detect_danger_in_rgb()
        if danger_pixel is not None:
            danger_3d = self.estimate_danger_position(danger_pixel)
            if danger_3d is not None:
                # 去重
                new_pos = danger_3d.tolist()
                too_close = any(
                    np.linalg.norm(np.array(d['position']) - np.array(new_pos)) < 0.5
                    for d in self.detected_dangers
                )
                if not too_close:
                    self.detected_dangers.append({
                        'position': [round(new_pos[0], 2),
                                     round(new_pos[1], 2),
                                     round(new_pos[2], 2)],
                        'confidence': min(area / 500.0, 1.0) if danger_pixel[2] > 0 else 0.5
                    })

        return obs, rewards, dones, infos

    def _build_observation(self) -> np.ndarray:
        """构建观测向量（只用允许的话题）

        观测结构 (48 维):
        - IMU 角速度 + 加速度: 6
        - IMU 估计 yaw: 1
        - Livox 2D 射线 (41 束): 41
        共 48 维
        """
        imu_data = self.get_imu_data()
        yaw = np.array([self._pose_yaw])
        rays = self.get_livox_rays()

        obs = np.concatenate([imu_data, yaw, rays])
        return obs.astype(np.float32)

    def _compute_rewards(self) -> np.ndarray:
        """计算奖励（不用真值里程计，只用局部信息）"""
        rewards = np.zeros(self.num_envs)
        reward = 0.0

        # IMU 稳定性奖励（角速度小 → 稳定）
        if self._imu is not None:
            ang_norm = abs(self._imu.angular_velocity.z)
            reward += 0.5 - 0.1 * ang_norm  # 鼓励减小旋转

        # 前进奖励（假设机器人主要向前走）
        # 用 IMU 加速度判断是否有运动
        if self._imu is not None:
            acc_mag = np.linalg.norm([
                self._imu.linear_acceleration.x,
                self._imu.linear_acceleration.y
            ])
            if 0.5 < acc_mag < 3.0:
                reward += 0.3  # 适中的加速度说明在移动

        # 障碍物惩罚
        rays = self.get_livox_rays()
        min_dist = np.min(rays)
        if min_dist < 0.3:
            reward -= 2.0
        elif min_dist < 0.5:
            reward -= 0.5
        else:
            reward += 0.1  # 有安全空间的奖励

        # 检测到危险源的奖励
        if self.step_count % 200 == 0:  # 每 10 秒检查
            if len(self.detected_dangers) > 0:
                reward += 2.0 * len(self.detected_dangers)

        rewards[0] = float(reward)
        return rewards

    def reset(self) -> np.ndarray:
        """重置环境状态"""
        self.step_count = 0
        self.detected_dangers = []
        self._pose_x = 0.0
        self._pose_y = 0.0
        self._pose_yaw = 0.0
        return self._build_observation()

    def set_goal(self, x: float, y: float):
        self.goal_position = (x, y)

    def get_num_envs(self) -> int:
        return self.num_envs

    def save_results(self, exploration_time: float, output_path: str = "results/detected_danger.json"):
        """保存比赛结果文件"""
        result = {
            "exploration_time": round(exploration_time, 2),
            "detected_danger_sources": [
                {"position": d['position']}
                for d in self.detected_dangers
                if d.get('confidence', 0) > 0.3
            ]
        }
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"结果已保存: {output_path} ({len(result['detected_danger_sources'])} 个检测)")

    def close(self):
        pass
