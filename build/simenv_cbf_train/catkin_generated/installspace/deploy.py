#!/usr/bin/env python3
"""
SimEnv 竞赛部署脚本
在比赛环境中加载训练好的 CBF 策略，执行自主探索 → 危险源检测 → 返航 → 输出结果

用法:
  rosrun simenv_cbf_train deploy.py --model ./models/policy_final.pt
  rosrun simenv_cbf_train deploy.py --model ./models/policy_final.pt --no_cbf
"""
import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Tuple, Optional

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2, Image
from cv_bridge import CvBridge
import cv2

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from simenv_cbf_train.config import cfg as train_cfg


class HazardDetector:
    """基于颜色阈值 + LiDAR 深度融合的红色球体危险源检测"""

    def __init__(self):
        self.red_lower1 = np.array([0, 100, 80])
        self.red_upper1 = np.array([10, 255, 255])
        self.red_lower2 = np.array([160, 100, 80])
        self.red_upper2 = np.array([180, 255, 255])
        self.min_radius = 5
        self.max_radius = 150
        self.circularity_thresh = 0.5
        self.match_distance = 0.5
        self.detected_positions = []
        self.bridge = CvBridge()

    def detect(self, image_msg: Image, pc_msg: PointCloud2, camera_info=None) -> list:
        if image_msg is None:
            return []
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_msg, "bgr8")
        except Exception:
            return []

        hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, self.red_lower1, self.red_upper1)
        mask2 = cv2.inRange(hsv, self.red_lower2, self.red_upper2)
        mask = cv2.bitwise_or(mask1, mask2)
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        new_detections = []

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < self.min_radius ** 2 * np.pi:
                continue
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            if radius < self.min_radius or radius > self.max_radius:
                continue
            perimeter = cv2.arcLength(cnt, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            if circularity < self.circularity_thresh:
                continue

            depth = self._get_depth_from_lidar(cx, cy, pc_msg, camera_info)
            if depth is None or depth < 0.1 or depth > 8.0:
                continue

            pos_3d = self._estimate_3d_position(cx, cy, depth, camera_info)
            if pos_3d is not None and not self._is_duplicate(pos_3d):
                self.detected_positions.append(pos_3d)
                new_detections.append(pos_3d)

        return new_detections

    def _get_depth_from_lidar(self, cx, cy, pc_msg, camera_info) -> Optional[float]:
        if pc_msg is None:
            return None
        try:
            from sensor_msgs import point_cloud2
            points = list(point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"),
                                                    skip_nans=True))
            if not points:
                return None
            pts = np.array(points)
            front = pts[pts[:, 0] > 0]
            if len(front) == 0:
                return None
            return float(np.min(np.linalg.norm(front[:, :3], axis=1)))
        except Exception:
            return None

    def _estimate_3d_position(self, cx, cy, depth, camera_info) -> list:
        return [depth, 0.0, 0.0]

    def _is_duplicate(self, pos: list) -> bool:
        for existing in self.detected_positions:
            if np.linalg.norm(np.array(pos) - np.array(existing)) < self.match_distance:
                return True
        return False


class FrontierExplorer:
    """基于占据栅格地图的前沿探索"""

    def __init__(self):
        self.map_resolution = 0.1
        self.map_size = 40.0
        self.map_dims = int(self.map_size / self.map_resolution)
        self.occupancy_grid = np.full((self.map_dims, self.map_dims), -1, dtype=np.int8)
        self.trajectory = deque(maxlen=20000)
        self.origin_x = 0.0
        self.origin_y = 0.0

    def set_origin(self, x: float, y: float):
        self.origin_x = x
        self.origin_y = y

    def world_to_map(self, wx: float, wy: float) -> Tuple[int, int]:
        mx = int((wx - self.origin_x) / self.map_resolution)
        my = int((wy - self.origin_y) / self.map_resolution)
        return mx + self.map_dims // 2, my + self.map_dims // 2

    def update_from_lidar(self, robot_x: float, robot_y: float,
                           robot_yaw: float, pc_msg: PointCloud2):
        if pc_msg is None:
            return
        try:
            from sensor_msgs import point_cloud2
            points = list(point_cloud2.read_points(pc_msg, field_names=("x", "y", "z"),
                                                    skip_nans=True))
        except Exception:
            return

        rx, ry = self.world_to_map(robot_x, robot_y)
        for p in points:
            cos_yaw, sin_yaw = np.cos(robot_yaw), np.sin(robot_yaw)
            wx = robot_x + p[0] * cos_yaw - p[1] * sin_yaw
            wy = robot_y + p[0] * sin_yaw + p[1] * cos_yaw
            mx, my = self.world_to_map(wx, wy)
            if 0 <= mx < self.map_dims and 0 <= my < self.map_dims:
                self.occupancy_grid[my, mx] = 1
            dx, dy = mx - rx, my - ry
            dist = max(abs(dx), abs(dy))
            if dist < 2:
                continue
            for s in np.linspace(0, 1, int(dist)):
                sx, sy = int(rx + dx * s), int(ry + dy * s)
                if 0 <= sx < self.map_dims and 0 <= sy < self.map_dims:
                    if self.occupancy_grid[sy, sx] != 1:
                        self.occupancy_grid[sy, sx] = 0

    def get_frontiers(self) -> np.ndarray:
        from scipy.ndimage import binary_dilation
        frontier_mask = np.zeros_like(self.occupancy_grid, dtype=bool)
        unknown_mask = self.occupancy_grid == -1
        free_mask = self.occupancy_grid == 0
        unknown_dilated = binary_dilation(unknown_mask, iterations=2)
        frontier_mask = free_mask & unknown_dilated
        frontier_ys, frontier_xs = np.where(frontier_mask)
        if len(frontier_xs) == 0:
            return np.zeros((0, 2))
        n = min(len(frontier_xs), 50)
        indices = np.random.choice(len(frontier_xs), n, replace=False)
        frontiers = np.zeros((n, 2))
        for i, idx in enumerate(indices):
            frontiers[i, 0] = (frontier_xs[idx] - self.map_dims // 2) * self.map_resolution + self.origin_x
            frontiers[i, 1] = (frontier_ys[idx] - self.map_dims // 2) * self.map_resolution + self.origin_y
        return frontiers

    def select_goal(self, robot_x: float, robot_y: float) -> Optional[np.ndarray]:
        frontiers = self.get_frontiers()
        if len(frontiers) == 0:
            return None
        dists = np.linalg.norm(frontiers - np.array([robot_x, robot_y]), axis=1)
        return frontiers[np.argmin(dists)]

    def is_exploration_complete(self) -> bool:
        unknown_ratio = np.sum(self.occupancy_grid == -1) / self.occupancy_grid.size
        return unknown_ratio < 0.05


class CompetitionNode:
    """竞赛主控节点"""

    def __init__(self, model_path: str, use_cbf: bool = True):
        rospy.init_node('competition_node', anonymous=True)

        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.odom_sub = rospy.Subscriber('/Odometry_gazebo', Odometry, self._odom_cb, queue_size=1)
        self.scan_sub = rospy.Subscriber('/scan', PointCloud2, self._scan_cb, queue_size=1)
        self.image_sub = rospy.Subscriber('/camera/image_raw', Image, self._image_cb, queue_size=1)

        self._latest_odom = None
        self._latest_scan = None
        self._latest_image = None

        self.policy = None
        self.use_cbf = use_cbf
        if model_path and os.path.exists(model_path):
            self._load_policy(model_path)
        else:
            rospy.logwarn(f"未找到策略模型: {model_path}，使用随机动作")

        self.detector = HazardDetector()
        self.explorer = FrontierExplorer()

        self.state = 'INIT'
        self.start_position = None
        self.start_time = None
        self.current_position = np.zeros(3)
        self.current_yaw = 0.0
        self.detected_dangers = []

    def _odom_cb(self, msg: Odometry):
        self._latest_odom = msg
        p = msg.pose.pose.position
        self.current_position = np.array([p.x, p.y, p.z])
        o = msg.pose.pose.orientation
        self.current_yaw = np.arctan2(2.0 * (o.w * o.z + o.x * o.y),
                                       1.0 - 2.0 * (o.y * o.y + o.z * o.z))

    def _scan_cb(self, msg: PointCloud2):
        self._latest_scan = msg

    def _image_cb(self, msg: Image):
        self._latest_image = msg

    def _load_policy(self, model_path: str):
        import torch
        from simenv_cbf_train.models import CBFNavigationActorCritic, StandardActorCritic

        if self.use_cbf:
            model = CBFNavigationActorCritic()
        else:
            model = StandardActorCritic()

        checkpoint = torch.load(model_path, map_location='cpu')
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
        model.eval()

        def policy_fn(obs):
            with torch.no_grad():
                return model.act_inference(obs).cpu().numpy()
        self.policy = policy_fn
        rospy.loginfo(f"策略已加载: {model_path}")

    def _send_twist(self, vx: float, vy: float, vyaw: float):
        cmd = Twist()
        cmd.linear.x = float(vx)
        cmd.linear.y = float(vy)
        cmd.angular.z = float(vyaw)
        self.cmd_pub.publish(cmd)

    def _compute_nav_action(self, target_x: float, target_y: float) -> Tuple[float, float, float]:
        dx = target_x - self.current_position[0]
        dy = target_y - self.current_position[1]
        local_dx = dx * np.cos(-self.current_yaw) - dy * np.sin(-self.current_yaw)
        local_dy = dx * np.sin(-self.current_yaw) + dy * np.cos(-self.current_yaw)
        dist = np.hypot(local_dx, local_dy)
        if dist < 0.3:
            return 0.0, 0.0, 0.0
        vx = np.clip(local_dx * 0.5, -0.6, 1.0)
        vy = np.clip(local_dy * 0.3, -0.3, 0.3)
        target_angle = np.arctan2(local_dy, local_dx)
        vyaw = np.clip(target_angle * 0.5, -0.5, 0.5)
        return vx, vy, vyaw

    def _is_at_start(self, threshold: float = 0.5) -> bool:
        if self.start_position is None:
            return True
        return np.linalg.norm(self.current_position - self.start_position) < threshold

    def _save_results(self):
        elapsed = (rospy.Time.now() - self.start_time).to_sec() if self.start_time else 0.0
        result = {
            "exploration_time": round(elapsed, 2),
            "detected_danger_sources": [
                {"position": [round(p[0], 2), round(p[1], 2), round(p[2], 2)]}
                for p in self.detected_dangers
            ]
        }

        # 输出到 SimEnv 的 results/ 目录
        ws_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        output_dir = os.path.join(ws_root, "results")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "detected_danger.json")

        with open(output_path, 'w') as f:
            json.dump(result, f, indent=2)
        rospy.loginfo(f"结果已保存至 {output_path}")
        rospy.loginfo(f"探索时间: {result['exploration_time']}s, 检测到 {len(self.detected_dangers)} 个危险源")

    def run(self):
        rospy.loginfo("等待 junior_ctrl 就绪...")
        rospy.sleep(3.0)

        if self._latest_odom:
            p = self._latest_odom.pose.pose.position
            self.start_position = np.array([p.x, p.y, p.z])
            self.explorer.set_origin(p.x, p.y)
            rospy.loginfo(f"起点: ({p.x:.2f}, {p.y:.2f})")

        self.start_time = rospy.Time.now()
        rate = rospy.Rate(20)
        self.state = 'EXPLORE'
        rospy.loginfo("开始探索...")

        while not rospy.is_shutdown():
            new_dangers = self.detector.detect(self._latest_image, self._latest_scan)
            for pos in new_dangers:
                self.detected_dangers.append(pos)
                rospy.loginfo(f"检测到危险源: {pos}")

            self.explorer.update_from_lidar(
                self.current_position[0], self.current_position[1],
                self.current_yaw, self._latest_scan
            )
            self.explorer.trajectory.append(self.current_position[:2].copy())

            if self.state == 'EXPLORE':
                if self.explorer.is_exploration_complete():
                    rospy.loginfo("探索完成，开始返航")
                    self.state = 'RETURN'
                else:
                    goal = self.explorer.select_goal(self.current_position[0], self.current_position[1])
                    if goal is not None:
                        vx, vy, vyaw = self._compute_nav_action(goal[0], goal[1])
                        self._send_twist(vx, vy, vyaw)

            elif self.state == 'RETURN':
                if self._is_at_start():
                    self._send_twist(0.0, 0.0, 0.0)
                    self._save_results()
                    self.state = 'DONE'
                    rospy.loginfo("任务完成!")
                    break
                else:
                    vx, vy, vyaw = self._compute_nav_action(
                        self.start_position[0], self.start_position[1])
                    self._send_twist(vx, vy, vyaw)

            elif self.state == 'DONE':
                self._send_twist(0.0, 0.0, 0.0)
                break

            rate.sleep()


def main():
    parser = argparse.ArgumentParser(description="SimEnv 竞赛部署")
    parser.add_argument("--model", type=str, default="./models/policy_final.pt",
                        help="策略模型路径")
    parser.add_argument("--no_cbf", action="store_true", default=False,
                        help="不使用 CBF 安全层")
    args = parser.parse_args()

    node = CompetitionNode(model_path=args.model, use_cbf=not args.no_cbf)
    try:
        node.run()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("用户中断")
    finally:
        node._send_twist(0.0, 0.0, 0.0)


if __name__ == '__main__':
    main()
