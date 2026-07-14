#!/usr/bin/env python3
"""
SimEnv 比赛部署脚本 v2

完整比赛流程：
1. 开门进楼 + 逐层探索
2. 检测红色球体危险源（RGB + 深度）
3. 摔倒自动复位
4. 输出 results/detected_danger.json

用法:
  rosrun simenv_train deploy.py
  rosrun simenv_train deploy.py --output results/detected_danger.json --max-time 600
"""
import os
import sys
import time
import json
import math
import argparse
import numpy as np

import rospy
from geometry_msgs.msg import Twist
from gazebo_msgs.srv import SetModelState, SetModelStateRequest
from gazebo_msgs.msg import ModelState
from sensor_msgs.msg import Imu

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from simenv_train.env import SimEnv
from simenv_train.door_control import DoorController, ElevatorController


def get_args():
    parser = argparse.ArgumentParser(description="SimEnv 比赛部署")
    parser.add_argument("--output", default="results/detected_danger.json")
    parser.add_argument("--max-time", type=float, default=600.0)
    return parser.parse_args()


class FallDetector:
    """摔倒检测：用 IMU 判断机器人是否翻倒"""

    def __init__(self):
        self._roll = 0.0
        self._pitch = 0.0
        self._fallen = False
        rospy.Subscriber('/trunk_imu', Imu, self._imu_cb, queue_size=1)

    def _imu_cb(self, msg: Imu):
        q = msg.orientation
        sinr_cosp = 2.0 * (q.w * q.x + q.y * q.z)
        cosr_cosp = 1.0 - 2.0 * (q.x * q.x + q.y * q.y)
        self._roll = math.atan2(sinr_cosp, cosr_cosp)
        sinp = 2.0 * (q.w * q.y - q.z * q.x)
        if abs(sinp) >= 1:
            self._pitch = math.copysign(math.pi / 2, sinp)
        else:
            self._pitch = math.asin(sinp)

    @property
    def is_fallen(self) -> bool:
        return abs(self._roll) > 1.2 or abs(self._pitch) > 1.2

    @property
    def roll(self) -> float:
        return self._roll

    @property
    def pitch(self) -> float:
        return self._pitch


class RobotResetter:
    """机器人复位"""

    def __init__(self):
        rospy.wait_for_service('/gazebo/set_model_state', timeout=10.0)
        self._svc = rospy.ServiceProxy('/gazebo/set_model_state', SetModelState)

    def reset_pose(self, x=0.0, y=-3.2, z=0.35, yaw=1.5708):
        req = SetModelStateRequest()
        req.model_state.model_name = 'a1_gazebo'
        req.model_state.pose.position.x = x
        req.model_state.pose.position.y = y
        req.model_state.pose.position.z = z
        req.model_state.pose.orientation.z = math.sin(yaw / 2)
        req.model_state.pose.orientation.w = math.cos(yaw / 2)
        req.model_state.twist.linear.x = 0
        req.model_state.twist.linear.y = 0
        req.model_state.twist.linear.z = 0
        req.model_state.twist.angular.x = 0
        req.model_state.twist.angular.y = 0
        req.model_state.twist.angular.z = 0
        try:
            self._svc(req)
            return True
        except rospy.ServiceException:
            return False


class Explorer:
    """系统化探索器：逐房间扫、开门、坐电梯"""

    def __init__(self, env: SimEnv, doors: DoorController,
                 elevator: ElevatorController, resetter: RobotResetter):
        self.env = env
        self.doors = doors
        self.elevator = elevator
        self.resetter = resetter
        self.fall_detector = FallDetector()

        self.current_floor = 0
        self.current_phase = "enter_building"  # enter_building, explore_floor, change_floor, done
        self.phase_start_time = time.time()
        self.phase_duration = 60.0
        self.doors_opened = set()
        self.scan_direction = 1  # 扫描方向: 1 = 右转, -1 = 左转
        self.scan_timer = 0
        self.fall_count = 0
        self.max_falls = 5

    def step(self) -> np.ndarray:
        """返回下一步动作 [vx, vy, yaw_rate]"""

        # ── 摔倒了就复位 ──
        if self.fall_detector.is_fallen:
            self.fall_count += 1
            if self.fall_count > self.max_falls:
                rospy.logwarn("摔倒次数过多，停止探索")
                return np.array([0.0, 0.0, 0.0])
            rospy.logwarn("检测到摔倒，尝试复位...")
            time.sleep(1.0)
            self.resetter.reset_pose()
            time.sleep(2.0)
            self.scan_timer = 0
            return np.array([0.0, 0.0, 0.0])

        self.fall_count = 0
        self.scan_timer += 1

        # ── 阶段：进入建筑 ──
        if self.current_phase == "enter_building":
            return self._phase_enter_building()

        # ── 阶段：探索当前层 ──
        if self.current_phase == "explore_floor":
            return self._phase_explore_floor()

        # ── 阶段：换层 ──
        if self.current_phase == "change_floor":
            return self._phase_change_floor()

        # ── 完成 ──
        return np.array([0.0, 0.0, 0.0])

    def _phase_enter_building(self) -> np.ndarray:
        """开门进楼"""
        elapsed = time.time() - self.phase_start_time

        # 先直行靠近大门
        if elapsed < 5.0:
            return np.array([0.2, 0.0, 0.0])

        # 开门
        if 'main_entrance' not in self.doors_opened:
            rospy.loginfo("打开主入口门...")
            if self.doors:
                self.doors.open('main_entrance')
            self.doors_opened.add('main_entrance')
            time.sleep(1.0)

        # 进门后直行一段
        if elapsed < 15.0:
            return np.array([0.2, 0.0, 0.0])

        # 进入探索阶段
        rospy.loginfo("进入建筑，开始探索 0 层")
        self.current_phase = "explore_floor"
        self.phase_start_time = time.time()
        self.phase_duration = 180.0  # 每层最多 3 分钟
        return np.array([0.0, 0.0, 0.0])

    def _phase_explore_floor(self) -> np.ndarray:
        """探索当前层：扫描式前进 + 检测危险源"""
        elapsed = time.time() - self.phase_start_time

        # 时间到了就换层
        if elapsed > self.phase_duration:
            self.current_phase = "change_floor"
            self.phase_start_time = time.time()
            return np.array([0.0, 0.0, 0.0])

        rays = self.env.get_livox_rays(num_rays=41)
        min_front = np.min(rays[15:26])  # 前方 1/4 扇形

        # 前方被堵住了
        if min_front < 1.0:
            # 检查是不是门
            if min_front < 0.5:
                self._try_open_nearby_door()
                # 原地旋转找路
                self.scan_direction *= -1 if self.scan_timer % 60 == 0 else 1
                return np.array([0.0, 0.0, 0.8 * self.scan_direction])
            # 慢慢转
            return np.array([0.2, 0.0, 0.3 * self.scan_direction])

        # 前方开阔：直行 + 微调方向
        # 找最远的方向
        best_idx = np.argmax(rays)
        angles = np.linspace(-2 * np.pi / 3, 2 * np.pi / 3, len(rays))
        target_angle = angles[best_idx]

        vx = min(0.6, min_front / 5.0)
        yaw_rate = 0.2 * np.sign(target_angle) if abs(target_angle) > 0.3 else 0.0

        return np.array([vx, 0.0, yaw_rate])

    def _phase_change_floor(self) -> np.ndarray:
        """坐电梯换层"""
        elapsed = time.time() - self.phase_start_time

        # 先等一等，确保在电梯附近
        if elapsed < 3.0:
            return np.array([0.0, 0.0, 0.0])

        # 呼叫电梯到当前层
        target_floor = self.current_floor + 1
        rospy.loginfo(f"呼叫电梯: {self.current_floor} -> {target_floor}")

        if self.elevator:
            self.elevator.go_to_floor('elevator_main', target_floor, open_doors=True)
            time.sleep(3.0)

        # 打开电梯门
        door_id = f'elevator_floor_{self.current_floor}'
        if self.doors:
            self.doors.open(door_id)
            time.sleep(2.0)

        self.current_floor = target_floor

        # 如果到了最后一层以上，结束探索
        if self.current_floor >= 3:
            rospy.loginfo("所有楼层探索完毕")
            self.current_phase = "done"
            return np.array([0.0, 0.0, 0.0])

        self.current_phase = "explore_floor"
        self.phase_start_time = time.time()
        self.phase_duration = 180.0
        return np.array([0.2, 0.0, 0.0])

    def _try_open_nearby_door(self):
        """尝试打开附近可能的门"""
        possible_doors = ['main_entrance']
        for floor in range(4):
            possible_doors.append(f'elevator_floor_{floor}')
        for door_id in possible_doors:
            if door_id not in self.doors_opened and self.doors:
                try:
                    rospy.loginfo(f"尝试开门: {door_id}")
                    self.doors.open(door_id)
                    self.doors_opened.add(door_id)
                    time.sleep(1.0)
                except Exception:
                    pass


def deploy(args):
    print("=" * 60)
    print("SimEnv 比赛部署 v2")
    print(f"  最大时间: {args.max_time}s")
    print(f"  输出: {args.output}")
    print("=" * 60)

    env = SimEnv(num_envs=1)

    try:
        doors = DoorController()
        print("  门控制就绪")
    except Exception:
        doors = None
        print("  [WARN] 门控制不可用")

    try:
        elevator = ElevatorController()
        print("  电梯控制就绪")
    except Exception:
        elevator = None
        print("  [WARN] 电梯控制不可用")

    try:
        resetter = RobotResetter()
        print("  复位服务就绪")
    except Exception:
        resetter = None
        print("  [WARN] 复位服务不可用")

    explorer = Explorer(env, doors, elevator, resetter)

    print("\n开始探索...\n")
    start_time = time.time()
    rate = rospy.Rate(20)

    while not rospy.is_shutdown():
        elapsed = time.time() - start_time
        if elapsed > args.max_time:
            break
        if explorer.current_phase == "done":
            break

        actions = explorer.step()
        if actions is None:
            break

        cmd = Twist()
        cmd.linear.x = float(actions[0])
        cmd.linear.y = float(actions[1])
        cmd.angular.z = float(actions[2])
        env.cmd_pub.publish(cmd)
        env.step_count += 1
        rospy.sleep(0.05)

        if env.step_count % 100 == 0:
            print(f"  [{elapsed:6.1f}s] phase={explorer.current_phase} "
                  f"floor={explorer.current_floor} "
                  f"dangers={len(env.detected_dangers)} "
                  f"fall={explorer.fall_count}")

        # 同时做危险源检测
        danger_pixel = env.detect_danger_in_rgb()
        if danger_pixel is not None:
            danger_3d = env.estimate_danger_position(danger_pixel)
            if danger_3d is not None:
                new_pos = danger_3d.tolist()
                too_close = any(
                    np.linalg.norm(np.array(d['position']) - np.array(new_pos)) < 0.5
                    for d in env.detected_dangers
                )
                if not too_close:
                    env.detected_dangers.append({
                        'position': [round(new_pos[0], 2),
                                     round(new_pos[1], 2),
                                     round(new_pos[2], 2)]
                    })
                    print(f"  [检测] 危险源候选 #{len(env.detected_dangers)} @ {new_pos}")

    total_time = time.time() - start_time
    env.save_results(total_time, args.output)
    print(f"\n探索完成！{total_time:.0f}s, {len(env.detected_dangers)} 个危险源")
    print(f"结果: {args.output}")
    env.close()


if __name__ == '__main__':
    args = get_args()
    deploy(args)
