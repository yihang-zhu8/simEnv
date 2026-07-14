#!/usr/bin/env python3
"""
SimEnv 比赛部署脚本

完整流程：
1. 连接 ROS-Gazebo 环境
2. 加载训练好的导航策略
3. 系统化探索楼栋各层
4. 使用 RGB 检测危险源
5. 输出 results/detected_danger.json

用法:
  rosrun simenv_train deploy.py
  rosrun simenv_train deploy.py --output results/detected_danger.json --max-time 600
"""
import os
import sys
import time
import json
import argparse
import numpy as np

import rospy

# 添加包路径
_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from simenv_train.env import SimEnv
from simenv_train.exploration import RoomExplorer
from simenv_train.door_control import DoorController, ElevatorController
from simenv_train.config import cfg


def get_args():
    parser = argparse.ArgumentParser(description="SimEnv 比赛部署")
    parser.add_argument("--output", default="results/detected_danger.json",
                        help="结果输出路径")
    parser.add_argument("--max-time", type=float, default=600.0,
                        help="最大探索时间 (秒)")
    parser.add_argument("--no-nn", action="store_true",
                        help="不使用神经网络，纯规则策略")
    parser.add_argument("--model", type=str, default=None,
                        help="训练好的模型路径")
    return parser.parse_args()


def rule_based_policy(env: SimEnv, explorer: RoomExplorer, step: int) -> np.ndarray:
    """规则策略（无模型时的 fallback）

    行为：
    1. 朝最空旷方向走
    2. 接近障碍物时转向
    3. 检测到危险源时靠近确认
    """
    vx, vy, yaw_rate = explorer.find_best_direction()
    return np.array([[vx, vy, yaw_rate]])


def nn_policy(env: SimEnv, actor_critic, step: int) -> np.ndarray:
    """神经网络策略"""
    obs = env._build_observation()
    import torch
    obs_t = torch.from_numpy(obs).float().unsqueeze(0)
    with torch.no_grad():
        actions = actor_critic.act_inference(obs_t)
    return actions.cpu().numpy()


def deploy(args):
    print("=" * 60)
    print("SimEnv 比赛部署")
    print("=" * 60)
    print(f"  最大时间: {args.max_time}s")
    print(f"  输出文件: {args.output}")
    print(f"  模式: {'规则策略' if args.no_nn else '神经网络策略'}")
    print("=" * 60)

    # 初始化环境
    env = SimEnv(num_envs=1)
    explorer = RoomExplorer(env)

    # 加载神经网络策略（如果可用）
    actor_critic = None
    if not args.no_nn and args.model:
        try:
            import torch
            from simenv_train.models import make_actor_critic
            actor_critic = make_actor_critic()
            checkpoint = torch.load(args.model, map_location='cpu')
            actor_critic.load_state_dict(checkpoint['actor_critic'])
            actor_critic.eval()
            print(f"  模型已加载: {args.model}")
        except Exception as e:
            print(f"  [WARN] 模型加载失败，使用规则策略: {e}")
            args.no_nn = True

    # 初始化门/电梯控制
    try:
        doors = DoorController()
        print("  门控制服务就绪")
    except Exception:
        doors = None
        print("  [WARN] 门控制服务不可用")

    try:
        elevator = ElevatorController()
        print("  电梯控制服务就绪")
    except Exception:
        elevator = None
        print("  [WARN] 电梯控制服务不可用")

    # 读取场景信息
    scene_info = {}
    try:
        with open("generated_building/team_scene_info.json", "r") as f:
            scene_info = json.load(f)
        print(f"  场景: {scene_info.get('building_info', {}).get('floor_count', '?')} 层")
    except Exception:
        pass

    print("\n开始探索...\n")

    start_time = time.time()
    rate = rospy.Rate(20)  # 20 Hz

    while not rospy.is_shutdown():
        elapsed = time.time() - start_time
        if elapsed > args.max_time:
            break

        step = env.step_count

        # 选择动作
        if args.no_nn or actor_critic is None:
            actions = rule_based_policy(env, explorer, step)
        else:
            actions = nn_policy(env, actor_critic, step)

        # 执行一步
        _, _, dones, _ = env.step(actions)

        # 每 5 秒打印状态
        if step % 100 == 0:
            remaining = args.max_time - elapsed
            print(f"  [{elapsed:6.1f}s] step={step} "
                  f"danger_detected={len(env.detected_dangers)} "
                  f"remaining={remaining:.0f}s")

        rate.sleep()

    # 保存结果
    total_time = time.time() - start_time
    env.save_results(total_time, args.output)

    print(f"\n探索完成！耗时 {total_time:.1f}s")
    print(f"检测到 {len(env.detected_dangers)} 个危险源候选")
    print(f"结果: {args.output}")

    env.close()


if __name__ == '__main__':
    args = get_args()
    deploy(args)
