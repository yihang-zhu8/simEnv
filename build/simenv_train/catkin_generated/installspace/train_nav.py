#!/usr/bin/env python3
"""
SimEnv 导航策略训练

两步训练：
1. 2D 快速预训练（--fast）：学习避障和导航到目标
2. ROS-Gazebo 精调：在真实仿真中适配

用法:
  # 2D 快速预训练
  rosrun simenv_train train_nav.py --fast --max_iterations 2000

  # ROS-Gazebo 精调（需先启动 auto.sh + junior_ctrl 进入 RL 模式）
  rosrun simenv_train train_nav.py --max_iterations 500
"""
import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

import torch

_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from simenv_train.config import cfg


def get_args():
    parser = argparse.ArgumentParser(description="SimEnv 导航训练")
    parser.add_argument("--fast", action="store_true",
                        help="2D 快速预训练（不依赖 ROS）")
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume", type=str, default=None,
                        help="从检查点恢复")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_envs", type=int, default=1024,
                        help="并行环境数（2D 模式）")
    return parser.parse_args()


def train_2d_fast(args):
    """2D 快速预训练"""
    from simenv_cbf_train.simple_env import SimpleNavEnv
    from simenv_cbf_train.ppo import PPO
    from simenv_cbf_train.models import make_actor_critic

    print("2D 快速预训练模式")
    env = SimpleNavEnv(num_envs=args.num_envs, seed=args.seed)

    actor_critic = make_actor_critic(with_cbf=True)
    ppo = PPO(actor_critic, cfg.ppo, args.device)

    from simenv_cbf_train.config import cfg as old_cfg
    obs_dim = old_cfg.env.num_observations  # (num_props + num_rays + num_goal_obs) * his_len = 540
    act_dim = cfg.env.num_actions
    ppo.init_storage(args.num_envs, obs_dim, act_dim)

    if args.resume:
        ppo.load(args.resume)

    max_iter = args.max_iterations or 2000
    num_steps = cfg.ppo.num_steps_per_env

    model_dir = os.path.join(cfg.model_dir, "2d_pretrain")
    os.makedirs(model_dir, exist_ok=True)

    obs = env.reset()
    start_time = time.time()

    for iteration in range(max_iter):
        ppo.train_mode()
        for step in range(num_steps):
            actions, log_probs, values = ppo.act(obs)
            next_obs, rewards, dones, infos = env.step(actions.cpu().numpy())

            rewards_t = torch.from_numpy(rewards).float().to(args.device)
            dones_t = torch.from_numpy(dones).bool().to(args.device)

            ppo.process_env_step(
                obs, next_obs, actions, log_probs,
                ppo.actor_critic.action_mean, ppo.actor_critic.action_std,
                rewards_t, dones_t, infos, values,
            )
            obs = next_obs

        train_info = ppo.update(obs)

        if iteration % 50 == 0:
            elapsed = time.time() - start_time
            print(f"[{iteration:4d}/{max_iter}] "
                  f"loss={train_info.get('surrogate_loss', 0):.3f} "
                  f"val={train_info.get('value_loss', 0):.3f} "
                  f"t={elapsed:.0f}s")

        if iteration % 200 == 0:
            ppo.save(os.path.join(model_dir, f"model_{iteration}.pt"))

    final = os.path.join(model_dir, "policy_2d.pt")
    ppo.save(final)
    print(f"2D 预训练完成: {final}")
    return final


def train_ros(args):
    """ROS-Gazebo 精调"""
    from simenv_train.env import SimEnv

    print("ROS-Gazebo 训练模式")
    print("确保: auto.sh 已启动, junior_ctrl 已于 RL 模式")
    env = SimEnv(num_envs=1)

    from simenv_cbf_train.ppo import PPO
    from simenv_cbf_train.models import make_actor_critic

    actor_critic = make_actor_critic(with_cbf=True)
    ppo = PPO(actor_critic, cfg.ppo, args.device)
    ppo.init_storage(1, cfg.env.num_obs, cfg.env.num_actions)

    if args.resume:
        ppo.load(args.resume)
        print(f"从 {args.resume} 恢复训练")

    max_iter = args.max_iterations or 500
    num_steps = cfg.ppo.num_steps_per_env

    timestamp = datetime.now().strftime("%m%d_%H%M")
    model_dir = os.path.join(cfg.model_dir, f"ros_finetune_{timestamp}")
    os.makedirs(model_dir, exist_ok=True)

    obs = env.reset()
    start_time = time.time()

    for iteration in range(max_iter):
        ppo.train_mode()
        for step in range(num_steps):
            actions, log_probs, values = ppo.act(obs)
            next_obs, rewards, dones, infos = env.step(actions.cpu().numpy())

            rewards_t = torch.from_numpy(rewards).float().to(args.device)
            dones_t = torch.from_numpy(dones).bool().to(args.device)

            ppo.process_env_step(
                obs, next_obs, actions, log_probs,
                ppo.actor_critic.action_mean, ppo.actor_critic.action_std,
                rewards_t, dones_t, infos, values,
            )
            obs = next_obs

        train_info = ppo.update(obs)

        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            print(f"[{iteration:4d}/{max_iter}] "
                  f"loss={train_info.get('surrogate_loss', 0):.3f} "
                  f"dangers={len(env.detected_dangers)} "
                  f"t={elapsed:.0f}s")

        if iteration % 100 == 0:
            ppo.save(os.path.join(model_dir, f"model_{iteration}.pt"))

    final = os.path.join(model_dir, "policy_final.pt")
    ppo.save(final)
    env.close()
    print(f"ROS 精调完成: {final}")
    return final


if __name__ == '__main__':
    args = get_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if args.max_iterations is not None:
        cfg.ppo.max_iterations = args.max_iterations
    cfg.device = args.device

    if args.fast:
        train_2d_fast(args)
    else:
        train_ros(args)
