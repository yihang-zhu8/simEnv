#!/usr/bin/env python3
"""
SimEnv CBF-PPO 训练入口
从 SEA-Nav-Code 的 train.py 移植，集成到 SimEnv ROS-Gazebo 环境

训练流程（与 SEA-Nav 一致）:
  1. 解析配置
  2. 创建环境（默认 ROS-Gazebo，--fast 切换 2D NumPy）
  3. 创建 PPO + CBF Agent
  4. 训练循环（collect → update → repeat）
  5. 保存模型（.pt，可被 junior_ctrl 的 LibTorch 加载）

用法:
  # 默认：SimEnv ROS-Gazebo 训练（需先启动 auto.sh）
  rosrun simenv_cbf_train train.py

  # 用 roslaunch 启动（可从参数服务器读取配置）
  roslaunch simenv_cbf_train train.launch max_iterations:=500

  # 快速 2D 预训练（不依赖 ROS/Gazebo）
  rosrun simenv_cbf_train train.py --fast --num_envs 1024 --max_iterations 2000

  # 从检查点恢复（SimEnv 精调）
  rosrun simenv_cbf_train train.py --resume --checkpoint ./models/model_2000.pt

  # 禁用 CBF（消融实验）
  rosrun simenv_cbf_train train.py --no_cbf
"""
import os
import sys
import time
import argparse
import numpy as np
from datetime import datetime

import torch

# 确保 ROS package 路径正确
_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from simenv_cbf_train.config import cfg
from simenv_cbf_train.ppo import PPO
from simenv_cbf_train.models import make_actor_critic


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="SimEnv CBF-PPO Training（默认 ROS-Gazebo 模式）"
    )

    # ── 环境模式 ──
    parser.add_argument("--fast", action="store_true", default=False,
                        help="使用快速 2D NumPy 环境（默认使用 ROS-Gazebo）")
    parser.add_argument("--num_envs", type=int, default=None,
                        help="并行环境数（覆盖配置）")
    parser.add_argument("--max_iterations", type=int, default=None,
                        help="最大迭代次数（覆盖配置）")

    # ── 模型 ──
    parser.add_argument("--no_cbf", action="store_true", default=False,
                        help="不使用 CBF 安全层（使用标准 ActorCritic）")
    parser.add_argument("--resume", action="store_true", default=False,
                        help="从检查点恢复训练")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="检查点路径")
    parser.add_argument("--experiment_name", type=str, default=None,
                        help="实验名称（覆盖配置）")

    # ── 设备 ──
    parser.add_argument("--device", type=str, default="cpu",
                        help="训练设备（cpu / cuda:0）")

    # ── 种子 ──
    parser.add_argument("--seed", type=int, default=None,
                        help="随机种子")

    return parser.parse_args()


def _read_ros_params(args: argparse.Namespace):
    """从 ROS 参数服务器读取配置（roslaunch 时使用）"""
    try:
        import rospy
        if rospy.get_node_uri():
            if rospy.has_param('~max_iterations'):
                args.max_iterations = rospy.get_param('~max_iterations')
            if rospy.has_param('~num_envs'):
                args.num_envs = rospy.get_param('~num_envs')
            if rospy.has_param('~fast_mode'):
                args.fast = rospy.get_param('~fast_mode')
            if rospy.has_param('~resume'):
                args.resume = rospy.get_param('~resume')
            if rospy.has_param('~checkpoint') and rospy.get_param('~checkpoint'):
                args.checkpoint = rospy.get_param('~checkpoint')
            if rospy.has_param('~no_cbf'):
                args.no_cbf = rospy.get_param('~no_cbf')
            if rospy.has_param('~experiment_name'):
                args.experiment_name = rospy.get_param('~experiment_name')
    except Exception:
        pass


def make_env(args) -> tuple:
    """创建训练环境（默认 ROS-Gazebo）"""
    if args.fast:
        # 快速 2D 环境
        from simenv_cbf_train.simple_env import SimpleNavEnv
        num_envs = args.num_envs or cfg.env.num_envs
        env = SimpleNavEnv(num_envs=num_envs, seed=args.seed or cfg.seed)
        print(f"  [环境] 快速 2D Nav（{env.get_num_envs()} 个环境）")
    else:
        # 默认 ROS-Gazebo 环境
        from simenv_cbf_train.simenv_env import SimEnvROSEnv
        env = SimEnvROSEnv(num_envs=args.num_envs or 1)
        print(f"  [环境] ROS-Gazebo SimEnv（{env.get_num_envs()} 个环境）")
        print(f"  提示: 确保已运行 auto.sh，且 junior_ctrl 已进入 RL 模式")

    return env, None


def make_agent(env, args) -> PPO:
    """创建 PPO Agent"""
    actor_critic = make_actor_critic(with_cbf=not args.no_cbf)
    ppo = PPO(actor_critic, cfg.ppo, cfg.device)
    obs_dim = cfg.env.num_observations
    act_dim = cfg.env.num_actions
    ppo.init_storage(env.get_num_envs(), obs_dim, act_dim)

    if args.resume and args.checkpoint:
        ppo.load(args.checkpoint)

    return ppo


def learn(env, agent: PPO, args) -> str:
    """训练主循环"""
    max_iter = args.max_iterations or cfg.ppo.max_iterations
    num_steps = cfg.ppo.num_steps_per_env
    save_interval = cfg.ppo.save_interval
    num_envs = env.get_num_envs()

    exp_name = args.experiment_name or cfg.ppo.experiment_name
    timestamp = datetime.now().strftime("%m_%d_%H-%M-%S")
    log_dir = os.path.join(cfg.log_dir, exp_name, timestamp)
    model_dir = os.path.join(cfg.model_dir, exp_name)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)

    start_time = time.time()

    print(f"\n{'='*60}")
    print(f"开始训练: {exp_name}")
    print(f"最大迭代: {max_iter}")
    print(f"每环境步数: {num_steps}")
    print(f"环境数: {num_envs}")
    print(f"设备: {cfg.device}")
    print(f"模型目录: {model_dir}")
    print(f"日志目录: {log_dir}")
    print(f"{'='*60}\n")

    obs = env.get_observations()

    for iteration in range(max_iter):
        agent.train_mode()

        # ── 收集经验 ──
        for step in range(num_steps):
            actions, log_probs, values = agent.act(obs)

            # 环境 step
            next_obs_np, rewards_np, dones_np, infos = env.step(actions.cpu().numpy())

            rewards = torch.from_numpy(rewards_np).float().to(cfg.device)
            dones = torch.from_numpy(dones_np).bool().to(cfg.device)
            next_obs = next_obs_np if isinstance(next_obs_np, torch.Tensor) else \
                torch.from_numpy(next_obs_np).float().to(cfg.device)

            agent.process_env_step(
                obs, next_obs, actions, log_probs,
                agent.actor_critic.action_mean,
                agent.actor_critic.action_std,
                rewards, dones, infos, values,
            )

            obs = next_obs

        # ── PPO 更新 ──
        train_info = agent.update(obs)

        # ── 日志 ──
        if iteration % 10 == 0:
            elapsed = time.time() - start_time
            print(
                f"[{iteration:4d}/{max_iter}]  "
                f"surr={train_info['surrogate_loss']:.3f}  "
                f"val={train_info['value_loss']:.3f}  "
                f"ent={train_info['entropy']:.3f}  "
                f"reg={train_info['reg_loss']:.3f}  "
                f"lr={agent.learning_rate:.2e}  "
                f"t={elapsed:.0f}s"
            )

        # ── 保存模型 ──
        if iteration % save_interval == 0 or iteration == max_iter - 1:
            model_path = os.path.join(model_dir, f"model_{iteration}.pt")
            agent.save(model_path)

    final_path = os.path.join(model_dir, "policy_final.pt")
    agent.save(final_path)

    total_time = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"训练完成!")
    print(f"总耗时: {total_time:.0f}s ({total_time / 60:.1f}min)")
    print(f"最终模型: {final_path}")
    print(f"{'='*60}")

    return final_path


def train(args):
    """训练主函数"""
    print_config()

    env, env_cfg = make_env(args)
    agent = make_agent(env, args)
    model_path = learn(env, agent, args)

    print(f"\n训练完毕！模型保存至: {model_path}")
    print(f"可在 SimEnv 中运行 deploy.py 部署测试")
    print(f"或: rosrun simenv_cbf_train deploy.py --model {model_path}")


def print_config():
    print('=' * 70)
    print('SimEnv CBF-PPO Navigation Training (ROS-Gazebo)')
    print('=' * 70)
    print(f"  观测: {cfg.env.num_observations} 维 ({cfg.env.num_obs_one_step} × {cfg.env.his_len} 步历史)")
    print(f"  动作: {cfg.env.num_actions} 维 (vx, vy, yaw_rate)")
    print(f"  射线: {cfg.cbf.num_rays} 条, FOV={cfg.cbf.fov_deg}°")
    print(f"  CBF 安全距离: {cfg.cbf.d_safe:.2f}m, κ={cfg.cbf.kappa:.1f}")
    print(f"  PPO 迭代: {cfg.ppo.max_iterations}")
    print(f"  PPO 学习率: {cfg.ppo.learning_rate}")
    print(f"  设备: {cfg.device}")
    print('=' * 70)


if __name__ == '__main__':
    args = get_args()

    # 尝试从 ROS 参数服务器读取（roslaunch 场景）
    _read_ros_params(args)

    # 覆盖配置
    if args.num_envs is not None:
        cfg.env.num_envs = args.num_envs
    if args.max_iterations is not None:
        cfg.ppo.max_iterations = args.max_iterations
    if args.device is not None:
        cfg.device = args.device
    if args.seed is not None:
        cfg.seed = args.seed

    # 随机种子
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # 设备检查
    if cfg.device.startswith("cuda") and not torch.cuda.is_available():
        print("[WARN] CUDA 不可用，回退到 CPU")
        cfg.device = "cpu"

    train(args)
