"""
PPO（近端策略优化）算法
从 rsl_rl 的 PPO + OnPolicyRunner 移植

关键特性：PPO-clip / GAE / KL 自适应学习率 / CBF 干预损失 / Lipschitz 平滑
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.distributions import Normal
import numpy as np
from collections import deque


class RolloutBuffer:
    def __init__(self, num_envs: int, num_steps: int, obs_dim: int, act_dim: int, device: str):
        self.device = device
        self.num_envs = num_envs
        self.num_steps = num_steps

        self.observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.next_observations = torch.zeros(num_steps, num_envs, obs_dim, device=device)
        self.actions = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.actions_log_prob = torch.zeros(num_steps, num_envs, device=device)
        self.action_means = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.action_stds = torch.zeros(num_steps, num_envs, act_dim, device=device)
        self.rewards = torch.zeros(num_steps, num_envs, device=device)
        self.values = torch.zeros(num_steps, num_envs, device=device)
        self.dones = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=device)
        self.bad_masks = torch.ones(num_steps, num_envs, dtype=torch.bool, device=device)
        self.time_outs = torch.zeros(num_steps, num_envs, dtype=torch.bool, device=device)
        self.step = 0

    def insert(self, obs, next_obs, actions, log_prob, mean, std,
               rewards, values, dones, bad_masks=None, time_outs=None):
        idx = self.step
        self.observations[idx] = obs
        self.next_observations[idx] = next_obs
        self.actions[idx] = actions
        self.actions_log_prob[idx] = log_prob
        self.action_means[idx] = mean
        self.action_stds[idx] = std
        self.rewards[idx] = rewards
        self.values[idx] = values
        self.dones[idx] = dones
        if bad_masks is not None:
            self.bad_masks[idx] = bad_masks
        if time_outs is not None:
            self.time_outs[idx] = time_outs
        self.step += 1

    def compute_returns(self, last_values: torch.Tensor, gamma: float, lam: float):
        advantages = torch.zeros_like(self.rewards)
        last_gae = 0.0
        for t in reversed(range(self.num_steps)):
            if t == self.num_steps - 1:
                next_values = last_values
                next_nonterminal = 1.0 - self.dones[t].float()
            else:
                next_values = self.values[t + 1]
                next_nonterminal = 1.0 - self.dones[t].float()

            delta = self.rewards[t] + gamma * next_values * next_nonterminal - self.values[t]
            last_gae = delta + gamma * lam * next_nonterminal * last_gae
            advantages[t] = last_gae

        returns = advantages + self.values
        return advantages, returns

    def clear(self):
        self.step = 0

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int):
        batch_size = self.num_envs * self.num_steps
        mini_batch_size = batch_size // num_mini_batches

        obs_flat = self.observations.view(batch_size, -1)
        next_obs_flat = self.next_observations.view(batch_size, -1)
        actions_flat = self.actions.view(batch_size, -1)
        log_prob_flat = self.actions_log_prob.view(batch_size)
        mean_flat = self.action_means.view(batch_size, -1)
        std_flat = self.action_stds.view(batch_size, -1)
        bad_masks_flat = self.bad_masks.view(batch_size)

        for _ in range(num_epochs):
            perm = torch.randperm(batch_size, device=self.device)
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = start + mini_batch_size
                inds = perm[start:end]
                yield (obs_flat[inds], next_obs_flat[inds],
                       actions_flat[inds], log_prob_flat[inds],
                       mean_flat[inds], std_flat[inds],
                       bad_masks_flat[inds], inds)


class PPO:
    def __init__(self, actor_critic: nn.Module, config, device: str):
        self.device = device
        self.actor_critic = actor_critic.to(device)
        self.cfg = config

        self.optimizer = optim.Adam(self.actor_critic.parameters(),
                                    lr=self.cfg.learning_rate, eps=1e-5)
        self.learning_rate = self.cfg.learning_rate
        self.schedule = self.cfg.schedule
        self.desired_kl = self.cfg.desired_kl
        self.buffer = None
        self.episode_infos = deque(maxlen=100)
        self.current_learning_iteration = 0

    def init_storage(self, num_envs: int, obs_dim: int, act_dim: int):
        self.buffer = RolloutBuffer(
            num_envs=num_envs, num_steps=self.cfg.num_steps_per_env,
            obs_dim=obs_dim, act_dim=act_dim, device=self.device,
        )

    def act(self, obs: torch.Tensor):
        obs = obs.to(self.device)
        with torch.no_grad():
            self.actor_critic.update_distribution(obs)
            actions = self.actor_critic.distribution.sample()
            log_prob = self.actor_critic.get_actions_log_prob(actions)
            values = self.actor_critic.evaluate(obs).squeeze(-1)
        return actions, log_prob, values

    def process_env_step(self, obs, next_obs, actions, log_prob, mean, std,
                         rewards, dones, infos, values):
        time_outs = infos.get('time_outs', None)
        if time_outs is not None and isinstance(time_outs, np.ndarray):
            time_outs = torch.from_numpy(time_outs).bool().to(self.device)
        bad_masks = infos.get('bad_masks', None)
        self.buffer.insert(
            obs.to(self.device), next_obs.to(self.device),
            actions.to(self.device), log_prob.to(self.device),
            mean.to(self.device), std.to(self.device),
            rewards.to(self.device), values.to(self.device), dones.to(self.device),
            bad_masks=bad_masks, time_outs=time_outs,
        )

    def update(self, last_obs: torch.Tensor) -> dict:
        last_obs = last_obs.to(self.device)
        with torch.no_grad():
            last_values = self.actor_critic.evaluate(last_obs).squeeze(-1)
            advantages, returns = self.buffer.compute_returns(
                last_values, self.cfg.gamma, self.cfg.lam
            )

        batch_size = self.buffer.num_envs * self.buffer.num_steps
        adv_flat = advantages.view(batch_size)
        ret_flat = returns.view(batch_size)
        val_flat = self.buffer.values.view(batch_size)
        adv_flat = (adv_flat - adv_flat.mean()) / (adv_flat.std() + 1e-8)

        mean_losses = {
            'value_loss': 0.0, 'surrogate_loss': 0.0, 'entropy': 0.0,
            'reg_loss': 0.0, 'smooth_loss': 0.0, 'interv_loss': 0.0,
        }
        num_updates = self.cfg.num_learning_epochs * self.cfg.num_mini_batches

        for epoch in range(self.cfg.num_learning_epochs):
            for batch_data in self.buffer.mini_batch_generator(
                    self.cfg.num_mini_batches, 1):
                (obs_b, next_obs_b, actions_b, old_log_prob_b,
                 old_mean_b, old_std_b, bad_mask_b, inds) = batch_data

                valid_mask = (~bad_mask_b.bool()).float()
                valid_sum = valid_mask.sum() + 1e-8

                self.actor_critic.update_distribution(obs_b)
                new_log_prob = self.actor_critic.get_actions_log_prob(actions_b)
                values = self.actor_critic.evaluate(obs_b).squeeze(-1)
                entropy = self.actor_critic.entropy
                new_mean = self.actor_critic.action_mean
                new_std = self.actor_critic.action_std

                # KL 自适应学习率
                if self.desired_kl is not None and self.schedule == 'adaptive':
                    with torch.no_grad():
                        kl = torch.sum(
                            torch.log(new_std / (old_std_b + 1e-5))
                            + (old_std_b ** 2 + (old_mean_b - new_mean) ** 2)
                            / (2.0 * new_std ** 2) - 0.5, dim=-1)
                        kl_mean = (kl * valid_mask).sum() / valid_sum
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.learning_rate

                # PPO-clip: 用正确的索引取 advantage
                adv_b = adv_flat[inds]
                ratio = torch.exp(new_log_prob - old_log_prob_b)
                surrogate = -adv_b * ratio
                surrogate_clipped = -adv_b * torch.clamp(
                    ratio, 1.0 - self.cfg.clip_param, 1.0 + self.cfg.clip_param)
                surrogate_loss = (torch.max(surrogate, surrogate_clipped) * valid_mask).sum() / valid_sum

                # Value loss
                ret_b = ret_flat[inds]
                val_b = val_flat[inds]
                if self.cfg.use_clipped_value_loss:
                    value_clipped = val_b + \
                        (values - val_b).clamp(-self.cfg.clip_param, self.cfg.clip_param)
                    value_losses = (values - ret_b).pow(2)
                    value_losses_clipped = (value_clipped - ret_b).pow(2)
                    value_loss = (torch.max(value_losses, value_losses_clipped)
                                  * valid_mask).sum() / valid_sum
                else:
                    value_loss = ((values - ret_b).pow(2) * valid_mask).sum() / valid_sum

                entropy_loss = (entropy * valid_mask).sum() / valid_sum

                # 动作范围正则化
                clip_mins = torch.tensor([-0.5, -0.8, -1.0], device=self.device)
                clip_maxs = torch.tensor([1.7, 0.8, 1.0], device=self.device)
                range_loss = (
                    torch.sum((new_mean - torch.clip(new_mean, min=clip_mins, max=clip_maxs)) ** 2, dim=-1)
                    * valid_mask
                ).sum() / valid_sum

                # Lipschitz 平滑
                smooth_loss = self._compute_smoothness_loss(obs_b, next_obs_b, valid_mask, valid_sum)
                reg_loss = range_loss + 0.05 * smooth_loss

                # CBF 干预损失
                if hasattr(self.actor_critic, 'u_bar') and self.actor_critic.u_bar is not None:
                    u_bar = self.actor_critic.u_bar
                    u_s = self.actor_critic.u_s
                    interv_loss = torch.mean(torch.sum((u_s - u_bar) ** 2, dim=-1))
                else:
                    interv_loss = torch.tensor(0.0)

                # 总损失
                loss = (
                    surrogate_loss
                    + self.cfg.value_loss_coef * value_loss
                    - self.cfg.entropy_coef * entropy_loss
                    + 1.0 * reg_loss
                    + 0.1 * interv_loss
                )

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.actor_critic.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                mean_losses['value_loss'] += value_loss.item()
                mean_losses['surrogate_loss'] += surrogate_loss.item()
                mean_losses['entropy'] += entropy_loss.item()
                mean_losses['reg_loss'] += reg_loss.item()
                mean_losses['smooth_loss'] += smooth_loss.item()
                mean_losses['interv_loss'] += interv_loss.item()

        for k in mean_losses:
            mean_losses[k] /= num_updates

        self.buffer.clear()
        self.current_learning_iteration += 1
        return mean_losses

    def _compute_smoothness_loss(self, obs, next_obs, valid_mask, valid_sum):
        batch_size = obs.size(0)
        _u = torch.rand(batch_size, 1, device=self.device)
        mix_weights = (_u - 0.5) * 2.0
        delta_states = next_obs - obs
        interp_states = obs + mix_weights * delta_states

        self.actor_critic.update_distribution(obs)
        orig_actions = self.actor_critic.action_mean
        self.actor_critic.update_distribution(interp_states)
        interp_actions = self.actor_critic.action_mean
        actor_smoothness = nn.functional.mse_loss(interp_actions, orig_actions, reduction='none')
        actor_smoothness = (actor_smoothness.mean(dim=-1) * valid_mask).sum() / valid_sum

        orig_values = self.actor_critic.evaluate(obs).squeeze(-1)
        interp_values = self.actor_critic.evaluate(interp_states).squeeze(-1)
        critic_smoothness = ((orig_values - interp_values).pow(2) * valid_mask).sum() / valid_sum

        return 1.0 * actor_smoothness + 0.1 * critic_smoothness

    def save(self, path: str):
        torch.save({
            'model_state_dict': self.actor_critic.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'iteration': self.current_learning_iteration,
        }, path)
        print(f"  [保存] 模型 → {path}")

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.actor_critic.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.current_learning_iteration = checkpoint['iteration']
        print(f"  [加载] 模型 ← {path} (迭代 {self.current_learning_iteration})")

    def get_inference_policy(self):
        def policy_fn(obs):
            with torch.no_grad():
                return self.actor_critic.act_inference(obs)
        return policy_fn

    def train_mode(self):
        self.actor_critic.train()

    def eval_mode(self):
        self.actor_critic.eval()
