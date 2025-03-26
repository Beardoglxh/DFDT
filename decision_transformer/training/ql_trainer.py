import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import time
import copy
from tqdm import tqdm, trange
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb


class EMA():
    '''
        empirical moving average
    '''

    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Trainer:

    def __init__(self,
                 model,
                 dynamics,
                 critic,
                 batch_size,
                 tau,
                 discount,
                 get_batch,
                 loss_fn,
                 eval_fns=None,
                 max_q_backup=False,
                 eta=1.0,
                 eta2=1.0,
                 ema_decay=0.995,
                 step_start_ema=1000,
                 update_ema_every=5,
                 lr=3e-4,
                 weight_decay=1e-4,
                 lr_decay=False,
                 lr_maxt=100000,
                 lr_min=0.,
                 grad_norm=1.0,
                 scale=1.0,
                 k_rewards=True,
                 use_discount=True
                 ):

        self.actor = model
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=lr, weight_decay=weight_decay)

        self.dynamics = dynamics

        self.step_start_ema = step_start_ema
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.actor)
        self.update_ema_every = update_ema_every

        self.critic = critic
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        if lr_decay:
            self.actor_lr_scheduler = CosineAnnealingLR(self.actor_optimizer, T_max=lr_maxt, eta_min=lr_min)
            self.critic_lr_scheduler = CosineAnnealingLR(self.critic_optimizer, T_max=lr_maxt, eta_min=lr_min)

        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.tau = tau
        self.max_q_backup = max_q_backup
        self.discount = discount
        self.grad_norm = grad_norm
        self.eta = eta
        self.eta2 = eta2
        self.lr_decay = lr_decay
        self.scale = scale
        self.k_rewards = k_rewards
        self.use_discount = use_discount

        self.start_time = time.time()
        self.step = 0

    def step_ema(self):
        if self.step > self.step_start_ema and self.step % self.update_ema_every == 0:
            self.ema.update_model_average(self.ema_model, self.actor)

    def train_iteration(self, num_steps, logger, iter_num=0, log_writer=None, action_num=10, max_uncertainty=0.8, pre_critic=None):

        logs = dict()

        train_start = time.time()

        self.actor.train()
        self.critic.train()
        loss_metric = {
            'bc_loss': [],
            'ql_loss': [],
            'actor_loss': [],
            'critic_loss': [],
            'target_q_mean': [],
        }
        for _ in trange(num_steps):
            loss_metric, out_action_num = self.train_step(log_writer, loss_metric, action_num, max_uncertainty, pre_critic)

        if self.lr_decay:
            self.actor_lr_scheduler.step()
            self.critic_lr_scheduler.step()

        logger.record_tabular('BC Loss', np.mean(loss_metric['bc_loss']))
        logger.record_tabular('QL Loss', np.mean(loss_metric['ql_loss']))
        logger.record_tabular('Actor Loss', np.mean(loss_metric['actor_loss']))
        logger.record_tabular('Critic Loss', np.mean(loss_metric['critic_loss']))
        logger.record_tabular('Target Q Mean', np.mean(loss_metric['target_q_mean']))
        logger.dump_tabular()

        wandb.log({
            'BC Loss': np.mean(loss_metric['bc_loss']),
            'QL Loss': np.mean(loss_metric['ql_loss']),
            'Actor Loss': np.mean(loss_metric['actor_loss']),
            'Critic Loss': np.mean(loss_metric['critic_loss']),
            'Target Q Mean': np.mean(loss_metric['target_q_mean'])
        })

        logs['time/training'] = time.time() - train_start

        eval_start = time.time()

        self.actor.eval()
        self.critic.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.actor, self.critic_target, action_num, out_action_num, pre_critic)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        logger.log('=' * 80)
        logger.log(f'Iteration {iter_num}')
        best_ret = -10000
        best_nor_ret = -10000
        for k, v in logs.items():
            if 'return_mean' in k:
                best_ret = max(best_ret, float(v))
            if 'normalized_score' in k:
                best_nor_ret = max(best_nor_ret, float(v))
            logger.record_tabular(k, float(v))
        logger.record_tabular('Current actor learning rate', self.actor_optimizer.param_groups[0]['lr'])
        logger.record_tabular('Current critic learning rate', self.critic_optimizer.param_groups[0]['lr'])
        logger.dump_tabular()

        logs['Best_return_mean'] = best_ret
        logs['Best_normalized_score'] = best_nor_ret
        return logs

    def scale_up_eta(self, lambda_):
        self.eta2 = self.eta2 / lambda_

    def train_step(self, log_writer=None, loss_metric={}, action_num=10, max_uncertainty=0.8, pre_critic=None):
        '''
            Train the model for one step
            states: (batch_size, max_len, state_dim)
        '''
        # 这里的action和action_target值一模一样
        states, actions, rewards, action_target, dones, rtg, timesteps, attention_mask = self.get_batch(self.batch_size)
        # action_target = torch.clone(actions)
        batch_size = states.shape[0]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        device = states.device

        """MY CODE"""

        def expectile_loss(diff, expectile=0.8):  # diff是一个张量, 表示实际值与预测值之间的差异
            weight = torch.where(diff > 0, expectile, (1 - expectile))  # 如果差距大于0, 则分配expectile的权重, 反之分配1-expectile的权重
            return weight * (diff ** 2)  # 计算分配权重之后的MSE.

        # def compute_out_action_num(pre_states, true_states, max_action_num, max_uncertainty):
        #     uncertainty = expectile_loss((true_states - pre_states), expectile=0.8).mean()
        #     out_action_num = int(
        #         torch.clamp(1 + (max_action_num - 1) * (uncertainty / max_uncertainty), 1, max_action_num))
        #     return out_action_num

        def compute_out_action_num(uncertainty, max_action_num, max_uncertainty): # for diffusion model
            uncertainty = uncertainty.mean()
            out_action_num = int(
                torch.clamp(1 + (max_action_num - 1) * (uncertainty / max_uncertainty), 1, max_action_num))
            return out_action_num

        # dynamics
        states_flat = states.view(-1, states.shape[-1])
        actions_flat = actions.view(-1, actions.shape[-1])
        _, uncertainty = self.dynamics.predict_uncertainty(states_flat, actions_flat)

        # next_states, coef_penalty, info = self.dynamics.step(states,
        #                                                     actions)
        # next_states = torch.from_numpy(next_states).to(states)
        # coef_penalty = torch.from_numpy(coef_penalty).to(rewards)
        # rewards = rewards - coef_penalty
        out_action_num = compute_out_action_num(uncertainty, action_num, max_uncertainty)
        states_expanded = states.unsqueeze(2)
        states_repeated = states_expanded.repeat(1, 1, out_action_num, 1)
        attention_mask_expanded = attention_mask.unsqueeze(2)
        attention_mask_repeated = attention_mask_expanded.expand(-1, -1, out_action_num)
        action_target_expanded = action_target.unsqueeze(2)
        action_target_repeated = action_target_expanded.expand(-1, -1, out_action_num, -1)

        '''Q Training'''
        # target_q1, target_q2 = self.critic_target(states, next_action)  # [B, T, 1]
        # target_q = torch.min(target_q1, target_q2)  # [B, T, 1]
        # target_q = rewards[:, :-1] + self.discount * target_q[:, 1:]
        # target_q = torch.cat([target_q, torch.zeros(batch_size, 1, 1).to(device)], dim=1)
        #
        # critic_loss = F.mse_loss(current_q1[:, :-1][attention_mask[:, :-1] > 0],
        #                          target_q[:, :-1][attention_mask[:, :-1] > 0]) \
        #               + F.mse_loss(current_q2[:, :-1][attention_mask[:, :-1] > 0],
        #                            target_q[:, :-1][attention_mask[:, :-1] > 0])

        if pre_critic:
            critic_loss = torch.tensor(0.0, device=device)  # 使用浮点数张量
            target_q = torch.tensor(0.0, device=device)
        else:
            current_q1, current_q2 = self.critic.forward(states, actions)
            current_v = self.critic.forward(states)  # [B, T, 1]

            next_states, next_actions, _ = self.ema_model(
                states, actions, rewards, action_target, rtg[:, :-1], timesteps, attention_mask=attention_mask,
                action_num=action_num, out_action_num=out_action_num
            )
            # next_action = next_actions[:, :, 0, :]

            # 计算V网络损失（关键修改点）
            with torch.no_grad():
                q_min = torch.min(current_q1, current_q2)  # 保守Q估计
            v_loss = torch.mean(expectile_loss(q_min - current_v, expectile=0.7))

            # 计算Q网络损失（基于V网络）
            with torch.no_grad():
                next_v = self.critic_target(next_states)  # 使用V网络而非Q网络
                next_v = next_v * (1 - dones)

            target_q = rewards[:, :-1] + self.discount * next_v[:, 1:]
            target_q = torch.cat([target_q, torch.zeros(batch_size, 1, 1).to(device)], dim=1)  # 对齐维度

            q1_loss = F.mse_loss(current_q1[:, :-1][attention_mask[:, :-1] > 0], target_q[:, :-1][attention_mask[:, :-1] > 0])
            q2_loss = F.mse_loss(current_q2[:, :-1][attention_mask[:, :-1] > 0], target_q[:, :-1][attention_mask[:, :-1] > 0])

            critic_loss = q1_loss + q2_loss + v_loss

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            if self.grad_norm > 0:
                critic_grad_norms = nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=self.grad_norm, norm_type=2)
            self.critic_optimizer.step()

        '''Policy Training'''
        state_preds, action_preds, reward_preds = self.actor.forward(
            states, actions, rewards, action_target, rtg[:, :-1], timesteps, attention_mask=attention_mask,
            action_num=action_num, out_action_num=out_action_num
        )

        """MYCODE"""
        if pre_critic:
            q1_scores, q2_scores = pre_critic.both(states_repeated, action_preds)
        else:
            q1_scores, q2_scores = self.critic.forward(states_repeated, action_preds)
        q_score = torch.min(q1_scores, q2_scores)
        probabilities = torch.softmax(q_score, dim=2)

        action_preds_ = action_preds.reshape(-1, action_dim)[attention_mask_repeated.reshape(-1) > 0]
        action_target_ = action_target_repeated.reshape(-1, action_dim)[attention_mask_repeated.reshape(-1) > 0]
        probabilities = probabilities.reshape(-1, 1)[attention_mask_repeated.reshape(-1, ) > 0]
        state_preds = state_preds[:, :-1]
        state_target = states[:, 1:]
        states_loss = ((state_preds - state_target) ** 2)[attention_mask[:, :-1] > 0].mean()
        if reward_preds is not None:  # 默认是None.
            reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
            reward_target = rewards.reshape(-1, 1)[attention_mask.reshape(-1) > 0] / self.scale
            rewards_loss = F.mse_loss(reward_preds, reward_target)
        else:
            rewards_loss = 0
        """MY CODE"""

        bc_loss = torch.mean(
            F.mse_loss(action_preds_, action_target_, reduction='none') * probabilities) + states_loss + rewards_loss

        actor_states = states_repeated.reshape(-1, state_dim)[attention_mask_repeated.reshape(-1) > 0]
        if pre_critic:
            q1_new_action, q2_new_action = pre_critic.both(actor_states, action_preds_)
        else:
            q1_new_action, q2_new_action = self.critic(actor_states, action_preds_)
        if pre_critic:
            q_loss = torch.tensor(0.0, device=device)
        else:
            if np.random.uniform() > 0.5:
                q_loss = - q1_new_action.mean() / q2_new_action.abs().mean().detach()
            else:
                q_loss = - q2_new_action.mean() / q1_new_action.abs().mean().detach()
        actor_loss = self.eta2 * bc_loss + self.eta * q_loss
        # actor_loss = self.eta * bc_loss + q_loss

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        if self.grad_norm > 0:
            actor_grad_norms = nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=self.grad_norm, norm_type=2)
        self.actor_optimizer.step()

        """ Step Target network """
        self.step_ema()

        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        self.step += 1

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean(
                (action_preds - action_target_repeated) ** 2).detach().cpu().item()

        if log_writer is not None:
            if self.grad_norm > 0:
                log_writer.add_scalar('Actor Grad Norm', actor_grad_norms.max().item(), self.step)
                log_writer.add_scalar('Critic Grad Norm', critic_grad_norms.max().item(), self.step)
            log_writer.add_scalar('BC Loss', bc_loss.item(), self.step)
            log_writer.add_scalar('QL Loss', q_loss.item(), self.step)
            log_writer.add_scalar('Critic Loss', critic_loss.item(), self.step)
            log_writer.add_scalar('Target_Q Mean', target_q.mean().item(), self.step)

        loss_metric['bc_loss'].append(bc_loss.item())
        loss_metric['ql_loss'].append(q_loss.item())
        loss_metric['critic_loss'].append(critic_loss.item())
        loss_metric['actor_loss'].append(actor_loss.item())
        loss_metric['target_q_mean'].append(target_q.mean().item())

        return loss_metric, out_action_num
