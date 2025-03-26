import pickle
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

datapath = "../scripts/data/walker2d-medium-replay-v2.pkl"


class D4RLDiffusionDataset:
    def __init__(self, dataset_path, rtg_scale=1000):
        with open(dataset_path, 'rb') as f:
            self.trajectories = pickle.load(f)

        # 计算全局归一化统计量
        states = []
        actions = []
        for traj in self.trajectories:
            states.append(traj['observations'])
            actions.append(traj['actions'])

        # 保存原始维度（用于后续分析）
        self.raw_state_dim = states[0].shape[-1]  # 原始状态维度
        self.raw_action_dim = actions[0].shape[-1]  # 原始动作维度

        self.state_mean = np.concatenate(states, axis=0).mean(axis=0)
        self.state_std = np.concatenate(states, axis=0).std(axis=0) + 1e-6
        self.action_mean = np.concatenate(actions, axis=0).mean(axis=0)
        self.action_std = np.concatenate(actions, axis=0).std(axis=0) + 1e-6

        # 归一化处理并收集有效transition
        self.states = []
        self.actions = []
        self.next_states = []

        for traj in self.trajectories:
            # 应用归一化
            norm_obs = (traj['observations'] - self.state_mean) / self.state_std
            norm_acts = (traj['actions'] - self.action_mean) / self.action_std
            norm_next_obs = (traj['next_observations'] - self.state_mean) / self.state_std

            # 过滤终止状态
            valid_mask = ~traj['terminals']

            self.states.append(norm_obs[valid_mask])
            self.actions.append(norm_acts[valid_mask])
            self.next_states.append(norm_next_obs[valid_mask])

        # 合并数据
        self.states = torch.tensor(np.concatenate(self.states), dtype=torch.float32)
        self.actions = torch.tensor(np.concatenate(self.actions), dtype=torch.float32)
        self.next_states = torch.tensor(np.concatenate(self.next_states), dtype=torch.float32)

        # 保存归一化后的维度（实际输入模型的维度）
        self.state_dim = self.states.shape[-1]  # 归一化后状态维度
        self.action_dim = self.actions.shape[-1]  # 归一化后动作维度

    def get_dataloader(self, batch_size=256):
        dataset = TensorDataset(self.states, self.actions, self.next_states)
        return DataLoader(dataset, batch_size=batch_size, shuffle=True)
