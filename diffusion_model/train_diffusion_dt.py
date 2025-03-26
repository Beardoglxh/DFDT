import torch
from torch import optim
from diffusion_model.diffusion_dt import DiffusionDynamics
from torch.utils.data import TensorDataset, DataLoader
import pickle
import numpy as np
from diffusion_model.data_processing import D4RLDiffusionDataset
from datetime import datetime


# 超参数
batch_size = 256
num_epochs = 3
num_diffusion_steps = 100
env_name = "halfcheetah-medium-expert-v2"
dataset_path = '../scripts/data/{}.pkl'.format(env_name)
context_len = 20
rtg_scale = 1000

# def normalize(data, mean, std):
#     return (data - mean) / (std + 1e-8)
#
# # 转换数据
#
# with open(dataset_path, 'rb') as f:
#     raw_data = pickle.load(f)  # 加载后的数据是包含多个字典的列表
#
# # 初始化存储容器
# states = []
# actions = []
# next_states = []
#
# # 处理每个轨迹片段
# for episode in raw_data:
#     # 提取数据片段 (假设每个字典的数组长度相同)
#     obs = episode['observations']  # shape (N, 17)
#     acts = episode['actions']  # shape (N, 6)
#     next_obs = episode['next_observations']  # shape (N, 17)
#     terminals = episode['terminals']  # shape (N,)
#
#     # 过滤终止状态的transition
#     valid_indices = np.where(terminals == False)[0]
#
#     # 收集有效数据
#     states.append(obs[valid_indices])
#     actions.append(acts[valid_indices])
#     next_states.append(next_obs[valid_indices])
#
# # 合并所有有效数据
# states = np.concatenate(states, axis=0)
# actions = np.concatenate(actions, axis=0)
# next_states = np.concatenate(next_states, axis=0)
#
# # 转换为PyTorch张量
# states = torch.tensor(states, dtype=torch.float32)
# actions = torch.tensor(actions, dtype=torch.float32)
# next_states = torch.tensor(next_states, dtype=torch.float32)
#
# # 计算统计量
# state_mean, state_std = states.mean(axis=0), states.std(axis=0)
# action_mean, action_std = actions.mean(axis=0), actions.std(axis=0)
#
# # 应用归一化
# states = normalize(states, state_mean, state_std)
# actions = normalize(actions, action_mean, action_std)
# next_states = normalize(next_states, state_mean, state_std)
#
# # 创建数据集和数据加载器
# dataset = TensorDataset(states, actions, next_states)
# dataloader = DataLoader(dataset, batch_size=256, shuffle=True)


# 初始化模型和优化器

dataset = D4RLDiffusionDataset(dataset_path, rtg_scale=rtg_scale)
dataloader = dataset.get_dataloader(batch_size=batch_size)

model = DiffusionDynamics(state_dim=dataset.state_dim, action_dim=dataset.action_dim)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 训练循环
for epoch in range(num_epochs):
    for s_t, a_t, s_next in dataloader:
        # 随机选择扩散步数
        diffusion_steps = torch.randint(0, num_diffusion_steps, (s_t.size(0),))

        # 正向扩散：添加噪声
        noisy_s_next, true_noise = model.add_noise(s_next, diffusion_steps)

        # 预测噪声
        pred_noise = model(s_t, a_t, noisy_s_next, diffusion_steps)

        # 计算损失
        loss = torch.mean((pred_noise - true_noise) ** 2)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

# 在训练代码最后添加保存逻辑
checkpoint = {
    'model_state': model.state_dict(),
    'state_mean': dataset.state_mean,  # 来自D4RLDiffusionDataset
    'state_std': dataset.state_std,
    'action_mean': dataset.action_mean,
    'action_std': dataset.action_std,
    'num_diffusion_steps': num_diffusion_steps  # 仅保存配置参数
}
current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
torch.save(checkpoint, f'./pretrained_model/diffusion_model-{env_name}-{current_time}.pth')
