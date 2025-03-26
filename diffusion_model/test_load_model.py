import torch
from diffusion_model.load_diffusion_dt_model import DiffusionWrapper

# 加载模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
diffusion_model = DiffusionWrapper('diffusion_model-walker2d-medium-v2-20250316-174140.pth', device=device)

# 准备输入数据（假设来自新采集的数据）
raw_s_t = torch.randn(32, 17)  # 原始状态
raw_a_t = torch.randn(32, 6)   # 原始动作

# 预测不确定性
next_states, uncertainties = diffusion_model.predict_uncertainty(
    raw_s_t.to(device),
    raw_a_t.to(device),
    num_samples=100
)

print(f"预测下一状态形状: {next_states.shape}")
print(f"不确定性度量形状: {uncertainties.shape}")
print(f"平均不确定性: {uncertainties.mean().item():.4f}")
