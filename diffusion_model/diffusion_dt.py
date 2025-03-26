import torch
import torch.nn as nn
from torch import optim
# from einops import rearrange


def cosine_beta_schedule(steps, s=0.008):
    x = torch.linspace(0, steps, steps + 1)
    alphas_cumprod = torch.cos((x / steps + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class ConditionalNoisePredictor(nn.Module):
    def __init__(self, state_dim, action_dim, time_dim=64, hidden_dim=512):
        super().__init__()
        # 初始投影层：将state_dim映射到hidden_dim
        self.input_proj = nn.Linear(state_dim, hidden_dim)

        # 条件投影层：state_dim + action_dim + time_dim → hidden_dim
        self.cond_proj = nn.Linear(state_dim + action_dim + time_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Mish()
            ) for _ in range(4)
        ])
        self.output = nn.Linear(hidden_dim, state_dim)

    def forward(self, noisy_s_next, condition):
        # 步骤1：将输入投影到hidden_dim
        x = self.input_proj(noisy_s_next)  # [batch, state_dim] → [batch, hidden_dim]

        # 步骤2：初始条件注入
        x = x + self.cond_proj(condition)  # 现在维度一致：都是[batch, hidden_dim]

        # 步骤3：分层处理
        for block in self.blocks:
            x = block(x) + self.cond_proj(condition)  # 保持维度一致

        return self.output(x)  # 映射回state_dim

class DiffusionDynamics(nn.Module):
    def __init__(self, state_dim=17, action_dim=6, num_diffusion_steps=100):
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.num_diffusion_steps = num_diffusion_steps

        # 时间编码器
        self.time_enc = nn.Sequential(
            nn.Linear(1, 64),
            nn.Mish(),
            nn.Linear(64, 64)
        )

        # 条件噪声预测器
        self.noise_predictor = ConditionalNoisePredictor(state_dim, action_dim)

        # 余弦扩散调度
        betas = cosine_beta_schedule(num_diffusion_steps)
        self.register_buffer('betas', betas)
        self.register_buffer('alphas', 1. - betas)  # 新增注册
        self.register_buffer('alpha_bars', torch.cumprod(1. - betas, dim=0))  # 新增注册

    def forward(self, s_t, a_t, noisy_s_next, diffusion_step):
        # 时间编码 [batch_size, 64]
        t = diffusion_step.float() / self.num_diffusion_steps
        time_emb = self.time_enc(t.unsqueeze(-1))

        # 拼接条件 [batch_size, state_dim + action_dim + 64]
        condition = torch.cat([s_t, a_t, time_emb], dim=-1)

        # 分层条件预测
        return self.noise_predictor(noisy_s_next, condition)

    def add_noise(self, s_next, diffusion_step):
        alpha_bar = self.alpha_bars[diffusion_step].unsqueeze(-1)
        noise = torch.randn_like(s_next)
        noisy_s_next = alpha_bar.sqrt() * s_next + (1 - alpha_bar).sqrt() * noise
        return noisy_s_next, noise

    def sample(self, s_t, a_t, num_samples=1):
        """
        反向去噪生成 s_{t+1} 样本
        输入:
            s_t: [batch_size, state_dim]
            a_t: [batch_size, action_dim]
            num_samples: 每个 (s_t, a_t) 生成多少样本
        输出:
            samples: [batch_size, num_samples, state_dim]
        """
        batch_size = s_t.shape[0]
        device = s_t.device

        # 扩展输入以生成多个样本
        s_t = s_t.repeat_interleave(num_samples, dim=0)  # [batch*num_samples, state_dim]
        a_t = a_t.repeat_interleave(num_samples, dim=0)  # [batch*num_samples, action_dim]

        # 初始化为随机噪声
        x = torch.randn(batch_size * num_samples, self.state_dim).to(device)

        # 逐步去噪
        for step in reversed(range(self.num_diffusion_steps)):
            diffusion_steps = torch.full((batch_size * num_samples,), step, device=device)
            predicted_noise = self.forward(s_t, a_t, x, diffusion_steps)

            # 计算当前步的 alpha 和 beta
            alpha = self.alphas[step]
            alpha_bar = self.alpha_bars[step]
            beta = self.betas[step]

            # 更新 x（反向过程的一步）
            if step > 0:
                noise = torch.randn_like(x)
            else:
                noise = 0

            x = (x - (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise) / torch.sqrt(alpha)
            x = x + torch.sqrt(beta) * noise

        # 恢复形状 [batch, num_samples, state_dim]
        x = x.view(batch_size, num_samples, self.state_dim)
        return x


