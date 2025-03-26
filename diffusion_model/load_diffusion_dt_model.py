import torch
from diffusion_model.diffusion_dt import DiffusionDynamics

class DiffusionWrapper:
    def __init__(self, checkpoint_path, device='cuda'):
        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # 初始化模型
        self.model = DiffusionDynamics(
            state_dim=checkpoint['state_mean'].shape[0],
            action_dim=checkpoint['action_mean'].shape[0],
            num_diffusion_steps=checkpoint['num_diffusion_steps']
        )
        self.model.load_state_dict(checkpoint['model_state'])
        self.model.to(device)
        self.model.eval()  # 切换到评估模式

        # 存储归一化参数
        self.state_mean = torch.tensor(checkpoint['state_mean'], device=device)
        self.state_std = torch.tensor(checkpoint['state_std'], device=device)
        self.action_mean = torch.tensor(checkpoint['action_mean'], device=device)
        self.action_std = torch.tensor(checkpoint['action_std'], device=device)

    def normalize(self, data, mean, std):
        """数据归一化"""
        return (data - mean) / std

    def denormalize(self, data, mean, std):
        """数据反归一化"""
        return data * std + mean

    def predict_uncertainty(self, s_t, a_t, num_samples=100):
        with torch.no_grad():
            # 归一化输入
            s_t_norm = self.normalize(s_t, self.state_mean, self.state_std)
            a_t_norm = self.normalize(a_t, self.action_mean, self.action_std)

            # 重复采样
            s_t_repeat = s_t_norm.repeat(num_samples, 1)  # [batch*num_samples, state_dim]
            a_t_repeat = a_t_norm.repeat(num_samples, 1)  # [batch*num_samples, action_dim]

            # 随机采样扩散步数
            t = torch.randint(0, self.model.num_diffusion_steps,
                              (s_t_repeat.size(0),), device=s_t.device)

            # 关键修复：调整alpha_bars的维度
            alpha_bars_t = self.model.alpha_bars[t].unsqueeze(-1)  # [batch*num_samples, 1]

            # 添加噪声并去噪
            noisy_s_next = torch.randn_like(s_t_repeat)
            pred_noise = self.model(s_t_repeat, a_t_repeat, noisy_s_next, t)

            # 维度匹配的计算
            denoised = (noisy_s_next - pred_noise * torch.sqrt(1 - alpha_bars_t)) / torch.sqrt(alpha_bars_t)

        # 反归一化
        denoised = self.denormalize(denoised, self.state_mean, self.state_std)

        # 计算统计量
        denoised = denoised.view(num_samples, -1, self.model.state_dim)
        next_state_pred = denoised.mean(dim=0)
        uncertainty = denoised.std(dim=0).mean(dim=-1)

        return next_state_pred, uncertainty

