import math, torch
import torch.nn as nn
import torch.nn.functional as F

class ConditionalDiffusion(nn.Module):
    def __init__(self, model, scheduler_type="linear", T=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.T = T

        if scheduler_type == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif scheduler_type == "cosine":
            betas = torch.cos(torch.linspace(0, math.pi/2, T))**2
            betas = (betas - betas.min())/(betas.max()-betas.min())
            betas = beta_start + (beta_end - beta_start) * betas
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")

        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        a_bar = self.alpha_bars[t].sqrt().view(-1,1,1,1)
        m_bar = (1 - self.alpha_bars[t]).sqrt().view(-1,1,1,1)
        return a_bar*x0 + m_bar*noise, noise

    def p_losses(self, x0, t, cond_iq, cond_rd):
        # x0: clean RD; cond_(iq|rd): cluttered maps
        x_noisy, noise = self.q_sample(x0, t)
        t_norm = t.float() / self.T
        # pass three tensors into your dual‑branch UNet
        noise_pred = self.model(x_noisy, cond_iq, cond_rd, t_norm)
        return F.mse_loss(noise_pred, noise)

    @torch.no_grad()
    def p_sample(self, x, t, cond_iq, cond_rd):
        t_int = int(t.item())
        β = self.betas[t_int].view(-1,1,1,1)
        α = self.alphas[t_int].view(-1,1,1,1)
        α_bar = self.alpha_bars[t_int].view(-1,1,1,1)
        t_norm = (torch.tensor([t_int], device=x.device).float() / self.T).repeat(x.shape[0])
        
        noise_pred = self.model(x, cond_iq, cond_rd, t_norm)
        coef1 = 1/torch.sqrt(α)
        coef2 = β/torch.sqrt(1-α_bar)
        mean = coef1 * (x - coef2 * noise_pred)
        noise = torch.randn_like(x) if t_int>0 else 0
        return mean + torch.sqrt(β)*noise

    @torch.no_grad()
    def sample(self, cond_iq, cond_rd, shape):
        x = torch.randn(shape, device=cond_iq.device)
        for ti in reversed(range(self.T)):
            t = torch.tensor([ti], device=x.device)
            x = self.p_sample(x, t, cond_iq, cond_rd)
        return x
