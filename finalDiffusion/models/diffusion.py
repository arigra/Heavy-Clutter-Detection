import math
import torch
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
            betas = torch.cos(torch.linspace(0, math.pi / 2, T)) ** 2
            betas = (betas - betas.min()) / (betas.max() - betas.min())
            betas = beta_start + (beta_end - beta_start) * betas
        else:
            raise ValueError(f"Unknown scheduler type: {scheduler_type}")
        
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", 1.0 - betas)
        self.register_buffer("alpha_bars", torch.cumprod(self.alphas, dim=0))

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alphas_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_bar = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alphas_bar * x0 + sqrt_one_minus_alphas_bar * noise, noise
    
    def p_losses(self, x0, t, cond):
        x_noisy, noise = self.q_sample(x0, t)
        t_norm = t.float() / self.T
        model_input = torch.cat([x_noisy, cond], dim=1)
        noise_pred = self.model(model_input, t_norm)
        return F.mse_loss(noise_pred, noise)
    
    @torch.no_grad()
    def p_sample(self, x, t, cond):
        t_int = t.item() if isinstance(t, torch.Tensor) else t
        betas_t = self.betas[t_int].view(-1, 1, 1, 1)
        alphas_t = self.alphas[t_int].view(-1, 1, 1, 1)
        alpha_bars_t = self.alpha_bars[t_int].view(-1, 1, 1, 1)
        t_norm = (torch.tensor([t_int], device=x.device).float() / self.T).repeat(x.shape[0])

        model_input = torch.cat([x, cond], dim=1)
        noise_pred = self.model(model_input, t_norm)

        coef1 = 1 / torch.sqrt(alphas_t)
        coef2 = betas_t / torch.sqrt(1-alpha_bars_t)
        mean = coef1 * (x - coef2 * noise_pred)
        noise=torch.randn_like(x) if t_int > 0 else 0
        return mean + torch.sqrt(betas_t) * noise
    
    @torch.no_grad()
    def sample(self, cond, shape):
        x = torch.randn(shape, device=cond.device)
        for t in reversed(range(self.T)):
            t_tensor = torch.tensor([t], device=x.device)
            x = self.p_sample(x, t_tensor, cond)
        return x

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import StudentT

class StudentTDiffusion(ConditionalDiffusion):
    def __init__(self,
                 model: nn.Module,
                 nu: float = 5.0,
                 gamma: float = 0.5,
                 scheduler_type: str = "linear",
                 T: int = 1000,
                 beta_start: float = 1e-4,
                 beta_end: float = 0.02):
        """
        ν: degrees of freedom for Student‐T prior
        γ: divergence parameter (γ>0)
        All other args identical to ConditionalDiffusion.
        """
        super().__init__(model, scheduler_type, T, beta_start, beta_end)
        self.nu = nu
        self.gamma = gamma

        # scale so that StudentT(df=ν, scale=scale) has unit variance:
        # Var = ν * scale^2 / (ν – 2) = 1  ⇒  scale = sqrt((ν–2)/ν)
        scale = math.sqrt((self.nu - 2) / self.nu)
        # we’ll reuse this distribution for sampling
        self.base_dist = StudentT(df=self.nu, loc=0.0, scale=scale)

    def q_sample(self, x0, t, noise=None):
        # Draw heavy‐tailed noise instead of Gaussian
        if noise is None:
            noise = self.base_dist.sample(x0.shape).to(x0.device)
        # same scheduling as before
        a_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
        one_minus = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
        return a_bar * x0 + one_minus * noise, noise

    def p_losses(self, x0, t, cond):
        # forward q: get noisy x and true Student‑T noise
        x_noisy, eps_true = self.q_sample(x0, t)
        t_norm = t.float() / self.T
        # predict the injected noise
        eps_pred = self.model(torch.cat([x_noisy, cond], dim=1), t_norm)

        # γ‑divergence weighting:
        # let σ_t = sqrt(1 - ᾱ_t)
        sigma_t = (1 - self.alpha_bars[t]).sqrt().view(-1,1,1,1)
        # normalized squared residual
        u = (eps_pred - eps_true).pow(2) / (self.nu * sigma_t.pow(2))
        # weight = (1 + u)^(-γ - 1)
        w = (1 + u).pow(- (self.gamma + 1))
        # final loss
        return (w * (eps_pred - eps_true).pow(2)).mean()

    @torch.no_grad()
    def p_sample(self, x, t, cond):
        # exactly as in your base, but draw Student‑T noise
        t_int = int(t.item() if isinstance(t, torch.Tensor) else t)
        β_t = self.betas[t_int].view(-1,1,1,1)
        α_t = self.alphas[t_int].view(-1,1,1,1)
        ᾱ_t = self.alpha_bars[t_int].view(-1,1,1,1)
        t_norm = (torch.tensor([t_int], device=x.device).float() / self.T).repeat(x.shape[0])

        # predict the noise
        noise_pred = self.model(torch.cat([x, cond], dim=1), t_norm)

        # standard DDPM update
        coef1 = 1 / α_t.sqrt()
        coef2 = β_t / (1 - ᾱ_t).sqrt()
        mean = coef1 * (x - coef2 * noise_pred)

        # sample heavy‑tailed noise (zero at t=0)
        noise = (self.base_dist.sample(x.shape).to(x.device) * β_t.sqrt()
                 if t_int > 0 else torch.zeros_like(x))
        return mean + noise

    # .sample(...) can remain unchanged
    @torch.no_grad()
    def sample(self, cond, shape):
        x = torch.randn(shape, device=cond.device)
        for t in reversed(range(self.T)):
            t_tensor = torch.tensor([t], device=x.device)
            x = self.p_sample(x, t_tensor, cond)
        return x
    
# heavy_tail_gamma_diffusion.py
import math, torch, torch.nn as nn, torch.nn.functional as F
from torch.distributions import StudentT, Gamma

# ---------- helpers ----------------------------------------------------------
def _student_t_noise(shape, nu, device):
    """ε ~ StudentT(ν) implemented via the Gaussian scale‑mixture trick."""
    gamma = Gamma(nu / 2, nu / 2).sample(shape).to(device)     # λ ~ Inv‑Gamma
    z     = torch.randn(shape, device=device)                  # z ~ N(0,1)
    return z / torch.sqrt(gamma)                               # ε = z / √λ

def _gamma_noise(shape, k, theta, device):
    """ε ~ Gamma(k,θ) but zero‑center it for symmetric residuals."""
    g = Gamma(k, 1 / theta).sample(shape).to(device)           # mean = kθ
    return g - (k * theta)                                     # zero mean

def _sample_noise(shape, device, noise_type, **kwargs):
    if noise_type == "gaussian":
        return torch.randn(shape, device=device)
    if noise_type == "studentt":
        return _student_t_noise(shape, kwargs.get("nu", 5.), device)
    if noise_type == "gamma":
        return _gamma_noise(shape,
                            kwargs.get("k", 2.0),
                            kwargs.get("theta", 1.0),
                            device)
    raise ValueError(f"Unknown noise_type {noise_type}")

def gamma_divergence(pred, target, gamma=0.3, eps=1e-8):
    """Γ‑divergence loss (γ∈(0,2); γ→1 gives MSE)."""
    diff2 = (pred - target).pow(2) + eps
    return (diff2.pow(gamma / 2)).mean()
# -----------------------------------------------------------------------------


class HTGDiffusion(nn.Module):
    """
    Heavy‑Tailed + Gamma Diffusion (HTG)
    Usage is identical to your original ConditionalDiffusion except for the
    extra constructor arguments that pick the noise family and its params.

    Parameters
    ----------
    model : nn.Module
        Your ε‑predictor (typically a U‑Net) that receives cat([x_t , cond], 1)
    noise_type : {"gaussian","studentt","gamma"}
    nu : float
        ν for Student‑t (ignored otherwise)
    k, theta : float
        shape / scale for Gamma (ignored otherwise)
    loss_type : {"mse","gamma_div"}
    gamma_param : float
        γ in γ‑divergence (ignored for mse)
    """
    def __init__(self, model,
                 T=1000,
                 scheduler_type="linear",
                 beta_start=1e-4, beta_end=0.02,
                 noise_type="studentt",
                 nu=5.0, k=2.0, theta=1.0,
                 loss_type="gamma_div", gamma_param=0.3):
        super().__init__()
        self.model, self.T = model, T
        self.noise_type, self.nu, self.k, self.theta = noise_type, nu, k, theta
        self.loss_type, self.gamma_param = loss_type, gamma_param

        # standard β schedule (unchanged – works for Student‑t/Gamma too)
        if scheduler_type == "linear":
            betas = torch.linspace(beta_start, beta_end, T)
        elif scheduler_type == "cosine":
            betas = torch.cos(torch.linspace(0, math.pi/2, T))**2
            betas = (betas - betas.min()) / (betas.max() - betas.min())
            betas = beta_start + (beta_end - beta_start) * betas
        else:
            raise ValueError(f"Unknown scheduler {scheduler_type}")

        self.register_buffer("betas",        betas)
        self.register_buffer("alphas",       1.0 - betas)
        self.register_buffer("alpha_bars",   torch.cumprod(1.0 - betas, dim=0))

    # --------------------------------------------------------------------- q_t
    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = _sample_noise(x0.shape, x0.device,
                                  self.noise_type, nu=self.nu,
                                  k=self.k, theta=self.theta)

        sqrt_ab   = self.alpha_bars[t].sqrt().view(-1,1,1,1)
        sqrt_omab = (1 - self.alpha_bars[t]).sqrt().view(-1,1,1,1)
        return sqrt_ab * x0 + sqrt_omab * noise, noise

    # ------------------------------------------------------------------ loss
    def p_losses(self, x0, t, cond):
        x_t, noise = self.q_sample(x0, t)
        t_norm     = t.float() / self.T
        noise_pred = self.model(torch.cat([x_t, cond], dim=1), t_norm)

        if self.loss_type == "mse":
            return F.mse_loss(noise_pred, noise)
        return gamma_divergence(noise_pred, noise, self.gamma_param)

    # --------------------------------------------------------------- p(x_{t-1})
    @torch.no_grad()
    def p_sample(self, x_t, t, cond):
        t_int   = int(t.item()) if isinstance(t, torch.Tensor) else int(t)
        betas_t = self.betas[t_int].view(1,1,1,1)
        alphas_t, ab_t = self.alphas[t_int], self.alpha_bars[t_int]
        t_norm = torch.full((x_t.size(0),), t_int / self.T, device=x_t.device)
        noise_pred = self.model(torch.cat([x_t, cond], 1), t_norm)

        # same closed‑form as EDM/t‑EDM (good empirical fit for Student‑t)
        coef1 = 1 / torch.sqrt(alphas_t)
        coef2 = betas_t / torch.sqrt(1 - ab_t)
        mean  = coef1 * (x_t - coef2 * noise_pred)

        if t_int > 0:
            noise = _sample_noise(x_t.shape, x_t.device,
                                  self.noise_type, nu=self.nu,
                                  k=self.k, theta=self.theta)
            return mean + torch.sqrt(betas_t) * noise
        return mean

    # ---------------------------------------------------------------- sample
    @torch.no_grad()
    def sample(self, cond, shape):
        x = torch.randn(shape, device=cond.device)              # start noise
        for t in reversed(range(self.T)):
            x = self.p_sample(x, torch.tensor([t], device=x.device), cond)
        return x
