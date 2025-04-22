import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def generate_range_steering_matrix(N=64, dR=64, B=50e6, c=3e8):
    rng_res = c / (2 * B)
    r_vals = torch.arange(dR) * rng_res
    n_vals = torch.arange(N)
    phase = -1j * 2 * torch.pi * (2 * B) / (c * N)
    R = torch.exp(phase * torch.outer(n_vals, r_vals))
    return R

def generate_doppler_steering_matrix(K=64, dV=64, fc=9.39e9, T0=1e-3, c=3e8):
    vel_res = c / (2 * fc * K * T0)
    v_vals = torch.linspace(-dV // 2, dV // 2, dV) * vel_res
    k_vals = torch.arange(K)
    phase = -1j * 2 * torch.pi * (2 * fc * T0) / c
    V = torch.exp(phase * torch.outer(k_vals, v_vals))
    return V

def create_rd_map_differentiable(IQ_map):
    if not torch.is_tensor(IQ_map):
        IQ_map = torch.from_numpy(IQ_map)
    if not torch.is_complex(IQ_map):
        IQ_map = IQ_map.to(torch.complex64)
    dev = IQ_map.device
    R = generate_range_steering_matrix().to(dev)
    V = generate_doppler_steering_matrix().to(dev)
    RD_map = torch.abs(R.T.conj() @ IQ_map @ V.conj())
    return RD_map

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


    def combined_loss(self, x0, t, cond,
                  use_standard_loss=True, use_rd_loss=True,
                  rd_threshold=100, rd_temperature=0.1, rd_loss_scale=100.0):
        """
        Computes a combined loss that can include:
        1. A standard MSE loss on noise prediction (if use_standard_loss is True)
        2. A soft RD loss in the Range-Doppler domain (if use_rd_loss is True)

        Parameters:
        x0             : Clean (ground truth) IQ map, shape (B,2,H,W)
        t              : Time indices for diffusion steps, shape (B,)
        cond           : Conditioning IQ map, shape (B,2,H,W)
        use_standard_loss : Boolean flag to enable the standard MSE loss.
        use_rd_loss    : Boolean flag to enable the RD soft loss.
        rd_threshold   : Threshold value for target presence in RD map.
        rd_temperature : Temperature parameter controlling the softness of the threshold.
        rd_loss_scale  : Scaling factor for balancing the RD loss.
        
        Returns:
        total_loss     : The sum of the enabled loss terms.
        """
        # Compute the shared components: noisy input and predicted noise.
        x_noisy, noise = self.q_sample(x0, t)
        t_norm = t.float() / self.T
        model_input = torch.cat([x_noisy, cond], dim=1)
        noise_pred = self.model(model_input, t_norm)

        # Initialize total loss.
        total_loss = 0.0
    
        # Standard MSE loss on noise prediction.
        if use_standard_loss:
            mse_loss = F.mse_loss(noise_pred, noise)
            total_loss += mse_loss

        # Soft RD loss.
        if use_rd_loss:
            # Estimate the denoised IQ map.
            sqrt_alphas_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
            sqrt_one_minus_alphas_bar = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
            x0_pred = (x_noisy - sqrt_one_minus_alphas_bar * noise_pred) / sqrt_alphas_bar

            # Compute the differentiable RD maps.
            rd_pred = create_rd_map_differentiable(x0_pred)
            rd_clean = create_rd_map_differentiable(x0)

            # Generate soft masks using a sigmoid activation.
            soft_mask_pred = torch.sigmoid((rd_pred - rd_threshold) / rd_temperature)
            soft_mask_clean = torch.sigmoid((rd_clean - rd_threshold) / rd_temperature)

            # Compute the loss between the soft masks (MSE or binary cross entropy can be used).
            rd_loss = F.mse_loss(soft_mask_pred, soft_mask_clean) * rd_loss_scale

            total_loss += rd_loss

        return total_loss


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







    # def rd_losses(self, x0, t, cond, threshold=100, rd_loss_scale=100.0):
    #     # 1. Add noise and predict noise
    #     x_noisy, noise = self.q_sample(x0, t)
    #     t_norm = t.float() / self.T
    #     model_input = torch.cat([x_noisy, cond], dim=1)
    #     noise_pred = self.model(model_input, t_norm)
        
    #     # 2. Estimate the clean map from the noisy one
    #     sqrt_alphas_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
    #     sqrt_one_minus_alphas_bar = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
    #     x0_pred = (x_noisy - sqrt_one_minus_alphas_bar * noise_pred) / sqrt_alphas_bar
        
    #     # 3. Compute differentiable RD maps for predicted and clean IQ maps
    #     rd_pred = create_rd_map_differentiable(x0_pred)
    #     rd_clean = create_rd_map_differentiable(x0)
        
    #     # 4. Create binary masks for target presence. Here a pixel is considered to have a target
    #     #    if its intensity in the RD map exceeds the threshold.
    #     #    Note: Using a hard threshold gives a non-differentiable indicator.
    #     mask_pred = (rd_pred > threshold).float()
    #     mask_clean = (rd_clean > threshold).float()
        
    #     # 5. Determine deletion: if a pixel was supposed to have a target (mask_clean==1)
    #     #    but the denoised map did not have it (mask_pred==0), then it counts as deletion.
    #     #    We create a deletion mask indicating these errors.
    #     deletion_mask = torch.clamp(mask_clean - mask_pred, min=0, max=1)
        
    #     # 6. Decide loss value: Here we assume that if any of the target pixels are "deleted",
    #     #    then the loss for that sample should be 1; else 0.
    #     #    This is computed per sample in the batch.
    #     batch_deletion = (deletion_mask.view(deletion_mask.shape[0], -1).sum(dim=1) > 0).float()
        
    #     # 7. Use the average of the binary indicators as the loss.
    #     rd_loss = batch_deletion.mean() * rd_loss_scale
    #     return rd_loss


    # def rd_soft_loss(self, x0, t, cond, threshold=100, temperature=0.1, rd_loss_scale=100.0):
    #     """
    #     Computes a soft, differentiable loss in the Range-Doppler domain.
        
    #     Parameters:
    #     x0           : Clean (ground truth) IQ map, shape (B,2,H,W)
    #     t            : Time indices for diffusion steps, shape (B,)
    #     cond         : Conditioning IQ map, shape (B,2,H,W)
    #     diffusion    : Instance of ConditionalDiffusion (provides q_sample, alpha_bars, etc.)
    #     threshold    : Intensity threshold to indicate target presence.
    #     temperature  : Controls softness of the threshold function.
    #     rd_loss_scale: Scaling factor to balance this loss with other loss terms.
    #     Returns:
    #     soft_rd_loss : A differentiable loss encouraging correct target representation in RD space.
    #     """
    #     # 1. Add noise and predict
    #     x_noisy, noise = self.q_sample(x0, t)
    #     t_norm = t.float() / self.T
    #     model_input = torch.cat([x_noisy, cond], dim=1)
    #     noise_pred = self.model(model_input, t_norm)
        
    #     # 2. Estimate the denoised IQ map
    #     sqrt_alphas_bar = self.alpha_bars[t].sqrt().view(-1, 1, 1, 1)
    #     sqrt_one_minus_alphas_bar = (1 - self.alpha_bars[t]).sqrt().view(-1, 1, 1, 1)
    #     x0_pred = (x_noisy - sqrt_one_minus_alphas_bar * noise_pred) / sqrt_alphas_bar
        
    #     # 3. Compute the differentiable RD maps
    #     rd_pred = create_rd_map_differentiable(x0_pred)
    #     rd_clean = create_rd_map_differentiable(x0)
        
    #     # 4. Generate soft masks using a sigmoid
    #     soft_mask_pred = torch.sigmoid((rd_pred - threshold) / temperature)
    #     soft_mask_clean = torch.sigmoid((rd_clean - threshold) / temperature)
        
    #     # 5. Compute the loss between soft masks (you can use MSE loss or binary cross entropy)
    #     # Using MSE loss for demonstration; BCE can be used as an alternative.
    #     loss_soft = F.mse_loss(soft_mask_pred, soft_mask_clean)
    #     return loss_soft * rd_loss_scale
