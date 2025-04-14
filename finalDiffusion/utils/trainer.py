import math
import torch
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


def train_one_epoch(diffusion, dataloader, optimizer, device, use_standard_loss=True, use_rd_loss=False):
    diffusion.train()
    epoch_loss = 0
    e_rd_loss = 0
    e_mse_loss = 0
    for i, batch in enumerate(dataloader):
        signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = batch

        if signal.real.ndim == 3:
            x0_real = signal.real.unsqueeze(1)
            x0_imag = signal.imag.unsqueeze(1)
            cond_real = IQ.real.unsqueeze(1)
            cond_imag = IQ.imag.unsqueeze(1)
        else:
            x0_real = signal.real
            x0_imag = signal.imag
            cond_real = IQ.real
            cond_imag = IQ.imag

        # Concatenate to form 2-channel tensors.
        x0 = torch.cat([x0_real, x0_imag], dim=1).to(device)   # (B,2,H,W)
        cond = torch.cat([cond_real, cond_imag], dim=1).to(device)  # (B,2,H,W)

        # Sample random timesteps for diffusion.
        t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device).long()
        
        # Compute loss.
        #rd_fac = 1e-9
        #rd_loss = diffusion.rd_losses(x0, t, cond)
        # rd_loss = diffusion.rd_soft_loss(x0, t, cond, threshold=100, temperature=0.1, rd_loss_scale=0.5)
        loss = diffusion.p_losses(x0, t, cond)
        # e_rd_loss += rd_loss.item()
        # e_mse_loss += mse_loss.item()
        # loss = mse_loss + rd_loss #* rd_fac
        #loss = diffusion.p_losses_rd_label(x0, t, cond, rd_label)
#        loss = diffusion.combined_loss(x0, t, cond, use_standard_loss, use_rd_loss)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)#, e_mse_loss/ len(dataloader), e_rd_loss/ len(dataloader)


@torch.no_grad()
def validate(diffusion, dataloader, device, use_standard_loss=True, use_rd_loss=False):
    diffusion.eval()
    total_val_loss = 0
    iq_val_loss = 0
    rd_val_loss = 0
    gen_mse, gen_psnr = None, None

    for i, batch in enumerate(dataloader):
        signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = batch
        
        if signal.real.ndim == 3:
            x0_real = signal.real.unsqueeze(1)
            x0_imag = signal.imag.unsqueeze(1)
            cond_real = IQ.real.unsqueeze(1)
            cond_imag = IQ.imag.unsqueeze(1)
        else:
            x0_real = signal.real
            x0_imag = signal.imag
            cond_real = IQ.real
            cond_imag = IQ.imag

        x0 = torch.cat([x0_real, x0_imag], dim=1).to(device)  # (B,2,H,W)
        cond = torch.cat([cond_real, cond_imag], dim=1).to(device)  # (B,2,H,W)
        
        t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device).long()
        #rd_loss = diffusion.rd_losses(x0, t, cond, threshold=100, rd_loss_scale=10.0)
        # rd_loss = diffusion.rd_soft_loss(x0, t, cond, threshold=0.5, temperature=0.1, rd_loss_scale=0.5)
        iq_loss = diffusion.p_losses(x0, t, cond)
        # loss = rd_loss+iq_loss
        # iq_val_loss += iq_loss.item()
        # rd_val_loss += rd_loss.item()
        #loss = diffusion.combined_loss(x0, t, cond, use_standard_loss, use_rd_loss)
        total_val_loss += iq_loss.item()

        # For the first batch, generate a sample and compute metrics.
        if i == 0:
            generated = diffusion.sample(cond, x0.shape)
            mse_val = F.mse_loss(generated, x0).item()
            #rd_mse = F.mse_loss(create_rd_map_differentiable(generated), create_rd_map_differentiable(x0)).item()
            psnr_val = 20 * math.log10(x0.max().item() / math.sqrt(mse_val)) if mse_val > 0 else 100
            gen_mse, gen_psnr = mse_val, psnr_val

    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss, gen_mse, gen_psnr#, rd_mse#, iq_val_loss/ len(dataloader), rd_val_loss/ len(dataloader),