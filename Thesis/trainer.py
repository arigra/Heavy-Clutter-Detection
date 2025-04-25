import math
import torch
import torch.nn.functional as F

def train_one_epoch(diffusion, dataloader, optimizer, device, use_standard_loss=True, use_rd_loss=False):
    diffusion.train()
    epoch_loss = 0
    e_rd_loss = 0
    e_mse_loss = 0
    for i, batch in enumerate(dataloader):
        #signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = batch
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs = batch
        IQ = RDs_norm
        signal = rd_signals_norm
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

        loss = diffusion.p_losses(x0, t, cond)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
    return epoch_loss / len(dataloader)

@torch.no_grad()
def validate(diffusion, dataloader, device, use_standard_loss=True, use_rd_loss=False):
    diffusion.eval()
    total_val_loss = 0
    iq_val_loss = 0
    rd_val_loss = 0
    gen_mse, gen_psnr = None, None

    for i, batch in enumerate(dataloader):
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs = batch
        IQ = RDs_norm
        signal = rd_signals_norm
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
        iq_loss = diffusion.p_losses(x0, t, cond)
        total_val_loss += iq_loss.item()

        # For the first batch, generate a sample and compute metrics.
        if i == 0:
            generated = diffusion.sample(cond, x0.shape)
            mse_val = F.mse_loss(generated, x0).item()
            psnr_val = 20 * math.log10(x0.max().item() / math.sqrt(mse_val)) if mse_val > 0 else 100
            gen_mse, gen_psnr = mse_val, psnr_val

    avg_val_loss = total_val_loss / len(dataloader)
    return avg_val_loss, gen_mse, gen_psnr