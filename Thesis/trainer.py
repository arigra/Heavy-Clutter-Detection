import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# def train_one_epoch(diffusion, dataloader, optimizer, device, use_standard_loss=True, use_rd_loss=False):
#     diffusion.train()
#     epoch_loss = 0
#     e_rd_loss = 0
#     e_mse_loss = 0
#     for i, batch in enumerate(dataloader):
#         #signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = batch
#         signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs = batch
#         IQ = RDs_norm
#         signal = rd_signals_norm
#         if signal.real.ndim == 3:
#             x0_real = signal.real.unsqueeze(1)
#             x0_imag = signal.imag.unsqueeze(1)
#             cond_real = IQ.real.unsqueeze(1)
#             cond_imag = IQ.imag.unsqueeze(1)
#         else:
#             x0_real = signal.real
#             x0_imag = signal.imag
#             cond_real = IQ.real
#             cond_imag = IQ.imag

#         # Concatenate to form 2-channel tensors.
#         x0 = torch.cat([x0_real, x0_imag], dim=1).to(device)   # (B,2,H,W)
#         cond = torch.cat([cond_real, cond_imag], dim=1).to(device)  # (B,2,H,W)

#         # Sample random timesteps for diffusion.
#         t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device).long()

#         loss = diffusion.p_losses(x0, t, cond)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         epoch_loss += loss.item()
        
#     return epoch_loss / len(dataloader)

# @torch.no_grad()
# def validate(diffusion, dataloader, device, use_standard_loss=True, use_rd_loss=False):
#     diffusion.eval()
#     total_val_loss = 0
#     iq_val_loss = 0
#     rd_val_loss = 0
#     gen_mse, gen_psnr = None, None

#     for i, batch in enumerate(dataloader):
#         signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs = batch
#         IQ = RDs_norm
#         signal = rd_signals_norm
#         if signal.real.ndim == 3:
#             x0_real = signal.real.unsqueeze(1)
#             x0_imag = signal.imag.unsqueeze(1)
#             cond_real = IQ.real.unsqueeze(1)
#             cond_imag = IQ.imag.unsqueeze(1)
#         else:
#             x0_real = signal.real
#             x0_imag = signal.imag
#             cond_real = IQ.real
#             cond_imag = IQ.imag

#         x0 = torch.cat([x0_real, x0_imag], dim=1).to(device)  # (B,2,H,W)
#         cond = torch.cat([cond_real, cond_imag], dim=1).to(device)  # (B,2,H,W)
        
#         t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device).long()
#         iq_loss = diffusion.p_losses(x0, t, cond)
#         total_val_loss += iq_loss.item()

#         # For the first batch, generate a sample and compute metrics.
#         if i == 0:
#             generated = diffusion.sample(cond, x0.shape)
#             mse_val = F.mse_loss(generated, x0).item()
#             psnr_val = 20 * math.log10(x0.max().item() / math.sqrt(mse_val)) if mse_val > 0 else 100
#             gen_mse, gen_psnr = mse_val, psnr_val

#     avg_val_loss = total_val_loss / len(dataloader)
#     return avg_val_loss, gen_mse, gen_psnr


#----------------------------------------------------------------------------#


def train_det_epoch(diffusion, dataloader, optimizer, device, 
                    lambda_det=0.2):
    """
    Run one epoch of training for the combined diffusion + detection model.

    diffusion: instance of ConditionalDiffusion wrapping a UNet with detection head
    dataloader: yields batches of (signals_norm, rd_signals_norm, IQs_norm, RDs_norm,
                                clutter_all, gauss_all, labels, scnr_dBs)
    optimizer: optimizer for diffusion.model parameters
    device: torch device
    lambda_det: weight for the detection loss

    Returns average total loss over batches.
    """
    diffusion.train()
    total_loss = 0.0
    for batch in dataloader:
        # Unpack batch
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, \
        clutter_all, gauss_all, labels, scnr_dBs = batch

        # Build x0 (clean RD) and cond (noisy RD)
        signal = rd_signals_norm.to(device)
        cond   = RDs_norm.to(device)

        # Ensure shape (B,2,H,W)
        if signal.real.ndim == 3:
            x0 = torch.cat([signal.real.unsqueeze(1), signal.imag.unsqueeze(1)], dim=1)
            cond = torch.cat([cond.real.unsqueeze(1), cond.imag.unsqueeze(1)], dim=1)
        else:
            x0 = torch.cat([signal.real, signal.imag], dim=1)
            cond = torch.cat([cond.real, cond.imag], dim=1)
        x0   = x0.to(device)
        cond = cond.to(device)

        # Detection labels: (B, H, W) -> add channel dim
        mask = labels.to(device).unsqueeze(1)  # (B,1,H,W)
        

        # Sample random timesteps
        t = torch.randint(0, diffusion.T, (x0.size(0),), device=device)
        # Add noise
        x_noisy, noise = diffusion.q_sample(x0, t)

        # Normalize timestep and forward
        t_norm = t.float() / diffusion.T
        inp    = torch.cat([x_noisy, cond], dim=1)
        noise_pred, det_pred = diffusion.model(inp, t_norm)

        # Noise prediction loss
        mse_loss = F.mse_loss(noise_pred, noise)
        # Detection loss (per-pixel BCE)
        pos = mask.sum()
        neg = mask.numel() - pos
        pos_weight = neg / (pos + 1e-6)           # scalar
        bce = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_weight, device=device))
        det_loss = bce(det_pred, mask)

#        det_loss = F.binary_cross_entropy_with_logits(det_pred, mask)

        loss = mse_loss + lambda_det * det_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

@torch.no_grad()
def det_validate(diffusion, dataloader, device, lambda_det=0):
    """
    Validate both denoising and detection performance.
    Returns average combined loss, plus generation MSE/PSNR for the first batch.
    """
    diffusion.eval()
    total_loss = 0.0
    gen_mse, gen_psnr = None, None

    for i, batch in enumerate(dataloader):
        # Unpack batch
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, \
        clutter_all, gauss_all, labels, scnr_dBs = batch

        # Build x0 (clean RD) and cond (noisy RD)
        signal = rd_signals_norm.to(device)
        cond   = RDs_norm.to(device)

        # Real/imag channels
        if signal.real.ndim == 3:
            x0   = torch.cat([signal.real.unsqueeze(1),
                              signal.imag.unsqueeze(1)], dim=1)
            cond = torch.cat([cond.real.unsqueeze(1),
                              cond.imag.unsqueeze(1)], dim=1)
        else:
            x0   = torch.cat([signal.real, signal.imag], dim=1)
            cond = torch.cat([cond.real,   cond.imag],   dim=1)
        x0   = x0.to(device)
        cond = cond.to(device)

        # Build mask (B,1,H,W)
        mask = labels.to(device).unsqueeze(1)

        # Sample timesteps & noise
        t       = torch.randint(0, diffusion.T, (x0.size(0),), device=device)
        x_noisy, noise = diffusion.q_sample(x0, t)

        # Forward through model
        t_norm       = t.float() / diffusion.T
        model_input  = torch.cat([x_noisy, cond], dim=1)
        noise_pred, det_pred = diffusion.model(model_input, t_norm)

        # Losses
        mse_loss = F.mse_loss(noise_pred, noise)
        det_loss = F.binary_cross_entropy_with_logits(det_pred, mask)
        loss     = mse_loss + lambda_det * det_loss
        total_loss += loss.item()

        # Generation metrics on first batch
        if i == 0:
            generated = diffusion.sample(cond, x0.shape)
            mse_val   = F.mse_loss(generated, x0).item()
            psnr_val  = 20 * math.log10(x0.max().item() / math.sqrt(mse_val)) \
                        if mse_val > 0 else float('inf')
            gen_mse, gen_psnr = mse_val, psnr_val

    avg_loss = total_loss / len(dataloader)
    return avg_loss, gen_mse, gen_psnr





