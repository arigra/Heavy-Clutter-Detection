import math, torch
import torch.nn as nn
import torch.nn.functional as F

def train_one_epoch(diffusion, dataloader, optimizer, device):
    diffusion.train()
    total_loss = 0
    for batch in dataloader:
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, *_ = batch

        # clean RD signal → x0
        x0_real = rd_signals_norm.real.unsqueeze(1)
        x0_imag = rd_signals_norm.imag.unsqueeze(1)
        x0 = torch.cat([x0_real, x0_imag], dim=1).to(device)  # (B,2,H,W)

        # two conditions: cluttered IQ & cluttered RD
        iq_real = IQs_norm.real.unsqueeze(1)
        iq_imag = IQs_norm.imag.unsqueeze(1)
        cond_iq = torch.cat([iq_real, iq_imag], dim=1).to(device)

        rd_real = RDs_norm.real.unsqueeze(1)
        rd_imag = RDs_norm.imag.unsqueeze(1)
        cond_rd = torch.cat([rd_real, rd_imag], dim=1).to(device)

        t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device).long()
        loss = diffusion.p_losses(x0, t, cond_iq, cond_rd)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
    return total_loss / len(dataloader)

@torch.no_grad()
def validate(diffusion, dataloader, device):
    diffusion.eval()
    total_loss = 0
    gen_mse = gen_psnr = None

    for i, batch in enumerate(dataloader):
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, *_ = batch

        x0 = torch.cat([
            rd_signals_norm.real.unsqueeze(1),
            rd_signals_norm.imag.unsqueeze(1)
        ], dim=1).to(device)

        cond_iq = torch.cat([
            IQs_norm.real.unsqueeze(1),
            IQs_norm.imag.unsqueeze(1)
        ], dim=1).to(device)

        cond_rd = torch.cat([
            RDs_norm.real.unsqueeze(1),
            RDs_norm.imag.unsqueeze(1)
        ], dim=1).to(device)

        t = torch.randint(0, diffusion.T, (x0.shape[0],), device=device).long()
        loss = diffusion.p_losses(x0, t, cond_iq, cond_rd)
        total_loss += loss.item()

        if i == 0:
            generated = diffusion.sample(cond_iq, cond_rd, x0.shape)
            mse = F.mse_loss(generated, x0).item()
            psnr = 20*math.log10(x0.max().item() / math.sqrt(mse)) if mse>0 else 100
            gen_mse, gen_psnr = mse, psnr

    return total_loss/len(dataloader), gen_mse, gen_psnr
