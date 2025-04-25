import math, torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb_factor = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb_factor)
        emb = x[:, None] * emb[None, :]  # shape: (B, half_dim)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb  # shape: (B, dim)

# Self-attention block operating on 2D features.
class SelfAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.norm = nn.GroupNorm(num_groups=8, num_channels=channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj_out = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h)  # (B, 3C, H, W)
        q, k, v = torch.chunk(qkv, 3, dim=1)

        # Reshape for multi-head attention.
        q = q.reshape(B, self.num_heads, C // self.num_heads, H * W)
        k = k.reshape(B, self.num_heads, C // self.num_heads, H * W)
        v = v.reshape(B, self.num_heads, C // self.num_heads, H * W)

        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) / math.sqrt(C // self.num_heads)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(B, C, H, W)
        out = self.proj_out(out)
        return x + out  # Residual connection

# A Double Convolution block with two convs (with GroupNorm and SiLU activation)
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU(),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.GroupNorm(num_groups=8, num_channels=out_ch),
            nn.SiLU()
        )
    def forward(self, x):
        return self.double_conv(x)

# Downsample with a double conv, then max pooling.
class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = DoubleConv(in_ch, out_ch)
        self.pool = nn.MaxPool2d(2)
    def forward(self, x):
        x_conv = self.conv(x)
        x_down = self.pool(x_conv)
        return x_conv, x_down

# Upsample with transpose convolution, concatenation of skip connection, and double conv.
class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)
    def forward(self, x, skip):
        x = self.up(x)
        # Adjust spatial dims if needed.
        if x.size() != skip.size():
            diffY = skip.size(2) - x.size(2)
            diffX = skip.size(3) - x.size(3)
            x = F.pad(x, [0, diffX, 0, diffY])
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)
# —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# 1) CROSS‑ATTENTION MODULE
# —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
class CrossAttention2d(nn.Module):
    def __init__(self, channels, num_heads=4):
        super().__init__()
        self.num_heads = num_heads
        self.channels  = channels
        self.norm_q    = nn.GroupNorm(8, channels)
        self.norm_kv   = nn.GroupNorm(8, channels)
        self.q_proj    = nn.Conv2d(channels, channels, kernel_size=1)
        self.k_proj    = nn.Conv2d(channels, channels, kernel_size=1)
        self.v_proj    = nn.Conv2d(channels, channels, kernel_size=1)
        self.proj_out  = nn.Conv2d(channels, channels, kernel_size=1)

    def forward(self, x_q, x_kv):
        """
        x_q:  (B,C,H,W)  → queries  
        x_kv: (B,C,H,W)  → keys & values  
        """
        B,C,H,W = x_q.shape

        q = self.q_proj(self.norm_q(x_q))   .view(B, self.num_heads, C//self.num_heads, H*W)
        k = self.k_proj(self.norm_kv(x_kv)) .view(B, self.num_heads, C//self.num_heads, H*W)
        v = self.v_proj(self.norm_kv(x_kv)) .view(B, self.num_heads, C//self.num_heads, H*W)

        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) \
             / math.sqrt(C//self.num_heads)
        attn = torch.softmax(attn, dim=-1)

        out = torch.einsum('bhnm,bhcm->bhcn', attn, v) \
            .reshape(B, C, H, W)
        out = self.proj_out(out)

        return x_q + out


# —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––
# 2) DUAL‑BRANCH U‑NET
# —–––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

class DualBranchConditionalUNet(nn.Module):
    def __init__(self, time_emb_dim=256):
        super().__init__()
        # time embedding
        self.time_emb = nn.Sequential(
            SinusoidalPosEmb(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )

        # — encoders — IQ branch, RD branch, **Noise branch** —
        self.iq_inc, self.rd_inc, self.noise_inc = (
            DoubleConv(2,64),
            DoubleConv(2,64),
            DoubleConv(2,64),
        )
        self.iq_down1, self.rd_down1, self.noise_down1 = (
            Down(64,128),
            Down(64,128),
            Down(64,128),
        )
        self.iq_down2, self.rd_down2, self.noise_down2 = (
            Down(128,256),
            Down(128,256),
            Down(128,256),
        )
        self.iq_down3, self.rd_down3, self.noise_down3 = (
            Down(256,256),
            Down(256,256),
            Down(256,256),
        )

        # cross‑attention on IQ↔RD skips (we won’t fuse noise skips here)
        self.cross1 = CrossAttention2d(128)
        self.cross2 = CrossAttention2d(256)

        # — bottleneck accepts 3×256=768 channels now —
        self.bot = DoubleConv(768, 512)
        self.bot_attn = CrossAttention2d(512)

        # time‑projections
        self.tp_bot = nn.Linear(time_emb_dim, 512)
        self.tp_up1 = nn.Linear(time_emb_dim, 256)
        self.tp_up2 = nn.Linear(time_emb_dim, 256)
        self.tp_up3 = nn.Linear(time_emb_dim, 128)

        # decoder (fused skip channels)
        self.up1 = Up(512, 256, 256)
        self.up2 = Up(256, 256, 256)
        self.up3 = Up(256, 128, 128)
        self.up4 = Up(128, 64, 64)
        self.outc = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x_noisy, cond_iq, cond_rd, t):
        te = self.time_emb(t)   # (B, time_emb_dim)

        # —— IQ branch —— 
        x1_i = self.iq_inc(cond_iq)
        s2_i, x2_i = self.iq_down1(x1_i)
        s3_i, x3_i = self.iq_down2(x2_i)
        s4_i, x4_i = self.iq_down3(x3_i)

        # —— RD branch —— 
        x1_r = self.rd_inc(cond_rd)
        s2_r, x2_r = self.rd_down1(x1_r)
        s3_r, x3_r = self.rd_down2(x2_r)
        s4_r, x4_r = self.rd_down3(x3_r)

        # —— Noise branch —— 
        x1_n = self.noise_inc(x_noisy)
        s2_n, x2_n = self.noise_down1(x1_n)
        s3_n, x3_n = self.noise_down2(x2_n)
        s4_n, x4_n = self.noise_down3(x3_n)

        # —— cross‑fuse IQ↔RD skip features —— 
        s2 = self.cross1(s2_i, s2_r)
        s3 = self.cross2(s3_i, s3_r)

        # —— fuse at bottleneck (now includes noise) —— 
        xb = torch.cat([x4_i, x4_r, x4_n], dim=1)   # → (B,768,H/8,W/8)
        xb = self.bot(xb)
        xb = self.bot_attn(xb, xb)
        tb = self.tp_bot(te).view(-1,512,1,1)
        xb = xb + tb

        # —— decoder —— 
        x = self.up1(xb, s4_i + s4_r)               # still just IQ+RD skips
        tu = self.tp_up1(te).view(-1,256,1,1); x = x + tu

        x = self.up2(x, s3)
        tu = self.tp_up2(te).view(-1,256,1,1); x = x + tu

        x = self.up3(x, s2)
        tu = self.tp_up3(te).view(-1,128,1,1); x = x + tu

        x = self.up4(x, x1_i + x1_r)
        return self.outc(x)