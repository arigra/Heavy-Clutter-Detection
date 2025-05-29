# import random
# import torch
# import matplotlib.pyplot as plt

# def run_inference(cond_diffusion , norm_val_dataset, checkpoint, device):

#     checkpoint_path = checkpoint
#     cond_diffusion.load_state_dict(torch.load(checkpoint_path, map_location=device))
#     cond_diffusion.eval()

#     #signal, C, W, IQ, _, _ =  norm_val_dataset[random.randint(1,50)] #RadarDataset(num_samples=100, n_targets=8, random_n_targets=True,snr=30, cnr=15)[0],
#     signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs = norm_val_dataset[random.randint(1,50)]
#     IQ = RDs_norm 
#     cond_img = torch.cat([IQ.real.unsqueeze(0), IQ.imag.unsqueeze(0)], dim=0)  # (2, H, W)
#     cond_img = cond_img.unsqueeze(0).to(device)  # (1, 2, H, W)
#     # The desired sample shape for the diffusion model is (1,2,H,W)
#     sample_shape = (1, 2, IQ.shape[0], IQ.shape[1])

#     # 5. Generate a denoised sample using the diffusion model
#     with torch.no_grad():
#         generated_sample = cond_diffusion.sample(cond_img, sample_shape)  # (1,2,H,W)

#     # 6. Convert the generated 2-channel tensor into a complex tensor
#     generated_complex = torch.complex(generated_sample[0,0,:,:], generated_sample[0,1,:,:])
    

#     # fig, axes = plt.subplots(1, 3, figsize=(15,5))
#     # for ax, img, title in zip(axes,
#     #                         [signal, IQ, generated_complex.cpu()],
#     #                         ["Clean IQ Map", "Noise Map", "Denoised IQ Map (Conditional)"]):
#     #     im = ax.imshow(abs(img), cmap='viridis')
#     #     ax.set_title(title)
#     #     ax.axis("off")
#     #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     # plt.tight_layout()
#     # plt.show()

#     clean_np = rd_signals_norm.cpu().numpy()
#     noisy_np = RDs_norm.cpu().numpy()

#     fig, axes = plt.subplots(1, 3, figsize=(15,5))
#     for ax, img, title in zip(axes,
#                             [abs(clean_np), abs(noisy_np), abs(generated_complex.cpu().numpy())],
#                             ["Clean RD Map", "Noisy RD Map", "Denoised RD Map (Conditional)"]):
#         im = ax.imshow(img, cmap='viridis')
#         ax.set_title(title)
#         ax.axis("off")
#         fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     plt.tight_layout()
#     plt.show()

#     # fig, axes = plt.subplots(1, 3, figsize=(15,5))
#     # for ax, img, title in zip(axes,
#     #                         [-create_rd_map(generated_complex)+create_rd_map(IQ),
#     #                         create_rd_map(C)+create_rd_map(W),
#     #                         create_rd_map(C)+create_rd_map(W) - (create_rd_map(IQ)-create_rd_map(generated_complex))],
#     #                         ["Predicted Noise", "Added Noise", "Added − Predicted"]):
#     #     im = ax.imshow(img, cmap='viridis')
#     #     ax.set_title(title)
#     #     ax.axis("off")
#     #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
#     # plt.tight_layout()
#     # plt.show()

import random
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter

def run_inference(cond_diffusion, norm_val_dataset, checkpoint, device,
                  nms_size: int = 5,
                  thr_frac: float = 0.5):
    """
    2×2 plot: Clean │ Noisy
             Denoised │ Detection Probabilities
    """

    # 1. Load weights, move model to device, set eval
    cond_diffusion.load_state_dict(torch.load(checkpoint, map_location=device))
    cond_diffusion.to(device)
    cond_diffusion.eval()

    # 2. Grab a random sample
    idx = random.randrange(len(norm_val_dataset))
    (signals_norm, rd_signals_norm, IQs_norm, RDs_norm,
     clutter_all, gauss_all, labels, scnr_dBs) = norm_val_dataset[idx]

    # 3. Build conditional input (real & imag)
    IQ = RDs_norm
    cond_img = torch.stack([IQ.real, IQ.imag], dim=0)[None].to(device)  # (1,2,H,W)
    sample_shape = (1, 2, IQ.shape[0], IQ.shape[1])

    # 4. Denoise
    with torch.no_grad():
        generated = cond_diffusion.sample(cond_img, sample_shape)       # (1,2,H,W)

    # 5. Run detection head at t=0
    inp    = torch.cat([generated, cond_img], dim=1)                   # (1,4,H,W)
    t_zero = torch.zeros(1, device=device)
    with torch.no_grad():
        _, det_logits = cond_diffusion.model(inp, t_zero)
        det_probs   = torch.sigmoid(det_logits)[0,0].cpu().numpy()     # (H,W)

    # 6. (Optional) you can still compute NMS/thr if you want to overlay later

    # 7. Prepare arrays
    clean_mag     = np.abs(rd_signals_norm.cpu().numpy())
    noisy_mag     = np.abs(RDs_norm.cpu().numpy())
    real_d, imag_d= generated[0,0].cpu().numpy(), generated[0,1].cpu().numpy()
    denoised_mag  = np.abs(real_d + 1j*imag_d)

    # 8. Plot in a 2×2 grid
    fig, axes = plt.subplots(2, 2, figsize=(12,12))
    # top-left: clean
    axes[0,0].imshow(clean_mag, cmap='viridis')
    axes[0,0].set_title("Clean RD Map")
    axes[0,0].axis('off')
    # top-right: noisy
    axes[0,1].imshow(noisy_mag, cmap='viridis')
    axes[0,1].set_title("Noisy RD Map")
    axes[0,1].axis('off')
    # bottom-left: denoised
    axes[1,0].imshow(denoised_mag, cmap='viridis')
    axes[1,0].set_title("Denoised RD Map")
    axes[1,0].axis('off')
    # bottom-right: detection probability heatmap
    im = axes[1,1].imshow(det_probs, cmap='hot', vmin=0, vmax=1)
    axes[1,1].set_title("Detection Probability Map")
    axes[1,1].axis('off')
    fig.colorbar(im, ax=axes[1,1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()

    # 9. Return for further use
    return {
        'clean':     clean_mag,
        'noisy':     noisy_mag,
        'denoised':  denoised_mag,
        'det_probs': det_probs,
        'labels':    labels.cpu().numpy()
    }
