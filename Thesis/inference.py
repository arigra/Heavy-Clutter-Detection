import random
import torch
import matplotlib.pyplot as plt

def run_inference(cond_diffusion , norm_val_dataset, checkpoint, device):

    checkpoint_path = checkpoint
    cond_diffusion.load_state_dict(torch.load(checkpoint_path, map_location=device))
    cond_diffusion.eval()

    #signal, C, W, IQ, _, _ =  norm_val_dataset[random.randint(1,50)] #RadarDataset(num_samples=100, n_targets=8, random_n_targets=True,snr=30, cnr=15)[0],
    signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs = norm_val_dataset[random.randint(1,50)]
    IQ = RDs_norm 
    cond_img = torch.cat([IQ.real.unsqueeze(0), IQ.imag.unsqueeze(0)], dim=0)  # (2, H, W)
    cond_img = cond_img.unsqueeze(0).to(device)  # (1, 2, H, W)
    # The desired sample shape for the diffusion model is (1,2,H,W)
    sample_shape = (1, 2, IQ.shape[0], IQ.shape[1])

    # 5. Generate a denoised sample using the diffusion model
    with torch.no_grad():
        generated_sample = cond_diffusion.sample(cond_img, sample_shape)  # (1,2,H,W)

    # 6. Convert the generated 2-channel tensor into a complex tensor
    generated_complex = torch.complex(generated_sample[0,0,:,:], generated_sample[0,1,:,:])
    

    # fig, axes = plt.subplots(1, 3, figsize=(15,5))
    # for ax, img, title in zip(axes,
    #                         [signal, IQ, generated_complex.cpu()],
    #                         ["Clean IQ Map", "Noise Map", "Denoised IQ Map (Conditional)"]):
    #     im = ax.imshow(abs(img), cmap='viridis')
    #     ax.set_title(title)
    #     ax.axis("off")
    #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # plt.tight_layout()
    # plt.show()

    clean_np = rd_signals_norm.cpu().numpy()
    noisy_np = RDs_norm.cpu().numpy()

    fig, axes = plt.subplots(1, 3, figsize=(15,5))
    for ax, img, title in zip(axes,
                            [abs(clean_np), abs(noisy_np), abs(generated_complex.cpu().numpy())],
                            ["Clean RD Map", "Noisy RD Map", "Denoised RD Map (Conditional)"]):
        im = ax.imshow(img, cmap='viridis')
        ax.set_title(title)
        ax.axis("off")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()

    # fig, axes = plt.subplots(1, 3, figsize=(15,5))
    # for ax, img, title in zip(axes,
    #                         [-create_rd_map(generated_complex)+create_rd_map(IQ),
    #                         create_rd_map(C)+create_rd_map(W),
    #                         create_rd_map(C)+create_rd_map(W) - (create_rd_map(IQ)-create_rd_map(generated_complex))],
    #                         ["Predicted Noise", "Added Noise", "Added − Predicted"]):
    #     im = ax.imshow(img, cmap='viridis')
    #     ax.set_title(title)
    #     ax.axis("off")
    #     fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    # plt.tight_layout()
    # plt.show()

    