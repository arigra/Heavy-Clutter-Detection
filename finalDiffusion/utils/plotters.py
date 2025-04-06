import torch
import wandb
import numpy as np
import matplotlib.pyplot as plt


def plot_with_colorbar(ax, data, title, xlabel, ylabel, cmap='viridis', aspect='auto'):
    im = ax.imshow(data, cmap=cmap, aspect=aspect)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    # Adding a colorbar for this axis
    plt.colorbar(im, ax=ax, shrink=0.8)

def visualize_sample(dataset, sample_index: int = 0, plot_here=True, save_to_wandb=True):
    # Assuming each sample returns these 6 elements
    signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = dataset[sample_index]
    print("SCNR (dB):", scnr_dB)

    # Plotting the signal, clutter, and noise components with colorbars
    fig1, axs1 = plt.subplots(1, 6, figsize=(20, 6))
    
    plot_with_colorbar(axs1[0], torch.real(signal), "Real Clean Signal", "Fast Time", "Slow Time")
    plot_with_colorbar(axs1[1], torch.imag(signal), "Imaginary Clean Signal", "Fast Time", "Slow Time")
    plot_with_colorbar(axs1[2], torch.real(clutter), "Real Clutter", "Fast Time", "Slow Time")
    plot_with_colorbar(axs1[3], torch.imag(clutter), "Imaginary Clutter", "Fast Time", "Slow Time")
    plot_with_colorbar(axs1[4], torch.real(gaus_noise), "Real White Gaussian Noise", "Fast Time", "Slow Time")
    plot_with_colorbar(axs1[5], torch.imag(gaus_noise), "Imaginary White Gaussian Noise", "Fast Time", "Slow Time")
    
    fig1.tight_layout()
    if save_to_wandb:
        plt.savefig("signal_clutter_noise.png", bbox_inches='tight')
        wandb.log({"signal_clutter_noise": wandb.Image("signal_clutter_noise.png")})
    if plot_here:
        plt.show()
    plt.close(fig1)

    # Plotting the IQ maps and ground truth label with colorbars
    fig2, axs2 = plt.subplots(1, 3, figsize=(20, 6))
    
    plot_with_colorbar(axs2[0], abs(IQ), "Noisy IQ Map", "Range", "Doppler")
    plot_with_colorbar(axs2[1], abs(signal), "Clean IQ Map", "Range", "Doppler")
    plot_with_colorbar(axs2[2], rd_label, "Ground Truth Label", "Range", "Doppler")
    
    fig2.tight_layout()
    if save_to_wandb:
        plt.savefig("iq_maps_labels.png", bbox_inches='tight')
        wandb.log({"iq_maps_labels": wandb.Image("iq_maps_labels.png")})
    if plot_here:
        plt.show()
    plt.close(fig2)

def compare_nu_scnr(nu_values, scnr_values, radar_dataset_class, plot_here=True, save_to_wandb=True):
    # First figure: Grid of IQ maps and clutter distribution histograms
    fig, axs = plt.subplots(
        nrows=len(nu_values),
        ncols=len(scnr_values) + 1,
        figsize=(20, 10),
        sharex=False,
        sharey=False
    )

    clutter_real_dict = {nu_val: [] for nu_val in nu_values}

    for i, nu_val in enumerate(nu_values):
        for j, scnr_val in enumerate(scnr_values):
            dataset = radar_dataset_class(
                num_samples=1,
                n_targets=1,
                random_n_targets=False,
                nu=nu_val,
                scnr=scnr_val
            )
            
            signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = dataset[0]

            iq_magnitude = torch.abs(IQ)
            iq_db = 20 * torch.log10(iq_magnitude + 1e-8)
            ax_rd = axs[i, j]
            im = ax_rd.imshow(iq_db, aspect='auto', cmap='viridis')
            ax_rd.set_title(f"nu={nu_val}, SCNR={scnr_val} dB", fontsize=9)
            plt.colorbar(im, ax=ax_rd)

        # Last column: Distribution of real parts of clutter samples
        ax_hist = axs[i, -1]
        num_clutter_samples = 50
        real_values = []

        for _ in range(num_clutter_samples):
            dataset_clutter = radar_dataset_class(
                num_samples=1,
                n_targets=0,
                random_n_targets=False,
                nu=nu_val
            )
            
            # Assuming gen_frame_and_labels returns a tuple with the clutter sample as the second element
            _, C_sample, _, _, _, _ = dataset_clutter.gen_frame_and_labels()
            real_part = torch.real(C_sample).view(-1).cpu().numpy()
            real_values.extend(real_part)

        ax_hist.hist(real_values, bins=50, density=True, alpha=0.7, color='gray')
        ax_hist.set_title(f"Real(Clutter) Dist\n(nu={nu_val})", fontsize=9)
        ax_hist.set_xlabel("Amplitude (Real Part)")
        ax_hist.set_ylabel("PDF")

        clutter_real_dict[nu_val].extend(real_values)

    plt.tight_layout()
    if save_to_wandb:
        plt.savefig("comparison_nu_scnr_grid.png", bbox_inches='tight')
        wandb.log({"comparison_nu_scnr_grid": wandb.Image("comparison_nu_scnr_grid.png")})
    if plot_here:
        plt.show()
    plt.close(fig)

    # Second figure: Overall comparison of clutter real-part distributions
    fig2 = plt.figure(figsize=(8, 6))
    all_vals = []
    for nu_val in nu_values:
        all_vals.extend(clutter_real_dict[nu_val])
    min_val = min(all_vals)
    max_val = max(all_vals)
    bins = np.linspace(min_val, max_val, 100)

    for nu_val in nu_values:
        arr = clutter_real_dict[nu_val]
        plt.hist(arr, bins=bins, alpha=0.5, density=True, label=f"nu={nu_val}")

    plt.yscale('log')
    plt.xlabel("Real(Clutter) amplitude")
    plt.ylabel("PDF (log scale)")
    plt.title("Comparison of Clutter Real-Part Distributions for Different nu")
    plt.legend()
    if save_to_wandb:
        plt.savefig("comparison_nu_scnr_hist.png", bbox_inches='tight')
        wandb.log({"comparison_nu_scnr_hist": wandb.Image("comparison_nu_scnr_hist.png")})
    if plot_here:
        plt.show()
    plt.close(fig2)


