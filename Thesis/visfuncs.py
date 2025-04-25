import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

def visualize_sample(dataset, sample_index: int = 0):
    """
    Works on either a Dataset or a DataLoader whose examples are
    (signals_norm, rd_signals_norm, IQs_norm, RDs_norm,
     clutter_all, gauss_all, labels, scnr_dBs).
    """
    # 1) Pull out one sample (or one batch + index into it)
    if isinstance(dataset, DataLoader):
        batch = next(iter(dataset))
        signals, rd_clean, IQ_maps, RD_maps, clutter, noise, labels, scnr_dBs = batch
        S        = signals[sample_index]
        rd_clean = rd_clean[sample_index]
        IQ_map   = IQ_maps[sample_index]
        RD_map   = RD_maps[sample_index]
        clutter  = clutter[sample_index]
        noise    = noise[sample_index]
        label    = labels[sample_index]
        scnr_db  = scnr_dBs[sample_index].item()
    else:
        S, rd_clean, IQ_map, RD_map, clutter, noise, label, scnr_db = dataset[sample_index]
        scnr_db = scnr_db.item()

    # 2) IQ‐domain plots
    plt.figure(figsize=(24, 6))
    plt.suptitle(f"Sample {sample_index} — SCNR: {scnr_db:.2f} dB", fontsize=16)
    iq_titles = ["Real S", "Imag S", "Real Clutter", "Imag Clutter", "Real IQ", "Imag IQ"]
    iq_data   = [S.real, S.imag, clutter.real, clutter.imag, IQ_map.real, IQ_map.imag]
    for i, (dat, title) in enumerate(zip(iq_data, iq_titles), 1):
        ax = plt.subplot(1, 6, i)
        ax.imshow(dat, aspect='auto', cmap='viridis')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Fast Time", fontsize=12)
        ax.set_ylabel("Slow Time", fontsize=12)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

    # 3) Range‐Doppler plots
    plt.figure(figsize=(24, 6))
    rd_titles = ["Real Noisy RD", "Imag Noisy RD", "Abs Noisy RD", "Abs Clean RD", "Ground Truth"]
    rd_data   = [RD_map.real, RD_map.imag, torch.abs(RD_map), torch.abs(rd_clean), label]
    for i, (dat, title) in enumerate(zip(rd_data, rd_titles), 1):
        ax = plt.subplot(1, 5, i)
        ax.imshow(dat, aspect='auto', cmap='viridis')
        ax.set_title(title, fontsize=14)
        ax.set_xlabel("Range", fontsize=12)
        ax.set_ylabel("Doppler", fontsize=12)
    plt.tight_layout()
    plt.show()
