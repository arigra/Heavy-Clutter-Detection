import plt.pyplot as plt
import torch

def map_plotter(S, C, IQ_map, clean_RD_map, RD_map, RD_label):
    plt.figure(figsize=(20, 6))

    # Subplot 1: Real part of IQ data
    plt.subplot(1, 6, 1)
    plt.imshow(torch.real(S), aspect='auto', cmap='viridis')
    plt.title("Real S", fontsize=14)
    plt.xlabel("Fast Time", fontsize=12)
    plt.ylabel("Slow Time", fontsize=12)

    # Subplot 2: Imaginary part of IQ data
    plt.subplot(1, 6, 2)
    plt.imshow(torch.imag(S), aspect='auto', cmap='viridis')
    plt.title("Imaginary S", fontsize=14)
    plt.xlabel("Fast Time", fontsize=12)
    plt.ylabel("Slow Time", fontsize=12)
    # Subplot 3: Magnitude of X_range
    plt.subplot(1, 6, 3)
    plt.imshow(torch.real(C) , aspect='auto', cmap='viridis')
    plt.title("Real C", fontsize=14)
    plt.xlabel("Fast Time", fontsize=12)
    plt.ylabel("Slow Time", fontsize=12)

    # Subplot 4: Magnitude of the Range-Doppler map
    plt.subplot(1, 6, 4)
    plt.imshow(torch.imag(C), aspect='auto', cmap='viridis')
    plt.title("Imaginary C", fontsize=14)
    plt.xlabel("Fast Time", fontsize=12)
    plt.ylabel("Slow Time", fontsize=12)

    # Subplot 5: Range-Doppler label (ground truth)
    plt.subplot(1, 6, 5)
    plt.imshow(torch.real(IQ_map), aspect='auto', cmap='viridis')
    plt.title("Real IQ map", fontsize=14)
    plt.xlabel("Fast Time", fontsize=12)
    plt.ylabel("Slow Time", fontsize=12)

    plt.subplot(1, 6, 6)
    plt.imshow(torch.imag(IQ_map), aspect='auto', cmap='viridis')
    plt.title("Real IQ map", fontsize=14)
    plt.xlabel("Fast Time", fontsize=12)
    plt.ylabel("Slow Time", fontsize=12)

    # Adjust the layout, save, and show
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("maps")
    plt.show()

    plt.figure(figsize=(20, 6))

    # Subplot 1: Real part of IQ data
    plt.subplot(1, 5, 1)
    plt.imshow(torch.real(RD_map), aspect='auto', cmap='viridis')
    plt.title("Real RD", fontsize=14)
    plt.xlabel("Range", fontsize=12)
    plt.ylabel("Doppler", fontsize=12)

    # Subplot 1: Real part of IQ data
    plt.subplot(1, 5, 2)
    plt.imshow(torch.imag(RD_map), aspect='auto', cmap='viridis')
    plt.title("Imaginary RD", fontsize=14)
    plt.xlabel("Range", fontsize=12)
    plt.ylabel("Doppler", fontsize=12)

    plt.subplot(1, 5, 3)
    plt.imshow(torch.abs(RD_map), aspect='auto', cmap='viridis')
    plt.title("RD Map (absolute value)", fontsize=14)
    plt.xlabel("Range", fontsize=12)
    plt.ylabel("Doppler", fontsize=12)

    plt.subplot(1, 5, 4)
    plt.imshow(torch.abs(clean_RD_map), aspect='auto', cmap='viridis')
    plt.title("clean RD Map (absolute value)", fontsize=14)
    plt.xlabel("Range", fontsize=12)
    plt.ylabel("Doppler", fontsize=12)

    # Subplot 2: Imaginary part of IQ data
    plt.subplot(1, 5, 5)
    plt.imshow(RD_label, aspect='auto', cmap='viridis')
    plt.title("Ground Truth", fontsize=14)
    plt.xlabel("Range", fontsize=12)
    plt.ylabel("Doppler", fontsize=12)

    # Adjust the layout, save, and show
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig("maps")
    plt.show()
