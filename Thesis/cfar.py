import numpy as np
from dataset import *
def ca_cfar_2d(signal, num_train, num_guard, Pfa):
    """
    Standard CA–CFAR on a 2D signal.
    """
    rows, cols = signal.shape
    detection_map = np.zeros_like(signal)
    
    win_size = 2 * (num_train + num_guard) + 1
    guard_size = 2 * num_guard + 1
    num_training_cells = win_size**2 - guard_size**2
    
    # Scaling factor for exponential noise
    alpha = num_training_cells * (Pfa**(-1/num_training_cells) - 1)
    
    pad = num_train + num_guard
    padded_signal = np.pad(signal, pad, mode='constant', constant_values=0)
    
    for i in range(pad, pad + rows):
        for j in range(pad, pad + cols):
            window = padded_signal[i - pad:i + pad + 1, j - pad:j + pad + 1]
            start = num_train
            end = num_train + 2 * num_guard + 1
            training_cells = np.concatenate((window[:start, :].ravel(),
                                             window[end:, :].ravel(),
                                             window[start:end, :start].ravel(),
                                             window[start:end, end:].ravel()))
            noise_level = np.mean(training_cells)
            threshold = alpha * noise_level
            if signal[i - pad, j - pad] > threshold:
                detection_map[i - pad, j - pad] = 1
    return detection_map

def tm_cfar_2d(signal, num_train, num_guard, trim_ratio, Pfa):
    """
    TM–CFAR on a 2D signal.
    """
    rows, cols = signal.shape
    detection_map = np.zeros_like(signal)
    
    win_size = 2 * (num_train + num_guard) + 1
    guard_size = 2 * num_guard + 1
    num_training_cells = win_size**2 - guard_size**2
    
    # Number of cells to trim from each end
    trim_cells = int(trim_ratio * num_training_cells)
    effective_cells = num_training_cells - 2 * trim_cells
    if effective_cells <= 0:
        effective_cells = num_training_cells  # fallback
    alpha = effective_cells * (Pfa**(-1/effective_cells) - 1)
    
    pad = num_train + num_guard
    padded_signal = np.pad(signal, pad, mode='constant', constant_values=0)
    
    for i in range(pad, pad + rows):
        for j in range(pad, pad + cols):
            window = padded_signal[i - pad:i + pad + 1, j - pad:j + pad + 1]
            start = num_train
            end = num_train + 2 * num_guard + 1
            training_cells = np.concatenate((window[:start, :].ravel(),
                                             window[end:, :].ravel(),
                                             window[start:end, :start].ravel(),
                                             window[start:end, end:].ravel()))
            sorted_cells = np.sort(training_cells)
            if 2 * trim_cells < num_training_cells:
                trimmed = sorted_cells[trim_cells: num_training_cells - trim_cells]
            else:
                trimmed = sorted_cells
            noise_level = np.mean(trimmed)
            threshold = alpha * noise_level
            if signal[i - pad, j - pad] > threshold:
                detection_map[i - pad, j - pad] = 1
    return detection_map

def simulate_cfar_performance(cfar_func, specified_Pfa, nu_val, num_trials=100,
                              n_targets=3, random_n_targets=False, **cfar_kwargs):
    """
    For a given CFAR function, specified false–alarm parameter, and clutter nu,
    simulate num_trials frames and compute the average probability of detection (Pd)
    and measured probability of false alarm (Pfa_meas).
    """
    dataset = DAFCDataset(num_samples=num_trials, n_targets=n_targets,
                           random_n_targets=random_n_targets, nu=nu_val)
    total_true_detections = 0
    total_targets = 0
    total_false_alarms = 0
    total_non_target_cells = 0
    for i in range(num_trials):
        _, _, _, _, RD_map, rd_label = dataset[i]
        RD_mag = torch.abs(RD_map).detach().numpy()
        detection_map = cfar_func(RD_mag, **cfar_kwargs, Pfa=specified_Pfa)
        gt = rd_label.detach().numpy()
        true_detections = np.sum((detection_map == 1) & (gt == 1))
        false_alarms = np.sum((detection_map == 1) & (gt == 0))
        total_targets += np.sum(gt)
        total_true_detections += true_detections
        total_false_alarms += false_alarms
        total_non_target_cells += (gt.size - np.sum(gt))
    pd_rate = total_true_detections / total_targets if total_targets > 0 else 0
    measured_pfa = total_false_alarms / total_non_target_cells if total_non_target_cells > 0 else 0
    return pd_rate, measured_pfa

def simulate_cfar_dif(dataset, cfar_func, specified_Pfa, nu_val, num_trials=100,
                              n_targets=3, random_n_targets=False, **cfar_kwargs):
    """
    For a given CFAR function, specified false–alarm parameter, and clutter nu,
    simulate num_trials frames and compute the average probability of detection (Pd)
    and measured probability of false alarm (Pfa_meas).
    """
    dataset = dataset
    total_true_detections = 0
    total_targets = 0
    total_false_alarms = 0
    total_non_target_cells = 0
    for i in range(num_trials):
        signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs = dataset[i]
        RD_mag = torch.abs(RDs_norm).detach().numpy()
        detection_map = cfar_func(RD_mag, **cfar_kwargs, Pfa=specified_Pfa)
        gt = labels.detach().numpy()
        true_detections = np.sum((detection_map == 1) & (gt == 1))
        false_alarms = np.sum((detection_map == 1) & (gt == 0))
        total_targets += np.sum(gt)
        total_true_detections += true_detections
        total_false_alarms += false_alarms
        total_non_target_cells += (gt.size - np.sum(gt))
    pd_rate = total_true_detections / total_targets if total_targets > 0 else 0
    measured_pfa = total_false_alarms / total_non_target_cells if total_non_target_cells > 0 else 0
    return pd_rate, measured_pfa