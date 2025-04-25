import math
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader, ConcatDataset, TensorDataset

class RadarDataset(Dataset):
    def __init__(self, num_samples, n_targets, random_n_targets, nu=None, scnr=None, snr=None, cnr=None):
        super().__init__()
        self.num_samples = num_samples
        self.n_targets = n_targets
        self.random_n_targets = random_n_targets
        self.with_targets = n_targets > 0
        self.snr_dB = snr
        self.cnr_dB = cnr
        self.scnr = scnr
        self.nu = torch.tensor([nu], dtype=torch.float) if nu is not None else None

        # Radar parameters
        self.N = 64       # fast–time samples per pulse
        self.K = 64       # slow–time pulses per frame
        self.B = 50e6     # Chirp bandwidth (Hz)
        self.T0 = 1e-3    # PRI (s)
        self.fc = 9.39e9  # Carrier frequency (Hz)
        self.c = 3e8      # Speed of light (m/s)
        self.CNR = 15     # in dB (only used if snr/cnr are NOT given)

        # Range and Doppler settings
        self.r_min, self.r_max = 0, 189    # meters
        self.v_min, self.v_max = -7.8, 7.8   # m/s (for targets)
        self.vc_min, self.vc_max = -7.8, 7.8 # m/s (for clutter)
        self.dr = 3       # Range resolution in m
        self.dv = 0.249   # Doppler resolution in m/s

        # Range and Doppler bins (for label maps)
        self.R = torch.arange(self.r_min, self.r_max + self.dr, self.dr)
        self.V = torch.arange(self.v_min, self.v_max + self.dv, self.dv)
        self.dR = len(self.R)
        self.dV = len(self.V)

        # Noise power calculation (only used if snr/cnr not specified)
        self.sigma2 = self.N / (2 * 10 ** (self.CNR / 10))
        
        # For old scnr logic, we also computed a "normalization" factor
        self.cn_norm = torch.sqrt(
            torch.tensor(
                self.N * self.K * (self.N // 2 + self.sigma2), dtype=torch.float
            )
        )

    def generate_target_signal(self, ranges, velocities, phases, gains_dB):

        w_r = (2 * torch.pi * 2 * self.B * ranges) / (self.c * self.N)
        range_steering = torch.exp(-1j * torch.outer(w_r, torch.arange(self.N, dtype=torch.float)))
        
        w_d = (2 * torch.pi * self.T0 * 2 * self.fc * velocities) / self.c
        doppler_steering = torch.exp(-1j * torch.outer(w_d, torch.arange(self.K, dtype=torch.float)))
        
        rd_signal = range_steering.unsqueeze(-1) * doppler_steering.unsqueeze(1)
        rd_signal = rd_signal * torch.exp(1j * phases)  # impart random phase per target
        

        if (self.snr_dB is None) or (self.cnr_dB is None):
            S_norm = torch.linalg.norm(rd_signal, dim=(1, 2)).real
            sig_amp = (10 ** (gains_dB / 20)) * (self.cn_norm / S_norm)
            rd_signal = (sig_amp.unsqueeze(-1).unsqueeze(-1) * rd_signal).sum(dim=0)
            return rd_signal
        else:

            rd_signal = rd_signal.sum(dim=0)
            return rd_signal

    def generate_clutter(self, nu):
        clutter_vel = torch.empty(1).uniform_(self.vc_min, self.vc_max)
        fd = 2 * torch.pi * (2 * self.fc * clutter_vel) / self.c 
        sigma_f = 0.05  # Correlation parameter (from the referenced paper)

        p, q = torch.meshgrid(
            torch.arange(self.N, dtype=torch.float),
            torch.arange(self.K, dtype=torch.float),
            indexing='ij'
        )
        M = torch.exp(
            -2 * torch.pi**2 * sigma_f**2 * (p - q)**2
            - 1j * (p - q) * fd * self.T0
        )

        z = torch.randn(self.K, self.dR, dtype=torch.cfloat) / torch.sqrt(torch.tensor(2.0))
        e, V_mat = torch.linalg.eigh(M)  # eigen-decomposition
        e_sqrt = torch.sqrt(torch.clamp(e.real, min=0.0))
        E = torch.diag(e_sqrt)
        A = V_mat @ E.to(V_mat.dtype)
        w_t = A @ z  # shaping the random draws to match M

        s = torch.distributions.Gamma(nu, nu).sample((self.dR,))
        c_t = (torch.sqrt(s).unsqueeze(0) * w_t.unsqueeze(-1)).squeeze(-1)


        c_r_steer = torch.exp(
            -1j 
            * 2 
            * torch.pi 
            * torch.outer(torch.arange(self.N, dtype=torch.float), self.R)
            * (2 * self.B) / (self.c * self.N)
        )
        C = c_r_steer @ c_t.transpose(0, 1)
        return C

    def gen_frame_and_labels(self):

        W_unscaled = torch.randn(self.N, self.K, dtype=torch.cfloat) / torch.sqrt(torch.tensor(2.0))
        
        nu = torch.empty(1).uniform_(0.1, 1.5) if self.nu is None else self.nu
        C_unscaled = self.generate_clutter(nu)
        

        S_unscaled = torch.zeros(self.N, self.K, dtype=torch.cfloat)
        rd_label = torch.zeros(self.dR, self.dV)

        if self.with_targets:
            n = (
                torch.randint(1, self.n_targets + 1, (1,)).item()
                if self.random_n_targets
                else self.n_targets
            )
            ranges = torch.empty(n).uniform_(self.r_min, self.r_max)
            velocities = torch.empty(n).uniform_(self.v_min, self.v_max)
            phases = torch.empty(n, 1, 1).uniform_(0, 2 * torch.pi)
            
            if (self.snr_dB is None) or (self.cnr_dB is None):
                SCNR_dBs = torch.empty(n).uniform_(-5, 10) if self.scnr is None else self.scnr * torch.ones(n)
                S_unscaled = self.generate_target_signal(ranges, velocities, phases, SCNR_dBs)
            else:

                S_raw = []
                for i in range(n):
                    s_i = self.generate_target_signal(
                        ranges[i].unsqueeze(-1),
                        velocities[i].unsqueeze(-1),
                        phases[i].unsqueeze(-1),
                        gains_dB=torch.tensor([0.0])  # placeholder
                    )
                    S_raw.append(s_i)
                S_unscaled = sum(S_raw)

            for r, v in zip(ranges, velocities):
                r_bin = torch.argmin(torch.abs(self.R - r))
                v_bin = torch.argmin(torch.abs(self.V - v))
                rd_label[r_bin, v_bin] = 1

        if (self.snr_dB is not None) and (self.cnr_dB is not None):
            noise_power  = W_unscaled.abs().pow(2).mean()
            clutter_power= C_unscaled.abs().pow(2).mean() if C_unscaled.numel() > 0 else 0.0
            signal_power = S_unscaled.abs().pow(2).mean() if S_unscaled.numel() > 0 else 0.0

            snr_lin = 10 ** (self.snr_dB / 10)
            cnr_lin = 10 ** (self.cnr_dB / 10)

            alpha_n = torch.sqrt(1.0 / noise_power)
            W = alpha_n * W_unscaled  # final noise
            final_noise_power = W.abs().pow(2).mean()

            if clutter_power > 0:
                alpha_c = torch.sqrt((cnr_lin * final_noise_power) / clutter_power)
                C = alpha_c * C_unscaled
            else:
                C = torch.zeros_like(C_unscaled)

            if signal_power > 0:
                alpha_s = torch.sqrt((snr_lin * final_noise_power) / signal_power)
                S = alpha_s * S_unscaled
            else:
                S = torch.zeros_like(S_unscaled)

        else:
            W = (W_unscaled / torch.sqrt(torch.tensor(self.sigma2)))  # old approach
            C = C_unscaled
            S = S_unscaled

        X = S + C + W
        
        signal_energy  = S.abs().pow(2).sum()
        clutter_energy = C.abs().pow(2).sum()
        noise_energy   = W.abs().pow(2).sum()
        scnr_lin = signal_energy / (clutter_energy + noise_energy + 1e-12)
        scnr_dB  = 10.0 * torch.log10(scnr_lin + 1e-12)

        return S, C, W, X, rd_label, scnr_dB

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = self.gen_frame_and_labels()
        return signal, clutter, gaus_noise, IQ, rd_label, scnr_dB

def generate_range_steering_matrix(N=64, dR=64, B=50e6, c=3e8):
    """
    Generates the range steering matrix R.
    """
    rng_res = c / (2 * B)
    r_vals = torch.arange(dR) * rng_res
    n_vals = torch.arange(N)
    phase = -1j * 2 * math.pi * (2 * B) / (c * N)
    R = torch.exp(phase * torch.outer(n_vals, r_vals))
    return R

def generate_doppler_steering_matrix(K=64, dV=64, fc=9.39e9, T0=1e-3, c=3e8):
    """
    Generates the Doppler steering matrix V.
    """
    vel_res = c / (2 * fc * K * T0)
    # Create a symmetric velocity vector
    v_vals = torch.linspace(-dV // 2, dV // 2, dV) * vel_res
    k_vals = torch.arange(K)
    phase = -1j * 2 * math.pi * (2 * fc * T0) / c
    V = torch.exp(phase * torch.outer(k_vals, v_vals))
    return V

def create_rd_map_differentiable(IQ_map):
    """
    Converts a complex IQ (or signal) map to a range–Doppler (RD) map.
    If the input is not already complex, it is cast to torch.complex64.
    Returns the absolute value of the radar return in the RD domain.
    """
    if not torch.is_tensor(IQ_map):
        IQ_map = torch.from_numpy(IQ_map)
    if not torch.is_complex(IQ_map):
        IQ_map = IQ_map.to(torch.complex64)
    device = IQ_map.device
    R = generate_range_steering_matrix().to(device)
    V = generate_doppler_steering_matrix().to(device)
    # The steering is applied to convert the input domain to the RD domain.
    RD_map = R.T.conj() @ IQ_map @ V.conj()
    return RD_map

def get_mean_std(radarloader, convert=False):
    IQ_total_sum = 0.0
    IQ_total_sq_sum = 0.0
    IQ_total_samples = 0
    signal_total_sum = 0.0
    signal_total_sq_sum = 0.0
    signal_total_samples = 0
    for signal, _, _, IQ, _, _ in radarloader:
        if convert:
            signal = create_rd_map_differentiable(signal)
            IQ = create_rd_map_differentiable(IQ)
        IQ_total_sum += IQ.real.sum() + IQ.imag.sum() #signal.abs().sum()
        IQ_total_sq_sum += (IQ.real.pow(2).sum() + IQ.imag.pow(2).sum()) #(signal.abs()**2).sum()
        IQ_total_samples += IQ.numel() * 2 # multiply by 2 for real and imaginary 
        signal_total_sum += signal.real.sum() + signal.imag.sum() #signal.abs().sum()
        signal_total_sq_sum += (signal.real.pow(2).sum() + signal.imag.pow(2).sum()) #(signal.abs()**2).sum()
        signal_total_samples += signal.numel() * 2 # multiply by 2 for real and imaginary 
    IQ_mean = IQ_total_sum / IQ_total_samples
    IQ_std = torch.sqrt((IQ_total_sq_sum / IQ_total_samples) - IQ_mean**2)
    signal_mean = signal_total_sum / signal_total_samples
    signal_std = torch.sqrt((signal_total_sq_sum / signal_total_samples) - signal_mean**2)
    return signal_mean, signal_std, IQ_mean, IQ_std

def normalize_and_cache_dataset(dataset, iq_signal_mean, iq_signal_std, iq_IQ_mean, iq_IQ_std, rd_signal_mean, rd_signal_std, rd_IQ_mean, rd_IQ_std):
    signals_norm = []
    rd_signals_norm = []
    IQs_norm = []
    RDs_norm = []
    labels = []
    scnr_dBs = []
    clutter_all = []  
    gauss_all = []   

    for idx in tqdm(range(len(dataset)), desc='Normalizing dataset'):
        signal, clutter, gaus_noise, IQ, rd_label, scnr_dB = dataset[idx]
        rd_signal = create_rd_map_differentiable(signal)
        rd_IQ = create_rd_map_differentiable(IQ)
        # Normalize signal
        signal_real_norm = (signal.real - iq_signal_mean) / iq_signal_std
        signal_imag_norm = (signal.imag - iq_signal_mean) / iq_signal_std
        signal_norm = torch.complex(signal_real_norm, signal_imag_norm)
        
        # Normalize IQ
        IQ_real_norm = (IQ.real - iq_IQ_mean) / iq_IQ_std
        IQ_imag_norm = (IQ.imag - iq_IQ_mean) / iq_IQ_std
        IQ_norm = torch.complex(IQ_real_norm, IQ_imag_norm)

        # Normalize rd signal
        rd_signal_real_norm = (rd_signal.real - rd_signal_mean) / rd_signal_std
        rd_signal_imag_norm = (rd_signal.imag - rd_signal_mean) / rd_signal_std
        rd_signal_norm = torch.complex(rd_signal_real_norm, rd_signal_imag_norm)
        
        # Normalize rd IQ
        RD_real_norm = (rd_IQ.real - rd_IQ_mean) / rd_IQ_std
        RD_imag_norm = (rd_IQ.imag - rd_IQ_mean) / rd_IQ_std
        RD_norm = torch.complex(RD_real_norm, RD_imag_norm)

        signals_norm.append(signal_norm)
        IQs_norm.append(IQ_norm)
        rd_signals_norm.append(rd_signal_norm)
        RDs_norm.append(RD_norm)
        labels.append(rd_label)
        scnr_dBs.append(scnr_dB)
        
        # Save clutter and gauss tensors as well
        clutter_all.append(clutter)      # <- added
        gauss_all.append(gaus_noise)     # <- added

    # Stack everything into tensors
    signals_norm = torch.stack(signals_norm)
    IQs_norm = torch.stack(IQs_norm)
    rd_signals_norm = torch.stack(rd_signals_norm)
    RDs_norm = torch.stack(RDs_norm)
    labels = torch.stack(labels)
    scnr_dBs = torch.tensor(scnr_dBs)
    clutter_all = torch.stack(clutter_all)  # <- added
    gauss_all = torch.stack(gauss_all)      # <- added

    # Return cached TensorDataset (now consistent)
    return TensorDataset(signals_norm, rd_signals_norm, IQs_norm, RDs_norm, clutter_all, gauss_all, labels, scnr_dBs)

def prep_dataset(config):
    snr_list = config.SNR
    cnr_list = config.CNR
    nu_list  = config.NU

    # how many distinct (snr, cnr, nu) combos
    C = len(snr_list) * len(cnr_list) * len(nu_list)

    train_datasets = []
    val_datasets   = []

    # for each ν, SNR, CNR triple
    for nu in nu_list:
        for snr in snr_list:
            for cnr in cnr_list:
                # compute per-combo sizes
                n_train_with_tgt  = config.dataset_size // C
                n_train_without_tg= config.dataset_size // (10 * C)
                n_val_with_tgt    = config.dataset_size // (10 * C)
                n_val_without_tg  = config.dataset_size // (100 * C)

                # create training splits
                train_w = RadarDataset(
                    num_samples    = n_train_with_tgt,
                    n_targets      = config.n_targets,
                    random_n_targets = config.rand_n_targets,
                    snr            = snr,
                    cnr            = cnr,
                    nu             = nu
                )
                train_wo = RadarDataset(
                    num_samples    = n_train_without_tg,
                    n_targets      = 0,
                    random_n_targets = False,
                    snr            = snr,
                    cnr            = cnr,
                    nu             = nu
                )
                train_datasets.append(ConcatDataset([train_w, train_wo]))

                # create validation splits
                val_w = RadarDataset(
                    num_samples    = n_val_with_tgt,
                    n_targets      = config.n_targets,
                    random_n_targets = config.rand_n_targets,
                    snr            = snr,
                    cnr            = cnr,
                    nu             = nu
                )
                val_wo = RadarDataset(
                    num_samples    = n_val_without_tg,
                    n_targets      = 0,
                    random_n_targets = False,
                    snr            = snr,
                    cnr            = cnr,
                    nu             = nu
                )
                val_datasets.append(ConcatDataset([val_w, val_wo]))

    # concat all combos
    full_train_dataset = ConcatDataset(train_datasets)
    full_val_dataset   = ConcatDataset(val_datasets)

    # loaders & normalization (unchanged)
    train_loader = DataLoader(full_train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader   = DataLoader(full_val_dataset,   batch_size=config.batch_size, shuffle=False)

    signal_mean,   signal_std,   IQ_mean,  IQ_std   = get_mean_std(train_loader, convert=False)
    rd_s_mean,     rd_s_std,     rd_IQ_mean, rd_IQ_std = get_mean_std(train_loader, convert=True)

    norm_train_dataset = normalize_and_cache_dataset(
        full_train_dataset,
        signal_mean, signal_std, IQ_mean, IQ_std,
        rd_s_mean,   rd_s_std,   rd_IQ_mean, rd_IQ_std
    )
    norm_val_dataset   = normalize_and_cache_dataset(
        full_val_dataset,
        signal_mean, signal_std, IQ_mean, IQ_std,
        rd_s_mean,   rd_s_std,   rd_IQ_mean, rd_IQ_std
    )

    norm_train_loader = DataLoader(norm_train_dataset, batch_size=config.batch_size, shuffle=True)
    norm_val_loader   = DataLoader(norm_val_dataset,   batch_size=config.batch_size, shuffle=False)

    return norm_train_loader, norm_val_loader, norm_train_dataset, norm_val_dataset


class DAFCDataset(Dataset):
    def __init__(self, num_samples, n_targets: int = 3, random_n_targets=False, nu=None, scnr=None):
        super().__init__()
        self.num_samples = num_samples
        self.n_targets = n_targets
        self.random_n_targets = random_n_targets
        self.with_targets = n_targets > 0
        self.scnr = scnr
        # If nu is provided, fix it; otherwise clutter nu will be drawn uniformly in [0.1, 1.5]
        self.nu = torch.tensor([nu], dtype=torch.float) if nu is not None else None

        # Radar parameters
        self.N = 64       # fast–time samples per pulse
        self.K = 64       # slow–time pulses per frame
        self.B = 50e6     # Chirp bandwidth (Hz)
        self.T0 = 1e-3    # PRI (s)
        self.fc = 9.39e9  # Carrier frequency (Hz)
        self.c = 3e8      # Speed of light (m/s)
        self.CNR = 15     # Clutter-to-noise ratio in dB

        # Range and Doppler settings
        self.r_min, self.r_max = 0, 189    # meters
        self.v_min, self.v_max = -7.8, 7.8   # m/s (for targets)
        self.vc_min, self.vc_max = -7.8, 7.8 # m/s (for clutter)
        self.dr = 3     # Range resolution in m
        self.dv = 0.249 # Doppler resolution in m/s

        # Range and Doppler bins (for label maps)
        self.R = torch.arange(self.r_min, self.r_max + self.dr, self.dr)
        self.V = torch.arange(self.v_min, self.v_max + self.dv, self.dv)
        self.dR = len(self.R)
        self.dV = len(self.V)

        # Noise power calculation
        self.sigma2 = self.N / (2 * 10 ** (self.CNR / 10))
        self.cn_norm = torch.sqrt(torch.tensor(self.N * self.K * (self.N // 2 + self.sigma2), dtype=torch.float))

        # Precompute the range steering matrix (matched filter in range)
        w_range = (2 * torch.pi * (2 * self.B) / (self.c * self.N)) * self.R.unsqueeze(-1) * torch.arange(self.N, dtype=torch.float)
        self.range_steering_full = torch.exp(-1j * w_range)

    def generate_target_signal(self, ranges, velocities, phases, SCNR_dBs):
        # Range steering vector (one per target)
        w_r = (2 * torch.pi * 2 * self.B * ranges) / (self.c * self.N)
        range_steering = torch.exp(-1j * torch.outer(w_r, torch.arange(self.N, dtype=torch.float)))
        # Doppler steering vector (one per target)
        w_d = (2 * torch.pi * self.T0 * 2 * self.fc * velocities) / self.c
        doppler_steering = torch.exp(-1j * torch.outer(w_d, torch.arange(self.K, dtype=torch.float)))
        # Form the fast–time × slow–time target signature for each target
        rd_signal = range_steering.unsqueeze(-1) * doppler_steering.unsqueeze(1)
        rd_signal = rd_signal * torch.exp(1j * phases)  # impart random phase per target

        # Scale the target signature to achieve the desired SCNR (in dB)
        S_norm = torch.linalg.norm(rd_signal, dim=(1, 2)).real
        sig_amp = (10 ** (SCNR_dBs / 20)) * (self.cn_norm / S_norm)
        rd_signal = (sig_amp.unsqueeze(-1).unsqueeze(-1) * rd_signal).sum(dim=0)
        return rd_signal

    def generate_clutter(self, nu):
        # Choose a clutter velocity uniformly within the allowed limits
        clutter_vel = torch.empty(1).uniform_(self.vc_min, self.vc_max)
        fd = 2 * torch.pi * (2 * self.fc * clutter_vel) / self.c 
        sigma_f = 0.05  # Correlation parameter (from the referenced paper)
        p, q = torch.meshgrid(torch.arange(self.N, dtype=torch.float),
                              torch.arange(self.K, dtype=torch.float), indexing='ij')
        M = torch.exp(-2 * torch.pi ** 2 * sigma_f ** 2 * (p - q) ** 2 - 1j * (p - q) * fd * self.T0)
        z = torch.randn(self.K, self.dR, dtype=torch.cfloat) / torch.sqrt(torch.tensor(2.0))
        e, V_mat = torch.linalg.eigh(M)
        e_sqrt = torch.sqrt(torch.clamp(e.real, min=0.0))
        E = torch.diag(e_sqrt)
        A = V_mat @ E.to(V_mat.dtype)
        w_t = A @ z

        # Impart heavy–tailed behavior via Gamma modulation (shape and scale = nu)
        s = torch.distributions.Gamma(nu, nu).sample((self.dR,))
        c_t = (torch.sqrt(s).unsqueeze(0) * w_t.unsqueeze(-1)).squeeze(-1)

        # Convert to fast–time × slow–time representation using a range–steering operation.
        c_r_steer = torch.exp(-1j * 2 * torch.pi * torch.outer(torch.arange(self.N, dtype=torch.float), self.R) *
                                (2 * self.B) / (self.c * self.N))
        C = c_r_steer @ c_t.transpose(0, 1)
        return C

    def gen_frame_and_labels(self):
        # Generate noise (complex Gaussian)
        W = (torch.randn(self.N, self.K, dtype=torch.cfloat) / 
             torch.sqrt(torch.tensor(2.0 * self.sigma2)))
        # Generate clutter; if self.nu is not set, choose nu uniformly in [0.1, 1.5]
        nu = torch.empty(1).uniform_(0.1, 1.5) if self.nu is None else self.nu
        C = self.generate_clutter(nu)

        # Initialize target signal and label map
        S = torch.zeros_like(W)
        rd_label = torch.zeros(self.dR, self.dV)
        if self.with_targets:
            # If not random, use fixed number of targets.
            n = torch.randint(1, self.n_targets + 1, (1,)).item() if self.random_n_targets else self.n_targets
            ranges = torch.empty(n).uniform_(self.r_min, self.r_max)
            velocities = torch.empty(n).uniform_(self.v_min, self.v_max)
            phases = torch.empty(n, 1, 1).uniform_(0, 2 * torch.pi)
            SCNR_dBs = torch.empty(n).uniform_(-5, 10) if self.scnr is None else self.scnr * torch.ones(n)
            S = self.generate_target_signal(ranges, velocities, phases, SCNR_dBs)
            # For each target, mark the closest range and Doppler bin.
            for r, v in zip(ranges, velocities):
                r_bin = torch.argmin(torch.abs(self.R - r))
                v_bin = torch.argmin(torch.abs(self.V - v))
                rd_label[r_bin, v_bin] = 1
        return S, C, W, rd_label

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int):
        S, C, W, rd_label = self.gen_frame_and_labels()
        X = S + C + W
        IQ_map = X
        # Range matched filtering using the precomputed steering matrix.
        X_range = self.range_steering_full.conj() @ X     
        S_range = self.range_steering_full.conj() @ S

        # Apply an FFT along slow–time, fftshift and flip to obtain the range–Doppler (RD) map.
        S_RD = torch.fft.fft(S_range, dim=1)
        S_RD = torch.fft.fftshift(S_RD, dim=1)
        clean_RD_map = S_RD.flip(dims=[1])
        X_RD = torch.fft.fft(X_range, dim=1)
        X_RD = torch.fft.fftshift(X_RD, dim=1)
        RD_map = X_RD.flip(dims=[1])
        return S, C, IQ_map, clean_RD_map, RD_map, rd_label

def prepDAFCdata():
    # Create datasets with and without targets
    train_dataset_with_targets = DAFCDataset(num_samples=10240, n_targets=8, random_n_targets=True)
    train_dataset_no_targets = DAFCDataset(num_samples=10240, n_targets=0)

    val_dataset_with_targets = DAFCDataset(num_samples=2048, n_targets=8, random_n_targets=True)
    val_dataset_no_targets = DAFCDataset(num_samples=2048, n_targets=0)

    # Combine datasets
    train_dataset = ConcatDataset([train_dataset_with_targets, train_dataset_no_targets])
    val_dataset = ConcatDataset([val_dataset_with_targets, val_dataset_no_targets])

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True, num_workers=2,
                                pin_memory=torch.cuda.is_available(), persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=256, shuffle=True, num_workers=2,
                            pin_memory=torch.cuda.is_available(), persistent_workers=True)
    return train_loader, val_loader