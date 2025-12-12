"""
thz_isac_world.py (Ultimate Fixed Version)

Description:
    Physics-driven simulation environment for Terahertz ISAC systems.
    Ref: DR-P2-1.

    Status:
    - [x] Includes 'theta_true' fix for training.
    - [x] Restored full visualization (Constellation + PA Compression).
    - [x] Auto-save plots to results folder.
"""

import numpy as np
import scipy.constants as const
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import os

# --- Configuration Structure ---

@dataclass
class SimConfig:
    """Simulation Configuration container."""
    # System Parameters
    fc: float = 300e9          # 300 GHz
    B: float = 20e9            # 20 GHz
    fs: float = 25e9           # 25 GSps
    N: int = 1024              # Block length

    # Power Amplifier
    enable_pa: bool = True
    P_in_dBm: float = 10.0
    ibo_dB: float = 3.0
    alpha_a: float = 2.15
    beta_a: float = 1.15
    alpha_phi: float = 4.0
    beta_phi: float = 2.1

    # Phase Noise
    enable_pn: bool = True
    pn_linewidth: float = 100e3
    pn_jitter: float = 50e-15

    # Channel Geometry
    enable_channel: bool = True
    R: float = 500e3            # 500 km
    v_rel: float = 7.5e3        # 7.5 km/s
    a_rel: float = 10.0         # 10 m/s^2

    # Receiver
    enable_quantization: bool = True
    snr_db: float = 20.0

    # Random Seed
    seed: int = 42

    @property
    def Ts(self):
        return 1.0 / self.fs

# --- Core Physics Modules ---

def generate_symbols(config: SimConfig, batch_size: int = 1) -> np.ndarray:
    rng = np.random.default_rng(config.seed)
    bits = rng.integers(0, 4, size=(batch_size, config.N))
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    x = constellation[bits]

    current_power = np.mean(np.abs(x)**2)
    target_power_lin = 10**((config.P_in_dBm - 30)/10)
    scale_factor = np.sqrt(target_power_lin / current_power) if current_power > 0 else 0
    return x * scale_factor

def apply_pa_saleh(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, float, float]:
    if not config.enable_pa:
        return x, 1.0, 0.0

    r = np.abs(x)
    phi = np.angle(x)

    A_r = (config.alpha_a * r) / (1 + config.beta_a * (r**2))
    Phi_r = (config.alpha_phi * (r**2)) / (1 + config.beta_phi * (r**2))

    z = A_r * np.exp(1j * (phi + Phi_r))

    # Bussgang decomposition
    num = np.mean(z * np.conj(x))
    den = np.mean(np.abs(x)**2)
    B_est = num / (den + 1e-12)

    eta = z - B_est * x
    sigma_eta = np.mean(np.abs(eta)**2)

    return z, B_est, sigma_eta

def apply_phase_noise(x: np.ndarray, config: SimConfig) -> np.ndarray:
    if not config.enable_pn:
        return x

    batch_size, N = x.shape
    rng = np.random.default_rng(config.seed + 1)
    sigma2_phi = 2 * np.pi * config.pn_linewidth * config.Ts
    delta = rng.normal(0, np.sqrt(sigma2_phi), size=(batch_size, N))
    phi = np.cumsum(delta, axis=1)
    return x * np.exp(1j * phi)

def apply_channel_squint(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    if not config.enable_channel:
        return x, np.ones_like(x)

    batch_size, N = x.shape
    t = np.arange(N) * config.Ts

    # 1. Delay
    tau_0 = 2 * config.R / const.c
    X_f = fft(x, axis=1)
    freqs = np.fft.fftfreq(N, d=config.Ts)
    delay_phase = np.exp(-1j * 2 * np.pi * freqs[np.newaxis, :] * tau_0)
    x_delayed = ifft(X_f * delay_phase, axis=1)

    # 2. Doppler & Squint
    f_D = config.fc * config.v_rel / const.c
    doppler_rate = config.fc * config.a_rel / const.c

    doppler_phase = np.exp(1j * 2 * np.pi * f_D * t)
    squint_phase = np.exp(1j * 2 * np.pi * 0.5 * doppler_rate * (t**2))

    h_diag = doppler_phase * squint_phase
    x_channel = x_delayed * h_diag

    return x_channel, h_diag

def add_thermal_noise(x: np.ndarray, config: SimConfig) -> np.ndarray:
    rng = np.random.default_rng(config.seed + 2)
    sig_power = np.mean(np.abs(x)**2)
    noise_power = sig_power / (10**(config.snr_db/10))
    noise = (rng.normal(0, 1, x.shape) + 1j * rng.normal(0, 1, x.shape)) / np.sqrt(2)
    return x + noise * np.sqrt(noise_power)

def quantize_1bit(y: np.ndarray, config: SimConfig) -> np.ndarray:
    if not config.enable_quantization:
        return y
    y_real = np.sign(np.real(y)); y_real[y_real==0] = 1
    y_imag = np.sign(np.imag(y)); y_imag[y_imag==0] = 1
    return (y_real + 1j * y_imag) / np.sqrt(2)

# --- Main Simulation Loop ---

def simulate_batch(config: SimConfig, batch_size: int = 64) -> Dict:
    """Executes the full dirty hardware chain."""

    # 1. Generate Tx
    x_true = generate_symbols(config, batch_size)

    # 2. Tx Chain
    x_pa, B_gain, sigma_eta = apply_pa_saleh(x_true, config)

    # 3. Phase Noise
    x_pn = apply_phase_noise(x_pa, config)

    # 4. Channel
    x_channel, h_diag = apply_channel_squint(x_pn, config)

    # 5. Noise
    y_analog = add_thermal_noise(x_channel, config)

    # 6. Rx Chain
    y_q = quantize_1bit(y_analog, config)

    # --- Fix: Construct Ground Truth Theta ---
    theta_val = np.array([config.R, config.v_rel, config.a_rel])
    theta_true = np.tile(theta_val, (batch_size, 1))

    return {
        "x_true": x_true,
        "y_raw": y_analog,
        "y_q": y_q,
        "theta_true": theta_true,  # Required for training
        "meta": {
            "B_gain": B_gain,
            "sigma_eta": sigma_eta,
            "h_diag": h_diag,
            "snr_linear": 10**(config.snr_db/10)
        }
    }

if __name__ == "__main__":
    # --- Visualization & Sanity Check ---
    cfg = SimConfig()
    print(f"--- THz-ISAC Simulation World (DR-P2-1) ---")
    print(f"Freq: {cfg.fc/1e9} GHz, BW: {cfg.B/1e9} GHz")

    # Run a small batch
    batch_data = simulate_batch(cfg, batch_size=100)

    # Extract data
    x = batch_data['x_true'][0]
    y_raw = batch_data['y_raw'][0]
    y_q = batch_data['y_q'][0]

    # Basic Stats
    print("Output Keys:", batch_data.keys())
    print("Theta Shape:", batch_data['theta_true'].shape)

    # Plotting
    fig = plt.figure(figsize=(12, 5))

    # Subplot 1: Constellation
    plt.subplot(1, 2, 1)
    plt.scatter(np.real(x), np.imag(x), label='Tx Original', alpha=0.5, s=10)
    plt.scatter(np.real(y_raw), np.imag(y_raw), label='Rx Analog (Distorted)', alpha=0.3, s=10)
    plt.scatter(np.real(y_q), np.imag(y_q), label='Rx 1-bit', color='red', marker='x', s=30)
    plt.title("Constellation: Tx -> Dirty HW -> 1-bit Rx")
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.legend()
    plt.grid(True)

    # Subplot 2: PA Compression (Re-running PA function to isolate effect)
    plt.subplot(1, 2, 2)
    plt.plot(np.abs(x[:100]), label='Input |x|')
    # Use PA output from a temp call to see pure PA effect
    z_pa, _, _ = apply_pa_saleh(x.reshape(1,-1), cfg)
    plt.plot(np.abs(z_pa[0][:100]), label='PA Output |z|', linestyle='--')
    plt.title("PA Amplitude Compression (First 100 samples)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    # Save Image
    save_dir = "results/validation_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "thz_world_sanity_check.png")
    plt.savefig(save_path, dpi=300)
    print(f"\n[Visual] Sanity check plot saved to: {save_path}")

    plt.show()