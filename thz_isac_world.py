"""
thz_isac_world.py (Gamma_eff/Chi Closure Version)

Description:
    Physics-driven simulation environment for Terahertz ISAC systems.
    Ref: DR-P2-1.

    **KEY UPDATE** per DR-P2-3 注意事项:
    - Added sim_stats with power decomposition
    - Computes gamma_eff from first principles
    - Computes chi using frozen definition
    - Exports sinr_eff for closed-loop integration

    Status:
    - [x] Includes 'theta_true' fix for training.
    - [x] Full visualization (Constellation + PA Compression).
    - [x] Auto-save plots to results folder.
    - [x] NEW: Gamma_eff & Chi closure.
"""

import numpy as np
import scipy.constants as const
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import os

# Import geometry_metrics for gamma_eff and chi computation
try:
    import geometry_metrics as gm
except ImportError:
    gm = None
    print("Warning: geometry_metrics.py not found. Using fallback calculations.")

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

def apply_phase_noise(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply phase noise and return both output and phase trajectory.

    Returns:
        x_pn: Signal after phase noise
        phi_trajectory: Phase noise trajectory (for power calculation)
    """
    if not config.enable_pn:
        return x, np.zeros_like(x, dtype=float)

    batch_size, N = x.shape
    rng = np.random.default_rng(config.seed + 1)
    sigma2_phi = 2 * np.pi * config.pn_linewidth * config.Ts
    delta = rng.normal(0, np.sqrt(sigma2_phi), size=(batch_size, N))
    phi = np.cumsum(delta, axis=1)

    return x * np.exp(1j * phi), phi

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

def quantize_1bit(y: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, float]:
    """
    1-bit quantization with Bussgang residual computation.

    Returns:
        y_q: Quantized signal
        P_quantization_loss: Bussgang residual power (for Gamma_eff)
    """
    if not config.enable_quantization:
        return y, 0.0

    y_real = np.sign(np.real(y))
    y_real[y_real == 0] = 1
    y_imag = np.sign(np.imag(y))
    y_imag[y_imag == 0] = 1
    y_q = (y_real + 1j * y_imag) / np.sqrt(2)

    # Compute Bussgang residual for quantization loss
    # A_q = E[y_q * conj(y)] / E[|y|^2]
    # q = y_q - A_q * y
    # P_quantization_loss = E[|q|^2]
    A_q = np.mean(y_q * np.conj(y)) / (np.mean(np.abs(y)**2) + 1e-12)
    q_residual = y_q - A_q * y
    P_quantization_loss = np.mean(np.abs(q_residual)**2)

    return y_q, P_quantization_loss


def compute_sim_stats(x_true: np.ndarray, x_pa: np.ndarray, x_pn: np.ndarray,
                      phi_trajectory: np.ndarray, y_analog: np.ndarray,
                      B_gain: float, sigma_eta: float, P_quant_loss: float,
                      config: SimConfig) -> Dict:
    """
    Computes power decomposition statistics for Gamma_eff calculation.

    This is the FIRST PRINCIPLES computation per DR-P2-2.5.

    Returns:
        sim_stats: Dictionary with P_signal, P_pa_distortion, P_phase_noise, P_quantization_loss
    """
    # P_signal: Main signal power (through effective channel)
    # Use E[|B * x|^2] as the reference signal power
    P_signal = np.mean(np.abs(x_true)**2) * (np.abs(B_gain)**2)

    # P_pa_distortion: Already computed in apply_pa_saleh as sigma_eta
    P_pa_distortion = sigma_eta

    # P_phase_noise: E[|x_after_pn - x_before_pn|^2] equivalent
    # For small angles: P_pn ≈ P_signal * var(phi)
    # Total phase variance = sigma^2 * N (for Wiener process)
    if config.enable_pn:
        # Direct computation: variance of phase rotation effect
        phase_var = np.var(phi_trajectory)
        # Approximate multiplicative noise power
        P_phase_noise = P_signal * phase_var
    else:
        P_phase_noise = 0.0

    # P_quantization_loss: Bussgang residual (already computed)
    # Note: This is the effective noise from 1-bit quantization

    return {
        'P_signal': float(P_signal),
        'P_pa_distortion': float(P_pa_distortion),
        'P_phase_noise': float(P_phase_noise),
        'P_quantization_loss': float(P_quant_loss)
    }


# --- Main Simulation Loop ---

def simulate_batch(config: SimConfig, batch_size: int = 64) -> Dict:
    """
    Executes the full dirty hardware chain.

    **KEY UPDATE**: Now includes sim_stats, gamma_eff, chi in meta.
    """

    # 1. Generate Tx
    x_true = generate_symbols(config, batch_size)

    # 2. Tx Chain - PA
    x_pa, B_gain, sigma_eta = apply_pa_saleh(x_true, config)

    # 3. Phase Noise (now also returns trajectory)
    x_pn, phi_trajectory = apply_phase_noise(x_pa, config)

    # 4. Channel
    x_channel, h_diag = apply_channel_squint(x_pn, config)

    # 5. Noise
    y_analog = add_thermal_noise(x_channel, config)

    # 6. Rx Chain - Quantization (now also returns loss)
    y_q, P_quant_loss = quantize_1bit(y_analog, config)

    # --- NEW: Compute sim_stats for Gamma_eff closure ---
    sim_stats = compute_sim_stats(
        x_true=x_true,
        x_pa=x_pa,
        x_pn=x_pn,
        phi_trajectory=phi_trajectory,
        y_analog=y_analog,
        B_gain=B_gain,
        sigma_eta=sigma_eta,
        P_quant_loss=P_quant_loss,
        config=config
    )

    # Compute Gamma_eff using geometry_metrics
    if gm is not None:
        gamma_eff = gm.estimate_gamma_eff(sim_stats)
        snr_linear = 10 ** (config.snr_db / 10)
        sinr_eff = gm.compute_sinr_eff(snr_linear, gamma_eff)
        chi = gm.approx_chi(sinr_eff)
    else:
        # Fallback calculation
        p_sig = sim_stats['P_signal']
        p_total_dist = (sim_stats['P_pa_distortion'] +
                        sim_stats['P_phase_noise'] +
                        sim_stats['P_quantization_loss'])
        gamma_eff = p_sig / (p_total_dist + 1e-12) if p_total_dist > 1e-12 else 1e9
        snr_linear = 10 ** (config.snr_db / 10)
        sinr_eff = 1.0 / ((1.0/snr_linear) + (1.0/gamma_eff))
        kappa = 1.0 - 2.0/np.pi
        chi = (2.0/np.pi) / (1.0 + kappa * sinr_eff)

    # --- Construct Ground Truth Theta ---
    theta_val = np.array([config.R, config.v_rel, config.a_rel])
    theta_true = np.tile(theta_val, (batch_size, 1))

    return {
        "x_true": x_true,
        "y_raw": y_analog,
        "y_q": y_q,
        "theta_true": theta_true,
        "meta": {
            # Original fields
            "B_gain": B_gain,
            "sigma_eta": sigma_eta,
            "h_diag": h_diag,
            "snr_linear": 10**(config.snr_db/10),
            "snr_db": config.snr_db,

            # NEW: Gamma_eff/Chi Closure Fields
            "sim_stats": sim_stats,
            "gamma_eff": gamma_eff,
            "sinr_eff": sinr_eff,
            "chi": chi,

            # Hardware config for diagnostics
            "enable_pa": config.enable_pa,
            "enable_pn": config.enable_pn,
            "enable_quantization": config.enable_quantization,
            "ibo_dB": config.ibo_dB,
            "pn_linewidth": config.pn_linewidth,
        }
    }


# --- Diagnostic Functions ---

def print_meta_summary(meta: Dict) -> None:
    """Pretty-print the meta dictionary for debugging."""
    print("\n--- Meta Summary ---")
    print(f"SNR: {meta['snr_db']:.1f} dB ({meta['snr_linear']:.2f} linear)")
    print(f"B_gain: {meta['B_gain']:.4f}")
    print(f"sigma_eta (PA distortion): {meta['sigma_eta']:.6f}")

    print(f"\n[Gamma_eff/Chi Closure]")
    print(f"  Gamma_eff: {meta['gamma_eff']:.4f} ({10*np.log10(meta['gamma_eff']+1e-12):.2f} dB)")
    print(f"  SINR_eff: {meta['sinr_eff']:.4f} ({10*np.log10(meta['sinr_eff']+1e-12):.2f} dB)")
    print(f"  Chi: {meta['chi']:.4f}")

    print(f"\n[Power Decomposition (sim_stats)]")
    ss = meta['sim_stats']
    print(f"  P_signal: {ss['P_signal']:.6f}")
    print(f"  P_pa_distortion: {ss['P_pa_distortion']:.6f}")
    print(f"  P_phase_noise: {ss['P_phase_noise']:.6f}")
    print(f"  P_quantization_loss: {ss['P_quantization_loss']:.6f}")


if __name__ == "__main__":
    # --- Visualization & Sanity Check ---
    print("=" * 60)
    print("THz-ISAC Simulation World (Gamma_eff/Chi Closure Version)")
    print("=" * 60)

    cfg = SimConfig()
    print(f"Freq: {cfg.fc/1e9} GHz, BW: {cfg.B/1e9} GHz")

    # Run a small batch
    batch_data = simulate_batch(cfg, batch_size=100)

    # Extract data
    x = batch_data['x_true'][0]
    y_raw = batch_data['y_raw'][0]
    y_q = batch_data['y_q'][0]
    meta = batch_data['meta']

    # Print meta summary
    print_meta_summary(meta)

    # Basic Stats
    print("\n--- Output Structure ---")
    print("Output Keys:", batch_data.keys())
    print("Theta Shape:", batch_data['theta_true'].shape)

    # --- Gamma_eff monotonicity check ---
    print("\n--- Gamma_eff Monotonicity Check ---")
    print("(Gamma_eff should DECREASE when enabling PA/PN/Quant)")

    configs_to_test = [
        ("Ideal (no impairments)", {"enable_pa": False, "enable_pn": False, "enable_quantization": False}),
        ("PA only", {"enable_pa": True, "enable_pn": False, "enable_quantization": False}),
        ("PA + PN", {"enable_pa": True, "enable_pn": True, "enable_quantization": False}),
        ("PA + PN + Quant (Full)", {"enable_pa": True, "enable_pn": True, "enable_quantization": True}),
    ]

    gamma_values = []
    for name, params in configs_to_test:
        test_cfg = SimConfig()
        for k, v in params.items():
            setattr(test_cfg, k, v)
        test_data = simulate_batch(test_cfg, batch_size=50)
        g = test_data['meta']['gamma_eff']
        gamma_values.append(g)
        print(f"  {name}: Gamma_eff = {g:.4f} ({10*np.log10(g+1e-12):.2f} dB)")

    # Check monotonicity
    is_monotonic = all(gamma_values[i] >= gamma_values[i+1] for i in range(len(gamma_values)-1))
    print(f"  Monotonicity Check: {'PASSED ✓' if is_monotonic else 'FAILED ✗'}")

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

    # Subplot 2: PA Compression
    plt.subplot(1, 2, 2)
    plt.plot(np.abs(x[:100]), label='Input |x|')
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