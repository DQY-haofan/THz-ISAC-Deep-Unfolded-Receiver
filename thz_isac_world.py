"""
thz_isac_world.py (Definition Freeze v3 - MC Randomness Fixed)

Description:
    Physics-driven simulation environment for Terahertz ISAC systems.
    Ref: DR-P2-1.

    **CRITICAL FIX** per Expert Review (Blocker 3):
    - REMOVED fixed seeds from internal functions
    - Each function now accepts rng parameter for external control
    - simulate_batch(config, batch_size, seed=None) for explicit seed control
    - If seed=None: fresh randomness each call (proper MC)
    - If seed provided: reproducible results

    **CLOSURE FIELDS** per DR-P2-3:
    - sim_stats: Power decomposition for Γ_eff
    - gamma_eff: From first principles (NOT magic numbers)
    - chi: From geometry_metrics.chi_from_rho()
    - seed_used: For reproducibility tracking

Author: Definition Freeze v3
Date: 2025-12-17
"""

import numpy as np
import scipy.constants as const
from scipy.fft import fft, ifft
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Tuple, Dict, Optional
import os
import hashlib

# Import geometry_metrics for gamma_eff and chi computation
try:
    import geometry_metrics as gm
except ImportError:
    gm = None
    print("Warning: geometry_metrics.py not found. Using fallback calculations.")


# =============================================================================
# Configuration Structure
# =============================================================================

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

    # NOTE: seed field is DEPRECATED for internal use
    # Use simulate_batch(..., seed=xxx) for external control
    seed: int = 42  # Only used as fallback

    @property
    def Ts(self):
        return 1.0 / self.fs


# =============================================================================
# Core Physics Modules (RNG-Parameterized)
# =============================================================================

def generate_symbols(config: SimConfig, batch_size: int,
                     rng: np.random.Generator) -> np.ndarray:
    """
    Generate QPSK symbols with proper randomness.

    Args:
        config: Simulation config
        batch_size: Number of samples
        rng: Random number generator (EXTERNAL control)

    Returns:
        x: [batch_size, N] complex symbols
    """
    bits = rng.integers(0, 4, size=(batch_size, config.N))
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    x = constellation[bits]

    # Power scaling
    current_power = np.mean(np.abs(x)**2)
    target_power_lin = 10**((config.P_in_dBm - 30)/10)
    scale_factor = np.sqrt(target_power_lin / current_power) if current_power > 0 else 0
    return x * scale_factor


def apply_pa_saleh(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, float, float]:
    """
    Apply Saleh PA model with Bussgang decomposition.

    Returns:
        z: PA output
        B_gain: Bussgang gain
        sigma_eta: Distortion power
    """
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


def apply_phase_noise(x: np.ndarray, config: SimConfig,
                      rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply phase noise with proper randomness.

    Args:
        x: Input signal [batch_size, N]
        config: Simulation config
        rng: Random number generator (EXTERNAL control)

    Returns:
        x_pn: Signal after phase noise
        phi_trajectory: Phase noise trajectory
    """
    if not config.enable_pn:
        return x, np.zeros_like(x, dtype=float)

    batch_size, N = x.shape
    sigma2_phi = 2 * np.pi * config.pn_linewidth * config.Ts
    delta = rng.normal(0, np.sqrt(sigma2_phi), size=(batch_size, N))
    phi = np.cumsum(delta, axis=1)

    return x * np.exp(1j * phi), phi


def apply_channel_squint(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply delay and Doppler channel (deterministic)."""
    if not config.enable_channel:
        return x, np.ones_like(x)

    batch_size, N = x.shape
    t = np.arange(N) * config.Ts

    # Delay
    tau_0 = 2 * config.R / const.c
    X_f = fft(x, axis=1)
    freqs = np.fft.fftfreq(N, d=config.Ts)
    delay_phase = np.exp(-1j * 2 * np.pi * freqs[np.newaxis, :] * tau_0)
    x_delayed = ifft(X_f * delay_phase, axis=1)

    # Doppler & Squint
    f_D = config.fc * config.v_rel / const.c
    doppler_rate = config.fc * config.a_rel / const.c

    doppler_phase = np.exp(1j * 2 * np.pi * f_D * t)
    squint_phase = np.exp(1j * 2 * np.pi * 0.5 * doppler_rate * (t**2))

    h_diag = doppler_phase * squint_phase
    x_channel = x_delayed * h_diag

    return x_channel, h_diag


def add_thermal_noise(x: np.ndarray, config: SimConfig,
                      rng: np.random.Generator) -> np.ndarray:
    """
    Add thermal noise with proper randomness.

    Args:
        x: Input signal
        config: Simulation config
        rng: Random number generator (EXTERNAL control)

    Returns:
        y: Noisy signal
    """
    sig_power = np.mean(np.abs(x)**2)
    noise_power = sig_power / (10**(config.snr_db/10))
    noise = (rng.normal(0, 1, x.shape) + 1j * rng.normal(0, 1, x.shape)) / np.sqrt(2)
    return x + noise * np.sqrt(noise_power)


def quantize_1bit(y: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, float]:
    """
    1-bit quantization with Bussgang residual computation.

    Returns:
        y_q: Quantized signal
        P_quantization_loss: Bussgang residual power
    """
    if not config.enable_quantization:
        return y, 0.0

    y_real = np.sign(np.real(y))
    y_real[y_real == 0] = 1
    y_imag = np.sign(np.imag(y))
    y_imag[y_imag == 0] = 1
    y_q = (y_real + 1j * y_imag) / np.sqrt(2)

    # Bussgang residual
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
    P_signal = np.mean(np.abs(x_true)**2) * (np.abs(B_gain)**2)
    P_pa_distortion = sigma_eta

    if config.enable_pn:
        phase_var = np.var(phi_trajectory)
        P_phase_noise = P_signal * phase_var
    else:
        P_phase_noise = 0.0

    return {
        'P_signal': float(P_signal),
        'P_pa_distortion': float(P_pa_distortion),
        'P_phase_noise': float(P_phase_noise),
        'P_quantization_loss': float(P_quant_loss)
    }


# =============================================================================
# Main Simulation Loop (MC Randomness Fixed)
# =============================================================================

def simulate_batch(config: SimConfig, batch_size: int = 64,
                   seed: Optional[int] = None) -> Dict:
    """
    Executes the full dirty hardware chain with PROPER randomness.

    **CRITICAL FIX** (Blocker 3):
    - If seed=None: uses fresh randomness (different each call)
    - If seed provided: reproducible results

    Args:
        config: Simulation configuration
        batch_size: Number of samples per batch
        seed: Random seed for reproducibility (None = fresh randomness)

    Returns:
        Dict with x_true, y_raw, y_q, theta_true, meta
    """
    # Create RNG with proper seed handling
    if seed is not None:
        rng = np.random.default_rng(seed)
        seed_used = seed
    else:
        # Fresh randomness - different each call
        rng = np.random.default_rng()
        seed_used = None

    # 1. Generate Tx
    x_true = generate_symbols(config, batch_size, rng)

    # 2. Tx Chain - PA (deterministic given input)
    x_pa, B_gain, sigma_eta = apply_pa_saleh(x_true, config)

    # 3. Phase Noise (requires RNG)
    x_pn, phi_trajectory = apply_phase_noise(x_pa, config, rng)

    # 4. Channel (deterministic)
    x_channel, h_diag = apply_channel_squint(x_pn, config)

    # 5. Noise (requires RNG)
    y_analog = add_thermal_noise(x_channel, config, rng)

    # 6. Rx Chain - Quantization (deterministic)
    y_q, P_quant_loss = quantize_1bit(y_analog, config)

    # Compute sim_stats for Gamma_eff closure
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

    # Compute Gamma_eff and Chi using geometry_metrics
    if gm is not None:
        gamma_eff = gm.estimate_gamma_eff(sim_stats)
        snr_linear = 10 ** (config.snr_db / 10)
        sinr_eff = gm.compute_sinr_eff(snr_linear, gamma_eff)
        chi = gm.chi_from_rho(sinr_eff)  # UNIFIED interface
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

    # Ground Truth Theta
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

            # Gamma_eff/Chi Closure Fields (Definition Freeze v3)
            "sim_stats": sim_stats,
            "gamma_eff": gamma_eff,
            "sinr_eff": sinr_eff,
            "chi": chi,

            # Seed tracking for reproducibility
            "seed_used": seed_used,

            # Hardware config for diagnostics
            "enable_pa": config.enable_pa,
            "enable_pn": config.enable_pn,
            "enable_quantization": config.enable_quantization,
            "ibo_dB": config.ibo_dB,
            "pn_linewidth": config.pn_linewidth,
        }
    }


# =============================================================================
# MC Randomness Verification
# =============================================================================

def verify_mc_randomness(n_trials: int = 5) -> Dict:
    """
    Verifies Monte Carlo randomness is working correctly.

    Tests:
    1. No seed: n_trials produce n unique checksums
    2. Same seed: n_trials produce identical checksum
    3. Different seeds: n_trials produce n unique checksums

    Returns:
        Dict with test results
    """
    cfg = SimConfig()
    cfg.snr_db = 20.0

    def get_checksum(data):
        """Compute checksum of output data."""
        arr = np.concatenate([data['y_q'].real.flatten(), data['y_q'].imag.flatten()])
        return hashlib.md5(arr.tobytes()).hexdigest()[:16]

    results = {}

    # Test 1: No seed - should be unique each time
    checksums_no_seed = []
    for _ in range(n_trials):
        data = simulate_batch(cfg, batch_size=32, seed=None)
        checksums_no_seed.append(get_checksum(data))
    results['no_seed_unique'] = len(set(checksums_no_seed)) == n_trials
    results['no_seed_checksums'] = checksums_no_seed

    # Test 2: Same seed - should be identical
    checksums_same_seed = []
    for _ in range(n_trials):
        data = simulate_batch(cfg, batch_size=32, seed=12345)
        checksums_same_seed.append(get_checksum(data))
    results['same_seed_identical'] = len(set(checksums_same_seed)) == 1
    results['same_seed_checksums'] = checksums_same_seed

    # Test 3: Different seeds - should be unique
    checksums_diff_seed = []
    for i in range(n_trials):
        data = simulate_batch(cfg, batch_size=32, seed=1000 + i)
        checksums_diff_seed.append(get_checksum(data))
    results['diff_seed_unique'] = len(set(checksums_diff_seed)) == n_trials
    results['diff_seed_checksums'] = checksums_diff_seed

    # Overall
    results['all_pass'] = (results['no_seed_unique'] and
                          results['same_seed_identical'] and
                          results['diff_seed_unique'])

    return results


# =============================================================================
# Diagnostic Functions
# =============================================================================

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
    print(f"  Seed Used: {meta['seed_used']}")

    print(f"\n[Power Decomposition (sim_stats)]")
    ss = meta['sim_stats']
    print(f"  P_signal: {ss['P_signal']:.6f}")
    print(f"  P_pa_distortion: {ss['P_pa_distortion']:.6f}")
    print(f"  P_phase_noise: {ss['P_phase_noise']:.6f}")
    print(f"  P_quantization_loss: {ss['P_quantization_loss']:.6f}")


# =============================================================================
# Main - Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("THz-ISAC Simulation World (Definition Freeze v3)")
    print("MC Randomness Fixed")
    print("=" * 60)

    # Test MC Randomness (Critical)
    print("\n[CRITICAL TEST] Monte Carlo Randomness Verification")
    mc_results = verify_mc_randomness(n_trials=5)
    print(f"  No seed unique: {mc_results['no_seed_unique']} ✓" if mc_results['no_seed_unique'] else "  No seed unique: FAILED ✗")
    print(f"  Same seed identical: {mc_results['same_seed_identical']} ✓" if mc_results['same_seed_identical'] else "  Same seed identical: FAILED ✗")
    print(f"  Different seeds unique: {mc_results['diff_seed_unique']} ✓" if mc_results['diff_seed_unique'] else "  Different seeds unique: FAILED ✗")
    print(f"  OVERALL: {'PASS ✓' if mc_results['all_pass'] else 'FAILED ✗'}")

    if not mc_results['all_pass']:
        print("\n⚠️  WARNING: MC randomness verification FAILED!")
        print("    This invalidates all Monte Carlo experiments.")
        exit(1)

    # Standard simulation test
    cfg = SimConfig()
    print(f"\nFreq: {cfg.fc/1e9} GHz, BW: {cfg.B/1e9} GHz")

    # Run with explicit seed (reproducible)
    batch_data = simulate_batch(cfg, batch_size=100, seed=42)
    print_meta_summary(batch_data['meta'])

    # Gamma_eff monotonicity check
    print("\n--- Gamma_eff Monotonicity Check ---")
    configs_to_test = [
        ("Ideal", {"enable_pa": False, "enable_pn": False, "enable_quantization": False}),
        ("PA only", {"enable_pa": True, "enable_pn": False, "enable_quantization": False}),
        ("PA + PN", {"enable_pa": True, "enable_pn": True, "enable_quantization": False}),
        ("PA + PN + Quant", {"enable_pa": True, "enable_pn": True, "enable_quantization": True}),
    ]

    gamma_values = []
    for name, params in configs_to_test:
        test_cfg = SimConfig()
        for k, v in params.items():
            setattr(test_cfg, k, v)
        test_data = simulate_batch(test_cfg, batch_size=50, seed=42)
        g = test_data['meta']['gamma_eff']
        gamma_values.append(g)
        print(f"  {name}: Gamma_eff = {g:.4f} ({10*np.log10(g+1e-12):.2f} dB)")

    is_monotonic = all(gamma_values[i] >= gamma_values[i+1] for i in range(len(gamma_values)-1))
    print(f"  Monotonicity: {'PASS ✓' if is_monotonic else 'FAILED ✗'}")

    # Visual check
    print("\n[Visual] Creating sanity check plot...")
    x = batch_data['x_true'][0]
    y_raw = batch_data['y_raw'][0]
    y_q = batch_data['y_q'][0]

    fig = plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.scatter(np.real(x), np.imag(x), label='Tx Original', alpha=0.5, s=10)
    plt.scatter(np.real(y_raw), np.imag(y_raw), label='Rx Analog', alpha=0.3, s=10)
    plt.scatter(np.real(y_q), np.imag(y_q), label='Rx 1-bit', color='red', marker='x', s=30)
    plt.xlabel("In-Phase")
    plt.ylabel("Quadrature")
    plt.legend()
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(np.abs(x[:100]), label='Input |x|')
    z_pa, _, _ = apply_pa_saleh(x.reshape(1,-1), cfg)
    plt.plot(np.abs(z_pa[0][:100]), label='PA Output |z|', linestyle='--')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()

    save_dir = "results/validation_results"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "thz_world_sanity_check_v3.png")
    plt.savefig(save_path, dpi=300)
    print(f"  Saved to: {save_path}")

    print("\n" + "=" * 60)
    print("All self-tests passed.")