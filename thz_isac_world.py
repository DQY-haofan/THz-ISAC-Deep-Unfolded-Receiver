"""
thz_isac_world.py (MC Randomness Fixed - v3.1)

CRITICAL FIX (Blocker 3): External seed control for Monte Carlo validity
- simulate_batch() now accepts optional `seed` parameter
- All random functions accept `rng: np.random.Generator` parameter
- If seed=None: fresh randomness each call (for training diversity)
- If seed=xxx: reproducible results (for experiments)

Usage:
    # Training - unique data each batch
    data = simulate_batch(config, batch_size=64, seed=None)

    # Experiments - reproducible
    data = simulate_batch(config, batch_size=64, seed=12345)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import hashlib

try:
    import geometry_metrics as gm
except ImportError:
    gm = None
    print("[Warning] geometry_metrics not found, using fallback calculations")


# --- Configuration ---

@dataclass
class SimConfig:
    # System
    fc: float = 300e9           # 300 GHz
    N: int = 1024
    fs: float = 10e9
    P_in_dBm: float = -10.0

    # PA - Saleh Model
    enable_pa: bool = True
    alpha_a: float = 2.1587
    beta_a: float = 1.1517
    alpha_phi: float = 4.0033
    beta_phi: float = 9.1040
    ibo_dB: float = 3.0

    # Phase Noise
    enable_pn: bool = True
    pn_linewidth: float = 100e3

    # Channel Geometry
    enable_channel: bool = True
    R: float = 500e3
    v_rel: float = 7.5e3
    a_rel: float = 10.0

    # Receiver
    enable_quantization: bool = True
    snr_db: float = 20.0

    # Random Seed (DEPRECATED - use simulate_batch(seed=xxx) instead)
    seed: int = 42

    @property
    def Ts(self):
        return 1.0 / self.fs


# --- Core Physics Modules (with RNG parameter) ---

def generate_symbols(config: SimConfig, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate QPSK symbols with external RNG."""
    bits = rng.integers(0, 4, size=(batch_size, config.N))
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    x = constellation[bits]

    current_power = np.mean(np.abs(x)**2)
    target_power_lin = 10**((config.P_in_dBm - 30)/10)
    scale_factor = np.sqrt(target_power_lin / current_power) if current_power > 0 else 0
    return x * scale_factor


def apply_pa_saleh(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, float, float]:
    """Apply Saleh PA model (deterministic, no RNG needed)."""
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


def apply_phase_noise(x: np.ndarray, config: SimConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Apply phase noise with external RNG."""
    if not config.enable_pn:
        return x, np.zeros(config.N)

    sigma = np.sqrt(2 * np.pi * config.pn_linewidth * config.Ts)
    increments = rng.normal(0, sigma, config.N)
    phi = np.cumsum(increments)

    x_out = x * np.exp(1j * phi)
    return x_out, phi


def apply_channel_squint(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """Apply channel with beam squint (deterministic)."""
    if not config.enable_channel:
        h = np.ones(config.N)
        return x * h, h

    t = np.arange(config.N) * config.Ts
    tau = config.R / 3e8
    tau_dot = config.v_rel / 3e8
    tau_ddot = config.a_rel / 3e8

    phase = 2 * np.pi * config.fc * (tau + tau_dot * t + 0.5 * tau_ddot * t**2)
    h = np.exp(-1j * phase)

    return x * h, h


def add_thermal_noise(x: np.ndarray, config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Add thermal noise with external RNG."""
    signal_power = np.mean(np.abs(x)**2)
    snr_lin = 10 ** (config.snr_db / 10)
    noise_power = signal_power / snr_lin

    noise = rng.normal(0, np.sqrt(noise_power/2), x.shape) + \
            1j * rng.normal(0, np.sqrt(noise_power/2), x.shape)

    return x + noise


def quantize_1bit(y: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, float]:
    """1-bit quantization (deterministic)."""
    if not config.enable_quantization:
        return y, 0.0

    y_q = np.sign(np.real(y)) + 1j * np.sign(np.imag(y))
    y_q = y_q / np.sqrt(2)

    # Bussgang loss
    rho = np.sqrt(2 / np.pi)
    d = y_q - rho * y / (np.std(y) + 1e-12)
    P_quant_loss = np.mean(np.abs(d)**2)

    return y_q, P_quant_loss


# --- Power Decomposition ---

def compute_sim_stats(x_true, x_pa, x_pn, phi_trajectory, y_analog, B_gain, sigma_eta, P_quant_loss, config) -> Dict:
    """Compute power decomposition for Gamma_eff calculation."""
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


# --- Main Simulation Loop ---

def simulate_batch(config: SimConfig, batch_size: int = 64, seed: Optional[int] = None) -> Dict:
    """
    Executes the full dirty hardware chain with external seed control.

    CRITICAL FIX (Blocker 3):
    - seed=None: Fresh randomness each call (for training)
    - seed=xxx: Reproducible results (for experiments)

    Args:
        config: SimConfig instance
        batch_size: Number of samples
        seed: Optional seed for reproducibility. If None, uses fresh random state.

    Returns:
        Dictionary with x_true, y_q, theta_true, meta (including gamma_eff, chi, seed_used)
    """
    # Create RNG with external seed control
    if seed is None:
        rng = np.random.default_rng()  # Fresh random state
        seed_used = -1  # Indicates non-reproducible
    else:
        rng = np.random.default_rng(seed)
        seed_used = seed

    # 1. Generate Tx (with RNG)
    x_true = generate_symbols(config, batch_size, rng)

    # 2. Tx Chain - PA (deterministic)
    x_pa, B_gain, sigma_eta = apply_pa_saleh(x_true, config)

    # 3. Phase Noise (with RNG)
    x_pn, phi_trajectory = apply_phase_noise(x_pa, config, rng)

    # 4. Channel (deterministic)
    x_channel, h_diag = apply_channel_squint(x_pn, config)

    # 5. Noise (with RNG)
    y_analog = add_thermal_noise(x_channel, config, rng)

    # 6. Rx Chain - Quantization (deterministic)
    y_q, P_quant_loss = quantize_1bit(y_analog, config)

    # --- Compute sim_stats for Gamma_eff closure ---
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
        chi = gm.chi_from_rho(sinr_eff)
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

            # Gamma_eff/Chi Closure Fields
            "sim_stats": sim_stats,
            "gamma_eff": gamma_eff,
            "sinr_eff": sinr_eff,
            "chi": chi,

            # Seed tracking (for reproducibility verification)
            "seed_used": seed_used,

            # Hardware config for diagnostics
            "enable_pa": config.enable_pa,
            "enable_pn": config.enable_pn,
            "enable_quantization": config.enable_quantization,
            "ibo_dB": config.ibo_dB,
            "pn_linewidth": config.pn_linewidth,
        }
    }


# --- Verification Functions ---

def verify_mc_randomness(n_trials: int = 5) -> Dict:
    """
    Verify Monte Carlo randomness is working correctly.

    Tests:
    1. No seed → unique data each call
    2. Same seed → identical data
    3. Different seeds → unique data

    Args:
        n_trials: Number of trials for each test (default: 5)
    """
    cfg = SimConfig()
    results = {}

    # Test 1: No seed - should be unique each call
    checksums_no_seed = []
    for _ in range(n_trials):
        data = simulate_batch(cfg, batch_size=16, seed=None)
        cs = hashlib.md5(data['y_q'].tobytes()).hexdigest()[:8]
        checksums_no_seed.append(cs)
    results['no_seed_unique'] = len(set(checksums_no_seed)) == n_trials
    results['no_seed_checksums'] = checksums_no_seed

    # Test 2: Same seed - should be identical
    checksums_same_seed = []
    for _ in range(n_trials):
        data = simulate_batch(cfg, batch_size=16, seed=12345)
        cs = hashlib.md5(data['y_q'].tobytes()).hexdigest()[:8]
        checksums_same_seed.append(cs)
    results['same_seed_identical'] = len(set(checksums_same_seed)) == 1
    results['same_seed_checksums'] = checksums_same_seed

    # Test 3: Different seeds - should be unique
    checksums_diff_seed = []
    for i in range(n_trials):
        data = simulate_batch(cfg, batch_size=16, seed=12345 + i)
        cs = hashlib.md5(data['y_q'].tobytes()).hexdigest()[:8]
        checksums_diff_seed.append(cs)
    results['diff_seed_unique'] = len(set(checksums_diff_seed)) == n_trials
    results['diff_seed_checksums'] = checksums_diff_seed

    results['all_pass'] = all([
        results['no_seed_unique'],
        results['same_seed_identical'],
        results['diff_seed_unique']
    ])

    return results


def verify_gamma_eff_monotonicity() -> Dict:
    """Verify Gamma_eff decreases with more hardware impairments."""
    results = {}

    # Ideal (no impairments)
    cfg_ideal = SimConfig(enable_pa=False, enable_pn=False, enable_quantization=False)
    data_ideal = simulate_batch(cfg_ideal, batch_size=64, seed=42)
    results['ideal'] = data_ideal['meta']['gamma_eff']

    # PA only
    cfg_pa = SimConfig(enable_pa=True, enable_pn=False, enable_quantization=False)
    data_pa = simulate_batch(cfg_pa, batch_size=64, seed=42)
    results['pa_only'] = data_pa['meta']['gamma_eff']

    # PA + PN
    cfg_pa_pn = SimConfig(enable_pa=True, enable_pn=True, enable_quantization=False)
    data_pa_pn = simulate_batch(cfg_pa_pn, batch_size=64, seed=42)
    results['pa_pn'] = data_pa_pn['meta']['gamma_eff']

    # PA + PN + Quant
    cfg_all = SimConfig(enable_pa=True, enable_pn=True, enable_quantization=True)
    data_all = simulate_batch(cfg_all, batch_size=64, seed=42)
    results['pa_pn_quant'] = data_all['meta']['gamma_eff']

    # Check monotonicity
    results['monotonic'] = (results['ideal'] >= results['pa_only'] >=
                           results['pa_pn'] >= results['pa_pn_quant'])

    return results


# --- Self-Test ---

if __name__ == "__main__":
    print("=" * 60)
    print("thz_isac_world.py Self-Test (MC Randomness Fixed)")
    print("=" * 60)

    print("\n[CRITICAL TEST] Monte Carlo Randomness Verification")
    mc_results = verify_mc_randomness()
    print(f"  No seed unique: {mc_results['no_seed_unique']} {'✓' if mc_results['no_seed_unique'] else '✗'}")
    print(f"  Same seed identical: {mc_results['same_seed_identical']} {'✓' if mc_results['same_seed_identical'] else '✗'}")
    print(f"  Different seeds unique: {mc_results['diff_seed_unique']} {'✓' if mc_results['diff_seed_unique'] else '✗'}")
    print(f"  OVERALL: {'PASS ✓' if mc_results['all_pass'] else 'FAIL ✗'}")

    print("\n--- Gamma_eff Monotonicity Check ---")
    mono_results = verify_gamma_eff_monotonicity()
    print(f"  Ideal: Gamma_eff = {mono_results['ideal']:.4f} ({10*np.log10(mono_results['ideal']):.2f} dB)")
    print(f"  PA only: Gamma_eff = {mono_results['pa_only']:.4f} ({10*np.log10(mono_results['pa_only']):.2f} dB)")
    print(f"  PA + PN: Gamma_eff = {mono_results['pa_pn']:.4f} ({10*np.log10(mono_results['pa_pn']):.2f} dB)")
    print(f"  PA + PN + Quant: Gamma_eff = {mono_results['pa_pn_quant']:.4f} ({10*np.log10(mono_results['pa_pn_quant']):.2f} dB)")
    print(f"  Monotonicity: {'PASS ✓' if mono_results['monotonic'] else 'FAIL ✗'}")

    print("\n" + "=" * 60)
    print("Self-Test Complete")
    print("=" * 60)