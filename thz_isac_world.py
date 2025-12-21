"""
thz_isac_world_v2.py (Wideband Delay Model - v4.1)

NEW in v4.1 (Expert Review):
1. Multi-bit ADC quantization support (1-8 bits)
2. AGC-normalized uniform quantization for bits > 1
3. Numerical FIM/CRLB computation for τ

Based on v4.0 Wideband Delay Model.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Tuple, Optional
import hashlib

try:
    import geometry_metrics as gm
except ImportError:
    gm = None


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class SimConfig:
    """Simulation configuration with wideband delay model."""

    # System parameters
    fc: float = 300e9           # Carrier frequency: 300 GHz
    fs: float = 10e9            # Sampling rate: 10 GHz (bandwidth B)
    N: int = 1024               # Number of symbols
    P_in_dBm: float = -10.0     # Input power

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
    R: float = 500e3            # Range [m]
    v_rel: float = 7.5e3        # Velocity [m/s]
    a_rel: float = 10.0         # Acceleration [m/s²]

    coarse_acquisition_error_samples: float = 0.0
    phi0_random: bool = True

    # Receiver
    enable_quantization: bool = True
    snr_db: float = 20.0

    # NEW: ADC bits (1 = sign quantization, 2-8 = uniform quantization)
    adc_bits: int = 1

    @property
    def Ts(self):
        return 1.0 / self.fs

    @property
    def wavelength(self):
        return 3e8 / self.fc

    @property
    def delay_resolution(self):
        return 1.0 / self.fs

    @property
    def range_resolution(self):
        return 3e8 * self.delay_resolution


# =============================================================================
# Core Physics Modules
# =============================================================================

def generate_symbols(config: SimConfig, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate QPSK symbols with proper power scaling."""
    bits = rng.integers(0, 4, size=(batch_size, config.N))
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    x = constellation[bits]

    current_power = np.mean(np.abs(x)**2)
    target_power_lin = 10**((config.P_in_dBm - 30)/10)
    scale_factor = np.sqrt(target_power_lin / current_power) if current_power > 0 else 0
    return x * scale_factor


def apply_pa_saleh(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, complex, float]:
    """Apply Saleh PA model."""
    if not config.enable_pa:
        return x, 1.0, 0.0

    r = np.abs(x)
    phi = np.angle(x)

    A_r = (config.alpha_a * r) / (1 + config.beta_a * (r**2))
    Phi_r = (config.alpha_phi * (r**2)) / (1 + config.beta_phi * (r**2))

    z = A_r * np.exp(1j * (phi + Phi_r))

    num = np.mean(z * np.conj(x))
    den = np.mean(np.abs(x)**2)
    B_est = num / (den + 1e-12)

    eta = z - B_est * x
    sigma_eta = np.mean(np.abs(eta)**2)

    return z, B_est, sigma_eta


def apply_phase_noise(x: np.ndarray, config: SimConfig, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    """Apply phase noise (Wiener process)."""
    if not config.enable_pn:
        return x, np.zeros(config.N)

    sigma = np.sqrt(2 * np.pi * config.pn_linewidth * config.Ts)
    increments = rng.normal(0, sigma, config.N)
    phi = np.cumsum(increments)

    x_out = x * np.exp(1j * phi)
    return x_out, phi


# =============================================================================
# Wideband Delay Channel Model
# =============================================================================

def wideband_delay_operator(x: np.ndarray, tau: float, fs: float) -> np.ndarray:
    """
    Apply wideband delay operator D(τ) to signal.

    D(τ)[x] = ifft(fft(x) × exp(-j × 2π × f_k × τ))
    """
    is_batched = x.ndim == 2
    if not is_batched:
        x = x[np.newaxis, :]

    N = x.shape[1]
    f_k = np.fft.fftfreq(N, d=1.0/fs)
    H_delay = np.exp(-1j * 2 * np.pi * f_k * tau)

    X = np.fft.fft(x, axis=1)
    Y = X * H_delay[np.newaxis, :]
    y = np.fft.ifft(Y, axis=1)

    if not is_batched:
        y = y[0]

    return y


def doppler_phase_operator(N: int, fs: float, fc: float, v: float, a: float) -> np.ndarray:
    """
    Compute Doppler/acceleration phase (WITHOUT absolute R term!).

    p(t) = exp(-j × 2π × fc × (v/c × t + 0.5 × a/c × t²))
    """
    c = 3e8
    Ts = 1.0 / fs
    t = np.arange(N) * Ts

    phase = 2 * np.pi * fc * (v/c * t + 0.5 * a/c * t**2)

    return np.exp(-1j * phase)


def apply_wideband_channel(
    x: np.ndarray,
    config: SimConfig,
    rng: np.random.Generator
) -> Tuple[np.ndarray, Dict]:
    """Apply wideband delay channel model."""
    if not config.enable_channel:
        batch_size = x.shape[0]
        return x, {
            'tau_true': 0.0,
            'tau_coarse': 0.0,
            'tau_res': 0.0,
            'phi0': np.zeros(batch_size),
            'v': 0.0,
            'a': 0.0,
            'p_t': np.ones(config.N, dtype=complex),
        }

    batch_size = x.shape[0]
    N = x.shape[1]
    c = 3e8

    tau_res_base = config.coarse_acquisition_error_samples * config.Ts

    if config.coarse_acquisition_error_samples > 0:
        tau_res_random = rng.uniform(-0.5, 0.5) * config.Ts
        tau_res = tau_res_base + tau_res_random
    else:
        tau_res = 0.0

    tau_true = config.R / c
    tau_coarse = tau_true - tau_res

    if abs(tau_res) > 1e-15:
        y = wideband_delay_operator(x, tau_res, config.fs)
    else:
        y = x.copy()

    p_t = doppler_phase_operator(N, config.fs, config.fc, config.v_rel, config.a_rel)
    y = y * p_t[np.newaxis, :]

    if config.phi0_random:
        phi0 = rng.uniform(0, 2*np.pi, size=(batch_size, 1))
    else:
        phi0 = np.zeros((batch_size, 1))

    y = y * np.exp(-1j * phi0)

    channel_info = {
        'tau_true': tau_true,
        'tau_coarse': tau_coarse,
        'tau_res': tau_res,
        'tau_res_samples': tau_res * config.fs,
        'phi0': phi0.flatten(),
        'v': config.v_rel,
        'a': config.a_rel,
        'p_t': p_t,
    }

    return y, channel_info


def add_thermal_noise(x: np.ndarray, config: SimConfig, rng: np.random.Generator) -> np.ndarray:
    """Add thermal noise."""
    signal_power = np.mean(np.abs(x)**2)
    snr_lin = 10 ** (config.snr_db / 10)
    noise_power = signal_power / snr_lin

    noise = rng.normal(0, np.sqrt(noise_power/2), x.shape) + \
            1j * rng.normal(0, np.sqrt(noise_power/2), x.shape)

    return x + noise


# =============================================================================
# NEW: Multi-bit Quantization
# =============================================================================

def quantize_1bit(y: np.ndarray) -> np.ndarray:
    """
    1-bit (sign) quantization for complex signal.

    Returns values in {±1 ± 1j}/sqrt(2) for consistent power.
    """
    y_q = np.sign(np.real(y)) + 1j * np.sign(np.imag(y))
    y_q = y_q / np.sqrt(2)
    return y_q


def quantize_uniform(y: np.ndarray, bits: int, kappa: float = 3.5) -> np.ndarray:
    """
    Uniform (mid-rise) quantization for complex signal.

    AGC-normalized: Full scale A = κ × RMS(y)

    Args:
        y: Complex signal [batch, N] or [N]
        bits: Number of bits per real/imag component (1-8)
        kappa: AGC scale factor (default 3.5, covers ~99.9% of Gaussian)

    Returns:
        y_q: Quantized signal (same shape)
    """
    if bits < 1:
        raise ValueError("bits must be >= 1")

    if bits == 1:
        return quantize_1bit(y)

    # AGC: Compute RMS and set full scale
    rms = np.sqrt(np.mean(np.abs(y)**2))
    A = kappa * rms  # Full scale amplitude

    if A < 1e-12:
        return np.zeros_like(y)

    # Number of quantization levels per component
    n_levels = 2 ** bits

    # Quantization step
    delta = 2 * A / n_levels

    # Quantize real and imaginary parts separately
    def quantize_real(x_real):
        # Clip to [-A, A]
        x_clipped = np.clip(x_real, -A, A)

        # Mid-rise quantizer: output levels are ±Δ/2, ±3Δ/2, ...
        # Map to integer: floor((x + A) / Δ)
        idx = np.floor((x_clipped + A) / delta)
        idx = np.clip(idx, 0, n_levels - 1)

        # Map back to quantized value (center of bin)
        x_q = -A + delta * (idx + 0.5)

        return x_q

    y_q_real = quantize_real(np.real(y))
    y_q_imag = quantize_real(np.imag(y))

    y_q = y_q_real + 1j * y_q_imag

    # Normalize to maintain consistent power (optional but recommended)
    # For fair comparison, normalize so E[|y_q|²] ≈ 1
    y_q_power = np.mean(np.abs(y_q)**2)
    if y_q_power > 1e-12:
        y_q = y_q / np.sqrt(y_q_power) * np.sqrt(np.mean(np.abs(y)**2))

    return y_q


def quantize(y: np.ndarray, bits: int, config: SimConfig = None) -> Tuple[np.ndarray, float]:
    """
    Universal quantization function.

    Args:
        y: Complex signal
        bits: Number of bits (1 = sign, 2-8 = uniform)
        config: Optional config for additional parameters

    Returns:
        y_q: Quantized signal
        P_quant_loss: Quantization distortion power
    """
    if bits == 1:
        y_q = quantize_1bit(y)
        # Bussgang loss for 1-bit
        rho = np.sqrt(2 / np.pi)
        d = y_q - rho * y / (np.std(y) + 1e-12)
        P_quant_loss = np.mean(np.abs(d)**2)
    else:
        y_q = quantize_uniform(y, bits)
        # Quantization noise power (approximate)
        P_quant_loss = np.mean(np.abs(y_q - y)**2)

    return y_q, P_quant_loss


# =============================================================================
# Power Decomposition (for Gamma_eff)
# =============================================================================

def compute_sim_stats(x_true, x_pa, x_pn, phi_trajectory, y_analog,
                      B_gain, sigma_eta, P_quant_loss, config) -> Dict:
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


# =============================================================================
# Main Simulation Function
# =============================================================================

def simulate_batch(config: SimConfig, batch_size: int = 64,
                   seed: Optional[int] = None) -> Dict:
    """
    Execute the full dirty hardware chain with wideband delay model.

    Signal flow:
        x → PA → PN → Channel[D(τ), p(t), φ₀] → Noise → Quantize(bits) → y_q
    """
    # Create RNG
    if seed is None:
        rng = np.random.default_rng()
        seed_used = -1
    else:
        rng = np.random.default_rng(seed)
        seed_used = seed

    # 1. Generate Tx symbols
    x_true = generate_symbols(config, batch_size, rng)

    # 2. PA (Tx chain)
    x_pa, B_gain, sigma_eta = apply_pa_saleh(x_true, config)

    # 3. Phase Noise
    x_pn, phi_trajectory = apply_phase_noise(x_pa, config, rng)

    # 4. Wideband Delay Channel
    x_channel, channel_info = apply_wideband_channel(x_pn, config, rng)

    # 5. Thermal Noise
    y_analog = add_thermal_noise(x_channel, config, rng)

    # 6. Quantization (supports 1-8 bits)
    if config.enable_quantization:
        y_q, P_quant_loss = quantize(y_analog, config.adc_bits, config)
    else:
        y_q = y_analog
        P_quant_loss = 0.0

    # Compute statistics
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

    # Compute Gamma_eff
    if gm is not None:
        gamma_eff = gm.estimate_gamma_eff(sim_stats)
        snr_linear = 10 ** (config.snr_db / 10)
        sinr_eff = gm.compute_sinr_eff(snr_linear, gamma_eff)
        chi = gm.chi_from_rho(sinr_eff)
    else:
        p_sig = sim_stats['P_signal']
        p_total_dist = (sim_stats['P_pa_distortion'] +
                        sim_stats['P_phase_noise'] +
                        sim_stats['P_quantization_loss'])
        gamma_eff = p_sig / (p_total_dist + 1e-12) if p_total_dist > 1e-12 else 1e9
        snr_linear = 10 ** (config.snr_db / 10)
        sinr_eff = 1.0 / ((1.0/snr_linear) + (1.0/gamma_eff))
        kappa = 1.0 - 2.0/np.pi
        chi = (2.0/np.pi) / (1.0 + kappa * sinr_eff)

    # Construct Ground Truth Theta
    tau_res = channel_info['tau_res']
    theta_val = np.array([tau_res, config.v_rel, config.a_rel])
    theta_true = np.tile(theta_val, (batch_size, 1))

    theta_R_based = np.tile(
        np.array([config.R, config.v_rel, config.a_rel]),
        (batch_size, 1)
    )

    return {
        "x_true": x_true,
        "y_raw": y_analog,
        "y_q": y_q,
        "theta_true": theta_true,
        "theta_R_based": theta_R_based,
        "meta": {
            "B_gain": B_gain,
            "sigma_eta": sigma_eta,
            "channel_info": channel_info,
            "tau_true": channel_info.get('tau_true', 0.0),
            "tau_coarse": channel_info.get('tau_coarse', 0.0),
            "tau_res": channel_info.get('tau_res', 0.0),
            "tau_res_samples": channel_info.get('tau_res_samples', 0.0),
            "phi0": channel_info.get('phi0', np.zeros(batch_size)),
            "snr_linear": 10**(config.snr_db/10),
            "snr_db": config.snr_db,
            "sim_stats": sim_stats,
            "gamma_eff": gamma_eff,
            "sinr_eff": sinr_eff,
            "chi": chi,
            "seed_used": seed_used,
            "enable_pa": config.enable_pa,
            "enable_pn": config.enable_pn,
            "enable_quantization": config.enable_quantization,
            "ibo_dB": config.ibo_dB,
            "pn_linewidth": config.pn_linewidth,
            "adc_bits": config.adc_bits,
            "delay_resolution": config.delay_resolution,
            "range_resolution": config.range_resolution,
            "wavelength": config.wavelength,
        }
    }


# =============================================================================
# BCRLB Computation (Diagonal Approximation)
# =============================================================================

def compute_bcrlb_diag(config: SimConfig, snr_linear: float,
                       gamma_eff: float) -> np.ndarray:
    """
    Compute diagonal Bayesian Cramér-Rao Lower Bound for [τ, v, a].
    """
    N = config.N
    B = config.fs
    T = N / config.fs
    fc = config.fc
    c = 3e8

    # Effective SNR with quantization loss
    if config.adc_bits == 1:
        quantization_loss = 2 / np.pi  # ~0.64
    else:
        # Higher bits have lower quantization loss
        quantization_loss = 1 - 2**(-2 * config.adc_bits)

    snr_eff = snr_linear * gamma_eff * quantization_loss
    snr_eff = max(snr_eff, 1e-6)

    # BCRLB for delay τ (wideband model)
    bcrlb_tau = 1 / (8 * np.pi**2 * B**2 * N * snr_eff)

    # BCRLB for velocity v
    bcrlb_v = (c / fc)**2 * 3 / (np.pi**2 * T**2 * N * snr_eff)

    # BCRLB for acceleration a
    bcrlb_a = (c / fc)**2 * 180 / (np.pi**2 * T**4 * N * snr_eff)

    return np.array([bcrlb_tau, bcrlb_v, bcrlb_a])


def compute_fim_diag(config: SimConfig, snr_linear: float,
                     gamma_eff: float) -> np.ndarray:
    """Compute diagonal Fisher Information Matrix."""
    bcrlb = compute_bcrlb_diag(config, snr_linear, gamma_eff)
    return 1.0 / (bcrlb + 1e-20)


# =============================================================================
# Numerical FIM for τ (Bussgang Approximation)
# =============================================================================

def compute_numeric_fim_tau_bussgang(config: SimConfig,
                                      n_samples: int = 1000,
                                      epsilon_samples: float = 1e-3) -> Dict:
    """
    Compute numerical FIM for τ using Bussgang approximation.

    Model: y_q ≈ α * y_noisefree + q (Bussgang linearization)

    FIM(τ) = 2 * Re{ (∂m/∂τ)^H * Σ^{-1} * (∂m/∂τ) }

    Where m(τ) = α * y_noisefree(τ)

    Returns dictionary with FIM, CRLB, and diagnostics.
    """
    fs = config.fs
    Ts = 1.0 / fs
    eps_seconds = epsilon_samples * Ts
    N = config.N

    # Bussgang gain for quantization
    if config.adc_bits == 1:
        alpha = np.sqrt(2 / np.pi)
    else:
        # For multi-bit, alpha ≈ 1 (small quantization error)
        alpha = 1.0 - 2**(-2 * config.adc_bits) * 0.1

    # Generate deterministic reference signal
    np.random.seed(42)
    x_ref = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

    # Compute derivative dm/dτ using central difference
    tau_0 = 0.0

    y_plus = wideband_delay_operator(x_ref, tau_0 + eps_seconds, fs)
    y_minus = wideband_delay_operator(x_ref, tau_0 - eps_seconds, fs)

    dm_dtau = alpha * (y_plus - y_minus) / (2 * eps_seconds)

    # Estimate noise variance from analytical model
    snr_linear = 10 ** (config.snr_db / 10)
    signal_power = 1.0  # Normalized
    noise_power = signal_power / snr_linear

    # Add quantization noise (approximate)
    if config.adc_bits == 1:
        # 1-bit: Bussgang distortion
        quant_noise_power = (1 - 2/np.pi) * alpha**2 * signal_power
    else:
        # Multi-bit: small quantization noise
        quant_noise_power = signal_power * 2**(-2 * config.adc_bits)

    sigma2 = noise_power + quant_noise_power

    # FIM for τ (scalar case)
    fim_tau = (2.0 / sigma2) * np.sum(np.abs(dm_dtau) ** 2)

    # CRLB
    crlb_tau = 1.0 / fim_tau if fim_tau > 0 else float('inf')
    crlb_tau_samples = np.sqrt(crlb_tau) * fs

    return {
        'fim_tau': fim_tau,
        'crlb_tau': crlb_tau,
        'crlb_tau_samples': crlb_tau_samples,
        'sigma2': sigma2,
        'alpha': alpha,
        'epsilon_samples': epsilon_samples,
        'snr_db': config.snr_db,
        'adc_bits': config.adc_bits,
    }


# =============================================================================
# Verification Functions
# =============================================================================

def verify_quantization(bits_list: list = None) -> Dict:
    """Verify multi-bit quantization is working correctly."""
    if bits_list is None:
        bits_list = [1, 2, 3, 4, 8]

    results = {}
    N = 1024

    # Generate test signal
    np.random.seed(42)
    y = np.random.randn(N) + 1j * np.random.randn(N)
    y = y / np.std(y)  # Normalize

    for bits in bits_list:
        y_q, P_loss = quantize(y, bits)

        # Compute metrics
        snr_q = 10 * np.log10(np.mean(np.abs(y)**2) / (P_loss + 1e-12))

        results[bits] = {
            'snr_q_db': snr_q,
            'P_loss': P_loss,
            'output_power': np.mean(np.abs(y_q)**2),
        }

        print(f"  {bits}-bit: SNR_q = {snr_q:.1f} dB, P_loss = {P_loss:.4f}")

    return results


def verify_wideband_delay() -> Dict:
    """Verify wideband delay operator is working correctly."""
    results = {}
    N = 1024
    fs = 10e9

    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, N) + 1j * rng.normal(0, 1, N)

    # Test 1: Zero delay
    y_zero = wideband_delay_operator(x, 0.0, fs)
    results['zero_delay_error'] = np.max(np.abs(y_zero - x))
    results['zero_delay_pass'] = results['zero_delay_error'] < 1e-10

    # Test 2: Integer sample delay
    tau_1sample = 1.0 / fs
    y_1sample = wideband_delay_operator(x, tau_1sample, fs)
    expected = np.roll(x, 1)
    results['1sample_delay_error'] = np.max(np.abs(y_1sample - expected))
    results['1sample_delay_pass'] = results['1sample_delay_error'] < 0.1

    # Test 3: Fractional delay
    tau_half = 0.5 / fs
    y_half = wideband_delay_operator(x, tau_half, fs)
    results['half_sample_smooth'] = np.isfinite(y_half).all()

    results['all_pass'] = all([
        results['zero_delay_pass'],
        results['1sample_delay_pass'],
        results['half_sample_smooth'],
    ])

    return results


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("thz_isac_world_v2.py v4.1 - Multi-bit Quantization Self-Test")
    print("=" * 70)

    print("\n[TEST 1] Wideband Delay Operator")
    delay_results = verify_wideband_delay()
    print(f"  Zero delay error: {delay_results['zero_delay_error']:.2e} "
          f"{'✓' if delay_results['zero_delay_pass'] else '✗'}")
    print(f"  OVERALL: {'PASS ✓' if delay_results['all_pass'] else 'FAIL ✗'}")

    print("\n[TEST 2] Multi-bit Quantization")
    quant_results = verify_quantization([1, 2, 3, 4, 8])

    print("\n[TEST 3] Full Simulation Chain (1-bit)")
    cfg = SimConfig(snr_db=20.0, adc_bits=1)
    data = simulate_batch(cfg, batch_size=32, seed=42)
    print(f"  y_q shape: {data['y_q'].shape}")
    print(f"  adc_bits: {data['meta']['adc_bits']}")
    print(f"  gamma_eff: {data['meta']['gamma_eff']:.2f}")

    print("\n[TEST 4] Full Simulation Chain (4-bit)")
    cfg = SimConfig(snr_db=20.0, adc_bits=4)
    data = simulate_batch(cfg, batch_size=32, seed=42)
    print(f"  y_q shape: {data['y_q'].shape}")
    print(f"  adc_bits: {data['meta']['adc_bits']}")
    print(f"  gamma_eff: {data['meta']['gamma_eff']:.2f}")

    print("\n[TEST 5] Numerical FIM/CRLB")
    cfg = SimConfig(snr_db=15.0, adc_bits=1)
    fim_result = compute_numeric_fim_tau_bussgang(cfg)
    print(f"  FIM(τ): {fim_result['fim_tau']:.2e}")
    print(f"  CRLB(τ): {fim_result['crlb_tau_samples']:.4f} samples")

    print("\n" + "=" * 70)
    print("Self-Test Complete")
    print("=" * 70)