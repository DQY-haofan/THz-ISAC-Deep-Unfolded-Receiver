"""
thz_isac_world.py (Wideband Delay Model - v4.0)

FUNDAMENTAL RESTRUCTURING (Expert Review):
===========================================
Previous model (v3.x):
    h[n] = exp(-j × 2π × fc × (R/c + v/c × t + 0.5 × a/c × t²))

    Problem: R enters ONLY the constant phase term.
    At fc=300GHz (λ=1mm), 10m R error → 10000 × 2π phase shift → random BER!

New model (v4.0 - Wideband Delay):
    y = exp(-j × φ₀) × D(τ) × x × p(t) + noise

    Where:
    - D(τ): Wideband delay operator (identifiable via bandwidth B)
        D(τ)[x] = ifft(fft(x) × exp(-j × 2π × f_k × τ))
        f_k = fftfreq(N, 1/fs)  # baseband frequencies

    - p(t): Doppler/acceleration phase (NO absolute R term!)
        p(t) = exp(-j × 2π × fc × (v/c × t + 0.5 × a/c × t²))

    - φ₀: Nuisance constant phase (absorbed by PN tracker / pilot calibration)

Key insight:
    - Delay τ is identifiable from GROUP DELAY (related to bandwidth B)
    - Resolution: δτ ~ 1/B, not 1/fc
    - For B=10GHz: δτ ~ 100ps → δR ~ 3cm (achievable!)
    - Carrier phase exp(-j×2π×fc×τ) is nuisance, tracked by PN module

This is the CORRECT physics for THz-ISAC sensing!

Author: Expert Review v4.0 (Wideband Delay Model)
Date: 2025-12-19
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

    # Channel Geometry - NEW PARAMETERIZATION
    enable_channel: bool = True
    R: float = 500e3            # Range [m] - used for coarse acquisition
    v_rel: float = 7.5e3        # Velocity [m/s]
    a_rel: float = 10.0         # Acceleration [m/s²]

    # Coarse acquisition (simulated)
    # tau_res will be the residual delay after coarse acquisition
    # For now, we assume perfect coarse acquisition (tau_res ~ 0)
    coarse_acquisition_error_samples: float = 0.0  # Error in samples (0 = perfect)

    # Nuisance initial phase (absorbed by PN tracker)
    phi0_random: bool = True    # Random initial phase each batch

    # Receiver
    enable_quantization: bool = True
    snr_db: float = 20.0

    @property
    def Ts(self):
        return 1.0 / self.fs

    @property
    def wavelength(self):
        """Carrier wavelength [m]."""
        return 3e8 / self.fc

    @property
    def delay_resolution(self):
        """Delay resolution from bandwidth [s]."""
        return 1.0 / self.fs  # One sample period

    @property
    def range_resolution(self):
        """Range resolution from bandwidth [m]."""
        return 3e8 * self.delay_resolution  # c × δτ


# =============================================================================
# Core Physics Modules
# =============================================================================

def generate_symbols(config: SimConfig, batch_size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate QPSK symbols with proper power scaling."""
    bits = rng.integers(0, 4, size=(batch_size, config.N))
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    x = constellation[bits]

    # Scale to target power
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

    # Bussgang decomposition
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
# NEW: Wideband Delay Channel Model
# =============================================================================

def wideband_delay_operator(x: np.ndarray, tau: float, fs: float) -> np.ndarray:
    """
    Apply wideband delay operator D(τ) to signal.

    D(τ)[x] = ifft(fft(x) × exp(-j × 2π × f_k × τ))

    Args:
        x: Input signal [batch, N] or [N]
        tau: Delay in seconds
        fs: Sampling frequency

    Returns:
        Delayed signal (same shape as x)

    Physics:
        This is the CORRECT model for wideband delay.
        The delay is identifiable from the LINEAR PHASE across frequency.
        Resolution: δτ ~ 1/B where B = fs (bandwidth)
    """
    is_batched = x.ndim == 2
    if not is_batched:
        x = x[np.newaxis, :]

    N = x.shape[1]

    # Baseband frequencies (Hz)
    f_k = np.fft.fftfreq(N, d=1.0/fs)  # [N]

    # Delay transfer function: exp(-j × 2π × f_k × τ)
    # Note: This creates LINEAR phase vs frequency
    H_delay = np.exp(-1j * 2 * np.pi * f_k * tau)  # [N]

    # Apply in frequency domain
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

    Args:
        N: Number of samples
        fs: Sampling frequency
        fc: Carrier frequency
        v: Velocity [m/s]
        a: Acceleration [m/s²]

    Returns:
        Phase vector [N] (complex)

    Note:
        The absolute range R does NOT appear here!
        R affects only the constant phase (nuisance) and the delay.
    """
    c = 3e8
    Ts = 1.0 / fs
    t = np.arange(N) * Ts

    # Doppler frequency shift: fd = fc × v/c
    # Phase from velocity: 2π × fc × (v/c) × t
    # Phase from acceleration: 2π × fc × (a/c) × t² / 2

    phase = 2 * np.pi * fc * (v/c * t + 0.5 * a/c * t**2)

    return np.exp(-1j * phase)


def apply_wideband_channel(
    x: np.ndarray,
    config: SimConfig,
    rng: np.random.Generator
) -> Tuple[np.ndarray, Dict]:
    """
    Apply wideband delay channel model.

    y = exp(-j × φ₀) × D(τ) × x × p(t)

    Args:
        x: Input signal [batch, N]
        config: Simulation configuration
        rng: Random number generator

    Returns:
        y: Output signal [batch, N]
        channel_info: Dictionary with channel parameters
    """
    if not config.enable_channel:
        return x, {'tau': 0.0, 'tau_res': 0.0, 'phi0': 0.0, 'v': 0.0, 'a': 0.0}

    batch_size = x.shape[0]
    N = x.shape[1]
    c = 3e8

    # === True delay from range ===
    tau_true = config.R / c  # True delay [s]

    # === Coarse acquisition (simulated) ===
    # In practice, coarse acquisition finds τ within ±1 sample
    # Here we add configurable acquisition error
    tau_coarse_error = config.coarse_acquisition_error_samples * config.Ts
    tau_coarse = tau_true + tau_coarse_error  # What coarse acquisition returns

    # Residual delay (what we need to estimate)
    # This is small (< 1 sample typically)
    tau_res = tau_true - tau_coarse  # Should be ~0 if acquisition is good

    # For now, apply the FULL delay (tau_true) to the signal
    # The model will need to estimate tau_res relative to tau_coarse

    # === Step 1: Apply wideband delay D(τ) ===
    y = wideband_delay_operator(x, tau_true, config.fs)

    # === Step 2: Apply Doppler/acceleration phase p(t) ===
    p_t = doppler_phase_operator(N, config.fs, config.fc, config.v_rel, config.a_rel)
    y = y * p_t[np.newaxis, :]

    # === Step 3: Apply nuisance constant phase φ₀ ===
    # This includes the carrier phase exp(-j×2π×fc×τ) which is NOT identifiable
    # In practice, this is absorbed by the PN tracker / pilot calibration
    if config.phi0_random:
        # Random initial phase for each batch
        phi0 = rng.uniform(0, 2*np.pi, size=(batch_size, 1))
    else:
        # Deterministic phase (for debugging)
        phi0 = np.zeros((batch_size, 1))

    # Apply constant phase rotation
    y = y * np.exp(-1j * phi0)

    # === Channel info for output ===
    channel_info = {
        'tau_true': tau_true,           # True delay [s]
        'tau_coarse': tau_coarse,       # Coarse acquisition estimate [s]
        'tau_res': tau_res,             # Residual delay to estimate [s]
        'phi0': phi0.flatten(),         # Nuisance phase [rad]
        'v': config.v_rel,              # Velocity [m/s]
        'a': config.a_rel,              # Acceleration [m/s²]
        'p_t': p_t,                     # Doppler phase vector
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


def quantize_1bit(y: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, float]:
    """1-bit quantization."""
    if not config.enable_quantization:
        return y, 0.0

    y_q = np.sign(np.real(y)) + 1j * np.sign(np.imag(y))
    y_q = y_q / np.sqrt(2)

    # Bussgang loss
    rho = np.sqrt(2 / np.pi)
    d = y_q - rho * y / (np.std(y) + 1e-12)
    P_quant_loss = np.mean(np.abs(d)**2)

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
        x → PA → PN → Channel[D(τ), p(t), φ₀] → Noise → 1-bit → y_q

    Args:
        config: SimConfig instance
        batch_size: Number of samples
        seed: Optional seed for reproducibility

    Returns:
        Dictionary with:
            - x_true: Original symbols [batch, N]
            - y_q: Quantized received signal [batch, N]
            - theta_true: Ground truth [batch, 3] - [tau_res, v, a] in SI units
            - meta: Dictionary with gamma_eff, chi, channel_info, etc.
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

    # 4. Wideband Delay Channel (NEW!)
    x_channel, channel_info = apply_wideband_channel(x_pn, config, rng)

    # 5. Thermal Noise
    y_analog = add_thermal_noise(x_channel, config, rng)

    # 6. 1-bit Quantization
    y_q, P_quant_loss = quantize_1bit(y_analog, config)

    # === Compute statistics ===
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
        # Fallback
        p_sig = sim_stats['P_signal']
        p_total_dist = (sim_stats['P_pa_distortion'] +
                        sim_stats['P_phase_noise'] +
                        sim_stats['P_quantization_loss'])
        gamma_eff = p_sig / (p_total_dist + 1e-12) if p_total_dist > 1e-12 else 1e9
        snr_linear = 10 ** (config.snr_db / 10)
        sinr_eff = 1.0 / ((1.0/snr_linear) + (1.0/gamma_eff))
        kappa = 1.0 - 2.0/np.pi
        chi = (2.0/np.pi) / (1.0 + kappa * sinr_eff)

    # === Construct Ground Truth Theta ===
    # NEW: theta = [tau_res, v, a] instead of [R, v, a]
    # This is the identifiable parameterization!
    tau_res = channel_info['tau_res']
    theta_val = np.array([tau_res, config.v_rel, config.a_rel])
    theta_true = np.tile(theta_val, (batch_size, 1))

    # Also provide R-based theta for backward compatibility
    theta_R_based = np.tile(
        np.array([config.R, config.v_rel, config.a_rel]),
        (batch_size, 1)
    )

    return {
        "x_true": x_true,
        "y_raw": y_analog,
        "y_q": y_q,
        "theta_true": theta_true,           # [tau_res, v, a] - NEW!
        "theta_R_based": theta_R_based,     # [R, v, a] - backward compat
        "meta": {
            # Bussgang parameters
            "B_gain": B_gain,
            "sigma_eta": sigma_eta,

            # Channel info (NEW!)
            "channel_info": channel_info,
            "tau_true": channel_info['tau_true'],
            "tau_coarse": channel_info['tau_coarse'],
            "tau_res": channel_info['tau_res'],
            "phi0": channel_info['phi0'],

            # SNR and Gamma_eff
            "snr_linear": 10**(config.snr_db/10),
            "snr_db": config.snr_db,
            "sim_stats": sim_stats,
            "gamma_eff": gamma_eff,
            "sinr_eff": sinr_eff,
            "chi": chi,

            # Seed tracking
            "seed_used": seed_used,

            # Hardware config
            "enable_pa": config.enable_pa,
            "enable_pn": config.enable_pn,
            "enable_quantization": config.enable_quantization,
            "ibo_dB": config.ibo_dB,
            "pn_linewidth": config.pn_linewidth,

            # NEW: Resolution info
            "delay_resolution": config.delay_resolution,
            "range_resolution": config.range_resolution,
            "wavelength": config.wavelength,
        }
    }


# =============================================================================
# Coarse Acquisition Stub
# =============================================================================

def coarse_acquisition_correlation(y: np.ndarray, x_pilot: np.ndarray,
                                    fs: float, search_range_samples: int = 100
                                    ) -> Tuple[float, float]:
    """
    Coarse acquisition via sliding correlation.

    This simulates the coarse synchronization that would happen before
    fine tracking. It finds the delay that maximizes correlation with
    known pilot symbols.

    Args:
        y: Received signal [N] or [batch, N]
        x_pilot: Known pilot symbols [N_pilot]
        fs: Sampling frequency
        search_range_samples: Search range in samples (±)

    Returns:
        tau_coarse: Estimated delay [s]
        correlation_peak: Peak correlation value (for confidence)

    Physics:
        Resolution is 1 sample = 1/fs
        For fs=10GHz: δτ = 100ps → δR = 3cm

    Note:
        In practice this would be done with a PN sequence or Zadoff-Chu.
        Here we use a simple correlation for simulation purposes.
    """
    if y.ndim == 2:
        y = y[0]  # Use first batch element

    N = len(y)
    N_pilot = len(x_pilot)

    # Search grid
    search_grid = np.arange(-search_range_samples, search_range_samples + 1)
    correlations = np.zeros(len(search_grid))

    for i, delay_samples in enumerate(search_grid):
        # Shift y by delay_samples
        if delay_samples >= 0:
            y_shifted = np.concatenate([y[delay_samples:], np.zeros(delay_samples)])
        else:
            y_shifted = np.concatenate([np.zeros(-delay_samples), y[:delay_samples]])

        # Correlate with pilot
        corr = np.abs(np.sum(y_shifted[:N_pilot] * np.conj(x_pilot)))
        correlations[i] = corr

    # Find peak
    peak_idx = np.argmax(correlations)
    delay_samples_est = search_grid[peak_idx]
    tau_coarse = delay_samples_est / fs
    correlation_peak = correlations[peak_idx] / (N_pilot + 1e-12)

    return tau_coarse, correlation_peak


def simulate_with_acquisition(config: SimConfig, batch_size: int = 64,
                               seed: Optional[int] = None,
                               do_acquisition: bool = True) -> Dict:
    """
    Simulate with optional coarse acquisition step.

    This is the recommended entry point for training:
    1. Generate signal with full delay
    2. Optionally run coarse acquisition to get tau_coarse
    3. Return theta_true as residual [tau_res, v, a]

    Args:
        config: SimConfig
        batch_size: Batch size
        seed: Random seed
        do_acquisition: Whether to run coarse acquisition

    Returns:
        Same as simulate_batch, but with acquisition info in meta
    """
    # Get base simulation
    data = simulate_batch(config, batch_size, seed)

    if do_acquisition and config.enable_channel:
        # Run coarse acquisition on first batch element
        x_pilot = data['x_true'][0, :64]  # Use first 64 symbols as pilot
        y_raw = data['y_raw'][0] if 'y_raw' in data else data['y_q'][0]

        tau_coarse_est, corr_peak = coarse_acquisition_correlation(
            y_raw, x_pilot, config.fs, search_range_samples=50
        )

        # Update meta with acquisition info
        data['meta']['acquisition'] = {
            'tau_coarse_est': tau_coarse_est,
            'correlation_peak': corr_peak,
            'acquisition_snr': 10 * np.log10(corr_peak + 1e-12),
        }

    return data


# =============================================================================
# BCRLB Computation (Diagonal Approximation)
# =============================================================================

def compute_bcrlb_diag(config: SimConfig, snr_linear: float,
                       gamma_eff: float) -> np.ndarray:
    """
    Compute diagonal Bayesian Cramér-Rao Lower Bound for [τ, v, a].

    This is a simplified diagonal approximation suitable for:
    1. Loss weighting during training
    2. Performance benchmarking

    For wideband delay model with 1-bit quantization:
        BCRLB(τ) ≈ 1 / (8π² B² N SNR_eff)
        BCRLB(v) ≈ (c / fc)² × 3 / (π² T² N SNR_eff)
        BCRLB(a) ≈ (c / fc)² × 180 / (π² T⁴ N SNR_eff)

    Where:
        B = fs (bandwidth)
        T = N × Ts (observation time)
        SNR_eff = SNR × γ_eff × (2/π) (1-bit loss factor)

    Args:
        config: Simulation configuration
        snr_linear: Linear SNR
        gamma_eff: Hardware efficiency factor

    Returns:
        bcrlb_diag: [3] array with BCRLB for [τ, v, a]
    """
    N = config.N
    B = config.fs
    T = N / config.fs  # Observation time
    fc = config.fc
    c = 3e8

    # Effective SNR with 1-bit quantization loss
    quantization_loss = 2 / np.pi  # ~0.64
    snr_eff = snr_linear * gamma_eff * quantization_loss

    # Prevent division by zero
    snr_eff = max(snr_eff, 1e-6)

    # BCRLB for delay τ (wideband model)
    # From Shen & Win, wideband ranging
    bcrlb_tau = 1 / (8 * np.pi**2 * B**2 * N * snr_eff)

    # BCRLB for velocity v
    # Doppler resolution ~ 1/T, then scale by c/fc
    bcrlb_v = (c / fc)**2 * 3 / (np.pi**2 * T**2 * N * snr_eff)

    # BCRLB for acceleration a
    bcrlb_a = (c / fc)**2 * 180 / (np.pi**2 * T**4 * N * snr_eff)

    return np.array([bcrlb_tau, bcrlb_v, bcrlb_a])


def compute_fim_diag(config: SimConfig, snr_linear: float,
                     gamma_eff: float) -> np.ndarray:
    """
    Compute diagonal Fisher Information Matrix.

    FIM = 1 / BCRLB (for diagonal approximation)

    Returns:
        fim_diag: [3] array with FIM diagonal for [τ, v, a]
    """
    bcrlb = compute_bcrlb_diag(config, snr_linear, gamma_eff)
    return 1.0 / (bcrlb + 1e-20)


# =============================================================================
# Legacy Compatibility Function
# =============================================================================

def apply_channel_squint(x: np.ndarray, config: SimConfig) -> Tuple[np.ndarray, np.ndarray]:
    """
    DEPRECATED: Legacy carrier-phase channel model.

    Use apply_wideband_channel() instead for correct physics!

    This function is kept for backward compatibility only.
    """
    import warnings
    warnings.warn(
        "apply_channel_squint() uses incorrect carrier-phase model. "
        "Use apply_wideband_channel() for correct wideband delay physics.",
        DeprecationWarning
    )

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


# =============================================================================
# Verification Functions
# =============================================================================

def verify_wideband_delay() -> Dict:
    """
    Verify wideband delay operator is working correctly.

    Tests:
    1. Zero delay should return input
    2. Integer sample delay should be exact
    3. Fractional delay should be smooth
    """
    results = {}
    N = 1024
    fs = 10e9

    # Test signal
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, N) + 1j * rng.normal(0, 1, N)

    # Test 1: Zero delay
    y_zero = wideband_delay_operator(x, 0.0, fs)
    results['zero_delay_error'] = np.max(np.abs(y_zero - x))
    results['zero_delay_pass'] = results['zero_delay_error'] < 1e-10

    # Test 2: Integer sample delay (1 sample = 1/fs)
    tau_1sample = 1.0 / fs
    y_1sample = wideband_delay_operator(x, tau_1sample, fs)
    # Should equal np.roll(x, 1) approximately
    expected = np.roll(x, 1)
    results['1sample_delay_error'] = np.max(np.abs(y_1sample - expected))
    results['1sample_delay_pass'] = results['1sample_delay_error'] < 0.1  # Some error due to edge effects

    # Test 3: Fractional delay (0.5 sample)
    tau_half = 0.5 / fs
    y_half = wideband_delay_operator(x, tau_half, fs)
    results['half_sample_smooth'] = np.isfinite(y_half).all()

    # Test 4: Phase linearity check
    # For a delay, the phase should be linear in frequency
    X = np.fft.fft(x)
    Y = np.fft.fft(y_1sample)
    H = Y / (X + 1e-12)
    phase = np.unwrap(np.angle(H))
    # Phase should be approximately linear
    f_k = np.fft.fftfreq(N, 1/fs)
    expected_phase = -2 * np.pi * f_k * tau_1sample
    phase_error = np.abs(phase - expected_phase)
    results['phase_linearity_error'] = np.mean(phase_error[10:-10])  # Exclude edges
    results['phase_linearity_pass'] = results['phase_linearity_error'] < 0.1

    results['all_pass'] = all([
        results['zero_delay_pass'],
        results['1sample_delay_pass'],
        results['half_sample_smooth'],
        results['phase_linearity_pass'],
    ])

    return results


def verify_mc_randomness(n_trials: int = 5) -> Dict:
    """Verify Monte Carlo randomness is working correctly."""
    cfg = SimConfig()
    results = {}

    # Test 1: No seed - should be unique each call
    checksums_no_seed = []
    for _ in range(n_trials):
        data = simulate_batch(cfg, batch_size=16, seed=None)
        cs = hashlib.md5(data['y_q'].tobytes()).hexdigest()[:8]
        checksums_no_seed.append(cs)
    results['no_seed_unique'] = len(set(checksums_no_seed)) == n_trials

    # Test 2: Same seed - should be identical
    checksums_same_seed = []
    for _ in range(n_trials):
        data = simulate_batch(cfg, batch_size=16, seed=12345)
        cs = hashlib.md5(data['y_q'].tobytes()).hexdigest()[:8]
        checksums_same_seed.append(cs)
    results['same_seed_identical'] = len(set(checksums_same_seed)) == 1

    results['all_pass'] = results['no_seed_unique'] and results['same_seed_identical']

    return results


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("thz_isac_world.py v4.0 - Wideband Delay Model Self-Test")
    print("=" * 70)

    print("\n[TEST 1] Wideband Delay Operator")
    delay_results = verify_wideband_delay()
    print(f"  Zero delay error: {delay_results['zero_delay_error']:.2e} "
          f"{'✓' if delay_results['zero_delay_pass'] else '✗'}")
    print(f"  1-sample delay error: {delay_results['1sample_delay_error']:.4f} "
          f"{'✓' if delay_results['1sample_delay_pass'] else '✗'}")
    print(f"  Phase linearity error: {delay_results['phase_linearity_error']:.4f} "
          f"{'✓' if delay_results['phase_linearity_pass'] else '✗'}")
    print(f"  OVERALL: {'PASS ✓' if delay_results['all_pass'] else 'FAIL ✗'}")

    print("\n[TEST 2] Monte Carlo Randomness")
    mc_results = verify_mc_randomness()
    print(f"  No seed unique: {'✓' if mc_results['no_seed_unique'] else '✗'}")
    print(f"  Same seed identical: {'✓' if mc_results['same_seed_identical'] else '✗'}")
    print(f"  OVERALL: {'PASS ✓' if mc_results['all_pass'] else 'FAIL ✗'}")

    print("\n[TEST 3] Full Simulation Chain")
    cfg = SimConfig(snr_db=20.0)
    data = simulate_batch(cfg, batch_size=32, seed=42)
    print(f"  x_true shape: {data['x_true'].shape}")
    print(f"  y_q shape: {data['y_q'].shape}")
    print(f"  theta_true shape: {data['theta_true'].shape}")
    print(f"  theta_true[0]: tau_res={data['theta_true'][0,0]:.6e}s, "
          f"v={data['theta_true'][0,1]:.1f}m/s, a={data['theta_true'][0,2]:.1f}m/s²")
    print(f"  gamma_eff: {data['meta']['gamma_eff']:.2f}")
    print(f"  chi: {data['meta']['chi']:.4f}")
    print(f"  Range resolution: {data['meta']['range_resolution']*100:.1f} cm")
    print(f"  Wavelength: {data['meta']['wavelength']*1000:.2f} mm")

    print("\n[INFO] Physical Interpretation:")
    print(f"  At fc=300GHz, λ=1mm:")
    print(f"    - Carrier phase changes 360° per 1mm (NOT identifiable alone)")
    print(f"    - Group delay (bandwidth B=10GHz) gives ~3cm resolution")
    print(f"  This is why we use WIDEBAND DELAY model, not carrier phase!")

    print("\n" + "=" * 70)
    print("Self-Test Complete")
    print("=" * 70)