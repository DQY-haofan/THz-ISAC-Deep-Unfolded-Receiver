"""
baselines_receivers.py

Description:
    Traditional Model-Driven Baselines for THz-ISAC "Dirty Hardware" Receiver.
    Implements non-learning based iterative algorithms to serve as benchmarks
    for the Deep Unfolded Network (GA-BV-Net).

    Algorithms:
    1. Bussgang-LMMSE: Linear Minimum Mean Square Error detector using Bussgang linearization.
    2. 1-bit GAMP: Generalized Approximate Message Passing for 1-bit quantization.

    Gap Analysis Reference:
    Based on DR-P0-1 [cite: 444-448], traditional iterative algorithms (AMP/LMMSE)
    struggle with compound impairments (PA + PN + Quantization). These implementations
    serve as the "Performance Floor" or "Weak Baseline" to demonstrate the necessity
    of Deep Unfolding.

Author: Gemini (AI Thought Partner)
Date: 2025-11-17
"""

import numpy as np
import scipy.stats as stats
from typing import Dict, Tuple, Optional

# Import Simulation World for config type hints
try:
    from thz_isac_world import SimConfig, simulate_batch
except ImportError:
    pass  # Allow standalone usage if needed


# --- Utility Functions ---

def qpsk_demod(x_hat: np.ndarray) -> np.ndarray:
    """Hard slicer for QPSK constellation."""
    # Constellation: (+/-1 +/- 1j) / sqrt(2)
    real_part = np.sign(np.real(x_hat))
    imag_part = np.sign(np.imag(x_hat))
    # Handle zero case
    real_part[real_part == 0] = 1
    imag_part[imag_part == 0] = 1
    return (real_part + 1j * imag_part) / np.sqrt(2)


def compute_ser(x_true: np.ndarray, x_hat: np.ndarray) -> float:
    """Computes Symbol Error Rate."""
    x_hat_sym = qpsk_demod(x_hat)
    # Compare with tolerance to avoid floating point issues, though exact match for QPSK is easy
    # We compare signs essentially
    diff = np.abs(x_hat_sym - x_true)
    errors = np.sum(diff > 1e-3)
    return errors / x_true.size


# --- Baseline 1: Bussgang-LMMSE ---

def detector_lmmse_bussgang(y_q: np.ndarray, H_eff: np.ndarray,
                            noise_var_total: float, P_signal: float) -> np.ndarray:
    """
    Linear MMSE Detector based on Bussgang Decomposition.
    Ref: DR-P0-1  - "Traditional linear processing paradigm".

    Model: y = H_eff * x + n_total
    where H_eff includes Channel, PA Bussgang Gain, and PN Phase rotation (mean).
    n_total includes Thermal Noise + PA Distortion + Quantization Noise + PN Variance.

    Args:
        y_q (np.ndarray): Received quantized signal [N, 1].
        H_eff (np.ndarray): Effective channel matrix [N, N].
        noise_var_total (float): Effective noise variance (sum of all impairments).
        P_signal (float): Signal power.

    Returns:
        x_hat (np.ndarray): Estimated symbols.
    """
    N = H_eff.shape[0]

    # W_LMMSE = P_x * H^H * (H * P_x * H^H + C_n)^-1
    # Assuming independent symbols: R_x = P_signal * I
    # Assuming white effective noise: C_n = noise_var_total * I

    # H_eff is often diagonal in OFDM/SC-FDE if we ignore ICI/Doppler Squint for LMMSE
    # or if we operate in Frequency Domain.
    # Here we perform generic Matrix inversion (Complexity O(N^3)).
    # For large N, this is slow, highlighting the need for efficient algorithms (Unfolding).

    # Regularization factor: noise_var / signal_power
    reg = noise_var_total / P_signal

    # Compute: (H^H H + reg * I)^-1 H^H y
    # Standard Ridge Regression form

    HH = H_eff.conj().T
    Gram = HH @ H_eff

    # Tikhonov Regularization
    A = Gram + reg * np.eye(N)

    # Solve linear system A * x = HH * y
    rhs = HH @ y_q

    try:
        x_hat = np.linalg.solve(A, rhs)
    except np.linalg.LinAlgError:
        # Fallback for singular matrix
        x_hat = np.linalg.lstsq(A, rhs, rcond=None)[0]

    return x_hat


# --- Baseline 2: 1-bit GAMP (Generalized AMP) ---

def phi_func(z):
    """Standard Normal CDF."""
    return stats.norm.cdf(z)


def phi_pdf(z):
    """Standard Normal PDF."""
    return stats.norm.pdf(z)


def output_update_1bit(p: np.ndarray, v_p: np.ndarray, y: np.ndarray, noise_var: float) -> Tuple[
    np.ndarray, np.ndarray]:
    """
    Output node update for 1-bit GAMP.
    Calculates E[z|y, p, v_p] and Var[z|y, p, v_p].

    Model: y = sign(z + n), z is the noiseless output of linear mixing.

    Args:
        p: Estimated output mean from linear step.
        v_p: Estimated output variance.
        y: Observed bits (+/- 1 complex, handled as separate Re/Im).
        noise_var: Thermal noise variance.
    """
    # Probit function calculation
    # We treat Real and Imaginary parts separately for QPSK/1-bit CS
    # z_real = Re(p), z_imag = Im(p)
    # y_real = Re(y), y_imag = Im(y)

    # Effective std dev
    tau = np.sqrt(v_p + noise_var / 2)  # /2 because noise splits into I/Q

    # Standardized variable
    # argument for phi: (y * p) / tau.  Note: y is +/- 1.
    # If y=1, we want Prob(z>0). If y=-1, Prob(z<0).
    xi_real = (np.real(y) * np.real(p)) / tau
    xi_imag = (np.imag(y) * np.imag(p)) / tau

    # Helper for Mill's ratio: pdf / cdf
    def mills(val):
        denom = phi_func(val)
        # Numerical stability clip
        denom = np.clip(denom, 1e-9, 1.0)
        return phi_pdf(val) / denom

    ratio_re = mills(xi_real)
    ratio_im = mills(xi_imag)

    # GAMP g_out function (First derivative of log-likelihood)
    # g = (y / tau) * ratio
    g_re = (np.real(y) / tau) * ratio_re
    g_im = (np.imag(y) / tau) * ratio_im
    g = g_re + 1j * g_im

    # GAMP -g'_out function (Second derivative)
    # -g' = (ratio * (xi + ratio)) / tau^2
    dg_re = (ratio_re * (xi_real + ratio_re)) / (tau ** 2)
    dg_im = (ratio_im * (xi_imag + ratio_im)) / (tau ** 2)
    dg = dg_re + dg_im  # Sum for complex variance update roughly

    # Return z_hat, v_z
    # z_hat = p + v_p * g
    z_hat = p + v_p * g

    # v_z = v_p - v_p^2 * (-g')
    # Note: This is an approximation for complex GAMP
    v_z = v_p - (v_p ** 2) * dg
    v_z = np.maximum(v_z, 1e-9)  # Safety

    return z_hat, v_z


def detector_gamp_1bit(y_q: np.ndarray, H_eff: np.ndarray,
                       noise_var: float, max_iter: int = 20) -> np.ndarray:
    """
    Generalized Approximate Message Passing (GAMP) for 1-bit quantization.
    Ref: DR-P0-1  - AMP family algorithms.

    This implementation simplifies the "Dirty Hardware" to just 1-bit quantization + AWGN.
    It does NOT inherently handle PA memory or Phase Noise (unless H_eff captures it).
    This limitation is exactly what the Deep Unfolding Network aims to solve.

    Args:
        y_q: Quantized received signal.
        H_eff: Effective Channel Matrix.
        noise_var: Thermal noise variance.
        max_iter: Maximum iterations.

    Returns:
        x_hat: Estimated symbols.
    """
    M, N = H_eff.shape

    # Initialization
    x_hat = np.zeros(N, dtype=complex)
    v_x = np.ones(N)  # Variance of x (normalized power)

    # Precompute squared magnitude of H for variance updates
    S = np.abs(H_eff) ** 2

    # Initial 'onsager' term
    s_hat = np.zeros(M, dtype=complex)  # Residual

    for it in range(max_iter):
        # 1. Linear Step (Output Node)
        # p = H * x_hat - v_p * s_hat (Onsager correction)
        v_p = S @ v_x  # Variance of p
        p = H_eff @ x_hat - v_p * s_hat

        # 2. Non-linear Step (Output Node - 1-bit Likelihood)
        z_hat, v_z = output_update_1bit(p, v_p, y_q, noise_var)

        # Calculate residual s
        # s_hat = (z_hat - p) / v_p  OR  g_out
        # v_s = (1 - v_z/v_p) / v_p

        # Using GAMP standard notation:
        # g_out is essentially (z_hat - p) / v_p
        g_out = (z_hat - p) / (v_p + 1e-9)
        dg_out = (1.0 - v_z / (v_p + 1e-9)) / (v_p + 1e-9)

        s_hat = g_out
        v_s = dg_out

        # 3. Linear Step (Input Node)
        # r = x_hat + v_x * (H^H * s_hat)
        # v_r = (v_x * S^T * v_s)^-1 ... simplified:

        inv_v_r = S.T @ v_s
        v_r = 1.0 / (inv_v_r + 1e-9)
        r = x_hat + v_r * (H_eff.conj().T @ s_hat)

        # 4. Non-linear Step (Input Node - Prior on X)
        # Assuming Gaussian Prior for X ~ CN(0, 1) or QPSK
        # For simplicity, we use a simple linear MMSE denoising (Gaussian Prior)
        # x_hat_new = E[x | r, v_r] = r * (1 / (1 + v_r)) if prior var is 1
        # Or soft thresholding if sparse. Here we assume dense constellation.

        # Simple LMMSE denoiser for Gaussian Signal
        # prior_var = 1.0.  post_var = 1 / (1/prior_var + 1/v_r)
        # x_post = post_var * (r / v_r)

        v_x_new = 1.0 / (1.0 + 1.0 / v_r)
        x_hat_new = v_x_new * (r / v_r)

        # Damping to prevent divergence
        damping = 0.5
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        v_x = damping * v_x + (1 - damping) * v_x_new

    return x_hat


# --- Evaluation Harness ---

def evaluate_baselines(config: SimConfig, n_trials: int = 100):
    """
    Runs baselines on a batch of data and reports BER/MSE.
    """
    print(f"\n--- Evaluating Baselines (N={n_trials}) ---")

    # 1. Generate Data using Sim World
    batch_data = simulate_batch(config, batch_size=n_trials)

    X_true = batch_data['x_true']
    Y_q = batch_data['y_q']
    meta = batch_data['meta']

    # Extract Parameters
    # Construct H_eff for the detector (Cheating a bit: assuming perfect CSI of the linear part)
    # In reality, H_eff would be estimated.
    # Here: H_eff = B * H_squint * Phi_mean (Identity for now)
    # We simplify H_eff to be diagonal H_squint * B_gain

    B_gain = meta['B_gain']  # Scalar approx
    # Diagonal channel from Doppler Squint
    H_diag_batch = meta['h_diag']

    # Metrics containers
    ser_lmmse = []
    ser_gamp = []

    for i in range(n_trials):
        y = Y_q[i]
        x = X_true[i]

        # Construct H_eff (Diagonal)
        h_diag = H_diag_batch[i] * B_gain
        # Convert to matrix if needed, but for diagonal we can optimize.
        # However, LMMSE implementation above assumes generic matrix.
        H_eff = np.diag(h_diag)

        # Estimate Noise Variance
        # noise_total = PA_Distortion + Thermal + Quantization
        # Roughly: sigma_eta + N0.
        # GAMP handles quantization internally. LMMSE needs total effective noise.
        sigma_eta = meta['sigma_eta']
        n0 = 1.0 / meta['snr_linear']
        noise_lmmse = sigma_eta + n0 + 0.5  # Add crude quantization noise floor
        noise_gamp = sigma_eta + n0  # GAMP models quantizer explicitly

        # 1. Run LMMSE
        x_lmmse = detector_lmmse_bussgang(y, H_eff, noise_lmmse, P_signal=1.0)
        ser_lmmse.append(compute_ser(x, x_lmmse))

        # 2. Run GAMP
        # Warning: GAMP is sensitive to H matrix properties (i.i.d gaussian is best).
        # Diagonal H might cause issues for standard GAMP without damping.
        x_gamp = detector_gamp_1bit(y, H_eff, noise_gamp)
        ser_gamp.append(compute_ser(x, x_gamp))

    print(f"Results [SNR={config.snr_db} dB, PA={config.enable_pa}, 1-bit={config.enable_quantization}]:")
    print(f"  LMMSE SER: {np.mean(ser_lmmse):.4f}")
    print(f"  GAMP  SER: {np.mean(ser_gamp):.4f}")

    return {"lmmse": np.mean(ser_lmmse), "gamp": np.mean(ser_gamp)}


if __name__ == "__main__":
    # Test Run
    # Need SimConfig from thz_isac_world
    # Ensure thz_isac_world.py is in the same directory
    try:
        cfg = SimConfig()
        cfg.N = 64  # Small size for fast GAMP test
        cfg.snr_db = 20
        evaluate_baselines(cfg, n_trials=10)
    except NameError:
        print("SimConfig not found. Please run this script alongside thz_isac_world.py")