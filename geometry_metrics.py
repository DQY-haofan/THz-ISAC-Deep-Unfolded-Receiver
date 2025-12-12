"""
geometry_metrics.py

Description:
    Information Geometry Toolbox for THz-ISAC "Dirty Hardware" analysis.
    Implements metrics defined in DR-P2-2.5 (Theory) and DR-P2-3.5 (Numerical Check).

    Functions:
    1. Effective Hardware Quality Factor (Gamma_eff) estimation.
    2. 1-bit Information Retention Factor (Chi) with asymptotic scaling.
    3. Block-Diagonal FIM approximation & Hutchinson Trace Estimation.
    4. CRLB calculation (Bayesian & Standard).
    5. Forbidden Region Detection.

Author: Gemini (AI Thought Partner)
Date: 2025-11-17
References:
    [1] DR-P2-2.5: Statistical Manifold & Info Geometry Hardening
    [2] DR-P2-3.5: Geometric Regularization & Sanity Check
"""

import numpy as np
import scipy.linalg
from typing import Dict, List, Tuple, Union, Optional

# --- Constants ---
CHI_LOW_SNR_LIMIT = 2.0 / np.pi  # ~0.6366


# --- Module 1: Hardware Quality Factor (Gamma_eff) ---

def estimate_gamma_eff(sim_stats: Dict) -> float:
    """
    Estimates the Effective Hardware Quality Factor (Gamma_eff) from simulation statistics.
    Based on First Principles definition in DR-P2-2.5.

    Gamma_eff = ( P_eta/P_sig + P_phi/P_sig + P_q/P_sig )^-1

    Args:
        sim_stats (dict): Dictionary containing power estimates from thz_isac_world.
                          Expected keys: 'P_signal', 'P_pa_distortion', 'P_phase_noise', 'P_quantization_loss'

    Returns:
        float: Gamma_eff (linear scale).
    """
    p_sig = sim_stats.get('P_signal', 1.0)

    # Extract distortion powers (default to 0 if perfect hardware)
    p_eta = sim_stats.get('P_pa_distortion', 0.0)
    p_phi = sim_stats.get('P_phase_noise', 0.0)  # Multiplicative noise power approx
    p_q = sim_stats.get('P_quantization_loss', 0.0)  # Effective noise from quantization

    # Avoid division by zero
    if p_sig < 1e-12:
        return 0.0

    # Calculate Inverse SINR components (Distortion-to-Signal Ratios)
    dsr_pa = p_eta / p_sig
    dsr_pn = p_phi / p_sig
    dsr_q = p_q / p_sig

    total_inverse_gamma = dsr_pa + dsr_pn + dsr_q

    if total_inverse_gamma < 1e-9:
        return 1e9  # Effectively infinite quality

    gamma_eff = 1.0 / total_inverse_gamma
    return gamma_eff


# --- Module 2: Information Retention Factor (Chi) ---

def chi_low_snr_limit() -> float:
    """Returns the theoretical limit of Chi at low SNR."""
    return CHI_LOW_SNR_LIMIT


def approx_chi(gamma_eff: float, snr_linear: float) -> float:
    """
    Approximates the Information Retention Factor Chi(theta).
    Captures the 'Hardware Wall' effect where Chi decays at high SNR due to dirty hardware.

    Formula based on DR-P2-3.5 [cite: 400-402] & DR-P2-2.5[cite: 557].

    Args:
        gamma_eff (float): Linear effective hardware quality factor.
        snr_linear (float): Linear Analog SNR (Thermal).

    Returns:
        float: Chi factor (0 < chi <= 2/pi).
    """
    # Theoretical max scaling
    chi_max = CHI_LOW_SNR_LIMIT

    # Scaling factor modeling the saturation.
    # Logic: As SNR increases, if Gamma_eff is finite, the effective SINR saturates.
    # The 1-bit quantizer loses info relative to the analog signal which keeps growing.
    # Ref[cite: 557]: chi <= C / (1 + Gamma^-1) * (1/SNR)
    # Ref[cite: 401]: Implementation logic: chi ~ 1 / (1 + SNR/Gamma_eff)

    # We use a smoothed logistic-like transition:
    # effective_saturation = 1 / (1 + SNR_analog / Gamma_eff)

    # Avoid div by zero
    safe_gamma = max(gamma_eff, 1e-6)

    saturation_term = 1.0 / (1.0 + (snr_linear / safe_gamma))

    chi = chi_max * saturation_term

    return chi


# --- Module 3: FIM & CRLB (Block Diagonal) ---

def compute_fim_block_diag(J_analog_full: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Approximates the full Fisher Information Matrix using Block-Diagonal strategy.
    Crucial for reducing complexity from O(N^3) to O(N * K^2).
    Ref: DR-P2-3.5 Section 3.1[cite: 246].

    Args:
        J_analog_full (np.ndarray): The full NxN FIM (or covariance inverse).
                                    In practice, this might be huge, so usually we assemble blocks directly.
                                    Here we simulate the operation on a matrix for verification.
        block_size (int): Size of the coherence window (K_block). Default 32[cite: 311].

    Returns:
        np.ndarray: Block-diagonal approximation of J.
    """
    N = J_analog_full.shape[0]
    J_approx = np.zeros_like(J_analog_full)

    for i in range(0, N, block_size):
        end = min(i + block_size, N)
        # Extract block
        block = J_analog_full[i:end, i:end]
        # Place block
        J_approx[i:end, i:end] = block

    return J_approx


def compute_crlb(fim: np.ndarray, param_indices: Optional[List[int]] = None) -> np.ndarray:
    """
    Computes Cramer-Rao Lower Bound from FIM.
    CRLB = diag(inv(FIM)).

    Args:
        fim (np.ndarray): Fisher Information Matrix.
        param_indices (list): Indices of parameters to return CRLB for. If None, return all.

    Returns:
        np.ndarray: Vector of CRLB values.
    """
    # 1. Invert FIM
    # Use pseudo-inverse for numerical stability near forbidden regions (singular matrices)
    try:
        inv_fim = scipy.linalg.inv(fim)
    except np.linalg.LinAlgError:
        # Fallback to pinv if singular
        inv_fim = scipy.linalg.pinv(fim)

    crlb_all = np.diagonal(inv_fim)

    if param_indices is not None:
        return crlb_all[param_indices]
    return crlb_all


def hutchinson_trace_est(operator_func, dim: int, num_probes: int = 10) -> float:
    """
    Performs Hutchinson Stochastic Trace Estimation.
    Tr(A) approx 1/K sum(z^T A z).
    Ref: DR-P2-3.5 Section 3.2[cite: 258].

    Args:
        operator_func: A function f(v) that computes A @ v.
                       Allows simplified computation without forming full Matrix A.
        dim (int): Dimension of the vector space.
        num_probes (int): Number of Rademacher vectors.

    Returns:
        float: Estimated trace.
    """
    trace_est = 0.0
    rng = np.random.default_rng()

    for _ in range(num_probes):
        # Rademacher vector (+1 or -1)
        z = 2 * rng.integers(0, 2, size=dim) - 1
        z = z.astype(float)

        # Apply operator: w = A z
        w = operator_func(z)

        # Accumulate z^T w
        trace_est += np.dot(z, w)

    return trace_est / num_probes


# --- Module 4: Forbidden Region Detection ---

def check_forbidden_region(logdet_j_eff: float, pn_variance_total: float, threshold: float = -10.0) -> Tuple[bool, str]:
    """
    Checks if the system is in the "Forbidden Region" based on Geometry and Physics.
    Ref: DR-P2-3.5 Section 2.3 [cite: 236] & DR-P2-2.5[cite: 616].

    Args:
        logdet_j_eff (float): Log-determinant of the effective FIM.
        pn_variance_total (float): Cumulative phase noise variance (sigma^2 * N).
        threshold (float): Empirical threshold for logdet.

    Returns:
        (bool, str): (is_forbidden, reason)
    """
    # Criterion 1: Physical Ambiguity (Phase Noise > Pi)
    # [cite: 236] sigma_PN^2 * N > pi
    if pn_variance_total > np.pi:
        return True, "Physical: PN Variance > Pi (Spectral Aliasing)"

    # Criterion 2: Geometric Collapse (LogDet too small)
    if logdet_j_eff < threshold:
        return True, "Geometric: FIM Singular (Information Collapse)"

    return False, "Safe"


# --- Module 5: Utilities & Error Metrics ---

def relative_frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    """
    Computes relative Frobenius norm error between two matrices.
    Used to verify Block-Diagonal Approximation accuracy.
    Ref: DR-P2-3.5 Protocol 3[cite: 306].

    Err = ||A - B||_F / ||A||_F
    """
    norm_A = np.linalg.norm(A, 'fro')
    if norm_A == 0:
        return 0.0
    return np.linalg.norm(A - B, 'fro') / norm_A


def analytic_fim_delay_doppler(snr_linear: float, N: int, B_rms_sq: float, T_rms_sq: float) -> np.ndarray:
    """
    Returns the theoretical ANALOG FIM for Delay and Doppler (Single Target).
    Useful as a baseline for Chi calculation.

    J_analog = 2 * SNR * N * diag(B_rms^2, T_rms^2) (Simplified)
    """
    # Simple diagonal approximation for baseline
    J = np.zeros((2, 2))
    factor = 2 * snr_linear * N

    # Delay element (related to bandwidth)
    J[0, 0] = factor * B_rms_sq

    # Doppler element (related to duration)
    J[1, 1] = factor * T_rms_sq

    return J


if __name__ == "__main__":
    print("--- Geometry Metrics Toolbox Test ---")

    # 1. Test Gamma_eff
    stats = {
        'P_signal': 1.0,
        'P_pa_distortion': 0.1,  # -10 dB
        'P_phase_noise': 0.01,  # -20 dB
        'P_quantization_loss': 0.05
    }
    g_eff = estimate_gamma_eff(stats)
    print(f"Gamma_eff: {g_eff:.4f} (Linear), {10 * np.log10(g_eff):.2f} dB")

    # 2. Test Chi approx
    snr_db_list = [0, 20, 40]
    print("\nChi Factor Behavior:")
    for snr in snr_db_list:
        snr_lin = 10 ** (snr / 10)
        chi = approx_chi(g_eff, snr_lin)
        print(f"  SNR={snr} dB: Chi={chi:.4f} (Limit={CHI_LOW_SNR_LIMIT:.4f})")

    # 3. Test Block Diag
    print("\nBlock Diagonal Approx:")
    # Create a dummy covariance matrix with decaying off-diagonals (Toeplitz-like)
    N = 64
    x = np.arange(N)
    # Exponential decay correlation
    cov = np.exp(-0.1 * np.abs(x[:, None] - x[None, :]))
    J_full = np.linalg.inv(cov + 0.01 * np.eye(N))  # FIM ~ inv(Cov)

    J_block = compute_fim_block_diag(J_full, block_size=16)
    err = relative_frobenius_error(J_full, J_block)
    print(f"  Matrix Size: {N}x{N}, Block Size: 16")
    print(f"  Approximation Error (Frobenius): {err * 100:.2f}%")

    # 4. Forbidden Region
    print("\nForbidden Region Check:")
    pn_var_high = 0.1 * 100  # sigma^2 * N > pi? 10 > 3.14
    is_forbid, reason = check_forbidden_region(-50, pn_var_high)
    print(f"  High PN Case: {is_forbid} -> {reason}")