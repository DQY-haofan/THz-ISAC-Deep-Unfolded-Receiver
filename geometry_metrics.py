"""
geometry_metrics.py (Definition Freeze v2 - Expert API Contract)

Description:
    Information Geometry Toolbox for THz-ISAC "Dirty Hardware" analysis.
    Implements metrics defined in DR-P2-2.5 (Theory) and DR-P2-3.5 (Numerical Check).

    **DEFINITION FREEZE v2** per Expert Review:
    - χ(ρ) formula with correct endpoint limits
    - Dual API interface per DR-P2-3.5 contract:
      * chi_from_rho(rho)               - One-parameter form
      * approx_chi(gamma_eff, snr_linear) - Two-parameter form
    - API consistency verification function

    Endpoint Properties (FROZEN):
    - χ(ρ→0) = 2/π ≈ 0.6366   (Low SNR limit)
    - χ(ρ→∞) → 0               (High SNR limit)
    - Monotonically decreasing

Author: Definition Freeze v2
Date: 2025-12-13
References:
    [1] DR-P2-2.5: Statistical Manifold & Info Geometry Hardening
    [2] DR-P2-3.5: Geometric Regularization & Sanity Check
    [3] DR-P2-3 注意事项: χ Definition Freeze & Errata
"""

import numpy as np
import scipy.linalg
from typing import Dict, List, Tuple, Union, Optional

# =============================================================================
# DEFINITION FREEZE CONSTANTS
# =============================================================================

CHI_LOW_SNR_LIMIT = 2.0 / np.pi  # ~0.6366 - This is the LOW SNR limit
KAPPA = 1.0 - 2.0 / np.pi        # ~0.3634 - Scaling coefficient for χ formula


# =============================================================================
# Module 1: Hardware Quality Factor (Gamma_eff)
# =============================================================================

def estimate_gamma_eff(sim_stats: Dict) -> float:
    """
    Estimates the Effective Hardware Quality Factor (Gamma_eff) from simulation statistics.
    Based on First Principles definition in DR-P2-2.5.

    Gamma_eff = ( P_eta/P_sig + P_phi/P_sig + P_q/P_sig )^-1

    Args:
        sim_stats (dict): Dictionary containing power estimates.
                          Expected keys: 'P_signal', 'P_pa_distortion',
                                        'P_phase_noise', 'P_quantization_loss'

    Returns:
        float: Gamma_eff (linear scale).
    """
    p_sig = sim_stats.get('P_signal', 1.0)

    # Extract distortion powers (default to 0 if perfect hardware)
    p_eta = sim_stats.get('P_pa_distortion', 0.0)
    p_phi = sim_stats.get('P_phase_noise', 0.0)
    p_q = sim_stats.get('P_quantization_loss', 0.0)

    # Avoid division by zero
    if p_sig < 1e-12:
        return 0.0

    # Calculate Inverse SINR components (Distortion-to-Signal Ratios)
    dsr_pa = p_eta / p_sig
    dsr_pn = p_phi / p_sig
    dsr_q = p_q / p_sig

    total_inverse_gamma = dsr_pa + dsr_pn + dsr_q

    if total_inverse_gamma < 1e-9:
        return 1e9  # Effectively infinite quality (ideal hardware)

    gamma_eff = 1.0 / total_inverse_gamma
    return gamma_eff


def compute_sinr_eff(snr_linear: float, gamma_eff: float) -> float:
    """
    Computes the effective SINR before quantization.

    SINR_eff = 1 / (1/SNR + 1/Gamma_eff)

    This is the "ρ" in the χ(ρ) formula - the pre-quantizer effective signal quality.

    Args:
        snr_linear (float): Linear thermal SNR.
        gamma_eff (float): Linear effective hardware quality factor.

    Returns:
        float: Effective SINR (linear scale), also known as rho_eff.
    """
    # Avoid division by zero
    safe_snr = max(snr_linear, 1e-12)
    safe_gamma = max(gamma_eff, 1e-12)

    # Harmonic combination: SINR_eff = (1/SNR + 1/Gamma)^-1
    inv_sinr_eff = (1.0 / safe_snr) + (1.0 / safe_gamma)

    return 1.0 / inv_sinr_eff


# =============================================================================
# Module 2: Information Retention Factor (Chi) - DEFINITION FREEZE
# =============================================================================

def chi_low_snr_limit() -> float:
    """Returns the theoretical limit of Chi at LOW SNR (ρ → 0)."""
    return CHI_LOW_SNR_LIMIT


def chi_high_snr_limit() -> float:
    """Returns the theoretical limit of Chi at HIGH SNR (ρ → ∞)."""
    return 0.0


def chi_from_rho(rho: float) -> float:
    """
    **DEFINITION FREEZE** - Primary interface for χ(ρ).

    Computes the 1-bit information retention ratio relative to analog observation.
    This captures the 'Hardware Wall' effect where χ decays at high SNR.

    Formula (per DR-P2-3 注意事项 Errata):
        χ(ρ) = (2/π) / (1 + κρ),  where κ = 1 - 2/π ≈ 0.3634

    Endpoint Consistency (FROZEN):
        - ρ → 0 (Low SNR):  χ → 2/π ≈ 0.6366
        - ρ → ∞ (High SNR): χ → 0
        - Monotonically decreasing

    Physical Meaning:
        χ represents "information retention" (NOT Bussgang linear gain).
        When χ is small, the system is in "information-sparse" or "quantization-saturated"
        state, and gradient updates should be suppressed to avoid fitting noise.

    Args:
        rho (float): Effective SINR before quantization (SINR_eff, linear scale).
                     Computed via compute_sinr_eff(snr_linear, gamma_eff).

    Returns:
        float: Chi factor (0 < chi <= 2/π).
    """
    # Ensure non-negative
    safe_rho = max(rho, 0.0)

    # χ(ρ) = (2/π) / (1 + κρ), where κ = 1 - 2/π
    chi = CHI_LOW_SNR_LIMIT / (1.0 + KAPPA * safe_rho)

    return chi


def approx_chi(gamma_eff: float, snr_linear: float) -> float:
    """
    **DEFINITION FREEZE** - Two-parameter interface for χ.

    Computes χ directly from Gamma_eff and SNR.
    This interface is per DR-P2-3.5 contract requirement.

    Internally:
        1. Computes ρ = SINR_eff(SNR, Γ_eff)
        2. Returns χ = chi_from_rho(ρ)

    API Contract (MUST satisfy):
        approx_chi(gamma_eff, snr_linear) == chi_from_rho(compute_sinr_eff(snr_linear, gamma_eff))

    Args:
        gamma_eff (float): Linear effective hardware quality factor.
        snr_linear (float): Linear thermal SNR.

    Returns:
        float: Chi factor (0 < chi <= 2/π).
    """
    rho = compute_sinr_eff(snr_linear, gamma_eff)
    return chi_from_rho(rho)


def verify_chi_api_consistency(snr_linear: float = 100.0, gamma_eff: float = 10.0) -> bool:
    """
    Verifies that both χ interfaces return consistent results.

    Test: approx_chi(gamma_eff, snr_linear) == chi_from_rho(compute_sinr_eff(snr_linear, gamma_eff))

    Args:
        snr_linear: Test SNR value
        gamma_eff: Test Gamma_eff value

    Returns:
        bool: True if APIs are consistent
    """
    rho = compute_sinr_eff(snr_linear, gamma_eff)
    chi_from_two_param = approx_chi(gamma_eff, snr_linear)
    chi_from_one_param = chi_from_rho(rho)

    return np.isclose(chi_from_two_param, chi_from_one_param, rtol=1e-10)


def verify_chi_endpoints(tolerance: float = 0.03) -> Tuple[bool, Dict]:
    """
    **MANDATORY SANITY CHECK** - Verifies χ formula satisfies endpoint constraints.

    Tests (per DR-P2-3 注意事项):
        1. χ(ρ=0) ≈ 2/π (error < tolerance)
        2. χ(ρ→∞) → 0
        3. Monotonically decreasing

    Args:
        tolerance: Maximum allowed error for low SNR limit test

    Returns:
        Tuple[bool, Dict]: (all_passed, details_dict)
    """
    results = {}

    # Test 1: Low SNR limit
    chi_at_zero = chi_from_rho(0.0)
    expected_low = CHI_LOW_SNR_LIMIT
    error_low = abs(chi_at_zero - expected_low)
    results['low_snr_limit'] = {
        'value': chi_at_zero,
        'expected': expected_low,
        'error': error_low,
        'passed': error_low < tolerance
    }

    # Test 2: High SNR limit
    chi_at_inf = chi_from_rho(1e9)
    results['high_snr_limit'] = {
        'value': chi_at_inf,
        'expected': 0.0,
        'error': chi_at_inf,
        'passed': chi_at_inf < 1e-6
    }

    # Test 3: Monotonicity
    rho_test = np.logspace(-2, 6, 100)
    chi_test = np.array([chi_from_rho(r) for r in rho_test])
    is_monotonic = np.all(np.diff(chi_test) <= 0)
    results['monotonicity'] = {
        'value': 'decreasing' if is_monotonic else 'NOT monotonic',
        'passed': is_monotonic
    }

    # Test 4: API consistency
    api_consistent = verify_chi_api_consistency()
    results['api_consistency'] = {
        'value': 'consistent' if api_consistent else 'INCONSISTENT',
        'passed': api_consistent
    }

    all_passed = all(r['passed'] for r in results.values())
    return all_passed, results


# =============================================================================
# Module 3: FIM & CRLB (Block Diagonal)
# =============================================================================

def compute_fim_block_diag(J_analog_full: np.ndarray, block_size: int = 32) -> np.ndarray:
    """
    Approximates the full Fisher Information Matrix using Block-Diagonal strategy.
    Crucial for reducing complexity from O(N^3) to O(N * K^2).
    Ref: DR-P2-3.5 Section 3.1.

    Args:
        J_analog_full (np.ndarray): The full NxN FIM (or covariance inverse).
        block_size (int): Size of the coherence window (K_block). Default 32.

    Returns:
        np.ndarray: Block-diagonal approximation of J.
    """
    N = J_analog_full.shape[0]
    if block_size >= N:
        return J_analog_full.copy()

    J_approx = np.zeros_like(J_analog_full)
    n_blocks = (N + block_size - 1) // block_size

    for i in range(n_blocks):
        start = i * block_size
        end = min(start + block_size, N)
        J_approx[start:end, start:end] = J_analog_full[start:end, start:end]

    return J_approx


def compute_fim_chi_scaled(J_analog: np.ndarray, chi: float) -> np.ndarray:
    """
    Computes χ-scaled FIM for 1-bit quantized observation.

    Theory (per DR-P2-3):
        J_1bit ≈ χ² · J_analog

    Args:
        J_analog (np.ndarray): Analog FIM matrix
        chi (float): Information retention factor

    Returns:
        np.ndarray: Scaled FIM for 1-bit case
    """
    chi_sq = chi ** 2
    return chi_sq * J_analog


def hutchinson_trace_estimate(matrix: np.ndarray, n_samples: int = 100, seed: int = 42) -> float:
    """
    Estimates trace using Hutchinson's method (for large matrices).

    tr(A) ≈ (1/n) Σᵢ zᵢᵀ A zᵢ
    where zᵢ are Rademacher random vectors.

    Args:
        matrix (np.ndarray): Square matrix to estimate trace of.
        n_samples (int): Number of random vectors to use. Default 100.
        seed (int): Random seed. Default 42.

    Returns:
        float: Estimated trace.
    """
    rng = np.random.default_rng(seed)
    N = matrix.shape[0]

    total = 0.0
    for _ in range(n_samples):
        z = rng.choice([-1, 1], size=N).astype(float)
        total += z @ matrix @ z

    return total / n_samples


def relative_frobenius_error(A: np.ndarray, B: np.ndarray) -> float:
    """Computes ||A - B||_F / ||A||_F."""
    norm_A = np.linalg.norm(A, 'fro')
    if norm_A < 1e-12:
        return 0.0
    return np.linalg.norm(A - B, 'fro') / norm_A


# =============================================================================
# Module 4: BCRLB Computation
# =============================================================================

def compute_bcrlb_from_fim(J: np.ndarray, prior_precision: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Computes Bayesian CRLB from FIM.

    Standard CRLB: var(θ) ≥ J⁻¹
    Bayesian CRLB: var(θ) ≥ (J + Λ)⁻¹, where Λ is prior precision

    Args:
        J (np.ndarray): Fisher Information Matrix
        prior_precision (np.ndarray, optional): Prior precision matrix

    Returns:
        np.ndarray: CRLB matrix (covariance lower bound)
    """
    if prior_precision is not None:
        J_total = J + prior_precision
    else:
        J_total = J

    # Regularize for numerical stability
    J_reg = J_total + 1e-10 * np.eye(J_total.shape[0])

    try:
        crlb = np.linalg.inv(J_reg)
    except np.linalg.LinAlgError:
        # Fallback to pseudoinverse
        crlb = np.linalg.pinv(J_reg)

    return crlb


def compute_crlb_range_simplified(snr_linear: float, chi: float,
                                   B: float = 20e9, N: int = 1024) -> float:
    """
    Simplified CRLB for range estimation with 1-bit quantization.

    CRLB_R_analog ≈ c² / (8π² B² SNR N)
    CRLB_R_1bit ≈ CRLB_R_analog / χ²

    Args:
        snr_linear: Linear SNR
        chi: Information retention factor
        B: Bandwidth (Hz)
        N: Number of samples

    Returns:
        float: CRLB for range (m²)
    """
    c = 3e8
    crlb_analog = (c ** 2) / (8 * (np.pi ** 2) * (B ** 2) * snr_linear * N + 1e-12)
    chi_safe = max(chi, 1e-6)
    return crlb_analog / (chi_safe ** 2)


# =============================================================================
# Module 5: Forbidden Region Detection
# =============================================================================

def check_forbidden_region(gamma_eff_db: float,
                            total_pn_variance: float,
                            pn_threshold: float = np.pi) -> Tuple[bool, str]:
    """
    Checks if current operating point is in the "Forbidden Region".

    Forbidden conditions (any triggers forbidden):
        1. Gamma_eff < -20 dB (hardware too degraded)
        2. Total phase noise variance > π (phase wrapping)

    Args:
        gamma_eff_db (float): Effective hardware quality in dB
        total_pn_variance (float): Sum of phase noise variance (σ² × N)
        pn_threshold (float): Phase variance threshold (default π)

    Returns:
        Tuple[bool, str]: (is_forbidden, reason)
    """
    reasons = []

    if gamma_eff_db < -20:
        reasons.append(f"Gamma_eff={gamma_eff_db:.1f}dB < -20dB")

    if total_pn_variance > pn_threshold:
        reasons.append(f"PN_var={total_pn_variance:.2f} > {pn_threshold:.2f}")

    if reasons:
        return True, "Forbidden: " + ", ".join(reasons)
    else:
        return False, "Safe Region"


def compute_region_label(snr_db: float, gamma_eff: float,
                         pn_linewidth: float, N: int, Ts: float) -> str:
    """
    Computes region label for a given operating point.

    Regions:
        - Safe: All metrics within bounds
        - Critical: Near boundary (warning zone)
        - Forbidden: Outside operational limits

    Args:
        snr_db: SNR in dB
        gamma_eff: Linear Gamma_eff
        pn_linewidth: Phase noise linewidth (Hz)
        N: Number of samples
        Ts: Sample period (s)

    Returns:
        str: Region label ('Safe', 'Critical', 'Forbidden')
    """
    gamma_eff_db = 10 * np.log10(max(gamma_eff, 1e-12))

    # Phase noise variance accumulation
    sigma2_phi = 2 * np.pi * pn_linewidth * Ts
    total_pn_var = sigma2_phi * N

    is_forbidden, _ = check_forbidden_region(gamma_eff_db, total_pn_var)

    if is_forbidden:
        return 'Forbidden'

    # Critical zone: close to boundaries
    if gamma_eff_db < -10 or total_pn_var > np.pi * 0.7:
        return 'Critical'

    return 'Safe'


# =============================================================================
# Module 6: Safe Zone Geometry (Diagonal Approximation)
# =============================================================================

def compute_safe_zone_diagonal(snr_db_range: Tuple[float, float],
                                gamma_eff_db_range: Tuple[float, float],
                                chi_threshold: float = 0.1) -> Dict:
    """
    Computes safe zone boundaries using diagonal approximation.

    This is a "lightweight energy-scale constraint" for training warm-up,
    not the final geometry computation (per DR-P2-3 §3.4 clarification).

    Args:
        snr_db_range: (min_snr_db, max_snr_db)
        gamma_eff_db_range: (min_gamma_db, max_gamma_db)
        chi_threshold: Minimum acceptable χ value

    Returns:
        Dict: Safe zone parameters
    """
    snr_min, snr_max = snr_db_range
    gamma_min, gamma_max = gamma_eff_db_range

    # Compute χ at corners
    corners = []
    for snr_db in [snr_min, snr_max]:
        for gamma_db in [gamma_min, gamma_max]:
            snr_lin = 10 ** (snr_db / 10)
            gamma_lin = 10 ** (gamma_db / 10)
            chi = approx_chi(gamma_lin, snr_lin)
            corners.append({
                'snr_db': snr_db,
                'gamma_db': gamma_db,
                'chi': chi,
                'in_safe': chi >= chi_threshold
            })

    return {
        'corners': corners,
        'chi_threshold': chi_threshold,
        'n_safe_corners': sum(1 for c in corners if c['in_safe'])
    }


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Geometry Metrics Module - Definition Freeze v2 Self-Test")
    print("=" * 60)

    # 1. Test Gamma_eff
    print("\n--- Test 1: Gamma_eff Estimation ---")
    stats = {
        'P_signal': 1.0,
        'P_pa_distortion': 0.1,
        'P_phase_noise': 0.01,
        'P_quantization_loss': 0.05
    }
    g_eff = estimate_gamma_eff(stats)
    print(f"Gamma_eff: {g_eff:.4f} (Linear), {10 * np.log10(g_eff):.2f} dB")

    # 2. Test χ endpoints (MANDATORY)
    print("\n--- Test 2: Chi Endpoint Verification (MANDATORY) ---")
    print(f"Constants: 2/π = {CHI_LOW_SNR_LIMIT:.6f}, κ = {KAPPA:.6f}")

    passed, details = verify_chi_endpoints()
    print(f"\nOverall: {'PASSED ✓' if passed else 'FAILED ✗'}")
    for test_name, result in details.items():
        status = "✓" if result['passed'] else "✗"
        print(f"  {status} {test_name}: {result}")

    # 3. Test dual API
    print("\n--- Test 3: Dual API Interface ---")
    snr_test = 100.0
    gamma_test = 10.0
    rho_test = compute_sinr_eff(snr_test, gamma_test)
    chi_1 = chi_from_rho(rho_test)
    chi_2 = approx_chi(gamma_test, snr_test)
    print(f"SNR={snr_test}, Gamma={gamma_test} → ρ={rho_test:.4f}")
    print(f"  chi_from_rho(ρ) = {chi_1:.6f}")
    print(f"  approx_chi(Γ, SNR) = {chi_2:.6f}")
    print(f"  Consistent: {'✓' if np.isclose(chi_1, chi_2) else '✗'}")

    # 4. Test χ vs SNR curve
    print("\n--- Test 4: Chi vs SNR Curve ---")
    print(f"{'SNR(dB)':<10} {'SNR_lin':<12} {'SINR_eff':<12} {'Chi':<12}")
    print("-" * 50)
    for snr_db in [-20, -10, 0, 10, 20, 30, 40]:
        snr_lin = 10 ** (snr_db / 10)
        sinr_eff = compute_sinr_eff(snr_lin, g_eff)
        chi = chi_from_rho(sinr_eff)
        print(f"{snr_db:<10} {snr_lin:<12.4f} {sinr_eff:<12.4f} {chi:<12.6f}")

    # 5. Test Block Diagonal
    print("\n--- Test 5: Block Diagonal Approximation ---")
    N = 64
    x = np.arange(N)
    cov = np.exp(-0.1 * np.abs(x[:, None] - x[None, :]))
    J_full = np.linalg.inv(cov + 0.01 * np.eye(N))
    J_block = compute_fim_block_diag(J_full, block_size=16)
    err = relative_frobenius_error(J_full, J_block)
    print(f"  Matrix Size: {N}x{N}, Block Size: 16")
    print(f"  Approximation Error (Frobenius): {err * 100:.2f}%")

    # 6. Test Forbidden Region
    print("\n--- Test 6: Forbidden Region Check ---")
    pn_var_high = 0.1 * 100
    is_forbid, reason = check_forbidden_region(-50, pn_var_high)
    print(f"  High PN Case: {is_forbid} -> {reason}")

    pn_var_safe = 0.01 * 100
    is_forbid2, reason2 = check_forbidden_region(0, pn_var_safe)
    print(f"  Safe PN Case: {is_forbid2} -> {reason2}")

    print("\n" + "=" * 60)
    print("All tests completed.")