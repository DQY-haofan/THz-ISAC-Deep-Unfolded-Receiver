"""
geometry_metrics.py (Definition Freeze v3 - Expert Approved)

Description:
    Geometric quantities for BCRLB-regularized deep unfolding.

    **DEFINITION FREEZE v3** per Expert Review:
    - CANONICAL Interface: chi_from_rho(rho) is the ONLY source of truth
    - Convenience Interface: approx_chi_from_components(gamma_eff, snr_linear)
    - Alias: approx_chi(rho_eff) -> chi_from_rho(rho_eff) for backward compat

    Formula (Frozen):
        ρ_eff = SINR_eff = 1 / (SNR^-1 + Γ_eff^-1)
        χ(ρ) = (2/π) / (1 + κρ),  κ = 1 - 2/π ≈ 0.3634

    Endpoint Constraints (MUST PASS Protocol 0):
        χ(ρ→0) = 2/π ≈ 0.6366 (low SNR limit)
        χ(ρ→∞) → 0 (high SNR limit)
        χ monotonically decreasing

Author: Definition Freeze v3
Date: 2025-12-17
"""

import numpy as np
from typing import Dict, Tuple, Optional, Union
import warnings

# =============================================================================
# FROZEN CONSTANTS (DO NOT MODIFY)
# =============================================================================

CHI_LOW_SNR_LIMIT = 2.0 / np.pi  # ≈ 0.6366197723675814
KAPPA = 1.0 - CHI_LOW_SNR_LIMIT  # ≈ 0.36338022763241865

# Sanity check thresholds
CHI_ZERO_TOLERANCE = 0.03  # χ(0) must be within this of 2/π
CHI_INF_THRESHOLD = 1e-6   # χ(∞) must be below this


# =============================================================================
# 1. CHI FACTOR COMPUTATION (Definition Freeze v3)
# =============================================================================

def chi_from_rho(rho: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    CANONICAL χ(ρ) function - THE source of truth.

    Formula:
        χ(ρ) = (2/π) / (1 + κρ)
        where κ = 1 - 2/π ≈ 0.3634

    Endpoint Behavior:
        χ(0) = 2/π ≈ 0.6366 (information MOST retained at low SNR)
        χ(∞) → 0 (quantization noise dominates at high SNR)

    This captures the "arcsine law" behavior of 1-bit quantization:
    - At low SNR, thermal noise dominates → 1-bit loses little info
    - At high SNR, signal is clipped → 1-bit loses significant info

    Args:
        rho: Effective SINR (linear scale), scalar or array

    Returns:
        chi: Information retention factor in [0, 2/π]
    """
    rho = np.asarray(rho)
    rho_safe = np.maximum(rho, 0.0)  # Ensure non-negative
    chi = CHI_LOW_SNR_LIMIT / (1.0 + KAPPA * rho_safe)
    return float(chi) if chi.ndim == 0 else chi


def approx_chi(rho_eff: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    ALIAS for chi_from_rho - backward compatibility.

    Note: This is a SINGLE-PARAMETER interface.
    If you have (gamma_eff, snr_linear), use approx_chi_from_components instead.

    Args:
        rho_eff: Effective SINR (linear scale)

    Returns:
        chi: Information retention factor
    """
    return chi_from_rho(rho_eff)


def approx_chi_from_components(gamma_eff: float, snr_linear: float) -> float:
    """
    TWO-PARAMETER convenience wrapper.

    Computes ρ_eff from components, then calls chi_from_rho.

    Formula:
        ρ_eff = 1 / (1/SNR + 1/Γ_eff) = (SNR * Γ_eff) / (SNR + Γ_eff)
        χ = chi_from_rho(ρ_eff)

    Args:
        gamma_eff: Hardware distortion ratio (linear)
        snr_linear: Thermal SNR (linear)

    Returns:
        chi: Information retention factor
    """
    rho_eff = compute_sinr_eff(snr_linear, gamma_eff)
    return chi_from_rho(rho_eff)


# =============================================================================
# 2. SINR / GAMMA COMPUTATION
# =============================================================================

def compute_sinr_eff(snr_linear: float, gamma_eff: float) -> float:
    """
    Computes effective SINR via harmonic mean.

    Formula:
        ρ_eff = 1 / (1/SNR + 1/Γ_eff)

    Physical Interpretation:
        - Combines thermal noise (1/SNR) and hardware distortion (1/Γ_eff)
        - Limited by the weaker of the two

    Args:
        snr_linear: Thermal SNR (linear scale)
        gamma_eff: Hardware distortion ratio (linear scale)

    Returns:
        rho_eff: Effective SINR (linear scale)
    """
    # Handle edge cases
    snr_safe = max(snr_linear, 1e-12)
    gamma_safe = max(gamma_eff, 1e-12)

    # Harmonic combination
    rho_eff = 1.0 / ((1.0 / snr_safe) + (1.0 / gamma_safe))

    return rho_eff


def estimate_gamma_eff(sim_stats: Dict[str, float]) -> float:
    """
    First-principles Γ_eff estimation from power decomposition.

    Formula (per DR-P2-2.5):
        Γ_eff = P_signal / (P_pa + P_pn + P_quant)

    This is the ONLY correct way to compute Γ_eff.
    DO NOT use magic values like 0.5 or 100.

    Args:
        sim_stats: Dict with keys:
            - P_signal: Main signal power
            - P_pa_distortion: PA Bussgang residual power
            - P_phase_noise: Phase noise equivalent power
            - P_quantization_loss: Quantization Bussgang residual

    Returns:
        gamma_eff: Hardware distortion ratio (linear scale)
    """
    P_sig = sim_stats.get('P_signal', 1.0)
    P_pa = sim_stats.get('P_pa_distortion', 0.0)
    P_pn = sim_stats.get('P_phase_noise', 0.0)
    P_quant = sim_stats.get('P_quantization_loss', 0.0)

    # Total distortion power
    P_dist = P_pa + P_pn + P_quant

    # Compute Gamma_eff
    if P_dist < 1e-12:
        # No distortion → infinite Γ_eff (ideal hardware)
        gamma_eff = 1e9
    else:
        gamma_eff = P_sig / P_dist

    return gamma_eff


# =============================================================================
# 3. FIM / CRLB UTILITIES
# =============================================================================

def compute_fim_block_diag(H: np.ndarray, snr: float, chi: float,
                           block_size: int = 32) -> np.ndarray:
    """
    Block-diagonal FIM approximation for O(NK²) complexity.

    Args:
        H: Channel matrix [N, K] or diag [N]
        snr: SNR (linear)
        chi: Information retention factor
        block_size: Size of diagonal blocks

    Returns:
        FIM: [K, K] approximate Fisher information matrix
    """
    H = np.atleast_2d(H)
    if H.ndim == 1:
        H = np.diag(H)

    N, K = H.shape

    # χ-scaled SNR
    eff_snr = chi * snr

    # Simple FIM: J = eff_snr * H^H @ H
    FIM = eff_snr * (np.conj(H.T) @ H)

    return FIM


def compute_crlb_from_fim(FIM: np.ndarray, regularization: float = 1e-6) -> np.ndarray:
    """
    Computes CRLB as inverse of FIM with regularization.

    Args:
        FIM: Fisher Information Matrix [K, K]
        regularization: Tikhonov regularization for stability

    Returns:
        CRLB: [K, K] Cramér-Rao Lower Bound matrix
    """
    K = FIM.shape[0]
    FIM_reg = FIM + regularization * np.eye(K)

    try:
        CRLB = np.linalg.inv(FIM_reg)
    except np.linalg.LinAlgError:
        warnings.warn("FIM inversion failed, using pseudo-inverse")
        CRLB = np.linalg.pinv(FIM_reg)

    return CRLB


def compute_log_det_fim(FIM: np.ndarray, regularization: float = 1e-6) -> float:
    """
    Computes log-determinant of FIM for volume computation.

    Args:
        FIM: Fisher Information Matrix
        regularization: For numerical stability

    Returns:
        log_det: log|FIM|
    """
    FIM_reg = FIM + regularization * np.eye(FIM.shape[0])
    sign, log_det = np.linalg.slogdet(FIM_reg)

    if sign <= 0:
        warnings.warn("FIM has non-positive determinant")
        return -1e10

    return log_det


def hutchinson_trace_estimator(FIM: np.ndarray, n_samples: int = 10,
                               rng: Optional[np.random.Generator] = None) -> float:
    """
    Stochastic trace estimation using Hutchinson's method.

    Useful for large FIM where exact trace is expensive.

    Args:
        FIM: Square matrix [K, K]
        n_samples: Number of random vectors
        rng: Random number generator

    Returns:
        trace_est: Estimated trace of FIM
    """
    if rng is None:
        rng = np.random.default_rng()

    K = FIM.shape[0]
    trace_sum = 0.0

    for _ in range(n_samples):
        # Rademacher random vector
        z = rng.choice([-1.0, 1.0], size=K)
        trace_sum += z @ FIM @ z

    return trace_sum / n_samples


# =============================================================================
# 4. FORBIDDEN REGION CHECK
# =============================================================================

def check_forbidden_region(gamma_eff: float, pn_var: float,
                           gamma_threshold_db: float = -20.0,
                           pn_threshold: float = np.pi) -> Dict[str, bool]:
    """
    Checks if operating point is in "forbidden region".

    Forbidden conditions (per DR-P2-3.5):
        1. Γ_eff < -20 dB (hardware too noisy)
        2. Phase noise variance > π (phase wrapping)

    Args:
        gamma_eff: Hardware distortion ratio (linear)
        pn_var: Phase noise variance (radians²)
        gamma_threshold_db: Minimum acceptable Γ_eff
        pn_threshold: Maximum acceptable PN variance

    Returns:
        Dict with:
            - in_forbidden: True if in forbidden region
            - gamma_ok: True if Γ_eff acceptable
            - pn_ok: True if PN variance acceptable
    """
    gamma_db = 10 * np.log10(max(gamma_eff, 1e-12))
    gamma_ok = gamma_db >= gamma_threshold_db
    pn_ok = pn_var <= pn_threshold

    return {
        'in_forbidden': not (gamma_ok and pn_ok),
        'gamma_ok': gamma_ok,
        'pn_ok': pn_ok,
        'gamma_eff_db': gamma_db,
        'pn_var': pn_var
    }


# =============================================================================
# 5. VERIFICATION UTILITIES
# =============================================================================

def verify_chi_endpoints() -> Dict[str, bool]:
    """
    Verifies χ formula satisfies endpoint constraints.

    This is Protocol 0 from sanity checks - MUST PASS before training.

    Returns:
        Dict with test results
    """
    results = {}

    # Test 1: χ(0) ≈ 2/π
    chi_zero = chi_from_rho(0.0)
    error_zero = abs(chi_zero - CHI_LOW_SNR_LIMIT)
    results['chi_at_zero'] = chi_zero
    results['chi_zero_error'] = error_zero
    results['chi_zero_pass'] = error_zero < CHI_ZERO_TOLERANCE

    # Test 2: χ(∞) → 0
    chi_inf = chi_from_rho(1e9)
    results['chi_at_inf'] = chi_inf
    results['chi_inf_pass'] = chi_inf < CHI_INF_THRESHOLD

    # Test 3: Monotonicity
    rho_test = np.logspace(-2, 8, 1000)
    chi_test = chi_from_rho(rho_test)
    is_monotonic = np.all(np.diff(chi_test) <= 1e-12)
    results['monotonic_pass'] = is_monotonic

    # Overall
    results['all_pass'] = (results['chi_zero_pass'] and
                          results['chi_inf_pass'] and
                          results['monotonic_pass'])

    return results


def verify_chi_api_consistency(snr_db: float = 20.0, gamma_db: float = 10.0) -> Dict:
    """
    Verifies all χ interfaces return consistent values.

    Tests:
        chi_from_rho(rho) == approx_chi(rho) == approx_chi_from_components(gamma, snr)

    Args:
        snr_db, gamma_db: Test values in dB

    Returns:
        Dict with consistency check results
    """
    snr_lin = 10 ** (snr_db / 10)
    gamma_lin = 10 ** (gamma_db / 10)

    # Compute rho
    rho = compute_sinr_eff(snr_lin, gamma_lin)

    # Three interfaces
    chi_1 = chi_from_rho(rho)
    chi_2 = approx_chi(rho)
    chi_3 = approx_chi_from_components(gamma_lin, snr_lin)

    # Check consistency
    tol = 1e-10
    consistent_12 = abs(chi_1 - chi_2) < tol
    consistent_23 = abs(chi_2 - chi_3) < tol
    consistent_13 = abs(chi_1 - chi_3) < tol

    return {
        'rho': rho,
        'chi_from_rho': chi_1,
        'approx_chi': chi_2,
        'approx_chi_from_components': chi_3,
        'all_consistent': consistent_12 and consistent_23 and consistent_13,
        'snr_db': snr_db,
        'gamma_db': gamma_db
    }


# =============================================================================
# 6. MAIN - Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("geometry_metrics.py - Definition Freeze v3 Self-Test")
    print("=" * 60)

    # Test 1: Endpoint verification
    print("\n[Test 1] Chi Endpoint Verification")
    results = verify_chi_endpoints()
    print(f"  χ(0) = {results['chi_at_zero']:.6f} (expected: {CHI_LOW_SNR_LIMIT:.6f})")
    print(f"  χ(0) error = {results['chi_zero_error']:.6e} ({'PASS' if results['chi_zero_pass'] else 'FAIL'})")
    print(f"  χ(1e9) = {results['chi_at_inf']:.2e} ({'PASS' if results['chi_inf_pass'] else 'FAIL'})")
    print(f"  Monotonic: {'PASS' if results['monotonic_pass'] else 'FAIL'}")
    print(f"  Overall: {'ALL PASS ✓' if results['all_pass'] else 'FAILED ✗'}")

    # Test 2: API consistency
    print("\n[Test 2] API Consistency Check")
    for snr_db in [0, 10, 20, 30]:
        for gamma_db in [0, 10, 20]:
            api_result = verify_chi_api_consistency(snr_db, gamma_db)
            status = "✓" if api_result['all_consistent'] else "✗"
            print(f"  SNR={snr_db:3d}dB, Γ={gamma_db:3d}dB: "
                  f"χ={api_result['chi_from_rho']:.4f} {status}")

    # Test 3: Gamma_eff estimation
    print("\n[Test 3] Gamma_eff Estimation")
    test_stats = {
        'P_signal': 1.0,
        'P_pa_distortion': 0.01,
        'P_phase_noise': 0.005,
        'P_quantization_loss': 0.02
    }
    gamma = estimate_gamma_eff(test_stats)
    print(f"  Test sim_stats: {test_stats}")
    print(f"  Estimated Γ_eff = {gamma:.4f} ({10*np.log10(gamma):.2f} dB)")

    # Test 4: Forbidden region
    print("\n[Test 4] Forbidden Region Check")
    fr1 = check_forbidden_region(gamma_eff=10.0, pn_var=0.1)
    fr2 = check_forbidden_region(gamma_eff=0.01, pn_var=0.1)
    fr3 = check_forbidden_region(gamma_eff=10.0, pn_var=4.0)
    print(f"  Γ=10, PN_var=0.1: {'FORBIDDEN' if fr1['in_forbidden'] else 'OK'}")
    print(f"  Γ=0.01, PN_var=0.1: {'FORBIDDEN' if fr2['in_forbidden'] else 'OK'} (low Γ)")
    print(f"  Γ=10, PN_var=4.0: {'FORBIDDEN' if fr3['in_forbidden'] else 'OK'} (high PN)")

    print("\n" + "=" * 60)
    print("Self-test complete.")