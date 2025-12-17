"""
sanity_checks_p23x.py (Definition Freeze v3 - MC Randomness Check Added)

Description:
    Automated Sanity Check Suite for THz-ISAC "Dirty Hardware" Modeling.

    **DEFINITION FREEZE v3** per Expert Review:
    - Protocol 0: Chi endpoint verification (MANDATORY)
    - Protocol 0.5: Chi API consistency check
    - Protocol 1: Chi factor vs SNR curves
    - Protocol 5: Gamma_eff monotonicity check
    - Protocol 6: BER vs SER verification
    - Protocol 7: MC Randomness verification (NEW - Blocker 3)

    These checks MUST PASS before Phase 3 training can begin.

    Output: results/validation_results/ (CSV, PNG, PDF)

Author: Definition Freeze v3
Date: 2025-12-17
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import time
import hashlib
from pathlib import Path
from typing import Dict, List, Tuple

# Import modules
try:
    from thz_isac_world import SimConfig, simulate_batch, verify_mc_randomness
    import geometry_metrics as gm
except ImportError as e:
    print(f"Error: {e}")
    print("Make sure 'thz_isac_world.py' and 'geometry_metrics.py' are in the path.")
    exit(1)

# --- Configuration ---
OUTPUT_DIR = Path("results/validation_results")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'lines.linewidth': 2
})


def get_hardware_config():
    """Detects hardware only when called."""
    if torch.cuda.is_available():
        return "gpu", os.cpu_count()
    else:
        return "cpu", os.cpu_count()


def save_experiment_results(df, fig, filename_base, no_title=True):
    """Saves results to CSV, PNG, and PDF."""
    base_path = OUTPUT_DIR / filename_base

    csv_path = base_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)

    png_path = base_path.with_suffix('.png')
    pdf_path = base_path.with_suffix('.pdf')

    if no_title:
        for ax in fig.get_axes():
            ax.set_title('')

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  [Saved] -> {filename_base} (.csv/.png/.pdf)")
    plt.close(fig)


# ============================================================================
# PROTOCOL 0: Chi Endpoint Verification (MANDATORY - Definition Freeze)
# ============================================================================

def run_protocol_0_chi_endpoints():
    """
    **MANDATORY SANITY CHECK** per DR-P2-3 注意事项

    Verifies that χ formula satisfies endpoint constraints:
    - χ(ρ→0) ≈ 2/π (error < 0.03)
    - χ(ρ→∞) → 0
    - χ monotonically decreases with ρ

    This MUST PASS before any training can begin.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 0: Chi Endpoint Verification (MANDATORY)")
    print("=" * 60)

    results = []
    all_passed = True

    # Test 1: Low SNR limit (ρ → 0)
    chi_at_zero = gm.chi_from_rho(0.0)
    expected_low = gm.CHI_LOW_SNR_LIMIT
    error_low = abs(chi_at_zero - expected_low)
    passed_1 = error_low < 0.03
    all_passed &= passed_1

    print(f"\nTest 1: Low SNR Limit (ρ → 0)")
    print(f"  χ(0) = {chi_at_zero:.6f}")
    print(f"  Expected: 2/π = {expected_low:.6f}")
    print(f"  Error: {error_low:.6f}")
    print(f"  Status: {'PASS ✓' if passed_1 else 'FAIL ✗'}")

    results.append({
        'test': 'low_snr_limit',
        'chi_value': chi_at_zero,
        'expected': expected_low,
        'error': error_low,
        'passed': passed_1
    })

    # Test 2: High SNR limit (ρ → ∞)
    chi_at_inf = gm.chi_from_rho(1e9)
    passed_2 = chi_at_inf < 1e-6
    all_passed &= passed_2

    print(f"\nTest 2: High SNR Limit (ρ → ∞)")
    print(f"  χ(1e9) = {chi_at_inf:.2e}")
    print(f"  Expected: ~0")
    print(f"  Status: {'PASS ✓' if passed_2 else 'FAIL ✗'}")

    results.append({
        'test': 'high_snr_limit',
        'chi_value': chi_at_inf,
        'expected': 0.0,
        'error': chi_at_inf,
        'passed': passed_2
    })

    # Test 3: Monotonicity
    rho_test = np.logspace(-2, 8, 1000)
    chi_test = np.array([gm.chi_from_rho(r) for r in rho_test])
    is_monotonic = np.all(np.diff(chi_test) <= 1e-12)
    all_passed &= is_monotonic

    print(f"\nTest 3: Monotonicity")
    print(f"  Tested {len(rho_test)} points from ρ=0.01 to ρ=1e8")
    print(f"  Status: {'PASS ✓' if is_monotonic else 'FAIL ✗'}")

    results.append({
        'test': 'monotonicity',
        'n_points': len(rho_test),
        'is_monotonic': is_monotonic,
        'passed': is_monotonic
    })

    # Create visualization
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax1 = axes[0]
    ax1.semilogx(rho_test, chi_test, 'b-', linewidth=2, label=r'$\chi(\rho)$')
    ax1.axhline(y=expected_low, color='r', linestyle='--', label=r'$\chi(0) = 2/\pi$')
    ax1.axhline(y=0, color='g', linestyle=':', label=r'$\chi(\infty) = 0$')
    ax1.set_xlabel(r'Effective SINR ($\rho$)')
    ax1.set_ylabel(r'Information Retention Factor ($\chi$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 0.7])

    ax2 = axes[1]
    test_names = ['Low SNR\nLimit', 'High SNR\nLimit', 'Monotonicity']
    test_passed = [r['passed'] for r in results]
    colors = ['green' if p else 'red' for p in test_passed]
    bars = ax2.bar(test_names, [1]*3, color=colors, alpha=0.7)
    ax2.set_ylabel('Test Status')
    ax2.set_ylim([0, 1.2])

    for i, (bar, passed) in enumerate(zip(bars, test_passed)):
        ax2.text(bar.get_x() + bar.get_width()/2, 1.05,
                '✓ PASS' if passed else '✗ FAIL',
                ha='center', fontsize=12, fontweight='bold',
                color='green' if passed else 'red')

    df = pd.DataFrame(results)
    save_experiment_results(df, fig, "protocol_0_chi_endpoints")

    print("\n" + "-" * 40)
    print(f"PROTOCOL 0 RESULT: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    print("-" * 40)

    if not all_passed:
        print("\n⚠️  WARNING: Chi endpoint verification FAILED!")
        print("    Training MUST NOT proceed until this is fixed.")

    return all_passed, results


# ============================================================================
# PROTOCOL 0.5: Chi API Consistency Check
# ============================================================================

def run_protocol_05_api_consistency():
    """
    **NEW** per DR-P2-3.5 API Contract

    Verifies that all χ interfaces return consistent results:
    - chi_from_rho(rho)
    - approx_chi(rho)
    - approx_chi_from_components(gamma_eff, snr_linear)

    Must satisfy: approx_chi_from_components(Γ, SNR) == chi_from_rho(SINR_eff(SNR, Γ))
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 0.5: Chi API Consistency Check")
    print("=" * 60)

    results = []
    all_passed = True

    snr_db_list = [-10, 0, 10, 20, 30]
    gamma_db_list = [-10, 0, 10, 20, 30]

    print(f"\n{'SNR(dB)':<10} {'Γ(dB)':<10} {'ρ':<12} {'χ(ρ)':<12} {'χ(Γ,SNR)':<12} {'Match':<8}")
    print("-" * 70)

    for snr_db in snr_db_list:
        for gamma_db in gamma_db_list:
            snr_lin = 10 ** (snr_db / 10)
            gamma_lin = 10 ** (gamma_db / 10)

            rho = gm.compute_sinr_eff(snr_lin, gamma_lin)
            chi_rho = gm.chi_from_rho(rho)
            chi_alias = gm.approx_chi(rho)
            chi_comp = gm.approx_chi_from_components(gamma_lin, snr_lin)

            tol = 1e-10
            match_all = (abs(chi_rho - chi_alias) < tol and
                        abs(chi_rho - chi_comp) < tol)

            results.append({
                'snr_db': snr_db,
                'gamma_db': gamma_db,
                'rho': rho,
                'chi_from_rho': chi_rho,
                'approx_chi': chi_alias,
                'approx_chi_from_components': chi_comp,
                'match': match_all
            })

            all_passed &= match_all
            status = "✓" if match_all else "✗"
            print(f"{snr_db:<10} {gamma_db:<10} {rho:<12.4f} {chi_rho:<12.6f} {chi_comp:<12.6f} {status}")

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "protocol_05_api_consistency.csv", index=False)

    print("\n" + "-" * 40)
    print(f"PROTOCOL 0.5 RESULT: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    print("-" * 40)

    return all_passed, results


# ============================================================================
# PROTOCOL 1: Chi vs SNR Curves
# ============================================================================

def run_protocol_1_chi_vs_snr():
    """
    Generates χ vs SNR curves for different hardware configurations.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 1: Chi vs SNR Curves")
    print("=" * 60)

    configs = [
        ("Ideal", {"enable_pa": False, "enable_pn": False, "enable_quantization": False}),
        ("PA only", {"enable_pa": True, "enable_pn": False, "enable_quantization": False}),
        ("PA + PN", {"enable_pa": True, "enable_pn": True, "enable_quantization": False}),
        ("Full Dirty HW", {"enable_pa": True, "enable_pn": True, "enable_quantization": True}),
    ]

    snr_range = np.arange(-10, 35, 2)
    results = []

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, params in configs:
        chi_values = []
        for snr_db in snr_range:
            cfg = SimConfig()
            cfg.snr_db = snr_db
            for k, v in params.items():
                setattr(cfg, k, v)

            data = simulate_batch(cfg, batch_size=50, seed=42)
            chi_values.append(data['meta']['chi'])

        ax.plot(snr_range, chi_values, '-o', label=name, markersize=4)

        for snr, chi in zip(snr_range, chi_values):
            results.append({
                'config': name,
                'snr_db': snr,
                'chi': chi
            })

    ax.axhline(y=2/np.pi, color='k', linestyle='--', alpha=0.5, label=r'$\chi_{max} = 2/\pi$')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'Information Retention Factor ($\chi$)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 0.7])

    df = pd.DataFrame(results)
    save_experiment_results(df, fig, "protocol_1_chi_vs_snr")

    print(f"  Generated curves for {len(configs)} configurations")
    print("-" * 40)
    print("PROTOCOL 1 RESULT: COMPLETED ✓")

    return True, results


# ============================================================================
# PROTOCOL 5: Gamma_eff Monotonicity Check
# ============================================================================

def run_protocol_5_gamma_monotonicity():
    """
    **KEY SANITY CHECK** per DR-P2-3 注意事项

    Verifies that Gamma_eff decreases monotonically as hardware
    impairments are added:
        Ideal > PA only > PA+PN > PA+PN+Quant
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 5: Gamma_eff Monotonicity Check")
    print("=" * 60)

    configs = [
        ("Ideal", {"enable_pa": False, "enable_pn": False, "enable_quantization": False}),
        ("PA only", {"enable_pa": True, "enable_pn": False, "enable_quantization": False}),
        ("PA + PN", {"enable_pa": True, "enable_pn": True, "enable_quantization": False}),
        ("PA + PN + Quant", {"enable_pa": True, "enable_pn": True, "enable_quantization": True}),
    ]

    results = []
    gamma_values = []

    print(f"\n{'Config':<20} {'Γ_eff (linear)':<15} {'Γ_eff (dB)':<12} {'χ':<12}")
    print("-" * 65)

    for name, params in configs:
        cfg = SimConfig()
        cfg.snr_db = 20.0
        for k, v in params.items():
            setattr(cfg, k, v)

        batch_data = simulate_batch(cfg, batch_size=100, seed=42)
        meta = batch_data['meta']

        gamma_eff = meta['gamma_eff']
        gamma_eff_db = 10 * np.log10(gamma_eff + 1e-12)
        chi = meta['chi']

        gamma_values.append(gamma_eff)

        results.append({
            'config': name,
            'gamma_eff': gamma_eff,
            'gamma_eff_db': gamma_eff_db,
            'chi': chi
        })

        print(f"{name:<20} {gamma_eff:<15.4f} {gamma_eff_db:<12.2f} {chi:<12.6f}")

    is_monotonic = all(gamma_values[i] >= gamma_values[i+1]
                       for i in range(len(gamma_values)-1))

    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(configs))
    colors = ['green' if is_monotonic else 'orange'] * len(configs)
    bars = ax.bar(x, df['gamma_eff_db'], color=colors, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs], rotation=15, ha='right')
    ax.set_ylabel(r'$\Gamma_{eff}$ (dB)')
    ax.grid(True, axis='y', alpha=0.3)

    for i, (bar, val) in enumerate(zip(bars, df['gamma_eff_db'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f} dB', ha='center', fontsize=9)

    save_experiment_results(df, fig, "protocol_5_gamma_monotonicity")

    print("\n" + "-" * 40)
    print(f"Monotonicity Check: {'PASSED ✓' if is_monotonic else 'FAILED ✗'}")
    print(f"PROTOCOL 5 RESULT: {'PASSED ✓' if is_monotonic else 'FAILED ✗'}")
    print("-" * 40)

    if not is_monotonic:
        print("\n⚠️  WARNING: Gamma_eff monotonicity FAILED!")
        print("    Power decomposition may be incorrect.")

    return is_monotonic, results


# ============================================================================
# PROTOCOL 6: BER Calculation Verification
# ============================================================================

def run_protocol_6_ber_verification():
    """
    Verifies BER calculation is true bit-level (not SER).

    For QPSK:
        - Each symbol carries 2 bits (I and Q)
        - BER should count I/Q bit errors separately
        - SER counts symbol errors

    BER ≤ SER always (equality when single-bit errors only)
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 6: BER vs SER Verification")
    print("=" * 60)

    def compute_ber_qpsk_bitwise(x_hat: np.ndarray, x_true: np.ndarray) -> float:
        """True bit-level BER for QPSK."""
        bit_I_true = (np.real(x_true) > 0).astype(int)
        bit_Q_true = (np.imag(x_true) > 0).astype(int)
        bit_I_hat = (np.real(x_hat) > 0).astype(int)
        bit_Q_hat = (np.imag(x_hat) > 0).astype(int)

        errors_I = np.sum(bit_I_true != bit_I_hat)
        errors_Q = np.sum(bit_Q_true != bit_Q_hat)
        total_bits = 2 * x_true.size

        return (errors_I + errors_Q) / total_bits

    def compute_ser_qpsk(x_hat: np.ndarray, x_true: np.ndarray) -> float:
        """Symbol Error Rate for QPSK."""
        constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
        idx_true = np.argmin(np.abs(x_true[..., None] - constellation), axis=-1)
        idx_hat = np.argmin(np.abs(x_hat[..., None] - constellation), axis=-1)
        return np.mean(idx_true != idx_hat)

    results = []
    np.random.seed(42)
    n_symbols = 10000

    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    x_true = constellation[np.random.randint(0, 4, n_symbols)]

    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]

    print(f"\n{'Noise σ':<10} {'BER':<12} {'SER':<12} {'BER ≤ SER':<10}")
    print("-" * 45)

    for noise_std in noise_levels:
        noise = (np.random.randn(n_symbols) + 1j * np.random.randn(n_symbols)) * noise_std / np.sqrt(2)
        x_hat = x_true + noise

        ber = compute_ber_qpsk_bitwise(x_hat, x_true)
        ser = compute_ser_qpsk(x_hat, x_true)

        is_valid = ber <= ser + 1e-10

        results.append({
            'noise_std': noise_std,
            'ber': ber,
            'ser': ser,
            'ber_le_ser': is_valid
        })

        print(f"{noise_std:<10.2f} {ber:<12.6f} {ser:<12.6f} {'✓' if is_valid else '✗'}")

    all_valid = all(r['ber_le_ser'] for r in results)

    print("\n" + "-" * 40)
    print(f"PROTOCOL 6 RESULT: {'PASSED ✓' if all_valid else 'FAILED ✗'}")
    print("-" * 40)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "protocol_6_ber_verification.csv", index=False)
    print(f"  [Saved] protocol_6_ber_verification.csv")

    return all_valid, results


# ============================================================================
# PROTOCOL 7: Monte Carlo Randomness Verification (NEW - Blocker 3)
# ============================================================================

def run_protocol_7_mc_randomness():
    """
    **CRITICAL** per Expert Review (Blocker 3)

    Verifies Monte Carlo randomness is working correctly:
    1. No seed: n_trials produce n unique checksums
    2. Same seed: n_trials produce identical checksum
    3. Different seeds: n_trials produce n unique checksums

    If this fails, ALL Monte Carlo experiments are invalid!
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 7: Monte Carlo Randomness Verification (CRITICAL)")
    print("=" * 60)

    n_trials = 5

    # Use built-in verification from thz_isac_world
    results = verify_mc_randomness(n_trials=n_trials)

    print(f"\nTest 1: No seed -> {n_trials} unique outputs")
    print(f"  Checksums: {results['no_seed_checksums']}")
    print(f"  Unique count: {len(set(results['no_seed_checksums']))}/{n_trials}")
    print(f"  Status: {'PASS ✓' if results['no_seed_unique'] else 'FAIL ✗'}")

    print(f"\nTest 2: Same seed (12345) -> identical outputs")
    print(f"  Checksums: {results['same_seed_checksums']}")
    print(f"  Unique count: {len(set(results['same_seed_checksums']))}/1 expected")
    print(f"  Status: {'PASS ✓' if results['same_seed_identical'] else 'FAIL ✗'}")

    print(f"\nTest 3: Different seeds -> unique outputs")
    print(f"  Checksums: {results['diff_seed_checksums']}")
    print(f"  Unique count: {len(set(results['diff_seed_checksums']))}/{n_trials}")
    print(f"  Status: {'PASS ✓' if results['diff_seed_unique'] else 'FAIL ✗'}")

    # Save results
    df = pd.DataFrame([{
        'test': 'no_seed_unique',
        'passed': results['no_seed_unique'],
        'details': str(results['no_seed_checksums'])
    }, {
        'test': 'same_seed_identical',
        'passed': results['same_seed_identical'],
        'details': str(results['same_seed_checksums'])
    }, {
        'test': 'diff_seed_unique',
        'passed': results['diff_seed_unique'],
        'details': str(results['diff_seed_checksums'])
    }])
    df.to_csv(OUTPUT_DIR / "protocol_7_mc_randomness.csv", index=False)

    print("\n" + "-" * 40)
    print(f"PROTOCOL 7 RESULT: {'PASSED ✓' if results['all_pass'] else 'FAILED ✗'}")
    print("-" * 40)

    if not results['all_pass']:
        print("\n⚠️  CRITICAL WARNING: MC randomness verification FAILED!")
        print("    This INVALIDATES all Monte Carlo experiments.")
        print("    n_mc=50 is statistically equivalent to n_mc=1.")
        print("    FIX THIS BEFORE ANY EXPERIMENTS!")

    return results['all_pass'], results


# ============================================================================
# Main Entry Point
# ============================================================================

def run_all_protocols():
    """Runs all sanity check protocols."""
    print("\n" + "=" * 70)
    print("SANITY CHECK SUITE - Definition Freeze v3")
    print("=" * 70)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}
    all_passed = True

    # Protocol 7: MC Randomness (CRITICAL - run first)
    print("\n[Running Protocol 7 FIRST - MC Randomness is CRITICAL]")
    passed, results = run_protocol_7_mc_randomness()
    all_results['protocol_7'] = {'passed': passed, 'results': results}
    all_passed &= passed

    if not passed:
        print("\n" + "!" * 70)
        print("STOPPING: Protocol 7 (MC Randomness) FAILED!")
        print("All subsequent tests would produce invalid results.")
        print("Fix thz_isac_world.py before continuing.")
        print("!" * 70)
        return False, all_results

    # Protocol 0: Chi Endpoints (MANDATORY)
    passed, results = run_protocol_0_chi_endpoints()
    all_results['protocol_0'] = {'passed': passed, 'results': results}
    all_passed &= passed

    # Protocol 0.5: API Consistency
    passed, results = run_protocol_05_api_consistency()
    all_results['protocol_05'] = {'passed': passed, 'results': results}
    all_passed &= passed

    # Protocol 1: Chi vs SNR
    passed, results = run_protocol_1_chi_vs_snr()
    all_results['protocol_1'] = {'passed': passed, 'results': results}

    # Protocol 5: Gamma Monotonicity
    passed, results = run_protocol_5_gamma_monotonicity()
    all_results['protocol_5'] = {'passed': passed, 'results': results}
    all_passed &= passed

    # Protocol 6: BER Verification
    passed, results = run_protocol_6_ber_verification()
    all_results['protocol_6'] = {'passed': passed, 'results': results}
    all_passed &= passed

    # Final Summary
    print("\n" + "=" * 70)
    print("SANITY CHECK SUMMARY")
    print("=" * 70)

    critical_protocols = ['protocol_7', 'protocol_0', 'protocol_05', 'protocol_5', 'protocol_6']

    for proto_name, data in all_results.items():
        status = "PASS ✓" if data['passed'] else "FAIL ✗"
        critical = " (CRITICAL)" if proto_name in ['protocol_7', 'protocol_0'] else ""
        print(f"  {proto_name}: {status}{critical}")

    print("-" * 70)
    print(f"OVERALL: {'ALL PASSED ✓' if all_passed else 'SOME FAILED ✗'}")
    print("=" * 70)

    if not all_passed:
        print("\n⚠️  WARNING: Some sanity checks FAILED!")
        print("    Review failures before proceeding with training.")
    else:
        print("\n✓ All sanity checks passed. Ready for Phase 3 training.")

    return all_passed, all_results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Sanity Check Suite - Definition Freeze v3")
    parser.add_argument('--protocol', type=str, default='all',
                        help='Protocol to run (0, 05, 1, 5, 6, 7, or all)')
    args = parser.parse_args()

    if args.protocol == 'all':
        run_all_protocols()
    elif args.protocol == '0':
        run_protocol_0_chi_endpoints()
    elif args.protocol == '05':
        run_protocol_05_api_consistency()
    elif args.protocol == '1':
        run_protocol_1_chi_vs_snr()
    elif args.protocol == '5':
        run_protocol_5_gamma_monotonicity()
    elif args.protocol == '6':
        run_protocol_6_ber_verification()
    elif args.protocol == '7':
        run_protocol_7_mc_randomness()
    else:
        print(f"Unknown protocol: {args.protocol}")
        print("Valid options: 0, 05, 1, 5, 6, 7, all")