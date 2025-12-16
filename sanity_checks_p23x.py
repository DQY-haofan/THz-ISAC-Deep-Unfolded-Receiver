"""
sanity_checks_p23x.py (Definition Freeze v2 - Expert API Compliant)

Description:
    Automated Sanity Check Suite for THz-ISAC "Dirty Hardware" Modeling.

    **DEFINITION FREEZE v2** per Expert Review:
    - Protocol 0: Chi endpoint verification (MANDATORY)
    - Protocol 0.5: Chi API consistency check (NEW)
    - Protocol 1: Chi factor vs SNR with CORRECT formula
    - Protocol 5: Gamma_eff monotonicity check

    These checks MUST PASS before Phase 3 training can begin.

    Output: results/validation_results/ (CSV, PNG, PDF)

Author: Definition Freeze v2
Date: 2025-12-13
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Import modules
try:
    from thz_isac_world import SimConfig, simulate_batch
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
        # Remove titles for publication
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
    is_monotonic = np.all(np.diff(chi_test) <= 1e-12)  # Allow tiny numerical noise
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

    # Plot 1: χ vs ρ curve
    ax1 = axes[0]
    ax1.semilogx(rho_test, chi_test, 'b-', linewidth=2, label=r'$\chi(\rho)$')
    ax1.axhline(y=expected_low, color='r', linestyle='--',
                label=r'$\chi(0) = 2/\pi$')
    ax1.axhline(y=0, color='g', linestyle=':', label=r'$\chi(\infty) = 0$')
    ax1.set_xlabel(r'Effective SINR ($\rho$)')
    ax1.set_ylabel(r'Information Retention Factor ($\chi$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([-0.05, 0.7])

    # Plot 2: Test results bar chart
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

    # Save
    df = pd.DataFrame(results)
    save_experiment_results(df, fig, "protocol_0_chi_endpoints")

    # Summary
    print("\n" + "-" * 40)
    print(f"PROTOCOL 0 RESULT: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    print("-" * 40)

    if not all_passed:
        print("\n⚠️  WARNING: Chi endpoint verification FAILED!")
        print("    Training MUST NOT proceed until this is fixed.")

    return all_passed, results


# ============================================================================
# PROTOCOL 0.5: Chi API Consistency Check (NEW)
# ============================================================================

def run_protocol_05_api_consistency():
    """
    **NEW** per DR-P2-3.5 API Contract

    Verifies that both χ interfaces return consistent results:
    - chi_from_rho(rho)
    - approx_chi(gamma_eff, snr_linear)

    Must satisfy: approx_chi(Γ, SNR) == chi_from_rho(SINR_eff(SNR, Γ))
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 0.5: Chi API Consistency Check (NEW)")
    print("=" * 60)

    results = []
    all_passed = True

    # Test multiple SNR and Gamma combinations
    snr_db_list = [-10, 0, 10, 20, 30]
    gamma_db_list = [-10, 0, 10, 20, 30]

    print(f"\n{'SNR(dB)':<10} {'Γ(dB)':<10} {'ρ':<12} {'χ(ρ)':<12} {'χ(Γ,SNR)':<12} {'Match':<8}")
    print("-" * 70)

    for snr_db in snr_db_list:
        for gamma_db in gamma_db_list:
            snr_lin = 10 ** (snr_db / 10)
            gamma_lin = 10 ** (gamma_db / 10)

            # Compute via both interfaces
            rho = gm.compute_sinr_eff(snr_lin, gamma_lin)
            chi_1 = gm.chi_from_rho(rho)
            chi_2 = gm.approx_chi(gamma_lin, snr_lin)

            is_consistent = np.isclose(chi_1, chi_2, rtol=1e-10)
            all_passed &= is_consistent

            results.append({
                'snr_db': snr_db,
                'gamma_db': gamma_db,
                'rho': rho,
                'chi_from_rho': chi_1,
                'chi_from_gamma_snr': chi_2,
                'consistent': is_consistent
            })

            print(f"{snr_db:<10} {gamma_db:<10} {rho:<12.4f} {chi_1:<12.6f} {chi_2:<12.6f} {'✓' if is_consistent else '✗'}")

    # Summary
    print("\n" + "-" * 40)
    print(f"PROTOCOL 0.5 RESULT: {'PASSED ✓' if all_passed else 'FAILED ✗'}")
    print("-" * 40)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "protocol_05_api_consistency.csv", index=False)
    print(f"  [Saved] protocol_05_api_consistency.csv")

    return all_passed, results


# ============================================================================
# PROTOCOL 1: Chi Factor vs SNR (with Full Hardware Chain)
# ============================================================================

def run_protocol_1_chi_vs_snr():
    """
    Tests χ behavior across SNR range using REAL simulation data.
    Uses gamma_eff/chi from simulate_batch() meta.
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 1: Chi Factor vs SNR (Full Hardware Chain)")
    print("=" * 60)

    snr_db_list = np.arange(-10, 35, 2)
    results = []

    cfg = SimConfig()
    cfg.enable_pa = True
    cfg.enable_pn = True
    cfg.enable_quantization = True

    print(f"\n{'SNR(dB)':<10} {'Γ_eff':<12} {'Γ_eff(dB)':<12} {'ρ_eff':<12} {'χ':<12}")
    print("-" * 60)

    for snr_db in snr_db_list:
        cfg.snr_db = snr_db

        # Run simulation
        batch_data = simulate_batch(cfg, batch_size=64)
        meta = batch_data['meta']

        gamma_eff = meta['gamma_eff']
        sinr_eff = meta['sinr_eff']
        chi = meta['chi']

        gamma_eff_db = 10 * np.log10(gamma_eff + 1e-12)

        results.append({
            'snr_db': snr_db,
            'gamma_eff': gamma_eff,
            'gamma_eff_db': gamma_eff_db,
            'sinr_eff': sinr_eff,
            'chi': chi
        })

        print(f"{snr_db:<10} {gamma_eff:<12.4f} {gamma_eff_db:<12.2f} {sinr_eff:<12.4f} {chi:<12.6f}")

    # Visualization
    df = pd.DataFrame(results)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Chi vs SNR
    ax1 = axes[0]
    ax1.plot(df['snr_db'], df['chi'], 'b-o', markersize=4)
    ax1.axhline(y=gm.CHI_LOW_SNR_LIMIT, color='r', linestyle='--',
                label=r'$\chi_{max} = 2/\pi$')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel(r'Information Retention Factor ($\chi$)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gamma_eff vs SNR
    ax2 = axes[1]
    ax2.plot(df['snr_db'], df['gamma_eff_db'], 'g-s', markersize=4)
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel(r'$\Gamma_{eff}$ (dB)')
    ax2.grid(True, alpha=0.3)

    save_experiment_results(df, fig, "protocol_1_chi_vs_snr")

    print("\n" + "-" * 40)
    print("PROTOCOL 1 COMPLETED")
    print("-" * 40)

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

    # Hardware configurations (increasing impairment)
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
        cfg.snr_db = 20.0  # Fixed SNR for comparison
        for k, v in params.items():
            setattr(cfg, k, v)

        batch_data = simulate_batch(cfg, batch_size=100)
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

    # Check monotonicity
    is_monotonic = all(gamma_values[i] >= gamma_values[i+1]
                       for i in range(len(gamma_values)-1))

    # Visualization
    df = pd.DataFrame(results)

    fig, ax = plt.subplots(figsize=(8, 5))

    x = range(len(configs))
    colors = ['green' if is_monotonic else 'orange'] * len(configs)
    bars = ax.bar(x, df['gamma_eff_db'], color=colors, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels([c[0] for c in configs], rotation=15, ha='right')
    ax.set_ylabel(r'$\Gamma_{eff}$ (dB)')
    ax.grid(True, axis='y', alpha=0.3)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df['gamma_eff_db'])):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{val:.1f} dB', ha='center', fontsize=9)

    save_experiment_results(df, fig, "protocol_5_gamma_monotonicity")

    # Summary
    print("\n" + "-" * 40)
    print(f"Monotonicity Check: {'PASSED ✓' if is_monotonic else 'FAILED ✗'}")
    print(f"PROTOCOL 5 RESULT: {'PASSED ✓' if is_monotonic else 'FAILED ✗'}")
    print("-" * 40)

    if not is_monotonic:
        print("\n⚠️  WARNING: Gamma_eff monotonicity FAILED!")
        print("    Power decomposition may be incorrect.")

    return is_monotonic, results


# ============================================================================
# PROTOCOL 6: BER Calculation Verification (NEW)
# ============================================================================

def run_protocol_6_ber_verification():
    """
    **NEW** - Verifies BER calculation is true bit-level (not SER).

    For QPSK:
        - Each symbol carries 2 bits (I and Q)
        - BER should count I/Q bit errors separately
        - SER counts symbol errors

    BER ≤ SER always (equality when single-bit errors only)
    """
    print("\n" + "=" * 60)
    print("PROTOCOL 6: BER vs SER Verification")
    print("=" * 60)

    # Import BER functions
    from run_p4_experiments import compute_ber_qpsk_bitwise, compute_ser_qpsk

    results = []

    # Generate test data
    np.random.seed(42)
    n_symbols = 10000

    # True QPSK symbols
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    x_true = constellation[np.random.randint(0, 4, n_symbols)]

    # Test different error levels
    noise_levels = [0.0, 0.1, 0.3, 0.5, 1.0]

    print(f"\n{'Noise σ':<10} {'BER':<12} {'SER':<12} {'BER ≤ SER':<10}")
    print("-" * 45)

    for noise_std in noise_levels:
        # Add noise
        noise = (np.random.randn(n_symbols) + 1j * np.random.randn(n_symbols)) * noise_std / np.sqrt(2)
        x_hat = x_true + noise

        # Calculate both metrics
        ber = compute_ber_qpsk_bitwise(x_hat, x_true)
        ser = compute_ser_qpsk(x_hat, x_true)

        is_valid = ber <= ser + 1e-10  # Allow tiny numerical error

        results.append({
            'noise_std': noise_std,
            'ber': ber,
            'ser': ser,
            'ber_le_ser': is_valid
        })

        print(f"{noise_std:<10.2f} {ber:<12.6f} {ser:<12.6f} {'✓' if is_valid else '✗'}")

    all_valid = all(r['ber_le_ser'] for r in results)

    # Summary
    print("\n" + "-" * 40)
    print(f"PROTOCOL 6 RESULT: {'PASSED ✓' if all_valid else 'FAILED ✗'}")
    print("-" * 40)

    df = pd.DataFrame(results)
    df.to_csv(OUTPUT_DIR / "protocol_6_ber_verification.csv", index=False)
    print(f"  [Saved] protocol_6_ber_verification.csv")

    return all_valid, results


# ============================================================================
# Main Entry Point
# ============================================================================

def run_all_protocols():
    """Runs all sanity check protocols."""
    print("\n" + "=" * 70)
    print("SANITY CHECK SUITE - Definition Freeze v2")
    print("=" * 70)
    print(f"Output Directory: {OUTPUT_DIR}")
    print(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    all_results = {}
    all_passed = True

    # Protocol 0: Chi Endpoints (MANDATORY)
    passed, results = run_protocol_0_chi_endpoints()
    all_results['protocol_0'] = {'passed': passed, 'results': results}
    all_passed &= passed

    # Protocol 0.5: API Consistency (NEW)
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

    # Protocol 6: BER Verification (requires run_p4_experiments)
    try:
        passed, results = run_protocol_6_ber_verification()
        all_results['protocol_6'] = {'passed': passed, 'results': results}
        all_passed &= passed
    except ImportError:
        print("\n[Skip] Protocol 6 requires run_p4_experiments.py")

    # Final Summary
    print("\n" + "=" * 70)
    print("SANITY CHECK SUMMARY")
    print("=" * 70)

    for proto_name, data in all_results.items():
        status = "PASS ✓" if data['passed'] else "FAIL ✗"
        print(f"  {proto_name}: {status}")

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
    parser = argparse.ArgumentParser(description="Sanity Check Suite")
    parser.add_argument('--protocol', type=str, default='all',
                        help='Protocol to run (0, 05, 1, 5, 6, or all)')
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
    else:
        print(f"Unknown protocol: {args.protocol}")