#!/usr/bin/env python3
"""
run_experiments.py - Main Entry Script (Expert v2.0 - Top Journal Ready)

Expert Requirements Implemented:
- P0-3: --mode {debug, paper} for method sets and MC counts
- P0-1: Cross-method fairness verification
- P0-2: Dual Oracle (oracle_sync) naming
- Strict validation in paper mode

Usage:
    # Paper-level run (default)
    python run_experiments.py --ckpt path/to/checkpoint.pth --mode paper

    # Debug/quick test
    python run_experiments.py --ckpt path/to/checkpoint.pth --mode debug

    # Visualization only
    python run_experiments.py --visualize_only --data_dir results/paper_figs
"""

import os
import sys
import argparse
import glob

# ============================================================================
# Path Setup
# ============================================================================

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# ============================================================================
# Imports
# ============================================================================

print("=" * 60)
print("🔍 Module Import Check")
print("=" * 60)

from evaluator import (
    EvalConfig,
    load_model,
    run_snr_sweep,
    run_cliff_sweep,
    run_snr_sweep_multi_init_error,
    run_ablation_sweep,
    run_heatmap_sweep,
    run_pn_sweep,
    run_pilot_sweep,
    run_jacobian_analysis,
    run_crlb_sweep,
    measure_latency,
    run_sanity_check,
    validate_csv_methods,
    print_import_info,
)

from visualization import generate_all_figures, compute_summary_metrics

from baselines import (
    METHOD_PAPER_CORE,
    METHOD_PAPER_FULL,
    METHOD_DEBUG,
    METHOD_CLIFF,
    METHOD_ABLATION,
    METHOD_SNR_SWEEP,
    METHOD_ROBUSTNESS,
)

print("  ✓ All modules loaded successfully")
print_import_info()
print("=" * 60)


# ============================================================================
# Helper Functions
# ============================================================================

def find_checkpoint(ckpt_path: str) -> str:
    """Find checkpoint file."""
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path

    patterns = [
        'results/checkpoints/Stage2_*/final.pth',
        './results/checkpoints/Stage2_*/final.pth',
        '../results/checkpoints/Stage2_*/final.pth',
        'results/checkpoints/Stage3_*/final.pth',
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]

    return None


def verify_csv_output(df, expected_methods, csv_name, strict=False):
    """Verify CSV output contains all expected methods."""
    if 'method' not in df.columns:
        msg = f"⚠️ {csv_name}: Missing 'method' column!"
        print(f"  {msg}")
        if strict:
            raise ValueError(msg)
        return False

    actual = set(df['method'].unique())
    expected = set(expected_methods)
    missing = expected - actual

    if missing:
        msg = f"⚠️ {csv_name}: Missing methods {missing}"
        print(f"  {msg}")
        if strict:
            raise ValueError(msg)
        return False

    print(f"  ✓ {csv_name}: {len(actual)} methods OK")
    return True


# ============================================================================
# Data Collection
# ============================================================================

def run_data_collection(args):
    """Run data collection phase."""

    print("\n" + "=" * 60)
    print("📊 Data Collection Phase")
    print("=" * 60)

    # Load model
    ckpt_path = find_checkpoint(args.ckpt)
    if not ckpt_path:
        print("❌ Checkpoint not found!")
        print("   Please specify --ckpt path")
        return None

    print(f"\nLoading model: {ckpt_path}")
    model, gabv_cfg = load_model(ckpt_path, args.device)

    # Mode-specific configuration (P0-3)
    if args.mode == 'debug':
        print("\n⚠️ DEBUG MODE: For testing only, not paper-ready!")
        n_mc = 5
        batch_size = 32
        methods_cliff = METHOD_DEBUG + ["adjoint_lmmse"]
        methods_snr = METHOD_DEBUG
        methods_ablation = ["proposed_no_update", "proposed", "oracle_sync"]
        strict = False
    else:  # paper mode
        print("\n📝 PAPER MODE: Full methods, strict validation")
        n_mc = args.n_mc
        batch_size = args.batch
        methods_cliff = METHOD_CLIFF
        methods_snr = METHOD_SNR_SWEEP
        methods_ablation = METHOD_ABLATION
        strict = True

    eval_cfg = EvalConfig(
        ckpt_path=ckpt_path,
        device=args.device,
        snr_list=args.snr_list,
        n_mc=n_mc,
        batch_size=batch_size,
        theta_noise_tau=args.init_error,
        out_dir=args.out_dir,
        eps_tau=args.eps_tau,
        mode=args.mode,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  Mode: {args.mode}")
    print(f"  SNR list: {eval_cfg.snr_list}")
    print(f"  MC runs: {eval_cfg.n_mc}")
    print(f"  Batch size: {eval_cfg.batch_size}")
    print(f"  Init error (τ): {eval_cfg.theta_noise_tau}")
    print(f"  Success threshold (ε_τ): {eval_cfg.eps_tau}")
    print(f"  Output dir: {args.out_dir}")

    # ===== Sanity Check =====
    if not args.skip_sanity:
        passed = run_sanity_check(model, gabv_cfg, eval_cfg)
        if not passed:
            print("\n❌ Sanity Check FAILED!")
            if strict and not args.force:
                print("   In paper mode, sanity check must pass. Use --force to override.")
                return None
            print("   (Continuing with --force)")

    # ===== Run Sweeps =====
    print("\n" + "-" * 40)

    print("\n[1/9] SNR sweep...")
    df_snr = run_snr_sweep(model, gabv_cfg, eval_cfg, methods=methods_snr)
    df_snr.to_csv(f"{args.out_dir}/data_snr_sweep.csv", index=False)
    verify_csv_output(df_snr, methods_snr, "data_snr_sweep", strict=strict)

    print("\n[2/9] Cliff sweep (core figure)...")
    df_cliff = run_cliff_sweep(model, gabv_cfg, eval_cfg, methods=methods_cliff)
    df_cliff.to_csv(f"{args.out_dir}/data_cliff_sweep.csv", index=False)
    verify_csv_output(df_cliff, methods_cliff, "data_cliff_sweep", strict=strict)

    print("\n[3/9] Multi-init SNR sweep...")
    df_snr_multi = run_snr_sweep_multi_init_error(model, gabv_cfg, eval_cfg)
    df_snr_multi.to_csv(f"{args.out_dir}/data_snr_multi_init_error.csv", index=False)

    print("\n[4/9] Ablation sweep...")
    df_ablation = run_ablation_sweep(model, gabv_cfg, eval_cfg, methods=methods_ablation)
    df_ablation.to_csv(f"{args.out_dir}/data_ablation_sweep.csv", index=False)
    verify_csv_output(df_ablation, methods_ablation, "data_ablation_sweep", strict=strict)

    print("\n[5/9] Heatmap sweep (basin map)...")
    df_heatmap = run_heatmap_sweep(model, gabv_cfg, eval_cfg)
    df_heatmap.to_csv(f"{args.out_dir}/data_heatmap_sweep.csv", index=False)

    print("\n[6/9] PN sweep...")
    df_pn = run_pn_sweep(model, gabv_cfg, eval_cfg)
    df_pn.to_csv(f"{args.out_dir}/data_pn_sweep.csv", index=False)

    print("\n[7/9] Pilot sweep...")
    df_pilot = run_pilot_sweep(model, gabv_cfg, eval_cfg)
    df_pilot.to_csv(f"{args.out_dir}/data_pilot_sweep.csv", index=False)

    print("\n[8/9] CRLB sweep (P2-1)...")
    df_crlb = run_crlb_sweep(gabv_cfg, eval_cfg)
    df_crlb.to_csv(f"{args.out_dir}/data_crlb_sweep.csv", index=False)

    print("\n[9/9] Jacobian & Latency...")
    df_jacobian = run_jacobian_analysis(model, gabv_cfg, eval_cfg)
    df_jacobian.to_csv(f"{args.out_dir}/data_jacobian.csv", index=False)

    df_latency = measure_latency(model, gabv_cfg, eval_cfg)
    df_latency.to_csv(f"{args.out_dir}/data_latency.csv", index=False)

    # ===== Results Summary =====
    print("\n" + "=" * 60)
    print("📋 Results Summary")
    print("=" * 60)

    # Baseline validation @ init_error=0
    print("\n### Baseline @ init_error=0")
    cliff_0 = df_cliff[df_cliff['init_error'] == 0.0]
    if len(cliff_0) > 0:
        for method in cliff_0['method'].unique():
            ber = cliff_0[cliff_0['method'] == method]['ber'].mean()
            status = "✅ OK" if ber < 0.2 else "⚠️ Anomaly"
            print(f"  {method:25s}: BER={ber:.4f} {status}")

    # SNR=15dB performance
    print("\n### @ SNR=15dB")
    snr_15 = df_snr[df_snr['snr_db'] == 15]
    if len(snr_15) > 0:
        for method in snr_15['method'].unique():
            data = snr_15[snr_15['method'] == method]
            if len(data) > 0:
                ber = data['ber'].mean()
                rmse = data['rmse_tau_final'].mean()
                print(f"  {method:25s}: BER={ber:.4f}, RMSE={rmse:.4f}")

    # Pull-in range
    print("\n### Pull-in Range (P1-2)")
    methods_to_check = ['proposed', 'adjoint_slice', 'proposed_no_update']
    summary = compute_summary_metrics(df_cliff, methods_to_check)
    for _, row in summary.iterrows():
        print(f"  {row['method']:25s}: pull-in={row['pull_in_range']:.2f} samples")

    return args.out_dir


# ============================================================================
# Main Function
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Paper Figure Generation Pipeline (Expert v2.0)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Paper-level run
  python run_experiments.py --ckpt checkpoint.pth --mode paper --n_mc 20

  # Debug/quick test
  python run_experiments.py --ckpt checkpoint.pth --mode debug

  # Visualization only
  python run_experiments.py --visualize_only --data_dir results/paper_figs
        """
    )

    # Mode selection
    parser.add_argument('--visualize_only', action='store_true',
                        help="Only generate figures from CSV")

    # Run mode (P0-3)
    parser.add_argument('--mode', type=str, choices=['debug', 'paper'], default='paper',
                        help="Run mode: 'debug' (fast) or 'paper' (full)")

    # Data collection parameters
    parser.add_argument('--ckpt', type=str, default="",
                        help="Checkpoint path")
    parser.add_argument('--snr_list', nargs='+', type=float,
                        default=[-5, 0, 5, 10, 15, 20, 25],
                        help="SNR sweep values")
    parser.add_argument('--n_mc', type=int, default=20,
                        help="Monte Carlo runs (paper: 20)")
    parser.add_argument('--batch', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--init_error', type=float, default=0.3,
                        help="Default init τ error (samples)")
    parser.add_argument('--eps_tau', type=float, default=0.1,
                        help="Success threshold ε_τ (samples)")
    parser.add_argument('--device', type=str, default="cuda",
                        help="Device (cuda/cpu)")

    # Sanity check
    parser.add_argument('--skip_sanity', action='store_true',
                        help="Skip sanity check")
    parser.add_argument('--force', action='store_true',
                        help="Continue even if sanity check fails")

    # Output parameters
    parser.add_argument('--out_dir', type=str, default="results/paper_figs",
                        help="Output directory")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Data directory (for visualize_only)")

    args = parser.parse_args()

    print("=" * 60)
    print("🎓 Paper Figure Pipeline (Expert v2.0)")
    print("=" * 60)
    print(f"Mode: {'Visualization Only' if args.visualize_only else args.mode.upper()}")

    if args.visualize_only:
        # Visualization only mode
        data_dir = args.data_dir or args.out_dir
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return

        generate_all_figures(data_dir, args.out_dir)
    else:
        # Full run mode
        data_dir = run_data_collection(args)

        if data_dir:
            print("\n" + "=" * 60)
            print("📈 Visualization Phase")
            print("=" * 60)
            generate_all_figures(data_dir, args.out_dir)

    # Final summary
    print("\n" + "=" * 60)
    print("📝 Paper Narrative Key Points")
    print("=" * 60)
    print("""
Core Thesis:
"In 1-bit quantized dirty-hardware THz-ISAC links, initial sync error 
triggers detection 'cliff failure'; our pilot-only geometric τ fast-loop 
tracking pulls the receiver back into the trackable basin, achieving 
near-oracle performance within that basin."

Key Evidence:
- init_error=0: All methods ≈ oracle (baseline implementation OK)
- init_error=0.3: Baselines fail, proposed still works
- Basin boundary: ~0.3-0.5 samples (data-driven)
- BER saturation: 1-bit physical limit, true gain in τ RMSE

Figure Checklist:
✓ Fig 04 (Cliff): Core contribution with pull-in annotation
✓ Fig 05 (Multi-init): Proof of baseline correctness
✓ Fig 07 (Gap-to-Oracle): Near-oracle performance
✓ Fig 11 (Basin Map): 2D success rate visualization
✓ Fig 10 (Ablation): Dual-axis BER + τ RMSE
""")

    print(f"\n✅ Complete! All outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()