#!/usr/bin/env python3
"""
run_experiments.py - Main Entry Script (Expert Review)

Usage:
    python run_experiments.py --ckpt path/to/checkpoint.pth --n_mc 20
    python run_experiments.py --ckpt path/to/checkpoint.pth --quick
    python run_experiments.py --visualize_only --data_dir results/paper_figs
"""

import os
import sys
import argparse
import glob

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

print("=" * 60)
print("🔍 Module Import Path Check")
print("=" * 60)

try:
    from evaluator import (
        EvalConfig, load_model, run_snr_sweep, run_cliff_sweep,
        run_snr_sweep_multi_init_error, run_ablation_sweep, run_heatmap_sweep,
        run_pn_sweep, run_pilot_sweep, run_jacobian_analysis, measure_latency,
        run_sanity_check, validate_csv_methods, print_import_info,
        run_adc_bits_sweep, run_crlb_sweep,
    )
    print("  ✓ Using evaluator.py (Trial-first)")
except ImportError as e:
    print(f"  ❌ evaluator import error: {e}")
    sys.exit(1)

try:
    from visualization import generate_all_figures
    print("  ✓ Using visualization.py")
except ImportError as e:
    print(f"  ❌ visualization import error: {e}")
    sys.exit(1)

from baselines import METHOD_ORDER, METHOD_CLIFF, METHOD_ABLATION, METHOD_SNR_SWEEP, get_method_info

if print_import_info:
    print_import_info()

print("=" * 60)


def find_checkpoint(ckpt_path: str) -> str:
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


def verify_csv_output(df, expected_methods, csv_name):
    if 'method' not in df.columns:
        print(f"  ⚠️ {csv_name}: Missing 'method' column!")
        return False
    actual = set(df['method'].unique())
    missing = set(expected_methods) - actual
    if missing:
        print(f"  ⚠️ {csv_name}: Missing methods {missing}")
        return False
    print(f"  ✓ {csv_name}: {len(actual)} methods OK")
    return True


def run_data_collection(args):
    print("\n" + "=" * 60)
    print("📊 Data Collection Phase")
    print("=" * 60)

    ckpt_path = find_checkpoint(args.ckpt)
    if not ckpt_path:
        print("❌ Checkpoint not found!")
        return None

    print(f"\nLoading model: {ckpt_path}")
    model, gabv_cfg = load_model(ckpt_path, args.device)

    if args.quick:
        print("\n⚠️ Quick mode: For debugging only!")
        n_mc, batch_size = 5, 32
    else:
        n_mc, batch_size = args.n_mc, args.batch

    eval_cfg = EvalConfig(
        ckpt_path=ckpt_path, device=args.device, snr_list=args.snr_list,
        n_mc=n_mc, batch_size=batch_size, theta_noise_tau=args.init_error, out_dir=args.out_dir,
    )
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  SNR list: {eval_cfg.snr_list}")
    print(f"  MC runs: {eval_cfg.n_mc}")
    print(f"  Batch size: {eval_cfg.batch_size}")
    print(f"  Init error (τ): {eval_cfg.theta_noise_tau}")
    print(f"  Output dir: {args.out_dir}")

    print("\n  Method Registry:")
    method_info = get_method_info()
    for name, info in method_info.items():
        extras = ""
        if 'grid_points' in info:
            extras = f" [grid={info['grid_points']}, range=±{info.get('default_search_half_range', info.get('search_half_range', 'N/A'))}]"
        print(f"    {name}: {info['class']}{extras}")

    if not args.skip_sanity:
        passed = run_sanity_check(model, gabv_cfg, eval_cfg)
        if not passed and not args.force:
            print("\n❌ Sanity Check FAILED!")
            return None

    print("\n" + "-" * 40)

    print("\n[1/10] SNR sweep...")
    df_snr = run_snr_sweep(model, gabv_cfg, eval_cfg)
    df_snr.to_csv(f"{args.out_dir}/data_snr_sweep.csv", index=False)
    verify_csv_output(df_snr, ["proposed", "oracle_sync", "adjoint_slice"], "data_snr_sweep")

    print("\n[2/10] Cliff sweep (CORE figure)...")
    df_cliff = run_cliff_sweep(model, gabv_cfg, eval_cfg)
    df_cliff.to_csv(f"{args.out_dir}/data_cliff_sweep.csv", index=False)
    verify_csv_output(df_cliff, METHOD_CLIFF, "data_cliff_sweep")

    print("\n[3/10] Multi-init SNR sweep...")
    df_snr_multi = run_snr_sweep_multi_init_error(model, gabv_cfg, eval_cfg)
    df_snr_multi.to_csv(f"{args.out_dir}/data_snr_multi_init_error.csv", index=False)

    print("\n[4/10] Ablation sweep...")
    df_ablation = run_ablation_sweep(model, gabv_cfg, eval_cfg)
    df_ablation.to_csv(f"{args.out_dir}/data_ablation_sweep.csv", index=False)
    verify_csv_output(df_ablation, METHOD_ABLATION, "data_ablation_sweep")

    print("\n[5/10] Heatmap sweep...")
    df_heatmap = run_heatmap_sweep(model, gabv_cfg, eval_cfg)
    df_heatmap.to_csv(f"{args.out_dir}/data_heatmap_sweep.csv", index=False)

    print("\n[6/10] PN sweep...")
    df_pn = run_pn_sweep(model, gabv_cfg, eval_cfg)
    df_pn.to_csv(f"{args.out_dir}/data_pn_sweep.csv", index=False)

    print("\n[7/10] Pilot sweep...")
    df_pilot = run_pilot_sweep(model, gabv_cfg, eval_cfg)
    df_pilot.to_csv(f"{args.out_dir}/data_pilot_sweep.csv", index=False)

    print("\n[8/10] Jacobian & Latency...")
    df_jacobian = run_jacobian_analysis(model, gabv_cfg, eval_cfg)
    df_jacobian.to_csv(f"{args.out_dir}/data_jacobian.csv", index=False)
    df_latency = measure_latency(model, gabv_cfg, eval_cfg)
    df_latency.to_csv(f"{args.out_dir}/data_latency.csv", index=False)

    print("\n[9/10] ADC bits sweep...")
    df_adc_bits = run_adc_bits_sweep(model, gabv_cfg, eval_cfg)
    df_adc_bits.to_csv(f"{args.out_dir}/data_adc_bits_sweep.csv", index=False)

    print("\n[10/10] CRLB computation...")
    df_crlb = run_crlb_sweep(gabv_cfg, eval_cfg)
    df_crlb.to_csv(f"{args.out_dir}/data_crlb.csv", index=False)

    print("\n" + "=" * 60)
    print("📋 Results Summary")
    print("=" * 60)

    print("\n### Baseline @ init_error=0")
    cliff_0 = df_cliff[df_cliff['init_error'] == 0.0]
    if len(cliff_0) > 0:
        for method in cliff_0['method'].unique():
            ber = cliff_0[cliff_0['method'] == method]['ber'].mean()
            status = "✅ OK" if ber < 0.2 else "⚠️ Anomaly"
            print(f"  {method:25s}: BER={ber:.4f} {status}")

    print("\n### @ SNR=15dB")
    snr_15 = df_snr[df_snr['snr_db'] == 15]
    if len(snr_15) > 0:
        for method in ['adjoint_slice', 'proposed', 'oracle_sync']:
            data = snr_15[snr_15['method'] == method]
            if len(data) > 0:
                ber = data['ber'].mean()
                rmse = data['rmse_tau_final'].mean()
                print(f"  {method:25s}: BER={ber:.4f}, RMSE={rmse:.4f}")

    print("\n### Gap-to-Oracle Check")
    oracle_ber = df_snr[df_snr['method'] == 'oracle_sync']['ber'].mean() if 'oracle_sync' in df_snr['method'].values else None
    proposed_ber = df_snr[df_snr['method'] == 'proposed']['ber'].mean() if 'proposed' in df_snr['method'].values else None
    if oracle_ber is not None and proposed_ber is not None:
        gap = proposed_ber - oracle_ber
        status = "✅ OK" if gap >= 0 else "⚠️ Negative"
        print(f"  Gap: {gap:+.4f} {status}")

    return args.out_dir


def main():
    parser = argparse.ArgumentParser(description="Run experiments and generate paper figures")
    parser.add_argument('--visualize_only', action='store_true')
    parser.add_argument('--ckpt', type=str, default="")
    parser.add_argument('--snr_list', nargs='+', type=float, default=[-5, 0, 5, 10, 15, 20, 25])
    parser.add_argument('--n_mc', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--init_error', type=float, default=0.3)
    parser.add_argument('--device', type=str, default="cuda")
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--skip_sanity', action='store_true')
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--out_dir', type=str, default="results/paper_figs")
    parser.add_argument('--data_dir', type=str, default=None)

    args = parser.parse_args()

    print("=" * 60)
    print("🎓 Paper Figure Generation Pipeline")
    print("=" * 60)
    print(f"Mode: {'Visualization only' if args.visualize_only else 'Full run'}")

    if args.visualize_only:
        data_dir = args.data_dir or args.out_dir
        if not os.path.exists(data_dir):
            print(f"❌ Data directory not found: {data_dir}")
            return
        generate_all_figures(data_dir, args.out_dir)
    else:
        data_dir = run_data_collection(args)
        if data_dir:
            print("\n" + "=" * 60)
            print("📈 Visualization Phase")
            print("=" * 60)
            generate_all_figures(data_dir, args.out_dir)

    print("\n" + "=" * 60)
    print("📝 Paper Narrative Guidance")
    print("=" * 60)
    print("""
"In 1-bit quantized dirty-hardware THz-ISAC links, initial sync error
triggers a 'cliff-like detection failure'; our proposed pilot-only
geometry-consistent τ fast-loop tracking pulls the receiver back into
the trackable basin, achieving near-oracle performance within that basin."
""")
    print(f"\n✅ Complete! All outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()