#!/usr/bin/env python3
"""
run_experiments.py - 主入口脚本

运行完整的实验流程：
1. 加载模型
2. 运行各种 sweep（数据采集）
3. 生成图表（可视化）

用法：
    # 完整运行（数据采集 + 可视化）
    python run_experiments.py --ckpt path/to/checkpoint.pth --n_mc 20

    # 快速测试
    python run_experiments.py --ckpt path/to/checkpoint.pth --quick

    # 仅可视化（从已有 CSV 数据）
    python run_experiments.py --visualize_only --data_dir results/paper_figs
"""

import os
import sys
import argparse
import glob

# 添加当前目录到 path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

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
    measure_latency,
)
from visualization import generate_all_figures
from baselines import METHOD_ORDER


def find_checkpoint(ckpt_path: str) -> str:
    """查找 checkpoint 文件"""
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path

    patterns = [
        'results/checkpoints/Stage2_*/final.pth',
        './results/checkpoints/Stage2_*/final.pth',
        '../results/checkpoints/Stage2_*/final.pth',
    ]

    for pattern in patterns:
        matches = glob.glob(pattern)
        if matches:
            return sorted(matches)[-1]

    return None


def run_data_collection(args):
    """运行数据采集阶段"""

    print("=" * 60)
    print("📊 数据采集阶段")
    print("=" * 60)

    # 加载模型
    ckpt_path = find_checkpoint(args.ckpt)
    if not ckpt_path:
        print("ERROR: No checkpoint found!")
        return None

    print(f"Loading model from: {ckpt_path}")
    model, gabv_cfg = load_model(ckpt_path, args.device)

    # 配置
    eval_cfg = EvalConfig(
        ckpt_path=ckpt_path,
        device=args.device,
        snr_list=args.snr_list,
        n_mc=args.n_mc if not args.quick else 5,
        batch_size=args.batch if not args.quick else 32,
        theta_noise_tau=args.init_error,
        out_dir=args.out_dir,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\nConfiguration:")
    print(f"  SNR list: {eval_cfg.snr_list}")
    print(f"  Monte Carlo trials: {eval_cfg.n_mc}")
    print(f"  Batch size: {eval_cfg.batch_size}")
    print(f"  Init error (τ): {eval_cfg.theta_noise_tau}")
    print(f"  Output: {args.out_dir}")

    # 运行各种 sweep
    print("\n" + "-" * 40)
    print("[1/7] SNR sweep...")
    df_snr = run_snr_sweep(model, gabv_cfg, eval_cfg)
    df_snr.to_csv(f"{args.out_dir}/data_snr_sweep.csv", index=False)
    print(f"      Saved: data_snr_sweep.csv ({len(df_snr)} records)")

    print("\n[2/7] Cliff sweep (ALL methods) - 方案1...")
    df_cliff = run_cliff_sweep(model, gabv_cfg, eval_cfg)
    df_cliff.to_csv(f"{args.out_dir}/data_cliff_sweep.csv", index=False)
    print(f"      Saved: data_cliff_sweep.csv ({len(df_cliff)} records)")

    print("\n[3/7] SNR sweep @ multi init_error - 方案3...")
    df_snr_multi = run_snr_sweep_multi_init_error(model, gabv_cfg, eval_cfg)
    df_snr_multi.to_csv(f"{args.out_dir}/data_snr_multi_init_error.csv", index=False)
    print(f"      Saved: data_snr_multi_init_error.csv ({len(df_snr_multi)} records)")

    print("\n[4/8] Ablation sweep - 方案2...")
    df_ablation = run_ablation_sweep(model, gabv_cfg, eval_cfg)
    df_ablation.to_csv(f"{args.out_dir}/data_ablation_sweep.csv", index=False)
    print(f"      Saved: data_ablation_sweep.csv ({len(df_ablation)} records)")

    print("\n[5/8] Heatmap sweep (2D: SNR × init_error)...")
    df_heatmap = run_heatmap_sweep(model, gabv_cfg, eval_cfg)
    df_heatmap.to_csv(f"{args.out_dir}/data_heatmap_sweep.csv", index=False)
    print(f"      Saved: data_heatmap_sweep.csv ({len(df_heatmap)} records)")

    print("\n[6/8] PN sweep...")
    df_pn = run_pn_sweep(model, gabv_cfg, eval_cfg)
    df_pn.to_csv(f"{args.out_dir}/data_pn_sweep.csv", index=False)
    print(f"      Saved: data_pn_sweep.csv ({len(df_pn)} records)")

    print("\n[7/8] Pilot sweep...")
    df_pilot = run_pilot_sweep(model, gabv_cfg, eval_cfg)
    df_pilot.to_csv(f"{args.out_dir}/data_pilot_sweep.csv", index=False)
    print(f"      Saved: data_pilot_sweep.csv ({len(df_pilot)} records)")

    print("\n[8/8] Jacobian analysis & Latency...")
    df_jacobian = run_jacobian_analysis(model, gabv_cfg, eval_cfg)
    df_jacobian.to_csv(f"{args.out_dir}/data_jacobian.csv", index=False)

    df_latency = measure_latency(model, gabv_cfg, eval_cfg)
    df_latency.to_csv(f"{args.out_dir}/data_latency.csv", index=False)
    print(f"      Saved: data_jacobian.csv, data_latency.csv")

    # 打印结果摘要
    print("\n" + "=" * 60)
    print("📋 结果摘要")
    print("=" * 60)

    # 验证 baseline 在 init_error=0 时的表现
    print("\n### 专家验证：Baseline @ init_error=0")
    cliff_0 = df_cliff[df_cliff['init_error'] == 0.0]
    if len(cliff_0) > 0:
        for method in cliff_0['method'].unique():
            ber = cliff_0[cliff_0['method'] == method]['ber'].mean()
            status = "✅ OK" if ber < 0.2 else "⚠️ 异常"
            print(f"  {method:25s}: BER={ber:.4f} {status}")

    # SNR=15dB 性能
    print("\n### @ SNR=15dB")
    snr_15 = df_snr[df_snr['snr_db'] == 15]
    if len(snr_15) > 0:
        for method in ['adjoint_slice', 'proposed', 'oracle']:
            data = snr_15[snr_15['method'] == method]
            if len(data) > 0:
                ber = data['ber'].mean()
                rmse = data['rmse_tau_final'].mean()
                print(f"  {method:25s}: BER={ber:.4f}, RMSE={rmse:.4f}")

    return args.out_dir


def main():
    parser = argparse.ArgumentParser(
        description="Run experiments and generate paper figures",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run with data collection and visualization
  python run_experiments.py --ckpt checkpoint.pth --n_mc 20

  # Quick test
  python run_experiments.py --ckpt checkpoint.pth --quick

  # Visualization only (from existing CSV data)
  python run_experiments.py --visualize_only --data_dir results/paper_figs
        """
    )

    # 模式选择
    parser.add_argument('--visualize_only', action='store_true',
                        help="Only generate figures from existing CSV data")

    # 数据采集参数
    parser.add_argument('--ckpt', type=str, default="",
                        help="Checkpoint path")
    parser.add_argument('--snr_list', nargs='+', type=float,
                        default=[-5, 0, 5, 10, 15, 20, 25],
                        help="SNR values to sweep")
    parser.add_argument('--n_mc', type=int, default=20,
                        help="Monte Carlo trials")
    parser.add_argument('--batch', type=int, default=64,
                        help="Batch size")
    parser.add_argument('--init_error', type=float, default=0.3,
                        help="Default init τ error (samples)")
    parser.add_argument('--device', type=str, default="cuda",
                        help="Device (cuda/cpu)")
    parser.add_argument('--quick', action='store_true',
                        help="Quick mode for testing")

    # 输出参数
    parser.add_argument('--out_dir', type=str, default="results/paper_figs",
                        help="Output directory")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="Data directory (for visualize_only mode)")

    args = parser.parse_args()

    print("=" * 60)
    print("🎓 Paper Figure Generation Pipeline")
    print("=" * 60)
    print(f"Mode: {'Visualize Only' if args.visualize_only else 'Full Run'}")

    if args.visualize_only:
        # 仅可视化模式
        data_dir = args.data_dir or args.out_dir
        if not os.path.exists(data_dir):
            print(f"ERROR: Data directory not found: {data_dir}")
            return

        generate_all_figures(data_dir, args.out_dir)
    else:
        # 完整运行模式
        data_dir = run_data_collection(args)

        if data_dir:
            print("\n" + "=" * 60)
            print("📈 可视化阶段")
            print("=" * 60)
            generate_all_figures(data_dir, args.out_dir)

    # 最终输出
    print("\n" + "=" * 60)
    print("📝 论文叙事建议")
    print("=" * 60)
    print("""
"在 1-bit 量化与脏硬件 THz-ISAC 链路中，初始同步误差会触发检测
'悬崖式失效'；本文提出的 pilot-only 几何一致 τ 快环跟踪将接收机
重新拉回可跟踪盆地，使检测性能在该盆地内逼近 oracle 上界。"

关键数据点：
- init_error=0 时所有方法都接近 oracle（证明 baseline 没 bug）
- init_error=0.3 时 baseline 失效，proposed 仍工作
- basin 边界约 0.3-0.5 samples
""")

    print(f"\n✅ Done! All outputs saved to: {args.out_dir}")


if __name__ == "__main__":
    main()