#!/usr/bin/env python3
"""
run_experiments.py - 主入口脚本 (修复版 v2)

修复内容：
1. 添加模块导入路径打印（防止版本混乱）
2. 添加 Sanity Check（init_error=0 时验证 baseline）
3. 添加 CSV 方法验证
4. 不再使用 --quick 时自动退化方法集

用法：
    # 完整运行（论文级）
    python run_experiments.py --ckpt path/to/checkpoint.pth --n_mc 20

    # 快速测试（仅 debug 用）
    python run_experiments.py --ckpt path/to/checkpoint.pth --quick

    # 仅可视化
    python run_experiments.py --visualize_only --data_dir results/paper_figs
"""

import os
import sys
import argparse
import glob

# ============================================================================
# 路径设置
# ============================================================================

_this_dir = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(_this_dir)

if _project_root not in sys.path:
    sys.path.insert(0, _project_root)
if _this_dir not in sys.path:
    sys.path.insert(0, _this_dir)

# ============================================================================
# 导入并打印路径（防止版本混乱）
# ============================================================================

print("=" * 60)
print("🔍 模块导入路径检查")
print("=" * 60)

try:
    from evaluator_v2 import (
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
        run_sanity_check,
        validate_csv_methods,
        print_import_info,
    )

    print("  ✓ 使用 evaluator_v2.py (修复版)")
except ImportError:
    print("  ⚠️ evaluator_v2 未找到，使用原版 evaluator")
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

    run_sanity_check = None
    validate_csv_methods = None
    print_import_info = None

try:
    from visualization_v5 import generate_all_figures

    print("  ✓ 使用 visualization_v5.py (修复版)")
except ImportError:
    print("  ⚠️ visualization_v5 未找到，使用原版 visualization")
    from visualization import generate_all_figures

from baselines import METHOD_ORDER, METHOD_CLIFF, METHOD_ABLATION

# 打印导入信息
if print_import_info:
    print_import_info()

print("=" * 60)


# ============================================================================
# 辅助函数
# ============================================================================

def find_checkpoint(ckpt_path: str) -> str:
    """查找 checkpoint 文件"""
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
    """验证 CSV 输出包含所有期望的方法"""
    if 'method' not in df.columns:
        print(f"  ⚠️ {csv_name}: 缺少 'method' 列！")
        return False

    actual = set(df['method'].unique())
    expected = set(expected_methods)
    missing = expected - actual

    if missing:
        print(f"  ⚠️ {csv_name}: 缺少方法 {missing}")
        return False

    print(f"  ✓ {csv_name}: {len(actual)} 个方法 OK")
    return True


# ============================================================================
# 数据采集
# ============================================================================

def run_data_collection(args):
    """运行数据采集阶段"""

    print("\n" + "=" * 60)
    print("📊 数据采集阶段")
    print("=" * 60)

    # 加载模型
    ckpt_path = find_checkpoint(args.ckpt)
    if not ckpt_path:
        print("❌ 未找到 checkpoint！")
        print("   请指定 --ckpt 路径")
        return None

    print(f"\n加载模型: {ckpt_path}")
    model, gabv_cfg = load_model(ckpt_path, args.device)

    # 配置
    if args.quick:
        print("\n⚠️ Quick 模式：仅用于 debug，不适合论文图！")
        n_mc = 5
        batch_size = 32
    else:
        n_mc = args.n_mc
        batch_size = args.batch

    eval_cfg = EvalConfig(
        ckpt_path=ckpt_path,
        device=args.device,
        snr_list=args.snr_list,
        n_mc=n_mc,
        batch_size=batch_size,
        theta_noise_tau=args.init_error,
        out_dir=args.out_dir,
    )

    os.makedirs(args.out_dir, exist_ok=True)

    print(f"\n配置:")
    print(f"  SNR 列表: {eval_cfg.snr_list}")
    print(f"  MC 次数: {eval_cfg.n_mc}")
    print(f"  Batch 大小: {eval_cfg.batch_size}")
    print(f"  Init error (τ): {eval_cfg.theta_noise_tau}")
    print(f"  输出目录: {args.out_dir}")

    # ===== Sanity Check =====
    if run_sanity_check and not args.skip_sanity:
        passed = run_sanity_check(model, gabv_cfg, eval_cfg)
        if not passed:
            print("\n❌ Sanity Check 失败！请检查 baseline 实现后再继续。")
            if not args.force:
                return None
            print("   (--force 模式：继续执行)")

    # ===== 运行各种 sweep =====
    print("\n" + "-" * 40)

    print("\n[1/8] SNR sweep...")
    df_snr = run_snr_sweep(model, gabv_cfg, eval_cfg)
    df_snr.to_csv(f"{args.out_dir}/data_snr_sweep.csv", index=False)
    verify_csv_output(df_snr, ["proposed", "oracle", "adjoint_slice"], "data_snr_sweep")

    print("\n[2/8] Cliff sweep (核心图)...")
    df_cliff = run_cliff_sweep(model, gabv_cfg, eval_cfg)
    df_cliff.to_csv(f"{args.out_dir}/data_cliff_sweep.csv", index=False)
    verify_csv_output(df_cliff, METHOD_CLIFF, "data_cliff_sweep")

    print("\n[3/8] Multi-init SNR sweep...")
    df_snr_multi = run_snr_sweep_multi_init_error(model, gabv_cfg, eval_cfg)
    df_snr_multi.to_csv(f"{args.out_dir}/data_snr_multi_init_error.csv", index=False)
    verify_csv_output(df_snr_multi, ["proposed", "oracle"], "data_snr_multi_init_error")

    print("\n[4/8] Ablation sweep...")
    df_ablation = run_ablation_sweep(model, gabv_cfg, eval_cfg)
    df_ablation.to_csv(f"{args.out_dir}/data_ablation_sweep.csv", index=False)
    verify_csv_output(df_ablation, METHOD_ABLATION, "data_ablation_sweep")

    print("\n[5/8] Heatmap sweep...")
    df_heatmap = run_heatmap_sweep(model, gabv_cfg, eval_cfg)
    df_heatmap.to_csv(f"{args.out_dir}/data_heatmap_sweep.csv", index=False)

    print("\n[6/8] PN sweep...")
    df_pn = run_pn_sweep(model, gabv_cfg, eval_cfg)
    df_pn.to_csv(f"{args.out_dir}/data_pn_sweep.csv", index=False)
    verify_csv_output(df_pn, ["proposed", "adjoint_slice"], "data_pn_sweep")

    print("\n[7/8] Pilot sweep...")
    df_pilot = run_pilot_sweep(model, gabv_cfg, eval_cfg)
    df_pilot.to_csv(f"{args.out_dir}/data_pilot_sweep.csv", index=False)
    verify_csv_output(df_pilot, ["proposed", "adjoint_slice"], "data_pilot_sweep")

    print("\n[8/8] Jacobian & Latency...")
    df_jacobian = run_jacobian_analysis(model, gabv_cfg, eval_cfg)
    df_jacobian.to_csv(f"{args.out_dir}/data_jacobian.csv", index=False)

    df_latency = measure_latency(model, gabv_cfg, eval_cfg)
    df_latency.to_csv(f"{args.out_dir}/data_latency.csv", index=False)

    # ===== 结果摘要 =====
    print("\n" + "=" * 60)
    print("📋 结果摘要")
    print("=" * 60)

    # Baseline 验证（init_error=0）
    print("\n### Baseline @ init_error=0")
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

    # 消融实验关键数据
    print("\n### 消融实验 @ 最高 SNR")
    target_snr = df_ablation['snr_db'].max()
    abl_high = df_ablation[df_ablation['snr_db'] == target_snr]
    if len(abl_high) > 0:
        for method in ['proposed_no_update', 'proposed_tau_slice', 'proposed', 'oracle']:
            data = abl_high[abl_high['method'] == method]
            if len(data) > 0:
                ber = data['ber'].mean()
                rmse = data['rmse_tau_final'].mean()
                print(f"  {method:25s}: BER={ber:.4f}, RMSE={rmse:.4f}")

    return args.out_dir


# ============================================================================
# 主函数
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="运行实验并生成论文图表 (修复版 v2)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 论文级运行
  python run_experiments.py --ckpt checkpoint.pth --n_mc 20

  # 快速测试 (仅 debug)
  python run_experiments.py --ckpt checkpoint.pth --quick

  # 仅可视化
  python run_experiments.py --visualize_only --data_dir results/paper_figs
        """
    )

    # 模式选择
    parser.add_argument('--visualize_only', action='store_true',
                        help="仅从 CSV 生成图表")

    # 数据采集参数
    parser.add_argument('--ckpt', type=str, default="",
                        help="Checkpoint 路径")
    parser.add_argument('--snr_list', nargs='+', type=float,
                        default=[-5, 0, 5, 10, 15, 20, 25],
                        help="SNR 扫描值")
    parser.add_argument('--n_mc', type=int, default=20,
                        help="Monte Carlo 次数（论文级：20）")
    parser.add_argument('--batch', type=int, default=64,
                        help="Batch 大小")
    parser.add_argument('--init_error', type=float, default=0.3,
                        help="默认 init τ 误差 (samples)")
    parser.add_argument('--device', type=str, default="cuda",
                        help="设备 (cuda/cpu)")
    parser.add_argument('--quick', action='store_true',
                        help="快速模式（仅 debug 用）")

    # Sanity check
    parser.add_argument('--skip_sanity', action='store_true',
                        help="跳过 sanity check")
    parser.add_argument('--force', action='store_true',
                        help="即使 sanity check 失败也继续")

    # 输出参数
    parser.add_argument('--out_dir', type=str, default="results/paper_figs",
                        help="输出目录")
    parser.add_argument('--data_dir', type=str, default=None,
                        help="数据目录 (visualize_only 模式)")

    args = parser.parse_args()

    print("=" * 60)
    print("🎓 论文图表生成管线 (修复版 v2)")
    print("=" * 60)
    print(f"模式: {'仅可视化' if args.visualize_only else '完整运行'}")

    if args.visualize_only:
        # 仅可视化模式
        data_dir = args.data_dir or args.out_dir
        if not os.path.exists(data_dir):
            print(f"❌ 数据目录不存在: {data_dir}")
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
- Basin 边界约 0.3-0.5 samples
- BER 饱和是 1-bit 物理极限，真正增益在 τ RMSE
""")

    print(f"\n✅ 完成！所有输出保存到: {args.out_dir}")


if __name__ == "__main__":
    main()