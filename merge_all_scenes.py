"""
merge_all_scenes.py - 合并所有场景结果到单一 CSV 和图表

功能：
1. 读取所有场景的 metrics_raw.csv
2. 合并到单一 CSV 文件
3. 生成合并的多子图 PNG/PDF (4个场景在一个图中)
4. 生成单独的对比图

Usage:
    python merge_all_scenes.py --input_dir results/p4_experiments --output_dir results/p4_merged
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from pathlib import Path

# 设置 matplotlib 风格
plt.rcParams.update({
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'legend.fontsize': 8,
    'figure.dpi': 150,
    'figure.facecolor': 'white',
    'axes.grid': True,
    'grid.alpha': 0.3
})

# 场景配置
SCENE_CONFIG = {
    'S1_full_hw': {'label': 'Full HW (PA+PN+1bit)', 'color': '#e41a1c', 'marker': 'o'},
    'S2_pa_only': {'label': 'PA Only', 'color': '#377eb8', 'marker': 's'},
    'S3_pn_only': {'label': 'PN Only', 'color': '#4daf4a', 'marker': '^'},
    'S4_ideal': {'label': 'Ideal', 'color': '#984ea3', 'marker': 'D'},
}

def load_all_scenes(input_dir: Path) -> pd.DataFrame:
    """Load and merge all scene results."""
    all_data = []

    for scene_id in SCENE_CONFIG.keys():
        scene_dir = input_dir / scene_id
        raw_csv = scene_dir / 'metrics_raw.csv'

        if raw_csv.exists():
            df = pd.read_csv(raw_csv)
            df['scene_id'] = scene_id
            all_data.append(df)
            print(f"  [✓] Loaded {scene_id}: {len(df)} rows")
        else:
            print(f"  [✗] Not found: {raw_csv}")

    if all_data:
        combined = pd.concat(all_data, ignore_index=True)
        return combined
    else:
        return pd.DataFrame()

def compute_mean_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and std for each scene and SNR."""
    # 定义聚合函数
    agg_funcs = {
        'gamma_eff': ['mean', 'std'],
        'chi': ['mean', 'std'],
        'BER_Net': ['mean', 'std'],
        'NMSE_Net': ['mean', 'std'],
    }

    # 添加可选列
    if 'BER_LMMSE' in df.columns:
        agg_funcs['BER_LMMSE'] = ['mean', 'std']
    if 'BER_GAMP' in df.columns:
        agg_funcs['BER_GAMP'] = ['mean', 'std']

    mean_df = df.groupby(['scene_id', 'snr_db']).agg(agg_funcs).reset_index()

    # 展平列名
    mean_df.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                       for col in mean_df.columns]

    # 计算 gamma_eff_db
    mean_df['gamma_eff_db_mean'] = 10 * np.log10(mean_df['gamma_eff_mean'] + 1e-12)

    return mean_df

def create_combined_4panel_plot(mean_df: pd.DataFrame, output_dir: Path, filename: str = 'combined_4panel'):
    """Create a 2x2 combined plot with all 4 metrics."""

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    scenes = mean_df['scene_id'].unique()

    # ===== Panel 1: BER vs SNR =====
    ax = axes[0, 0]
    for scene_id in scenes:
        if scene_id not in SCENE_CONFIG:
            continue
        cfg = SCENE_CONFIG[scene_id]
        df_s = mean_df[mean_df['scene_id'] == scene_id]

        # GA-BV-Net
        ax.semilogy(df_s['snr_db'], df_s['BER_Net_mean'],
                   marker=cfg['marker'], color=cfg['color'],
                   label=f"Net: {cfg['label']}", linewidth=2, markersize=6)

    # 添加 LMMSE baseline (只画一条)
    if 'BER_LMMSE_mean' in mean_df.columns:
        df_s1 = mean_df[mean_df['scene_id'] == 'S1_full_hw']
        if len(df_s1) > 0:
            ax.semilogy(df_s1['snr_db'], df_s1['BER_LMMSE_mean'],
                       'k--', label='LMMSE (S1)', linewidth=1.5, alpha=0.7)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random (0.5)')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('BER')
    ax.set_title('(a) BER vs SNR')
    ax.legend(loc='upper right', fontsize=7)
    ax.set_ylim([1e-4, 1])
    ax.set_xlim([mean_df['snr_db'].min()-1, mean_df['snr_db'].max()+1])

    # ===== Panel 2: NMSE vs SNR =====
    ax = axes[0, 1]
    for scene_id in scenes:
        if scene_id not in SCENE_CONFIG:
            continue
        cfg = SCENE_CONFIG[scene_id]
        df_s = mean_df[mean_df['scene_id'] == scene_id]

        ax.plot(df_s['snr_db'], df_s['NMSE_Net_mean'],
               marker=cfg['marker'], color=cfg['color'],
               label=cfg['label'], linewidth=2, markersize=6)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('NMSE (dB)')
    ax.set_title('(b) NMSE vs SNR')
    ax.legend(loc='upper right', fontsize=8)

    # ===== Panel 3: Gamma_eff vs SNR =====
    ax = axes[1, 0]
    for scene_id in scenes:
        if scene_id not in SCENE_CONFIG:
            continue
        cfg = SCENE_CONFIG[scene_id]
        df_s = mean_df[mean_df['scene_id'] == scene_id]

        ax.plot(df_s['snr_db'], df_s['gamma_eff_db_mean'],
               marker=cfg['marker'], color=cfg['color'],
               label=cfg['label'], linewidth=2, markersize=6)

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Γ_eff (dB)')
    ax.set_title('(c) Effective SNR Degradation (Γ_eff)')
    ax.legend(loc='lower right', fontsize=8)

    # ===== Panel 4: Chi vs SNR =====
    ax = axes[1, 1]
    for scene_id in scenes:
        if scene_id not in SCENE_CONFIG:
            continue
        cfg = SCENE_CONFIG[scene_id]
        df_s = mean_df[mean_df['scene_id'] == scene_id]

        ax.plot(df_s['snr_db'], df_s['chi_mean'],
               marker=cfg['marker'], color=cfg['color'],
               label=cfg['label'], linewidth=2, markersize=6)

    ax.axhline(y=2/np.pi, color='gray', linestyle='--', alpha=0.5, label='χ_max = 2/π')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('χ (Information Retention)')
    ax.set_title('(d) Information Retention Factor (χ)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim([0, 0.7])

    plt.tight_layout()

    # 保存
    fig.savefig(output_dir / f'{filename}.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / f'{filename}.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  [✓] Saved: {output_dir}/{filename}.png/pdf")

def create_ber_comparison_plot(mean_df: pd.DataFrame, output_dir: Path):
    """Create a detailed BER comparison plot (Net vs Baselines)."""

    fig, ax = plt.subplots(figsize=(10, 6))

    scenes = mean_df['scene_id'].unique()

    for scene_id in scenes:
        if scene_id not in SCENE_CONFIG:
            continue
        cfg = SCENE_CONFIG[scene_id]
        df_s = mean_df[mean_df['scene_id'] == scene_id]

        # GA-BV-Net (实线)
        ax.semilogy(df_s['snr_db'], df_s['BER_Net_mean'],
                   marker=cfg['marker'], color=cfg['color'],
                   label=f"Net: {cfg['label']}", linewidth=2, markersize=6)

        # LMMSE (虚线)
        if 'BER_LMMSE_mean' in df_s.columns:
            ax.semilogy(df_s['snr_db'], df_s['BER_LMMSE_mean'],
                       linestyle='--', color=cfg['color'],
                       label=f"LMMSE: {cfg['label']}", linewidth=1.5, alpha=0.6)

    ax.axhline(y=0.5, color='gray', linestyle=':', alpha=0.5, label='Random Guess')
    ax.set_xlabel('SNR (dB)', fontsize=12)
    ax.set_ylabel('BER', fontsize=12)
    ax.set_title('BER Comparison: GA-BV-Net vs LMMSE Baseline', fontsize=14)
    ax.legend(loc='upper right', fontsize=8, ncol=2)
    ax.set_ylim([1e-4, 1])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.savefig(output_dir / 'ber_comparison.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'ber_comparison.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  [✓] Saved: {output_dir}/ber_comparison.png/pdf")

def create_gamma_chi_scatter(raw_df: pd.DataFrame, output_dir: Path):
    """Create Gamma_eff vs Chi scatter plot."""

    fig, ax = plt.subplots(figsize=(8, 6))

    for scene_id in raw_df['scene_id'].unique():
        if scene_id not in SCENE_CONFIG:
            continue
        cfg = SCENE_CONFIG[scene_id]
        df_s = raw_df[raw_df['scene_id'] == scene_id]

        gamma_db = 10 * np.log10(df_s['gamma_eff'] + 1e-12)
        ax.scatter(gamma_db, df_s['chi'],
                  c=cfg['color'], marker=cfg['marker'],
                  label=cfg['label'], alpha=0.5, s=20)

    ax.set_xlabel('Γ_eff (dB)', fontsize=12)
    ax.set_ylabel('χ', fontsize=12)
    ax.set_title('Hardware Impairment Distribution: Γ_eff vs χ', fontsize=14)
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    fig.savefig(output_dir / 'gamma_chi_scatter.png', dpi=150, bbox_inches='tight')
    fig.savefig(output_dir / 'gamma_chi_scatter.pdf', bbox_inches='tight')
    plt.close(fig)

    print(f"  [✓] Saved: {output_dir}/gamma_chi_scatter.png/pdf")

def print_summary(mean_df: pd.DataFrame):
    """Print summary statistics."""

    print("\n" + "=" * 70)
    print("SUMMARY STATISTICS")
    print("=" * 70)

    for scene_id in mean_df['scene_id'].unique():
        if scene_id not in SCENE_CONFIG:
            continue
        df_s = mean_df[mean_df['scene_id'] == scene_id]
        cfg = SCENE_CONFIG[scene_id]

        print(f"\n{scene_id} ({cfg['label']}):")
        print(f"  Γ_eff: {df_s['gamma_eff_db_mean'].mean():.1f} dB (range: {df_s['gamma_eff_db_mean'].min():.1f} - {df_s['gamma_eff_db_mean'].max():.1f})")
        print(f"  χ: {df_s['chi_mean'].mean():.4f} (range: {df_s['chi_mean'].min():.4f} - {df_s['chi_mean'].max():.4f})")
        print(f"  BER_Net: {df_s['BER_Net_mean'].mean():.4f} (range: {df_s['BER_Net_mean'].min():.4f} - {df_s['BER_Net_mean'].max():.4f})")

        if df_s['BER_Net_mean'].mean() > 0.4:
            print(f"  ⚠️ WARNING: BER ≈ 0.5 - Model may not be working!")

        if 'BER_LMMSE_mean' in df_s.columns:
            print(f"  BER_LMMSE: {df_s['BER_LMMSE_mean'].mean():.4f} (range: {df_s['BER_LMMSE_mean'].min():.4f} - {df_s['BER_LMMSE_mean'].max():.4f})")

def main():
    parser = argparse.ArgumentParser(description="Merge all scene results")
    parser.add_argument('--input_dir', type=str, default='results/p4_experiments',
                        help='Input directory with scene subfolders')
    parser.add_argument('--output_dir', type=str, default='results/p4_merged',
                        help='Output directory for merged results')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Merging All Scene Results")
    print("=" * 70)
    print(f"Input: {input_dir}")
    print(f"Output: {output_dir}")

    # 1. Load all scenes
    print("\n[1/5] Loading scene data...")
    raw_df = load_all_scenes(input_dir)

    if raw_df.empty:
        print("\n❌ No data found! Please run experiments first.")
        return

    # 2. Save merged raw CSV
    print("\n[2/5] Saving merged raw CSV...")
    raw_df.to_csv(output_dir / 'all_scenes_raw.csv', index=False)
    print(f"  [✓] Saved: {output_dir}/all_scenes_raw.csv ({len(raw_df)} rows)")

    # 3. Compute mean stats
    print("\n[3/5] Computing mean statistics...")
    mean_df = compute_mean_stats(raw_df)
    mean_df.to_csv(output_dir / 'all_scenes_mean.csv', index=False)
    print(f"  [✓] Saved: {output_dir}/all_scenes_mean.csv ({len(mean_df)} rows)")

    # 4. Create plots
    print("\n[4/5] Creating plots...")
    create_combined_4panel_plot(mean_df, output_dir)
    create_ber_comparison_plot(mean_df, output_dir)
    create_gamma_chi_scatter(raw_df, output_dir)

    # 5. Print summary
    print_summary(mean_df)

    print("\n" + "=" * 70)
    print("COMPLETE!")
    print("=" * 70)
    print(f"\nOutput files in: {output_dir}")
    print("  - all_scenes_raw.csv (所有原始数据)")
    print("  - all_scenes_mean.csv (平均统计)")
    print("  - combined_4panel.png/pdf (4子图合并)")
    print("  - ber_comparison.png/pdf (BER对比)")
    print("  - gamma_chi_scatter.png/pdf (Γ_eff vs χ 散点图)")

if __name__ == "__main__":
    main()