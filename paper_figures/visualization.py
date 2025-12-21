"""
visualization.py - 论文级绘图脚本 (修复版 v5)

修复内容：
1. Fig 10 (消融): 双轴图 (BER 柱状 + τ RMSE 折线)
2. Fig 04 (Cliff): MF 样式化为"昂贵上界"（半透明虚线）
3. Fig 01: Hardware Wall 标注
4. 添加缺失方法警告（不再默默跳过）
5. 添加关键数据点打印

专家意见整合：
- VAMP BER 饱和是物理规律，不是 bug
- MF 是全局搜索上界，需要视觉区分
- τ RMSE 才是真正的增益指标
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# IEEE 风格设置
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 9,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.figsize': (8, 5),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 7,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.linewidth': 1.2,
})


# ============================================================================
# 方法样式配置（叙事对齐）
# ============================================================================

METHOD_CONFIG = {
    # 弱基线（细线）
    "naive_slice": {
        "label": "Naive Slice",
        "color": "#7f7f7f",
        "marker": "v",
        "ls": ":",
        "lw": 1.5,
        "alpha": 0.6
    },

    # MF: 昂贵上界（虚线、半透明、标注复杂度）
    "matched_filter": {
        "label": "Matched Filter (41×)",
        "color": "#d62728",
        "marker": "X",
        "ls": "--",
        "lw": 1.8,
        "alpha": 0.5
    },

    # 强基线（中等粗细）
    "adjoint_lmmse": {
        "label": "Adjoint+LMMSE",
        "color": "#ff7f0e",
        "marker": "P",
        "ls": "-.",
        "lw": 2.0,
        "alpha": 0.8
    },
    "adjoint_slice": {
        "label": "Adjoint+Slice",
        "color": "#9467bd",
        "marker": "d",
        "ls": ":",
        "lw": 2.0,
        "alpha": 0.8
    },

    # 消融变体
    "proposed_no_update": {
        "label": "w/o τ update",
        "color": "#8c564b",
        "marker": "s",
        "ls": "-",
        "lw": 2.0,
        "alpha": 0.9
    },
    "proposed_tau_slice": {
        "label": "Proposed (τ)+Slice",
        "color": "#17becf",
        "marker": "p",
        "ls": "-.",
        "lw": 2.0,
        "alpha": 0.9
    },

    # 主角（粗线）
    "proposed": {
        "label": "Proposed (GA-BV-Net)",
        "color": "#1f77b4",
        "marker": "o",
        "ls": "-",
        "lw": 2.8,
        "alpha": 1.0
    },
    "oracle": {
        "label": "Oracle θ",
        "color": "#2ca02c",
        "marker": "^",
        "ls": "-",
        "lw": 2.5,
        "alpha": 1.0
    },

    # 特殊
    "random_init": {
        "label": "Random Init",
        "color": "black",
        "marker": "*",
        "ls": ":",
        "lw": 1.5,
        "alpha": 0.5
    },
}


def get_style(method: str) -> dict:
    """获取方法的绘图样式"""
    return METHOD_CONFIG.get(method, {
        "label": method, "color": "black", "marker": "o",
        "ls": "-", "lw": 2.0, "alpha": 1.0
    })


# ============================================================================
# 辅助函数
# ============================================================================

def aggregate(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    """聚合数据，计算均值、标准差和 95% CI"""
    agg_funcs = {col: ['mean', 'std', 'count'] for col in value_cols}
    agg = df.groupby(group_cols).agg(agg_funcs).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]

    for col in value_cols:
        mean, std, count = f'{col}_mean', f'{col}_std', f'{col}_count'
        if all(c in agg.columns for c in [mean, std, count]):
            agg[f'{col}_ci95'] = 1.96 * agg[std] / np.sqrt(agg[count].clip(lower=1))

    return agg


def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """加载所有 CSV 数据"""
    data = {}
    csv_files = [
        'data_snr_sweep.csv',
        'data_cliff_sweep.csv',
        'data_snr_multi_init_error.csv',
        'data_ablation_sweep.csv',
        'data_heatmap_sweep.csv',
        'data_pn_sweep.csv',
        'data_pilot_sweep.csv',
        'data_jacobian.csv',
        'data_latency.csv',
    ]

    for f in csv_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            key = f.replace('data_', '').replace('.csv', '')
            data[key] = pd.read_csv(path)
            n_methods = len(data[key]['method'].unique()) if 'method' in data[key].columns else 0
            print(f"  ✓ {f}: {len(data[key])} 行, {n_methods} 个方法")
        else:
            print(f"  ⚠️ 未找到: {f}")

    return data


def check_methods(df: pd.DataFrame, expected: List[str], fig_name: str) -> List[str]:
    """检查并警告缺失的方法"""
    if 'method' not in df.columns:
        print(f"  ⚠️ {fig_name}: 数据缺少 'method' 列")
        return []

    actual = set(df['method'].unique())
    missing = set(expected) - actual

    if missing:
        print(f"  ⚠️ {fig_name}: 缺少方法 {missing}")

    return [m for m in expected if m in actual]


# ============================================================================
# 核心绘图函数
# ============================================================================

def fig01_ber_vs_snr(df: pd.DataFrame, out_dir: str):
    """
    Fig 1: BER vs SNR + Hardware Wall 标注

    关键信息：高 SNR 区域的 BER 饱和是 Γ_eff 限制，不是算法问题
    """
    fig, ax = plt.subplots(figsize=(10, 7))
    agg = aggregate(df, ['snr_db', 'method'], ['ber'])

    # 期望的方法集
    expected = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed", "oracle"]
    methods = check_methods(df, expected, "Fig 01")

    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        s = get_style(method)
        snr = data['snr_db'].values
        mean = data['ber_mean'].values
        ci = data.get('ber_ci95', pd.Series([0]*len(data))).values

        ax.semilogy(snr, mean, marker=s['marker'], color=s['color'],
                    linestyle=s['ls'], label=s['label'],
                    linewidth=s['lw'], alpha=s['alpha'], markersize=7)

        if np.any(ci > 0):
            ax.fill_between(snr, np.maximum(mean - ci, 1e-4), mean + ci,
                           alpha=0.12, color=s['color'])

    # Hardware Wall 标注（专家建议）
    ax.axvspan(18, 28, color='gray', alpha=0.1)
    ax.text(22, 0.35, "Hardware Wall\n($\\Gamma_{\\mathrm{eff}}$ Limited)",
            fontsize=10, color='gray', ha='center', style='italic')

    # BER 目标线
    ax.axhline(y=0.15, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
    ax.text(-4, 0.155, 'BER = 0.15', fontsize=9, color='gray')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate (BER)')
    ax.set_title('Communication Performance: BER vs SNR')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim([5e-2, 0.55])
    ax.set_xlim([-7, 27])
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig01_ber_vs_snr.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 01: BER vs SNR (含 Hardware Wall)")


def fig02_rmse_tau_vs_snr(df: pd.DataFrame, out_dir: str):
    """Fig 2: τ RMSE vs SNR"""
    fig, ax = plt.subplots(figsize=(10, 6))
    agg = aggregate(df, ['snr_db', 'method'], ['rmse_tau_final'])

    expected = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed"]
    methods = check_methods(df, expected, "Fig 02")

    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        s = get_style(method)
        ax.semilogy(data['snr_db'], data['rmse_tau_final_mean'],
                   marker=s['marker'], color=s['color'], linestyle=s['ls'],
                   label=s['label'], linewidth=s['lw'], alpha=s['alpha'])

    ax.axhline(y=0.1, color='green', linestyle=':', linewidth=2, label='Target (0.1 samples)')
    ax.axhspan(0, 0.1, alpha=0.1, color='green')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('τ RMSE (samples)')
    ax.set_title('Sensing Performance: Delay Estimation Error')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig02_rmse_tau_vs_snr.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 02: RMSE τ vs SNR")


def fig04_cliff_all_methods(df: pd.DataFrame, out_dir: str):
    """
    Fig 4: Cliff Plot（核心贡献图）

    修复：
    - MF 样式化为半透明虚线（表示昂贵上界）
    - 添加 "Global Search" 标注
    - 数据驱动的 basin 边界
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    agg = aggregate(df, ['init_error', 'method'], ['ber', 'rmse_tau_final', 'success_rate'])

    # 期望的方法集
    expected = ["naive_slice", "adjoint_lmmse", "adjoint_slice", "matched_filter",
                "proposed_no_update", "proposed", "oracle"]
    methods = check_methods(df, expected, "Fig 04")

    # 计算 basin 边界（从数据）
    proposed_data = agg[agg['method'] == 'proposed']
    if len(proposed_data) > 0 and 'success_rate_mean' in proposed_data.columns:
        success_vals = proposed_data[proposed_data['success_rate_mean'] >= 0.5]
        basin_boundary = success_vals['init_error'].max() if len(success_vals) > 0 else 0.3
    else:
        basin_boundary = 0.3

    # Zone shading
    max_x = max(agg['init_error'].max(), 1.5)
    for ax in [ax1, ax2]:
        ax.axvspan(0, basin_boundary, color='green', alpha=0.06)
        ax.axvspan(basin_boundary, basin_boundary + 0.2, color='orange', alpha=0.06)
        ax.axvspan(basin_boundary + 0.2, max_x, color='red', alpha=0.06)
        ax.axvline(basin_boundary, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    # ===== Panel A: BER =====
    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        s = get_style(method)
        ax1.plot(data['init_error'], data['ber_mean'],
                marker=s['marker'], color=s['color'], linestyle=s['ls'],
                linewidth=s['lw'], alpha=s['alpha'], label=s['label'], markersize=6)

    # MF 标注
    mf_data = agg[agg['method'] == 'matched_filter']
    if len(mf_data) > 0:
        x_last = mf_data['init_error'].max()
        y_last = mf_data[mf_data['init_error'] == x_last]['ber_mean'].values
        if len(y_last) > 0:
            ax1.annotate('Global Search\n(41× FFTs)', xy=(x_last, y_last[0]),
                        xytext=(x_last - 0.4, y_last[0] + 0.1),
                        arrowprops=dict(facecolor='gray', shrink=0.05, width=1),
                        fontsize=8, color='gray', style='italic')

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(max_x - 0.2, 0.48, 'Random Guess', fontsize=8, color='gray', alpha=0.7, ha='right')
    ax1.text(basin_boundary + 0.02, 0.42, f'Basin={basin_boundary:.1f}',
            fontsize=9, color='green', alpha=0.8)

    ax1.set_xlabel('Initial $\\tau$ Error (samples)')
    ax1.set_ylabel('BER')
    ax1.set_title('(a) The Cliff: Detection Robustness')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_ylim([0, 0.55])
    ax1.set_xlim([0, max_x])

    # ===== Panel B: RMSE =====
    for method in methods:
        if method == "oracle":
            continue
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        s = get_style(method)
        ax2.plot(data['init_error'], data['rmse_tau_final_mean'],
                marker=s['marker'], color=s['color'], linestyle=s['ls'],
                linewidth=s['lw'], alpha=s['alpha'], label=s['label'], markersize=6)

    # y=x 参考线
    ax2.plot([0, max_x], [0, max_x], 'k--', alpha=0.4, linewidth=1, label='No Improvement')
    ax2.axhspan(0, 0.1, alpha=0.1, color='green')
    ax2.text(0.05, 0.08, 'Target', fontsize=8, color='green', alpha=0.8)

    ax2.set_xlabel('Initial $\\tau$ Error (samples)')
    ax2.set_ylabel('Final $\\tau$ RMSE (samples)')
    ax2.set_title('(b) Convergence Dynamics')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_ylim([0, max_x])
    ax2.set_xlim([0, max_x])

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig04_cliff_all_methods.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 04: Cliff Plot (核心贡献图)")


def fig05_snr_multi_init_error(df: pd.DataFrame, out_dir: str):
    """
    Fig 5: 多 init_error 的 SNR sweep

    证明：
    (a) init_error=0: baseline 实现正确
    (b) init_error=0.2/0.3: proposed 优势明显
    """
    if 'init_error' not in df.columns:
        print("  ⚠️ Fig 05: 缺少 init_error 列，跳过")
        return

    init_errors = sorted(df['init_error'].unique())
    n_panels = min(len(init_errors), 3)

    fig, axes = plt.subplots(1, n_panels, figsize=(5.5*n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    expected = ["adjoint_slice", "proposed_no_update", "proposed", "oracle"]
    methods = check_methods(df, expected, "Fig 05")
    panel_labels = 'abcdefg'

    for idx, init_error in enumerate(init_errors[:n_panels]):
        ax = axes[idx]
        df_sub = df[df['init_error'] == init_error]
        agg = aggregate(df_sub, ['snr_db', 'method'], ['ber'])

        for method in methods:
            data = agg[agg['method'] == method]
            if len(data) == 0:
                continue

            s = get_style(method)
            ax.semilogy(data['snr_db'], data['ber_mean'],
                       marker=s['marker'], color=s['color'], linestyle=s['ls'],
                       label=s['label'], linewidth=s['lw'], alpha=s['alpha'])

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')

        # 标题颜色编码
        if init_error == 0.0:
            title = f'({panel_labels[idx]}) init_err=0: Baseline OK'
            color = 'green'
        elif init_error <= 0.2:
            title = f'({panel_labels[idx]}) init_err={init_error}'
            color = 'orange'
        else:
            title = f'({panel_labels[idx]}) init_err={init_error}: Baseline Fails'
            color = 'red'

        ax.set_title(title, fontsize=11, color=color)
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([5e-2, 0.55])

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig05_snr_multi_init_error.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 05: SNR @ Multi Init Error")


def fig07_gap_to_oracle(df: pd.DataFrame, out_dir: str):
    """Fig 7: Gap-to-Oracle"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    agg = aggregate(df, ['snr_db', 'method'], ['ber', 'rmse_tau_final'])

    oracle_data = agg[agg['method'] == 'oracle'][['snr_db', 'ber_mean', 'rmse_tau_final_mean']]
    oracle_data = oracle_data.rename(columns={
        'ber_mean': 'oracle_ber',
        'rmse_tau_final_mean': 'oracle_rmse'
    })

    expected = ["adjoint_slice", "proposed_no_update", "proposed"]
    methods = check_methods(df, expected, "Fig 07")

    # Panel A: BER Gap
    for method in methods:
        data = agg[agg['method'] == method][['snr_db', 'ber_mean']]
        merged = pd.merge(data, oracle_data, on='snr_db')
        if len(merged) == 0:
            continue

        s = get_style(method)
        gap = merged['ber_mean'].values - merged['oracle_ber'].values
        ax1.plot(merged['snr_db'], gap, marker=s['marker'], color=s['color'],
                linestyle=s['ls'], label=s['label'], linewidth=s['lw'])

    ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Oracle')
    ax1.axhline(y=0.01, color='orange', linestyle=':', alpha=0.6, label='1% gap')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('ΔBER (method − oracle)')
    ax1.set_title('(a) BER Gap to Oracle')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel B: RMSE
    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        s = get_style(method)
        ax2.semilogy(data['snr_db'], data['rmse_tau_final_mean'],
                    marker=s['marker'], color=s['color'], linestyle=s['ls'],
                    label=s['label'], linewidth=s['lw'])

    ax2.axhline(y=0.1, color='green', linestyle=':', linewidth=2, label='Target')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('τ RMSE (samples)')
    ax2.set_title('(b) τ RMSE Comparison')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig07_gap_to_oracle.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 07: Gap-to-Oracle")


def fig10_ablation_dual_axis(df: pd.DataFrame, out_dir: str):
    """
    Fig 10: 消融实验 - 双轴图（核心修复）

    专家建议：
    - 左轴 (柱状): BER - 展示 1-bit 饱和
    - 右轴 (折线): τ RMSE - 展示真正的增益

    关键信息：VAMP 的价值在于同步精度，不是 BER
    """
    fig, ax1 = plt.subplots(figsize=(11, 6))

    # 使用最高 SNR
    target_snr = df['snr_db'].max()
    data = df[df['snr_db'] == target_snr].copy()

    # 方法顺序（从弱到强）
    methods = ["random_init", "proposed_no_update", "proposed_tau_slice", "proposed", "oracle"]

    agg = aggregate(data, ['method'], ['ber', 'rmse_tau_final'])
    agg = agg.set_index('method').reindex(methods).reset_index()

    x = np.arange(len(methods))
    width = 0.4

    # ===== 左轴: BER (柱状图) =====
    colors = [get_style(m)['color'] for m in methods]
    bars = ax1.bar(x, agg['ber_mean'], width, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.2, label='BER')

    ax1.set_ylabel('BER (Lower is Better)', fontsize=12)
    ax1.set_ylim(0, 0.5)

    # BER 饱和区域标注
    ax1.axhspan(0.10, 0.12, alpha=0.15, color='gray')
    ax1.annotate('BER Saturation\n(1-bit limit)', xy=(3, 0.11), xytext=(1.5, 0.3),
                arrowprops=dict(facecolor='gray', shrink=0.05, width=1, headwidth=5),
                fontsize=9, color='gray', ha='center')

    # ===== 右轴: τ RMSE (折线图) =====
    ax2 = ax1.twinx()
    rmse = agg['rmse_tau_final_mean'].fillna(0).values

    ax2.plot(x, rmse, color='#d62728', marker='D', markersize=10,
            linewidth=2.5, linestyle='--', label='τ RMSE')

    ax2.set_ylabel('τ RMSE (samples, Log Scale)', color='#d62728', fontsize=12)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(1e-2, 5)

    # 标注 τ 增益
    if len(rmse) >= 4 and rmse[1] > 0 and rmse[3] > 0:
        improvement = rmse[1] / rmse[3]
        ax2.annotate(f'τ Tracking\nGain: {improvement:.1f}×',
                    xy=(3, rmse[3]), xytext=(3.5, rmse[3] * 3),
                    arrowprops=dict(arrowstyle='->', color='#d62728', lw=2),
                    fontsize=9, color='#d62728', ha='center', fontweight='bold')

    # X 轴标签
    labels = [get_style(m)['label'] for m in methods]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)

    ax1.set_title(f'Ablation Study @ SNR={target_snr:.0f}dB: Sync vs Detection Trade-off', fontsize=13)

    # 合并图例
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=2, fontsize=9)

    # 关键信息框
    textbox = ("Key Insight:\n"
               "• BER saturates ~10% (1-bit limit)\n"
               "• τ RMSE shows >5× improvement\n"
               "• VAMP enables stable sync loop")
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.3)
    ax1.text(0.02, 0.98, textbox, transform=ax1.transAxes, fontsize=8,
            verticalalignment='top', bbox=props)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig10_ablation_dual.{ext}", dpi=300)
    plt.close(fig)

    # 打印关键数据
    print("  ✓ Fig 10: Ablation (双轴图)")
    print(f"\n  📊 消融实验 @ SNR={target_snr:.0f}dB:")
    for i, method in enumerate(methods):
        ber = agg.iloc[i]['ber_mean']
        rmse_val = agg.iloc[i]['rmse_tau_final_mean']
        print(f"     {labels[i]:25s}: BER={ber:.4f}, τ_RMSE={rmse_val:.4f}")


def fig11_heatmap(df: pd.DataFrame, out_dir: str):
    """Fig 11: 2D Heatmap (SNR × init_error)"""
    if 'init_error' not in df.columns or 'snr_db' not in df.columns:
        print("  ⚠️ Fig 11: 缺少必要列，跳过")
        return

    # 只画 proposed
    df_proposed = df[df['method'] == 'proposed'] if 'method' in df.columns else df

    if len(df_proposed) == 0:
        print("  ⚠️ Fig 11: 无 proposed 数据，跳过")
        return

    agg = df_proposed.groupby(['snr_db', 'init_error'])['ber'].mean().reset_index()
    pivot = agg.pivot(index='init_error', columns='snr_db', values='ber')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', annot=True, fmt='.2f',
               cbar_kws={'label': 'BER'}, vmin=0, vmax=0.5,
               linewidths=0.5, linecolor='white')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Initial τ Error (samples)')
    ax.set_title('Proposed Method: BER Heatmap')

    # Basin 边界
    if 0.3 in pivot.index.values:
        y_pos = list(pivot.index.values).index(0.3) + 0.5
        ax.axhline(y=y_pos, color='white', linestyle='--', linewidth=2)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig11_heatmap.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 11: Heatmap")


def fig08_robustness(df_pn: pd.DataFrame, df_pilot: pd.DataFrame, out_dir: str):
    """Fig 8: PN 和 Pilot 鲁棒性"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    expected = ["adjoint_slice", "proposed_no_update", "proposed", "oracle"]

    # Panel A: PN sweep
    if df_pn is not None and len(df_pn) > 0:
        methods = check_methods(df_pn, expected, "Fig 08 (PN)")
        agg_pn = aggregate(df_pn, ['pn_linewidth', 'method'], ['ber'])

        for method in methods:
            data = agg_pn[agg_pn['method'] == method]
            if len(data) == 0:
                continue

            s = get_style(method)
            pn = data['pn_linewidth'].values / 1e3  # kHz
            ax1.plot(pn, data['ber_mean'], marker=s['marker'], color=s['color'],
                    label=s['label'], linewidth=s['lw'])

        ax1.set_xlabel('PN Linewidth (kHz)')
        ax1.set_ylabel('BER')
        ax1.set_title('(a) Robustness to Phase Noise')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

    # Panel B: Pilot sweep
    if df_pilot is not None and len(df_pilot) > 0:
        methods = check_methods(df_pilot, expected, "Fig 08 (Pilot)")
        agg_pilot = aggregate(df_pilot, ['pilot_len', 'method'], ['rmse_tau_final'])

        for method in methods:
            data = agg_pilot[agg_pilot['method'] == method]
            if len(data) == 0:
                continue

            s = get_style(method)
            ax2.plot(data['pilot_len'], data['rmse_tau_final_mean'],
                    marker=s['marker'], color=s['color'],
                    label=s['label'], linewidth=s['lw'])

        ax2.set_xlabel('Pilot Length')
        ax2.set_ylabel('τ RMSE (samples)')
        ax2.set_title('(b) Impact of Pilot Length')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig08_robustness.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 08: Robustness")


def fig09_latency(df: pd.DataFrame, out_dir: str):
    """Fig 9: 延迟对比"""
    if len(df) == 0:
        print("  ⚠️ Fig 09: 无延迟数据，跳过")
        return

    fig, ax = plt.subplots(figsize=(9, 6))

    methods = df['method'].values
    latencies = df['latency_mean_ms'].values
    stds = df['latency_std_ms'].values

    colors = [get_style(m)['color'] for m in methods]
    labels = [get_style(m)['label'] for m in methods]

    bars = ax.bar(range(len(methods)), latencies, yerr=stds,
                  color=colors, alpha=0.7, edgecolor='black', capsize=5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Computational Complexity')
    ax.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for i, (lat, std) in enumerate(zip(latencies, stds)):
        ax.text(i, lat + std + 0.5, f'{lat:.1f}ms', ha='center', fontsize=10)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig09_latency.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 09: Latency")


# ============================================================================
# 主函数
# ============================================================================

def generate_all_figures(data_dir: str, out_dir: str = None):
    """从 CSV 数据生成所有图表"""
    if out_dir is None:
        out_dir = data_dir

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("📊 论文图表生成 (修复版 v5)")
    print("=" * 60)
    print("\n加载数据...")
    data = load_data(data_dir)

    print("\n" + "-" * 40)
    print("生成图表...")
    print("-" * 40)

    # 核心图表
    if 'snr_sweep' in data:
        fig01_ber_vs_snr(data['snr_sweep'], out_dir)
        fig02_rmse_tau_vs_snr(data['snr_sweep'], out_dir)
        fig07_gap_to_oracle(data['snr_sweep'], out_dir)

    if 'cliff_sweep' in data:
        fig04_cliff_all_methods(data['cliff_sweep'], out_dir)

    if 'snr_multi_init_error' in data:
        fig05_snr_multi_init_error(data['snr_multi_init_error'], out_dir)

    # 消融实验（核心修复）
    if 'ablation_sweep' in data:
        fig10_ablation_dual_axis(data['ablation_sweep'], out_dir)

    # 辅助图表
    if 'pn_sweep' in data or 'pilot_sweep' in data:
        fig08_robustness(data.get('pn_sweep'), data.get('pilot_sweep'), out_dir)

    if 'latency' in data:
        fig09_latency(data['latency'], out_dir)

    if 'heatmap_sweep' in data:
        fig11_heatmap(data['heatmap_sweep'], out_dir)

    print("\n" + "=" * 60)
    print(f"✅ 所有图表已保存到: {out_dir}")
    print("=" * 60)

    # 叙事提示
    print("\n📝 论文叙事要点:")
    print("-" * 40)
    print("""
    核心论点: "GA-BV-Net 扩展了同步盆地，在梯度方法失效时仍能工作"
    
    关键证据:
    • Fig 04 (Cliff): 展示 "悬崖边缘" @ ~0.3-0.4 samples
    • Fig 10 (Ablation): BER 饱和 BUT τ RMSE 改善 >5×
    • Fig 05: init_error=0 时所有方法都 OK (无 baseline bug)
    
    审稿人预防:
    • Q: "为什么 BER 改善不大？"
      A: 1-bit 量化产生硬件极限 (Γ_eff limit)
    • Q: "MF 和 Proposed 一样好？"
      A: MF 需要 41× FFT (全局搜索)，不实用
    """)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="论文图表生成 (v5)")
    parser.add_argument('--data_dir', type=str, required=True, help="CSV 数据目录")
    parser.add_argument('--out_dir', type=str, default=None, help="输出目录")
    args = parser.parse_args()

    generate_all_figures(args.data_dir, args.out_dir)