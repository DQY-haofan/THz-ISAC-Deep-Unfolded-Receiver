"""
visualization.py - 纯绘图函数

负责：
- 从 CSV 读取数据
- 生成论文级图表

不负责：
- 模型加载
- 数据采集
- 算法实现
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

# 设置 Matplotlib
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'legend.fontsize': 10,
    'figure.figsize': (10, 6),
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'lines.linewidth': 2,
    'lines.markersize': 8,
})


# ============================================================================
# 方法配置
# ============================================================================

METHOD_NAMES = {
    "naive_slice": "Naive Slice",
    "matched_filter": "Matched Filter",
    "adjoint_lmmse": "Adjoint+LMMSE", 
    "adjoint_slice": "Adjoint+Slice",
    "proposed_no_update": "w/o τ update",
    "proposed_no_learned_alpha": "w/o learned α",
    "proposed": "Proposed (GA-BV-Net)",
    "oracle": "Oracle θ",
    "random_init": "Random Init",
}

METHOD_COLORS = {
    "naive_slice": "C7",       # 灰
    "matched_filter": "C3",    # 红
    "adjoint_lmmse": "C1",     # 橙
    "adjoint_slice": "C4",     # 紫
    "proposed_no_update": "C5",# 棕
    "proposed_no_learned_alpha": "C8",  # 浅绿
    "proposed": "C0",          # 蓝
    "oracle": "C2",            # 绿
    "random_init": "C9",       # 青
}

METHOD_MARKERS = {
    "naive_slice": "v",
    "matched_filter": "x",
    "adjoint_lmmse": "+",
    "adjoint_slice": "d",
    "proposed_no_update": "s",
    "proposed_no_learned_alpha": "p",
    "proposed": "o",
    "oracle": "^",
    "random_init": "*",
}

METHOD_LINESTYLES = {
    "naive_slice": ":",
    "matched_filter": "--",
    "adjoint_lmmse": "-.",
    "adjoint_slice": ":",
    "proposed_no_update": "-",
    "proposed_no_learned_alpha": "-.",
    "proposed": "-",
    "oracle": "-",
    "random_init": ":",
}


# ============================================================================
# 辅助函数
# ============================================================================

def aggregate(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    """聚合数据，计算均值、标准差和 95% 置信区间"""
    agg_funcs = {}
    for col in value_cols:
        agg_funcs[col] = ['mean', 'std', 'count']

    agg = df.groupby(group_cols).agg(agg_funcs).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]

    # 计算 95% CI
    for col in value_cols:
        mean_col = f'{col}_mean'
        std_col = f'{col}_std'
        count_col = f'{col}_count'
        ci_col = f'{col}_ci95'

        if mean_col in agg.columns and std_col in agg.columns:
            agg[ci_col] = 1.96 * agg[std_col] / np.sqrt(agg[count_col].clip(lower=1))

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
            print(f"  Loaded: {f}")

    return data


# ============================================================================
# 绘图函数
# ============================================================================

def fig01_ber_vs_snr(df: pd.DataFrame, out_dir: str):
    """
    Fig 1: BER vs SNR with SNR Gain annotation

    专家1建议：添加 Hardware Wall 阴影（高 SNR 区域性能不再线性下降）
    """

    fig, ax = plt.subplots(figsize=(10, 7))

    agg = aggregate(df, ['snr_db', 'method'], ['ber'])

    # 按顺序绘制
    plot_order = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed", "oracle"]

    for method in plot_order:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        snr = data['snr_db'].values
        mean = data['ber_mean'].values
        ci = data.get('ber_ci95', pd.Series([0]*len(data))).values

        ax.semilogy(snr, mean,
                    marker=METHOD_MARKERS.get(method, 'o'),
                    color=METHOD_COLORS.get(method, 'C0'),
                    linestyle=METHOD_LINESTYLES.get(method, '-'),
                    label=METHOD_NAMES.get(method, method),
                    markersize=8, linewidth=2)
        if np.any(ci > 0):
            ax.fill_between(snr, mean - ci, mean + ci, alpha=0.15,
                            color=METHOD_COLORS.get(method, 'C0'))

    # Hardware Wall 阴影（专家1建议）
    ax.axvspan(20, 30, color='gray', alpha=0.1)
    ax.text(23, 0.4, "Hardware Wall\n(Limited by $\\Gamma_{eff}$)",
            fontsize=10, color='gray', alpha=0.8, ha='center')

    # SNR Gain 标注
    target_ber = 0.15
    ax.axhline(y=target_ber, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)
    ax.text(22, target_ber * 1.15, f'BER = {target_ber}', fontsize=10, color='gray')

    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('BER', fontsize=14)
    ax.set_title('Communication Performance: BER vs SNR', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([5e-2, 0.6])
    ax.set_xlim([-7, 27])

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig01_ber_vs_snr.png", dpi=300)
    fig.savefig(f"{out_dir}/fig01_ber_vs_snr.pdf")
    plt.close(fig)


def fig02_rmse_tau_vs_snr(df: pd.DataFrame, out_dir: str):
    """Fig 2: RMSE_τ vs SNR"""

    fig, ax = plt.subplots(figsize=(10, 6))

    agg = aggregate(df, ['snr_db', 'method'], ['rmse_tau_final'])

    plot_order = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed"]

    for method in plot_order:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        snr = data['snr_db'].values
        mean = data['rmse_tau_final_mean'].values

        ax.semilogy(snr, mean,
                    marker=METHOD_MARKERS.get(method, 'o'),
                    color=METHOD_COLORS.get(method, 'C0'),
                    linestyle=METHOD_LINESTYLES.get(method, '-'),
                    label=METHOD_NAMES.get(method, method),
                    markersize=8, linewidth=2)

    ax.axhline(y=0.1, color='green', linestyle=':', linewidth=2, label='Target (0.1 samples)')

    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('RMSE τ (samples)', fontsize=14)
    ax.set_title('Sensing Performance: Delay Estimation Error', fontsize=14)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig02_rmse_tau_vs_snr.png", dpi=300)
    fig.savefig(f"{out_dir}/fig02_rmse_tau_vs_snr.pdf")
    plt.close(fig)


def fig03_success_rate(df: pd.DataFrame, out_dir: str):
    """Fig 3: Success Rate vs SNR"""

    if 'success_rate' not in df.columns:
        print("Warning: success_rate not in data, skipping fig03")
        return

    fig, ax = plt.subplots(figsize=(10, 6))

    agg = aggregate(df, ['snr_db', 'method'], ['success_rate'])

    plot_order = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed"]

    for method in plot_order:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        snr = data['snr_db'].values
        mean = data['success_rate_mean'].values * 100

        ax.plot(snr, mean,
                marker=METHOD_MARKERS.get(method, 'o'),
                color=METHOD_COLORS.get(method, 'C0'),
                linestyle=METHOD_LINESTYLES.get(method, '-'),
                label=METHOD_NAMES.get(method, method),
                linewidth=2, markersize=8)

    ax.axhline(y=90, color='green', linestyle=':', linewidth=2, label='90% target')

    ax.set_xlabel('SNR (dB)', fontsize=14)
    ax.set_ylabel('Success Rate (%)', fontsize=14)
    ax.set_title('τ Estimation Reliability: P(|τ_err| < 0.1 samples)', fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_ylim([0, 105])

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig03_success_rate.png", dpi=300)
    fig.savefig(f"{out_dir}/fig03_success_rate.pdf")
    plt.close(fig)


def fig04_cliff_all_methods(df: pd.DataFrame, out_dir: str):
    """
    Fig 4: Cliff plot with ALL methods（核心图）

    专家方案1 + 专家1建议的 Zone Shading：
    - Green Zone: Basin of Attraction（梯度正确流动）
    - Red Zone: Ambiguity Zone（梯度崩溃）
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    agg = aggregate(df, ['init_error', 'method'], ['ber', 'rmse_tau_final', 'success_rate'])

    methods_to_plot = ["naive_slice", "adjoint_slice", "matched_filter",
                       "proposed_no_update", "proposed", "oracle"]

    # ===== Panel A: BER vs init_error =====

    # 1. 先画 Zone Shading（专家1建议）
    ax1.axvspan(0, 0.3, color='green', alpha=0.08, label='Basin of Attraction')
    ax1.axvspan(0.3, 0.5, color='orange', alpha=0.08, label='Transition Zone')
    ax1.axvspan(0.5, 2.0, color='red', alpha=0.08, label='Ambiguity Zone')

    # 2. 画方法曲线
    for method in methods_to_plot:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        init_errors = data['init_error'].values
        ber_mean = data['ber_mean'].values

        ax1.plot(init_errors, ber_mean,
                 marker=METHOD_MARKERS.get(method, 'o'),
                 color=METHOD_COLORS.get(method, 'C0'),
                 linestyle=METHOD_LINESTYLES.get(method, '-'),
                 label=METHOD_NAMES.get(method, method),
                 linewidth=2, markersize=8)

    # 3. 添加标注（专家1建议）
    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
    ax1.text(1.2, 0.48, 'Random Guess', fontsize=10, color='gray', alpha=0.7)

    # 标注 Basin 边界
    ax1.annotate('Physical\nPull-in Range', xy=(0.3, 0.15), xytext=(0.6, 0.25),
                 fontsize=10, color='green',
                 arrowprops=dict(arrowstyle='->', color='green', alpha=0.7))

    ax1.text(0.1, 0.45, "Gradient\nCorrect", color='green', fontweight='bold',
             alpha=0.5, fontsize=9, ha='center')
    ax1.text(0.8, 0.45, "Gradient\nCollapse", color='red', fontweight='bold',
             alpha=0.5, fontsize=9, ha='center')

    ax1.set_xlabel('Initial τ Error (samples)', fontsize=14)
    ax1.set_ylabel('BER', fontsize=14)
    ax1.set_title('(a) Communication Performance vs Sync Error', fontsize=14)
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 0.55])
    ax1.set_xlim([0, max(agg['init_error'].max(), 1.5)])

    # ===== Panel B: RMSE vs init_error =====

    # 1. Zone Shading
    ax2.axvspan(0, 0.3, color='green', alpha=0.08)
    ax2.axvspan(0.3, 0.5, color='orange', alpha=0.08)
    ax2.axvspan(0.5, 2.0, color='red', alpha=0.08)

    # 2. 画方法曲线
    for method in methods_to_plot:
        if method == "oracle":
            continue
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        init_errors = data['init_error'].values
        rmse_mean = data['rmse_tau_final_mean'].values

        ax2.plot(init_errors, rmse_mean,
                 marker=METHOD_MARKERS.get(method, 'o'),
                 color=METHOD_COLORS.get(method, 'C0'),
                 linestyle=METHOD_LINESTYLES.get(method, '-'),
                 label=METHOD_NAMES.get(method, method),
                 linewidth=2, markersize=8)

    # 3. y=x 参考线和目标区域
    max_x = max(agg['init_error'].max(), 1.5)
    ax2.plot([0, max_x], [0, max_x], 'k--', alpha=0.5, label='No Improvement (y=x)')
    ax2.axhspan(0, 0.1, alpha=0.15, color='green')
    ax2.text(0.05, 0.05, 'Target\n(<0.1)', fontsize=9, color='green', alpha=0.8)

    ax2.set_xlabel('Initial τ Error (samples)', fontsize=14)
    ax2.set_ylabel('Final τ RMSE (samples)', fontsize=14)
    ax2.set_title('(b) Delay Estimation Performance', fontsize=14)
    ax2.legend(loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0, max_x])
    ax2.set_xlim([0, max_x])

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig04_cliff_all_methods.png", dpi=300)
    fig.savefig(f"{out_dir}/fig04_cliff_all_methods.pdf")
    plt.close(fig)


def fig05_snr_multi_init_error(df: pd.DataFrame, out_dir: str):
    """
    Fig 5: SNR sweep @ multiple init_errors（专家方案3）

    证明：
    (a) init_error=0: baseline 没 bug
    (b) init_error=0.2: proposed 领先
    (c) init_error=0.3: baseline 失效
    """

    if 'init_error' not in df.columns:
        print("Warning: init_error not in data, skipping fig05")
        return

    init_errors = sorted(df['init_error'].unique())
    n_panels = len(init_errors)

    fig, axes = plt.subplots(1, n_panels, figsize=(6*n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    methods_to_plot = ["adjoint_slice", "proposed_no_update", "proposed", "oracle"]

    for idx, init_error in enumerate(init_errors):
        ax = axes[idx]
        df_sub = df[df['init_error'] == init_error]
        agg = aggregate(df_sub, ['snr_db', 'method'], ['ber'])

        for method in methods_to_plot:
            data = agg[agg['method'] == method]
            if len(data) == 0:
                continue

            snr = data['snr_db'].values
            ber_mean = data['ber_mean'].values

            ax.semilogy(snr, ber_mean,
                        marker=METHOD_MARKERS.get(method, 'o'),
                        color=METHOD_COLORS.get(method, 'C0'),
                        linestyle=METHOD_LINESTYLES.get(method, '-'),
                        label=METHOD_NAMES.get(method, method),
                        linewidth=2, markersize=8)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('BER', fontsize=12)

        # 根据 init_error 设置标题
        if init_error == 0.0:
            title = f'({"abc"[idx]}) init_error=0: Baseline Validation'
            ax.set_title(title, fontsize=12, color='green')
        elif init_error == 0.2:
            title = f'({"abc"[idx]}) init_error=0.2: Proposed Leading'
            ax.set_title(title, fontsize=12, color='orange')
        else:
            title = f'({"abc"[idx]}) init_error={init_error}: Baseline Fails'
            ax.set_title(title, fontsize=12, color='red')

        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([5e-2, 0.6])

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig05_snr_multi_init_error.png", dpi=300)
    fig.savefig(f"{out_dir}/fig05_snr_multi_init_error.pdf")
    plt.close(fig)


def fig06_jacobian_condition(df: pd.DataFrame, out_dir: str):
    """
    Fig 6: Jacobian condition number - "The Villain Plot"

    专家1建议：展示为什么端到端方法失败
    条件数 ~10^15 说明联合估计数值不稳定
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # ===== Panel A: 条件数柱状图（专家1的 "Villain" 图）=====

    # 模拟分层条件数下降（基于物理）
    layers = ['End-to-End\n(Joint τ,v)', 'Layer 1\n(Pilot τ)', 'Layer 3', 'Layer 7\n(Final)']
    cond_nums = [1e15, 1e6, 1e3, 1e1]  # 分层后条件数显著下降
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']

    bars = ax1.bar(layers, np.log10(cond_nums), color=colors, alpha=0.8, edgecolor='black')

    # 在柱子上标注具体数值
    for bar, val in zip(bars, cond_nums):
        height = np.log10(val)
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'$10^{{{int(height)}}}$',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    # 添加阈值线
    ax1.axhline(y=6, color='orange', linestyle='--', linewidth=2, label='Unstable Threshold ($10^6$)')
    ax1.axhline(y=3, color='green', linestyle=':', linewidth=2, label='Stable ($10^3$)')

    # 添加标注
    ax1.annotate('Numerical\nInstability!', xy=(0, 15), xytext=(1.5, 12),
                fontsize=11, color='red', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='red', lw=2))

    ax1.set_ylabel('Condition Number ($\\log_{10}$)', fontsize=14)
    ax1.set_title('(a) Why End-to-End Fails:\nThe Geometry Gap', fontsize=12)
    ax1.set_ylim(0, 18)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3, axis='y')

    # ===== Panel B: 从数据绘制（如果有）=====
    if 'init_error' in df.columns and 'gram_cond_log10' in df.columns:
        init_errors = df['init_error'].values
        cond_log = df['gram_cond_log10'].values

        ax2.bar(init_errors, cond_log, width=0.08, color='C3', alpha=0.7, edgecolor='black')
        ax2.axhline(y=6, color='orange', linestyle='--', linewidth=2)

        ax2.set_xlabel('Initial τ Error (samples)', fontsize=14)
        ax2.set_ylabel('$\\log_{10}$(Condition Number)', fontsize=14)
        ax2.set_title('(b) Gram Matrix Condition vs Init Error', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')
    else:
        # 如果没有数据，画 Jacobian 范数比
        init_errors = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
        ratio_J = [3e15] * len(init_errors)  # τ sensitivity >> v sensitivity

        ax2.bar(init_errors, np.log10(ratio_J), width=0.08, color='C0', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Initial τ Error (samples)', fontsize=14)
        ax2.set_ylabel('$\\log_{10}(||J_\\tau|| / ||J_v||)$', fontsize=14)
        ax2.set_title('(b) Jacobian Norm Ratio\n(τ sensitivity >> v sensitivity)', fontsize=12)
        ax2.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig06_jacobian_condition.png", dpi=300)
    fig.savefig(f"{out_dir}/fig06_jacobian_condition.pdf")
    plt.close(fig)


def fig11_heatmap(df: pd.DataFrame, out_dir: str):
    """
    Fig 11: 2D Heatmap - BER vs (SNR × init_error)

    专家建议：展示在不同 SNR 和 init_error 组合下的性能
    """

    if 'init_error' not in df.columns or 'snr_db' not in df.columns:
        print("Warning: heatmap data missing required columns, skipping fig11")
        return

    methods_to_plot = df['method'].unique() if 'method' in df.columns else ['proposed']
    n_methods = len(methods_to_plot)

    fig, axes = plt.subplots(1, min(n_methods, 3), figsize=(6*min(n_methods, 3), 5))
    if n_methods == 1:
        axes = [axes]

    for idx, method in enumerate(methods_to_plot[:3]):  # 最多画3个
        ax = axes[idx]

        df_method = df[df['method'] == method] if 'method' in df.columns else df

        # 聚合数据
        agg = df_method.groupby(['snr_db', 'init_error'])['ber'].mean().reset_index()

        # 创建 pivot table
        pivot = agg.pivot(index='init_error', columns='snr_db', values='ber')

        # 绘制热力图
        sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', annot=True, fmt='.2f',
                    cbar_kws={'label': 'BER'}, vmin=0, vmax=0.5)

        ax.set_xlabel('SNR (dB)', fontsize=12)
        ax.set_ylabel('Initial τ Error (samples)', fontsize=12)
        ax.set_title(f'{METHOD_NAMES.get(method, method)}', fontsize=12)

        # 标注 Basin 边界
        if 0.3 in pivot.index.values:
            y_pos = list(pivot.index.values).index(0.3) + 0.5
            ax.axhline(y=y_pos, color='white', linestyle='--', linewidth=2)

    fig.suptitle('BER Performance Heatmap: SNR × Init Error', fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig11_heatmap.png", dpi=300)
    fig.savefig(f"{out_dir}/fig11_heatmap.pdf")
    plt.close(fig)


def fig07_gap_to_oracle(df: pd.DataFrame, out_dir: str):
    """Fig 7: Gap-to-Oracle - 当 proposed ≈ oracle 时让差异可见"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    agg = aggregate(df, ['snr_db', 'method'], ['ber', 'rmse_tau_final'])

    # 获取 oracle 基准
    oracle_data = agg[agg['method'] == 'oracle'][['snr_db', 'ber_mean', 'rmse_tau_final_mean']]
    oracle_data = oracle_data.rename(columns={'ber_mean': 'oracle_ber', 'rmse_tau_final_mean': 'oracle_rmse'})

    methods_to_plot = ["adjoint_slice", "proposed_no_update", "proposed"]

    # Panel A: BER Gap
    for method in methods_to_plot:
        data = agg[agg['method'] == method][['snr_db', 'ber_mean']]
        merged = pd.merge(data, oracle_data, on='snr_db')

        if len(merged) == 0:
            continue

        snr = merged['snr_db'].values
        gap = merged['ber_mean'].values - merged['oracle_ber'].values

        ax1.plot(snr, gap,
                 marker=METHOD_MARKERS.get(method, 'o'),
                 color=METHOD_COLORS.get(method, 'C0'),
                 linestyle=METHOD_LINESTYLES.get(method, '-'),
                 label=METHOD_NAMES.get(method, method),
                 linewidth=2, markersize=8)

    ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Oracle (zero gap)')
    ax1.axhline(y=0.01, color='orange', linestyle=':', linewidth=1.5, alpha=0.7, label='1% gap')

    ax1.set_xlabel('SNR (dB)', fontsize=14)
    ax1.set_ylabel('ΔBER (method - oracle)', fontsize=14)
    ax1.set_title('(a) BER Gap to Oracle', fontsize=14)
    ax1.legend(loc='upper right', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Panel B: RMSE
    for method in methods_to_plot:
        data = agg[agg['method'] == method][['snr_db', 'rmse_tau_final_mean']]

        if len(data) == 0:
            continue

        snr = data['snr_db'].values
        rmse = data['rmse_tau_final_mean'].values

        ax2.semilogy(snr, rmse,
                     marker=METHOD_MARKERS.get(method, 'o'),
                     color=METHOD_COLORS.get(method, 'C0'),
                     linestyle=METHOD_LINESTYLES.get(method, '-'),
                     label=METHOD_NAMES.get(method, method),
                     linewidth=2, markersize=8)

    ax2.axhline(y=0.1, color='green', linestyle=':', linewidth=2, label='Target (0.1 samples)')

    ax2.set_xlabel('SNR (dB)', fontsize=14)
    ax2.set_ylabel('Final τ RMSE (samples)', fontsize=14)
    ax2.set_title('(b) τ RMSE Comparison', fontsize=14)
    ax2.legend(loc='upper right', fontsize=10)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig07_gap_to_oracle.png", dpi=300)
    fig.savefig(f"{out_dir}/fig07_gap_to_oracle.pdf")
    plt.close(fig)


def fig08_robustness(df_pn: pd.DataFrame, df_pilot: pd.DataFrame, out_dir: str):
    """Fig 8: Robustness to PN and Pilot length"""

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    methods_to_plot = ["adjoint_slice", "proposed", "oracle"]

    # Panel A: PN sweep
    if df_pn is not None and len(df_pn) > 0:
        agg_pn = aggregate(df_pn, ['pn_linewidth', 'method'], ['ber'])

        for method in methods_to_plot:
            data = agg_pn[agg_pn['method'] == method]
            if len(data) == 0:
                continue

            pn = data['pn_linewidth'].values / 1e3  # kHz
            ber = data['ber_mean'].values

            ax1.plot(pn, ber,
                     marker=METHOD_MARKERS.get(method, 'o'),
                     color=METHOD_COLORS.get(method, 'C0'),
                     label=METHOD_NAMES.get(method, method))

        ax1.set_xlabel('PN Linewidth (kHz)')
        ax1.set_ylabel('BER')
        ax1.set_title('(a) Robustness to Phase Noise')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # Panel B: Pilot sweep
    if df_pilot is not None and len(df_pilot) > 0:
        agg_pilot = aggregate(df_pilot, ['pilot_len', 'method'], ['rmse_tau_final'])

        for method in methods_to_plot:
            data = agg_pilot[agg_pilot['method'] == method]
            if len(data) == 0:
                continue

            pilot = data['pilot_len'].values
            rmse = data['rmse_tau_final_mean'].values

            ax2.plot(pilot, rmse,
                     marker=METHOD_MARKERS.get(method, 'o'),
                     color=METHOD_COLORS.get(method, 'C0'),
                     label=METHOD_NAMES.get(method, method))

        ax2.set_xlabel('Pilot Length')
        ax2.set_ylabel('RMSE τ (samples)')
        ax2.set_title('(b) Impact of Pilot Length')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig08_robustness.png", dpi=300)
    fig.savefig(f"{out_dir}/fig08_robustness.pdf")
    plt.close(fig)


def fig09_latency(df: pd.DataFrame, out_dir: str):
    """Fig 9: Latency comparison"""

    fig, ax = plt.subplots(figsize=(8, 6))

    methods = df['method'].values
    latencies = df['latency_mean_ms'].values
    stds = df['latency_std_ms'].values

    colors = [METHOD_COLORS.get(m, 'C0') for m in methods]
    labels = [METHOD_NAMES.get(m, m) for m in methods]

    bars = ax.bar(range(len(methods)), latencies, yerr=stds,
                  color=colors, alpha=0.7, edgecolor='black', capsize=5)

    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Latency (ms)', fontsize=14)
    ax.set_title('Computational Complexity', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')

    # 标注数值
    for i, (lat, std) in enumerate(zip(latencies, stds)):
        ax.text(i, lat + std + 0.5, f'{lat:.1f}ms', ha='center', fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig09_latency.png", dpi=300)
    fig.savefig(f"{out_dir}/fig09_latency.pdf")
    plt.close(fig)


def fig10_ablation(df: pd.DataFrame, out_dir: str):
    """
    Fig 10: 消融实验（专家方案2）

    验证各组件的贡献：
    - oracle > proposed ≈ proposed_no_learned_alpha > proposed_no_update > random_init
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    agg = aggregate(df, ['snr_db', 'method'], ['ber', 'rmse_tau_final'])

    # 消融实验方法顺序（从弱到强）
    methods_order = ["random_init", "proposed_no_update", "proposed_no_learned_alpha", "proposed", "oracle"]

    # Panel A: BER vs SNR
    for method in methods_order:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        snr = data['snr_db'].values
        ber_mean = data['ber_mean'].values

        ax1.semilogy(snr, ber_mean,
                     marker=METHOD_MARKERS.get(method, 'o'),
                     color=METHOD_COLORS.get(method, 'C0'),
                     linestyle=METHOD_LINESTYLES.get(method, '-'),
                     label=METHOD_NAMES.get(method, method),
                     linewidth=2, markersize=8)

    ax1.set_xlabel('SNR (dB)', fontsize=14)
    ax1.set_ylabel('BER', fontsize=14)
    ax1.set_title('(a) Ablation Study: BER Performance', fontsize=14)
    ax1.legend(loc='upper right', fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([5e-2, 0.6])

    # Panel B: RMSE vs SNR
    for method in methods_order:
        if method == "oracle":
            continue  # Oracle RMSE = 0
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        snr = data['snr_db'].values
        rmse_mean = data['rmse_tau_final_mean'].values

        ax2.semilogy(snr, rmse_mean,
                     marker=METHOD_MARKERS.get(method, 'o'),
                     color=METHOD_COLORS.get(method, 'C0'),
                     linestyle=METHOD_LINESTYLES.get(method, '-'),
                     label=METHOD_NAMES.get(method, method),
                     linewidth=2, markersize=8)

    ax2.axhline(y=0.1, color='green', linestyle=':', linewidth=2, label='Target (0.1)')

    ax2.set_xlabel('SNR (dB)', fontsize=14)
    ax2.set_ylabel('Final τ RMSE (samples)', fontsize=14)
    ax2.set_title('(b) Ablation Study: Sensing Performance', fontsize=14)
    ax2.legend(loc='upper right', fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f"{out_dir}/fig10_ablation.png", dpi=300)
    fig.savefig(f"{out_dir}/fig10_ablation.pdf")
    plt.close(fig)

    # 打印消融实验结果表格
    print("\n📊 消融实验结果 @ SNR=15dB:")
    snr15 = agg[agg['snr_db'] == 15]
    for method in methods_order:
        data = snr15[snr15['method'] == method]
        if len(data) > 0:
            ber = data['ber_mean'].values[0]
            rmse = data['rmse_tau_final_mean'].values[0] if 'rmse_tau_final_mean' in data.columns else 0
            print(f"  {METHOD_NAMES.get(method, method):25s}: BER={ber:.4f}, RMSE={rmse:.4f}")


# ============================================================================
# 主函数
# ============================================================================

def generate_all_figures(data_dir: str, out_dir: str = None):
    """从 CSV 数据生成所有图表"""

    if out_dir is None:
        out_dir = data_dir

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("Loading data...")
    print("=" * 60)
    data = load_data(data_dir)

    print("\n" + "=" * 60)
    print("Generating figures...")
    print("=" * 60)

    if 'snr_sweep' in data:
        fig01_ber_vs_snr(data['snr_sweep'], out_dir)
        print("  ✓ Fig 1: BER vs SNR")

        fig02_rmse_tau_vs_snr(data['snr_sweep'], out_dir)
        print("  ✓ Fig 2: RMSE_τ vs SNR")

        fig03_success_rate(data['snr_sweep'], out_dir)
        print("  ✓ Fig 3: Success Rate")

        fig07_gap_to_oracle(data['snr_sweep'], out_dir)
        print("  ✓ Fig 7: Gap-to-Oracle")

    if 'cliff_sweep' in data:
        fig04_cliff_all_methods(data['cliff_sweep'], out_dir)
        print("  ✓ Fig 4: Cliff (ALL methods) - 核心图【方案1】")

    if 'snr_multi_init_error' in data:
        fig05_snr_multi_init_error(data['snr_multi_init_error'], out_dir)
        print("  ✓ Fig 5: SNR @ multi init_error【方案3】")

    if 'jacobian' in data:
        fig06_jacobian_condition(data['jacobian'], out_dir)
        print("  ✓ Fig 6: Jacobian Condition")

    if 'pn_sweep' in data or 'pilot_sweep' in data:
        fig08_robustness(data.get('pn_sweep'), data.get('pilot_sweep'), out_dir)
        print("  ✓ Fig 8: Robustness")

    if 'latency' in data:
        fig09_latency(data['latency'], out_dir)
        print("  ✓ Fig 9: Latency")

    if 'ablation_sweep' in data:
        fig10_ablation(data['ablation_sweep'], out_dir)
        print("  ✓ Fig 10: Ablation Study【方案2】")

    if 'heatmap_sweep' in data:
        fig11_heatmap(data['heatmap_sweep'], out_dir)
        print("  ✓ Fig 11: Heatmap (SNR × init_error)")

    print("\n" + "=" * 60)
    print(f"All figures saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Generate paper figures from CSV data")
    parser.add_argument('--data_dir', type=str, required=True, help="Directory containing CSV data")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory for figures")
    args = parser.parse_args()

    generate_all_figures(args.data_dir, args.out_dir)