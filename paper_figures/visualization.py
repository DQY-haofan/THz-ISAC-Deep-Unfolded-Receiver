"""
visualization.py - Paper-grade Visualization (Expert Review)

Features:
1. ADC bits sweep figure
2. CRLB overlay on τ RMSE plots
3. Updated matched filter label: "41-point τ-search"
4. Oracle-B support
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11, 'axes.labelsize': 12,
    'axes.titlesize': 12, 'legend.fontsize': 9, 'xtick.labelsize': 10,
    'ytick.labelsize': 10, 'figure.figsize': (8, 5), 'savefig.dpi': 300,
    'savefig.bbox': 'tight', 'lines.linewidth': 2, 'lines.markersize': 7,
    'axes.grid': True, 'grid.alpha': 0.3, 'axes.linewidth': 1.2,
})


METHOD_CONFIG = {
    "naive_slice": {"label": "Naive Slice", "color": "#7f7f7f", "marker": "v", "ls": ":", "lw": 1.5, "alpha": 0.6},
    "matched_filter": {"label": "Matched Filter (41-point τ-search)", "color": "#d62728", "marker": "X", "ls": "--", "lw": 1.8, "alpha": 0.5},
    "adjoint_lmmse": {"label": "Adjoint+LMMSE", "color": "#ff7f0e", "marker": "P", "ls": "-.", "lw": 2.0, "alpha": 0.8},
    "adjoint_slice": {"label": "Adjoint+Slice", "color": "#9467bd", "marker": "d", "ls": ":", "lw": 2.0, "alpha": 0.8},
    "proposed_no_update": {"label": "w/o τ update", "color": "#8c564b", "marker": "s", "ls": "-", "lw": 2.0, "alpha": 0.9},
    "proposed_tau_slice": {"label": "Proposed (τ)+Slice", "color": "#17becf", "marker": "p", "ls": "-.", "lw": 2.0, "alpha": 0.9},
    "proposed": {"label": "Proposed (GA-BV-Net)", "color": "#1f77b4", "marker": "o", "ls": "-", "lw": 2.8, "alpha": 1.0},
    "oracle_sync": {"label": "Oracle-A (Genie θ)", "color": "#2ca02c", "marker": "^", "ls": "-", "lw": 2.5, "alpha": 1.0},
    "oracle_local_best": {"label": "Oracle-B (Local Best)", "color": "#bcbd22", "marker": "v", "ls": "--", "lw": 2.0, "alpha": 0.8},
    "oracle": {"label": "Oracle θ", "color": "#2ca02c", "marker": "^", "ls": "-", "lw": 2.5, "alpha": 1.0},
    "random_init": {"label": "Random Init", "color": "black", "marker": "*", "ls": ":", "lw": 1.5, "alpha": 0.5},
}


def get_style(method: str) -> dict:
    return METHOD_CONFIG.get(method, {"label": method, "color": "black", "marker": "o", "ls": "-", "lw": 2.0, "alpha": 1.0})


def aggregate(df: pd.DataFrame, group_cols: List[str], value_cols: List[str]) -> pd.DataFrame:
    agg_funcs = {col: ['mean', 'std', 'count'] for col in value_cols}
    agg = df.groupby(group_cols).agg(agg_funcs).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]
    for col in value_cols:
        mean, std, count = f'{col}_mean', f'{col}_std', f'{col}_count'
        if all(c in agg.columns for c in [mean, std, count]):
            agg[f'{col}_ci95'] = 1.96 * agg[std] / np.sqrt(agg[count].clip(lower=1))
    return agg


def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    data = {}
    csv_files = ['data_snr_sweep.csv', 'data_cliff_sweep.csv', 'data_snr_multi_init_error.csv',
                 'data_ablation_sweep.csv', 'data_heatmap_sweep.csv', 'data_pn_sweep.csv',
                 'data_pilot_sweep.csv', 'data_jacobian.csv', 'data_latency.csv',
                 'data_adc_bits_sweep.csv', 'data_crlb.csv']

    for f in csv_files:
        path = os.path.join(data_dir, f)
        if os.path.exists(path):
            key = f.replace('data_', '').replace('.csv', '')
            data[key] = pd.read_csv(path)
            n_methods = len(data[key]['method'].unique()) if 'method' in data[key].columns else 0
            print(f"  ✓ {f}: {len(data[key])} rows, {n_methods} methods")
        else:
            print(f"  ⚠️ Not found: {f}")
    return data


def check_methods(df: pd.DataFrame, expected: List[str], fig_name: str) -> List[str]:
    if 'method' not in df.columns:
        return []
    actual = set(df['method'].unique())
    missing = set(expected) - actual
    if missing:
        print(f"  ⚠️ {fig_name}: Missing methods {missing}")
    return [m for m in expected if m in actual]


def fig01_ber_vs_snr(df: pd.DataFrame, out_dir: str, df_crlb: pd.DataFrame = None):
    fig, ax = plt.subplots(figsize=(10, 7))
    agg = aggregate(df, ['snr_db', 'method'], ['ber'])

    expected = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]
    methods = check_methods(df, expected, "Fig 01")

    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue
        s = get_style(method)
        ax.semilogy(data['snr_db'], data['ber_mean'], marker=s['marker'], color=s['color'],
                    linestyle=s['ls'], label=s['label'], linewidth=s['lw'], alpha=s['alpha'], markersize=7)

    ax.axvspan(18, 28, color='gray', alpha=0.1)
    ax.text(22, 0.35, "Hardware Wall\n($\\Gamma_{\\mathrm{eff}}$ Limited)", fontsize=10, color='gray', ha='center', style='italic')
    ax.axhline(y=0.15, color='gray', linestyle=':', alpha=0.6, linewidth=1.5)
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
    print("  ✓ Fig 01: BER vs SNR")


def fig02_rmse_tau_vs_snr(df: pd.DataFrame, out_dir: str, df_crlb: pd.DataFrame = None):
    fig, ax = plt.subplots(figsize=(10, 6))
    agg = aggregate(df, ['snr_db', 'method'], ['rmse_tau_final'])

    expected = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed"]
    methods = check_methods(df, expected, "Fig 02")

    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue
        s = get_style(method)
        ax.semilogy(data['snr_db'], data['rmse_tau_final_mean'], marker=s['marker'], color=s['color'],
                   linestyle=s['ls'], label=s['label'], linewidth=s['lw'], alpha=s['alpha'])

    if df_crlb is not None and 'crlb_tau_samples' in df_crlb.columns:
        ax.semilogy(df_crlb['snr_db'], df_crlb['crlb_tau_samples'], 'k--', linewidth=2.5, label='CRLB (Bussgang)', alpha=0.7)

    ax.axhline(y=0.1, color='green', linestyle=':', linewidth=2, label='Target (0.1 samples)')
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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    agg = aggregate(df, ['init_error', 'method'], ['ber', 'rmse_tau_final', 'success_rate'])

    expected = ["naive_slice", "adjoint_lmmse", "adjoint_slice", "matched_filter",
                "proposed_no_update", "proposed", "oracle_sync"]
    methods = check_methods(df, expected, "Fig 04")

    proposed_data = agg[agg['method'] == 'proposed']
    basin_boundary = 0.3
    if len(proposed_data) > 0 and 'success_rate_mean' in proposed_data.columns:
        success_vals = proposed_data[proposed_data['success_rate_mean'] >= 0.5]
        if len(success_vals) > 0:
            basin_boundary = success_vals['init_error'].max()

    max_x = max(agg['init_error'].max(), 1.5)
    for ax in [ax1, ax2]:
        ax.axvspan(0, basin_boundary, color='green', alpha=0.06)
        ax.axvspan(basin_boundary, basin_boundary + 0.2, color='orange', alpha=0.06)
        ax.axvspan(basin_boundary + 0.2, max_x, color='red', alpha=0.06)
        ax.axvline(basin_boundary, color='green', linestyle='--', linewidth=1.5, alpha=0.7)

    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue
        s = get_style(method)
        ax1.plot(data['init_error'], data['ber_mean'], marker=s['marker'], color=s['color'],
                linestyle=s['ls'], linewidth=s['lw'], alpha=s['alpha'], label=s['label'], markersize=6)

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.set_xlabel('Initial $\\tau$ Error (samples)')
    ax1.set_ylabel('BER')
    ax1.set_title('(a) The Cliff: Detection Robustness')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_ylim([0, 0.55])
    ax1.set_xlim([0, max_x])

    for method in methods:
        if method in ["oracle_sync", "oracle"]:
            continue
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue
        s = get_style(method)
        ax2.plot(data['init_error'], data['rmse_tau_final_mean'], marker=s['marker'], color=s['color'],
                linestyle=s['ls'], linewidth=s['lw'], alpha=s['alpha'], label=s['label'], markersize=6)

    ax2.plot([0, max_x], [0, max_x], 'k--', alpha=0.4, linewidth=1, label='No Improvement')
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
    print("  ✓ Fig 04: Cliff Plot")


def fig05_snr_multi_init_error(df: pd.DataFrame, out_dir: str):
    if 'init_error' not in df.columns:
        return

    init_errors = sorted(df['init_error'].unique())
    n_panels = min(len(init_errors), 3)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.5*n_panels, 5))
    if n_panels == 1:
        axes = [axes]

    expected = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]
    methods = check_methods(df, expected, "Fig 05")

    for idx, init_error in enumerate(init_errors[:n_panels]):
        ax = axes[idx]
        df_sub = df[df['init_error'] == init_error]
        agg = aggregate(df_sub, ['snr_db', 'method'], ['ber'])

        for method in methods:
            data = agg[agg['method'] == method]
            if len(data) == 0:
                continue
            s = get_style(method)
            ax.semilogy(data['snr_db'], data['ber_mean'], marker=s['marker'], color=s['color'],
                       linestyle=s['ls'], label=s['label'], linewidth=s['lw'], alpha=s['alpha'])

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')
        ax.set_title(f'init_err={init_error}')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([5e-2, 0.55])

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig05_snr_multi_init_error.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 05: SNR @ Multi Init Error")


def fig07_gap_to_oracle(df: pd.DataFrame, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    agg = aggregate(df, ['snr_db', 'method'], ['ber', 'rmse_tau_final'])

    oracle_method = 'oracle_sync' if 'oracle_sync' in df['method'].unique() else 'oracle'
    oracle_data = agg[agg['method'] == oracle_method][['snr_db', 'ber_mean', 'rmse_tau_final_mean']]
    oracle_data = oracle_data.rename(columns={'ber_mean': 'oracle_ber', 'rmse_tau_final_mean': 'oracle_rmse'})

    expected = ["adjoint_slice", "proposed_no_update", "proposed"]
    methods = check_methods(df, expected, "Fig 07")

    for method in methods:
        data = agg[agg['method'] == method][['snr_db', 'ber_mean']]
        merged = pd.merge(data, oracle_data, on='snr_db')
        if len(merged) == 0:
            continue
        s = get_style(method)
        gap = merged['ber_mean'].values - merged['oracle_ber'].values
        ax1.plot(merged['snr_db'], gap, marker=s['marker'], color=s['color'], linestyle=s['ls'],
                label=s['label'], linewidth=s['lw'])

    ax1.axhline(y=0, color='green', linestyle='--', linewidth=2, label='Oracle')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('ΔBER (method − oracle)')
    ax1.set_title('(a) BER Gap to Oracle')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue
        s = get_style(method)
        ax2.semilogy(data['snr_db'], data['rmse_tau_final_mean'], marker=s['marker'], color=s['color'],
                    linestyle=s['ls'], label=s['label'], linewidth=s['lw'])

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
    fig, ax1 = plt.subplots(figsize=(11, 6))

    target_snr = df['snr_db'].max()
    data = df[df['snr_db'] == target_snr].copy()

    methods = ["random_init", "proposed_no_update", "proposed_tau_slice", "proposed", "oracle_sync"]
    available_methods = [m for m in methods if m in data['method'].unique()]
    if len(available_methods) == 0:
        return

    agg = aggregate(data, ['method'], ['ber', 'rmse_tau_final'])
    agg = agg.set_index('method').reindex(available_methods).reset_index()

    x = np.arange(len(available_methods))
    colors = [get_style(m)['color'] for m in available_methods]
    ax1.bar(x, agg['ber_mean'], 0.4, color=colors, alpha=0.7, edgecolor='black', linewidth=1.2, label='BER')
    ax1.set_ylabel('BER (Lower is Better)', fontsize=12)
    ax1.set_ylim(0, 0.5)

    ax2 = ax1.twinx()
    rmse = agg['rmse_tau_final_mean'].fillna(0).values
    ax2.plot(x, rmse, color='#d62728', marker='D', markersize=10, linewidth=2.5, linestyle='--', label='τ RMSE')
    ax2.set_ylabel('τ RMSE (samples, Log Scale)', color='#d62728', fontsize=12)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(1e-2, 5)

    labels = [get_style(m)['label'] for m in available_methods]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax1.set_title(f'Ablation Study @ SNR={target_snr:.0f}dB', fontsize=13)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig10_ablation_dual.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 10: Ablation")


def fig12_adc_bits_sweep(df: pd.DataFrame, out_dir: str):
    if 'adc_bits' not in df.columns:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    agg = aggregate(df, ['adc_bits', 'method'], ['ber', 'rmse_tau_final'])

    expected = ["naive_slice", "adjoint_slice", "proposed", "oracle_sync"]
    methods = check_methods(df, expected, "Fig 12")

    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue
        s = get_style(method)
        ax1.plot(data['adc_bits'], data['ber_mean'], marker=s['marker'], color=s['color'],
                linestyle=s['ls'], label=s['label'], linewidth=s['lw'], alpha=s['alpha'], markersize=8)

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.axvspan(0.8, 1.2, color='red', alpha=0.1)
    ax1.set_xlabel('ADC Resolution (bits)')
    ax1.set_ylabel('BER')
    ax1.set_title('(a) BER vs ADC Bits: 1-bit Cliff')
    ax1.legend(loc='upper right', fontsize=9)
    ax1.set_xlim([0.5, 8.5])
    ax1.set_ylim([0, 0.55])
    ax1.grid(True, alpha=0.3)

    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue
        s = get_style(method)
        ax2.semilogy(data['adc_bits'], data['rmse_tau_final_mean'], marker=s['marker'], color=s['color'],
                    linestyle=s['ls'], label=s['label'], linewidth=s['lw'], alpha=s['alpha'], markersize=8)

    ax2.axhline(y=0.1, color='green', linestyle=':', linewidth=2, label='Target')
    ax2.set_xlabel('ADC Resolution (bits)')
    ax2.set_ylabel('τ RMSE (samples)')
    ax2.set_title('(b) τ RMSE vs ADC Bits')
    ax2.legend(loc='upper right', fontsize=9)
    ax2.set_xlim([0.5, 8.5])
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig12_adc_bits_sweep.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 12: ADC Bits Sweep")


def fig08_robustness(df_pn: pd.DataFrame, df_pilot: pd.DataFrame, out_dir: str):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    expected = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]

    if df_pn is not None and len(df_pn) > 0:
        methods = check_methods(df_pn, expected, "Fig 08 (PN)")
        agg_pn = aggregate(df_pn, ['pn_linewidth', 'method'], ['ber'])
        for method in methods:
            data = agg_pn[agg_pn['method'] == method]
            if len(data) == 0:
                continue
            s = get_style(method)
            pn = data['pn_linewidth'].values / 1e3
            ax1.plot(pn, data['ber_mean'], marker=s['marker'], color=s['color'], label=s['label'], linewidth=s['lw'])
        ax1.set_xlabel('PN Linewidth (kHz)')
        ax1.set_ylabel('BER')
        ax1.set_title('(a) Robustness to Phase Noise')
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.3)

    if df_pilot is not None and len(df_pilot) > 0:
        methods = check_methods(df_pilot, expected, "Fig 08 (Pilot)")
        agg_pilot = aggregate(df_pilot, ['pilot_len', 'method'], ['rmse_tau_final'])
        for method in methods:
            data = agg_pilot[agg_pilot['method'] == method]
            if len(data) == 0:
                continue
            s = get_style(method)
            ax2.plot(data['pilot_len'], data['rmse_tau_final_mean'], marker=s['marker'], color=s['color'],
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
    if len(df) == 0:
        return
    fig, ax = plt.subplots(figsize=(9, 6))
    methods = df['method'].values
    latencies = df['latency_mean_ms'].values
    stds = df['latency_std_ms'].values
    colors = [get_style(m)['color'] for m in methods]
    labels = [get_style(m)['label'] for m in methods]

    ax.bar(range(len(methods)), latencies, yerr=stds, color=colors, alpha=0.7, edgecolor='black', capsize=5)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(labels, rotation=15, ha='right')
    ax.set_ylabel('Latency (ms)')
    ax.set_title('Computational Complexity')
    ax.grid(True, alpha=0.3, axis='y')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig09_latency.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 09: Latency")


def fig11_heatmap(df: pd.DataFrame, out_dir: str):
    if 'init_error' not in df.columns or 'snr_db' not in df.columns:
        return
    df_proposed = df[df['method'] == 'proposed'] if 'method' in df.columns else df
    if len(df_proposed) == 0:
        return

    agg = df_proposed.groupby(['snr_db', 'init_error'])['ber'].mean().reset_index()
    pivot = agg.pivot(index='init_error', columns='snr_db', values='ber')

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(pivot, ax=ax, cmap='RdYlGn_r', annot=True, fmt='.2f',
               cbar_kws={'label': 'BER'}, vmin=0, vmax=0.5, linewidths=0.5, linecolor='white')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Initial τ Error (samples)')
    ax.set_title('Proposed Method: BER Heatmap')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig11_heatmap.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 11: Heatmap")


def generate_all_figures(data_dir: str, out_dir: str = None):
    if out_dir is None:
        out_dir = data_dir
    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("📊 Paper Figure Generation")
    print("=" * 60)
    print("\nLoading data...")
    data = load_data(data_dir)

    print("\n" + "-" * 40)
    print("Generating figures...")
    print("-" * 40)

    df_crlb = data.get('crlb', None)

    if 'snr_sweep' in data:
        fig01_ber_vs_snr(data['snr_sweep'], out_dir, df_crlb)
        fig02_rmse_tau_vs_snr(data['snr_sweep'], out_dir, df_crlb)
        fig07_gap_to_oracle(data['snr_sweep'], out_dir)

    if 'cliff_sweep' in data:
        fig04_cliff_all_methods(data['cliff_sweep'], out_dir)

    if 'snr_multi_init_error' in data:
        fig05_snr_multi_init_error(data['snr_multi_init_error'], out_dir)

    if 'ablation_sweep' in data:
        fig10_ablation_dual_axis(data['ablation_sweep'], out_dir)

    if 'adc_bits_sweep' in data:
        fig12_adc_bits_sweep(data['adc_bits_sweep'], out_dir)

    if 'pn_sweep' in data or 'pilot_sweep' in data:
        fig08_robustness(data.get('pn_sweep'), data.get('pilot_sweep'), out_dir)

    if 'latency' in data:
        fig09_latency(data['latency'], out_dir)

    if 'heatmap_sweep' in data:
        fig11_heatmap(data['heatmap_sweep'], out_dir)

    print("\n" + "=" * 60)
    print(f"✅ All figures saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--out_dir', type=str, default=None)
    args = parser.parse_args()
    generate_all_figures(args.data_dir, args.out_dir)