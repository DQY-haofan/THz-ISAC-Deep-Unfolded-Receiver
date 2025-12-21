"""
visualization.py - Journal-Quality Figures (Expert v2.0 - Top Journal Ready)

Expert Requirements Implemented:
- NO TITLES on figures (use axis labels only)
- P1-2: Pull-in range auto-computation and visualization
- P1-3: Gap-to-oracle figure
- P1-4: Basin map (SNR × init_error) with success rate
- IEEE-style formatting

Key Changes:
1. All ax.set_title() calls removed or made optional
2. compute_pull_in_range() function added
3. fig_basin_map() for 2D success rate visualization
4. fig_gap_to_oracle() with proper oracle_sync naming
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple


# ============================================================================
# IEEE Journal Style Configuration
# ============================================================================

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
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
    'mathtext.fontset': 'stix',
})


# ============================================================================
# Method Style Configuration (Narrative-Aligned)
# ============================================================================

METHOD_CONFIG = {
    "naive_slice": {
        "label": "Naive Slice",
        "color": "#7f7f7f",
        "marker": "v",
        "ls": ":",
        "lw": 1.5,
        "alpha": 0.6
    },
    "matched_filter": {
        "label": "Matched Filter (41×)",
        "color": "#d62728",
        "marker": "X",
        "ls": "--",
        "lw": 1.8,
        "alpha": 0.5
    },
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
    "proposed_no_update": {
        "label": r"w/o $\tau$ update",
        "color": "#8c564b",
        "marker": "s",
        "ls": "-",
        "lw": 2.0,
        "alpha": 0.9
    },
    "proposed_tau_slice": {
        "label": r"Proposed ($\tau$)+Slice",
        "color": "#17becf",
        "marker": "p",
        "ls": "-.",
        "lw": 2.0,
        "alpha": 0.9
    },
    "proposed": {
        "label": "Proposed (GA-BV-Net)",
        "color": "#1f77b4",
        "marker": "o",
        "ls": "-",
        "lw": 2.8,
        "alpha": 1.0
    },
    "oracle_sync": {
        "label": r"Oracle (true $\theta$)",
        "color": "#2ca02c",
        "marker": "^",
        "ls": "-",
        "lw": 2.5,
        "alpha": 1.0
    },
    "oracle": {
        "label": r"Oracle (true $\theta$)",
        "color": "#2ca02c",
        "marker": "^",
        "ls": "-",
        "lw": 2.5,
        "alpha": 1.0
    },
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
    """Get plotting style for method."""
    return METHOD_CONFIG.get(method, {
        "label": method, "color": "black", "marker": "o",
        "ls": "-", "lw": 2.0, "alpha": 1.0
    })


# ============================================================================
# Helper Functions
# ============================================================================

def aggregate(df: pd.DataFrame, group_cols: List[str],
              value_cols: List[str]) -> pd.DataFrame:
    """Aggregate data with mean, std, and 95% CI."""
    agg_funcs = {col: ['mean', 'std', 'count'] for col in value_cols}
    agg = df.groupby(group_cols).agg(agg_funcs).reset_index()
    agg.columns = ['_'.join(col).strip('_') for col in agg.columns]

    for col in value_cols:
        mean, std, count = f'{col}_mean', f'{col}_std', f'{col}_count'
        if all(c in agg.columns for c in [mean, std, count]):
            agg[f'{col}_ci95'] = 1.96 * agg[std] / np.sqrt(agg[count].clip(lower=1))

    return agg


def load_data(data_dir: str) -> Dict[str, pd.DataFrame]:
    """Load all CSV data files."""
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
        'data_crlb_sweep.csv',
    ]

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
    """Check and warn about missing methods."""
    if 'method' not in df.columns:
        print(f"  ⚠️ {fig_name}: Missing 'method' column")
        return []

    actual = set(df['method'].unique())
    missing = set(expected) - actual

    if missing:
        print(f"  ⚠️ {fig_name}: Missing methods {missing}")

    return [m for m in expected if m in actual]


# ============================================================================
# P1-2: Pull-in Range Computation
# ============================================================================

def compute_pull_in_range(df: pd.DataFrame, method: str,
                          threshold: float = 0.95) -> float:
    """
    Compute pull-in range: max init_error where success_rate >= threshold.

    Args:
        df: DataFrame with 'init_error', 'method', 'success_rate' columns
        method: Method name
        threshold: Success rate threshold (default 95%)

    Returns:
        pull_in: Pull-in range in samples, or NaN if not computable
    """
    if 'init_error' not in df.columns or 'success_rate' not in df.columns:
        return float('nan')

    method_data = df[df['method'] == method]
    if len(method_data) == 0:
        return float('nan')

    agg = method_data.groupby('init_error')['success_rate'].mean().reset_index()

    # Find max init_error where success_rate >= threshold
    passing = agg[agg['success_rate'] >= threshold]

    if len(passing) == 0:
        return 0.0

    return passing['init_error'].max()


def compute_summary_metrics(df: pd.DataFrame, methods: List[str]) -> pd.DataFrame:
    """Compute summary metrics including pull-in range."""
    records = []

    for method in methods:
        method_data = df[df['method'] == method]
        if len(method_data) == 0:
            continue

        pull_in = compute_pull_in_range(df, method)

        records.append({
            'method': method,
            'pull_in_range': pull_in,
            'mean_ber': method_data['ber'].mean(),
            'mean_rmse_tau': method_data['rmse_tau_final'].mean() if 'rmse_tau_final' in method_data.columns else float('nan'),
            'mean_success_rate': method_data['success_rate'].mean() if 'success_rate' in method_data.columns else float('nan'),
        })

    return pd.DataFrame(records)


# ============================================================================
# Figure Functions (NO TITLES - Journal Style)
# ============================================================================

def fig01_ber_vs_snr(df: pd.DataFrame, out_dir: str):
    """Fig 1: BER vs SNR (no title, axes-only information)."""
    fig, ax = plt.subplots(figsize=(10, 7))
    agg = aggregate(df, ['snr_db', 'method'], ['ber'])

    expected = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed", "oracle_sync", "oracle"]
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

    # Hardware Wall region
    ax.axvspan(18, 28, color='gray', alpha=0.1)
    ax.text(22, 0.35, r"$\Gamma_{\mathrm{eff}}$ Limited",
            fontsize=10, color='gray', ha='center', style='italic')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel('Bit Error Rate (BER)')
    ax.legend(loc='upper right', framealpha=0.9)
    ax.set_ylim([5e-2, 0.55])
    ax.set_xlim([-7, 27])
    ax.grid(True, which='both', alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig01_ber_vs_snr.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 01: BER vs SNR")


def fig02_rmse_tau_vs_snr(df: pd.DataFrame, out_dir: str):
    """Fig 2: τ RMSE vs SNR."""
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

    ax.axhline(y=0.1, color='green', linestyle=':', linewidth=2,
               label=r'Target ($\epsilon_\tau=0.1$)')
    ax.axhspan(0, 0.1, alpha=0.1, color='green')

    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'$\tau$ RMSE (samples)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig02_rmse_tau_vs_snr.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 02: RMSE τ vs SNR")


def fig04_cliff_all_methods(df: pd.DataFrame, out_dir: str):
    """
    Fig 4: Cliff Plot (core contribution figure).

    P1-2: Include pull-in range visualization.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
    agg = aggregate(df, ['init_error', 'method'], ['ber', 'rmse_tau_final', 'success_rate'])

    expected = ["naive_slice", "adjoint_lmmse", "adjoint_slice", "matched_filter",
                "proposed_no_update", "proposed", "oracle_sync", "oracle"]
    methods = check_methods(df, expected, "Fig 04")

    # Compute pull-in range from data
    pull_in_proposed = compute_pull_in_range(df, 'proposed', threshold=0.95)
    pull_in_baseline = compute_pull_in_range(df, 'adjoint_slice', threshold=0.95)

    if np.isnan(pull_in_proposed):
        pull_in_proposed = 0.3  # Default

    max_x = max(agg['init_error'].max(), 1.5)

    # Zone shading
    for ax in [ax1, ax2]:
        ax.axvspan(0, pull_in_proposed, color='green', alpha=0.08, label='_nolegend_')
        ax.axvline(pull_in_proposed, color='green', linestyle='--', linewidth=2, alpha=0.8)

    # Panel A: BER vs init_error
    for method in methods:
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        s = get_style(method)
        ax1.plot(data['init_error'], data['ber_mean'],
                marker=s['marker'], color=s['color'], linestyle=s['ls'],
                linewidth=s['lw'], alpha=s['alpha'], label=s['label'], markersize=6)

    ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=1)
    ax1.text(max_x - 0.1, 0.48, 'Random', fontsize=9, color='gray', alpha=0.7, ha='right')

    # Pull-in annotation
    ax1.annotate(f'Pull-in={pull_in_proposed:.2f}',
                xy=(pull_in_proposed, 0.15), xytext=(pull_in_proposed + 0.3, 0.25),
                arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
                fontsize=10, color='green', fontweight='bold')

    ax1.set_xlabel(r'Initial $\tau$ Error (samples)')
    ax1.set_ylabel('BER')
    ax1.legend(loc='upper left', fontsize=8, ncol=2)
    ax1.set_ylim([0, 0.55])
    ax1.set_xlim([0, max_x])
    ax1.grid(True, alpha=0.3)

    # Panel B: RMSE vs init_error
    for method in methods:
        if method in ["oracle", "oracle_sync"]:
            continue
        data = agg[agg['method'] == method]
        if len(data) == 0:
            continue

        s = get_style(method)
        ax2.plot(data['init_error'], data['rmse_tau_final_mean'],
                marker=s['marker'], color=s['color'], linestyle=s['ls'],
                linewidth=s['lw'], alpha=s['alpha'], label=s['label'], markersize=6)

    # y=x reference line
    ax2.plot([0, max_x], [0, max_x], 'k--', alpha=0.4, linewidth=1, label='No Improvement')
    ax2.axhspan(0, 0.1, alpha=0.1, color='green')
    ax2.text(0.05, 0.08, r'$\epsilon_\tau$', fontsize=9, color='green', alpha=0.8)

    ax2.set_xlabel(r'Initial $\tau$ Error (samples)')
    ax2.set_ylabel(r'Final $\tau$ RMSE (samples)')
    ax2.legend(loc='upper left', fontsize=8)
    ax2.set_ylim([0, max_x])
    ax2.set_xlim([0, max_x])
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig04_cliff_all_methods.{ext}", dpi=300)
    plt.close(fig)
    print(f"  ✓ Fig 04: Cliff Plot (pull-in={pull_in_proposed:.2f})")


def fig05_snr_multi_init_error(df: pd.DataFrame, out_dir: str):
    """Fig 5: Multi init_error SNR sweep."""
    if 'init_error' not in df.columns:
        print("  ⚠️ Fig 05: Missing init_error column, skipping")
        return

    init_errors = sorted(df['init_error'].unique())
    n_panels = min(len(init_errors), 3)

    fig, axes = plt.subplots(1, n_panels, figsize=(5*n_panels, 4.5))
    if n_panels == 1:
        axes = [axes]

    expected = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync", "oracle"]
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
            ax.semilogy(data['snr_db'], data['ber_mean'],
                       marker=s['marker'], color=s['color'], linestyle=s['ls'],
                       label=s['label'], linewidth=s['lw'], alpha=s['alpha'])

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')

        # Color-coded init_error label
        if init_error == 0.0:
            color = 'green'
            label = r'$\Delta\tau_0=0$'
        elif init_error <= 0.2:
            color = 'orange'
            label = f'$\\Delta\\tau_0={init_error}$'
        else:
            color = 'red'
            label = f'$\\Delta\\tau_0={init_error}$'

        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12,
                color=color, fontweight='bold', verticalalignment='top')

        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([5e-2, 0.55])

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig05_snr_multi_init_error.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 05: SNR @ Multi Init Error")


def fig07_gap_to_oracle(df: pd.DataFrame, out_dir: str):
    """
    Fig 7: Gap-to-Oracle (P1-3).

    Shows gap = method - oracle_sync for both BER and τ RMSE.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    agg = aggregate(df, ['snr_db', 'method'], ['ber', 'rmse_tau_final'])

    # Get oracle data (try oracle_sync first, then oracle)
    oracle_data = agg[agg['method'].isin(['oracle_sync', 'oracle'])]
    if len(oracle_data) == 0:
        print("  ⚠️ Fig 07: No oracle data, skipping")
        return

    oracle_data = oracle_data.groupby('snr_db').agg({
        'ber_mean': 'first',
        'rmse_tau_final_mean': 'first'
    }).reset_index()
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
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel(r'$\Delta$BER (method $-$ oracle)')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Panel B: RMSE Gap (ratio)
    for method in methods:
        data = agg[agg['method'] == method][['snr_db', 'rmse_tau_final_mean']]
        merged = pd.merge(data, oracle_data, on='snr_db')
        if len(merged) == 0:
            continue

        s = get_style(method)
        ratio = merged['rmse_tau_final_mean'].values / (merged['oracle_rmse'].values + 1e-10)
        ax2.semilogy(merged['snr_db'], ratio, marker=s['marker'], color=s['color'],
                    linestyle=s['ls'], label=s['label'], linewidth=s['lw'])

    ax2.axhline(y=1, color='green', linestyle='--', linewidth=2, label='Oracle')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel(r'$\tau$ RMSE Ratio (method / oracle)')
    ax2.legend(loc='upper right')
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig07_gap_to_oracle.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 07: Gap-to-Oracle")


def fig08_robustness(df_pn: pd.DataFrame, df_pilot: pd.DataFrame, out_dir: str):
    """Fig 8: PN and Pilot robustness."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    expected = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync", "oracle"]

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
        ax2.set_ylabel(r'$\tau$ RMSE (samples)')
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig08_robustness.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 08: Robustness")


def fig09_latency(df: pd.DataFrame, out_dir: str):
    """Fig 9: Latency comparison."""
    if len(df) == 0:
        print("  ⚠️ Fig 09: No latency data, skipping")
        return

    fig, ax = plt.subplots(figsize=(8, 5))

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
    ax.grid(True, alpha=0.3, axis='y')

    # Value labels
    for i, (lat, std) in enumerate(zip(latencies, stds)):
        ax.text(i, lat + std + 0.5, f'{lat:.1f}', ha='center', fontsize=10)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig09_latency.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 09: Latency")


def fig10_ablation_dual_axis(df: pd.DataFrame, out_dir: str):
    """Fig 10: Ablation study - dual axis (BER bar + τ RMSE line)."""
    fig, ax1 = plt.subplots(figsize=(10, 5.5))

    target_snr = df['snr_db'].max()
    data = df[df['snr_db'] == target_snr].copy()

    methods = ["random_init", "proposed_no_update", "proposed_tau_slice", "proposed", "oracle_sync"]
    # Filter to available methods
    methods = [m for m in methods if m in data['method'].unique() or
               (m == 'oracle_sync' and 'oracle' in data['method'].unique())]

    agg = aggregate(data, ['method'], ['ber', 'rmse_tau_final'])

    # Reorder
    method_order = {m: i for i, m in enumerate(methods)}
    agg['order'] = agg['method'].map(method_order)
    agg = agg.dropna(subset=['order']).sort_values('order')

    x = np.arange(len(agg))
    width = 0.4

    # Left axis: BER bars
    colors = [get_style(m)['color'] for m in agg['method']]
    bars = ax1.bar(x, agg['ber_mean'], width, color=colors, alpha=0.7,
                   edgecolor='black', linewidth=1.2, label='BER')

    ax1.set_ylabel('BER (Lower is Better)', fontsize=11)
    ax1.set_ylim(0, 0.5)

    # Right axis: τ RMSE line
    ax2 = ax1.twinx()
    rmse = agg['rmse_tau_final_mean'].fillna(0).values

    ax2.plot(x, rmse, color='#d62728', marker='D', markersize=10,
            linewidth=2.5, linestyle='--', label=r'$\tau$ RMSE')

    ax2.set_ylabel(r'$\tau$ RMSE (samples)', color='#d62728', fontsize=11)
    ax2.set_yscale('log')
    ax2.tick_params(axis='y', labelcolor='#d62728')
    ax2.set_ylim(1e-2, 5)

    # X axis labels
    labels = [get_style(m)['label'] for m in agg['method']]
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)

    # Combined legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', ncol=2, fontsize=9)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig10_ablation_dual.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 10: Ablation (dual axis)")


def fig11_basin_map(df: pd.DataFrame, out_dir: str):
    """
    Fig 11: Basin Map (P1-4).

    2D heatmap of success rate: SNR × init_error.
    """
    if 'init_error' not in df.columns or 'snr_db' not in df.columns:
        print("  ⚠️ Fig 11: Missing columns, skipping")
        return

    # Get proposed and baseline
    methods_to_plot = ['proposed', 'adjoint_slice']
    available = [m for m in methods_to_plot if m in df['method'].unique()]

    if len(available) == 0:
        print("  ⚠️ Fig 11: No methods available, skipping")
        return

    n_methods = len(available)
    fig, axes = plt.subplots(1, n_methods, figsize=(6*n_methods, 5))
    if n_methods == 1:
        axes = [axes]

    for idx, method in enumerate(available):
        ax = axes[idx]
        df_method = df[df['method'] == method]

        # Aggregate success rate
        agg = df_method.groupby(['snr_db', 'init_error'])['success_rate'].mean().reset_index()
        pivot = agg.pivot(index='init_error', columns='snr_db', values='success_rate')

        sns.heatmap(pivot, ax=ax, cmap='RdYlGn', annot=True, fmt='.2f',
                   cbar_kws={'label': 'Success Rate'}, vmin=0, vmax=1,
                   linewidths=0.5, linecolor='white')

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(r'Initial $\tau$ Error (samples)')
        ax.text(0.5, 1.02, get_style(method)['label'], transform=ax.transAxes,
                fontsize=11, ha='center', fontweight='bold')

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig11_basin_map.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 11: Basin Map")


def fig12_efficiency(df_snr: pd.DataFrame, df_crlb: pd.DataFrame, out_dir: str):
    """
    Fig 12: Efficiency plot (RMSE / sqrt(CRLB)).

    P2-1: Shows how close each method is to the theoretical bound.
    """
    if df_crlb is None or len(df_crlb) == 0:
        print("  ⚠️ Fig 12: No CRLB data, skipping")
        return

    fig, ax = plt.subplots(figsize=(9, 5.5))

    agg_snr = aggregate(df_snr, ['snr_db', 'method'], ['rmse_tau_final'])

    expected = ["adjoint_slice", "proposed", "oracle_sync", "oracle"]
    methods = check_methods(df_snr, expected, "Fig 12")

    for method in methods:
        data = agg_snr[agg_snr['method'] == method]
        if len(data) == 0:
            continue

        merged = pd.merge(data, df_crlb, on='snr_db')
        if len(merged) == 0:
            continue

        s = get_style(method)
        efficiency = merged['rmse_tau_final_mean'] / merged['sqrt_crlb_tau']

        ax.semilogy(merged['snr_db'], efficiency, marker=s['marker'], color=s['color'],
                   linestyle=s['ls'], label=s['label'], linewidth=s['lw'])

    ax.axhline(y=1, color='green', linestyle='--', linewidth=2, label='CRLB')
    ax.set_xlabel('SNR (dB)')
    ax.set_ylabel(r'Efficiency: RMSE / $\sqrt{\mathrm{CRLB}}$')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    for ext in ['png', 'pdf']:
        fig.savefig(f"{out_dir}/fig12_efficiency.{ext}", dpi=300)
    plt.close(fig)
    print("  ✓ Fig 12: Efficiency")


# ============================================================================
# Main Function
# ============================================================================

def generate_all_figures(data_dir: str, out_dir: str = None):
    """Generate all figures from CSV data."""
    if out_dir is None:
        out_dir = data_dir

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 60)
    print("📊 Journal Figure Generation (Expert v2.0)")
    print("=" * 60)
    print("\nLoading data...")
    data = load_data(data_dir)

    print("\n" + "-" * 40)
    print("Generating figures (NO TITLES - journal style)...")
    print("-" * 40)

    # Core figures
    if 'snr_sweep' in data:
        fig01_ber_vs_snr(data['snr_sweep'], out_dir)
        fig02_rmse_tau_vs_snr(data['snr_sweep'], out_dir)
        fig07_gap_to_oracle(data['snr_sweep'], out_dir)

    if 'cliff_sweep' in data:
        fig04_cliff_all_methods(data['cliff_sweep'], out_dir)

        # Compute and save summary metrics
        methods = data['cliff_sweep']['method'].unique().tolist()
        summary = compute_summary_metrics(data['cliff_sweep'], methods)
        summary.to_csv(f"{out_dir}/summary_metrics.csv", index=False)
        print(f"  ✓ Summary metrics saved")

    if 'snr_multi_init_error' in data:
        fig05_snr_multi_init_error(data['snr_multi_init_error'], out_dir)

    if 'ablation_sweep' in data:
        fig10_ablation_dual_axis(data['ablation_sweep'], out_dir)

    # Basin map
    if 'heatmap_sweep' in data:
        fig11_basin_map(data['heatmap_sweep'], out_dir)

    # Robustness
    if 'pn_sweep' in data or 'pilot_sweep' in data:
        fig08_robustness(data.get('pn_sweep'), data.get('pilot_sweep'), out_dir)

    if 'latency' in data:
        fig09_latency(data['latency'], out_dir)

    # Efficiency (if CRLB data available)
    if 'crlb_sweep' in data and 'snr_sweep' in data:
        fig12_efficiency(data['snr_sweep'], data['crlb_sweep'], out_dir)

    print("\n" + "=" * 60)
    print(f"✅ All figures saved to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Journal Figure Generation (v2.0)")
    parser.add_argument('--data_dir', type=str, required=True, help="CSV data directory")
    parser.add_argument('--out_dir', type=str, default=None, help="Output directory")
    args = parser.parse_args()

    generate_all_figures(args.data_dir, args.out_dir)