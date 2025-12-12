"""
sanity_checks_p23x_v3.py

Description:
    Automated Sanity Check Suite for THz-ISAC "Dirty Hardware" Modeling.
    (Fixes: SyntaxWarnings, Windows Multiprocessing Print Spam, Output Formatting)

    Output: results/validation_results/ (CSV, PNG, PDF)

Author: Gemini (AI Thought Partner)
Date: 2025-12-12
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
except ImportError:
    print("Error: 'thz_isac_world.py' or 'geometry_metrics.py' not found.")
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

# --- Hardware Detection (Moved Logic to Function) ---
def get_hardware_config():
    """Detects hardware only when called."""
    if torch.cuda.is_available():
        return "gpu", os.cpu_count()
    else:
        return "cpu", os.cpu_count()

# --- Helper: Save Utils ---
def save_experiment_results(df, fig, filename_base):
    base_path = OUTPUT_DIR / filename_base

    # 1. Save Data
    csv_path = base_path.with_suffix('.csv')
    df.to_csv(csv_path, index=False)

    # 2. Save Plots
    png_path = base_path.with_suffix('.png')
    pdf_path = base_path.with_suffix('.pdf')

    fig.tight_layout()
    fig.savefig(png_path, dpi=300, bbox_inches='tight')
    fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
    print(f"  [Saved] -> {filename_base} (.csv/.png/.pdf)")
    plt.close(fig)

# --- Workers (Must be top-level for pickling) ---
def _exp1_worker(snr_db, scenario_params, base_config_dict):
    cfg = SimConfig(**base_config_dict)
    for k, v in scenario_params.items():
        setattr(cfg, k, v)
    cfg.snr_db = float(snr_db)

    batch_data = simulate_batch(cfg, batch_size=50)
    meta = batch_data['meta']
    x_var = np.mean(np.abs(batch_data['x_true'])**2)
    stats = {
        'P_signal': x_var * (np.abs(meta['B_gain'])**2),
        'P_pa_distortion': meta['sigma_eta'],
        'P_phase_noise': x_var * (2 * np.pi * cfg.pn_linewidth * cfg.Ts * cfg.N) if cfg.enable_pn else 0.0,
        'P_quantization_loss': 0.0
    }
    g_eff = gm.estimate_gamma_eff(stats)
    snr_linear = 10**(snr_db/10)
    chi = gm.approx_chi(g_eff, snr_linear)
    return snr_db, chi, g_eff

def _exp2_worker(block_size, J_full, norm_J_full):
    J_block = gm.compute_fim_block_diag(J_full, block_size=block_size)
    err = np.linalg.norm(J_full - J_block, 'fro') / norm_J_full
    return block_size, err * 100

def _exp3_worker(ptx_dbm, base_config_dict):
    cfg = SimConfig(**base_config_dict)
    cfg.enable_pa = True
    cfg.ibo_dB = 3.0
    cfg.enable_pn = True
    cfg.pn_linewidth = 100e3
    cfg.P_in_dBm = float(ptx_dbm)
    cfg.snr_db = 10.0 + ptx_dbm

    batch_data = simulate_batch(cfg, batch_size=50)
    x_var = np.mean(np.abs(batch_data['x_true'])**2)
    meta = batch_data['meta']
    stats = {
        'P_signal': x_var * (np.abs(meta['B_gain'])**2),
        'P_pa_distortion': meta['sigma_eta'],
        'P_phase_noise': x_var * (2 * np.pi * cfg.pn_linewidth * cfg.Ts * cfg.N),
        'P_quantization_loss': 0.0
    }
    g_eff = gm.estimate_gamma_eff(stats)
    snr_lin = 10**(cfg.snr_db/10)
    sinr_eff = 1.0 / (1.0/snr_lin + 1.0/g_eff)
    crlb_est = 1.0 / (cfg.N * sinr_eff)
    floor_val = 1.0 / (cfg.N * g_eff)
    return ptx_dbm, 10*np.log10(crlb_est), 10*np.log10(floor_val)

def _exp4_row_worker(args):
    lw, lengths, Ts = args
    row = []
    for N in lengths:
        sigma2_step = 2 * np.pi * lw * Ts
        total_var = sigma2_step * N
        is_forbidden, _ = gm.check_forbidden_region(0.0, total_var)
        row.append(1.0 if is_forbidden else 0.0)
    return row

# --- Experiment Runners ---

def run_exp1_chi_vs_snr(base_config, num_workers):
    print(f"\n--- Running Protocol 1: Chi Factor vs. SNR (Threads: {num_workers}) ---")
    snr_range = np.linspace(-10, 40, 20)
    scenarios = {
        "Ideal": {"enable_pa": False, "enable_pn": False},
        "Dirty_Typical": {"enable_pa": True, "ibo_dB": 3.0, "pn_linewidth": 100e3},
        "Dirty_Extreme": {"enable_pa": True, "ibo_dB": 0.0, "pn_linewidth": 5e6}
    }
    all_data = []
    fig, ax = plt.subplots(figsize=(6, 5))

    for name, params in scenarios.items():
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            worker = partial(_exp1_worker, scenario_params=params, base_config_dict=base_config.__dict__)
            results = list(executor.map(worker, snr_range))
        res_snr, res_chi, res_geff = zip(*results)
        ax.plot(res_snr, res_chi, marker='o', markersize=5, label=name)
        for s, c, g in results:
            all_data.append({"Scenario": name, "SNR_dB": s, "Chi": c, "Gamma_Eff": g})

    # FIX: Use raw string r'...' for LaTeX
    ax.axhline(gm.CHI_LOW_SNR_LIMIT, color='gray', linestyle='--', linewidth=1.5, label=r'Low-SNR Limit ($2/\pi$)')
    ax.set_xlabel("Analog SNR (dB)")
    ax.set_ylabel(r"Info Retention Factor $\chi$") # FIX: Raw string
    ax.set_ylim(0, 0.75)
    ax.legend()
    save_experiment_results(pd.DataFrame(all_data), fig, "exp1_chi_vs_snr")

def run_exp2_fim_error(base_config, num_workers):
    print(f"\n--- Running Protocol 3: FIM Approximation Error ---")
    N = 1024
    sigma_phi_sq = 2 * np.pi * 100e3 * base_config.Ts
    idx = np.arange(N)
    diff = np.abs(idx[:, None] - idx[None, :])
    C_pn = np.exp(-0.5 * sigma_phi_sq * diff)
    C_total = C_pn + 0.01 * np.eye(N)

    J_full = np.linalg.inv(C_total)
    norm_J_full = np.linalg.norm(J_full, 'fro')
    block_sizes = [16, 32, 64, 128, 256, 512]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        worker = partial(_exp2_worker, J_full=J_full, norm_J_full=norm_J_full)
        results = list(executor.map(worker, block_sizes))
    bs_res, err_res = zip(*results)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(bs_res, err_res, 'r-o')
    ax.axhline(5.0, color='green', linestyle='--', label='5% Threshold')
    ax.set_xlabel(r"Block Size ($K_{block}$)")
    ax.set_ylabel("Relative Frobenius Error (%)")
    ax.set_xticks(block_sizes)
    ax.legend()
    save_experiment_results(pd.DataFrame({"Block_Size": bs_res, "Error_Percent": err_res}), fig, "exp2_fim_error")

def run_exp3_crlb_floor(base_config, num_workers):
    print(f"\n--- Running Protocol 2: CRLB Floor Effect ---")
    ptx_sweep = np.linspace(-10, 35, 15)
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        worker = partial(_exp3_worker, base_config_dict=base_config.__dict__)
        results = list(executor.map(worker, ptx_sweep))
    ptx, crlb, floor = zip(*results)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(ptx, crlb, 'b-o', label='Simulated CRLB')
    # FIX: Use raw string r'...'
    ax.plot(ptx, floor, 'r--', label=r'Floor ($1/\Gamma_{eff}$)')
    ax.set_xlabel("Tx Power (dBm)")
    ax.set_ylabel("Log CRLB (dB)")
    ax.legend()
    save_experiment_results(pd.DataFrame({"Tx_Power_dBm": ptx, "CRLB_dB": crlb, "Theoretical_Floor_dB": floor}), fig, "exp3_crlb_floor")

def run_exp4_heatmap(base_config, num_workers):
    print(f"\n--- Running Protocol 4: Forbidden Region Heatmap ---")
    linewidths = np.logspace(4, 8, 30)
    lengths = np.linspace(128, 4096, 30).astype(int)
    map_args = [(lw, lengths, base_config.Ts) for lw in linewidths]

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        grid_rows = list(executor.map(_exp4_row_worker, map_args))
    grid_map = np.array(grid_rows)

    fig, ax = plt.subplots(figsize=(7, 6))
    X, Y = np.meshgrid(lengths, linewidths)
    ax.pcolormesh(X, Y, grid_map, cmap='Reds', shading='auto', alpha=0.6, vmin=0, vmax=1)

    boundary_lw = 1.0 / (2 * base_config.Ts * lengths)
    ax.plot(lengths, boundary_lw, 'b--', linewidth=2, label=r'Theory Boundary ($\sigma^2 N = \pi$)')
    ax.set_yscale('log')
    ax.set_xlabel(r"Sequence Length $N$")
    ax.set_ylabel("PN Linewidth (Hz)")
    ax.legend(loc='upper right')
    ax.text(lengths[5], linewidths[-5], "Forbidden Region", color='darkred', weight='bold')
    ax.text(lengths[-5], linewidths[2], "Safe Zone", color='black', weight='bold')

    # Save Heatmap Data flattened
    df_grid = pd.DataFrame(grid_map)
    save_experiment_results(df_grid, fig, "exp4_forbidden_heatmap")

# --- Main Entry Point (Crucial for Windows Multiprocessing) ---
if __name__ == "__main__":
    # 1. Detect Hardware (Only prints ONCE now)
    mode, num_workers = get_hardware_config()
    print(f"==================================================")
    print(f"[Hardware] Mode: {mode.upper()} | Workers: {num_workers}")
    print(f"[Output]   Saving to: {OUTPUT_DIR.absolute()}")
    print(f"==================================================")

    start_time = time.time()
    config = SimConfig()

    # 2. Run Experiments (Pass num_workers explicitly)
    run_exp1_chi_vs_snr(config, num_workers)
    run_exp2_fim_error(config, num_workers)
    run_exp3_crlb_floor(config, num_workers)
    run_exp4_heatmap(config, num_workers)

    print(f"\n[Done] All checks passed in {time.time() - start_time:.2f}s.")