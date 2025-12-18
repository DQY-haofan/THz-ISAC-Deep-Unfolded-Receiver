"""
run_p4_experiments.py (Fixed Version v4 - Expert Review Applied)

Description:
    Phase 4 Evaluation & Visualization for GA-BV-Net.

    **FIXES APPLIED** per Expert Review:
    - [FIX 1] noise_var uses actual signal power (not assumed 1.0)
    - [FIX 2] BER calculation → true bit-level BER
    - [FIX 3] BCRLB_ref → independent χ-scaled bound
    - [FIX 4] numpy→torch conversion fixed
    - [FIX 5] MC Randomness → unique seed per trial
    - [FIX 6] fs consistency check
    - [FIX 7] Optional debug mode with theta_init = theta_true
    - [NEW] Enhanced GPU detection for Colab compatibility

    Output:
    - metrics_mean.csv (averaged results)
    - metrics_raw.csv (per-MC results with seed tracking)
    - fig_*.png / fig_*.pdf (publication-quality)
    - config.json (reproducibility metadata)

Author: Expert Review Fixed v4
Date: 2025-12-18
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import pandas as pd
import json
import sys
import os
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import time

# --- GPU Utilities ---
try:
    from gpu_utils import setup_device, print_gpu_info, clear_gpu_memory, memory_summary
    HAS_GPU_UTILS = True
except ImportError:
    HAS_GPU_UTILS = False

    def setup_device(preferred_device=None, verbose=True):
        """Fallback device setup."""
        if preferred_device:
            device = torch.device(preferred_device)
        elif torch.cuda.is_available():
            device = torch.device('cuda')
        else:
            device = torch.device('cpu')

        if verbose:
            print(f"[Device] Using: {device}")
            if device.type == 'cuda':
                print(f"[GPU] {torch.cuda.get_device_name(0)}")
                print(f"[GPU] Memory: {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
        return device

    def print_gpu_info():
        if torch.cuda.is_available():
            print(f"[GPU] {torch.cuda.get_device_name(0)}")
        else:
            print("[GPU] CUDA not available, using CPU")

    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def memory_summary():
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**3
            total = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"[GPU Memory] {allocated:.2f}/{total:.1f} GB")

# Import modules
try:
    from thz_isac_world import SimConfig, simulate_batch
    import geometry_metrics as gm
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

try:
    from gabv_net_model import GABVNet, GABVConfig
    HAS_MODEL = True
except ImportError:
    HAS_MODEL = False
    print("Warning: gabv_net_model.py not found. GA-BV-Net disabled.")

try:
    from baselines_receivers import detector_lmmse_bussgang_torch, detector_gamp_1bit_torch
    HAS_BASELINES = True
except ImportError:
    HAS_BASELINES = False
    print("Warning: baselines_receivers.py not found. Baselines disabled.")


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class ExperimentConfig:
    """Single experiment configuration."""
    scene_id: str = "S1"
    description: str = "Default Scenario"

    # Hardware config
    enable_pa: bool = True
    ibo_dB: float = 3.0
    enable_pn: bool = True
    pn_linewidth: float = 100e3
    enable_quantization: bool = True

    # Geometry
    R: float = 500e3
    v_rel: float = 7.5e3

    # Evaluation
    snr_grid: List[float] = field(default_factory=lambda: list(np.arange(-5, 26, 2)))
    n_mc: int = 10
    batch_size: int = 64

    # MC Randomness
    base_seed: int = 42

    # Toggles
    use_ga_bv_net: bool = True
    use_baselines: bool = True

    # [FIX 7] Debug mode: use exact theta (no noise)
    debug_theta_exact: bool = False
    theta_noise_scale: Tuple[float, float, float] = (100.0, 10.0, 0.5)


# Plot Configuration
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.figsize': (7, 5),
    'lines.linewidth': 1.5,
    'lines.markersize': 6,
})


# =============================================================================
# 1. Meta Feature Construction
# =============================================================================

def construct_meta_features(meta: dict, batch_size: int) -> torch.Tensor:
    """Constructs the FROZEN meta feature vector for GA-BV-Net."""
    snr_db = float(meta.get('snr_db', 20.0))
    gamma_eff = float(meta.get('gamma_eff', 1e6))
    chi = float(meta.get('chi', 0.6366))
    sigma_eta = float(meta.get('sigma_eta', 0.0))

    # [FIX] When enable_pn=False, pn_linewidth should be 0
    enable_pn = meta.get('enable_pn', True)
    if enable_pn:
        pn_linewidth = float(meta.get('pn_linewidth', 100e3))
    else:
        pn_linewidth = 0.0  # No PN

    ibo_dB = float(meta.get('ibo_dB', 3.0))

    snr_db_norm = (snr_db - 15.0) / 15.0
    gamma_eff_db = 10.0 * np.log10(max(gamma_eff, 1e-12))
    gamma_eff_db_norm = (gamma_eff_db - 10.0) / 20.0
    chi_raw = chi
    sigma_eta_norm = sigma_eta / 0.1
    pn_linewidth_norm = np.log10(pn_linewidth + 1.0) / 6.0
    ibo_db_norm = (ibo_dB - 3.0) / 3.0

    features = torch.tensor([
        snr_db_norm,
        gamma_eff_db_norm,
        chi_raw,
        sigma_eta_norm,
        pn_linewidth_norm,
        ibo_db_norm
    ], dtype=torch.float32)

    meta_t = features.unsqueeze(0).expand(batch_size, -1).clone()
    return meta_t


# =============================================================================
# 2. Independent BCRLB Reference
# =============================================================================

def compute_bcrlb_ref_independent(snr_linear: float, chi: float,
                                   R: float, v_rel: float,
                                   fc: float = 300e9, B: float = 20e9,
                                   N: int = 1024) -> Dict[str, float]:
    """Computes INDEPENDENT BCRLB reference using χ-scaling."""
    c = 3e8

    crlb_R_analog = (c ** 2) / (8 * (np.pi ** 2) * (B ** 2) * snr_linear * N + 1e-12)

    Ts = 1.0 / 10e9  # [FIX 6] Use correct fs=10e9
    T_obs = N * Ts
    crlb_v_analog = (c ** 2) / (8 * (np.pi ** 2) * (fc ** 2) * (T_obs ** 2) * snr_linear * N + 1e-12)

    crlb_a_analog = crlb_v_analog * 100

    chi_safe = max(chi, 1e-6)
    chi_sq = chi_safe ** 2

    crlb_R_1bit = crlb_R_analog / chi_sq
    crlb_v_1bit = crlb_v_analog / chi_sq
    crlb_a_1bit = crlb_a_analog / chi_sq

    return {
        'BCRLB_R_analog': crlb_R_analog,
        'BCRLB_v_analog': crlb_v_analog,
        'BCRLB_a_analog': crlb_a_analog,
        'BCRLB_R_ref': crlb_R_1bit,
        'BCRLB_v_ref': crlb_v_1bit,
        'BCRLB_a_ref': crlb_a_1bit,
        'chi_used': chi_safe
    }


# =============================================================================
# 3. Metrics Calculation
# =============================================================================

def compute_ber_qpsk_bitwise(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Computes TRUE bit-level BER for QPSK."""
    bit_I_true = (np.real(x_true) > 0).astype(int)
    bit_Q_true = (np.imag(x_true) > 0).astype(int)

    bit_I_hat = (np.real(x_hat) > 0).astype(int)
    bit_Q_hat = (np.imag(x_hat) > 0).astype(int)

    errors_I = np.sum(bit_I_true != bit_I_hat)
    errors_Q = np.sum(bit_Q_true != bit_Q_hat)

    total_bits = 2 * x_true.size
    ber = (errors_I + errors_Q) / total_bits

    return ber


def compute_ser_qpsk(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Symbol Error Rate for QPSK."""
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    idx_true = np.argmin(np.abs(x_true[..., None] - constellation), axis=-1)
    idx_hat = np.argmin(np.abs(x_hat[..., None] - constellation), axis=-1)
    return np.mean(idx_true != idx_hat)


def compute_nmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Normalized MSE in dB."""
    mse = np.mean(np.abs(x_hat - x_true)**2)
    power = np.mean(np.abs(x_true)**2) + 1e-12
    return 10 * np.log10(mse / power + 1e-12)


def compute_rmse_theta(theta_hat: np.ndarray, theta_true: np.ndarray) -> Tuple[float, float, float]:
    """RMSE for R, v, a."""
    err = theta_hat - theta_true
    rmse_R = np.sqrt(np.mean(err[:, 0]**2))
    rmse_v = np.sqrt(np.mean(err[:, 1]**2))
    rmse_a = np.sqrt(np.mean(err[:, 2]**2))
    return rmse_R, rmse_v, rmse_a


# =============================================================================
# 4. Baseline Runners
# =============================================================================

def run_baselines_on_batch(y_q_np: np.ndarray, h_diag_np: np.ndarray,
                           x_true_np: np.ndarray, snr_linear: float,
                           device: str) -> Dict[str, np.ndarray]:
    """
    Run baseline detectors on a batch.

    [FIX 1] noise_var now uses actual signal power, not assumed 1.0
    """
    if not HAS_BASELINES:
        return {}

    y_q = torch.from_numpy(y_q_np).cfloat().to(device)
    h_diag = torch.from_numpy(h_diag_np).cfloat().to(device)

    # [FIX 1] Use actual signal power for noise_var calculation
    signal_power = np.mean(np.abs(x_true_np)**2)
    noise_var = signal_power / snr_linear

    results = {}

    try:
        x_lmmse = detector_lmmse_bussgang_torch(y_q, h_diag, noise_var)
        results['x_lmmse'] = x_lmmse.cpu().numpy()
    except Exception as e:
        print(f"  LMMSE failed: {e}")

    try:
        x_gamp = detector_gamp_1bit_torch(y_q, h_diag, noise_var)
        results['x_gamp'] = x_gamp.cpu().numpy()
    except Exception as e:
        print(f"  GAMP failed: {e}")

    return results


# =============================================================================
# 5. Model Loading
# =============================================================================

def load_gabv_model(ckpt_path: str, device: str) -> Optional[GABVNet]:
    """Load GA-BV-Net from checkpoint."""
    if not HAS_MODEL:
        return None

    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        cfg = GABVConfig(n_layers=checkpoint['config'].get('n_layers', 8))

        # [FIX 6] Check fs in checkpoint
        saved_fs = checkpoint['config'].get('fs', None)
        if saved_fs and abs(saved_fs - cfg.fs) > 1e6:
            print(f"  ⚠️  WARNING: Checkpoint fs={saved_fs/1e9:.1f}GHz != config fs={cfg.fs/1e9:.1f}GHz")

        model = GABVNet(cfg)
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        print(f"[Model] Loaded from: {ckpt_path}")
        print(f"[Model] Version: {checkpoint.get('version', 'unknown')}")
        return model
    except Exception as e:
        print(f"[Model] Failed to load: {e}")
        return None


# =============================================================================
# 6. Single Scenario Runner
# =============================================================================

def run_single_scenario(exp_cfg: ExperimentConfig, model: Optional[GABVNet],
                        device: str, ckpt_tag: str, out_base: str):
    """Run evaluation for a single scenario."""
    out_dir = Path(out_base) / exp_cfg.scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scene: {exp_cfg.scene_id} - {exp_cfg.description}")
    if exp_cfg.debug_theta_exact:
        print(f"DEBUG MODE: theta_init = theta_true (NO NOISE)")
    print(f"{'='*60}")

    seed_rng = np.random.default_rng(exp_cfg.base_seed)

    sim_cfg = SimConfig()
    sim_cfg.enable_pa = exp_cfg.enable_pa
    sim_cfg.ibo_dB = exp_cfg.ibo_dB
    sim_cfg.enable_pn = exp_cfg.enable_pn
    sim_cfg.pn_linewidth = exp_cfg.pn_linewidth
    sim_cfg.enable_quantization = exp_cfg.enable_quantization
    sim_cfg.R = exp_cfg.R
    sim_cfg.v_rel = exp_cfg.v_rel

    all_results = []
    mean_results = []

    for snr_db in tqdm(exp_cfg.snr_grid, desc="SNR sweep"):
        sim_cfg.snr_db = snr_db
        snr_linear = 10 ** (snr_db / 10)

        mc_metrics = {
            'BER_Net': [], 'BER_LMMSE': [], 'BER_GAMP': [],
            'NMSE_Net': [], 'NMSE_LMMSE': [], 'NMSE_GAMP': [],
            'RMSE_R': [], 'RMSE_v': [], 'RMSE_a': [],
            'gamma_eff': [], 'chi': [],
            'mc_seed': []
        }

        for mc_idx in range(exp_cfg.n_mc):
            mc_seed = int(seed_rng.integers(0, 2**31))
            mc_metrics['mc_seed'].append(mc_seed)

            data = simulate_batch(sim_cfg, batch_size=exp_cfg.batch_size, seed=mc_seed)
            meta = data['meta']

            mc_metrics['gamma_eff'].append(meta['gamma_eff'])
            mc_metrics['chi'].append(meta['chi'])

            # Run GA-BV-Net
            if model is not None and exp_cfg.use_ga_bv_net:
                with torch.no_grad():
                    y_q_t = torch.from_numpy(data['y_q']).cfloat().to(device)
                    x_true_t = torch.from_numpy(data['x_true']).cfloat().to(device)
                    theta_true_t = torch.from_numpy(data['theta_true']).float().to(device)

                    # [FIX 7] Optional exact theta for debugging
                    if exp_cfg.debug_theta_exact:
                        theta_init_t = theta_true_t.clone()
                    else:
                        noise_scale = torch.tensor(exp_cfg.theta_noise_scale, device=device)
                        theta_init_t = theta_true_t + torch.randn_like(theta_true_t) * noise_scale

                    meta_t = construct_meta_features(meta, exp_cfg.batch_size).to(device)

                    batch = {
                        'y_q': y_q_t,
                        'x_true': x_true_t,
                        'theta_init': theta_init_t,
                        'meta': meta_t
                    }
                    outputs = model(batch)

                x_hat_np = outputs['x_hat'].cpu().numpy()
                theta_hat_np = outputs['theta_hat'].cpu().numpy()

                ber_net = compute_ber_qpsk_bitwise(x_hat_np, data['x_true'])
                nmse_net = compute_nmse(x_hat_np, data['x_true'])
                rmse_R, rmse_v, rmse_a = compute_rmse_theta(theta_hat_np, data['theta_true'])

                mc_metrics['BER_Net'].append(ber_net)
                mc_metrics['NMSE_Net'].append(nmse_net)
                mc_metrics['RMSE_R'].append(rmse_R)
                mc_metrics['RMSE_v'].append(rmse_v)
                mc_metrics['RMSE_a'].append(rmse_a)

            # Run baselines
            if exp_cfg.use_baselines and HAS_BASELINES:
                h_diag = meta['h_diag']

                baseline_out = run_baselines_on_batch(
                    data['y_q'], h_diag, data['x_true'], snr_linear, device
                )

                if 'x_lmmse' in baseline_out:
                    ber_lmmse = compute_ber_qpsk_bitwise(baseline_out['x_lmmse'], data['x_true'])
                    nmse_lmmse = compute_nmse(baseline_out['x_lmmse'], data['x_true'])
                    mc_metrics['BER_LMMSE'].append(ber_lmmse)
                    mc_metrics['NMSE_LMMSE'].append(nmse_lmmse)

                if 'x_gamp' in baseline_out:
                    ber_gamp = compute_ber_qpsk_bitwise(baseline_out['x_gamp'], data['x_true'])
                    nmse_gamp = compute_nmse(baseline_out['x_gamp'], data['x_true'])
                    mc_metrics['BER_GAMP'].append(ber_gamp)
                    mc_metrics['NMSE_GAMP'].append(nmse_gamp)

            all_results.append({
                'snr_db': snr_db,
                'mc_idx': mc_idx,
                'mc_seed': mc_seed,
                'gamma_eff': meta['gamma_eff'],
                'chi': meta['chi'],
                'BER_Net': mc_metrics['BER_Net'][-1] if mc_metrics['BER_Net'] else np.nan,
                'NMSE_Net': mc_metrics['NMSE_Net'][-1] if mc_metrics['NMSE_Net'] else np.nan,
                'BER_LMMSE': mc_metrics['BER_LMMSE'][-1] if mc_metrics['BER_LMMSE'] else np.nan,
                'BER_GAMP': mc_metrics['BER_GAMP'][-1] if mc_metrics['BER_GAMP'] else np.nan,
            })

        def safe_mean(arr):
            return np.mean(arr) if arr else np.nan

        def safe_std(arr):
            return np.std(arr) if len(arr) > 1 else 0.0

        avg_chi = safe_mean(mc_metrics['chi'])
        bcrlb_ref = compute_bcrlb_ref_independent(
            snr_linear, avg_chi, exp_cfg.R, exp_cfg.v_rel
        )

        mean_results.append({
            'snr_db': snr_db,
            'gamma_eff': safe_mean(mc_metrics['gamma_eff']),
            'chi': avg_chi,
            'BER_Net': safe_mean(mc_metrics['BER_Net']),
            'BER_Net_std': safe_std(mc_metrics['BER_Net']),
            'BER_LMMSE': safe_mean(mc_metrics['BER_LMMSE']),
            'BER_GAMP': safe_mean(mc_metrics['BER_GAMP']),
            'NMSE_Net': safe_mean(mc_metrics['NMSE_Net']),
            'RMSE_R': safe_mean(mc_metrics['RMSE_R']),
            'RMSE_R_std': safe_std(mc_metrics['RMSE_R']),
            'RMSE_v': safe_mean(mc_metrics['RMSE_v']),
            'RMSE_a': safe_mean(mc_metrics['RMSE_a']),
            'BCRLB_R_ref': bcrlb_ref['BCRLB_R_ref'],
            'BCRLB_R_analog': bcrlb_ref['BCRLB_R_analog'],
        })

    # Save results
    df_mean = pd.DataFrame(mean_results)
    df_mean.to_csv(out_dir / "metrics_mean.csv", index=False)

    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(out_dir / "metrics_raw.csv", index=False)

    config_data = {
        'experiment': asdict(exp_cfg),
        'checkpoint': ckpt_tag,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
        'version': 'v4.0_expert_fixed',
        'fixes_applied': [
            'noise_var uses actual signal power',
            'fs=10GHz in BCRLB',
            'debug_theta_exact option',
        ]
    }
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    generate_plots(df_mean, out_dir)

    print(f"  [Results] Saved to {out_dir}")

    unique_seeds = df_raw['mc_seed'].nunique()
    total_trials = len(df_raw)
    print(f"  [MC Check] Unique seeds: {unique_seeds}/{total_trials}")

    return df_mean


# =============================================================================
# 7. Plot Generation
# =============================================================================

def generate_plots(df: pd.DataFrame, out_dir: Path):
    """Generate publication-quality plots."""

    # Plot 1: BER vs SNR
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    if "BER_Net" in df.columns and not df["BER_Net"].isna().all():
        ax1.semilogy(df["snr_db"], df["BER_Net"], 'b-o', label='GA-BV-Net')
    if "BER_LMMSE" in df.columns and not df["BER_LMMSE"].isna().all():
        ax1.semilogy(df["snr_db"], df["BER_LMMSE"], 'r--s', label='Bussgang-LMMSE')
    if "BER_GAMP" in df.columns and not df["BER_GAMP"].isna().all():
        ax1.semilogy(df["snr_db"], df["BER_GAMP"], 'g-.^', label='1-bit GAMP')

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Bit Error Rate')
    ax1.legend(loc='upper right')
    ax1.grid(True, which='both', linestyle='-', alpha=0.3)
    ax1.set_ylim([1e-5, 1])

    fig1.tight_layout()
    fig1.savefig(out_dir / "fig_ber.png", dpi=300, bbox_inches='tight')
    fig1.savefig(out_dir / "fig_ber.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig1)

    # Plot 2: RMSE Range with BCRLB
    fig2, ax2 = plt.subplots(figsize=(7, 5))

    if "RMSE_R" in df.columns and not df["RMSE_R"].isna().all():
        ax2.semilogy(df["snr_db"], df["RMSE_R"], 'b-o', label='GA-BV-Net RMSE')

    if "BCRLB_R_ref" in df.columns and not df["BCRLB_R_ref"].isna().all():
        lb_ref = np.sqrt(df["BCRLB_R_ref"])
        ax2.semilogy(df["snr_db"], lb_ref, 'k-', linewidth=2,
                     label=r'$\sqrt{\mathrm{BCRLB}_\mathrm{ref}}$ (χ-scaled)')

    if "BCRLB_R_analog" in df.columns and not df["BCRLB_R_analog"].isna().all():
        lb_analog = np.sqrt(df["BCRLB_R_analog"])
        ax2.semilogy(df["snr_db"], lb_analog, 'k:', linewidth=1,
                     label=r'$\sqrt{\mathrm{CRLB}_\mathrm{analog}}$')

    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel('Range RMSE (m)')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle='-', alpha=0.3)

    fig2.tight_layout()
    fig2.savefig(out_dir / "fig_rmse_range.png", dpi=300, bbox_inches='tight')
    fig2.savefig(out_dir / "fig_rmse_range.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig2)

    # Plot 3: NMSE vs SNR
    fig3, ax3 = plt.subplots(figsize=(7, 5))

    if "NMSE_Net" in df.columns and not df["NMSE_Net"].isna().all():
        ax3.plot(df["snr_db"], df["NMSE_Net"], 'b-o', label='GA-BV-Net')

    ax3.set_xlabel('SNR (dB)')
    ax3.set_ylabel('NMSE (dB)')
    ax3.legend(loc='upper right')
    ax3.grid(True, linestyle='-', alpha=0.3)

    fig3.tight_layout()
    fig3.savefig(out_dir / "fig_nmse.png", dpi=300, bbox_inches='tight')
    fig3.savefig(out_dir / "fig_nmse.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig3)

    # Plot 4: Gamma_eff and Chi vs SNR
    fig4, ax4a = plt.subplots(figsize=(7, 5))

    if "gamma_eff" in df.columns and not df["gamma_eff"].isna().all():
        gamma_db = 10 * np.log10(df["gamma_eff"].values + 1e-12)
        ax4a.plot(df["snr_db"], gamma_db, 'b-o', label=r'$\Gamma_\mathrm{eff}$ (dB)')
        ax4a.set_ylabel(r'$\Gamma_\mathrm{eff}$ (dB)', color='b')
        ax4a.tick_params(axis='y', labelcolor='b')

    ax4b = ax4a.twinx()
    if "chi" in df.columns and not df["chi"].isna().all():
        ax4b.plot(df["snr_db"], df["chi"], 'r--s', label=r'$\chi$')
        ax4b.set_ylabel(r'$\chi$ (Information Retention)', color='r')
        ax4b.tick_params(axis='y', labelcolor='r')
        ax4b.axhline(y=2/np.pi, color='r', linestyle=':', alpha=0.5,
                     label=r'$\chi_\mathrm{max} = 2/\pi$')

    ax4a.set_xlabel('SNR (dB)')
    ax4a.grid(True, linestyle='-', alpha=0.3)

    lines1, labels1 = ax4a.get_legend_handles_labels()
    lines2, labels2 = ax4b.get_legend_handles_labels()
    ax4a.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig4.tight_layout()
    fig4.savefig(out_dir / "fig_gamma_chi.png", dpi=300, bbox_inches='tight')
    fig4.savefig(out_dir / "fig_gamma_chi.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig4)

    print(f"  [Plots] Saved 4 figures to {out_dir}")


# =============================================================================
# 8. Experiment Configurations
# =============================================================================

def get_default_experiments() -> List[ExperimentConfig]:
    """Returns default experiment configurations."""
    return [
        ExperimentConfig(
            scene_id="S1_full_hw",
            description="Full Hardware (PA+PN+1bit)",
            enable_pa=True, ibo_dB=3.0,
            enable_pn=True, pn_linewidth=100e3,
            enable_quantization=True,
            n_mc=10, batch_size=64,
            base_seed=42
        ),
        ExperimentConfig(
            scene_id="S2_pa_only",
            description="PA Only (no PN, no 1-bit)",
            enable_pa=True, ibo_dB=3.0,
            enable_pn=False,
            enable_quantization=False,
            n_mc=10, batch_size=64,
            base_seed=42
        ),
        ExperimentConfig(
            scene_id="S3_pn_only",
            description="PN Only (no PA)",
            enable_pa=False,
            enable_pn=True, pn_linewidth=100e3,
            enable_quantization=True,
            n_mc=10, batch_size=64,
            base_seed=42
        ),
        ExperimentConfig(
            scene_id="S4_ideal",
            description="Ideal (no impairments)",
            enable_pa=False, enable_pn=False,
            enable_quantization=False,
            n_mc=10, batch_size=64,
            base_seed=42
        ),
    ]


# =============================================================================
# 9. Main Entry Point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P4 Experiments (Expert Fixed v4)")
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to GA-BV-Net checkpoint')
    parser.add_argument('--out', type=str, default='results/p4_experiments',
                        help='Output directory')
    parser.add_argument('--scene', type=str, default='all',
                        help='Scene ID to run (or "all")')
    parser.add_argument('--n_mc', type=int, default=10,
                        help='Number of Monte Carlo trials')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size per trial')
    parser.add_argument('--no_baselines', action='store_true',
                        help='Disable baselines')
    parser.add_argument('--no_model', action='store_true',
                        help='Disable GA-BV-Net')
    parser.add_argument('--debug_theta', action='store_true',
                        help='Debug: use theta_init = theta_true')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    # =========================================================================
    # GPU Setup - Important for Colab!
    # =========================================================================
    device = setup_device(preferred_device=args.device, verbose=True)
    device_str = str(device)

    print("\n" + "=" * 60)
    print("P4 Experiments - Expert Fixed v4")
    print("=" * 60)
    print(f"Device: {device_str}")
    if args.debug_theta:
        print("⚠️  DEBUG MODE: theta_init = theta_true (NO NOISE)")
        print("    Use this to verify BER decreases with SNR")
    print("=" * 60)

    # Clear GPU memory before starting
    clear_gpu_memory()

    model = None
    ckpt_tag = "none"
    if HAS_MODEL and args.ckpt and not args.no_model:
        model = load_gabv_model(args.ckpt, device_str)
        ckpt_tag = Path(args.ckpt).stem
    elif not args.no_model:
        print("[Model] No checkpoint provided, GA-BV-Net disabled")

    experiments = get_default_experiments()

    if args.scene != 'all':
        experiments = [e for e in experiments if e.scene_id == args.scene]
        if not experiments:
            print(f"[Error] Scene '{args.scene}' not found")
            sys.exit(1)

    # Auto-adjust batch size based on GPU memory
    batch_size = args.batch_size
    if device.type == 'cuda':
        total_mem = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        if total_mem < 8 and batch_size > 32:
            batch_size = 32
            print(f"⚠️  Limited GPU memory ({total_mem:.1f}GB), reducing batch_size to {batch_size}")

    for exp in experiments:
        if args.no_baselines:
            exp.use_baselines = False
        if args.no_model:
            exp.use_ga_bv_net = False
        exp.n_mc = args.n_mc
        exp.batch_size = batch_size
        exp.debug_theta_exact = args.debug_theta

    print(f"[Config] n_mc={args.n_mc}, batch_size={batch_size}")
    print(f"[Config] Scenes: {[e.scene_id for e in experiments]}")

    # Run all experiments and collect results
    all_scene_results = {}
    for exp in experiments:
        df_result = run_single_scenario(exp, model, device_str, ckpt_tag, args.out)
        all_scene_results[exp.scene_id] = df_result
        # Clear memory between experiments
        clear_gpu_memory()

    # ==========================================================================
    # Save Combined Results for All Scenes
    # ==========================================================================
    if len(all_scene_results) > 1:
        print("\n" + "=" * 60)
        print("📊 Generating Combined Results for All Scenes")
        print("=" * 60)

        combined_dir = Path(args.out) / "combined"
        combined_dir.mkdir(parents=True, exist_ok=True)

        # 1. Combined CSV with scene column
        combined_rows = []
        for scene_id, df in all_scene_results.items():
            if df is not None:
                df_copy = df.copy()
                df_copy.insert(0, 'scene_id', scene_id)
                combined_rows.append(df_copy)

        if combined_rows:
            df_combined = pd.concat(combined_rows, ignore_index=True)
            df_combined.to_csv(combined_dir / "all_scenes_metrics.csv", index=False)
            print(f"  ✅ Saved: {combined_dir}/all_scenes_metrics.csv")

        # 2. Generate combined comparison plots
        generate_combined_plots(all_scene_results, combined_dir)

        print(f"  ✅ Combined results saved to: {combined_dir}")

    print("\n[Done] All experiments completed.")
    memory_summary()


def generate_combined_plots(results: Dict[str, pd.DataFrame], out_dir: Path):
    """Generate combined comparison plots for all scenes."""

    # Define colors and markers for each scene
    scene_styles = {
        'S1_full_hw': {'color': 'blue', 'marker': 'o', 'label': 'Full HW (PA+PN+1bit)'},
        'S2_pa_only': {'color': 'red', 'marker': 's', 'label': 'PA Only'},
        'S3_pn_only': {'color': 'green', 'marker': '^', 'label': 'PN + 1bit'},
        'S4_ideal': {'color': 'black', 'marker': 'd', 'label': 'Ideal (No HW)'},
    }

    # =========================================================================
    # Plot 1: BER Comparison (All Scenes)
    # =========================================================================
    fig1, ax1 = plt.subplots(figsize=(8, 6))

    for scene_id, df in results.items():
        if df is None:
            continue
        style = scene_styles.get(scene_id, {'color': 'gray', 'marker': 'x', 'label': scene_id})

        # Plot GA-BV-Net if available
        if 'BER_Net' in df.columns and not df['BER_Net'].isna().all():
            ax1.semilogy(df['snr_db'], df['BER_Net'],
                        color=style['color'], marker=style['marker'],
                        linestyle='-', label=f"GA-BV-Net ({style['label']})")

        # Plot LMMSE baseline
        if 'BER_LMMSE' in df.columns and not df['BER_LMMSE'].isna().all():
            ax1.semilogy(df['snr_db'], df['BER_LMMSE'],
                        color=style['color'], marker=style['marker'],
                        linestyle='--', alpha=0.6, label=f"LMMSE ({style['label']})")

    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Bit Error Rate')
    ax1.set_title('BER vs SNR - All Scenarios Comparison')
    ax1.legend(loc='upper right', fontsize=8, ncol=2)
    ax1.grid(True, which='both', linestyle='-', alpha=0.3)
    ax1.set_ylim([1e-5, 1])

    fig1.tight_layout()
    fig1.savefig(out_dir / "combined_ber.png", dpi=300, bbox_inches='tight')
    fig1.savefig(out_dir / "combined_ber.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig1)

    # =========================================================================
    # Plot 2: Chi Comparison (All Scenes)
    # =========================================================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    for scene_id, df in results.items():
        if df is None:
            continue
        style = scene_styles.get(scene_id, {'color': 'gray', 'marker': 'x', 'label': scene_id})

        if 'chi' in df.columns and not df['chi'].isna().all():
            ax2.plot(df['snr_db'], df['chi'],
                    color=style['color'], marker=style['marker'],
                    linestyle='-', label=style['label'])

    ax2.axhline(y=2/np.pi, color='gray', linestyle=':', alpha=0.7, label=r'$\chi_{max}=2/\pi$')
    ax2.set_xlabel('SNR (dB)')
    ax2.set_ylabel(r'$\chi$ (Information Retention)')
    ax2.set_title('Chi vs SNR - All Scenarios Comparison')
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='-', alpha=0.3)
    ax2.set_ylim([0, 0.7])

    fig2.tight_layout()
    fig2.savefig(out_dir / "combined_chi.png", dpi=300, bbox_inches='tight')
    fig2.savefig(out_dir / "combined_chi.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig2)

    # =========================================================================
    # Plot 3: RMSE Range Comparison (All Scenes)
    # =========================================================================
    fig3, ax3 = plt.subplots(figsize=(8, 6))

    has_rmse_data = False
    for scene_id, df in results.items():
        if df is None:
            continue
        style = scene_styles.get(scene_id, {'color': 'gray', 'marker': 'x', 'label': scene_id})

        if 'RMSE_R' in df.columns and not df['RMSE_R'].isna().all():
            ax3.semilogy(df['snr_db'], df['RMSE_R'],
                        color=style['color'], marker=style['marker'],
                        linestyle='-', label=style['label'])
            has_rmse_data = True

    if has_rmse_data:
        ax3.set_xlabel('SNR (dB)')
        ax3.set_ylabel('Range RMSE (m)')
        ax3.set_title('Range RMSE vs SNR - All Scenarios Comparison')
        ax3.legend(loc='upper right')
        ax3.grid(True, which='both', linestyle='-', alpha=0.3)

        fig3.tight_layout()
        fig3.savefig(out_dir / "combined_rmse.png", dpi=300, bbox_inches='tight')
        fig3.savefig(out_dir / "combined_rmse.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig3)

    # =========================================================================
    # Plot 4: Summary Table as Figure
    # =========================================================================
    fig4, ax4 = plt.subplots(figsize=(10, 4))
    ax4.axis('off')

    # Create summary table
    table_data = []
    for scene_id, df in results.items():
        if df is None:
            continue

        # Get metrics at SNR=15dB (or closest)
        idx = (df['snr_db'] - 15).abs().idxmin()
        row = df.iloc[idx]

        ber_net = f"{row.get('BER_Net', np.nan):.4f}" if not pd.isna(row.get('BER_Net')) else "N/A"
        ber_lmmse = f"{row.get('BER_LMMSE', np.nan):.4f}" if not pd.isna(row.get('BER_LMMSE')) else "N/A"
        chi_val = f"{row.get('chi', np.nan):.3f}" if not pd.isna(row.get('chi')) else "N/A"

        table_data.append([
            scene_id,
            scene_styles.get(scene_id, {}).get('label', scene_id),
            ber_net,
            ber_lmmse,
            chi_val
        ])

    if table_data:
        table = ax4.table(
            cellText=table_data,
            colLabels=['Scene ID', 'Description', 'BER (Net)', 'BER (LMMSE)', 'χ'],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 1.5)
        ax4.set_title('Summary at SNR = 15 dB', fontsize=12, fontweight='bold', y=0.9)

        fig4.tight_layout()
        fig4.savefig(out_dir / "summary_table.png", dpi=300, bbox_inches='tight')
        fig4.savefig(out_dir / "summary_table.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig4)

    print(f"  ✅ Generated 4 combined plots")


if __name__ == "__main__":
    main()