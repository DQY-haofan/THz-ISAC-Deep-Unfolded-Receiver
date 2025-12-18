"""
run_p4_experiments_v2.py (Enhanced Save Version)

Description:
    Phase 4 Evaluation & Visualization for GA-BV-Net.

    **ENHANCED FEATURES**:
    - 统一输出文件夹管理
    - 每个图保存 CSV + PNG + PDF
    - 多场景对比图 (4个阶段合并)
    - 完整的数据溯源支持

    Output Structure:
    results/all_outputs/
    ├── data/                    # 所有CSV数据
    │   ├── S1_full_hw_metrics.csv
    │   ├── combined_all_scenes.csv
    │   └── fig_*_data.csv
    ├── figures_individual/      # 单场景图
    ├── figures_combined/        # 多场景对比图
    └── config/                  # 配置文件

Author: Enhanced Save Version
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

# Import results manager
try:
    from results_manager import ResultsManager, integrate_with_run_p4
    HAS_RESULTS_MANAGER = True
except ImportError:
    HAS_RESULTS_MANAGER = False
    print("Warning: results_manager.py not found. Using basic save mode.")


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
    pn_linewidth = float(meta.get('pn_linewidth', 100e3))
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

    Ts = 1.0 / 25e9
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
                           noise_var: float, device: str) -> Dict[str, np.ndarray]:
    """Run baseline detectors on a batch."""
    if not HAS_BASELINES:
        return {}

    y_q = torch.from_numpy(y_q_np).cfloat().to(device)
    h_diag = torch.from_numpy(h_diag_np).cfloat().to(device)

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
        model = GABVNet(cfg)
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        print(f"[Model] Loaded from: {ckpt_path}")
        return model
    except Exception as e:
        print(f"[Model] Failed to load: {e}")
        return None


# =============================================================================
# 6. Single Scenario Runner
# =============================================================================

def run_single_scenario(exp_cfg: ExperimentConfig, model: Optional[GABVNet],
                        device: str, ckpt_tag: str, out_base: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Run evaluation for a single scenario.

    Returns:
        (df_mean, df_raw): Mean results and raw MC results
    """
    out_dir = Path(out_base) / exp_cfg.scene_id
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Scene: {exp_cfg.scene_id} - {exp_cfg.description}")
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
                    theta_init_t = theta_true_t + torch.randn_like(theta_true_t) * torch.tensor([100., 10., 0.5], device=device)
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
                noise_var = 1.0 / snr_linear

                baseline_out = run_baselines_on_batch(
                    data['y_q'], h_diag, noise_var, device
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

            # Store per-MC result
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

        # Compute means
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

    # Save results (原有保存逻辑保留)
    df_mean = pd.DataFrame(mean_results)
    df_mean.to_csv(out_dir / "metrics_mean.csv", index=False)

    df_raw = pd.DataFrame(all_results)
    df_raw.to_csv(out_dir / "metrics_raw.csv", index=False)

    # Save config
    config_data = {
        'experiment': asdict(exp_cfg),
        'checkpoint': ckpt_tag,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    print(f"  [Results] Saved to {out_dir}")

    return df_mean, df_raw


# =============================================================================
# 7. Experiment Configurations
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
# 8. Main Entry Point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P4 Experiments (Enhanced Save Version)")
    parser.add_argument('--ckpt', type=str, default=None,
                        help='Path to GA-BV-Net checkpoint')
    parser.add_argument('--out', type=str, default='results/p4_experiments',
                        help='Output directory for individual scenes')
    parser.add_argument('--unified_out', type=str, default='results/all_outputs',
                        help='Unified output directory for combined results')
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
    parser.add_argument('--no_combined', action='store_true',
                        help='Skip combined plots')
    parser.add_argument('--device', type=str, default=None,
                        help='Device to use (cuda/cpu)')
    args = parser.parse_args()

    if args.device:
        device = args.device
    else:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] Using: {device}")

    print("\n" + "=" * 60)
    print("P4 Experiments - Enhanced Save Version")
    print("=" * 60)

    model = None
    ckpt_tag = "none"
    if HAS_MODEL and args.ckpt and not args.no_model:
        model = load_gabv_model(args.ckpt, device)
        ckpt_tag = Path(args.ckpt).stem
    elif not args.no_model:
        print("[Model] No checkpoint provided, GA-BV-Net disabled")

    experiments = get_default_experiments()

    if args.scene != 'all':
        experiments = [e for e in experiments if e.scene_id == args.scene]
        if not experiments:
            print(f"[Error] Scene '{args.scene}' not found")
            sys.exit(1)

    # Apply command line overrides
    for exp in experiments:
        if args.no_baselines:
            exp.use_baselines = False
        if args.no_model:
            exp.use_ga_bv_net = False
        exp.n_mc = args.n_mc
        exp.batch_size = args.batch_size

    print(f"[Config] n_mc={args.n_mc}, batch_size={args.batch_size}")
    print(f"[Config] Scenes: {[e.scene_id for e in experiments]}")

    # 收集所有场景结果
    scene_results = {}

    for exp in experiments:
        df_mean, df_raw = run_single_scenario(exp, model, device, ckpt_tag, args.out)
        scene_results[exp.scene_id] = {
            'df_mean': df_mean,
            'df_raw': df_raw,
            'config': asdict(exp)
        }

    # =================================================================
    # 使用 ResultsManager 生成统一输出
    # =================================================================

    if HAS_RESULTS_MANAGER and not args.no_combined:
        print("\n" + "=" * 60)
        print("Generating Unified Outputs with ResultsManager")
        print("=" * 60)

        manager = ResultsManager(base_dir=args.unified_out)

        for scene_id, results in scene_results.items():
            manager.add_scene_data(
                scene_id,
                results['df_mean'],
                results['config']
            )

        # 保存所有输出
        manager.save_all(additional_config={
            'checkpoint': ckpt_tag,
            'n_mc': args.n_mc,
            'batch_size': args.batch_size,
            'device': device,
        })

    elif not HAS_RESULTS_MANAGER:
        print("\n[Warning] results_manager.py not found, skipping unified outputs")
        print("Run: python results_manager.py --demo to test the module")

    print("\n[Done] All experiments completed.")


if __name__ == "__main__":
    main()