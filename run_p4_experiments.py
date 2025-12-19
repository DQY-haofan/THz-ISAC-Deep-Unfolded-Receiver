"""
run_p4_experiments.py (Expert Review v5.0 - Enhanced Theta Metrics)

Description:
    Phase 4 Evaluation & Visualization for GA-BV-Net.

    **ENHANCEMENTS v5.0** per Expert Review:
    - [NEW] Enhanced theta metrics logging (RMSE_R vs init, RMSE_v, RMSE_a)
    - [NEW] g_theta and accept_rate tracking per MC trial
    - [NEW] RMSE improvement ratio (vs theta_init)
    - [NEW] Score feature statistics from ThetaUpdater
    - [FIX] All previous v4 fixes preserved

    Output:
    - metrics_mean.csv (averaged results with enhanced theta metrics)
    - metrics_raw.csv (per-MC results with full theta diagnostics)
    - fig_*.png / fig_*.pdf (publication-quality)
    - config.json (reproducibility metadata)

Author: Expert Review v5.0 - Enhanced
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

    # Debug mode: use exact theta (no noise)
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

    # When enable_pn=False, pn_linewidth should be 0
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

    Ts = 1.0 / 10e9  # Use correct fs=10e9
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


def compute_rmse_vs_init(theta_hat: np.ndarray, theta_init: np.ndarray) -> Tuple[float, float, float]:
    """
    [NEW] RMSE of theta_hat vs theta_init (to check if network improved).

    A good network should have RMSE(hat, true) < RMSE(init, true)
    Equivalently, we can check if RMSE(hat, init) is reasonable.
    """
    err = theta_hat - theta_init
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
    """Run baseline detectors on a batch."""
    if not HAS_BASELINES:
        return {}

    y_q = torch.from_numpy(y_q_np).cfloat().to(device)
    h_diag = torch.from_numpy(h_diag_np).cfloat().to(device)

    # Use actual signal power for noise_var calculation
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
# 5. Phase Bias Calibration
# =============================================================================

def calibrate_phase_bias(model: GABVNet, device: str, verbose: bool = True) -> float:
    """Pilot-aided phase bias calibration."""
    if verbose:
        print("\n[Calibration] Pilot-aided phase_bias search...")

    # Setup test scenario (S3_pn_only at SNR=20dB)
    sim_cfg = SimConfig()
    sim_cfg.snr_db = 20.0
    sim_cfg.enable_pa = False
    sim_cfg.enable_pn = True
    sim_cfg.pn_linewidth = 100e3
    sim_cfg.enable_quantization = True

    data = simulate_batch(sim_cfg, batch_size=64, seed=42)
    meta_t = construct_meta_features(data['meta'], 64).to(device)
    x_true_np = data['x_true']

    batch = {
        'y_q': torch.from_numpy(data['y_q']).cfloat().to(device),
        'theta_init': torch.from_numpy(data['theta_true']).float().to(device),
        'meta': meta_t
    }

    # Save original value
    original_bias = model.pn_tracker.phase_bias.item()

    # Pilot-aided metric
    N_pilot = 64
    x_pilot = torch.from_numpy(x_true_np[:, :N_pilot]).cfloat().to(device)

    def compute_pilot_metric(x_hat):
        x_hat_pilot = x_hat[:, :N_pilot]
        corr = torch.real(
            torch.sum(x_hat_pilot * torch.conj(x_pilot)) /
            (torch.sqrt(torch.sum(torch.abs(x_hat_pilot) ** 2) *
                        torch.sum(torch.abs(x_pilot) ** 2)) + 1e-8)
        )
        return corr.item()

    # Grid search
    best_bias_deg = 0
    best_metric = -1.0

    for bias_deg in range(-180, 181, 15):
        model.pn_tracker.phase_bias.data = torch.tensor(
            np.radians(bias_deg), device=device
        )

        with torch.no_grad():
            outputs = model(batch)

        metric = compute_pilot_metric(outputs['x_hat'])

        if metric > best_metric:
            best_metric = metric
            best_bias_deg = bias_deg

    # Fine search
    for bias_deg in range(best_bias_deg - 15, best_bias_deg + 16, 5):
        model.pn_tracker.phase_bias.data = torch.tensor(
            np.radians(bias_deg), device=device
        )

        with torch.no_grad():
            outputs = model(batch)

        metric = compute_pilot_metric(outputs['x_hat'])

        if metric > best_metric:
            best_metric = metric
            best_bias_deg = bias_deg

    # Apply best value
    model.pn_tracker.phase_bias.data = torch.tensor(
        np.radians(best_bias_deg), device=device
    )

    # Compute actual BER
    with torch.no_grad():
        outputs = model(batch)
    final_ber = compute_ber_qpsk_bitwise(outputs['x_hat'].cpu().numpy(), x_true_np)

    if verbose:
        print(f"[Calibration] Method: Pilot-aided ({N_pilot} pilots)")
        print(f"[Calibration] Original: {np.degrees(original_bias):.1f}°")
        print(f"[Calibration] Optimal:  {best_bias_deg}° (correlation={best_metric:.4f}, BER={final_ber:.4f})")

    return best_bias_deg


# =============================================================================
# 6. Model Loading
# =============================================================================

def load_gabv_model(ckpt_path: str, device: str) -> Optional[GABVNet]:
    """Load GA-BV-Net from checkpoint."""
    if not HAS_MODEL:
        return None

    try:
        checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
        cfg = GABVConfig(n_layers=checkpoint['config'].get('n_layers', 8))

        saved_fs = checkpoint['config'].get('fs', None)
        if saved_fs and abs(saved_fs - cfg.fs) > 1e6:
            print(f"  ⚠️  WARNING: Checkpoint fs={saved_fs / 1e9:.1f}GHz != config fs={cfg.fs / 1e9:.1f}GHz")

        model = GABVNet(cfg)
        model.load_state_dict(checkpoint['model_state'])
        model.to(device)
        model.eval()
        print(f"[Model] Loaded from: {ckpt_path}")
        print(f"[Model] Version: {checkpoint.get('version', 'unknown')}")

        # Auto-calibrate phase_bias
        calibrate_phase_bias(model, device, verbose=True)

        return model
    except Exception as e:
        print(f"[Model] Failed to load: {e}")
        return None


# =============================================================================
# 7. [ENHANCED] Single Scenario Runner with Theta Metrics
# =============================================================================

def extract_theta_diagnostics(outputs: Dict) -> Dict[str, float]:
    """
    [NEW] Extract theta-related diagnostics from model outputs.

    Returns:
        Dict with g_theta, accept_rate, delta_R, etc.
    """
    diagnostics = {
        'g_theta': 0.0,
        'g_theta_eff': 0.0,
        'accept_rate': 1.0,
        'delta_R': 0.0,
        'delta_v': 0.0,
        'delta_a': 0.0,
        'confidence': 0.0,
    }

    try:
        last_layer = outputs['layers'][-1]

        # Gate values
        if 'gates' in last_layer:
            diagnostics['g_theta'] = last_layer['gates']['g_theta'].mean().item()

        # Theta info
        if 'theta_info' in last_layer:
            theta_info = last_layer['theta_info']
            diagnostics['g_theta_eff'] = theta_info.get('g_theta_effective', 0.0)
            diagnostics['accept_rate'] = theta_info.get('accept_rate', 1.0)
            diagnostics['delta_R'] = theta_info.get('delta_R', 0.0)
            diagnostics['delta_v'] = theta_info.get('delta_v', 0.0)
            diagnostics['delta_a'] = theta_info.get('delta_a', 0.0)
            diagnostics['confidence'] = theta_info.get('confidence', 0.0)
    except Exception as e:
        pass  # Return defaults if extraction fails

    return diagnostics


def run_single_scenario(exp_cfg: ExperimentConfig, model: Optional[GABVNet],
                        device: str, ckpt_tag: str, out_base: str):
    """Run evaluation for a single scenario with enhanced theta metrics."""
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
            'RMSE_R_init': [],  # [NEW] RMSE of init (to compare)
            'gamma_eff': [], 'chi': [],
            # [NEW] Theta diagnostics
            'g_theta': [], 'g_theta_eff': [], 'accept_rate': [],
            'delta_R': [], 'delta_v': [], 'delta_a': [],
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

                    # Theta initialization
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
                theta_init_np = theta_init_t.cpu().numpy()

                # Communication metrics
                ber_net = compute_ber_qpsk_bitwise(x_hat_np, data['x_true'])
                nmse_net = compute_nmse(x_hat_np, data['x_true'])

                # Theta RMSE vs true
                rmse_R, rmse_v, rmse_a = compute_rmse_theta(theta_hat_np, data['theta_true'])

                # [NEW] RMSE of init vs true (baseline)
                rmse_R_init, _, _ = compute_rmse_theta(theta_init_np, data['theta_true'])

                # [NEW] Extract theta diagnostics
                diag = extract_theta_diagnostics(outputs)

                mc_metrics['BER_Net'].append(ber_net)
                mc_metrics['NMSE_Net'].append(nmse_net)
                mc_metrics['RMSE_R'].append(rmse_R)
                mc_metrics['RMSE_v'].append(rmse_v)
                mc_metrics['RMSE_a'].append(rmse_a)
                mc_metrics['RMSE_R_init'].append(rmse_R_init)

                # [NEW] Theta diagnostics
                mc_metrics['g_theta'].append(diag['g_theta'])
                mc_metrics['g_theta_eff'].append(diag['g_theta_eff'])
                mc_metrics['accept_rate'].append(diag['accept_rate'])
                mc_metrics['delta_R'].append(diag['delta_R'])
                mc_metrics['delta_v'].append(diag['delta_v'])
                mc_metrics['delta_a'].append(diag['delta_a'])

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

            # [ENHANCED] Raw results with full diagnostics
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
                'RMSE_R': mc_metrics['RMSE_R'][-1] if mc_metrics['RMSE_R'] else np.nan,
                'RMSE_v': mc_metrics['RMSE_v'][-1] if mc_metrics['RMSE_v'] else np.nan,
                'RMSE_a': mc_metrics['RMSE_a'][-1] if mc_metrics['RMSE_a'] else np.nan,
                'RMSE_R_init': mc_metrics['RMSE_R_init'][-1] if mc_metrics['RMSE_R_init'] else np.nan,
                # [NEW] Theta diagnostics
                'g_theta': mc_metrics['g_theta'][-1] if mc_metrics['g_theta'] else np.nan,
                'g_theta_eff': mc_metrics['g_theta_eff'][-1] if mc_metrics['g_theta_eff'] else np.nan,
                'accept_rate': mc_metrics['accept_rate'][-1] if mc_metrics['accept_rate'] else np.nan,
                'delta_R': mc_metrics['delta_R'][-1] if mc_metrics['delta_R'] else np.nan,
            })

        def safe_mean(arr):
            return np.mean(arr) if arr else np.nan

        def safe_std(arr):
            return np.std(arr) if len(arr) > 1 else 0.0

        avg_chi = safe_mean(mc_metrics['chi'])
        bcrlb_ref = compute_bcrlb_ref_independent(
            snr_linear, avg_chi, exp_cfg.R, exp_cfg.v_rel
        )

        # [NEW] Compute improvement ratio
        rmse_R_mean = safe_mean(mc_metrics['RMSE_R'])
        rmse_R_init_mean = safe_mean(mc_metrics['RMSE_R_init'])
        improvement_ratio = rmse_R_init_mean / (rmse_R_mean + 1e-12) if rmse_R_mean > 0 else np.nan

        mean_results.append({
            'snr_db': snr_db,
            'gamma_eff': safe_mean(mc_metrics['gamma_eff']),
            'chi': avg_chi,
            'BER_Net': safe_mean(mc_metrics['BER_Net']),
            'BER_Net_std': safe_std(mc_metrics['BER_Net']),
            'BER_LMMSE': safe_mean(mc_metrics['BER_LMMSE']),
            'BER_GAMP': safe_mean(mc_metrics['BER_GAMP']),
            'NMSE_Net': safe_mean(mc_metrics['NMSE_Net']),
            'RMSE_R': rmse_R_mean,
            'RMSE_R_std': safe_std(mc_metrics['RMSE_R']),
            'RMSE_R_init': rmse_R_init_mean,  # [NEW]
            'RMSE_improvement_ratio': improvement_ratio,  # [NEW]
            'RMSE_v': safe_mean(mc_metrics['RMSE_v']),
            'RMSE_a': safe_mean(mc_metrics['RMSE_a']),
            'BCRLB_R_ref': bcrlb_ref['BCRLB_R_ref'],
            'BCRLB_R_analog': bcrlb_ref['BCRLB_R_analog'],
            # [NEW] Theta diagnostics
            'g_theta': safe_mean(mc_metrics['g_theta']),
            'g_theta_eff': safe_mean(mc_metrics['g_theta_eff']),
            'accept_rate': safe_mean(mc_metrics['accept_rate']),
            'delta_R_mean': safe_mean(mc_metrics['delta_R']),
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
        'version': 'v5.0_enhanced_theta',
        'fixes_applied': [
            'noise_var uses actual signal power',
            'fs=10GHz in BCRLB',
            'debug_theta_exact option',
            'Enhanced theta metrics (RMSE_R_init, improvement_ratio)',
            'Theta diagnostics (g_theta, accept_rate, delta_R)',
        ]
    }
    with open(out_dir / "config.json", 'w') as f:
        json.dump(config_data, f, indent=2, default=str)

    generate_plots(df_mean, out_dir)
    generate_theta_plots(df_mean, out_dir)  # [NEW]

    print(f"  [Results] Saved to {out_dir}")

    unique_seeds = df_raw['mc_seed'].nunique()
    total_trials = len(df_raw)
    print(f"  [MC Check] Unique seeds: {unique_seeds}/{total_trials}")

    # [NEW] Print theta summary
    if 'RMSE_improvement_ratio' in df_mean.columns:
        avg_improvement = df_mean['RMSE_improvement_ratio'].mean()
        print(f"  [Theta] Avg RMSE improvement ratio: {avg_improvement:.2f}x")

    return df_mean


# =============================================================================
# 8. Plot Generation
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

    # [NEW] Plot RMSE_R_init for comparison
    if "RMSE_R_init" in df.columns and not df["RMSE_R_init"].isna().all():
        ax2.semilogy(df["snr_db"], df["RMSE_R_init"], 'g--^', alpha=0.7,
                     label=r'$\mathrm{RMSE}(\theta_{init})$')

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


def generate_theta_plots(df: pd.DataFrame, out_dir: Path):
    """
    [NEW] Generate theta-specific diagnostic plots.
    """
    # Plot 5: g_theta and accept_rate vs SNR
    fig5, ax5a = plt.subplots(figsize=(7, 5))

    if "g_theta" in df.columns and not df["g_theta"].isna().all():
        ax5a.plot(df["snr_db"], df["g_theta"], 'b-o', label=r'$g_\theta$ (gate)')
        ax5a.set_ylabel(r'$g_\theta$', color='b')
        ax5a.tick_params(axis='y', labelcolor='b')
        ax5a.set_ylim([0, 1])

    ax5b = ax5a.twinx()
    if "accept_rate" in df.columns and not df["accept_rate"].isna().all():
        ax5b.plot(df["snr_db"], df["accept_rate"], 'r--s', label='Accept Rate')
        ax5b.set_ylabel('Accept Rate', color='r')
        ax5b.tick_params(axis='y', labelcolor='r')
        ax5b.set_ylim([0, 1])

    ax5a.set_xlabel('SNR (dB)')
    ax5a.grid(True, linestyle='-', alpha=0.3)
    ax5a.set_title('Theta Update Diagnostics')

    lines1, labels1 = ax5a.get_legend_handles_labels()
    lines2, labels2 = ax5b.get_legend_handles_labels()
    ax5a.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

    fig5.tight_layout()
    fig5.savefig(out_dir / "fig_theta_diag.png", dpi=300, bbox_inches='tight')
    fig5.savefig(out_dir / "fig_theta_diag.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig5)

    # Plot 6: Improvement ratio vs SNR
    fig6, ax6 = plt.subplots(figsize=(7, 5))

    if "RMSE_improvement_ratio" in df.columns and not df["RMSE_improvement_ratio"].isna().all():
        ax6.plot(df["snr_db"], df["RMSE_improvement_ratio"], 'g-o', linewidth=2)
        ax6.axhline(y=1.0, color='k', linestyle='--', alpha=0.7, label='No Improvement')
        ax6.fill_between(df["snr_db"], 1.0, df["RMSE_improvement_ratio"],
                         where=df["RMSE_improvement_ratio"] > 1,
                         alpha=0.3, color='green', label='Improved')
        ax6.fill_between(df["snr_db"], 1.0, df["RMSE_improvement_ratio"],
                         where=df["RMSE_improvement_ratio"] < 1,
                         alpha=0.3, color='red', label='Degraded')

    ax6.set_xlabel('SNR (dB)')
    ax6.set_ylabel('RMSE Improvement Ratio (init/hat)')
    ax6.set_title('Theta Estimation Improvement')
    ax6.legend(loc='best')
    ax6.grid(True, linestyle='-', alpha=0.3)

    fig6.tight_layout()
    fig6.savefig(out_dir / "fig_theta_improvement.png", dpi=300, bbox_inches='tight')
    fig6.savefig(out_dir / "fig_theta_improvement.pdf", format='pdf', bbox_inches='tight')
    plt.close(fig6)

    print(f"  [Plots] Saved 2 theta diagnostic figures to {out_dir}")


# =============================================================================
# 9. Experiment Configurations
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
# 10. Main Entry Point
# =============================================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(description="P4 Experiments (Expert v5.0 Enhanced)")
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

    # GPU Setup
    device = setup_device(preferred_device=args.device, verbose=True)
    device_str = str(device)

    print("\n" + "=" * 60)
    print("P4 Experiments - Expert v5.0 (Enhanced Theta Metrics)")
    print("=" * 60)
    print(f"Device: {device_str}")
    if args.debug_theta:
        print("⚠️  DEBUG MODE: theta_init = theta_true (NO NOISE)")
    print("=" * 60)

    # Clear GPU memory
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

    # Auto-adjust batch size
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

    # Run all experiments
    all_scene_results = {}
    for exp in experiments:
        df_result = run_single_scenario(exp, model, device_str, ckpt_tag, args.out)
        all_scene_results[exp.scene_id] = df_result
        clear_gpu_memory()

    print("\n[Done] All experiments completed.")
    memory_summary()


if __name__ == "__main__":
    main()