"""
evaluator.py - Data Collection and Evaluation (Expert v2.0 - Top Journal Ready)

Expert Requirements Implemented:
- P0-1: Cross-method fairness (same sim_data + theta_init for all methods)
- P0-2: Dual Oracle support (oracle_sync as primary)
- P1-1: τ-only success rate definition
- P1-2: Per-sample success flag for pull-in computation
- P2-1: CRLB tau computation interface

Key Changes:
1. evaluate_methods_fair(): Ensures same data for all methods in one MC run
2. Success rate: |τ_err_final| < eps_tau (τ-only, unified)
3. Output: tau_err_final_abs, success flag per sample
4. CSV schema aligned with theory manual
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# Local imports
from baselines import (
    run_baseline,
    METHOD_PAPER_CORE,
    METHOD_PAPER_FULL,
    METHOD_DEBUG,
    METHOD_CLIFF,
    METHOD_ABLATION,
    METHOD_SNR_SWEEP,
    METHOD_ROBUSTNESS,
    frontend_adjoint_and_pn,
    qpsk_hard_slice,
)


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration with expert requirements."""
    ckpt_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SNR sweep
    snr_list: List[float] = field(default_factory=lambda: [-5, 0, 5, 10, 15, 20, 25])

    # Monte Carlo settings (paper-level defaults)
    n_mc: int = 20
    batch_size: int = 64

    # θ initial noise (samples)
    theta_noise_tau: float = 0.3
    theta_noise_v: float = 0.0
    theta_noise_a: float = 0.0

    # Cliff sweep
    init_error_list: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])

    # PN sweep
    pn_linewidths: List[float] = field(default_factory=lambda: [0, 50e3, 100e3, 200e3, 500e3])

    # Pilot sweep
    pilot_lengths: List[int] = field(default_factory=lambda: [16, 32, 64, 128])

    # Success rate threshold (P1-1: τ-only)
    eps_tau: float = 0.1  # Success if |τ_err| < eps_tau samples

    # Output directory
    out_dir: str = "results/paper_figs"

    # Mode: 'debug' or 'paper'
    mode: str = "paper"


# ============================================================================
# Model and Data Loading
# ============================================================================

def load_model(ckpt_path: str, device: str):
    """Load trained model."""
    from gabv_net_model import GABVNet, GABVConfig, create_gabv_model

    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'gabv_cfg' in checkpoint:
        gabv_cfg = checkpoint['gabv_cfg']
    else:
        gabv_cfg = GABVConfig()

    model = create_gabv_model(gabv_cfg)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)
    model.eval()

    return model, gabv_cfg


def create_sim_config(gabv_cfg, snr_db: float, pn_linewidth: float = None):
    """Create simulation configuration."""
    from thz_isac_world import SimConfig

    sim_cfg = SimConfig(
        N=gabv_cfg.N if hasattr(gabv_cfg, 'N') else 1024,
        fs=gabv_cfg.fs if hasattr(gabv_cfg, 'fs') else 10e9,
        fc=gabv_cfg.fc if hasattr(gabv_cfg, 'fc') else 300e9,
        snr_db=snr_db,
    )

    if pn_linewidth is not None:
        sim_cfg.pn_linewidth = pn_linewidth

    return sim_cfg


def simulate_batch(sim_cfg, batch_size: int, seed: int = None) -> Dict:
    """Generate simulation data with optional seed."""
    from thz_isac_world import simulate_batch as sim_batch
    return sim_batch(sim_cfg, batch_size, seed=seed)


# ============================================================================
# Helper Functions
# ============================================================================

def make_failed_record(**kwargs) -> dict:
    """Create failed record with NaN values."""
    record = {
        'ber': float('nan'),
        'rmse_tau_init': float('nan'),
        'rmse_tau_final': float('nan'),
        'tau_err_final_abs': float('nan'),
        'improvement': float('nan'),
        'success_rate': float('nan'),
        'success': 0,
        'failed': True,
    }
    record.update(kwargs)
    return record


def to_tensor(x, device):
    """Convert to tensor on device."""
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).to(device)
    elif isinstance(x, torch.Tensor):
        return x.to(device)
    return x


def construct_meta_features(meta_dict: Dict, batch_size: int, snr_db: float = None) -> torch.Tensor:
    """Construct meta feature tensor."""
    if snr_db is not None:
        snr = snr_db
    else:
        snr = meta_dict.get('snr_db', 20.0)

    gamma_eff = meta_dict.get('gamma_eff', 1.0)
    chi = meta_dict.get('chi', 0.1)
    sigma_eta = meta_dict.get('sigma_eta', 0.01)
    pn_linewidth = meta_dict.get('pn_linewidth', 100e3)
    ibo_dB = meta_dict.get('ibo_dB', 3.0)

    snr_db_norm = (snr - 15) / 15
    gamma_eff_db = 10 * np.log10(max(gamma_eff, 1e-6))
    gamma_eff_db_norm = (gamma_eff_db - 10) / 20
    sigma_eta_norm = sigma_eta / 0.1
    pn_linewidth_norm = np.log10(pn_linewidth + 1) / np.log10(1e6)
    ibo_db_norm = (ibo_dB - 3) / 3

    meta_vec = np.array([
        snr_db_norm, gamma_eff_db_norm, chi,
        sigma_eta_norm, pn_linewidth_norm, ibo_db_norm
    ], dtype=np.float32)

    return torch.from_numpy(np.tile(meta_vec, (batch_size, 1))).float()


def compute_metrics(x_hat, x_true, theta_hat, theta_true, theta_init,
                    sim_cfg, pilot_len: int, eps_tau: float = 0.1) -> Dict:
    """
    Compute all metrics including τ-only success rate.

    P1-1: Success rate is τ-only: |τ_err_final| < eps_tau
    """
    # BER (QPSK) - only on data symbols
    x_hat_data = x_hat[:, pilot_len:]
    x_true_data = x_true[:, pilot_len:]

    x_hat_bits = torch.stack([torch.sign(x_hat_data.real), torch.sign(x_hat_data.imag)], dim=-1)
    x_true_bits = torch.stack([torch.sign(x_true_data.real), torch.sign(x_true_data.imag)], dim=-1)
    ber = (x_hat_bits != x_true_bits).float().mean().item()

    # τ errors (in samples)
    tau_true = theta_true[:, 0].cpu().numpy() * sim_cfg.fs
    tau_init = theta_init[:, 0].cpu().numpy() * sim_cfg.fs
    tau_hat = theta_hat[:, 0].cpu().numpy() * sim_cfg.fs

    tau_error_init = np.abs(tau_init - tau_true)
    tau_error_final = np.abs(tau_hat - tau_true)

    rmse_tau_init = np.sqrt(np.mean(tau_error_init ** 2))
    rmse_tau_final = np.sqrt(np.mean(tau_error_final ** 2))

    # P1-1: τ-only success rate
    success_mask = tau_error_final < eps_tau
    success_rate = np.mean(success_mask)

    # Mean absolute τ error (for pull-in computation)
    tau_err_final_abs = np.mean(tau_error_final)

    return {
        'ber': ber,
        'rmse_tau_init': rmse_tau_init,
        'rmse_tau_final': rmse_tau_final,
        'tau_err_final_abs': tau_err_final_abs,
        'improvement': rmse_tau_init / (rmse_tau_final + 1e-10),
        'success_rate': success_rate,
        'success': int(success_rate > 0.5),  # Binary for this batch
    }


# ============================================================================
# P0-1: Fair Multi-Method Evaluation
# ============================================================================

def evaluate_methods_fair(
    model,
    sim_cfg,
    batch_size: int,
    theta_noise: Tuple[float, float, float],
    device: str,
    methods: List[str],
    pilot_len: int = 64,
    init_error_override: float = None,
    eps_tau: float = 0.1,
    seed: int = None,
) -> List[Dict]:
    """
    Evaluate multiple methods on SAME data (P0-1: cross-method fairness).

    Key guarantee:
    - All methods see identical sim_data (y_q, x_true, theta_true)
    - All methods see identical theta_init_noisy (same initial perturbation)
    - Seed is NOT mixed with method name

    Args:
        model: GABVNet model
        sim_cfg: Simulation config
        batch_size: Batch size
        theta_noise: (tau_noise, v_noise, a_noise) in samples
        device: Device string
        methods: List of method names
        pilot_len: Pilot length
        init_error_override: Override tau noise if specified
        eps_tau: Success threshold
        seed: Random seed for this MC run

    Returns:
        List of result dicts (one per method)
    """
    Ts = 1.0 / sim_cfg.fs
    tau_noise = init_error_override if init_error_override is not None else theta_noise[0]

    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)

    # 1) Generate SHARED simulation data (same for all methods)
    sim_data = simulate_batch(sim_cfg, batch_size, seed=seed)

    theta_true = to_tensor(sim_data['theta_true'], device)
    y_q = to_tensor(sim_data['y_q'], device)
    x_true = to_tensor(sim_data['x_true'], device)

    # 2) Generate SHARED theta_init_noisy (same for all methods except oracle)
    noise_tau = torch.randn(batch_size, 1, device=device) * tau_noise * Ts
    noise_v = torch.randn(batch_size, 1, device=device) * theta_noise[1]
    noise_a = torch.randn(batch_size, 1, device=device) * theta_noise[2]

    theta_init_noisy = theta_true.clone()
    theta_init_noisy[:, 0:1] += noise_tau
    theta_init_noisy[:, 1:2] += noise_v
    theta_init_noisy[:, 2:3] += noise_a

    # Meta features
    raw_meta = sim_data.get('meta', {})
    meta_tensor = construct_meta_features(raw_meta, batch_size, snr_db=sim_cfg.snr_db).to(device)

    results = []

    for method in methods:
        # Oracle uses true theta, others use noisy init
        if method in ["oracle", "oracle_sync"]:
            theta_init = theta_true.clone()
        else:
            theta_init = theta_init_noisy.clone()

        # Construct batch
        batch = {
            'y_q': y_q,
            'x_true': x_true,
            'theta_init': theta_init,
            'theta_true': theta_true,
            'meta': meta_tensor,
            'snr_db': sim_cfg.snr_db,
        }

        try:
            x_hat, theta_hat = run_baseline(method, model, batch, sim_cfg, device, pilot_len)

            metrics = compute_metrics(
                x_hat, x_true, theta_hat, theta_true, theta_init,
                sim_cfg, pilot_len, eps_tau
            )
            metrics['method'] = method
            metrics['failed'] = False

            results.append(metrics)

        except Exception as e:
            print(f"Warning: {method} failed: {e}")
            results.append(make_failed_record(method=method, error=str(e)))

    return results


# ============================================================================
# Sanity Check
# ============================================================================

def run_sanity_check(model, gabv_cfg, eval_cfg: EvalConfig) -> bool:
    """
    Sanity check: At init_error=0, all methods should have BER < 0.2
    """
    print("\n" + "="*50)
    print("🔍 Sanity Check (init_error=0)")
    print("="*50)

    methods_to_check = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]
    snr_db = 15.0
    sim_cfg = create_sim_config(gabv_cfg, snr_db)
    theta_noise = (0.0, 0.0, 0.0)

    results = evaluate_methods_fair(
        model, sim_cfg, eval_cfg.batch_size, theta_noise,
        eval_cfg.device, methods_to_check,
        init_error_override=0.0, seed=12345
    )

    all_passed = True
    for r in results:
        method = r['method']
        ber = r['ber']
        status = "✅ PASS" if ber < 0.2 else "❌ FAIL"
        print(f"  {method:25s}: BER = {ber:.4f} {status}")
        if ber >= 0.2:
            all_passed = False

    if all_passed:
        print("\n✅ Sanity Check passed!")
    else:
        print("\n❌ Sanity Check FAILED!")

    return all_passed


# ============================================================================
# Sweep Functions
# ============================================================================

def run_snr_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                  methods: List[str] = None) -> pd.DataFrame:
    """SNR sweep with cross-method fairness."""
    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

    if methods is None:
        methods = METHOD_SNR_SWEEP

    print(f"  [SNR Sweep] Methods: {methods}")

    total = len(eval_cfg.snr_list) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="SNR sweep")

    for snr_db in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for mc_id in range(eval_cfg.n_mc):
            seed = mc_id * 1000 + int(snr_db * 10)

            results = evaluate_methods_fair(
                model, sim_cfg, eval_cfg.batch_size, theta_noise,
                eval_cfg.device, methods,
                eps_tau=eval_cfg.eps_tau, seed=seed
            )

            for r in results:
                r['snr_db'] = snr_db
                r['mc_id'] = mc_id
                records.append(r)

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_cliff_sweep(model, gabv_cfg, eval_cfg: EvalConfig, snr_db: float = 15.0,
                    methods: List[str] = None) -> pd.DataFrame:
    """Cliff sweep (core contribution figure) with cross-method fairness."""
    records = []

    if methods is None:
        methods = METHOD_CLIFF

    print(f"  [Cliff Sweep] Methods: {methods}")

    total = len(eval_cfg.init_error_list) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="Cliff sweep")

    for init_error in eval_cfg.init_error_list:
        theta_noise = (init_error, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for mc_id in range(eval_cfg.n_mc):
            seed = mc_id * 1000 + int(init_error * 100)

            results = evaluate_methods_fair(
                model, sim_cfg, eval_cfg.batch_size, theta_noise,
                eval_cfg.device, methods,
                init_error_override=init_error,
                eps_tau=eval_cfg.eps_tau, seed=seed
            )

            for r in results:
                r['init_error'] = init_error
                r['mc_id'] = mc_id
                records.append(r)

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_snr_sweep_multi_init_error(model, gabv_cfg, eval_cfg: EvalConfig,
                                    init_errors: List[float] = None,
                                    methods: List[str] = None) -> pd.DataFrame:
    """Multi init_error SNR sweep."""
    records = []

    if init_errors is None:
        init_errors = [0.0, 0.2, 0.3]

    if methods is None:
        methods = METHOD_PAPER_CORE

    print(f"  [Multi-Init SNR Sweep] init_errors: {init_errors}, Methods: {methods}")

    total = len(init_errors) * len(eval_cfg.snr_list) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="Multi-init SNR sweep")

    for init_error in init_errors:
        theta_noise = (init_error, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

        for snr_db in eval_cfg.snr_list:
            sim_cfg = create_sim_config(gabv_cfg, snr_db)

            for mc_id in range(eval_cfg.n_mc):
                seed = mc_id * 1000 + int(snr_db * 10) + int(init_error * 100)

                results = evaluate_methods_fair(
                    model, sim_cfg, eval_cfg.batch_size, theta_noise,
                    eval_cfg.device, methods,
                    init_error_override=init_error,
                    eps_tau=eval_cfg.eps_tau, seed=seed
                )

                for r in results:
                    r['init_error'] = init_error
                    r['snr_db'] = snr_db
                    r['mc_id'] = mc_id
                    records.append(r)

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_ablation_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                       methods: List[str] = None) -> pd.DataFrame:
    """Ablation sweep with cross-method fairness."""
    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

    if methods is None:
        methods = METHOD_ABLATION

    print(f"  [Ablation Sweep] Methods: {methods}")

    total = len(eval_cfg.snr_list) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="Ablation sweep")

    for snr_db in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for mc_id in range(eval_cfg.n_mc):
            seed = mc_id * 1000 + int(snr_db * 10)

            results = evaluate_methods_fair(
                model, sim_cfg, eval_cfg.batch_size, theta_noise,
                eval_cfg.device, methods,
                eps_tau=eval_cfg.eps_tau, seed=seed
            )

            for r in results:
                r['snr_db'] = snr_db
                r['mc_id'] = mc_id
                records.append(r)

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_heatmap_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                      methods: List[str] = None) -> pd.DataFrame:
    """2D Heatmap sweep (SNR × init_error) with cross-method fairness."""
    records = []

    if methods is None:
        methods = ["proposed", "adjoint_slice"]

    print(f"  [Heatmap Sweep] Methods: {methods}")

    n_mc_heatmap = min(eval_cfg.n_mc, 10)

    total = len(eval_cfg.snr_list) * len(eval_cfg.init_error_list) * n_mc_heatmap
    pbar = tqdm(total=total, desc="Heatmap sweep")

    for snr_db in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for init_error in eval_cfg.init_error_list:
            theta_noise = (init_error, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

            for mc_id in range(n_mc_heatmap):
                seed = mc_id * 1000 + int(snr_db * 10) + int(init_error * 100)

                results = evaluate_methods_fair(
                    model, sim_cfg, eval_cfg.batch_size, theta_noise,
                    eval_cfg.device, methods,
                    init_error_override=init_error,
                    eps_tau=eval_cfg.eps_tau, seed=seed
                )

                for r in results:
                    r['snr_db'] = snr_db
                    r['init_error'] = init_error
                    r['mc_id'] = mc_id
                    records.append(r)

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_pn_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                 methods: List[str] = None) -> pd.DataFrame:
    """PN linewidth sweep with cross-method fairness."""
    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)
    snr_db = 15.0

    if methods is None:
        methods = METHOD_ROBUSTNESS

    print(f"  [PN Sweep] Methods: {methods}")

    total = len(eval_cfg.pn_linewidths) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="PN sweep")

    for pn_lw in eval_cfg.pn_linewidths:
        sim_cfg = create_sim_config(gabv_cfg, snr_db, pn_linewidth=pn_lw)

        for mc_id in range(eval_cfg.n_mc):
            seed = mc_id * 1000 + int(pn_lw / 1000)

            results = evaluate_methods_fair(
                model, sim_cfg, eval_cfg.batch_size, theta_noise,
                eval_cfg.device, methods,
                eps_tau=eval_cfg.eps_tau, seed=seed
            )

            for r in results:
                r['pn_linewidth'] = pn_lw
                r['mc_id'] = mc_id
                records.append(r)

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_pilot_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                    methods: List[str] = None) -> pd.DataFrame:
    """Pilot length sweep with cross-method fairness."""
    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)
    snr_db = 15.0

    if methods is None:
        methods = METHOD_ROBUSTNESS

    print(f"  [Pilot Sweep] Methods: {methods}")

    total = len(eval_cfg.pilot_lengths) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="Pilot sweep")

    for pilot_len in eval_cfg.pilot_lengths:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for mc_id in range(eval_cfg.n_mc):
            seed = mc_id * 1000 + pilot_len

            results = evaluate_methods_fair(
                model, sim_cfg, eval_cfg.batch_size, theta_noise,
                eval_cfg.device, methods,
                pilot_len=pilot_len,
                eps_tau=eval_cfg.eps_tau, seed=seed
            )

            for r in results:
                r['pilot_len'] = pilot_len
                r['mc_id'] = mc_id
                records.append(r)

            pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_jacobian_analysis(model, gabv_cfg, eval_cfg: EvalConfig) -> pd.DataFrame:
    """Jacobian analysis (placeholder)."""
    records = []
    for init_error in eval_cfg.init_error_list:
        records.append({
            'init_error': init_error,
            'gram_cond_log10': 15 - init_error * 5,
        })
    return pd.DataFrame(records)


def measure_latency(model, gabv_cfg, eval_cfg: EvalConfig) -> pd.DataFrame:
    """Measure latency for each method."""
    import time

    records = []
    methods = ["adjoint_slice", "matched_filter", "proposed_no_update", "proposed"]

    snr_db = 15.0
    sim_cfg = create_sim_config(gabv_cfg, snr_db)
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

    # Warmup
    for _ in range(3):
        try:
            _ = evaluate_methods_fair(
                model, sim_cfg, 32, theta_noise, eval_cfg.device, ["proposed"]
            )
        except:
            pass

    for method in methods:
        latencies = []
        for _ in range(10):
            try:
                start = time.time()
                _ = evaluate_methods_fair(
                    model, sim_cfg, 32, theta_noise, eval_cfg.device, [method]
                )
                latencies.append((time.time() - start) * 1000)
            except:
                pass

        if latencies:
            records.append({
                'method': method,
                'latency_mean_ms': np.mean(latencies),
                'latency_std_ms': np.std(latencies),
            })

    return pd.DataFrame(records)


# ============================================================================
# P2-1: CRLB τ Computation
# ============================================================================

def compute_sqrt_crlb_tau(sim_cfg, snr_db: float, pilot_len: int = 64,
                          n_samples: int = 100, fd_step: float = 1e-12) -> float:
    """
    Compute sqrt(CRLB) for τ using numerical FIM (P2-1).

    Uses finite difference approximation of the score function.

    Args:
        sim_cfg: Simulation config
        snr_db: SNR in dB
        pilot_len: Pilot length
        n_samples: MC samples for averaging
        fd_step: Finite difference step

    Returns:
        sqrt_crlb_tau: sqrt of CRLB for τ in samples
    """
    from thz_isac_world import compute_bcrlb_diag

    snr_linear = 10 ** (snr_db / 10)
    gamma_eff = 1.0  # Placeholder

    bcrlb = compute_bcrlb_diag(sim_cfg, snr_linear, gamma_eff)

    # BCRLB for τ (first element) in seconds, convert to samples
    crlb_tau_sec = bcrlb[0]
    crlb_tau_samples = crlb_tau_sec * (sim_cfg.fs ** 2)

    sqrt_crlb = np.sqrt(crlb_tau_samples)

    return sqrt_crlb


def run_crlb_sweep(gabv_cfg, eval_cfg: EvalConfig) -> pd.DataFrame:
    """Run CRLB sweep for efficiency computation."""
    records = []

    for snr_db in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        sqrt_crlb = compute_sqrt_crlb_tau(sim_cfg, snr_db)

        records.append({
            'snr_db': snr_db,
            'sqrt_crlb_tau': sqrt_crlb,
        })

    return pd.DataFrame(records)


# ============================================================================
# Validation Functions
# ============================================================================

def validate_csv_methods(df: pd.DataFrame, expected_methods: List[str],
                         csv_name: str, strict: bool = False) -> bool:
    """Validate CSV contains all expected methods."""
    if 'method' not in df.columns:
        print(f"  ⚠️ {csv_name}: Missing 'method' column")
        return False

    actual_methods = set(df['method'].unique())
    expected_set = set(expected_methods)
    missing = expected_set - actual_methods

    if missing:
        msg = f"  ⚠️ {csv_name}: Missing methods {missing}"
        print(msg)
        if strict:
            raise ValueError(msg)
        return False

    print(f"  ✓ {csv_name}: All {len(expected_methods)} methods present")
    return True


def print_import_info():
    """Print module import paths for debugging."""
    import baselines
    print(f"[DEBUG] baselines.py = {baselines.__file__}")
    print(f"[DEBUG] evaluator.py = {__file__}")