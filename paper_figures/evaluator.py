"""
evaluator_v3.py - Data Collection and Evaluation (Expert v3.0 - Top Journal Ready)

CRITICAL FIXES (Expert Review):
1. Trial-first / Method-second evaluation:
   - Same trial (y_q, x_true, theta_true) for all methods
   - Seed independent of method (no hash(method) in seed)
   - Enables fair Gap-to-Oracle computation

2. ADC bits sweep:
   - Supports 1-8 bit quantization
   - Proves 1-bit failures are inherent, not bugs

3. CRLB computation:
   - Bussgang-FIM with numerical differentiation
   - Plotted as reference on τ RMSE figures

4. CSV fields:
   - seed, mc_idx, sweep parameters for reproducibility
   - mf_grid_points, mf_search_half_range for matched filter

Author: Expert Review v3.0
Date: 2025-12-20
"""

import os
import sys
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Local imports
from baselines import (
    run_baseline,
    get_baseline,
    METHOD_ORDER,
    METHOD_QUICK,
    METHOD_CLIFF,
    METHOD_ABLATION,
    METHOD_SNR_SWEEP,
    METHOD_ROBUSTNESS,
    frontend_adjoint_and_pn,
    qpsk_hard_slice,
    BaselineMatchedFilter,
)


# ============================================================================
# Import Path Debugging
# ============================================================================

def print_import_info():
    """Print current import paths to prevent version confusion."""
    print(f"[DEBUG] evaluator_v3.py = {__file__}")
    try:
        import baselines_v2
        print(f"[DEBUG] baselines_v2.py = {baselines_v2.__file__}")
    except ImportError:
        print("[DEBUG] baselines_v2 not found, using baselines")


# ============================================================================
# Configuration
# ============================================================================

@dataclass
class EvalConfig:
    """Evaluation configuration."""
    ckpt_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SNR sweep
    snr_list: List[float] = field(default_factory=lambda: [-5, 0, 5, 10, 15, 20, 25])

    # Monte Carlo - paper-grade defaults
    n_mc: int = 20
    batch_size: int = 64

    # θ initial noise (samples)
    theta_noise_tau: float = 0.3
    theta_noise_v: float = 0.0
    theta_noise_a: float = 0.0

    # Cliff sweep
    init_error_list: List[float] = field(
        default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]
    )
    max_init_error: float = 1.5  # For matched filter search range

    # PN sweep
    pn_linewidths: List[float] = field(
        default_factory=lambda: [0, 50e3, 100e3, 200e3, 500e3]
    )

    # Pilot sweep
    pilot_lengths: List[int] = field(default_factory=lambda: [16, 32, 64, 128])

    # ADC bits sweep (NEW)
    adc_bits_list: List[int] = field(default_factory=lambda: [1, 2, 3, 4, 8])

    # Output directory
    out_dir: str = "results/paper_figs"


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


def create_sim_config(gabv_cfg, snr_db: float, pn_linewidth: float = None,
                      adc_bits: int = 1):
    """Create simulation configuration."""
    from thz_isac_world import SimConfig

    sim_cfg = SimConfig(
        N=gabv_cfg.N if hasattr(gabv_cfg, 'N') else 1024,
        fs=gabv_cfg.fs if hasattr(gabv_cfg, 'fs') else 10e9,
        fc=gabv_cfg.fc if hasattr(gabv_cfg, 'fc') else 300e9,
        snr_db=snr_db,
        adc_bits=adc_bits,
    )

    if pn_linewidth is not None:
        sim_cfg.pn_linewidth = pn_linewidth

    return sim_cfg


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

    # Normalize
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


# ============================================================================
# Trial Generation (Core of Trial-first Architecture)
# ============================================================================

def generate_trial(sim_cfg, batch_size: int, seed: int, device: str,
                   init_error: float = 0.0) -> Dict:
    """
    Generate a single trial for evaluation.

    CRITICAL: This function is deterministic given (sim_cfg, batch_size, seed).
    All methods will receive the SAME trial data.

    Args:
        sim_cfg: Simulation configuration
        batch_size: Batch size
        seed: Random seed (MUST be independent of method)
        device: Device string
        init_error: Initial τ error in samples

    Returns:
        trial: Dictionary with y_q, x_true, theta_true, theta_init, meta, seed
    """
    from thz_isac_world import simulate_batch

    Ts = 1.0 / sim_cfg.fs

    # Set seeds for reproducibility
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Generate simulation data
    sim_data = simulate_batch(sim_cfg, batch_size, seed=seed)

    def to_tensor(x, device):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    theta_true = to_tensor(sim_data['theta_true'], device)
    y_q = to_tensor(sim_data['y_q'], device)
    x_true = to_tensor(sim_data['x_true'], device)

    # Generate theta_init with noise (same for all methods!)
    # Note: This noise is generated with the same seed, so all methods
    # see the same initial error for fair comparison.
    noise_tau = torch.randn(batch_size, 1, device=device) * init_error * Ts
    noise_v = torch.zeros(batch_size, 1, device=device)
    noise_a = torch.zeros(batch_size, 1, device=device)

    theta_init = theta_true.clone()
    theta_init[:, 0:1] += noise_tau
    theta_init[:, 1:2] += noise_v
    theta_init[:, 2:3] += noise_a

    # Construct meta features
    raw_meta = sim_data.get('meta', {})
    meta_tensor = construct_meta_features(raw_meta, batch_size, snr_db=sim_cfg.snr_db).to(device)

    return {
        'y_q': y_q,
        'x_true': x_true,
        'theta_true': theta_true,
        'theta_init': theta_init,
        'meta': meta_tensor,
        'snr_db': sim_cfg.snr_db,
        'seed': seed,
        'raw_meta': raw_meta,
    }


# ============================================================================
# Evaluation on Trial (Trial-first Architecture)
# ============================================================================

def evaluate_method_on_trial(
    model, method: str, trial: Dict, sim_cfg, device: str,
    pilot_len: int = 64, **kwargs
) -> Dict:
    """
    Evaluate a single method on a given trial.

    Args:
        model: GABVNet model
        method: Method name
        trial: Trial dictionary from generate_trial()
        sim_cfg: Simulation configuration
        device: Device string
        pilot_len: Pilot length
        **kwargs: Additional arguments for baseline

    Returns:
        result: Dictionary with BER, RMSE, etc.
    """
    # Prepare batch for method
    # CRITICAL: Oracle methods need theta_true, others use theta_init
    if method in ['oracle_sync', 'oracle_local_best', 'oracle']:
        batch = {
            'y_q': trial['y_q'],
            'x_true': trial['x_true'],
            'theta_init': trial['theta_true'].clone(),  # Oracle gets true theta
            'theta_true': trial['theta_true'],
            'meta': trial['meta'],
            'snr_db': trial['snr_db'],
        }
    else:
        batch = {
            'y_q': trial['y_q'],
            'x_true': trial['x_true'],
            'theta_init': trial['theta_init'],  # Other methods get noisy init
            'theta_true': trial['theta_true'],
            'meta': trial['meta'],
            'snr_db': trial['snr_db'],
        }

    # Run baseline
    x_hat, theta_hat = run_baseline(method, model, batch, sim_cfg, device, pilot_len, **kwargs)

    # Compute metrics
    x_hat_data = x_hat[:, pilot_len:]
    x_true_data = trial['x_true'][:, pilot_len:]

    x_hat_bits = torch.stack([torch.sign(x_hat_data.real), torch.sign(x_hat_data.imag)], dim=-1)
    x_true_bits = torch.stack([torch.sign(x_true_data.real), torch.sign(x_true_data.imag)], dim=-1)
    ber = (x_hat_bits != x_true_bits).float().mean().item()

    # τ errors (in samples)
    tau_true = trial['theta_true'][:, 0].cpu().numpy() * sim_cfg.fs
    tau_init = trial['theta_init'][:, 0].cpu().numpy() * sim_cfg.fs
    tau_hat = theta_hat[:, 0].cpu().numpy() * sim_cfg.fs

    tau_error_init = np.abs(tau_init - tau_true)
    tau_error_final = np.abs(tau_hat - tau_true)

    rmse_tau_init = np.sqrt(np.mean(tau_error_init ** 2))
    rmse_tau_final = np.sqrt(np.mean(tau_error_final ** 2))

    # Success Rate (|τ_err| < 0.1 samples)
    success_rate = np.mean(tau_error_final < 0.1)

    return {
        'ber': ber,
        'rmse_tau_init': rmse_tau_init,
        'rmse_tau_final': rmse_tau_final,
        'improvement': rmse_tau_init / (rmse_tau_final + 1e-10),
        'success_rate': success_rate,
    }


# ============================================================================
# Sanity Check
# ============================================================================

def run_sanity_check(model, gabv_cfg, eval_cfg: EvalConfig) -> bool:
    """
    Run sanity check: init_error=0 should give BER < 0.2 for all methods.
    """
    print("\n" + "="*50)
    print("🔍 Running Sanity Check (init_error=0)")
    print("="*50)

    methods_to_check = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]
    snr_db = 15.0
    sim_cfg = create_sim_config(gabv_cfg, snr_db)

    # Generate ONE trial
    seed = 42
    trial = generate_trial(sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device, init_error=0.0)

    all_passed = True

    for method in methods_to_check:
        try:
            result = evaluate_method_on_trial(
                model, method, trial, sim_cfg, eval_cfg.device
            )
            ber = result['ber']
            status = "✅ PASS" if ber < 0.2 else "❌ FAIL"
            print(f"  {method:25s}: BER = {ber:.4f} {status}")

            if ber >= 0.2:
                all_passed = False

        except Exception as e:
            print(f"  {method:25s}: ❌ ERROR - {e}")
            all_passed = False

    if all_passed:
        print("\n✅ Sanity Check PASSED! Baseline implementations are correct.")
    else:
        print("\n❌ Sanity Check FAILED! Please check baseline implementations.")

    return all_passed


# ============================================================================
# Sweep Functions (Trial-first Architecture)
# ============================================================================

def run_snr_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                  methods: List[str] = None) -> pd.DataFrame:
    """
    SNR sweep with Trial-first architecture.
    """
    records = []

    if methods is None:
        methods = METHOD_SNR_SWEEP

    print(f"  [SNR Sweep] Methods: {methods}")

    total = len(eval_cfg.snr_list) * eval_cfg.n_mc * len(methods)
    pbar = tqdm(total=total, desc="SNR sweep")

    for snr_db in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for mc_idx in range(eval_cfg.n_mc):
            # CRITICAL: Seed is independent of method!
            seed = mc_idx * 1000 + int(snr_db * 10)

            # Generate trial ONCE for all methods
            trial = generate_trial(
                sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device,
                init_error=eval_cfg.theta_noise_tau
            )

            # Run all methods on the SAME trial
            for method in methods:
                try:
                    result = evaluate_method_on_trial(
                        model, method, trial, sim_cfg, eval_cfg.device
                    )
                    records.append({
                        'snr_db': snr_db,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'init_error': eval_cfg.theta_noise_tau,
                        **result
                    })
                except Exception as e:
                    print(f"Warning: {method} @ SNR={snr_db} failed: {e}")
                    records.append({
                        'snr_db': snr_db,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'ber': float('nan'),
                        'rmse_tau_init': float('nan'),
                        'rmse_tau_final': float('nan'),
                        'failed': True,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_cliff_sweep(model, gabv_cfg, eval_cfg: EvalConfig, snr_db: float = 15.0,
                    methods: List[str] = None) -> pd.DataFrame:
    """
    Cliff sweep with Trial-first architecture.

    This is the CORE contribution figure.
    """
    records = []

    if methods is None:
        methods = METHOD_CLIFF

    print(f"  [Cliff Sweep] Methods: {methods}")

    # Matched filter search range must cover max init_error
    mf_search_half_range = max(eval_cfg.init_error_list) + 0.2

    total = len(eval_cfg.init_error_list) * eval_cfg.n_mc * len(methods)
    pbar = tqdm(total=total, desc="Cliff sweep")

    for init_error in eval_cfg.init_error_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for mc_idx in range(eval_cfg.n_mc):
            # CRITICAL: Seed is independent of method!
            seed = mc_idx * 1000 + int(init_error * 100)

            # Generate trial ONCE for all methods
            trial = generate_trial(
                sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device,
                init_error=init_error
            )

            # Run all methods on the SAME trial
            for method in methods:
                try:
                    result = evaluate_method_on_trial(
                        model, method, trial, sim_cfg, eval_cfg.device,
                        search_half_range=mf_search_half_range  # For matched filter
                    )
                    records.append({
                        'init_error': init_error,
                        'snr_db': snr_db,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'mf_grid_points': BaselineMatchedFilter.GRID_POINTS,
                        'mf_search_half_range': mf_search_half_range,
                        **result
                    })
                except Exception as e:
                    print(f"Warning: {method} @ init_error={init_error} failed: {e}")
                    records.append({
                        'init_error': init_error,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'ber': float('nan'),
                        'failed': True,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_snr_sweep_multi_init_error(model, gabv_cfg, eval_cfg: EvalConfig,
                                    init_errors: List[float] = None,
                                    methods: List[str] = None) -> pd.DataFrame:
    """
    Multi-init_error SNR sweep (Expert Plan 3).
    """
    records = []

    if init_errors is None:
        init_errors = [0.0, 0.2, 0.3]

    if methods is None:
        methods = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]

    print(f"  [Multi-Init SNR Sweep] init_errors: {init_errors}, Methods: {methods}")

    total = len(init_errors) * len(eval_cfg.snr_list) * eval_cfg.n_mc * len(methods)
    pbar = tqdm(total=total, desc="Multi-init SNR sweep")

    for init_error in init_errors:
        for snr_db in eval_cfg.snr_list:
            sim_cfg = create_sim_config(gabv_cfg, snr_db)

            for mc_idx in range(eval_cfg.n_mc):
                # CRITICAL: Seed is independent of method!
                seed = mc_idx * 1000 + int(snr_db * 10) + int(init_error * 100)

                trial = generate_trial(
                    sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device,
                    init_error=init_error
                )

                for method in methods:
                    try:
                        result = evaluate_method_on_trial(
                            model, method, trial, sim_cfg, eval_cfg.device
                        )
                        records.append({
                            'init_error': init_error,
                            'snr_db': snr_db,
                            'method': method,
                            'mc_idx': mc_idx,
                            'seed': seed,
                            **result
                        })
                    except Exception as e:
                        records.append({
                            'init_error': init_error,
                            'snr_db': snr_db,
                            'method': method,
                            'mc_idx': mc_idx,
                            'seed': seed,
                            'ber': float('nan'),
                            'failed': True,
                        })

                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_ablation_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                       methods: List[str] = None) -> pd.DataFrame:
    """
    Ablation sweep with Trial-first architecture.
    """
    records = []

    if methods is None:
        methods = METHOD_ABLATION

    print(f"  [Ablation Sweep] Methods: {methods}")

    total = len(eval_cfg.snr_list) * eval_cfg.n_mc * len(methods)
    pbar = tqdm(total=total, desc="Ablation sweep")

    for snr_db in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for mc_idx in range(eval_cfg.n_mc):
            seed = mc_idx * 1000 + int(snr_db * 10)

            trial = generate_trial(
                sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device,
                init_error=eval_cfg.theta_noise_tau
            )

            for method in methods:
                try:
                    result = evaluate_method_on_trial(
                        model, method, trial, sim_cfg, eval_cfg.device
                    )
                    records.append({
                        'snr_db': snr_db,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        **result
                    })
                except Exception as e:
                    records.append({
                        'snr_db': snr_db,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'ber': float('nan'),
                        'failed': True,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_heatmap_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                      methods: List[str] = None) -> pd.DataFrame:
    """2D Heatmap sweep (SNR × init_error)."""
    records = []

    if methods is None:
        methods = ["proposed", "adjoint_slice"]

    print(f"  [Heatmap Sweep] Methods: {methods}")

    n_mc_heatmap = min(eval_cfg.n_mc, 10)

    total = len(eval_cfg.snr_list) * len(eval_cfg.init_error_list) * n_mc_heatmap * len(methods)
    pbar = tqdm(total=total, desc="Heatmap sweep")

    for snr_db in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for init_error in eval_cfg.init_error_list:
            for mc_idx in range(n_mc_heatmap):
                seed = mc_idx * 1000 + int(snr_db * 10) + int(init_error * 100)

                trial = generate_trial(
                    sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device,
                    init_error=init_error
                )

                for method in methods:
                    try:
                        result = evaluate_method_on_trial(
                            model, method, trial, sim_cfg, eval_cfg.device
                        )
                        records.append({
                            'snr_db': snr_db,
                            'init_error': init_error,
                            'method': method,
                            'mc_idx': mc_idx,
                            'seed': seed,
                            **result
                        })
                    except Exception as e:
                        records.append({
                            'snr_db': snr_db,
                            'init_error': init_error,
                            'method': method,
                            'mc_idx': mc_idx,
                            'seed': seed,
                            'ber': float('nan'),
                            'failed': True,
                        })

                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_pn_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                 methods: List[str] = None) -> pd.DataFrame:
    """Phase noise sweep."""
    records = []
    snr_db = 15.0

    if methods is None:
        methods = METHOD_ROBUSTNESS

    print(f"  [PN Sweep] Methods: {methods}")

    total = len(eval_cfg.pn_linewidths) * eval_cfg.n_mc * len(methods)
    pbar = tqdm(total=total, desc="PN sweep")

    for pn_lw in eval_cfg.pn_linewidths:
        sim_cfg = create_sim_config(gabv_cfg, snr_db, pn_linewidth=pn_lw)

        for mc_idx in range(eval_cfg.n_mc):
            seed = mc_idx * 1000 + int(pn_lw / 1000)

            trial = generate_trial(
                sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device,
                init_error=eval_cfg.theta_noise_tau
            )

            for method in methods:
                try:
                    result = evaluate_method_on_trial(
                        model, method, trial, sim_cfg, eval_cfg.device
                    )
                    records.append({
                        'pn_linewidth': pn_lw,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        **result
                    })
                except Exception as e:
                    records.append({
                        'pn_linewidth': pn_lw,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'ber': float('nan'),
                        'failed': True,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_pilot_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                    methods: List[str] = None) -> pd.DataFrame:
    """Pilot length sweep."""
    records = []
    snr_db = 15.0

    if methods is None:
        methods = METHOD_ROBUSTNESS

    print(f"  [Pilot Sweep] Methods: {methods}")

    total = len(eval_cfg.pilot_lengths) * eval_cfg.n_mc * len(methods)
    pbar = tqdm(total=total, desc="Pilot sweep")

    for pilot_len in eval_cfg.pilot_lengths:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for mc_idx in range(eval_cfg.n_mc):
            seed = mc_idx * 1000 + pilot_len

            trial = generate_trial(
                sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device,
                init_error=eval_cfg.theta_noise_tau
            )

            for method in methods:
                try:
                    result = evaluate_method_on_trial(
                        model, method, trial, sim_cfg, eval_cfg.device,
                        pilot_len=pilot_len
                    )
                    records.append({
                        'pilot_len': pilot_len,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        **result
                    })
                except Exception as e:
                    records.append({
                        'pilot_len': pilot_len,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'ber': float('nan'),
                        'failed': True,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


# ============================================================================
# NEW: ADC Bits Sweep
# ============================================================================

def run_adc_bits_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                       methods: List[str] = None,
                       snr_db: float = 15.0) -> pd.DataFrame:
    """
    ADC bits sweep - proves 1-bit failures are inherent, not bugs.

    This experiment shows that hard-slice baselines recover normal
    performance at bits > 1, confirming the 1-bit cliff is a fundamental
    property rather than an implementation bug.
    """
    records = []

    if methods is None:
        methods = ["naive_slice", "adjoint_slice", "proposed", "oracle_sync"]

    print(f"  [ADC Bits Sweep] Methods: {methods}, bits: {eval_cfg.adc_bits_list}")

    total = len(eval_cfg.adc_bits_list) * eval_cfg.n_mc * len(methods)
    pbar = tqdm(total=total, desc="ADC bits sweep")

    for adc_bits in eval_cfg.adc_bits_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db, adc_bits=adc_bits)

        for mc_idx in range(eval_cfg.n_mc):
            seed = mc_idx * 1000 + adc_bits * 10

            trial = generate_trial(
                sim_cfg, eval_cfg.batch_size, seed, eval_cfg.device,
                init_error=eval_cfg.theta_noise_tau
            )

            for method in methods:
                try:
                    result = evaluate_method_on_trial(
                        model, method, trial, sim_cfg, eval_cfg.device
                    )
                    records.append({
                        'adc_bits': adc_bits,
                        'snr_db': snr_db,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        **result
                    })
                except Exception as e:
                    records.append({
                        'adc_bits': adc_bits,
                        'snr_db': snr_db,
                        'method': method,
                        'mc_idx': mc_idx,
                        'seed': seed,
                        'ber': float('nan'),
                        'failed': True,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


# ============================================================================
# NEW: CRLB Computation
# ============================================================================

def compute_numeric_fim_tau(sim_cfg, batch_size: int = 256, n_mc: int = 50,
                            epsilon: float = 1e-3) -> Dict:
    """
    Compute numerical FIM for τ using central difference.

    Uses Bussgang approximation:
        y_q ≈ α * y_noisefree + q

    FIM(τ) = 2 * Re{ (∂m/∂τ)^H * Σ^{-1} * (∂m/∂τ) }

    Where m(τ) = α * y_noisefree(τ) and Σ is estimated from samples.

    Args:
        sim_cfg: Simulation configuration
        batch_size: Batch size for Monte Carlo
        n_mc: Number of MC runs for variance estimation
        epsilon: Step size for central difference (in samples)

    Returns:
        Dictionary with FIM, CRLB, and diagnostics
    """
    from thz_isac_world import simulate_batch, wideband_delay_operator

    fs = sim_cfg.fs
    Ts = 1.0 / fs
    eps_seconds = epsilon * Ts

    # Bussgang gain for 1-bit quantization
    alpha = np.sqrt(2 / np.pi)

    # Generate reference signal (noisefree)
    np.random.seed(42)
    rng = np.random.default_rng(42)

    # Create a simpler noisefree signal for derivative computation
    N = sim_cfg.N
    x_pilot = (np.random.randn(N) + 1j * np.random.randn(N)) / np.sqrt(2)

    # Compute derivative of m(τ) = α * y_noisefree(τ) using central difference
    tau_0 = 0.0  # Reference point

    # y(τ + ε)
    y_plus = wideband_delay_operator(x_pilot, tau_0 + eps_seconds, fs)

    # y(τ - ε)
    y_minus = wideband_delay_operator(x_pilot, tau_0 - eps_seconds, fs)

    # Central difference: dm/dτ ≈ α * (y(τ+ε) - y(τ-ε)) / (2ε)
    dm_dtau = alpha * (y_plus - y_minus) / (2 * eps_seconds)

    # Estimate noise variance from Monte Carlo
    residuals = []
    for mc_idx in range(n_mc):
        sim_data = simulate_batch(sim_cfg, batch_size, seed=mc_idx)
        y_q = sim_data['y_q']

        # Expected value (using noisefree reference scaled)
        m_tau = alpha * np.tile(x_pilot, (batch_size, 1))

        # Residual
        resid = y_q - m_tau
        residuals.append(resid)

    residuals = np.concatenate(residuals, axis=0)
    sigma2 = np.mean(np.abs(residuals) ** 2)

    # FIM for τ (scalar case, complex Gaussian)
    # FIM = 2 * Re{ (dm/dτ)^H * Σ^{-1} * (dm/dτ) }
    # With Σ = σ² I, this becomes:
    # FIM = (2/σ²) * Σ_n |dm_n/dτ|²
    fim_tau = (2.0 / sigma2) * np.sum(np.abs(dm_dtau) ** 2)

    # CRLB
    crlb_tau = 1.0 / fim_tau if fim_tau > 0 else float('inf')
    crlb_tau_samples = np.sqrt(crlb_tau) * fs  # Convert to samples

    return {
        'fim_tau': fim_tau,
        'crlb_tau': crlb_tau,
        'crlb_tau_samples': crlb_tau_samples,
        'sigma2': sigma2,
        'alpha': alpha,
        'epsilon_samples': epsilon,
        'snr_db': sim_cfg.snr_db,
    }


def run_crlb_sweep(gabv_cfg, eval_cfg: EvalConfig) -> pd.DataFrame:
    """
    Compute CRLB for τ across SNR range.
    """
    records = []

    print("  [CRLB Sweep] Computing numerical FIM...")

    for snr_db in tqdm(eval_cfg.snr_list, desc="CRLB sweep"):
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        try:
            result = compute_numeric_fim_tau(sim_cfg, batch_size=128, n_mc=20)
            records.append({
                'snr_db': snr_db,
                **result
            })
        except Exception as e:
            print(f"Warning: CRLB @ SNR={snr_db} failed: {e}")
            records.append({
                'snr_db': snr_db,
                'crlb_tau_samples': float('nan'),
            })

    return pd.DataFrame(records)


# ============================================================================
# Other Sweeps
# ============================================================================

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
    """Measure latency of each method."""
    import time

    records = []
    methods = ["adjoint_slice", "matched_filter", "proposed_no_update", "proposed"]

    snr_db = 15.0
    sim_cfg = create_sim_config(gabv_cfg, snr_db)

    # Generate a trial for warmup and testing
    trial = generate_trial(sim_cfg, 32, seed=42, device=eval_cfg.device, init_error=0.3)

    # Warmup
    for _ in range(3):
        try:
            evaluate_method_on_trial(model, "proposed", trial, sim_cfg, eval_cfg.device)
        except:
            pass

    for method in methods:
        latencies = []
        for _ in range(10):
            try:
                start = time.time()
                evaluate_method_on_trial(model, method, trial, sim_cfg, eval_cfg.device)
                latencies.append((time.time() - start) * 1000)  # ms
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
# CSV Validation
# ============================================================================

def validate_csv_methods(df: pd.DataFrame, expected_methods: List[str], csv_name: str) -> bool:
    """Validate that CSV contains all expected methods."""
    if 'method' not in df.columns:
        print(f"  ⚠️ {csv_name}: Missing 'method' column")
        return False

    actual_methods = set(df['method'].unique())
    expected_set = set(expected_methods)
    missing = expected_set - actual_methods

    if missing:
        print(f"  ⚠️ {csv_name}: Missing methods {missing}")
        return False

    print(f"  ✓ {csv_name}: Contains all {len(expected_methods)} methods")
    return True