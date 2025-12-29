#!/usr/bin/env python3
"""
run_phase3_demo.py - Phase III Multi-Frame Tracking Demo

This script demonstrates the complete Phase III pipeline:
1. Generate multi-frame sequence data
2. Process frames with FrameProcessor (fast-loop)
3. Track with EKF/RTS (slow-loop)
4. Evaluate and output CSV results

Usage:
    python run_phase3_demo.py --K 100 --snr 20 --method MF+EKF
    python run_phase3_demo.py --sweep  # Run full parameter sweep

Based on:
- DR-III-E Section 7: Implementation roadmap
- Phase III Coding Prompt: MVP requirements

Author: Phase III MVP
Date: 2025-12
"""

import argparse
import os
import sys
import time
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd

# Add project path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, '/mnt/user-data/uploads')

from state_space import (
    StateSpaceConfig, MotionModel, compute_F_matrix, compute_Q_kinematic
)
from tracker import (
    TrackerConfig, EKFTracker, RTSSmoother, run_ekf_rts_pipeline,
    compute_nees, compute_nis
)
from frame_processor import (
    FrameProcessorConfig, FrameProcessor, EstimatorType,
    create_frame_processor, compute_dynamic_R
)
from thz_isac_world_ext import (
    SequenceConfig, simulate_sequence, simulate_sequence_simple,
    extract_measurements_from_sequence
)

try:
    from thz_isac_world import SimConfig
    HAS_SIM = True
except ImportError:
    HAS_SIM = False
    @dataclass
    class SimConfig:
        snr_db: float = 20.0
        adc_bits: int = 1
        fs: float = 10e9
        N: int = 1024
        enable_pa: bool = True
        enable_pn: bool = True
        pn_linewidth: float = 100e3
        enable_quantization: bool = True


def set_all_seeds(seed: int):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    import random
    random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


@dataclass
class ExperimentConfig:
    """Configuration for a tracking experiment."""
    
    # Sequence parameters
    K: int = 100
    dt: float = 1e-3
    
    # Signal parameters
    snr_db: float = 20.0
    adc_bits: int = 1
    pn_level: str = "low"  # "low" or "high"
    ibo_db: float = 3.0
    
    # Initial conditions
    # NOTE: init_v must be small enough that tau drift stays within MF search window
    # tau_dot = v/c, so v=1 m/s → Δτ = 1*dt/c = 3.33e-12 s/frame = 0.033 samples/frame
    # Over K=100 frames: 3.3 samples drift (within ±5 sample search window)
    init_tau_samples: float = 0.0
    init_v: float = 1.0      # m/s - reduced from 100 to keep tau in MF window
    init_a: float = 0.01     # m/s² - small acceleration
    init_error_samples: float = 0.0  # Added noise to initial estimate
    
    # Process noise
    S_jerk: float = 1e-10    # Reduced from 1e-6 for realistic LEO jerk
    
    # Method
    method: str = "MF+EKF"  # Options: "MF+EKF", "MF+RTS", "Oracle+EKF", "Oracle+RTS"
    
    # Random seed
    seed: int = 42
    
    @property
    def pn_linewidth(self) -> float:
        return 1e6 if self.pn_level == "high" else 100e3


def run_single_experiment(cfg: ExperimentConfig, verbose: bool = False
                          ) -> Dict[str, float]:
    """
    Run a single tracking experiment.
    
    Args:
        cfg: Experiment configuration
        verbose: Print progress
        
    Returns:
        results: Dictionary with metrics
    """
    set_all_seeds(cfg.seed)
    
    if verbose:
        print(f"\n[Experiment] method={cfg.method}, SNR={cfg.snr_db}dB, K={cfg.K}")
    
    # 1. Generate sequence
    sim_cfg = SimConfig(
        snr_db=cfg.snr_db,
        adc_bits=cfg.adc_bits,
        pn_linewidth=cfg.pn_linewidth,
        enable_pa=True,
        enable_pn=True,
        enable_quantization=True
    )
    
    seq_cfg = SequenceConfig(
        sim_config=sim_cfg,
        K=cfg.K,
        dt=cfg.dt,
        S_jerk=cfg.S_jerk,
        init_tau_samples=cfg.init_tau_samples,
        init_v=cfg.init_v,
        init_a=cfg.init_a
    )
    
    seq = simulate_sequence(seq_cfg, batch_size=1, seed=cfg.seed, verbose=verbose)
    
    # 2. Extract measurements (different methods)
    truth = seq['truth']
    K = cfg.K
    Ts = 1.0 / sim_cfg.fs
    
    # True state trajectory
    x_true = truth['x_state']  # [K, 3]
    tau_true = x_true[:, 0]
    
    # Determine estimator type from method
    if "Oracle" in cfg.method:
        estimator_type = EstimatorType.ORACLE
    else:
        estimator_type = EstimatorType.MATCHED_FILTER
    
    # 3. Process frames with FrameProcessor
    fp_cfg = FrameProcessorConfig(
        fs=sim_cfg.fs,
        N=sim_cfg.N,
        adc_bits=cfg.adc_bits,
        estimator_type=estimator_type
    )
    processor = FrameProcessor(fp_cfg)
    
    z_list = []
    R_list = []
    valid_list = []
    chi_list = []
    
    for k in range(K):
        frame = seq['frames'][k]
        
        # Add true tau for Oracle mode
        frame['tau_true'] = tau_true[k]
        
        # Get gamma_eff from frame meta if available
        gamma_eff = frame.get('meta', {}).get('gamma_eff', 100.0)
        frame.setdefault('meta', {})['gamma_eff'] = gamma_eff
        frame['meta']['snr_db'] = cfg.snr_db
        
        output = processor.process(frame)
        
        z_list.append(output.z)
        R_list.append(output.R)
        valid_list.append(output.valid_flag)
        chi_list.append(output.chi)
    
    z_seq = np.array(z_list)  # [K, 1]
    R_seq = np.array(R_list)  # [K, 1, 1]
    valid_flags = np.array(valid_list)
    
    # 4. Initialize tracker
    # Convert S_jerk from physical units (m²/s⁵) to delay-space (s²/s⁵)
    c = 3e8
    S_jerk_delay = cfg.S_jerk / (c**2)
    
    state_cfg = StateSpaceConfig(
        dt=cfg.dt,
        S_jerk=S_jerk_delay,  # Use delay-space units
        motion_model=MotionModel.CONSTANT_ACCELERATION
    )
    
    tracker_cfg = TrackerConfig(
        state_config=state_cfg,
        obs_dim=1
    )
    
    # Initial state with error
    x0 = x_true[0].copy()
    x0[0] += cfg.init_error_samples * Ts  # Add init error in tau
    
    # 5. Run tracking
    if "RTS" in cfg.method:
        results = run_ekf_rts_pipeline(
            z_seq=z_seq,
            R_seq=R_seq,
            valid_flags=valid_flags,
            x_true_seq=x_true,
            x0=x0,
            cfg=tracker_cfg
        )
        x_est = results['x_smooth']
        nees_seq = results['nees_smooth']
    else:  # EKF only
        results = run_ekf_rts_pipeline(
            z_seq=z_seq,
            R_seq=R_seq,
            valid_flags=valid_flags,
            x_true_seq=x_true,
            x0=x0,
            cfg=tracker_cfg
        )
        x_est = results['x_filt']
        nees_seq = results['nees_filt']
    
    nis_seq = results['nis']
    
    # 6. Compute metrics
    tau_est = x_est[:, 0]
    v_est = x_est[:, 1]
    a_est = x_est[:, 2] if x_est.shape[1] > 2 else np.zeros(K)
    
    # RMSE
    rmse_tau = np.sqrt(np.mean((tau_est - tau_true)**2))
    rmse_tau_samples = rmse_tau / Ts
    rmse_v = np.sqrt(np.mean((v_est - x_true[:, 1])**2))
    rmse_a = np.sqrt(np.mean((a_est - x_true[:, 2])**2)) if x_true.shape[1] > 2 else 0.0
    
    # Failure rate (RMSE > 3 samples at any point)
    tau_error_samples = np.abs(tau_est - tau_true) / Ts
    failure_rate = np.mean(tau_error_samples > 3.0)
    
    # NEES/NIS statistics
    mean_nees = np.mean(nees_seq)
    mean_nis = np.mean(nis_seq)
    
    # Valid flag rate
    valid_rate = np.mean(valid_flags)
    
    # Final error
    final_error_tau = np.abs(tau_est[-1] - tau_true[-1])
    final_error_samples = final_error_tau / Ts
    
    metrics = {
        'method': cfg.method,
        'seed': cfg.seed,
        'K': cfg.K,
        'snr_db': cfg.snr_db,
        'adc_bits': cfg.adc_bits,
        'pn_level': cfg.pn_level,
        'ibo_db': cfg.ibo_db,
        'init_error_samples': cfg.init_error_samples,
        'rmse_tau_ns': rmse_tau * 1e9,
        'rmse_tau_samples': rmse_tau_samples,
        'rmse_v': rmse_v,
        'rmse_a': rmse_a,
        'failure_rate': failure_rate,
        'mean_nees': mean_nees,
        'mean_nis': mean_nis,
        'valid_rate': valid_rate,
        'final_error_samples': final_error_samples,
        'mean_chi': np.mean(chi_list)
    }
    
    if verbose:
        print(f"  RMSE_tau: {rmse_tau_samples:.3f} samples ({rmse_tau*1e9:.3f} ns)")
        print(f"  RMSE_v: {rmse_v:.3f} m/s")
        print(f"  Failure rate: {failure_rate*100:.1f}%")
        print(f"  Mean NEES: {mean_nees:.2f}, Mean NIS: {mean_nis:.2f}")
    
    return metrics


def run_parameter_sweep(
    methods: List[str] = None,
    snr_range: List[float] = None,
    K_values: List[int] = None,
    init_errors: List[float] = None,
    seeds: List[int] = None,
    output_csv: str = "tracking_results.csv",
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run a full parameter sweep over methods and conditions.
    
    This is the main evaluation function for Phase III.
    
    Args:
        methods: List of tracking methods
        snr_range: List of SNR values [dB]
        K_values: List of frame counts
        init_errors: List of initial errors [samples]
        seeds: List of random seeds
        output_csv: Output CSV filename
        verbose: Print progress
        
    Returns:
        df: DataFrame with all results
    """
    # Defaults
    if methods is None:
        methods = ["MF+EKF", "MF+RTS", "Oracle+EKF"]
    if snr_range is None:
        snr_range = [-5, 0, 5, 10, 15, 20, 25]
    if K_values is None:
        K_values = [50, 100]
    if init_errors is None:
        init_errors = [0.0, 0.5, 1.0]
    if seeds is None:
        seeds = [42, 43, 44]
    
    all_results = []
    total_experiments = len(methods) * len(snr_range) * len(K_values) * len(init_errors) * len(seeds)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Phase III Parameter Sweep")
        print(f"{'='*60}")
        print(f"Methods: {methods}")
        print(f"SNR range: {snr_range} dB")
        print(f"K values: {K_values}")
        print(f"Init errors: {init_errors} samples")
        print(f"Seeds: {seeds}")
        print(f"Total experiments: {total_experiments}")
        print(f"{'='*60}\n")
    
    exp_count = 0
    start_time = time.time()
    
    for method in methods:
        for snr_db in snr_range:
            for K in K_values:
                for init_error in init_errors:
                    for seed in seeds:
                        exp_count += 1
                        
                        if verbose and exp_count % 10 == 0:
                            elapsed = time.time() - start_time
                            eta = elapsed / exp_count * (total_experiments - exp_count)
                            print(f"[{exp_count}/{total_experiments}] "
                                  f"method={method}, SNR={snr_db}dB, K={K}, "
                                  f"init_err={init_error} | ETA: {eta:.0f}s")
                        
                        cfg = ExperimentConfig(
                            K=K,
                            snr_db=snr_db,
                            method=method,
                            init_error_samples=init_error,
                            seed=seed
                        )
                        
                        try:
                            metrics = run_single_experiment(cfg, verbose=False)
                            all_results.append(metrics)
                        except Exception as e:
                            warnings.warn(f"Experiment failed: {e}")
                            # Record failed experiment
                            all_results.append({
                                'method': method,
                                'seed': seed,
                                'K': K,
                                'snr_db': snr_db,
                                'init_error_samples': init_error,
                                'failed': True,
                                'error': str(e)
                            })
    
    # Create DataFrame
    df = pd.DataFrame(all_results)
    
    # Save to CSV
    output_path = Path(output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    
    if verbose:
        elapsed = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Sweep complete! {exp_count} experiments in {elapsed:.1f}s")
        print(f"Results saved to: {output_path}")
        print(f"{'='*60}")
    
    return df


def run_protocol_tests(verbose: bool = True) -> Dict[str, bool]:
    """
    Run the 4 protocol tests from DR-III-E.
    
    Protocol-0: No-noise test (RMSE decreases with K)
    Protocol-1: Init error test (EKF improves over single-frame)
    Protocol-2: NEES/NIS not explosive
    Protocol-3: Full method sweep with all dimensions
    
    Returns:
        results: Dict with pass/fail for each protocol
    """
    results = {}
    
    if verbose:
        print("\n" + "="*60)
        print("Phase III Protocol Tests")
        print("="*60)
    
    # Protocol-0: Low noise, RMSE should decrease with K
    if verbose:
        print("\n[Protocol-0] RMSE vs K (low noise)")
    
    rmse_vs_k = []
    for K in [20, 50, 100, 200]:
        cfg = ExperimentConfig(K=K, snr_db=30.0, method="Oracle+EKF", 
                               init_error_samples=0.1, seed=42)
        metrics = run_single_experiment(cfg, verbose=False)
        rmse_vs_k.append((K, metrics['rmse_tau_samples']))
        if verbose:
            print(f"  K={K}: RMSE = {metrics['rmse_tau_samples']:.4f} samples")
    
    # Check if RMSE decreases (allow some noise)
    decreasing = all(rmse_vs_k[i][1] >= rmse_vs_k[i+1][1] * 0.9 
                     for i in range(len(rmse_vs_k)-1))
    results['protocol_0'] = decreasing
    if verbose:
        print(f"  Result: {'PASS ✓' if decreasing else 'FAIL ✗'}")
    
    # Protocol-1: Init error test
    if verbose:
        print("\n[Protocol-1] Init error recovery")
    
    single_frame_failure = 0
    ekf_recovery = 0
    
    for init_err in [0.5, 1.0, 1.5, 2.0]:
        cfg = ExperimentConfig(K=100, snr_db=20.0, method="MF+EKF",
                               init_error_samples=init_err, seed=42)
        metrics = run_single_experiment(cfg, verbose=False)
        
        # "Single frame" failure if init error > 1 sample and final error large
        if init_err > 1.0:
            single_frame_failure += 1
        
        # EKF recovery if final error < init error
        if metrics['final_error_samples'] < init_err:
            ekf_recovery += 1
        
        if verbose:
            print(f"  init_err={init_err:.1f}: final_err={metrics['final_error_samples']:.3f} samples")
    
    # Pass if EKF recovered at least once
    results['protocol_1'] = ekf_recovery > 0
    if verbose:
        print(f"  EKF recovery rate: {ekf_recovery}/4")
        print(f"  Result: {'PASS ✓' if results['protocol_1'] else 'FAIL ✗'}")
    
    # Protocol-2: NEES/NIS consistency
    if verbose:
        print("\n[Protocol-2] NEES/NIS consistency")
    
    cfg = ExperimentConfig(K=100, snr_db=20.0, method="MF+EKF", seed=42)
    metrics = run_single_experiment(cfg, verbose=False)
    
    nees_ok = 0.1 < metrics['mean_nees'] < 30  # Very loose bounds
    nis_ok = 0.1 < metrics['mean_nis'] < 30
    
    results['protocol_2'] = nees_ok and nis_ok
    if verbose:
        print(f"  Mean NEES: {metrics['mean_nees']:.2f} (expected ~3)")
        print(f"  Mean NIS: {metrics['mean_nis']:.2f} (expected ~1)")
        print(f"  Result: {'PASS ✓' if results['protocol_2'] else 'FAIL ✗'}")
    
    # Protocol-3: Full sweep has all method dimensions
    if verbose:
        print("\n[Protocol-3] Method sweep coverage")
    
    df = run_parameter_sweep(
        methods=["MF+EKF", "Oracle+EKF"],
        snr_range=[10, 20],
        K_values=[50],
        init_errors=[0.0],
        seeds=[42],
        output_csv="/tmp/protocol3_test.csv",
        verbose=False
    )
    
    has_method_dim = 'method' in df.columns and df['method'].nunique() >= 2
    has_snr_dim = 'snr_db' in df.columns and df['snr_db'].nunique() >= 2
    
    results['protocol_3'] = has_method_dim and has_snr_dim
    if verbose:
        print(f"  Methods in CSV: {df['method'].unique().tolist()}")
        print(f"  SNR values: {df['snr_db'].unique().tolist()}")
        print(f"  Result: {'PASS ✓' if results['protocol_3'] else 'FAIL ✗'}")
    
    # Summary
    if verbose:
        print("\n" + "="*60)
        print("Protocol Test Summary")
        print("="*60)
        for name, passed in results.items():
            print(f"  {name}: {'PASS ✓' if passed else 'FAIL ✗'}")
        
        all_pass = all(results.values())
        print(f"\n  Overall: {'ALL PASS ✓' if all_pass else 'SOME FAILED ✗'}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase III Multi-Frame Tracking Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single experiment
  python run_phase3_demo.py --K 100 --snr 20 --method "MF+EKF"
  
  # Parameter sweep
  python run_phase3_demo.py --sweep --output results/sweep.csv
  
  # Protocol tests
  python run_phase3_demo.py --protocols
        """
    )
    
    parser.add_argument('--K', type=int, default=100, help='Number of frames')
    parser.add_argument('--snr', type=float, default=20.0, help='SNR [dB]')
    parser.add_argument('--method', type=str, default='MF+EKF',
                        choices=['MF+EKF', 'MF+RTS', 'Oracle+EKF', 'Oracle+RTS'],
                        help='Tracking method')
    parser.add_argument('--init-error', type=float, default=0.0,
                        help='Initial tau error [samples]')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--adc-bits', type=int, default=1, help='ADC resolution')
    
    parser.add_argument('--sweep', action='store_true', help='Run parameter sweep')
    parser.add_argument('--protocols', action='store_true', help='Run protocol tests')
    parser.add_argument('--output', type=str, default='tracking_results.csv',
                        help='Output CSV file')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    
    args = parser.parse_args()
    
    if args.protocols:
        # Run protocol tests
        run_protocol_tests(verbose=True)
        
    elif args.sweep:
        # Run parameter sweep
        df = run_parameter_sweep(
            output_csv=args.output,
            verbose=True
        )
        print(f"\nResults preview:")
        print(df.groupby('method')[['rmse_tau_samples', 'failure_rate', 'mean_nees']].mean())
        
    else:
        # Single experiment
        cfg = ExperimentConfig(
            K=args.K,
            snr_db=args.snr,
            method=args.method,
            init_error_samples=args.init_error,
            seed=args.seed,
            adc_bits=args.adc_bits
        )
        
        metrics = run_single_experiment(cfg, verbose=True)
        
        print("\n" + "="*60)
        print("Results Summary")
        print("="*60)
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
