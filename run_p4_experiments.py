#!/usr/bin/env python3
"""
run_p4_experiments.py - GA-BV-Net Evaluation (Wideband Delay Model - v6.0)

Evaluation script for the wideband delay model.

Key Features:
1. Consistency test: simulator == PhysicsEncoder
2. BER vs SNR curves
3. RMSE(tau, v, a) vs SNR
4. Identifiability cliff: performance vs initial error
5. Acceptance rate and gate diagnostics

Author: Expert Review v6.0 (Wideband Delay)
Date: 2025-12-19
"""

import argparse
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# Local imports
try:
    from gabv_net_model import GABVNet, GABVConfig, create_gabv_model, PhysicsEncoder
    from thz_isac_world import SimConfig, simulate_batch, wideband_delay_operator, doppler_phase_operator

    HAS_MODEL = True
except ImportError as e:
    print(f"[Warning] Import error: {e}")
    HAS_MODEL = False

# =============================================================================
# Configuration
# =============================================================================

Ts_default = 1e-10  # 100 ps for 10 GHz


# =============================================================================
# Consistency Tests
# =============================================================================

def test_simulator_physics_encoder_consistency(verbose: bool = True) -> Dict:
    """
    Test that simulator and PhysicsEncoder produce consistent results.

    This is CRITICAL - if they don't match, the model will never converge!
    """
    if verbose:
        print("\n[TEST] Simulator ↔ PhysicsEncoder Consistency")

    results = {}

    # Create PhysicsEncoder
    cfg = GABVConfig()
    phys_enc = PhysicsEncoder(cfg)

    # Test parameters
    B, N = 4, cfg.N
    tau = 1e-10  # 1 sample delay
    v = 1000.0  # 1000 m/s
    a = 10.0  # 10 m/s²

    # Generate test signal
    rng = np.random.default_rng(42)
    x_np = rng.normal(0, 1, (B, N)) + 1j * rng.normal(0, 1, (B, N))
    x_np = x_np / np.sqrt(2)

    # === Simulator path ===
    # D(τ) in numpy
    y_sim = wideband_delay_operator(x_np, tau, cfg.fs)
    # p(t) in numpy
    p_t = doppler_phase_operator(N, cfg.fs, cfg.fc, v, a)
    y_sim = y_sim * p_t[np.newaxis, :]

    # === PhysicsEncoder path ===
    x_torch = torch.from_numpy(x_np).to(torch.cfloat)
    theta = torch.tensor([[tau, v, a]] * B)
    y_phys = phys_enc.forward_operator(x_torch, theta)
    y_phys_np = y_phys.detach().numpy()

    # Compare
    error = np.max(np.abs(y_sim - y_phys_np))
    results['max_error'] = error
    results['pass'] = error < 1e-6

    if verbose:
        print(f"  Max error: {error:.2e}")
        print(f"  Status: {'PASS ✓' if results['pass'] else 'FAIL ✗'}")

    return results


def test_adjoint_consistency(verbose: bool = True) -> Dict:
    """Test that forward and adjoint are consistent."""
    if verbose:
        print("\n[TEST] Forward ↔ Adjoint Consistency")

    results = {}

    cfg = GABVConfig()
    phys_enc = PhysicsEncoder(cfg)

    B, N = 4, cfg.N
    x = torch.randn(B, N, dtype=torch.cfloat)
    theta = torch.zeros(B, 3)
    theta[:, 0] = 5e-11  # Small delay

    # Forward then adjoint
    y = phys_enc.forward_operator(x, theta)
    x_rec = phys_enc.adjoint_operator(y, theta)

    # Should recover x (approximately for small delay)
    error = torch.mean(torch.abs(x - x_rec) ** 2).item()
    results['reconstruction_error'] = error
    results['pass'] = error < 0.1  # Allow some error for non-zero delay

    if verbose:
        print(f"  Reconstruction MSE: {error:.6f}")
        print(f"  Status: {'PASS ✓' if results['pass'] else 'FAIL ✗'}")

    return results


# =============================================================================
# Model Loading
# =============================================================================

def load_model(ckpt_path: str, device: str) -> Optional[GABVNet]:
    """Load GA-BV-Net from checkpoint."""
    if not HAS_MODEL:
        return None

    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)

        cfg = GABVConfig(
            n_layers=ckpt['config'].get('n_layers', 8),
            enable_theta_update=ckpt['config'].get('enable_theta_update', True),
        )

        model = GABVNet(cfg)
        model.load_state_dict(ckpt['model_state'], strict=False)
        model.to(device)
        model.eval()

        print(f"[Model] Loaded from: {ckpt_path}")
        print(f"[Model] Version: {ckpt.get('version', 'unknown')}")
        print(f"[Model] enable_theta_update: {cfg.enable_theta_update}")

        return model
    except Exception as e:
        print(f"[Model] Failed to load: {e}")
        return None


# =============================================================================
# Meta Feature Construction
# =============================================================================

def construct_meta_features(meta_dict: Dict, batch_size: int) -> torch.Tensor:
    """Construct meta features tensor."""
    snr_db = meta_dict.get('snr_db', 20.0)
    gamma_eff = meta_dict.get('gamma_eff', 1.0)
    chi = meta_dict.get('chi', 0.1)
    sigma_eta = meta_dict.get('sigma_eta', 0.01)
    pn_linewidth = meta_dict.get('pn_linewidth', 100e3)
    ibo_dB = meta_dict.get('ibo_dB', 3.0)

    snr_db_norm = (snr_db - 15) / 15
    gamma_eff_db = 10 * np.log10(max(gamma_eff, 1e-6))
    gamma_eff_db_norm = (gamma_eff_db - 10) / 20
    sigma_eta_norm = sigma_eta / 0.1
    pn_linewidth_norm = np.log10(pn_linewidth + 1) / np.log10(1e6)
    ibo_db_norm = (ibo_dB - 3) / 3

    meta_vec = np.array([
        snr_db_norm, gamma_eff_db_norm, chi,
        sigma_eta_norm, pn_linewidth_norm, ibo_db_norm,
    ], dtype=np.float32)

    return torch.from_numpy(np.tile(meta_vec, (batch_size, 1)))


# =============================================================================
# Metrics Computation
# =============================================================================

def compute_ber(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """
    Compute BER for QPSK.

    QPSK has 2 bits per symbol:
    - I bit = sign(real)
    - Q bit = sign(imag)

    BER = average of I and Q bit error rates
    """
    # I bits
    bits_I_hat = (np.real(x_hat) > 0).astype(float)
    bits_I_true = (np.real(x_true) > 0).astype(float)
    ber_I = np.mean(bits_I_hat != bits_I_true)

    # Q bits
    bits_Q_hat = (np.imag(x_hat) > 0).astype(float)
    bits_Q_true = (np.imag(x_true) > 0).astype(float)
    ber_Q = np.mean(bits_Q_hat != bits_Q_true)

    return (ber_I + ber_Q) / 2


def compute_rmse_theta(theta_hat: np.ndarray, theta_true: np.ndarray) -> Tuple[float, float, float]:
    """Compute RMSE for each theta component."""
    rmse_tau = np.sqrt(np.mean((theta_hat[:, 0] - theta_true[:, 0]) ** 2))
    rmse_v = np.sqrt(np.mean((theta_hat[:, 1] - theta_true[:, 1]) ** 2))
    rmse_a = np.sqrt(np.mean((theta_hat[:, 2] - theta_true[:, 2]) ** 2))
    return rmse_tau, rmse_v, rmse_a


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_snr_sweep(model: GABVNet, device: str,
                       snr_range: np.ndarray,
                       n_mc: int = 10,
                       theta_noise_samples: float = 0.5) -> Dict:
    """
    Evaluate model across SNR range.

    Args:
        model: GA-BV-Net model
        device: Device
        snr_range: Array of SNR values [dB]
        n_mc: Monte Carlo trials per SNR
        theta_noise_samples: Initial theta noise in samples

    Returns:
        Dictionary with BER, RMSE vs SNR
    """
    results = {
        'snr_db': [],
        'ber': [],
        'rmse_tau_samples': [],
        'rmse_v': [],
        'rmse_a': [],
        'rmse_tau_init_samples': [],  # Initial RMSE for comparison
        'accept_rate': [],
        'g_theta': [],
    }

    sim_cfg = SimConfig()

    for snr_db in snr_range:
        sim_cfg.snr_db = snr_db

        bers = []
        rmse_taus = []
        rmse_vs = []
        rmse_as = []
        rmse_tau_inits = []
        accept_rates = []
        g_thetas = []

        for mc in range(n_mc):
            # Generate data with positive seed
            seed_val = abs(int(snr_db * 1000)) + mc + 10000
            data = simulate_batch(sim_cfg, batch_size=64, seed=seed_val)

            # Prepare tensors
            y_q = torch.from_numpy(data['y_q']).to(device)
            x_true = torch.from_numpy(data['x_true']).to(device)
            theta_true = torch.from_numpy(data['theta_true']).float().to(device)
            meta = construct_meta_features(data['meta'], 64).to(device)

            # Add theta noise
            tau_noise_si = theta_noise_samples * Ts_default
            noise_std = torch.tensor([tau_noise_si, 50.0, 5.0], device=device)
            theta_init = theta_true + torch.randn_like(theta_true) * noise_std

            # Forward pass
            batch = {
                'y_q': y_q,
                'x_true': x_true,
                'theta_init': theta_init,
                'meta': meta,
            }

            with torch.no_grad():
                outputs = model(batch)

            x_hat = outputs['x_hat'].cpu().numpy()
            theta_hat = outputs['theta_hat'].cpu().numpy()
            theta_init_np = theta_init.cpu().numpy()
            theta_true_np = data['theta_true']

            # Compute metrics
            ber = compute_ber(x_hat, data['x_true'])
            rmse_tau, rmse_v, rmse_a = compute_rmse_theta(theta_hat, theta_true_np)
            rmse_tau_init, _, _ = compute_rmse_theta(theta_init_np, theta_true_np)

            # Extract diagnostics
            accept_rate = 1.0
            g_theta = 0.0
            if outputs['layers'] and 'theta_info' in outputs['layers'][-1]:
                accept_rate = outputs['layers'][-1]['theta_info'].get('accept_rate', 1.0)
            if 'gates' in outputs:
                g_theta = outputs['gates']['g_theta'].mean().item()

            bers.append(ber)
            rmse_taus.append(rmse_tau / Ts_default)  # Convert to samples
            rmse_vs.append(rmse_v)
            rmse_as.append(rmse_a)
            rmse_tau_inits.append(rmse_tau_init / Ts_default)
            accept_rates.append(accept_rate)
            g_thetas.append(g_theta)

        results['snr_db'].append(snr_db)
        results['ber'].append(np.mean(bers))
        results['rmse_tau_samples'].append(np.mean(rmse_taus))
        results['rmse_v'].append(np.mean(rmse_vs))
        results['rmse_a'].append(np.mean(rmse_as))
        results['rmse_tau_init_samples'].append(np.mean(rmse_tau_inits))
        results['accept_rate'].append(np.mean(accept_rates))
        results['g_theta'].append(np.mean(g_thetas))

    return results


def evaluate_identifiability_cliff(model: GABVNet, device: str,
                                   error_range: np.ndarray,
                                   snr_db: float = 20.0,
                                   n_mc: int = 10) -> Dict:
    """
    Evaluate identifiability cliff: performance vs initial error.

    This shows the "basin of attraction" - how far from truth
    the network can recover from.
    """
    results = {
        'init_error_samples': [],
        'ber': [],
        'rmse_tau_samples': [],
        'improvement_ratio': [],  # rmse_init / rmse_hat
    }

    sim_cfg = SimConfig(snr_db=snr_db)

    for error_samples in error_range:
        bers = []
        rmse_taus = []
        improvement_ratios = []

        for mc in range(n_mc):
            data = simulate_batch(sim_cfg, batch_size=64, seed=2000 + mc)

            y_q = torch.from_numpy(data['y_q']).to(device)
            x_true = torch.from_numpy(data['x_true']).to(device)
            theta_true = torch.from_numpy(data['theta_true']).float().to(device)
            meta = construct_meta_features(data['meta'], 64).to(device)

            # Add specified error
            tau_noise_si = error_samples * Ts_default
            noise_std = torch.tensor([tau_noise_si, 50.0, 5.0], device=device)
            theta_init = theta_true + torch.randn_like(theta_true) * noise_std

            batch = {
                'y_q': y_q,
                'x_true': x_true,
                'theta_init': theta_init,
                'meta': meta,
            }

            with torch.no_grad():
                outputs = model(batch)

            x_hat = outputs['x_hat'].cpu().numpy()
            theta_hat = outputs['theta_hat'].cpu().numpy()
            theta_init_np = theta_init.cpu().numpy()
            theta_true_np = data['theta_true']

            ber = compute_ber(x_hat, data['x_true'])
            rmse_tau, _, _ = compute_rmse_theta(theta_hat, theta_true_np)
            rmse_tau_init, _, _ = compute_rmse_theta(theta_init_np, theta_true_np)

            improvement = rmse_tau_init / (rmse_tau + 1e-12)

            bers.append(ber)
            rmse_taus.append(rmse_tau / Ts_default)
            improvement_ratios.append(improvement)

        results['init_error_samples'].append(error_samples)
        results['ber'].append(np.mean(bers))
        results['rmse_tau_samples'].append(np.mean(rmse_taus))
        results['improvement_ratio'].append(np.mean(improvement_ratios))

    return results


# =============================================================================
# Plotting
# =============================================================================

def plot_snr_sweep(results: Dict, out_dir: Path):
    """Plot SNR sweep results."""
    if not HAS_PLT:
        return

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # BER vs SNR
    ax = axes[0, 0]
    ax.semilogy(results['snr_db'], results['ber'], 'b-o', label='GA-BV-Net')
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('BER')
    ax.set_title('BER vs SNR')
    ax.grid(True)
    ax.legend()

    # RMSE_tau vs SNR
    ax = axes[0, 1]
    ax.plot(results['snr_db'], results['rmse_tau_samples'], 'r-o', label='Estimated')
    ax.plot(results['snr_db'], results['rmse_tau_init_samples'], 'k--', label='Initial')
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('RMSE_τ [samples]')
    ax.set_title('Delay RMSE vs SNR')
    ax.grid(True)
    ax.legend()

    # Accept rate vs SNR
    ax = axes[1, 0]
    ax.plot(results['snr_db'], results['accept_rate'], 'g-o')
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('Accept Rate')
    ax.set_title('Acceptance Rate vs SNR')
    ax.grid(True)

    # g_theta vs SNR
    ax = axes[1, 1]
    ax.plot(results['snr_db'], results['g_theta'], 'm-o')
    ax.set_xlabel('SNR [dB]')
    ax.set_ylabel('g_θ')
    ax.set_title('Theta Gate vs SNR')
    ax.grid(True)

    plt.tight_layout()
    plt.savefig(out_dir / 'snr_sweep.png', dpi=150)
    plt.close()


def plot_identifiability_cliff(results: Dict, out_dir: Path):
    """Plot identifiability cliff."""
    if not HAS_PLT:
        return

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # BER vs initial error
    ax = axes[0]
    ax.plot(results['init_error_samples'], results['ber'], 'b-o')
    ax.set_xlabel('Initial Error [samples]')
    ax.set_ylabel('BER')
    ax.set_title('BER vs Initial Error (Identifiability Cliff)')
    ax.grid(True)
    ax.axvline(x=1.0, color='r', linestyle='--', label='1 sample = basin edge')
    ax.legend()

    # RMSE vs initial error
    ax = axes[1]
    ax.plot(results['init_error_samples'], results['rmse_tau_samples'], 'r-o')
    ax.plot(results['init_error_samples'], results['init_error_samples'], 'k--', label='No improvement')
    ax.set_xlabel('Initial Error [samples]')
    ax.set_ylabel('Final RMSE [samples]')
    ax.set_title('Final RMSE vs Initial Error')
    ax.grid(True)
    ax.legend()

    # Improvement ratio
    ax = axes[2]
    ax.plot(results['init_error_samples'], results['improvement_ratio'], 'g-o')
    ax.axhline(y=1.0, color='k', linestyle='--', label='No improvement')
    ax.set_xlabel('Initial Error [samples]')
    ax.set_ylabel('Improvement Ratio (init/final)')
    ax.set_title('Improvement Ratio vs Initial Error')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    plt.savefig(out_dir / 'identifiability_cliff.png', dpi=150)
    plt.close()


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Evaluate GA-BV-Net (Wideband Delay Model)")
    parser.add_argument('--ckpt', type=str, default=None, help='Checkpoint path (supports glob patterns)')
    parser.add_argument('--n_mc', type=int, default=10, help='Monte Carlo trials')
    parser.add_argument('--out_dir', type=str, default='results/evaluation', help='Output directory')

    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"[Device] {device}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # === Run Consistency Tests ===
    print("\n" + "=" * 60)
    print("CONSISTENCY TESTS")
    print("=" * 60)

    sim_phys_test = test_simulator_physics_encoder_consistency()
    adjoint_test = test_adjoint_consistency()

    if not sim_phys_test['pass']:
        print("\n⚠️  WARNING: Simulator and PhysicsEncoder are NOT consistent!")
        print("    This MUST be fixed before training will converge.")

    # === Resolve Checkpoint Path ===
    import glob

    ckpt_path = args.ckpt
    if ckpt_path is None:
        # Auto-find latest checkpoint
        patterns = [
            'results/checkpoints/Stage3_*/final.pth',
            'results/checkpoints/Stage2_*/final.pth',
            'results/checkpoints/Stage1_*/final.pth',
        ]
        for pattern in patterns:
            matches = glob.glob(pattern)
            if matches:
                # Sort by modification time, get newest
                matches.sort(key=os.path.getmtime, reverse=True)
                ckpt_path = matches[0]
                print(f"[Auto] Found checkpoint: {ckpt_path}")
                break
    elif '*' in ckpt_path:
        # Glob pattern provided
        matches = glob.glob(ckpt_path)
        if matches:
            matches.sort(key=os.path.getmtime, reverse=True)
            ckpt_path = matches[0]
            print(f"[Glob] Resolved to: {ckpt_path}")
        else:
            print(f"[Error] No files match pattern: {ckpt_path}")
            ckpt_path = None

    if ckpt_path is None:
        print("[Error] No checkpoint found. Run training first.")
        return

    # === Load Model ===
    model = load_model(ckpt_path, device)
    if model is None:
        print("Failed to load model. Exiting.")
        return

    # === SNR Sweep ===
    print("\n" + "=" * 60)
    print("SNR SWEEP")
    print("=" * 60)

    snr_range = np.arange(-5, 26, 2)
    snr_results = evaluate_snr_sweep(model, device, snr_range, n_mc=args.n_mc)

    print("\nResults:")
    for i, snr in enumerate(snr_results['snr_db']):
        print(f"  SNR={snr:3.0f}dB: BER={snr_results['ber'][i]:.4f}, "
              f"RMSE_τ={snr_results['rmse_tau_samples'][i]:.3f} samples "
              f"(init={snr_results['rmse_tau_init_samples'][i]:.3f})")

    plot_snr_sweep(snr_results, out_dir)

    # === Identifiability Cliff ===
    print("\n" + "=" * 60)
    print("IDENTIFIABILITY CLIFF")
    print("=" * 60)

    error_range = np.array([0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0])
    cliff_results = evaluate_identifiability_cliff(model, device, error_range, n_mc=args.n_mc)

    print("\nResults:")
    for i, err in enumerate(cliff_results['init_error_samples']):
        print(f"  Init={err:.1f} samples: BER={cliff_results['ber'][i]:.4f}, "
              f"RMSE_τ={cliff_results['rmse_tau_samples'][i]:.3f} samples, "
              f"Improvement={cliff_results['improvement_ratio'][i]:.2f}x")

    plot_identifiability_cliff(cliff_results, out_dir)

    print(f"\n[Done] Results saved to {out_dir}")


if __name__ == "__main__":
    main()