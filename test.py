#!/usr/bin/env python3
"""
debug_theta_updater.py - Diagnose theta updater with P0-1/2/3 fixes

Expert fixes applied:
- P0-1: Sync z to new theta after update
- P0-2: Use Bussgang-linearized residual instead of quantized
- P0-3: Use pilots only for theta update

Key metrics to watch:
- residual_improvement > 0: theta update is helping
- RMSE_τ < init: theta is being tracked
"""

import torch
import numpy as np
import math

# Local imports
from gabv_net_model import GABVNet, GABVConfig, create_gabv_model, PhysicsEncoder
from thz_isac_world import SimConfig, simulate_batch


def main():
    print("=" * 70)
    print("THETA UPDATER DIAGNOSTIC (P0-1/2/3 Fixes)")
    print("=" * 70)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    # Create model
    cfg = GABVConfig(enable_theta_update=True)
    model = create_gabv_model(cfg).to(device)
    model.eval()

    # Generate test data
    sim_cfg = SimConfig(snr_db=20.0)
    data = simulate_batch(sim_cfg, batch_size=4, seed=42)

    y_q = torch.from_numpy(data['y_q']).to(device)
    x_true = torch.from_numpy(data['x_true']).to(device)
    theta_true = torch.from_numpy(data['theta_true']).float().to(device)

    # DEBUG: Check x_true power
    x_true_power = torch.mean(torch.abs(x_true) ** 2).item()
    print(f"\n[Data Check]")
    print(f"  x_true shape: {x_true.shape}")
    print(f"  x_true power: {x_true_power:.6f} (should be ~1.0)")
    print(f"  x_true[0,:5]: {x_true[0, :5]}")

    # Add theta noise (0.5 samples)
    Ts = 1e-10
    noise_std = torch.tensor([0.5 * Ts, 50.0, 5.0], device=device)
    theta_init = theta_true + torch.randn_like(theta_true) * noise_std

    print(f"\n[Data]")
    print(f"  theta_true[0]: tau={theta_true[0, 0] / Ts:.3f} samples, v={theta_true[0, 1]:.1f} m/s")
    print(f"  theta_init[0]: tau={theta_init[0, 0] / Ts:.3f} samples, v={theta_init[0, 1]:.1f} m/s")
    print(f"  theta_error[0]: tau={abs(theta_init[0, 0] - theta_true[0, 0]) / Ts:.3f} samples")

    # === Check: Full Model Forward Pass ===
    print(f"\n[Full Model Forward Pass]")

    # Construct meta features
    snr_db = sim_cfg.snr_db
    snr_db_norm = (snr_db - 15) / 15
    gamma_eff = data['meta'].get('gamma_eff', 1.0)
    gamma_eff_db = 10 * np.log10(max(gamma_eff, 1e-6))
    gamma_eff_db_norm = (gamma_eff_db - 10) / 20
    chi = data['meta'].get('chi', 1.0)

    meta_np = np.array([
        snr_db_norm, gamma_eff_db_norm, chi, 0.0,
        np.log10(100e3) / np.log10(1e6), 0.0,
    ], dtype=np.float32)
    meta = torch.from_numpy(np.tile(meta_np, (4, 1))).to(device)

    batch = {
        'y_q': y_q,
        'x_true': x_true,
        'theta_init': theta_init.clone(),
        'meta': meta,
        'snr_db': snr_db,
    }

    with torch.no_grad():
        outputs = model(batch, g_theta_sched=1.0)

    theta_hat = outputs['theta_hat']

    print(f"  theta_init[0]:  tau={theta_init[0, 0] / Ts:.4f} samples")
    print(f"  theta_hat[0]:   tau={theta_hat[0, 0] / Ts:.4f} samples")
    print(f"  theta_true[0]:  tau={theta_true[0, 0] / Ts:.4f} samples")

    init_error = abs(theta_init[0, 0] - theta_true[0, 0]) / Ts
    final_error = abs(theta_hat[0, 0] - theta_true[0, 0]) / Ts
    improvement = init_error - final_error

    print(f"\n  init_error:  {init_error:.4f} samples")
    print(f"  final_error: {final_error:.4f} samples")
    print(f"  improvement: {improvement:.4f} samples")

    # Check layer diagnostics
    if outputs['layers']:
        last_layer = outputs['layers'][-1]
        if 'theta_info' in last_layer and last_layer['theta_info']:
            theta_info = last_layer['theta_info']
            print(f"\n[Theta Update Diagnostics - 3x3 Gauss-Newton]")
            print(f"  effective_gate: {theta_info.get('effective_gate', 'N/A'):.4f}")
            print(f"  residual_improvement: {theta_info.get('residual_improvement', 'N/A'):.6f}")

            # Delta in samples (after clamp and gate)
            delta_tau = theta_info.get('delta_tau', 0)
            print(f"  delta_tau: {delta_tau / Ts:.4f} samples")
            print(f"  delta_v: {theta_info.get('delta_v', 0):.2f} m/s")

            # Raw GN deltas (before clamp)
            print(f"\n[Raw Gauss-Newton Deltas (before clamp)]")
            gn_tau = theta_info.get('delta_gn_tau', 0)
            gn_v = theta_info.get('delta_gn_v', 0)
            print(f"  delta_gn_tau: {gn_tau / Ts:.4f} samples ({gn_tau:.2e} s)")
            print(f"  delta_gn_v: {gn_v:.2f} m/s")
            print(f"  G_cond: {theta_info.get('G_cond', 'N/A'):.2f}")

            # Normalized deltas (from GN solve, before scale conversion)
            print(f"\n[Normalized Deltas (GN solution)]")
            print(f"  delta_n_tau: {theta_info.get('delta_n_tau', 'N/A'):.6f}")
            print(f"  delta_n_v: {theta_info.get('delta_n_v', 'N/A'):.6f}")
            print(f"  scale_tau: {theta_info.get('scale_tau', 'N/A'):.2e}")
            print(f"  scale_v: {theta_info.get('scale_v', 'N/A'):.2e}")

            # Residual & projections
            print(f"\n[Residual & Projections]")
            using_pilot = theta_info.get('using_x_pilot', 0)
            print(f"  using_x_pilot: {'YES ✓' if using_pilot > 0.5 else 'NO (fallback to x_est)'}")

            # Power diagnostics (NEW)
            print(f"\n[Power Diagnostics]")
            print(f"  ||x_pilot input||²: {theta_info.get('x_pilot_input_power', 'N/A'):.6f}")
            print(f"  ||x_for_pred||²: {theta_info.get('x_power', 'N/A'):.6f}")
            print(f"  ||x_est||²:      {theta_info.get('x_est_power', 'N/A'):.6f}")
            print(f"  ||y_pred_full||²: {theta_info.get('y_pred_full_power', 'N/A'):.6f}")
            print(f"  ||y_pred (pilot)||²: {theta_info.get('y_pred_power', 'N/A'):.6f}")

            # GN solve diagnostics (NEW)
            print(f"\n[Gauss-Newton Solve]")
            print(f"  solve_success: {'YES ✓' if theta_info.get('solve_success', 0) > 0.5 else 'NO ✗'}")
            print(
                f"  G_diag: [{theta_info.get('G_diag_0', 0):.4f}, {theta_info.get('G_diag_1', 0):.4f}, {theta_info.get('G_diag_2', 0):.4f}] (should be ~1)")
            print(
                f"  b_vec:  [{theta_info.get('b_vec_0', 0):.4f}, {theta_info.get('b_vec_1', 0):.4f}, {theta_info.get('b_vec_2', 0):.4f}]")

            # Expert recommendation status
            print(f"\n[Expert Strategy: τ-only Update]")
            print(f"  v_info_weak: {'YES' if theta_info.get('v_info_weak', 0) > 0.5 else 'NO'}")
            print(f"  Reason: 64 pilots (6.4ns) → Δφ_v ≈ 0.001 rad (unidentifiable)")
            print(f"  Strategy: Fast loop (τ per-frame) + Slow loop (v cross-frame, future)")

            print(f"\n[Gradient Info]")
            print(f"  ||r||: {theta_info.get('r_norm', 'N/A'):.4f}")
            print(f"  b1 (J_tau^H @ r, norm): {theta_info.get('b1', 'N/A'):.6f}")
            print(f"  b2 (J_v^H @ r, norm): {theta_info.get('b2', 'N/A'):.6f}")
            print(f"  b1_raw: {theta_info.get('b1_raw', 'N/A'):.2e}")
            print(f"  ||J_tau||: {theta_info.get('norm_J_tau', 'N/A'):.2e}")
            print(f"  ||J_v||: {theta_info.get('norm_J_v', 'N/A'):.2e}")

            print(f"\n  bussgang_alpha: {theta_info.get('bussgang_alpha', 'N/A'):.4f}")

    # Check phi_est
    if 'phi_est' in outputs:
        phi_est = outputs['phi_est']
        print(f"\n[Phase Estimate]")
        print(
            f"  phi_est[0] mean: {phi_est[0].mean().item():.4f} rad ({phi_est[0].mean().item() * 180 / 3.14159:.1f}°)")

    # === Summary ===
    print(f"\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    if improvement > 0.01:
        print(f"  ✓ Theta update is IMPROVING tau estimation by {improvement:.4f} samples")
    elif improvement > -0.01:
        print(f"  ? Theta update has MINIMAL effect ({improvement:.4f} samples)")
    else:
        print(f"  ✗ Theta update is MAKING THINGS WORSE by {-improvement:.4f} samples")

    # Check gates
    g_theta = outputs['gates']['g_theta'].mean().item()
    if g_theta > 0.3:
        print(f"  ✓ g_theta = {g_theta:.3f} (healthy)")
    else:
        print(f"  ⚠ g_theta = {g_theta:.3f} (may be too low)")


if __name__ == "__main__":
    main()