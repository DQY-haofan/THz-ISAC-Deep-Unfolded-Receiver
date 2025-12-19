#!/usr/bin/env python3
"""
debug_theta_updater.py - Diagnose why theta update is not working

Key observations from evaluation:
- g_theta = 0.005-0.07 (almost zero!)
- RMSE_τ ≈ init (no improvement)
- Curve C ≈ Curve B (theta update has no effect)

This script will check each component of theta updater.
"""

import torch
import numpy as np
import math

# Local imports
from gabv_net_model import GABVNet, GABVConfig, create_gabv_model, PhysicsEncoder, quantize_1bit_torch
from thz_isac_world import SimConfig, simulate_batch


def main():
    print("=" * 70)
    print("THETA UPDATER DIAGNOSTIC")
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

    # Add theta noise (0.5 samples)
    Ts = 1e-10
    noise_std = torch.tensor([0.5 * Ts, 50.0, 5.0], device=device)
    theta_init = theta_true + torch.randn_like(theta_true) * noise_std

    print(f"\n[Data]")
    print(f"  theta_true[0]: tau={theta_true[0, 0] / Ts:.3f} samples, v={theta_true[0, 1]:.1f} m/s")
    print(f"  theta_init[0]: tau={theta_init[0, 0] / Ts:.3f} samples, v={theta_init[0, 1]:.1f} m/s")
    print(f"  theta_error[0]: tau={abs(theta_init[0, 0] - theta_true[0, 0]) / Ts:.3f} samples")

    # === Check 1: Meta features ===
    print(f"\n[Check 1: Meta Features]")
    snr_db = sim_cfg.snr_db
    snr_db_norm = (snr_db - 15) / 15
    gamma_eff = data['meta'].get('gamma_eff', 1.0)
    gamma_eff_db = 10 * np.log10(max(gamma_eff, 1e-6))
    gamma_eff_db_norm = (gamma_eff_db - 10) / 20
    chi = data['meta'].get('chi', 1.0)

    meta_np = np.array([
        snr_db_norm,  # -0.33 to 0.67 for SNR -5 to 25
        gamma_eff_db_norm,  # normalized gamma_eff
        chi,  # Bussgang factor
        0.0,  # sigma_eta_norm
        np.log10(100e3) / np.log10(1e6),  # pn_linewidth_norm
        0.0,  # ibo_db_norm
    ], dtype=np.float32)
    meta = torch.from_numpy(np.tile(meta_np, (4, 1))).to(device)

    print(f"  meta[0]: {meta[0].cpu().numpy()}")
    print(f"  snr_db_norm: {snr_db_norm:.3f}")

    # === Check 2: Gate Network Output ===
    print(f"\n[Check 2: Gate Network (PilotNavigator)]")
    gates = model.pilot(meta)
    print(f"  g_data:  {gates['g_data'].mean().item():.4f}")
    print(f"  g_prior: {gates['g_prior'].mean().item():.4f}")
    print(f"  g_pn:    {gates['g_pn'].mean().item():.4f}")
    print(f"  g_theta: {gates['g_theta'].mean().item():.4f}")

    if gates['g_theta'].mean().item() < 0.1:
        print(f"  ⚠️ WARNING: g_theta too small! Theta update will be ineffective.")
        print(f"     This is likely the ROOT CAUSE of the problem.")

    # === Check 3: Theta Updater Internals ===
    print(f"\n[Check 3: Theta Updater Internals]")

    # Run forward pass to get x_est
    with torch.no_grad():
        z = model.phys_enc.adjoint_operator(y_q, theta_init)
        z_derotated, phi_est = model.pn_tracker(z, meta, x_true[:, :64])

        # Get x_est from one VAMP layer
        gamma = torch.ones(4, 1, device=device)
        z_vamp, x_est = model.solver_layers[0](z_derotated, gamma, model.phys_enc, theta_init)

        # Compute residual manually
        y_pred = model.phys_enc.forward_operator(x_est, theta_init)
        y_pred_q = quantize_1bit_torch(y_pred)
        residual = y_q - y_pred_q

        print(f"  x_est amplitude: {torch.abs(x_est).mean().item():.6f}")
        print(f"  residual power: {torch.mean(torch.abs(residual) ** 2).item():.6f}")

        # Compute Jacobian
        dy_dtheta = model.phys_enc.compute_channel_jacobian(theta_init, x_est)
        dy_dtau, dy_dv, dy_da = dy_dtheta

        # Compute score using UNIT-NORMALIZED Jacobians (new method)
        eps = 1e-10

        dy_dtau_norm = torch.sqrt(torch.sum(torch.abs(dy_dtau) ** 2, dim=1, keepdim=True) + eps)
        dy_dtau_hat = dy_dtau / dy_dtau_norm
        score_tau = torch.real(torch.sum(torch.conj(dy_dtau_hat) * residual, dim=1, keepdim=True))

        dy_dv_norm = torch.sqrt(torch.sum(torch.abs(dy_dv) ** 2, dim=1, keepdim=True) + eps)
        dy_dv_hat = dy_dv / dy_dv_norm
        score_v = torch.real(torch.sum(torch.conj(dy_dv_hat) * residual, dim=1, keepdim=True))

        dy_da_norm = torch.sqrt(torch.sum(torch.abs(dy_da) ** 2, dim=1, keepdim=True) + eps)
        dy_da_hat = dy_da / dy_da_norm
        score_a = torch.real(torch.sum(torch.conj(dy_da_hat) * residual, dim=1, keepdim=True))

        print(f"  |dy/dtau|: {dy_dtau_norm.mean().item():.6e}")
        print(f"  |dy/dv|:   {dy_dv_norm.mean().item():.6e}")
        print(f"  score_tau (unit-norm): {score_tau.mean().item():.6f}")
        print(f"  score_v (unit-norm):   {score_v.mean().item():.6f}")
        print(f"  score_a (unit-norm):   {score_a.mean().item():.6f}")

        # Compute step sizes from network
        residual_power = torch.mean(torch.abs(residual) ** 2, dim=1, keepdim=True)
        log_power = torch.log10(residual_power + 1e-10)
        score_magnitude = torch.abs(torch.cat([score_tau, score_v, score_a], dim=1))

        x_est_amplitude = torch.mean(torch.abs(x_est), dim=1, keepdim=True).clamp(min=1e-6)
        x_est_normalized = x_est / x_est_amplitude
        confidence = 1.0 - torch.mean(
            torch.abs(torch.abs(x_est_normalized.real) - 1 / math.sqrt(2)) ** 2 +
            torch.abs(torch.abs(x_est_normalized.imag) - 1 / math.sqrt(2)) ** 2,
            dim=1, keepdim=True
        )
        confidence = torch.clamp(confidence, 0, 1)

        snr_norm = torch.tensor([[(snr_db - 15) / 15]], device=device).expand(4, 1)

        feat = torch.cat([
            log_power, score_magnitude, gates['g_theta'], confidence, snr_norm
        ], dim=1).float()

        print(f"  feat[0]: {feat[0].cpu().numpy()}")
        print(f"  confidence: {confidence.mean().item():.4f}")

        step_sizes = model.theta_updater.step_net(feat)
        print(f"  step_sizes (raw): {step_sizes[0].cpu().numpy()}")

        step_sizes_scaled = step_sizes * model.theta_updater.bcrlb_scale.unsqueeze(0)
        print(f"  step_sizes (scaled): {step_sizes_scaled[0].cpu().numpy()}")

        # Compute delta
        score = torch.cat([score_tau, score_v, score_v], dim=1)
        delta_theta = -step_sizes_scaled * score
        print(
            f"  delta_theta (raw): tau={delta_theta[0, 0].item() / Ts:.6f} samples, v={delta_theta[0, 1].item():.6f} m/s")

        # Apply clamp
        delta_theta_clamped = torch.clamp(delta_theta, -model.theta_updater.max_delta, model.theta_updater.max_delta)
        print(
            f"  delta_theta (clamped): tau={delta_theta_clamped[0, 0].item() / Ts:.6f} samples, v={delta_theta_clamped[0, 1].item():.6f} m/s")

        # Apply gate (simplified - matches new model)
        effective_gate = gates['g_theta'] * 1.0  # g_theta_sched = 1.0
        min_gate = 0.1
        effective_gate = torch.maximum(effective_gate, torch.tensor(min_gate, device=device))
        print(f"  effective_gate: {effective_gate.mean().item():.4f}")

        delta_theta_gated = effective_gate * delta_theta_clamped
        print(
            f"  delta_theta (gated): tau={delta_theta_gated[0, 0].item() / Ts:.6f} samples, v={delta_theta_gated[0, 1].item():.6f} m/s")

        # Final update
        theta_new = theta_init + delta_theta_gated
        print(f"\n  theta_init[0]:  tau={theta_init[0, 0].item() / Ts:.6f} samples")
        print(f"  theta_new[0]:   tau={theta_new[0, 0].item() / Ts:.6f} samples")
        print(f"  theta_true[0]:  tau={theta_true[0, 0].item() / Ts:.6f} samples")

        improvement = abs(theta_init[0, 0] - theta_true[0, 0]) - abs(theta_new[0, 0] - theta_true[0, 0])
        print(f"  improvement: {improvement.item() / Ts:.6f} samples ({'better' if improvement > 0 else 'WORSE'})")

    # === Summary ===
    print(f"\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    issues = []

    if gates['g_theta'].mean().item() < 0.2:
        issues.append("g_theta too small (gate almost closed)")

    if confidence.mean().item() < 0.5:
        issues.append("confidence too low (confidence gate reduces update)")

    if abs(delta_theta_gated[0, 0].item()) < 1e-14:
        issues.append("delta_theta_gated ≈ 0 (no update happening)")

    if improvement < 0:
        issues.append("theta update made things WORSE")

    if not issues:
        print("  No obvious issues found. May need deeper investigation.")
    else:
        print("  ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"    {i}. {issue}")

    print("\n[RECOMMENDED FIXES]")
    print("  1. Initialize g_theta gate to higher value (e.g., 0.5 not ~0)")
    print("  2. Remove or simplify the gate mechanism during initial training")
    print("  3. Check if score direction is correct (should point toward theta_true)")


if __name__ == "__main__":
    main()