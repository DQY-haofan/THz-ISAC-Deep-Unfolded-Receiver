#!/usr/bin/env python3
"""
train_gabv_net.py - GA-BV-Net Training Script (Wideband Delay Model - v6.0)

FUNDAMENTAL RESTRUCTURING:
=========================
This script trains GA-BV-Net with the CORRECT wideband delay model.

Key Changes:
1. theta = [tau_res, v, a] instead of [R, v, a]
2. tau_res is residual delay after coarse acquisition (small, ~samples)
3. THETA_SCALE normalized to physical resolution
4. Curriculum designed around IDENTIFIABILITY basin

Physical Insight:
    At B=10GHz, delay resolution δτ ~ 100ps → δR ~ 3cm
    We can realistically estimate tau_res within ±1-2 samples
    Carrier phase (λ=1mm) is nuisance, tracked by PN module

Curriculum Design:
    Stage 1: Communication only (theta exact, learn denoising)
    Stage 2: Fine tracking (tau_res noise < 1 sample, v/a noise moderate)
    Stage 3: Full noise (tau_res noise ~ 1 sample, v/a noise large)

Author: Expert Review v6.0 (Wideband Delay)
Date: 2025-12-19
"""

import argparse
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

# Local imports
try:
    from gabv_net_model import GABVNet, GABVConfig, create_gabv_model
    from thz_isac_world import SimConfig, simulate_batch, compute_bcrlb_diag

    HAS_DEPS = True
except ImportError as e:
    print(f"[Warning] Import error: {e}")
    HAS_DEPS = False

try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False

# =============================================================================
# Theta Normalization Scales (Physical Units!)
# =============================================================================

# theta = [tau_res, v, a] in SI units
# tau_res: residual delay [s], typical noise ~ 0.1 × Ts = 10ps for 10GHz
# v: velocity [m/s], typical noise ~ 100 m/s
# a: acceleration [m/s²], typical noise ~ 10 m/s²

# For B=10GHz: Ts = 100ps = 1e-10 s
Ts_default = 1e-10  # 100 ps

THETA_SCALE = torch.tensor([
    1e-10,  # 1 sample period (100ps for 10GHz) - tau_res scale
    100.0,  # 100 m/s - v scale
    10.0,  # 10 m/s² - a scale
])


# =============================================================================
# Training Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""

    # Training params
    n_steps: int = 1000
    batch_size: int = 32
    lr: float = 2e-3
    seed: int = 42

    # Hardware randomization
    randomize_hardware: bool = True

    # Debug mode
    debug_mode: bool = False

    # Warmup
    theta_noise_warmup_steps: int = 200
    g_theta_warmup_steps: int = 100

    # Output
    out_dir: str = "results/checkpoints"


@dataclass
class StageConfig:
    """Stage configuration for curriculum learning."""

    stage: int = 1
    name: str = "Stage1"
    description: str = ""
    n_steps: int = 1000
    lr_multiplier: float = 1.0

    # Theta noise std: [tau_res, v, a]
    # tau_res in samples (will be multiplied by Ts)
    # v in m/s, a in m/s²
    theta_noise_samples: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # (samples, m/s, m/s²)

    enable_theta_update: bool = False

    # Loss weights
    loss_weight_comm: float = 1.0
    loss_weight_sens: float = 0.0
    loss_weight_prior: float = 0.0

    # SNR range
    snr_range: Tuple[float, float] = (-5, 25)

    # Hardware
    enable_pn: bool = True
    freeze_comm: bool = False
    freeze_pn: bool = False


def get_curriculum_stages(base_steps: int) -> List[StageConfig]:
    """
    Get curriculum learning stages designed for IDENTIFIABILITY.

    Key insight:
        tau_res is identifiable within ±1-2 samples (basin of attraction)
        Beyond that, we need coarse acquisition (not GA-BV-Net's job)

    Curriculum:
        Stage 0: Ultra-simple debugging (no hardware impairments)
        Stage 1: Learn communication (exact theta)
        Stage 2: Learn fine tracking (< 0.5 sample noise)
        Stage 3: Push to limit (~1 sample noise)
    """
    return [
        # Stage 0: Ultra-Debug (no impairments at all!)
        StageConfig(
            stage=0,
            name="Stage0_Debug",
            description="Debug mode: no impairments, v=0, a=0",
            n_steps=base_steps // 2,
            theta_noise_samples=(0.0, 0.0, 0.0),
            enable_theta_update=False,
            loss_weight_comm=1.0,
            loss_weight_sens=0.0,
            loss_weight_prior=0.0,
            snr_range=(20, 30),
            enable_pn=False,  # No phase noise!
            freeze_comm=False,
            freeze_pn=True,
        ),

        # Stage 1: Communication Only
        StageConfig(
            stage=1,
            name="Stage1_CommOnly",
            description="Communication only, exact theta (learn denoising)",
            n_steps=base_steps,
            theta_noise_samples=(0.0, 0.0, 0.0),  # Exact theta
            enable_theta_update=False,
            loss_weight_comm=1.0,
            loss_weight_sens=0.0,
            loss_weight_prior=0.0,
            snr_range=(-5, 25),
            enable_pn=True,
            freeze_comm=False,
            freeze_pn=False,
        ),

        # Stage 2: Fine Tracking (within basin)
        StageConfig(
            stage=2,
            name="Stage2_FineTrak",
            description="Fine tracking (tau_res < 0.5 sample, within basin)",
            n_steps=base_steps,
            theta_noise_samples=(0.3, 50.0, 5.0),  # 0.3 samples = 30ps = 9mm
            enable_theta_update=True,
            loss_weight_comm=1.0,
            loss_weight_sens=0.5,
            loss_weight_prior=0.1,
            snr_range=(5, 25),
            enable_pn=True,
            lr_multiplier=0.5,
            freeze_comm=True,  # Freeze comm modules
            freeze_pn=True,
        ),

        # Stage 3: Full Tracking (at basin edge)
        StageConfig(
            stage=3,
            name="Stage3_FullTrak",
            description="Full tracking (tau_res ~ 1 sample, basin edge)",
            n_steps=base_steps,
            theta_noise_samples=(1.0, 100.0, 10.0),  # 1 sample = 100ps = 3cm
            enable_theta_update=True,
            loss_weight_comm=1.0,
            loss_weight_sens=1.0,
            loss_weight_prior=0.2,
            snr_range=(-5, 25),
            enable_pn=True,
            lr_multiplier=0.3,
            freeze_comm=True,
            freeze_pn=True,
        ),
    ]


# =============================================================================
# Meta Feature Construction
# =============================================================================

def construct_meta_features(meta_dict: Dict, batch_size: int) -> torch.Tensor:
    """
    Construct meta feature tensor from simulation metadata.

    Schema (must match model!):
        meta[:, 0] = snr_db_norm      = (snr_db - 15) / 15
        meta[:, 1] = gamma_eff_db_norm = (10*log10(gamma_eff) - 10) / 20
        meta[:, 2] = chi              (raw, range [0, 2/π])
        meta[:, 3] = sigma_eta_norm   = sigma_eta / 0.1
        meta[:, 4] = pn_linewidth_norm = log10(pn_linewidth + 1) / log10(1e6)
        meta[:, 5] = ibo_db_norm      = (ibo_dB - 3) / 3
    """
    snr_db = meta_dict.get('snr_db', 20.0)
    gamma_eff = meta_dict.get('gamma_eff', 1.0)
    chi = meta_dict.get('chi', 0.1)
    sigma_eta = meta_dict.get('sigma_eta', 0.01)
    pn_linewidth = meta_dict.get('pn_linewidth', 100e3)
    ibo_dB = meta_dict.get('ibo_dB', 3.0)

    # Normalize
    snr_db_norm = (snr_db - 15) / 15
    gamma_eff_db = 10 * np.log10(max(gamma_eff, 1e-6))
    gamma_eff_db_norm = (gamma_eff_db - 10) / 20
    sigma_eta_norm = sigma_eta / 0.1
    pn_linewidth_norm = np.log10(pn_linewidth + 1) / np.log10(1e6)
    ibo_db_norm = (ibo_dB - 3) / 3

    meta_vec = np.array([
        snr_db_norm,
        gamma_eff_db_norm,
        chi,
        sigma_eta_norm,
        pn_linewidth_norm,
        ibo_db_norm,
    ], dtype=np.float32)

    return torch.from_numpy(np.tile(meta_vec, (batch_size, 1)))


# =============================================================================
# Loss Functions
# =============================================================================

def compute_comm_loss(x_hat: torch.Tensor, x_true: torch.Tensor) -> torch.Tensor:
    """Compute communication loss (MSE on symbols)."""
    return torch.mean(torch.abs(x_hat - x_true) ** 2)


def compute_sensing_loss(theta_hat: torch.Tensor,
                         theta_true: torch.Tensor,
                         device: torch.device,
                         bcrlb_diag: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict]:
    """
    Compute normalized sensing loss with optional BCRLB weighting.

    Two modes:
    1. Simple normalized loss (if bcrlb_diag is None):
        loss = mean((theta_hat - theta_true)² / THETA_SCALE²)

    2. BCRLB-weighted Mahalanobis loss (recommended):
        loss = mean((theta_hat - theta_true)² / (BCRLB + ε))

        This weights each component by its Fisher information,
        giving more weight to parameters that are easier to estimate.

    Args:
        theta_hat: Estimated theta [B, 3]
        theta_true: True theta [B, 3]
        device: PyTorch device
        bcrlb_diag: Optional BCRLB diagonal [3] (detached!)

    Returns:
        loss: Scalar loss
        metrics: Dictionary with per-component RMSE
    """
    if bcrlb_diag is not None:
        # BCRLB-weighted loss (Mahalanobis-like)
        # Detach BCRLB to prevent gradient flow through it
        bcrlb = bcrlb_diag.detach().to(device)
        weights = 1.0 / (bcrlb + 1e-20)

        diff = theta_hat - theta_true
        loss = torch.mean(diff ** 2 * weights)
    else:
        # Simple normalized loss
        scales = THETA_SCALE.to(device)
        diff_normalized = (theta_hat - theta_true) / scales
        loss = torch.mean(diff_normalized ** 2)

    # Per-component RMSE for logging (in original units)
    with torch.no_grad():
        rmse_tau = torch.sqrt(torch.mean((theta_hat[:, 0] - theta_true[:, 0]) ** 2)).item()
        rmse_v = torch.sqrt(torch.mean((theta_hat[:, 1] - theta_true[:, 1]) ** 2)).item()
        rmse_a = torch.sqrt(torch.mean((theta_hat[:, 2] - theta_true[:, 2]) ** 2)).item()

    metrics = {
        'rmse_tau': rmse_tau,
        'rmse_v': rmse_v,
        'rmse_a': rmse_a,
        'rmse_tau_samples': rmse_tau / Ts_default,  # In sample units
    }

    return loss, metrics


def compute_prior_loss(theta_hat: torch.Tensor,
                       theta_init: torch.Tensor,
                       device: torch.device) -> torch.Tensor:
    """
    Compute prior/regularization loss.

    Penalizes large deviations from initial estimate.
    This acts as a soft constraint to keep estimates reasonable.
    """
    scales = THETA_SCALE.to(device)
    diff_normalized = (theta_hat - theta_init) / scales
    return torch.mean(diff_normalized ** 2)


# =============================================================================
# Module Freezing
# =============================================================================

def freeze_module(module: nn.Module, freeze: bool = True):
    """Freeze or unfreeze module parameters."""
    for param in module.parameters():
        param.requires_grad = not freeze


def apply_freeze_schedule(model: GABVNet, stage_cfg: StageConfig):
    """Apply freeze schedule based on stage config."""
    if stage_cfg.freeze_comm:
        freeze_module(model.phys_enc, True)
        freeze_module(model.solver_layers, True)
        freeze_module(model.refiner, True)
        print("  [Freeze] phys_enc, solver, refiner FROZEN")
    else:
        freeze_module(model.phys_enc, False)
        freeze_module(model.solver_layers, False)
        freeze_module(model.refiner, False)
        print("  [Freeze] phys_enc, solver, refiner ACTIVE")

    if stage_cfg.freeze_pn:
        freeze_module(model.pn_tracker, True)
        print("  [Freeze] pn_tracker FROZEN")
    else:
        freeze_module(model.pn_tracker, False)
        print("  [Freeze] pn_tracker ACTIVE")

    # Theta updater and pilot are always trainable
    freeze_module(model.theta_updater, False)
    freeze_module(model.pilot, False)


# =============================================================================
# Single Stage Training
# =============================================================================

def train_one_stage(
        cfg: TrainConfig,
        stage_cfg: StageConfig,
        model: Optional[GABVNet] = None,
        prev_ckpt: Optional[str] = None,
) -> Tuple[GABVNet, str]:
    """
    Train one stage of curriculum learning.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Move scales to device
    theta_scale = THETA_SCALE.to(device)

    print(f"\n{'=' * 60}")
    print(f"Stage {stage_cfg.stage}: {stage_cfg.name}")
    print(f"Description: {stage_cfg.description}")
    print(f"Theta noise (samples, m/s, m/s²): {stage_cfg.theta_noise_samples}")
    print(f"Enable theta update: {stage_cfg.enable_theta_update}")
    print(f"Loss weights: comm={stage_cfg.loss_weight_comm}, "
          f"sens={stage_cfg.loss_weight_sens}, prior={stage_cfg.loss_weight_prior}")
    print(f"{'=' * 60}\n")

    # Set seeds
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create or load model
    if model is None:
        model_cfg = GABVConfig()
        model_cfg.enable_theta_update = stage_cfg.enable_theta_update
        model = create_gabv_model(model_cfg).to(device)

        if prev_ckpt and os.path.exists(prev_ckpt):
            print(f"Loading from: {prev_ckpt}")
            ckpt = torch.load(prev_ckpt, map_location=device, weights_only=False)
            model.load_state_dict(ckpt['model_state'], strict=False)
    else:
        model.cfg.enable_theta_update = stage_cfg.enable_theta_update

    model.to(device)

    # Apply freeze schedule
    apply_freeze_schedule(model, stage_cfg)

    # Create optimizer
    lr = cfg.lr * stage_cfg.lr_multiplier
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    print(f"[Optimizer] lr={lr:.6f}")

    # Training loop
    model.train()
    metrics_history = []

    pbar = tqdm(range(stage_cfg.n_steps), desc=f"Stage {stage_cfg.stage}")

    for step in pbar:
        # === Generate Data ===
        sim_cfg = SimConfig()

        # Randomize SNR
        snr_db = np.random.uniform(*stage_cfg.snr_range)
        sim_cfg.snr_db = snr_db

        # Stage 0: Ultra-debug mode - disable everything
        if stage_cfg.stage == 0:
            sim_cfg.enable_pa = False
            sim_cfg.enable_pn = False
            sim_cfg.enable_quantization = True  # Keep quantization to test it
            sim_cfg.v_rel = 0.0  # No Doppler!
            sim_cfg.a_rel = 0.0
            sim_cfg.phi0_random = False  # No random phase
            sim_cfg.coarse_acquisition_error_samples = 0.0
        # Normal randomization
        elif cfg.randomize_hardware:
            sim_cfg.enable_pa = np.random.random() > 0.3
            sim_cfg.enable_pn = stage_cfg.enable_pn and (np.random.random() > 0.3)
            sim_cfg.pn_linewidth = np.random.uniform(10e3, 1e6)
            sim_cfg.ibo_dB = np.random.uniform(1, 6)

        data = simulate_batch(sim_cfg, cfg.batch_size, seed=None)

        # === Prepare Tensors ===
        y_q = torch.from_numpy(data['y_q']).to(device)
        x_true = torch.from_numpy(data['x_true']).to(device)
        theta_true = torch.from_numpy(data['theta_true']).float().to(device)

        meta = construct_meta_features(data['meta'], cfg.batch_size).to(device)

        # === Apply Theta Noise with Warmup ===
        if cfg.debug_mode:
            theta_init = theta_true.clone()
            warmup_factor = 0.0
        else:
            warmup_factor = min(1.0, step / max(cfg.theta_noise_warmup_steps, 1))

            # Convert sample noise to SI units
            tau_noise_samples, v_noise, a_noise = stage_cfg.theta_noise_samples
            tau_noise_si = tau_noise_samples * Ts_default  # samples → seconds

            noise_std = torch.tensor([tau_noise_si, v_noise, a_noise], device=device)
            theta_init = theta_true + warmup_factor * torch.randn_like(theta_true) * noise_std

        # g_theta scheduling
        g_theta_sched = min(1.0, step / max(cfg.g_theta_warmup_steps, 1))

        # === Forward Pass ===
        batch = {
            'y_q': y_q,
            'x_true': x_true,
            'theta_init': theta_init,
            'meta': meta,
            'snr_db': snr_db,  # For BCRLB scaling in ThetaUpdater
        }

        optimizer.zero_grad()
        outputs = model(batch, g_theta_sched=g_theta_sched)

        x_hat = outputs['x_hat']
        theta_hat = outputs['theta_hat']

        # === Compute BCRLB for weighting (if sensing loss enabled) ===
        bcrlb_diag = None
        if stage_cfg.loss_weight_sens > 0:
            try:
                bcrlb_np = compute_bcrlb_diag(
                    sim_cfg,
                    10 ** (sim_cfg.snr_db / 10),
                    data['meta'].get('gamma_eff', 1.0)
                )
                bcrlb_diag = torch.from_numpy(bcrlb_np).float().to(device)
            except Exception:
                bcrlb_diag = None  # Fall back to simple loss

        # === Compute Losses ===
        loss_comm = compute_comm_loss(x_hat, x_true)
        loss_sens, sens_metrics = compute_sensing_loss(theta_hat, theta_true, device, bcrlb_diag)
        loss_prior = compute_prior_loss(theta_hat, theta_init, device)

        loss = (stage_cfg.loss_weight_comm * loss_comm +
                stage_cfg.loss_weight_sens * loss_sens +
                stage_cfg.loss_weight_prior * loss_prior)

        # === Backward ===
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        # === Metrics ===
        with torch.no_grad():
            # QPSK BER: compare I and Q bits separately
            # I bit = sign(real), Q bit = sign(imag)
            x_true_I = (x_true.real > 0).float()
            x_true_Q = (x_true.imag > 0).float()
            x_hat_I = (x_hat.real > 0).float()
            x_hat_Q = (x_hat.imag > 0).float()

            # BER = average of I and Q bit errors
            ber_I = torch.mean((x_true_I != x_hat_I).float()).item()
            ber_Q = torch.mean((x_true_Q != x_hat_Q).float()).item()
            ber = (ber_I + ber_Q) / 2

            # G_theta
            g_theta_mean = outputs['gates']['g_theta'].mean().item()

            # Accept rate
            accept_rate = 1.0
            if outputs['layers'] and 'theta_info' in outputs['layers'][-1]:
                accept_rate = outputs['layers'][-1]['theta_info'].get('accept_rate', 1.0)

        metrics = {
            'step': step,
            'loss': loss.item(),
            'loss_comm': loss_comm.item(),
            'loss_sens': loss_sens.item(),
            'BER': ber,
            'RMSE_tau_samples': sens_metrics['rmse_tau_samples'],
            'RMSE_v': sens_metrics['rmse_v'],
            'RMSE_a': sens_metrics['rmse_a'],
            'g_theta': g_theta_mean,
            'accept_rate': accept_rate,
        }
        metrics_history.append(metrics)

        # Update progress bar
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'BER': f"{ber:.4f}",
            'RMSE_τ': f"{sens_metrics['rmse_tau_samples']:.2f}samp",
            'g_θ': f"{g_theta_mean:.3f}",
            'acc': f"{accept_rate:.2f}",
        })

        # Periodic logging
        if (step + 1) % 200 == 0:
            print(f"\n  [Step {step + 1}] loss={loss.item():.4f}, BER={ber:.4f}, "
                  f"RMSE_τ={sens_metrics['rmse_tau_samples']:.2f} samples, "
                  f"g_theta={g_theta_mean:.3f}, accept={accept_rate:.2f}")

    # === Save Checkpoint ===
    timestamp = int(time.time())
    ckpt_dir = Path(cfg.out_dir) / f"{stage_cfg.name}_{timestamp}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / "final.pth"
    torch.save({
        'model_state': model.state_dict(),
        'config': {
            'stage': stage_cfg.stage,
            'n_steps': stage_cfg.n_steps,
            'n_layers': model.cfg.n_layers,
            'enable_theta_update': stage_cfg.enable_theta_update,
            'fs': model.cfg.fs,
        },
        'version': 'v6.0_wideband',
        'metrics': metrics_history[-100:],
    }, ckpt_path)

    print(f"\n[Checkpoint] Saved to: {ckpt_path}")

    # Save metrics
    df = pd.DataFrame(metrics_history)
    df.to_csv(ckpt_dir / "metrics.csv", index=False)

    print(f"Training Complete. Stage {stage_cfg.stage} complete.")

    return model, str(ckpt_path)


# =============================================================================
# Curriculum Training
# =============================================================================

def run_curriculum(cfg: TrainConfig, stages: List[int] = [1, 2, 3]):
    """Run curriculum learning across multiple stages."""

    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING (Wideband Delay Model)")
    print(f"Stages: {stages}")
    print(f"Steps per stage: {cfg.n_steps}")
    print("=" * 70)

    all_stages = get_curriculum_stages(cfg.n_steps)

    model = None
    prev_ckpt = None
    global_step = 0

    for stage_num in stages:
        stage_cfg = all_stages[stage_num - 1]

        model, ckpt_path = train_one_stage(
            cfg=cfg,
            stage_cfg=stage_cfg,
            model=model,
            prev_ckpt=prev_ckpt,
        )

        prev_ckpt = ckpt_path
        global_step += stage_cfg.n_steps

        print(f"\n{'=' * 60}")
        print(f"Stage {stage_num} complete. Global step: {global_step}")
        print(f"{'=' * 60}\n")

    print("\n" + "=" * 70)
    print("CURRICULUM TRAINING COMPLETE")
    print(f"Total steps: {global_step}")
    print("=" * 70)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GA-BV-Net (Wideband Delay Model)")
    parser.add_argument('--curriculum', action='store_true', help='Run curriculum learning')
    parser.add_argument('--stage', type=int, default=1, help='Single stage to train')
    parser.add_argument('--steps', type=int, default=1000, help='Steps per stage')
    parser.add_argument('--batch', type=int, default=32, help='Batch size')
    parser.add_argument('--lr', type=float, default=2e-3, help='Learning rate')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')

    args = parser.parse_args()

    cfg = TrainConfig(
        n_steps=args.steps,
        batch_size=args.batch,
        lr=args.lr,
        seed=args.seed,
        debug_mode=args.debug,
    )

    print("=" * 60)
    print("GA-BV-Net Training (Wideband Delay Model v6.0)")
    print("=" * 60)
    print(f"Mode: {'Curriculum' if args.curriculum else f'Stage {args.stage}'}")
    print(f"Steps per stage: {cfg.n_steps}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Debug Mode: {cfg.debug_mode}")
    print("=" * 60)

    if args.curriculum:
        run_curriculum(cfg, stages=[1, 2, 3])
    else:
        stages = get_curriculum_stages(cfg.n_steps)
        train_one_stage(cfg, stages[args.stage - 1])