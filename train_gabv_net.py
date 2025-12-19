#!/usr/bin/env python3
"""
train_gabv_net.py - GA-BV-Net Training Script (Expert Review v5.0 - Stable)

Expert Review Changes v5.0:
- [CRITICAL FIX] Normalized sensing loss (THETA_SCALE) to prevent gradient explosion
- [CRITICAL FIX] Warmup applied to Stage 2/3 (was only Stage 1)
- [FIX] Reduced theta_params LR from 5x to 1x base_lr
- [FIX] Added g_theta scheduling (curriculum)
- [FIX] Renamed loss_bcrlb to loss_prior (correct terminology)
- [NEW] Stage2a/2b for freeze-then-unfreeze training
- [NEW] Mahalanobis-style loss option with FIM weighting
- [NEW] Enhanced logging (RMSE per component, accept_rate, effective_gate)
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
from thz_isac_world import SimConfig, simulate_batch
from gabv_net_model import GABVNet, GABVConfig, create_gabv_model

# Optional: matplotlib for training curves
try:
    import matplotlib.pyplot as plt

    HAS_PLT = True
except ImportError:
    HAS_PLT = False


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class TrainConfig:
    """Training configuration."""
    stage: int = 1
    n_steps: int = 1000
    batch_size: int = 32
    lr: float = 0.002
    seed: int = 42

    # Hardware randomization
    randomize_hardware: bool = True

    # Debug mode
    debug_mode: bool = False

    # Theta noise warmup (applies to ALL stages now)
    theta_noise_warmup_steps: int = 500

    # g_theta scheduling warmup steps
    g_theta_warmup_steps: int = 100  # [FIX] Reduced from 300 for faster warmup

    # Output
    out_dir: str = "results/checkpoints"


# =============================================================================
# [CRITICAL FIX] Theta Normalization Scales
# =============================================================================

# Physical scales for theta = [R, v, a]
# These represent "acceptable tolerance" for each parameter
# A 100m error in R contributes same loss as 10m/s error in v
THETA_SCALE = torch.tensor([1000.0, 10.0, 1.0])  # [1km, 10m/s, 1m/s²]

# Prior variance for MAP-style regularization (squared of noise std)
THETA_PRIOR_VAR = torch.tensor([100.0 ** 2, 10.0 ** 2, 0.5 ** 2])  # Stage3 noise level


@dataclass
class StageConfig:
    """Configuration for a curriculum learning stage."""
    stage: int
    name: str
    description: str
    n_steps: int

    # Theta configuration
    theta_noise_std: Tuple[float, float, float]  # (R, v, a) in physical units
    enable_theta_update: bool

    # Loss weights
    loss_weight_comm: float = 1.0
    loss_weight_sens: float = 0.0
    loss_weight_prior: float = 0.0  # Renamed from loss_weight_bcrlb

    # Training settings
    snr_range: Tuple[float, float] = (-5, 25)
    enable_pn: bool = True
    lr_multiplier: float = 1.0

    # [NEW] Freeze settings for Stage2a/2b
    freeze_comm_modules: bool = False
    freeze_pn_tracker: bool = False


# =============================================================================
# Meta Feature Construction (with FIX 11)
# =============================================================================

def construct_meta_features(raw_meta: dict, batch_size: int) -> torch.Tensor:
    """
    Construct normalized meta features for GA-BV-Net.

    [FIX 11] When enable_pn=False, force pn_linewidth=0
    """
    snr_db = raw_meta.get('snr_db', 20.0)
    gamma_eff = raw_meta.get('gamma_eff', 1e6)
    chi = raw_meta.get('chi', 0.6366)
    sigma_eta = raw_meta.get('sigma_eta', 0.0)

    # [FIX 11] Check enable_pn flag
    enable_pn = raw_meta.get('enable_pn', True)
    if enable_pn:
        pn_linewidth = raw_meta.get('pn_linewidth', 100e3)
    else:
        pn_linewidth = 0.0  # No PN

    ibo_dB = raw_meta.get('ibo_dB', 3.0)

    # Normalize features
    snr_db_norm = (snr_db - 15.0) / 15.0
    gamma_eff_db = 10.0 * math.log10(max(gamma_eff, 1e-12))
    gamma_eff_db_norm = (gamma_eff_db - 10.0) / 20.0
    chi_raw = chi
    sigma_eta_norm = sigma_eta / 0.1
    pn_linewidth_norm = math.log10(pn_linewidth + 1.0) / 6.0
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
# [CRITICAL FIX] Normalized Loss Functions
# =============================================================================

def compute_normalized_sensing_loss(theta_hat: torch.Tensor,
                                    theta_true: torch.Tensor,
                                    device: torch.device) -> Tuple[torch.Tensor, Dict]:
    """
    Compute normalized sensing loss to prevent gradient explosion.

    The key insight: R ~ 5e5 m, so raw MSE(R) ~ 1e10 dominates everything.
    By normalizing with THETA_SCALE, we get balanced gradients across all parameters.

    Args:
        theta_hat: Predicted theta [B, 3]
        theta_true: Ground truth theta [B, 3]
        device: Torch device

    Returns:
        loss_sens: Normalized MSE loss (scalar)
        metrics: Dict with per-component RMSE for logging
    """
    scales = THETA_SCALE.to(device)

    # Normalized error: (theta_hat - theta_true) / scale
    # This makes each component contribute equally to the loss
    diff_normalized = (theta_hat - theta_true) / scales
    loss_sens = torch.mean(diff_normalized ** 2)

    # Compute per-component RMSE for logging (in original units)
    with torch.no_grad():
        rmse_R = torch.sqrt(torch.mean((theta_hat[:, 0] - theta_true[:, 0]) ** 2)).item()
        rmse_v = torch.sqrt(torch.mean((theta_hat[:, 1] - theta_true[:, 1]) ** 2)).item()
        rmse_a = torch.sqrt(torch.mean((theta_hat[:, 2] - theta_true[:, 2]) ** 2)).item()

    metrics = {
        'rmse_R': rmse_R,
        'rmse_v': rmse_v,
        'rmse_a': rmse_a,
    }

    return loss_sens, metrics


def compute_prior_loss(theta_hat: torch.Tensor,
                       theta_init: torch.Tensor,
                       theta_noise_std: Tuple[float, float, float],
                       device: torch.device) -> torch.Tensor:
    """
    Compute prior/regularization loss (MAP-style).

    This is the correct implementation of "BCRLB regularization":
    - Acts as a prior term preventing theta from drifting too far from init
    - Weighted by the inverse variance of the prior (1/σ²)

    Args:
        theta_hat: Predicted theta [B, 3]
        theta_init: Initial theta estimate [B, 3]
        theta_noise_std: Standard deviation of theta noise (R, v, a)
        device: Torch device

    Returns:
        loss_prior: Prior regularization loss (scalar)
    """
    # Use the theta noise std as the prior standard deviation
    # This makes physical sense: the network shouldn't update theta
    # more than the expected noise range
    prior_std = torch.tensor(theta_noise_std, device=device) + 1e-6

    # Normalized prior loss: (theta_hat - theta_init)² / σ²
    diff = (theta_hat - theta_init) / prior_std
    loss_prior = torch.mean(diff ** 2)

    return loss_prior


def compute_mahalanobis_loss(theta_hat: torch.Tensor,
                             theta_true: torch.Tensor,
                             fim_inv_diag: torch.Tensor,
                             device: torch.device) -> torch.Tensor:
    """
    [OPTIONAL] Compute Mahalanobis-style loss using FIM inverse.

    This is the "proper BCRLB loss" - weights errors by their estimability.
    High FIM = easy to estimate = high weight on error.
    Low FIM = hard to estimate = low weight on error.

    Args:
        theta_hat: Predicted theta [B, 3]
        theta_true: Ground truth theta [B, 3]
        fim_inv_diag: Diagonal FIM inverse (approx CRLB variance) [B, 3]
        device: Torch device

    Returns:
        loss: Mahalanobis loss (scalar)
    """
    # FIM inverse is approximately the variance lower bound
    # Mahalanobis: (θ - θ*)ᵀ FIM (θ - θ*) = (θ - θ*)² / var_lb
    var_lb = fim_inv_diag.detach() + 1e-9  # Detach to avoid backprop through FIM

    diff_sq = (theta_hat - theta_true) ** 2
    loss = torch.mean(torch.sum(diff_sq / var_lb, dim=1))

    return loss


# =============================================================================
# BER Computation
# =============================================================================

def compute_ber(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    """Compute bit error rate for QPSK."""
    bits_hat_r = (np.real(x_hat) < 0).astype(int)
    bits_hat_i = (np.imag(x_hat) < 0).astype(int)
    bits_true_r = (np.real(x_true) < 0).astype(int)
    bits_true_i = (np.imag(x_true) < 0).astype(int)
    return 0.5 * (np.mean(bits_hat_r != bits_true_r) + np.mean(bits_hat_i != bits_true_i))


# =============================================================================
# Curriculum Stage Definitions (Fixed)
# =============================================================================

def get_curriculum_stages(base_steps: int) -> List[StageConfig]:
    """
    Define curriculum learning stages per expert recommendations.

    Stage 1: Communication only, exact theta (warm-up)
    Stage 2a: Freeze comm, train theta_updater only (NEW)
    Stage 2b: Unfreeze PN tracker, continue theta training (NEW)
    Stage 3: Full theta noise (100m) - paper target
    """
    return [
        StageConfig(
            stage=1,
            name="Stage1_CommOnly",
            description="Communication only, exact theta, PN tracking",
            n_steps=base_steps,
            theta_noise_std=(0.0, 0.0, 0.0),  # Exact theta
            enable_theta_update=False,
            loss_weight_comm=1.0,
            loss_weight_sens=0.0,
            loss_weight_prior=0.0,
            snr_range=(-5, 25),
            enable_pn=True,
            freeze_comm_modules=False,
            freeze_pn_tracker=False,
        ),
        # Stage 2a: Freeze comm modules, train only theta_updater
        StageConfig(
            stage=2,
            name="Stage2a_ThetaFreeze",
            description="Freeze comm, train theta_updater with 10m noise",
            n_steps=base_steps // 2,
            theta_noise_std=(10.0, 1.0, 0.1),  # Small noise
            enable_theta_update=True,
            loss_weight_comm=0.5,  # Lower weight since comm is frozen
            loss_weight_sens=1.0,
            loss_weight_prior=0.2,
            snr_range=(10, 25),  # Higher SNR for stability
            enable_pn=True,
            lr_multiplier=1.0,  # Normal LR for theta
            freeze_comm_modules=True,  # Freeze phys_enc, solver, refiner
            freeze_pn_tracker=True,  # Freeze PN tracker too
        ),
        # Stage 2b: Unfreeze PN tracker
        StageConfig(
            stage=3,
            name="Stage2b_ThetaUnfreeze",
            description="Unfreeze PN tracker, continue theta training",
            n_steps=base_steps // 2,
            theta_noise_std=(10.0, 1.0, 0.1),  # Same noise as 2a
            enable_theta_update=True,
            loss_weight_comm=1.0,
            loss_weight_sens=1.0,
            loss_weight_prior=0.1,
            snr_range=(5, 25),
            enable_pn=True,
            lr_multiplier=0.5,
            freeze_comm_modules=True,  # Keep solver frozen
            freeze_pn_tracker=False,  # Unfreeze PN tracker
        ),
        # Stage 3: Full noise, full fine-tuning
        StageConfig(
            stage=4,
            name="Stage3_ThetaFull",
            description="Full theta noise (100m) - paper target",
            n_steps=base_steps,
            theta_noise_std=(100.0, 10.0, 0.5),  # Full noise
            enable_theta_update=True,
            loss_weight_comm=1.0,
            loss_weight_sens=1.0,
            loss_weight_prior=0.2,
            snr_range=(-5, 25),
            enable_pn=True,
            lr_multiplier=0.3,
            freeze_comm_modules=False,  # Full fine-tuning
            freeze_pn_tracker=False,
        ),
    ]


def get_simple_curriculum_stages(base_steps: int) -> List[StageConfig]:
    """
    Simplified 3-stage curriculum (backward compatible).

    Use this if you don't need the freeze/unfreeze stages.
    """
    return [
        StageConfig(
            stage=1,
            name="Stage1_CommOnly",
            description="Communication only, exact theta, PN tracking",
            n_steps=base_steps,
            theta_noise_std=(0.0, 0.0, 0.0),
            enable_theta_update=False,
            loss_weight_comm=1.0,
            loss_weight_sens=0.0,
            loss_weight_prior=0.0,
            snr_range=(-5, 25),
            enable_pn=True,
        ),
        StageConfig(
            stage=2,
            name="Stage2_ThetaMicro",
            description="Enable theta update with 10m noise",
            n_steps=base_steps,
            theta_noise_std=(10.0, 1.0, 0.1),
            enable_theta_update=True,
            loss_weight_comm=1.0,
            loss_weight_sens=0.5,
            loss_weight_prior=0.1,
            snr_range=(5, 25),
            enable_pn=True,
            lr_multiplier=0.5,
        ),
        StageConfig(
            stage=3,
            name="Stage3_ThetaFull",
            description="Full theta noise (100m) - paper target",
            n_steps=base_steps,
            theta_noise_std=(100.0, 10.0, 0.5),
            enable_theta_update=True,
            loss_weight_comm=1.0,
            loss_weight_sens=1.0,
            loss_weight_prior=0.2,
            snr_range=(-5, 25),
            enable_pn=True,
            lr_multiplier=0.3,
        ),
    ]


# =============================================================================
# Module Freezing Utilities
# =============================================================================

def freeze_module(module: nn.Module, freeze: bool = True):
    """Freeze or unfreeze a module's parameters."""
    for param in module.parameters():
        param.requires_grad = not freeze


def apply_freeze_schedule(model: GABVNet, stage_cfg: StageConfig):
    """Apply freeze schedule based on stage configuration."""
    if stage_cfg.freeze_comm_modules:
        # Freeze communication modules
        freeze_module(model.phys_enc, True)
        freeze_module(model.solver, True)
        freeze_module(model.refiner, True)
        print("  [Freeze] phys_enc, solver, refiner FROZEN")
    else:
        # Unfreeze
        freeze_module(model.phys_enc, False)
        freeze_module(model.solver, False)
        freeze_module(model.refiner, False)
        print("  [Freeze] phys_enc, solver, refiner ACTIVE")

    if stage_cfg.freeze_pn_tracker:
        freeze_module(model.pn_tracker, True)
        print("  [Freeze] pn_tracker FROZEN")
    else:
        freeze_module(model.pn_tracker, False)
        print("  [Freeze] pn_tracker ACTIVE")

    # theta_updater and pilot are always trainable
    freeze_module(model.theta_updater, False)
    freeze_module(model.pilot, False)


# =============================================================================
# Single Stage Training
# =============================================================================

def train_one_stage(
        cfg: TrainConfig,
        stage_cfg: Optional[StageConfig] = None,
        model: Optional[GABVNet] = None,
        prev_ckpt: Optional[str] = None,
) -> Tuple[GABVNet, str]:
    """
    Train one stage of curriculum learning.

    Returns:
        model: Trained model
        ckpt_path: Path to saved checkpoint
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Move THETA_SCALE to device
    theta_scale = THETA_SCALE.to(device)

    # Use default stage config if not provided
    if stage_cfg is None:
        stages = get_simple_curriculum_stages(cfg.n_steps)
        stage_cfg = stages[cfg.stage - 1]

    print(f"\n{'=' * 60}")
    print(f"Stage {stage_cfg.stage}: {stage_cfg.name}")
    print(f"Description: {stage_cfg.description}")
    print(f"Theta noise std: {stage_cfg.theta_noise_std}")
    print(f"Enable theta update: {stage_cfg.enable_theta_update}")
    print(f"Loss weights: comm={stage_cfg.loss_weight_comm}, "
          f"sens={stage_cfg.loss_weight_sens}, prior={stage_cfg.loss_weight_prior}")
    print(f"{'=' * 60}\n")

    # Set random seed
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create or load model
    if model is None:
        model_cfg = GABVConfig()
        model_cfg.enable_theta_update = stage_cfg.enable_theta_update
        model = create_gabv_model(model_cfg).to(device)

        # Load previous checkpoint if available
        if prev_ckpt and os.path.exists(prev_ckpt):
            print(f"Loading from: {prev_ckpt}")
            ckpt = torch.load(prev_ckpt, map_location=device)
            model.load_state_dict(ckpt['model_state'], strict=False)
    else:
        # Update theta update setting
        model.cfg.enable_theta_update = stage_cfg.enable_theta_update

    model.to(device)

    # [NEW] Apply freeze schedule
    apply_freeze_schedule(model, stage_cfg)

    # [FIX] Create optimizer with SAME LR for theta updater (was 5x, now 1x)
    theta_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # Skip frozen params
        if 'theta_updater' in name:
            theta_params.append(param)
        else:
            other_params.append(param)

    base_lr = cfg.lr * stage_cfg.lr_multiplier

    # [FIX] theta_params now use SAME lr as others (was 5x)
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': base_lr},
        {'params': theta_params, 'lr': base_lr},  # Changed from base_lr * 5
    ], weight_decay=1e-4)

    print(f"[Optimizer] base_lr={base_lr:.6f}, theta_lr={base_lr:.6f}")
    print(f"[Optimizer] other_params={len(other_params)}, theta_params={len(theta_params)}")

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=stage_cfg.n_steps, eta_min=base_lr * 0.1
    )

    # Training loop
    model.train()
    metrics_history = []

    pbar = tqdm(range(stage_cfg.n_steps), desc=f"Stage {stage_cfg.stage}")
    for step in pbar:
        # Generate batch with randomized hardware
        sim_cfg = SimConfig()
        sim_cfg.snr_db = np.random.uniform(*stage_cfg.snr_range)

        if cfg.randomize_hardware:
            sim_cfg.enable_pn = stage_cfg.enable_pn and (np.random.random() > 0.2)
            sim_cfg.pn_linewidth = np.random.uniform(50e3, 200e3) if sim_cfg.enable_pn else 0
            sim_cfg.enable_pa = np.random.random() > 0.3
            sim_cfg.ibo_dB = np.random.uniform(2, 5) if sim_cfg.enable_pa else 3.0
        else:
            sim_cfg.enable_pn = stage_cfg.enable_pn
            sim_cfg.pn_linewidth = 100e3 if sim_cfg.enable_pn else 0
            sim_cfg.enable_pa = True
            sim_cfg.ibo_dB = 3.0

        # Simulate batch
        data = simulate_batch(sim_cfg, batch_size=cfg.batch_size, seed=None)

        # Prepare tensors
        y_q = torch.from_numpy(data['y_q']).cfloat().to(device)
        x_true = torch.from_numpy(data['x_true']).cfloat().to(device)
        theta_true = torch.from_numpy(data['theta_true']).float().to(device)

        # [CRITICAL FIX] Apply theta noise with warmup for ALL stages (not just Stage 1)
        if cfg.debug_mode:
            theta_init = theta_true.clone()
            warmup_factor = 0.0
        else:
            # Progressive noise warmup - applies to ALL stages now
            warmup_factor = min(1.0, step / max(cfg.theta_noise_warmup_steps, 1))

            theta_noise_std = torch.tensor(stage_cfg.theta_noise_std, device=device)
            theta_init = theta_true + warmup_factor * torch.randn_like(theta_true) * theta_noise_std

        # [NEW] g_theta scheduling - ramp up gradually
        g_theta_sched = min(1.0, step / max(cfg.g_theta_warmup_steps, 1))

        # Construct meta features
        meta_t = construct_meta_features(data['meta'], cfg.batch_size).to(device)

        # Forward pass
        batch = {
            'y_q': y_q,
            'x_true': x_true,
            'theta_init': theta_init,
            'theta_true': theta_true,
            'meta': meta_t,
            'g_theta_sched': g_theta_sched,  # Pass to model
        }

        optimizer.zero_grad()
        outputs = model(batch)

        # Extract outputs
        x_hat = outputs['x_hat']
        theta_hat = outputs['theta_hat']

        # === Compute losses with NORMALIZATION ===

        # Communication loss (MSE on symbols) - unchanged
        loss_comm = F.mse_loss(
            torch.stack([x_hat.real, x_hat.imag], dim=-1),
            torch.stack([x_true.real, x_true.imag], dim=-1)
        )

        # [CRITICAL FIX] Sensing loss with NORMALIZATION
        loss_sens, sens_metrics = compute_normalized_sensing_loss(
            theta_hat, theta_true, device
        )

        # [FIX] Prior loss (renamed from BCRLB - correct terminology)
        loss_prior = compute_prior_loss(
            theta_hat, theta_init, stage_cfg.theta_noise_std, device
        )

        # Wiener prior loss from PN tracker
        loss_wiener = outputs.get('wiener_loss', torch.tensor(0.0, device=device))

        # Total loss with all components
        loss = (stage_cfg.loss_weight_comm * loss_comm +
                stage_cfg.loss_weight_sens * loss_sens +
                stage_cfg.loss_weight_prior * loss_prior +
                0.1 * loss_wiener)

        # Backward pass with gradient clipping
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # === Compute metrics ===
        with torch.no_grad():
            ber = compute_ber(x_hat.cpu().numpy(), data['x_true'])

            # Get gate values from last layer
            last_layer = outputs['layers'][-1]
            g_pn = last_layer['gates']['g_PN'].mean().item()
            g_theta = last_layer['gates']['g_theta'].mean().item()

            # Get theta update info if available
            theta_info = last_layer.get('theta_info', {})
            delta_R = theta_info.get('delta_R', 0.0)
            delta_v = theta_info.get('delta_v', 0.0)
            delta_a = theta_info.get('delta_a', 0.0)
            accept_rate = theta_info.get('accept_rate', 1.0)
            g_theta_eff = theta_info.get('g_theta_effective', g_theta)

        metrics_history.append({
            'step': step,
            'loss': loss.item(),
            'loss_comm': loss_comm.item(),
            'loss_sens': loss_sens.item(),
            'loss_prior': loss_prior.item(),
            'ber': ber,
            'rmse_R': sens_metrics['rmse_R'],
            'rmse_v': sens_metrics['rmse_v'],
            'rmse_a': sens_metrics['rmse_a'],
            'g_pn': g_pn,
            'g_theta': g_theta,
            'g_theta_eff': g_theta_eff,
            'delta_R': delta_R,
            'delta_v': delta_v,
            'delta_a': delta_a,
            'accept_rate': accept_rate,
            'warmup_factor': warmup_factor,
            'g_theta_sched': g_theta_sched,
            'snr_db': sim_cfg.snr_db,
        })

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'BER': f'{ber:.4f}',
            'RMSE_R': f'{sens_metrics["rmse_R"]:.1f}m',
            'g_θ': f'{g_theta:.3f}',
            'ΔR': f'{delta_R:.1f}',
            'acc': f'{accept_rate:.2f}',
        })

        # Periodic logging
        if step > 0 and step % 200 == 0:
            recent = metrics_history[-200:]
            avg_loss = np.mean([m['loss'] for m in recent])
            avg_ber = np.mean([m['ber'] for m in recent])
            avg_rmse_R = np.mean([m['rmse_R'] for m in recent])
            avg_g_theta = np.mean([m['g_theta'] for m in recent])
            avg_accept = np.mean([m['accept_rate'] for m in recent])
            print(f"\n  [Step {step}] loss={avg_loss:.4f}, BER={avg_ber:.4f}, "
                  f"RMSE_R={avg_rmse_R:.1f}m, g_theta={avg_g_theta:.4f}, accept={avg_accept:.2f}")

    # === Check MC diversity ===
    gamma_effs = [simulate_batch(SimConfig(), 1, seed=None)['meta']['gamma_eff']
                  for _ in range(1000)]
    unique_gamma = len(set([f'{g:.6f}' for g in gamma_effs]))
    print(f"\n[MC Diversity Check] Unique gamma_eff values: {unique_gamma}/1000 "
          f"({unique_gamma / 10:.1f}%) {'✓' if unique_gamma > 900 else '✗'}")

    # === Save checkpoint ===
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
            'lr': cfg.lr,
            'fs': model.cfg.fs,
            'theta_noise_std': stage_cfg.theta_noise_std,
            'enable_theta_update': stage_cfg.enable_theta_update,
        },
        'version': 'v5.0_stable',
        'metrics': metrics_history[-100:],
    }, ckpt_path)

    print(f"\n[Checkpoint] Saved to: {ckpt_path}")

    # === Save training curves ===
    if HAS_PLT:
        save_training_curves(metrics_history, ckpt_dir, stage_cfg.name)

    # === Save metrics CSV ===
    df = pd.DataFrame(metrics_history)
    df.to_csv(ckpt_dir / "metrics.csv", index=False)

    print(f"[Result] Saved training curves to {ckpt_dir}/")
    print(f"Training Complete. Stage {stage_cfg.stage} complete.")

    return model, str(ckpt_path)


def save_training_curves(metrics: List[Dict], out_dir: Path, name: str):
    """Save training curves as plots."""
    df = pd.DataFrame(metrics)

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f'Training Curves: {name}', fontsize=14)

    # Row 0: Losses
    axes[0, 0].plot(df['step'], df['loss'], alpha=0.7, linewidth=0.8)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(df['step'], df['loss_comm'], alpha=0.7, linewidth=0.8, label='comm')
    axes[0, 1].plot(df['step'], df['loss_sens'], alpha=0.7, linewidth=0.8, label='sens')
    axes[0, 1].plot(df['step'], df['loss_prior'], alpha=0.7, linewidth=0.8, label='prior')
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Loss Components')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].semilogy(df['step'], df['ber'], alpha=0.7, linewidth=0.8)
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('BER')
    axes[0, 2].set_title('Bit Error Rate')
    axes[0, 2].grid(True, alpha=0.3)

    # Row 1: RMSE components
    axes[1, 0].plot(df['step'], df['rmse_R'], alpha=0.7, linewidth=0.8)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('RMSE_R (m)')
    axes[1, 0].set_title('Range RMSE')
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(df['step'], df['rmse_v'], alpha=0.7, linewidth=0.8)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('RMSE_v (m/s)')
    axes[1, 1].set_title('Velocity RMSE')
    axes[1, 1].grid(True, alpha=0.3)

    axes[1, 2].plot(df['step'], df['rmse_a'], alpha=0.7, linewidth=0.8)
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('RMSE_a (m/s²)')
    axes[1, 2].set_title('Acceleration RMSE')
    axes[1, 2].grid(True, alpha=0.3)

    # Row 2: Gates and deltas
    axes[2, 0].plot(df['step'], df['g_pn'], alpha=0.7, linewidth=0.8, label='g_PN')
    axes[2, 0].plot(df['step'], df['g_theta'], alpha=0.7, linewidth=0.8, label='g_θ')
    axes[2, 0].plot(df['step'], df['g_theta_eff'], alpha=0.7, linewidth=0.8, label='g_θ_eff')
    axes[2, 0].set_xlabel('Step')
    axes[2, 0].set_ylabel('Gate Value')
    axes[2, 0].set_title('Gates')
    axes[2, 0].set_ylim([0, 1])
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)

    axes[2, 1].plot(df['step'], df['delta_R'], alpha=0.7, linewidth=0.8)
    axes[2, 1].set_xlabel('Step')
    axes[2, 1].set_ylabel('delta_R (m)')
    axes[2, 1].set_title('Theta Update (ΔR)')
    axes[2, 1].grid(True, alpha=0.3)

    axes[2, 2].plot(df['step'], df['accept_rate'], alpha=0.7, linewidth=0.8)
    axes[2, 2].set_xlabel('Step')
    axes[2, 2].set_ylabel('Accept Rate')
    axes[2, 2].set_title('Acceptance Rate')
    axes[2, 2].set_ylim([0, 1])
    axes[2, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'train_curve_{name}.png', dpi=150, bbox_inches='tight')
    plt.savefig(out_dir / f'train_curve_{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# Curriculum Learning
# =============================================================================

def run_curriculum(cfg: TrainConfig, stages: List[int] = [1, 2, 3], use_extended: bool = False):
    """
    Run curriculum learning through specified stages.

    Args:
        cfg: Training configuration
        stages: List of stage numbers to run
        use_extended: If True, use 4-stage (freeze/unfreeze) curriculum
    """
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING")
    print(f"Stages: {stages}")
    print(f"Steps per stage: {cfg.n_steps}")
    print(f"Extended (freeze/unfreeze): {use_extended}")
    print("=" * 70)

    if use_extended:
        stage_configs = get_curriculum_stages(cfg.n_steps)
    else:
        stage_configs = get_simple_curriculum_stages(cfg.n_steps)

    model = None
    prev_ckpt = None
    global_step = 0

    for stage_num in stages:
        if stage_num > len(stage_configs):
            print(f"[Warning] Stage {stage_num} not defined, skipping")
            continue

        stage_cfg = stage_configs[stage_num - 1]

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
    print(f"Total steps across all stages: {global_step}")
    print("=" * 70)


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GA-BV-Net Training (Expert v5.0 Stable)")
    parser.add_argument("--stage", type=int, default=1, help="Single stage to run (1, 2, or 3)")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per stage")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Base learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--curriculum", action="store_true", help="Run stages 1→2→3")
    parser.add_argument("--extended", action="store_true", help="Use extended 4-stage curriculum")
    parser.add_argument("--no-hw-random", action="store_true", help="Disable hardware randomization")
    parser.add_argument("--debug", action="store_true", help="Debug mode: theta_init = theta_true")
    parser.add_argument("--theta-warmup", type=int, default=500, help="Theta noise warmup steps")
    parser.add_argument("--out", type=str, default="results/checkpoints", help="Output directory")
    args = parser.parse_args()

    cfg = TrainConfig(
        stage=args.stage,
        n_steps=args.steps,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
        randomize_hardware=not args.no_hw_random,
        debug_mode=args.debug,
        theta_noise_warmup_steps=args.theta_warmup,
        out_dir=args.out,
    )

    # Print configuration summary
    print("\n" + "=" * 60)
    print("GA-BV-Net Training (Expert Fixed v5.0 - Stable)")
    print("=" * 60)
    print(f"Mode: {'Curriculum (1→2→3)' if args.curriculum else f'Single Stage {cfg.stage}'}")
    print(f"Extended curriculum: {args.extended}")
    print(f"Steps per stage: {cfg.n_steps}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Hardware randomization: {cfg.randomize_hardware}")
    print(f"Debug Mode: {cfg.debug_mode}")
    print(f"Theta warmup steps: {cfg.theta_noise_warmup_steps}")
    if cfg.debug_mode:
        print("  → theta_init = theta_true (NO NOISE)")
    print("=" * 60 + "\n")

    if args.curriculum:
        if args.extended:
            run_curriculum(cfg, stages=[1, 2, 3, 4], use_extended=True)
        else:
            run_curriculum(cfg, stages=[1, 2, 3], use_extended=False)
    else:
        train_one_stage(cfg)