#!/usr/bin/env python3
"""
train_gabv_net.py - GA-BV-Net Training Script (Expert Review v4.1)

Expert Review Changes:
- Three-stage curriculum learning with ThetaUpdater progression
- Sensing loss and BCRLB regularization
- Enhanced monitoring (g_theta, delta_R, RMSE_R)
- Confidence-based gating support
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

    # Theta noise (progressive)
    theta_noise_warmup_steps: int = 500

    # Output
    out_dir: str = "results/checkpoints"


@dataclass
class StageConfig:
    """Configuration for a curriculum learning stage."""
    stage: int
    name: str
    description: str
    n_steps: int

    # Theta configuration
    theta_noise_std: Tuple[float, float, float]  # (R, v, a)
    enable_theta_update: bool

    # Loss weights
    loss_weight_comm: float = 1.0
    loss_weight_sens: float = 0.0
    loss_weight_bcrlb: float = 0.0

    # Training settings
    snr_range: Tuple[float, float] = (-5, 25)
    enable_pn: bool = True
    lr_multiplier: float = 1.0


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
# Curriculum Stage Definitions
# =============================================================================

def get_curriculum_stages(base_steps: int) -> List[StageConfig]:
    """
    Define curriculum learning stages per expert recommendations.

    Stage 1: Communication only, exact theta (warm-up)
    Stage 2: Enable theta update with small noise (10m)
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
            loss_weight_bcrlb=0.0,
            snr_range=(-5, 25),
            enable_pn=True,
        ),
        StageConfig(
            stage=2,
            name="Stage2_ThetaMicro",
            description="Enable theta update with 10m noise",
            n_steps=base_steps,
            theta_noise_std=(10.0, 1.0, 0.1),  # Small noise
            enable_theta_update=True,
            loss_weight_comm=1.0,
            loss_weight_sens=0.5,
            loss_weight_bcrlb=0.1,
            snr_range=(5, 25),  # Higher SNR for stability
            enable_pn=True,
            lr_multiplier=0.5,  # Lower LR for fine-tuning
        ),
        StageConfig(
            stage=3,
            name="Stage3_ThetaFull",
            description="Full theta noise (100m) - paper target",
            n_steps=base_steps,
            theta_noise_std=(100.0, 10.0, 0.5),  # Full noise
            enable_theta_update=True,
            loss_weight_comm=1.0,
            loss_weight_sens=1.0,
            loss_weight_bcrlb=0.2,
            snr_range=(-5, 25),
            enable_pn=True,
            lr_multiplier=0.3,
        ),
    ]


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

    # Use default stage config if not provided
    if stage_cfg is None:
        stages = get_curriculum_stages(cfg.n_steps)
        stage_cfg = stages[cfg.stage - 1]

    print(f"\n{'=' * 60}")
    print(f"Stage {stage_cfg.stage}: {stage_cfg.name}")
    print(f"Description: {stage_cfg.description}")
    print(f"Theta noise std: {stage_cfg.theta_noise_std}")
    print(f"Enable theta update: {stage_cfg.enable_theta_update}")
    print(f"Loss weights: comm={stage_cfg.loss_weight_comm}, "
          f"sens={stage_cfg.loss_weight_sens}, bcrlb={stage_cfg.loss_weight_bcrlb}")
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

    # Create optimizer with different LR for theta updater
    theta_params = []
    other_params = []
    for name, param in model.named_parameters():
        if 'theta_updater' in name:
            theta_params.append(param)
        else:
            other_params.append(param)

    base_lr = cfg.lr * stage_cfg.lr_multiplier
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': base_lr},
        {'params': theta_params, 'lr': base_lr * 5},  # Higher LR for theta updater
    ], weight_decay=1e-4)

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

        # Add theta noise (progressive warmup in Stage 1)
        if cfg.debug_mode:
            theta_init = theta_true.clone()
        else:
            # Progressive noise warmup
            if stage_cfg.stage == 1:
                warmup_factor = min(1.0, step / max(cfg.theta_noise_warmup_steps, 1))
            else:
                warmup_factor = 1.0

            theta_noise_std = torch.tensor(stage_cfg.theta_noise_std, device=device)
            theta_init = theta_true + warmup_factor * torch.randn_like(theta_true) * theta_noise_std

        # Construct meta features
        meta_t = construct_meta_features(data['meta'], cfg.batch_size).to(device)

        # Forward pass
        batch = {
            'y_q': y_q,
            'x_true': x_true,
            'theta_init': theta_init,
            'theta_true': theta_true,
            'meta': meta_t,
        }

        optimizer.zero_grad()
        outputs = model(batch)

        # Extract outputs
        x_hat = outputs['x_hat']
        theta_hat = outputs['theta_hat']

        # === Compute losses ===

        # Communication loss (MSE on symbols)
        # Communication loss (MSE on symbols)
        loss_comm = F.mse_loss(
            torch.stack([x_hat.real, x_hat.imag], dim=-1),
            torch.stack([x_true.real, x_true.imag], dim=-1)
        )

        # Sensing loss (MSE on theta)
        loss_sens = F.mse_loss(theta_hat, theta_true)

        # BCRLB regularization
        snr_weight = torch.sigmoid(torch.tensor((sim_cfg.snr_db - 10) / 5))
        loss_bcrlb = snr_weight * F.mse_loss(theta_hat, theta_init)

        # [NEW B-lite] Wiener prior loss from PN tracker
        # This is accumulated across layers in the forward pass
        loss_wiener = outputs.get('wiener_loss', torch.tensor(0.0))

        # Total loss with all components
        loss = (stage_cfg.loss_weight_comm * loss_comm +
                stage_cfg.loss_weight_sens * loss_sens +
                stage_cfg.loss_weight_bcrlb * loss_bcrlb +
                0.1 * loss_wiener)  # Wiener prior weight

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()

        # === Compute metrics ===
        with torch.no_grad():
            ber = compute_ber(x_hat.cpu().numpy(), data['x_true'])
            rmse_R = torch.sqrt(F.mse_loss(theta_hat[:, 0], theta_true[:, 0])).item()

            # Get gate values from last layer
            last_layer = outputs['layers'][-1]
            g_pn = last_layer['gates']['g_PN'].mean().item()
            g_theta = last_layer['gates']['g_theta'].mean().item()

            # Get theta update info if available
            theta_info = last_layer.get('theta_info', {})
            delta_R = theta_info.get('delta_R', 0.0)

        metrics_history.append({
            'step': step,
            'loss': loss.item(),
            'loss_comm': loss_comm.item(),
            'loss_sens': loss_sens.item(),
            'ber': ber,
            'rmse_R': rmse_R,
            'g_pn': g_pn,
            'g_theta': g_theta,
            'delta_R': delta_R,
            'snr_db': sim_cfg.snr_db,
        })

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'BER': f'{ber:.4f}',
            'RMSE': f'{rmse_R:.1f}m',
            'g_θ': f'{g_theta:.3f}',
            'ΔR': f'{delta_R:.1f}',
        })

        # Periodic logging
        if step > 0 and step % 200 == 0:
            recent = metrics_history[-200:]
            avg_loss = np.mean([m['loss'] for m in recent])
            avg_ber = np.mean([m['ber'] for m in recent])
            avg_rmse = np.mean([m['rmse_R'] for m in recent])
            avg_g_theta = np.mean([m['g_theta'] for m in recent])
            print(f"\n  [Step {step}] loss={avg_loss:.4f}, BER={avg_ber:.4f}, "
                  f"RMSE_R={avg_rmse:.1f}m, g_theta={avg_g_theta:.4f}")

    # === Check MC diversity ===
    gamma_effs = [simulate_batch(SimConfig(), 1, seed=None)['meta']['gamma_eff']
                  for _ in range(1000)]
    unique_gamma = len(set([f'{g:.6f}' for g in gamma_effs]))
    print(f"\n[MC Diversity Check] Unique gamma_eff values: {unique_gamma}/1000 "
          f"({unique_gamma/10:.1f}%) {'✓' if unique_gamma > 900 else '✗'}")

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
        'version': 'v4.1_curriculum',
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
    print(f"Training Complete.  Stage {stage_cfg.stage} complete. Global step: {stage_cfg.n_steps}")

    return model, str(ckpt_path)


def save_training_curves(metrics: List[Dict], out_dir: Path, name: str):
    """Save training curves as plots."""
    df = pd.DataFrame(metrics)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f'Training Curves: {name}', fontsize=14)

    # Loss
    axes[0, 0].plot(df['step'], df['loss'], alpha=0.7, linewidth=0.8)
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(True, alpha=0.3)

    # BER
    axes[0, 1].semilogy(df['step'], df['ber'], alpha=0.7, linewidth=0.8)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('BER')
    axes[0, 1].set_title('Bit Error Rate')
    axes[0, 1].grid(True, alpha=0.3)

    # RMSE_R
    axes[0, 2].plot(df['step'], df['rmse_R'], alpha=0.7, linewidth=0.8)
    axes[0, 2].set_xlabel('Step')
    axes[0, 2].set_ylabel('RMSE_R (m)')
    axes[0, 2].set_title('Range RMSE')
    axes[0, 2].grid(True, alpha=0.3)

    # g_PN
    axes[1, 0].plot(df['step'], df['g_pn'], alpha=0.7, linewidth=0.8)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('g_PN')
    axes[1, 0].set_title('PN Gate')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(True, alpha=0.3)

    # g_theta
    axes[1, 1].plot(df['step'], df['g_theta'], alpha=0.7, linewidth=0.8)
    axes[1, 1].set_xlabel('Step')
    axes[1, 1].set_ylabel('g_theta')
    axes[1, 1].set_title('Theta Gate')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(True, alpha=0.3)

    # delta_R
    axes[1, 2].plot(df['step'], df['delta_R'], alpha=0.7, linewidth=0.8)
    axes[1, 2].set_xlabel('Step')
    axes[1, 2].set_ylabel('delta_R (m)')
    axes[1, 2].set_title('Theta Update (ΔR)')
    axes[1, 2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_dir / f'train_curve_{name}.png', dpi=150, bbox_inches='tight')
    plt.savefig(out_dir / f'train_curve_{name}.pdf', format='pdf', bbox_inches='tight')
    plt.close()


# =============================================================================
# Curriculum Learning
# =============================================================================

def run_curriculum(cfg: TrainConfig, stages: List[int] = [1, 2, 3]):
    """
    Run curriculum learning through specified stages.

    Each stage builds on the previous stage's checkpoint.
    """
    print("\n" + "=" * 70)
    print("CURRICULUM LEARNING")
    print(f"Stages: {stages}")
    print(f"Steps per stage: {cfg.n_steps}")
    print("=" * 70)

    stage_configs = get_curriculum_stages(cfg.n_steps)

    model = None
    prev_ckpt = None
    global_step = 0

    for stage_num in stages:
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
    parser = argparse.ArgumentParser(description="GA-BV-Net Training (Expert v4.1)")
    parser.add_argument("--stage", type=int, default=1, help="Single stage to run (1, 2, or 3)")
    parser.add_argument("--steps", type=int, default=1000, help="Steps per stage")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.002, help="Base learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--curriculum", action="store_true", help="Run stages 1→2→3")
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
    print("GA-BV-Net Training (Expert Fixed v4.1)")
    print("=" * 60)
    print(f"Mode: {'Curriculum (1→2→3)' if args.curriculum else f'Single Stage {cfg.stage}'}")
    print(f"Steps per stage: {cfg.n_steps}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.lr}")
    print(f"Hardware randomization: {cfg.randomize_hardware}")
    print(f"Debug Mode: {cfg.debug_mode}")
    if cfg.debug_mode:
        print("  → theta_init = theta_true (NO NOISE)")
    print("=" * 60 + "\n")

    if args.curriculum:
        run_curriculum(cfg, stages=[1, 2, 3])
    else:
        train_one_stage(cfg)