"""
train_gabv_net.py (MC Randomness Fixed - v3.1)

CRITICAL FIXES:
1. Global RNG with unique seeds per batch (not per stage)
2. Hardware parameter randomization for gamma_eff diversity
3. Seed tracking in output for verification

Usage:
    python train_gabv_net.py --stage 1 --steps 2000
    python train_gabv_net.py --curriculum  # Run stages 1→2→3 with continuous RNG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import os
import argparse
import time
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from dataclasses import dataclass

# --- Import Project Modules ---
try:
    from thz_isac_world import SimConfig, simulate_batch
    import geometry_metrics as gm
    from gabv_net_model import GABVNet, GABVConfig, create_gabv_model
except ImportError as e:
    print(f"Error importing modules: {e}")
    exit(1)


# --- 1. Configuration ---

@dataclass
class TrainConfig:
    exp_name: str = "GABV"
    seed: int = 42
    stage: int = 1
    n_steps: int = 2000
    batch_size: int = 512
    lr: float = 2e-3
    grad_clip: float = 1.0
    n_layers: int = 4
    hidden_dim_pn: int = 64
    cg_steps: int = 5

    # Loss Weights (Defaults)
    w_comm: float = 1.0
    w_sens: float = 1.0
    w_geom: float = 0.0
    w_bound: float = 0.0

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "results/train_logs"
    ckpt_dir: str = "results/checkpoints"
    eval_dir: str = "results/eval_results"

    # --- Meta Feature Normalization Constants (Definition Freeze) ---
    snr_db_center: float = 15.0
    snr_db_scale: float = 15.0
    gamma_eff_db_center: float = 10.0
    gamma_eff_db_scale: float = 20.0
    sigma_eta_scale: float = 0.1
    pn_linewidth_scale: float = 1e6
    ibo_db_center: float = 3.0
    ibo_db_scale: float = 3.0

    # --- NEW: Hardware Randomization for Gamma_eff Diversity ---
    randomize_hardware: bool = True
    ibo_range: tuple = (1.0, 8.0)        # IBO range in dB
    pn_linewidth_range: tuple = (1e3, 1e6)  # PN linewidth range in Hz


# Global step counter for unique seeds across all stages
_global_step_counter = 0


def get_global_step():
    """Get current global step (unique across all stages)."""
    global _global_step_counter
    return _global_step_counter


def increment_global_step():
    """Increment and return global step."""
    global _global_step_counter
    _global_step_counter += 1
    return _global_step_counter


def reset_global_step():
    """Reset global step (only for fresh training runs)."""
    global _global_step_counter
    _global_step_counter = 0


def set_stage_params(cfg: TrainConfig, sim_cfg: SimConfig):
    """Configures Physics based on Stage (BASE values - will be randomized)."""
    print(f"[Curriculum] Configuring Stage {cfg.stage}...")
    if cfg.stage == 0:
        sim_cfg.enable_pa = False
        sim_cfg.enable_pn = False
        sim_cfg.enable_quantization = False
        cfg.w_geom = 0.0
        cfg.w_bound = 0.0
    elif cfg.stage == 1:
        sim_cfg.enable_pa = True
        sim_cfg.ibo_dB = 6.0  # Base value
        sim_cfg.enable_pn = True
        sim_cfg.pn_linewidth = 1e3  # Base value
        sim_cfg.enable_quantization = True
        cfg.w_geom = 0.01
        cfg.w_bound = 0.0
    elif cfg.stage == 2:
        sim_cfg.enable_pa = True
        sim_cfg.ibo_dB = 3.0  # Base value
        sim_cfg.enable_pn = True
        sim_cfg.pn_linewidth = 500e3  # Base value
        sim_cfg.enable_quantization = True
        cfg.w_geom = 0.05
        cfg.w_bound = 0.01
    elif cfg.stage == 3:
        sim_cfg.enable_pa = True
        sim_cfg.ibo_dB = 0.0  # Base value
        sim_cfg.enable_pn = True
        sim_cfg.pn_linewidth = 1e6  # Base value
        sim_cfg.enable_quantization = True
        cfg.w_geom = 0.1
        cfg.w_bound = 0.1


# --- 2. Meta Features ---

def construct_meta_features(raw_meta: dict, train_cfg: TrainConfig, batch_size: int) -> np.ndarray:
    """
    Construct meta feature vector from simulation metadata.

    Definition Freeze v3 (6-dim):
        [0] snr_db_norm       - Normalized SNR in dB
        [1] gamma_eff_db_norm - Normalized Gamma_eff in dB
        [2] chi               - Information retention factor (raw)
        [3] sigma_eta_norm    - Normalized PA distortion power
        [4] pn_linewidth_norm - Normalized PN linewidth (log scale)
        [5] ibo_db_norm       - Normalized IBO
    """
    meta = np.zeros((batch_size, 6), dtype=np.float32)

    # [0] SNR normalized
    snr_db = raw_meta.get('snr_db', 20.0)
    meta[:, 0] = (snr_db - train_cfg.snr_db_center) / train_cfg.snr_db_scale

    # [1] Gamma_eff normalized (in dB)
    gamma_eff = raw_meta.get('gamma_eff', 1.0)
    gamma_eff_db = 10 * np.log10(gamma_eff + 1e-12)
    meta[:, 1] = (gamma_eff_db - train_cfg.gamma_eff_db_center) / train_cfg.gamma_eff_db_scale

    # [2] Chi (raw value, already in [0, 1])
    chi = raw_meta.get('chi', 0.6)
    meta[:, 2] = chi

    # [3] Sigma_eta normalized
    sigma_eta = raw_meta.get('sigma_eta', 0.0)
    meta[:, 3] = sigma_eta / train_cfg.sigma_eta_scale

    # [4] PN Linewidth normalized (log-scale)
    pn_linewidth = raw_meta.get('pn_linewidth', 100e3)
    meta[:, 4] = np.log10(pn_linewidth + 1) / np.log10(train_cfg.pn_linewidth_scale)

    # [5] IBO normalized
    ibo_db = raw_meta.get('ibo_dB', 3.0)
    meta[:, 5] = (ibo_db - train_cfg.ibo_db_center) / train_cfg.ibo_db_scale

    return meta


# --- 3. Dataset ---

class THzISACDataset:
    """
    Dataset with proper MC randomness:
    - Uses global master RNG for consistent seed generation
    - Unique seed for each batch
    - Optional hardware parameter randomization
    """

    def __init__(self, train_cfg: TrainConfig, master_rng: np.random.Generator):
        self.train_cfg = train_cfg
        self.sim_cfg = SimConfig()
        set_stage_params(train_cfg, self.sim_cfg)
        self.master_rng = master_rng  # Shared across stages
        self.step_counter = 0

        # Track gamma_eff diversity
        self.gamma_eff_history = []

    def __iter__(self):
        return self

    def __next__(self):
        # Randomize SNR
        current_snr = self.master_rng.uniform(0, 30)
        self.sim_cfg.snr_db = current_snr

        # NEW: Randomize hardware parameters for diversity
        if self.train_cfg.randomize_hardware:
            self.sim_cfg.ibo_dB = self.master_rng.uniform(*self.train_cfg.ibo_range)
            self.sim_cfg.pn_linewidth = 10 ** self.master_rng.uniform(
                np.log10(self.train_cfg.pn_linewidth_range[0]),
                np.log10(self.train_cfg.pn_linewidth_range[1])
            )

        # Generate unique seed for this batch
        batch_seed = self.master_rng.integers(0, 2**31)

        # Simulate with unique seed
        raw_data = simulate_batch(self.sim_cfg, batch_size=self.train_cfg.batch_size, seed=batch_seed)

        theta_true = torch.tensor(raw_data['theta_true'], dtype=torch.float32)
        tle_noise = torch.randn_like(theta_true) * torch.tensor([100.0, 10.0, 0.5])
        theta_init = theta_true + tle_noise

        # Construct meta features
        meta = construct_meta_features(
            raw_meta=raw_data['meta'],
            train_cfg=self.train_cfg,
            batch_size=self.train_cfg.batch_size
        )

        # Store raw values for logging
        raw_gamma_eff = raw_data['meta']['gamma_eff']
        raw_chi = raw_data['meta']['chi']
        raw_sinr_eff = raw_data['meta']['sinr_eff']
        seed_used = raw_data['meta'].get('seed_used', batch_seed)

        # Track diversity
        self.gamma_eff_history.append(raw_gamma_eff)
        self.step_counter += 1
        increment_global_step()

        return {
            'y_q': torch.from_numpy(raw_data['y_q']).cfloat(),
            'x_true': torch.from_numpy(raw_data['x_true']).cfloat(),
            'theta_true': theta_true,
            'theta_init': theta_init,
            'meta': meta,
            # Raw values for monitoring
            '_raw_gamma_eff': raw_gamma_eff,
            '_raw_chi': raw_chi,
            '_raw_sinr_eff': raw_sinr_eff,
            '_raw_snr_db': current_snr,
            '_seed_used': seed_used,
            '_ibo_dB': self.sim_cfg.ibo_dB,
            '_pn_linewidth': self.sim_cfg.pn_linewidth,
        }

    def get_diversity_stats(self) -> dict:
        """Return diversity statistics for gamma_eff."""
        if not self.gamma_eff_history:
            return {'unique_ratio': 0, 'total': 0, 'unique': 0}

        # Round to 2 decimal places for comparison
        rounded = [round(g, 2) for g in self.gamma_eff_history]
        unique = len(set(rounded))
        total = len(rounded)

        return {
            'unique_ratio': unique / total,
            'total': total,
            'unique': unique
        }


# --- 4. Loss Engine ---

class GABVLoss(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.mse = nn.MSELoss()

    def forward(self, outputs, batch, stage_cfg):
        x_hat, theta_hat = outputs['x_hat'], outputs['theta_hat']
        x_true = batch['x_true'].to(self.cfg.device)
        theta_true = batch['theta_true'].to(self.cfg.device)
        geom_cache = outputs['geom_cache']

        L_comm = self.mse(torch.view_as_real(x_hat), torch.view_as_real(x_true))

        scale = torch.tensor([1e5, 1e4, 10.0], device=self.cfg.device)
        L_sens = torch.mean(((theta_hat - theta_true)/scale)**2)

        L_geom = torch.tensor(0.0, device=self.cfg.device)
        if geom_cache['log_vols']:
            L_geom = -torch.mean(geom_cache['log_vols'][-1])

        L_bound = torch.tensor(0.0, device=self.cfg.device)

        total = (stage_cfg.w_comm * L_comm + stage_cfg.w_sens * L_sens +
                 stage_cfg.w_geom * L_geom + stage_cfg.w_bound * L_bound)

        return total, {
            "L_comm": L_comm.item(),
            "L_sens": L_sens.item(),
            "L_geom": L_geom.item(),
            "L_bound": L_bound.item()
        }


# --- 5. Training Loop ---

def train_one_stage(cfg: TrainConfig, master_rng: np.random.Generator = None):
    """
    Train one stage with proper MC randomness.

    Args:
        cfg: TrainConfig instance
        master_rng: Optional shared RNG (for curriculum training)

    Returns:
        master_rng: Updated RNG for next stage
    """
    # Create or use shared master RNG
    if master_rng is None:
        master_rng = np.random.default_rng(cfg.seed)
        reset_global_step()

    # Unique Run ID
    run_id = f"Stage{cfg.stage}_{int(time.time())}"
    log_path = os.path.join(cfg.log_dir, run_id)
    ckpt_path = os.path.join(cfg.ckpt_dir, run_id)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(cfg.eval_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_path)

    model_cfg = GABVConfig(n_layers=cfg.n_layers)
    model = create_gabv_model(model_cfg).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    dataset = THzISACDataset(cfg, master_rng)
    criterion = GABVLoss(cfg)

    print(f"--- Starting Training: {run_id} ---")
    print(f"    Using REAL gamma_eff/chi (gamma_proxy REMOVED)")
    print(f"    MC Randomness: FIXED (unique seed per batch)")
    print(f"    Hardware Randomization: {'ON' if cfg.randomize_hardware else 'OFF'}")

    model.train()
    data_iter = iter(dataset)

    loss_history = []

    for step in range(1, cfg.n_steps + 1):
        batch = next(data_iter)
        batch_gpu = {k: v.to(cfg.device) for k, v in batch.items()
                     if torch.is_tensor(v) and not k.startswith('_')}

        outputs = model(batch_gpu)
        loss, metrics = criterion(outputs, batch, cfg)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            # Extract Gate Values
            last_layer_gates = outputs['layers'][-1]['gates']
            g_pn_val = last_layer_gates['g_PN'].mean().item()
            g_nl_val = last_layer_gates['g_NL'].mean().item()

            # Extract raw metrics
            raw_gamma = batch.get('_raw_gamma_eff', 0)
            raw_chi = batch.get('_raw_chi', 0)
            raw_snr = batch.get('_raw_snr_db', 0)
            seed_used = batch.get('_seed_used', 0)
            ibo_db = batch.get('_ibo_dB', 0)
            pn_lw = batch.get('_pn_linewidth', 0)

            print(f"Step {step} | Loss: {loss.item():.4f} | Comm: {metrics['L_comm']:.4f} | "
                  f"g_PN: {g_pn_val:.2f} | χ: {raw_chi:.3f} | Γ_eff: {raw_gamma:.2f} | "
                  f"seed: {seed_used}")

            writer.add_scalar("Loss/Total", loss.item(), step)
            writer.add_scalar("Gates/g_PN", g_pn_val, step)
            writer.add_scalar("Physics/chi", raw_chi, step)
            writer.add_scalar("Physics/gamma_eff", raw_gamma, step)
            writer.add_scalar("Physics/snr_db", raw_snr, step)
            writer.add_scalar("Physics/ibo_dB", ibo_db, step)
            writer.add_scalar("Physics/pn_linewidth", pn_lw, step)
            writer.add_scalar("MC/seed", seed_used, step)

            # Record detailed metrics
            record = {
                "Step": step,
                "Loss": loss.item(),
                **metrics,
                "g_PN": g_pn_val,
                "g_NL": g_nl_val,
                "gamma_eff": raw_gamma,
                "chi": raw_chi,
                "snr_db": raw_snr,
                "ibo_dB": ibo_db,
                "pn_linewidth": pn_lw,
                "seed": seed_used,
            }
            loss_history.append(record)

    # Diversity check
    diversity = dataset.get_diversity_stats()
    print(f"\n [MC Diversity Check] Unique gamma_eff values: {diversity['unique']}/{diversity['total']} "
          f"({diversity['unique_ratio']*100:.1f}%)")

    if diversity['unique_ratio'] < 0.5:
        print("⚠️  WARNING: Low diversity in gamma_eff - check hardware randomization!")
    else:
        print("✓  Good diversity in gamma_eff")

    # Save checkpoint with meta
    checkpoint = {
        'model_state': model.state_dict(),
        'config': {
            'stage': cfg.stage,
            'n_steps': cfg.n_steps,
            'n_layers': cfg.n_layers,
            'lr': cfg.lr,
        },
        'version': 'v3.1_mc_fixed',
        'mc_randomness_info': {
            'fixed': True,
            'diversity_ratio': diversity['unique_ratio'],
            'hardware_randomization': cfg.randomize_hardware,
            'global_step_end': get_global_step(),
        }
    }
    torch.save(checkpoint, os.path.join(ckpt_path, "final.pth"))

    # Save training history
    df = pd.DataFrame(loss_history)
    csv_path = os.path.join(cfg.eval_dir, f"train_curve_{run_id}.csv")
    df.to_csv(csv_path, index=False)

    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    axes[0, 0].plot(df['Step'], df['Loss'])
    axes[0, 0].set_xlabel('Step')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training Loss')

    axes[0, 1].plot(df['Step'], df['gamma_eff'], label='Γ_eff', alpha=0.7)
    axes[0, 1].set_xlabel('Step')
    axes[0, 1].set_ylabel('Γ_eff')
    axes[0, 1].set_title('Gamma_eff Over Training')

    axes[1, 0].plot(df['Step'], df['chi'], label='χ', alpha=0.7)
    axes[1, 0].set_xlabel('Step')
    axes[1, 0].set_ylabel('χ')
    axes[1, 0].set_title('Chi Over Training')

    axes[1, 1].scatter(df['gamma_eff'], df['chi'], alpha=0.3, s=10)
    axes[1, 1].set_xlabel('Γ_eff')
    axes[1, 1].set_ylabel('χ')
    axes[1, 1].set_title('Γ_eff vs χ (should show variation)')

    fig.tight_layout()
    fig.savefig(os.path.join(cfg.eval_dir, f"train_curve_{run_id}.png"), dpi=150)
    fig.savefig(os.path.join(cfg.eval_dir, f"train_curve_{run_id}.pdf"))
    plt.close(fig)

    print(f"[Result] Saved training curves to {cfg.eval_dir}/train_curve_{run_id}.*")
    print("Training Complete.\n")

    writer.close()

    return master_rng  # Return for curriculum training


def run_curriculum(cfg: TrainConfig, stages=[1, 2, 3]):
    """
    Run curriculum training with shared RNG across stages.

    This ensures:
    - Each stage uses different seeds (not the same sequence)
    - Global step counter increases continuously
    """
    print("=" * 60)
    print("CURRICULUM TRAINING (MC Randomness Fixed)")
    print("=" * 60)
    print(f"Stages: {stages}")
    print(f"Base seed: {cfg.seed}")
    print(f"Hardware randomization: {cfg.randomize_hardware}")
    print("=" * 60)

    # Create master RNG once
    master_rng = np.random.default_rng(cfg.seed)
    reset_global_step()

    for stage in stages:
        cfg.stage = stage
        print(f"\n{'='*60}")
        print(f"STAGE {stage}")
        print(f"{'='*60}")

        # Pass and update master RNG
        master_rng = train_one_stage(cfg, master_rng)

        print(f"Stage {stage} complete. Global step: {get_global_step()}")

    print("\n" + "=" * 60)
    print("CURRICULUM TRAINING COMPLETE")
    print(f"Total steps across all stages: {get_global_step()}")
    print("=" * 60)


# --- Main ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--curriculum", action="store_true", help="Run stages 1→2→3")
    parser.add_argument("--no-hw-random", action="store_true", help="Disable hardware randomization")
    args = parser.parse_args()

    cfg = TrainConfig(
        stage=args.stage,
        n_steps=args.steps,
        seed=args.seed,
        randomize_hardware=not args.no_hw_random
    )

    if args.curriculum:
        run_curriculum(cfg, stages=[1, 2, 3])
    else:
        train_one_stage(cfg)