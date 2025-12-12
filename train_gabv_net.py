"""
train_gabv_net.py (Fixed Version)

Description:
    Training Pipeline for GA-BV-Net.

    Fixes:
    - TypeError in GABVLoss.forward (Added 'stage_cfg' argument).
    - Ensure all metrics (L_geom, L_bound) are returned.
    - Added tensorboard flush.
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
    exp_name: str = "GABV_Stage1"
    seed: int = 42

    # Curriculum
    stage: int = 1

    # Training
    n_steps: int = 2000
    batch_size: int = 32
    lr: float = 1e-3
    grad_clip: float = 1.0

    # Model
    n_layers: int = 4
    hidden_dim_pn: int = 64
    cg_steps: int = 5

    # Loss Weights
    w_comm: float = 1.0
    w_sens: float = 1.0
    w_geom: float = 0.0
    w_bound: float = 0.0

    # System
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    log_dir: str = "results/train_logs"
    ckpt_dir: str = "results/checkpoints"
    eval_dir: str = "results/eval_results"

def set_stage_params(cfg: TrainConfig, sim_cfg: SimConfig):
    """Configures Physics based on Stage."""
    print(f"[Curriculum] Configuring Stage {cfg.stage}...")
    if cfg.stage == 0:
        sim_cfg.enable_pa = False; sim_cfg.enable_pn = False; sim_cfg.enable_quantization = False
        cfg.w_geom = 0.0; cfg.w_bound = 0.0
    elif cfg.stage == 1:
        sim_cfg.enable_pa = True; sim_cfg.ibo_dB = 6.0; sim_cfg.enable_pn = True; sim_cfg.pn_linewidth = 1e3
        sim_cfg.enable_quantization = True
        cfg.w_geom = 0.01; cfg.w_bound = 0.0
    elif cfg.stage == 2:
        sim_cfg.ibo_dB = 3.0; sim_cfg.pn_linewidth = 100e3
        cfg.w_geom = 0.1; cfg.w_bound = 0.1
    elif cfg.stage == 3:
        sim_cfg.ibo_dB = 0.0; sim_cfg.pn_linewidth = 500e3
        cfg.w_geom = 0.2; cfg.w_bound = 1.0

# --- 2. Dataset ---

class THzISACDataset:
    def __init__(self, train_cfg: TrainConfig):
        self.train_cfg = train_cfg
        self.sim_cfg = SimConfig()
        set_stage_params(train_cfg, self.sim_cfg)
        self.rng = np.random.default_rng(train_cfg.seed)

    def __iter__(self):
        return self

    def __next__(self):
        current_snr = self.rng.uniform(0, 30)
        self.sim_cfg.snr_db = current_snr

        # Online Simulation
        raw_data = simulate_batch(self.sim_cfg, batch_size=self.train_cfg.batch_size)

        theta_true = torch.tensor(raw_data['theta_true'], dtype=torch.float32)
        tle_noise = torch.randn_like(theta_true) * torch.tensor([100.0, 10.0, 0.5])
        theta_init = theta_true + tle_noise

        meta = torch.zeros(self.train_cfg.batch_size, 6)
        meta[:, 0] = 10**(current_snr/10) # SNR Linear
        meta[:, 1] = 0.5 if self.sim_cfg.enable_pa else 100.0

        return {
            'y_q': torch.from_numpy(raw_data['y_q']).cfloat(),
            'x_true': torch.from_numpy(raw_data['x_true']).cfloat(),
            'theta_true': theta_true,
            'theta_init': theta_init,
            'meta': meta
        }

# --- 3. Loss Engine (FIXED) ---

class GABVLoss(nn.Module):
    def __init__(self, cfg: TrainConfig):
        super().__init__()
        self.cfg = cfg
        self.mse = nn.MSELoss()

    def forward(self, outputs, batch, stage_cfg): # <--- FIX: Added stage_cfg
        # Unpack
        x_hat, theta_hat = outputs['x_hat'], outputs['theta_hat']
        x_true = batch['x_true'].to(self.cfg.device)
        theta_true = batch['theta_true'].to(self.cfg.device)
        geom_cache = outputs['geom_cache']

        # 1. Comm Loss
        L_comm = self.mse(torch.view_as_real(x_hat), torch.view_as_real(x_true))

        # 2. Sens Loss (Normalized)
        scale = torch.tensor([1e5, 1e4, 10.0], device=self.cfg.device)
        L_sens = torch.mean(((theta_hat - theta_true)/scale)**2)

        # 3. Geom Loss (LogDet)
        L_geom = torch.tensor(0.0, device=self.cfg.device)
        if geom_cache['log_vols']:
            # geom_cache['log_vols'] is a list of tensors. Use the last one.
            L_geom = -torch.mean(geom_cache['log_vols'][-1])

        # 4. Bound Loss Placeholder
        L_bound = torch.tensor(0.0, device=self.cfg.device)

        # Total Weighted Loss (Using dynamic weights from stage_cfg)
        total = (stage_cfg.w_comm * L_comm + stage_cfg.w_sens * L_sens +
                 stage_cfg.w_geom * L_geom + stage_cfg.w_bound * L_bound)

        return total, {
            "L_comm": L_comm.item(),
            "L_sens": L_sens.item(),
            "L_geom": L_geom.item(),
            "L_bound": L_bound.item()
        }

# --- 4. Training Loop ---

def train_one_stage(cfg: TrainConfig):
    # Setup Dirs
    run_id = f"Stage{cfg.stage}_{int(time.time())}"
    log_path = os.path.join(cfg.log_dir, run_id)
    ckpt_path = os.path.join(cfg.ckpt_dir, run_id)
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(ckpt_path, exist_ok=True)
    os.makedirs(cfg.eval_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=log_path)

    # Setup Model
    model_cfg = GABVConfig(n_layers=cfg.n_layers)
    model = create_gabv_model(model_cfg).to(cfg.device)
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
    dataset = THzISACDataset(cfg)
    criterion = GABVLoss(cfg)

    print(f"--- Starting Training: {run_id} ---")
    model.train()
    data_iter = iter(dataset)

    loss_history = []

    for step in range(1, cfg.n_steps + 1):
        batch = next(data_iter)
        batch_gpu = {k: v.to(cfg.device) for k, v in batch.items() if torch.is_tensor(v)}

        outputs = model(batch_gpu)

        # FIX: Pass 'cfg' (TrainConfig) to forward as stage_cfg
        loss, metrics = criterion(outputs, batch, cfg)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        # Logging
        if step % 50 == 0:
            print(f"Step {step} | Loss: {loss.item():.4f} | Comm: {metrics['L_comm']:.4f}")
            writer.add_scalar("Loss/Total", loss.item(), step)
            loss_history.append({"Step": step, "Loss": loss.item(), **metrics})

    # Save Checkpoint
    torch.save(model.state_dict(), os.path.join(ckpt_path, "final.pth"))

    # Save Training Curves
    save_training_curves(loss_history, cfg.eval_dir, run_id)

    writer.close()
    print("Training Complete.")
    return model

# --- 5. Utils ---

def save_training_curves(history, save_dir, run_id):
    """Saves training loss curves."""
    df = pd.DataFrame(history)
    base_name = os.path.join(save_dir, f"train_curve_{run_id}")

    # Save CSV
    df.to_csv(f"{base_name}.csv", index=False)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(df["Step"], df["Loss"], label="Total Loss")
    ax.plot(df["Step"], df["L_comm"], label="Comm MSE", linestyle="--")
    ax.plot(df["Step"], df["L_sens"], label="Sens MSE", linestyle=":")
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(True)

    fig.savefig(f"{base_name}.png", dpi=300)
    fig.savefig(f"{base_name}.pdf")
    plt.close(fig)
    print(f"[Result] Saved training curves to {base_name}.*")

if __name__ == "__main__":
    # Ensure dirs exist
    for d in ["results/train_logs", "results/checkpoints", "results/eval_results"]:
        os.makedirs(d, exist_ok=True)

    cfg = TrainConfig(n_steps=500, stage=1)
    train_one_stage(cfg)