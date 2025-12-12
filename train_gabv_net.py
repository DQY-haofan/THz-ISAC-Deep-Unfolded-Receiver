"""
train_gabv_net.py (Journal-Ready Version)

Description:
    Training Pipeline for GA-BV-Net.

    Fixes:
    - Command line arguments now correctly override defaults.
    - Logs Gate values (g_PN, g_NL) to CSV for physics-aware analysis.
    - Unique filenames for each run.
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
    batch_size: int = 1024
    lr: float = 1e-3
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
        raw_data = simulate_batch(self.sim_cfg, batch_size=self.train_cfg.batch_size)

        theta_true = torch.tensor(raw_data['theta_true'], dtype=torch.float32)
        tle_noise = torch.randn_like(theta_true) * torch.tensor([100.0, 10.0, 0.5])
        theta_init = theta_true + tle_noise

        meta = torch.zeros(self.train_cfg.batch_size, 6)
        meta[:, 0] = 10**(current_snr/10)
        meta[:, 1] = 0.5 if self.sim_cfg.enable_pa else 100.0

        return {
            'y_q': torch.from_numpy(raw_data['y_q']).cfloat(),
            'x_true': torch.from_numpy(raw_data['x_true']).cfloat(),
            'theta_true': theta_true,
            'theta_init': theta_init,
            'meta': meta
        }

# --- 3. Loss Engine ---

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

# --- 4. Training Loop ---

def train_one_stage(cfg: TrainConfig):
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
        loss, metrics = criterion(outputs, batch, cfg)

        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()

        if step % 50 == 0:
            # Extract Gate Values for Physics Analysis
            last_layer_gates = outputs['layers'][-1]['gates']
            g_pn_val = last_layer_gates['g_PN'].mean().item()
            g_nl_val = last_layer_gates['g_NL'].mean().item()

            print(f"Step {step} | Loss: {loss.item():.4f} | Comm: {metrics['L_comm']:.4f} | g_PN: {g_pn_val:.2f}")

            writer.add_scalar("Loss/Total", loss.item(), step)
            writer.add_scalar("Gates/g_PN", g_pn_val, step)

            # Record detailed metrics for CSV
            record = {
                "Step": step,
                "Loss": loss.item(),
                **metrics,
                "g_PN": g_pn_val,
                "g_NL": g_nl_val
            }
            loss_history.append(record)

    torch.save(model.state_dict(), os.path.join(ckpt_path, "final.pth"))
    save_training_curves(loss_history, cfg.eval_dir, run_id)

    writer.close()
    print("Training Complete.")
    return model

# --- 5. Utils ---

def save_training_curves(history, save_dir, run_id):
    """Saves training loss & gate curves."""
    df = pd.DataFrame(history)
    base_name = os.path.join(save_dir, f"train_curve_{run_id}")

    # 1. Save CSV (Journal Data Source)
    df.to_csv(f"{base_name}.csv", index=False)

    # 2. Plot Loss
    fig, ax = plt.subplots(figsize=(10, 4))

    # Left Axis: Loss
    ax.plot(df["Step"], df["Loss"], label="Total Loss", color='blue')
    ax.plot(df["Step"], df["L_comm"], label="Comm MSE", linestyle="--", color='cyan')
    ax.set_xlabel("Steps")
    ax.set_ylabel("Loss")
    ax.set_yscale("linear") # or log if needed
    ax.legend(loc='upper left')
    ax.grid(True)

    # Right Axis: Gates
    ax2 = ax.twinx()
    ax2.plot(df["Step"], df["g_PN"], label="g_PN (Gate)", color='red', alpha=0.6)
    ax2.set_ylabel("Gate Value (0-1)")
    ax2.legend(loc='upper right')

    plt.title(f"Training Dynamics: {run_id}")
    fig.tight_layout()

    fig.savefig(f"{base_name}.png", dpi=300)
    fig.savefig(f"{base_name}.pdf")
    plt.close(fig)
    print(f"[Result] Saved training curves to {base_name}.*")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # Add arguments to override Config defaults
    parser.add_argument("--stage", type=int, default=1, help="Curriculum Stage")
    parser.add_argument("--steps", type=int, default=2000, help="Training Steps")
    args = parser.parse_args()

    # Ensure dirs exist
    for d in ["results/train_logs", "results/checkpoints", "results/eval_results"]:
        os.makedirs(d, exist_ok=True)

    # Init Config with Args
    cfg = TrainConfig(stage=args.stage, n_steps=args.steps)

    train_one_stage(cfg)