"""
train_gabv_net.py (Gamma_eff/Chi Closure Version)

Description:
    Training Pipeline for GA-BV-Net.

    **KEY UPDATE** per DR-P2-3 注意事项:
    - REMOVED gamma_proxy (0.5/100) - NO LONGER USED
    - Meta features now use REAL gamma_eff, chi from thz_isac_world
    - Meta feature vector DEFINITION FREEZE (6-dim)

    Meta Feature Definition (Frozen):
        [0] snr_db_norm      - Normalized SNR in dB
        [1] gamma_eff_db_norm - Normalized Gamma_eff in dB
        [2] chi              - Information retention factor (raw)
        [3] sigma_eta_norm   - Normalized PA distortion power
        [4] pn_linewidth_norm - Normalized PN linewidth
        [5] ibo_db_norm      - Normalized IBO
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
    # These are used to normalize meta features to roughly [-1, 1] or [0, 1]
    snr_db_center: float = 15.0       # Center of SNR range
    snr_db_scale: float = 15.0        # Half-width of SNR range
    gamma_eff_db_center: float = 10.0 # Center of Gamma_eff range in dB
    gamma_eff_db_scale: float = 20.0  # Half-width
    sigma_eta_scale: float = 0.1      # Typical PA distortion power
    pn_linewidth_scale: float = 1e6   # 1 MHz reference
    ibo_db_center: float = 3.0        # Typical IBO
    ibo_db_scale: float = 3.0         # Half-width


def set_stage_params(cfg: TrainConfig, sim_cfg: SimConfig):
    """Configures Physics based on Stage."""
    print(f"[Curriculum] Configuring Stage {cfg.stage}...")
    if cfg.stage == 0:
        sim_cfg.enable_pa = False
        sim_cfg.enable_pn = False
        sim_cfg.enable_quantization = False
        cfg.w_geom = 0.0
        cfg.w_bound = 0.0
    elif cfg.stage == 1:
        sim_cfg.enable_pa = True
        sim_cfg.ibo_dB = 6.0
        sim_cfg.enable_pn = True
        sim_cfg.pn_linewidth = 1e3
        sim_cfg.enable_quantization = True
        cfg.w_geom = 0.01
        cfg.w_bound = 0.0
    elif cfg.stage == 2:
        sim_cfg.ibo_dB = 3.0
        sim_cfg.pn_linewidth = 500e3
        cfg.w_geom = 0.1
        cfg.w_bound = 0.1
    elif cfg.stage == 3:
        sim_cfg.ibo_dB = 0.0
        sim_cfg.pn_linewidth = 100e4
        cfg.w_geom = 0.2
        cfg.w_bound = 1.0


# --- 2. Meta Feature Construction (Definition Freeze) ---

def construct_meta_features(raw_meta: dict, train_cfg: TrainConfig, batch_size: int) -> torch.Tensor:
    """
    Constructs normalized meta feature tensor from simulation metadata.

    **DEFINITION FREEZE** - This is the canonical meta feature vector:
        [0] snr_db_norm       - (snr_db - center) / scale
        [1] gamma_eff_db_norm - (gamma_eff_dB - center) / scale
        [2] chi               - Raw chi value (already in [0, 2/π])
        [3] sigma_eta_norm    - sigma_eta / scale
        [4] pn_linewidth_norm - pn_linewidth / scale (log-scaled)
        [5] ibo_db_norm       - (ibo_dB - center) / scale

    Args:
        raw_meta: Meta dict from simulate_batch()
        train_cfg: Training config with normalization constants
        batch_size: Number of samples

    Returns:
        meta_tensor: [B, 6] float tensor
    """
    meta = torch.zeros(batch_size, 6)

    # [0] SNR normalized
    snr_db = raw_meta['snr_db']
    meta[:, 0] = (snr_db - train_cfg.snr_db_center) / train_cfg.snr_db_scale

    # [1] Gamma_eff normalized (in dB)
    gamma_eff = raw_meta['gamma_eff']
    gamma_eff_db = 10 * np.log10(gamma_eff + 1e-12)
    meta[:, 1] = (gamma_eff_db - train_cfg.gamma_eff_db_center) / train_cfg.gamma_eff_db_scale

    # [2] Chi (raw value, already normalized to [0, 2/π])
    chi = raw_meta['chi']
    meta[:, 2] = chi

    # [3] Sigma_eta normalized
    sigma_eta = raw_meta['sigma_eta']
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

        # --- KEY CHANGE: Use construct_meta_features instead of gamma_proxy ---
        meta = construct_meta_features(
            raw_meta=raw_data['meta'],
            train_cfg=self.train_cfg,
            batch_size=self.train_cfg.batch_size
        )

        # Store raw values for logging
        raw_gamma_eff = raw_data['meta']['gamma_eff']
        raw_chi = raw_data['meta']['chi']
        raw_sinr_eff = raw_data['meta']['sinr_eff']

        return {
            'y_q': torch.from_numpy(raw_data['y_q']).cfloat(),
            'x_true': torch.from_numpy(raw_data['x_true']).cfloat(),
            'theta_true': theta_true,
            'theta_init': theta_init,
            'meta': meta,
            # Raw values for monitoring (not used in network)
            '_raw_gamma_eff': raw_gamma_eff,
            '_raw_chi': raw_chi,
            '_raw_sinr_eff': raw_sinr_eff,
            '_raw_snr_db': current_snr,
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
    print(f"    Using REAL gamma_eff/chi (gamma_proxy REMOVED)")
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
            # Extract Gate Values for Physics Analysis
            last_layer_gates = outputs['layers'][-1]['gates']
            g_pn_val = last_layer_gates['g_PN'].mean().item()
            g_nl_val = last_layer_gates['g_NL'].mean().item()

            # Extract raw metrics for logging
            raw_gamma = batch.get('_raw_gamma_eff', 0)
            raw_chi = batch.get('_raw_chi', 0)
            raw_snr = batch.get('_raw_snr_db', 0)

            print(f"Step {step} | Loss: {loss.item():.4f} | Comm: {metrics['L_comm']:.4f} | "
                  f"g_PN: {g_pn_val:.2f} | χ: {raw_chi:.3f} | Γ_eff: {raw_gamma:.2f}")

            writer.add_scalar("Loss/Total", loss.item(), step)
            writer.add_scalar("Gates/g_PN", g_pn_val, step)
            writer.add_scalar("Physics/chi", raw_chi, step)
            writer.add_scalar("Physics/gamma_eff", raw_gamma, step)
            writer.add_scalar("Physics/snr_db", raw_snr, step)

            # Record detailed metrics for CSV
            record = {
                "Step": step,
                "Loss": loss.item(),
                **metrics,
                "g_PN": g_pn_val,
                "g_NL": g_nl_val,
                "gamma_eff": raw_gamma,
                "chi": raw_chi,
                "snr_db": raw_snr,
            }
            loss_history.append(record)

    # Save checkpoint with meta
    checkpoint = {
        'model_state': model.state_dict(),
        'config': {
            'stage': cfg.stage,
            'n_layers': cfg.n_layers,
            'n_steps': cfg.n_steps,
        },
        'meta_feature_definition': {
            'version': 'v2_definition_freeze',
            'dims': ['snr_db_norm', 'gamma_eff_db_norm', 'chi',
                     'sigma_eta_norm', 'pn_linewidth_norm', 'ibo_db_norm'],
            'note': 'gamma_proxy REMOVED, using real gamma_eff/chi'
        }
    }
    torch.save(checkpoint, os.path.join(ckpt_path, "final.pth"))

    save_training_curves(loss_history, cfg.eval_dir, run_id)

    writer.close()
    print("Training Complete.")
    return model


# --- 6. Utils ---

def save_training_curves(history, save_dir, run_id):
    """Saves training loss & gate curves with gamma_eff/chi info."""
    df = pd.DataFrame(history)
    base_name = os.path.join(save_dir, f"train_curve_{run_id}")

    # 1. Save CSV (Journal Data Source)
    df.to_csv(f"{base_name}.csv", index=False)

    # 2. Plot Loss and Physics
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # Top-Left: Loss
    axes[0, 0].plot(df["Step"], df["Loss"], label="Total Loss", color='blue')
    axes[0, 0].plot(df["Step"], df["L_comm"], label="Comm MSE", linestyle="--", color='cyan')
    axes[0, 0].set_xlabel("Steps")
    axes[0, 0].set_ylabel("Loss")
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    axes[0, 0].set_title("Training Loss")

    # Top-Right: Gates
    axes[0, 1].plot(df["Step"], df["g_PN"], label="g_PN", color='red')
    axes[0, 1].plot(df["Step"], df["g_NL"], label="g_NL", color='orange')
    axes[0, 1].set_xlabel("Steps")
    axes[0, 1].set_ylabel("Gate Value")
    axes[0, 1].legend()
    axes[0, 1].grid(True)
    axes[0, 1].set_title("Physics Gates")

    # Bottom-Left: Gamma_eff and Chi
    ax3 = axes[1, 0]
    ax3.plot(df["Step"], df["gamma_eff"], label=r"$\Gamma_{eff}$", color='green')
    ax3.set_xlabel("Steps")
    ax3.set_ylabel(r"$\Gamma_{eff}$ (linear)")
    ax3.legend(loc='upper left')
    ax3.grid(True)

    ax3_twin = ax3.twinx()
    ax3_twin.plot(df["Step"], df["chi"], label=r"$\chi$", color='purple', linestyle='--')
    ax3_twin.set_ylabel(r"$\chi$")
    ax3_twin.legend(loc='upper right')
    axes[1, 0].set_title(r"Hardware Quality: $\Gamma_{eff}$ and $\chi$")

    # Bottom-Right: SNR variation
    axes[1, 1].plot(df["Step"], df["snr_db"], label="SNR (dB)", color='brown', alpha=0.7)
    axes[1, 1].set_xlabel("Steps")
    axes[1, 1].set_ylabel("SNR (dB)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    axes[1, 1].set_title("Training SNR Variation")

    plt.suptitle(f"Training Dynamics: {run_id}\n(gamma_proxy REMOVED, using real gamma_eff/chi)")
    fig.tight_layout()

    fig.savefig(f"{base_name}.png", dpi=300)
    fig.savefig(f"{base_name}.pdf")
    plt.close(fig)
    print(f"[Result] Saved training curves to {base_name}.*")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, default=1, help="Curriculum Stage")
    parser.add_argument("--steps", type=int, default=2000, help="Training Steps")
    args = parser.parse_args()

    # Ensure dirs exist
    for d in ["results/train_logs", "results/checkpoints", "results/eval_results"]:
        os.makedirs(d, exist_ok=True)

    # Init Config with Args
    cfg = TrainConfig(stage=args.stage, n_steps=args.steps)

    print("=" * 60)
    print("GA-BV-Net Training (Gamma_eff/Chi Closure Version)")
    print("=" * 60)
    print("NOTE: gamma_proxy has been REMOVED")
    print("      Now using REAL gamma_eff and chi from thz_isac_world")
    print("=" * 60)

    train_one_stage(cfg)