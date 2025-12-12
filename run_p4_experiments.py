"""
run_p4_experiments.py (Final Hardened Version)

Description:
    Final Experiment Engine for Direction-1 Top-Tier Journal Paper (DR-P4).

    Updates vs v1:
    1. Robust FIM dimension handling ([B,3] vs [B,3,3]).
    2. Safe 'geom_cache' checking.
    3. CLI argument '--baselines' to toggle CPU-heavy benchmarks.
    4. Preserved 'h_diag' broadcasting fix.

Author: Gemini (AI Thought Partner)
Date: 2025-12-12
"""

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os
import sys
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from tqdm import tqdm

# --- Import Project Modules ---
try:
    from thz_isac_world import SimConfig, simulate_batch
    import geometry_metrics as gm
    from gabv_net_model import GABVNet, GABVConfig
    # Try importing baselines
    try:
        from baselines_receivers import detector_lmmse_bussgang, detector_gamp_1bit
        HAS_BASELINES = True
    except ImportError:
        print("Warning: 'baselines_receivers.py' not found. Baselines will be skipped.")
        HAS_BASELINES = False
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

# --- 1. Experiment Configuration ---

@dataclass
class ExperimentConfig:
    scene_id: str
    description: str
    snr_grid: List[float]
    n_mc: int = 50
    batch_size: int = 64

    # Methods to run (Controlled dynamically via CLI now)
    use_ga_bv_net: bool = True
    use_baselines: bool = False # Default off, enabled via CLI

    # Hardware Profile
    enable_pa: bool = True
    ibo_dB: float = 3.0
    enable_pn: bool = True
    pn_linewidth: float = 100e3
    enable_quantization: bool = True

    # Geometry
    R: float = 500e3
    v_rel: float = 7.5e3

def get_default_experiments() -> Dict[str, ExperimentConfig]:
    """Defines the Scenario Matrix from DR-P4."""
    base_snr = list(np.arange(0, 31, 5)) # 0 to 30 dB

    scenarios = {}

    # 1. Clean
    scenarios["clean"] = ExperimentConfig(
        scene_id="clean",
        description="Ideal Hardware (Linear PA, No PN, High Res)",
        snr_grid=base_snr,
        enable_pa=False,
        enable_pn=False,
        enable_quantization=False
    )

    # 2. Dirty Typical
    scenarios["dirty_typical"] = ExperimentConfig(
        scene_id="dirty_typical",
        description="Typical THz Impairments (IBO=3dB, LW=100kHz, 1-bit)",
        snr_grid=base_snr,
        ibo_dB=3.0,
        pn_linewidth=100e3
    )

    # 3. Dirty Extreme
    scenarios["dirty_extreme"] = ExperimentConfig(
        scene_id="dirty_extreme",
        description="Extreme Impairments (IBO=0dB, LW=500kHz, 1-bit)",
        snr_grid=base_snr,
        ibo_dB=0.0,
        pn_linewidth=500e3
    )

    # 4. Near Forbidden
    scenarios["near_forbidden"] = ExperimentConfig(
        scene_id="near_forbidden",
        description="Near Forbidden Region (LW=5MHz)",
        snr_grid=base_snr,
        ibo_dB=3.0,
        pn_linewidth=5e6
    )

    return scenarios

# --- 2. Metrics Calculation ---

def compute_ber_qpsk(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    x_hat_sign = np.sign(x_hat.real) + 1j * np.sign(x_hat.imag)
    x_true_sign = np.sign(x_true.real) + 1j * np.sign(x_true.imag)
    x_hat_sign.real[x_hat_sign.real == 0] = 1
    x_hat_sign.imag[x_hat_sign.imag == 0] = 1
    errors = np.sum(x_hat_sign != x_true_sign)
    return errors / x_true.size

def compute_nmse(x_hat: np.ndarray, x_true: np.ndarray) -> float:
    power = np.mean(np.abs(x_true)**2)
    mse = np.mean(np.abs(x_hat - x_true)**2)
    return 10 * np.log10(mse / (power + 1e-12))

def compute_rmse(theta_hat: np.ndarray, theta_true: np.ndarray) -> Tuple[float, float, float]:
    diff = theta_hat - theta_true
    mse = np.mean(diff**2, axis=0)
    rmse = np.sqrt(mse)
    return rmse[0], rmse[1], rmse[2]

# --- 3. Model & Baseline Wrappers ---

def load_gabv_model(ckpt_path: str, device: str) -> GABVNet:
    print(f"[Model] Loading GA-BV-Net from {ckpt_path}...")
    cfg = GABVConfig(n_layers=6, hidden_dim_pn=64)
    model = GABVNet(cfg)
    try:
        checkpoint = torch.load(ckpt_path, map_location=device)
        state_dict = checkpoint.get('model_state', checkpoint)
        model.load_state_dict(state_dict)
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        sys.exit(1)
    model.to(device)
    model.eval()
    return model

def run_baselines_on_batch(batch_data: dict, sim_cfg: SimConfig) -> dict:
    if not HAS_BASELINES: return {}

    y_q = batch_data['y_q']
    x_true = batch_data['x_true']
    meta = batch_data['meta']

    B_batch = y_q.shape[0]
    ber_lmmse_list, ber_gamp_list = [], []

    B_gain = meta['B_gain']
    H_diag_all = meta['h_diag']

    # Broadcast Fix
    if H_diag_all.ndim == 1:
        H_diag_all = np.tile(H_diag_all, (B_batch, 1))

    snr_lin = meta['snr_linear']
    sigma_eta = meta['sigma_eta']

    n0 = 1.0 / snr_lin
    noise_lmmse = sigma_eta + n0 + 0.5
    noise_gamp = sigma_eta + n0

    for i in range(B_batch):
        y = y_q[i]
        h_diag = H_diag_all[i] * B_gain
        H_eff = np.diag(h_diag)
        x_t = x_true[i]

        # 1. LMMSE
        x_lmmse = detector_lmmse_bussgang(y, H_eff, noise_lmmse, P_signal=1.0)
        ber_lmmse_list.append(compute_ber_qpsk(x_lmmse, x_t))

        # 2. GAMP
        try:
            x_gamp = detector_gamp_1bit(y, H_eff, noise_gamp)
            ber_gamp_list.append(compute_ber_qpsk(x_gamp, x_t))
        except:
            ber_gamp_list.append(0.5)

    return {
        "BER_LMMSE": np.mean(ber_lmmse_list),
        "BER_GAMP": np.mean(ber_gamp_list)
    }

# --- 4. Core Evaluation Loop ---

def run_single_scenario(cfg: ExperimentConfig, model: Optional[GABVNet],
                        device: str, ckpt_tag: str, out_dir: str):
    print(f"\n>>> Running Scenario: {cfg.scene_id} [{cfg.description}]")
    print(f"    Baselines: {'ON' if cfg.use_baselines else 'OFF'}")

    sim_cfg = SimConfig()
    sim_cfg.enable_pa = cfg.enable_pa
    sim_cfg.ibo_dB = cfg.ibo_dB
    sim_cfg.enable_pn = cfg.enable_pn
    sim_cfg.pn_linewidth = cfg.pn_linewidth
    sim_cfg.enable_quantization = cfg.enable_quantization
    sim_cfg.R = cfg.R
    sim_cfg.v_rel = cfg.v_rel

    results = []

    for snr_db in tqdm(cfg.snr_grid, desc="SNR Sweep"):
        sim_cfg.snr_db = snr_db

        metrics_accum = {
            "BER_Net": [], "NMSE_Net": [],
            "RMSE_R": [], "RMSE_v": [], "RMSE_a": [],
            "BCRLB_R": [], "Eff_R": []
        }
        if cfg.use_baselines:
            metrics_accum.update({"BER_LMMSE": [], "BER_GAMP": []})

        for _ in range(cfg.n_mc):
            # A. Simulate
            raw_data = simulate_batch(sim_cfg, batch_size=cfg.batch_size)

            # B. GA-BV-Net Inference
            if cfg.use_ga_bv_net and model is not None:
                theta_true_t = torch.tensor(raw_data['theta_true'], dtype=torch.float32)
                tle_noise = torch.randn_like(theta_true_t) * torch.tensor([100.0, 10.0, 0.5])
                theta_init_t = theta_true_t + tle_noise

                gamma_proxy = 0.5 if cfg.enable_pa else 100.0
                meta_t = torch.zeros(cfg.batch_size, 6)
                meta_t[:, 0] = 10**(snr_db/10)
                meta_t[:, 1] = gamma_proxy

                batch_gpu = {
                    'y_q': torch.from_numpy(raw_data['y_q']).cfloat().to(device),
                    'theta_init': theta_init_t.to(device),
                    'meta': meta_t.to(device)
                }

                with torch.no_grad():
                    outputs = model(batch_gpu)

                x_hat = outputs['x_hat'].cpu().numpy()
                x_true = raw_data['x_true']
                theta_hat = outputs['theta_hat'].cpu().numpy()
                theta_true = raw_data['theta_true']

                metrics_accum["BER_Net"].append(compute_ber_qpsk(x_hat, x_true))
                metrics_accum["NMSE_Net"].append(compute_nmse(x_hat, x_true))

                r_r, r_v, r_a = compute_rmse(theta_hat, theta_true)
                metrics_accum["RMSE_R"].append(r_r)
                metrics_accum["RMSE_v"].append(r_v)
                metrics_accum["RMSE_a"].append(r_a)

                # --- FIX 1 & 2: Robust FIM / Geom Cache Handling ---
                geom_cache = outputs.get('geom_cache', {})
                fim_invs = geom_cache.get('fim_invs', [])

                if isinstance(fim_invs, list) and len(fim_invs) > 0:
                    fim_inv = fim_invs[-1].cpu().numpy()

                    # Robust Dimension Check
                    crlb_r = np.nan
                    if fim_inv.ndim == 3: # [B, 3, 3] -> [0,0]
                        crlb_r = np.mean(fim_inv[:, 0, 0])
                    elif fim_inv.ndim == 2: # [B, 3] -> [0]
                        crlb_r = np.mean(fim_inv[:, 0])

                    if not np.isnan(crlb_r):
                        metrics_accum["BCRLB_R"].append(crlb_r)
                        mse_r = r_r**2 + 1e-12
                        metrics_accum["Eff_R"].append(crlb_r / mse_r)
                # ----------------------------------------------------

            # C. Baselines
            if cfg.use_baselines and HAS_BASELINES:
                bl_res = run_baselines_on_batch(raw_data, sim_cfg)
                metrics_accum["BER_LMMSE"].append(bl_res.get("BER_LMMSE", 1.0))
                metrics_accum["BER_GAMP"].append(bl_res.get("BER_GAMP", 1.0))

        # Aggregate
        row = {"scene_id": cfg.scene_id, "snr_db": snr_db}
        for k, v in metrics_accum.items():
            if v: row[k] = np.mean(v)
            else: row[k] = np.nan
        results.append(row)

    df = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, f"{cfg.scene_id}_{ckpt_tag}_metrics.csv")
    df.to_csv(csv_path, index=False)
    print(f"  [Saved] Metrics -> {csv_path}")
    plot_scenario_results(df, cfg, ckpt_tag, out_dir)

# --- 5. Visualization ---

def plot_scenario_results(df: pd.DataFrame, cfg: ExperimentConfig, tag: str, out_dir: str):
    fig_dir = os.path.join(out_dir, "figs")
    os.makedirs(fig_dir, exist_ok=True)

    # Plot 1: BER vs SNR
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    if cfg.use_ga_bv_net:
        ax1.semilogy(df["snr_db"], df["BER_Net"], 'b-o', label='GA-BV-Net')
    if cfg.use_baselines and "BER_LMMSE" in df.columns:
        ax1.semilogy(df["snr_db"], df["BER_LMMSE"], 'g--s', label='LMMSE (Bussgang)')
    if cfg.use_baselines and "BER_GAMP" in df.columns:
        ax1.semilogy(df["snr_db"], df["BER_GAMP"], 'r-.^', label='1-bit GAMP')

    ax1.set_xlabel("SNR (dB)")
    ax1.set_ylabel("Bit Error Rate (BER)")
    ax1.grid(True, which='both', alpha=0.3)
    ax1.legend()
    fig1.savefig(os.path.join(fig_dir, f"fig_ber_{cfg.scene_id}_{tag}.png"), dpi=300)
    fig1.savefig(os.path.join(fig_dir, f"fig_ber_{cfg.scene_id}_{tag}.pdf"))
    plt.close(fig1)

    # Plot 2: RMSE Range
    if cfg.use_ga_bv_net and "RMSE_R" in df.columns:
        fig2, ax2 = plt.subplots(figsize=(6, 5))
        ax2.semilogy(df["snr_db"], df["RMSE_R"], 'b-o', label='RMSE Range (Net)')
        if "BCRLB_R" in df.columns and not df["BCRLB_R"].isna().all():
            lb = np.sqrt(df["BCRLB_R"])
            ax2.semilogy(df["snr_db"], lb, 'k--', label='$\sqrt{BCRLB}$')
        ax2.set_xlabel("SNR (dB)")
        ax2.set_ylabel("RMSE Range (m)")
        ax2.grid(True, which='both', alpha=0.3)
        ax2.legend()
        fig2.savefig(os.path.join(fig_dir, f"fig_rmse_{cfg.scene_id}_{tag}.png"), dpi=300)
        fig2.savefig(os.path.join(fig_dir, f"fig_rmse_{cfg.scene_id}_{tag}.pdf"))
        plt.close(fig2)

# --- 6. Main Entry Point ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Path to checkpoint")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--scenes", type=str, default="all")
    parser.add_argument("--n_mc", type=int, default=20)

    # --- FIX 3: CLI Control for Baselines ---
    parser.add_argument("--baselines", type=int, default=0, help="Run Baselines? 0=No, 1=Yes")
    # ----------------------------------------

    parser.add_argument("--out_dir", type=str, default="results/p4")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    ckpt_tag = os.path.basename(os.path.dirname(args.ckpt))
    if not ckpt_tag: ckpt_tag = "custom"

    model = load_gabv_model(args.ckpt, args.device)

    all_scenarios = get_default_experiments()
    if args.scenes == "all":
        target_scenes = all_scenarios.values()
    else:
        keys = args.scenes.split(",")
        target_scenes = [all_scenarios[k] for k in keys if k in all_scenarios]

    print(f"--- Launching P4 Experiments (Tag: {ckpt_tag}) ---")
    print(f"    Mode: {'WITH Baselines' if args.baselines else 'Net ONLY (Fast)'}")

    for sc in target_scenes:
        sc.n_mc = args.n_mc
        sc.use_baselines = bool(args.baselines) # Override config
        run_single_scenario(sc, model, args.device, ckpt_tag, args.out_dir)

    print(f"\n[Done] All results saved to {args.out_dir}")

if __name__ == "__main__":
    main()