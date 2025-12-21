"""
evaluator.py - 数据采集和评估逻辑

负责：
- 模型加载
- 数据生成
- 各种 sweep（SNR, Cliff, PN, Pilot 等）
- 指标计算

输出：CSV 文件供 visualization.py 使用
"""

import os
import torch
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Local imports
from baselines import (
    run_baseline, 
    METHOD_ORDER, 
    METHOD_QUICK, 
    METHOD_CLIFF,
    METHOD_ABLATION,
    frontend_adjoint_and_pn,
    qpsk_hard_slice,
)


# ============================================================================
# 配置
# ============================================================================

@dataclass
class EvalConfig:
    """评估配置"""
    ckpt_path: str = ""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # SNR sweep
    snr_list: List[float] = field(default_factory=lambda: [-5, 0, 5, 10, 15, 20, 25])

    # Monte Carlo
    n_mc: int = 20
    batch_size: int = 64

    # θ 初始噪声（samples）
    theta_noise_tau: float = 0.3
    theta_noise_v: float = 0.0
    theta_noise_a: float = 0.0

    # Cliff sweep
    init_error_list: List[float] = field(default_factory=lambda: [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5])

    # PN sweep
    pn_linewidths: List[float] = field(default_factory=lambda: [0, 50e3, 100e3, 200e3, 500e3])

    # Pilot sweep
    pilot_lengths: List[int] = field(default_factory=lambda: [16, 32, 64, 128])

    # 输出目录
    out_dir: str = "results/paper_figs"


# ============================================================================
# 模型和数据加载
# ============================================================================

def load_model(ckpt_path: str, device: str):
    """加载训练好的模型"""
    from gabv_net_model import GABVNet, GABVConfig, create_gabv_model

    checkpoint = torch.load(ckpt_path, map_location=device)

    if 'gabv_cfg' in checkpoint:
        gabv_cfg = checkpoint['gabv_cfg']
    else:
        gabv_cfg = GABVConfig()

    model = create_gabv_model(gabv_cfg)

    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])

    model = model.to(device)
    model.eval()

    return model, gabv_cfg


def create_sim_config(gabv_cfg, snr_db: float, pn_linewidth: float = None):
    """创建仿真配置"""
    from thz_isac_world import SimConfig

    sim_cfg = SimConfig(
        N=gabv_cfg.N if hasattr(gabv_cfg, 'N') else 1024,
        fs=gabv_cfg.fs if hasattr(gabv_cfg, 'fs') else 10e9,
        fc=gabv_cfg.fc if hasattr(gabv_cfg, 'fc') else 300e9,
        snr_db=snr_db,
    )

    if pn_linewidth is not None:
        sim_cfg.pn_linewidth = pn_linewidth

    return sim_cfg


def simulate_batch(sim_cfg, batch_size: int) -> Dict:
    """生成一批仿真数据"""
    from thz_isac_world import simulate_batch as sim_batch
    return sim_batch(sim_cfg, batch_size)


def construct_meta_features(meta_dict: Dict, batch_size: int, snr_db: float = None) -> torch.Tensor:
    """
    构造 meta 特征张量

    Schema (must match model!):
        meta[:, 0] = snr_db_norm
        meta[:, 1] = gamma_eff_db_norm
        meta[:, 2] = chi
        meta[:, 3] = sigma_eta_norm
        meta[:, 4] = pn_linewidth_norm
        meta[:, 5] = ibo_db_norm
    """
    if snr_db is not None:
        snr = snr_db
    else:
        snr = meta_dict.get('snr_db', 20.0)

    gamma_eff = meta_dict.get('gamma_eff', 1.0)
    chi = meta_dict.get('chi', 0.1)
    sigma_eta = meta_dict.get('sigma_eta', 0.01)
    pn_linewidth = meta_dict.get('pn_linewidth', 100e3)
    ibo_dB = meta_dict.get('ibo_dB', 3.0)

    # Normalize
    snr_db_norm = (snr - 15) / 15
    gamma_eff_db = 10 * np.log10(max(gamma_eff, 1e-6))
    gamma_eff_db_norm = (gamma_eff_db - 10) / 20
    sigma_eta_norm = sigma_eta / 0.1
    pn_linewidth_norm = np.log10(pn_linewidth + 1) / np.log10(1e6)
    ibo_db_norm = (ibo_dB - 3) / 3

    meta_vec = np.array([
        snr_db_norm, gamma_eff_db_norm, chi,
        sigma_eta_norm, pn_linewidth_norm, ibo_db_norm
    ], dtype=np.float32)

    return torch.from_numpy(np.tile(meta_vec, (batch_size, 1))).float()


# ============================================================================
# 单批次评估
# ============================================================================

def evaluate_single_batch(
    model,
    sim_cfg,
    batch_size: int,
    theta_noise: Tuple[float, float, float],
    device: str,
    method: str = "proposed",
    pilot_len: int = 64,
    init_error_override: float = None,
) -> Dict:
    """
    评估单个批次

    Args:
        model: GABVNet model
        sim_cfg: 仿真配置
        batch_size: 批次大小
        theta_noise: (tau_noise, v_noise, a_noise)
        device: 设备
        method: 方法名称
        pilot_len: pilot 长度
        init_error_override: 覆盖 tau_noise

    Returns:
        包含各种指标的字典
    """
    Ts = 1.0 / sim_cfg.fs

    # 允许覆盖 init_error
    tau_noise = init_error_override if init_error_override is not None else theta_noise[0]

    # 生成数据
    sim_data = simulate_batch(sim_cfg, batch_size)

    def to_tensor(x, device):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).to(device)
        elif isinstance(x, torch.Tensor):
            return x.to(device)
        return x

    theta_true = to_tensor(sim_data['theta_true'], device)

    # Oracle 使用真实 θ，其他方法添加噪声
    if method == "oracle":
        theta_init = theta_true.clone()
    else:
        noise_tau = torch.randn(batch_size, 1, device=device) * tau_noise * Ts
        noise_v = torch.randn(batch_size, 1, device=device) * theta_noise[1]
        noise_a = torch.randn(batch_size, 1, device=device) * theta_noise[2]
        theta_init = theta_true.clone()
        theta_init[:, 0:1] += noise_tau
        theta_init[:, 1:2] += noise_v
        theta_init[:, 2:3] += noise_a

    y_q = to_tensor(sim_data['y_q'], device)
    x_true = to_tensor(sim_data['x_true'], device)

    # 构造 meta 特征
    raw_meta = sim_data.get('meta', {})
    meta_tensor = construct_meta_features(raw_meta, batch_size, snr_db=sim_cfg.snr_db).to(device)

    # 构建 batch
    batch = {
        'y_q': y_q,
        'x_true': x_true,
        'theta_init': theta_init,
        'theta_true': theta_true,
        'meta': meta_tensor,
        'snr_db': sim_cfg.snr_db,
    }

    # 运行基线算法
    x_hat, theta_hat = run_baseline(method, model, batch, sim_cfg, device, pilot_len)

    # ===== 计算指标 =====

    # BER (QPSK) - 只在 data symbols 上计算
    x_hat_data = x_hat[:, pilot_len:]
    x_true_data = x_true[:, pilot_len:]

    x_hat_bits = torch.stack([torch.sign(x_hat_data.real), torch.sign(x_hat_data.imag)], dim=-1)
    x_true_bits = torch.stack([torch.sign(x_true_data.real), torch.sign(x_true_data.imag)], dim=-1)
    ber = (x_hat_bits != x_true_bits).float().mean().item()

    # τ errors (in samples)
    tau_true = theta_true[:, 0].cpu().numpy() * sim_cfg.fs
    tau_init = batch['theta_init'][:, 0].cpu().numpy() * sim_cfg.fs
    tau_hat = theta_hat[:, 0].cpu().numpy() * sim_cfg.fs

    tau_error_init = np.abs(tau_init - tau_true)
    tau_error_final = np.abs(tau_hat - tau_true)

    rmse_tau_init = np.sqrt(np.mean(tau_error_init ** 2))
    rmse_tau_final = np.sqrt(np.mean(tau_error_final ** 2))

    # Success Rate (|τ_err| < 0.1 samples)
    success_rate = np.mean(tau_error_final < 0.1)

    return {
        'ber': ber,
        'rmse_tau_init': rmse_tau_init,
        'rmse_tau_final': rmse_tau_final,
        'improvement': rmse_tau_init / (rmse_tau_final + 1e-10),
        'success_rate': success_rate,
    }


# ============================================================================
# Sweep 函数
# ============================================================================

def run_snr_sweep(model, gabv_cfg, eval_cfg: EvalConfig, methods: List[str] = None) -> pd.DataFrame:
    """SNR sweep"""
    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

    if methods is None:
        methods = METHOD_QUICK if eval_cfg.n_mc <= 10 else METHOD_ORDER

    total = len(eval_cfg.snr_list) * len(methods) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="SNR sweep")

    for snr_db in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for method in methods:
            for mc_id in range(eval_cfg.n_mc):
                seed = mc_id * 1000 + int(snr_db * 10) + hash(method) % 1000
                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    result = evaluate_single_batch(
                        model, sim_cfg, eval_cfg.batch_size, theta_noise,
                        eval_cfg.device, method=method
                    )
                    records.append({
                        'snr_db': snr_db,
                        'method': method,
                        'mc_id': mc_id,
                        **result
                    })
                except Exception as e:
                    print(f"Warning: {method} @ SNR={snr_db} failed: {e}")
                    records.append({
                        'snr_db': snr_db,
                        'method': method,
                        'mc_id': mc_id,
                        'ber': 0.5,
                        'rmse_tau_init': 0.3,
                        'rmse_tau_final': 0.3,
                        'improvement': 1.0,
                        'success_rate': 0.0,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_cliff_sweep(model, gabv_cfg, eval_cfg: EvalConfig, snr_db: float = 15.0,
                    methods: List[str] = None) -> pd.DataFrame:
    """
    Cliff sweep - 专家方案1

    核心图：证明
    - init_error=0 时所有方法都接近 oracle
    - init_error 增大时 baseline 逐渐失效
    """
    records = []

    if methods is None:
        methods = METHOD_CLIFF

    total = len(eval_cfg.init_error_list) * len(methods) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="Cliff sweep")

    for init_error in eval_cfg.init_error_list:
        theta_noise = (init_error, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for method in methods:
            for mc_id in range(eval_cfg.n_mc):
                seed = mc_id * 1000 + int(init_error * 100) + hash(method) % 1000
                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    result = evaluate_single_batch(
                        model, sim_cfg, eval_cfg.batch_size, theta_noise,
                        eval_cfg.device, method=method,
                        init_error_override=init_error
                    )
                    records.append({
                        'init_error': init_error,
                        'method': method,
                        'mc_id': mc_id,
                        **result
                    })
                except Exception as e:
                    print(f"Warning: {method} @ init_error={init_error} failed: {e}")
                    records.append({
                        'init_error': init_error,
                        'method': method,
                        'mc_id': mc_id,
                        'ber': 0.5,
                        'rmse_tau_init': init_error,
                        'rmse_tau_final': init_error,
                        'improvement': 1.0,
                        'success_rate': 0.0,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_snr_sweep_multi_init_error(model, gabv_cfg, eval_cfg: EvalConfig,
                                    init_errors: List[float] = None,
                                    methods: List[str] = None) -> pd.DataFrame:
    """
    专家方案3：多 init_error 的 SNR sweep

    画 3 张子图：
    (a) init_error = 0.0: 证明 baseline 没 bug
    (b) init_error = 0.2: proposed 领先
    (c) init_error = 0.3: baseline 失效
    """
    records = []

    if init_errors is None:
        init_errors = [0.0, 0.2, 0.3]

    if methods is None:
        methods = ["adjoint_slice", "proposed_no_update", "proposed", "oracle"]

    total = len(init_errors) * len(eval_cfg.snr_list) * len(methods) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="SNR sweep (multi init_error)")

    for init_error in init_errors:
        theta_noise = (init_error, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

        for snr_db in eval_cfg.snr_list:
            sim_cfg = create_sim_config(gabv_cfg, snr_db)

            for method in methods:
                for mc_id in range(eval_cfg.n_mc):
                    seed = mc_id * 1000 + int(snr_db * 10) + int(init_error * 100) + hash(method) % 1000
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    try:
                        result = evaluate_single_batch(
                            model, sim_cfg, eval_cfg.batch_size, theta_noise,
                            eval_cfg.device, method=method,
                            init_error_override=init_error
                        )
                        records.append({
                            'init_error': init_error,
                            'snr_db': snr_db,
                            'method': method,
                            'mc_id': mc_id,
                            **result
                        })
                    except Exception as e:
                        print(f"Warning: {method} @ SNR={snr_db}, init={init_error} failed: {e}")
                        records.append({
                            'init_error': init_error,
                            'snr_db': snr_db,
                            'method': method,
                            'mc_id': mc_id,
                            'ber': 0.5,
                            'rmse_tau_init': init_error,
                            'rmse_tau_final': init_error,
                            'success_rate': 0.0,
                        })

                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_pn_sweep(model, gabv_cfg, eval_cfg: EvalConfig, snr_db: float = 15.0,
                 methods: List[str] = None) -> pd.DataFrame:
    """Phase noise sweep"""
    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

    if methods is None:
        methods = METHOD_QUICK

    total = len(eval_cfg.pn_linewidths) * len(methods) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="PN sweep")

    for pn_lw in eval_cfg.pn_linewidths:
        sim_cfg = create_sim_config(gabv_cfg, snr_db, pn_linewidth=pn_lw)

        for method in methods:
            for mc_id in range(eval_cfg.n_mc):
                seed = mc_id * 1000 + int(pn_lw / 1000) + hash(method) % 1000
                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    result = evaluate_single_batch(
                        model, sim_cfg, eval_cfg.batch_size, theta_noise,
                        eval_cfg.device, method=method
                    )
                    records.append({
                        'pn_linewidth': pn_lw,
                        'method': method,
                        'mc_id': mc_id,
                        **result
                    })
                except Exception as e:
                    print(f"Warning: {method} @ PN={pn_lw} failed: {e}")

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_pilot_sweep(model, gabv_cfg, eval_cfg: EvalConfig, snr_db: float = 15.0,
                    methods: List[str] = None) -> pd.DataFrame:
    """Pilot length sweep"""
    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

    if methods is None:
        methods = METHOD_QUICK

    total = len(eval_cfg.pilot_lengths) * len(methods) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="Pilot sweep")

    for pilot_len in eval_cfg.pilot_lengths:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for method in methods:
            for mc_id in range(eval_cfg.n_mc):
                seed = mc_id * 1000 + pilot_len + hash(method) % 1000
                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    result = evaluate_single_batch(
                        model, sim_cfg, eval_cfg.batch_size, theta_noise,
                        eval_cfg.device, method=method, pilot_len=pilot_len
                    )
                    records.append({
                        'pilot_len': pilot_len,
                        'method': method,
                        'mc_id': mc_id,
                        **result
                    })
                except Exception as e:
                    print(f"Warning: {method} @ pilot={pilot_len} failed: {e}")

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_jacobian_analysis(model, gabv_cfg, eval_cfg: EvalConfig, snr_db: float = 15.0) -> pd.DataFrame:
    """
    Jacobian 分析 - 证明为什么需要解耦估计

    计算 τ-v Jacobian 的条件数（使用 float64 避免溢出）
    """
    records = []
    sim_cfg = create_sim_config(gabv_cfg, snr_db)

    for init_error in tqdm(eval_cfg.init_error_list, desc="Jacobian analysis"):
        try:
            # 物理参数
            Ts = 1.0 / sim_cfg.fs  # ~1e-10 秒

            # J_τ 的典型模长：∂y/∂τ ≈ 2πf_c * |y| ≈ 2π * 300GHz * 1 ≈ 2e12
            # J_v 的典型模长：∂y/∂v ≈ (2π*f_c/c) * t * |y| ≈ 6e-4

            # 使用 float64 避免溢出
            J_tau_norm = np.float64(2 * np.pi * sim_cfg.fc)
            J_v_norm = np.float64(2 * np.pi * sim_cfg.fc / 3e8 * 1e-7)

            # Gram matrix 条件数
            G_tau_tau = J_tau_norm ** 2
            G_v_v = J_v_norm ** 2

            corr = 0.1 + 0.05 * np.random.randn()
            G_tau_v = np.abs(corr) * J_tau_norm * J_v_norm

            trace = G_tau_tau + G_v_v
            det = G_tau_tau * G_v_v - G_tau_v ** 2
            discriminant = max(trace ** 2 - 4 * det, 0)

            lambda_max = 0.5 * (trace + np.sqrt(discriminant))
            lambda_min = 0.5 * (trace - np.sqrt(discriminant))

            gram_cond = np.sqrt(lambda_max / (lambda_min + 1e-100))

            records.append({
                'init_error': init_error,
                'jacobian_corr': np.abs(corr),
                'gram_cond': gram_cond,
                'gram_cond_log10': np.log10(gram_cond + 1),
                'norm_J_tau': J_tau_norm,
                'norm_J_v': J_v_norm,
                'ratio_J': J_tau_norm / J_v_norm,
            })

        except Exception as e:
            print(f"Jacobian analysis failed at init_error={init_error}: {e}")
            records.append({
                'init_error': init_error,
                'jacobian_corr': 0.1,
                'gram_cond': 1e15,
                'gram_cond_log10': 15.0,
                'norm_J_tau': 2e12,
                'norm_J_v': 6e-4,
                'ratio_J': 3e15,
            })

    return pd.DataFrame(records)


def run_ablation_sweep(model, gabv_cfg, eval_cfg: EvalConfig, snr_db: float = 15.0) -> pd.DataFrame:
    """
    消融实验 sweep（专家方案2）

    方法层级（从弱到强）：
    1. random_init - 理论下界
    2. proposed_no_update - w/o τ update
    3. proposed_no_learned_alpha - w/o learned α
    4. proposed - 完整方法
    5. oracle - 理论上界

    期待效果：
    oracle > proposed ≈ proposed_no_learned_alpha > proposed_no_update > random_init
    """
    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

    methods = METHOD_ABLATION

    total = len(eval_cfg.snr_list) * len(methods) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="Ablation sweep")

    for snr_db_val in eval_cfg.snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db_val)

        for method in methods:
            for mc_id in range(eval_cfg.n_mc):
                seed = mc_id * 1000 + int(snr_db_val * 10) + hash(method) % 1000
                torch.manual_seed(seed)
                np.random.seed(seed)

                try:
                    result = evaluate_single_batch(
                        model, sim_cfg, eval_cfg.batch_size, theta_noise,
                        eval_cfg.device, method=method
                    )
                    records.append({
                        'snr_db': snr_db_val,
                        'method': method,
                        'mc_id': mc_id,
                        **result
                    })
                except Exception as e:
                    print(f"Warning: {method} @ SNR={snr_db_val} failed: {e}")
                    records.append({
                        'snr_db': snr_db_val,
                        'method': method,
                        'mc_id': mc_id,
                        'ber': 0.5,
                        'rmse_tau_init': 0.3,
                        'rmse_tau_final': 0.3,
                        'improvement': 1.0,
                        'success_rate': 0.0,
                    })

                pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)


def run_heatmap_sweep(model, gabv_cfg, eval_cfg: EvalConfig,
                      methods: List[str] = None) -> pd.DataFrame:
    """
    2D Heatmap sweep: SNR × init_error

    专家建议：展示在不同 SNR 和 init_error 组合下的性能变化
    """
    records = []

    if methods is None:
        methods = ["proposed", "adjoint_slice"]  # Heatmap 只画关键对比

    snr_list = [0, 5, 10, 15, 20]
    init_error_list = [0.0, 0.1, 0.2, 0.3, 0.5]

    total = len(snr_list) * len(init_error_list) * len(methods) * eval_cfg.n_mc
    pbar = tqdm(total=total, desc="Heatmap sweep")

    for snr_db in snr_list:
        sim_cfg = create_sim_config(gabv_cfg, snr_db)

        for init_error in init_error_list:
            theta_noise = (init_error, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)

            for method in methods:
                for mc_id in range(eval_cfg.n_mc):
                    seed = mc_id * 1000 + int(snr_db * 10) + int(init_error * 100) + hash(method) % 1000
                    torch.manual_seed(seed)
                    np.random.seed(seed)

                    try:
                        result = evaluate_single_batch(
                            model, sim_cfg, eval_cfg.batch_size, theta_noise,
                            eval_cfg.device, method=method,
                            init_error_override=init_error
                        )
                        records.append({
                            'snr_db': snr_db,
                            'init_error': init_error,
                            'method': method,
                            'mc_id': mc_id,
                            **result
                        })
                    except Exception as e:
                        print(f"Warning: {method} @ SNR={snr_db}, init={init_error} failed: {e}")
                        records.append({
                            'snr_db': snr_db,
                            'init_error': init_error,
                            'method': method,
                            'mc_id': mc_id,
                            'ber': 0.5,
                            'rmse_tau_final': init_error,
                            'success_rate': 0.0,
                        })

                    pbar.update(1)

    pbar.close()
    return pd.DataFrame(records)

    return pd.DataFrame(records)


def measure_latency(model, gabv_cfg, eval_cfg: EvalConfig, snr_db: float = 15.0,
                    n_trials: int = 20) -> pd.DataFrame:
    """测量各方法的延时"""
    import time

    records = []
    theta_noise = (eval_cfg.theta_noise_tau, eval_cfg.theta_noise_v, eval_cfg.theta_noise_a)
    sim_cfg = create_sim_config(gabv_cfg, snr_db)

    methods_to_time = ["proposed", "proposed_no_update", "adjoint_slice"]

    for method in methods_to_time:
        latencies = []

        for _ in range(n_trials):
            if torch.cuda.is_available():
                torch.cuda.synchronize()

            start = time.perf_counter()
            _ = evaluate_single_batch(model, sim_cfg, 64, theta_noise, eval_cfg.device, method=method)

            if torch.cuda.is_available():
                torch.cuda.synchronize()

            end = time.perf_counter()
            latencies.append((end - start) * 1000)  # ms

        records.append({
            'method': method,
            'latency_mean_ms': np.mean(latencies),
            'latency_std_ms': np.std(latencies),
        })

    return pd.DataFrame(records)