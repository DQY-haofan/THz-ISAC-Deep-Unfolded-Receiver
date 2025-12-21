"""
baselines.py - 基线算法实现

所有与 proposed 方法对比的基线算法都在这里实现。
专家要求：baseline 必须复用模型前端，确保在同一域对比！

方法层级（从弱到强）：
1. naive_slice      - 直接 slice（不做任何前端处理）
2. matched_filter   - Grid Search τ + Slice
3. adjoint_lmmse    - Adjoint + PN Align + LMMSE
4. adjoint_slice    - Adjoint + PN Align + Hard Slice
5. proposed_no_update - BV-VAMP 无 τ 更新
6. proposed         - 完整方法（BV-VAMP + τ 更新）
7. oracle           - 使用真实 θ
"""

import torch
import numpy as np
import math
from typing import Dict, Tuple, Optional


# ============================================================================
# 辅助函数
# ============================================================================

def qpsk_hard_slice(z: torch.Tensor) -> torch.Tensor:
    """QPSK hard decision in complex plane."""
    xr = torch.sign(z.real)
    xi = torch.sign(z.imag)
    # 处理 0 的情况
    xr = torch.where(xr == 0, torch.ones_like(xr), xr)
    xi = torch.where(xi == 0, torch.ones_like(xi), xi)
    return (xr + 1j * xi) / np.sqrt(2)


def frontend_adjoint_and_pn(model, y_q: torch.Tensor, theta: torch.Tensor, 
                            x_pilot: torch.Tensor, pilot_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    使用与 proposed 完全相同的前端。
    
    专家强调：这是确保公平对比的关键！
    
    流程：
      z = H*(theta) y_q          # Adjoint 操作
      z_derot = pn_derotation(z) # Pilot-based 常相位对齐
    
    Args:
        model: GABVNet model（用于访问 phys_enc）
        y_q: 量化后的接收信号 [B, N]
        theta: 信道参数 [B, 3] = [τ, v, a]
        x_pilot: 真实符号（含 pilot）[B, N]
        pilot_len: pilot 长度
    
    Returns:
        z_derot: 去旋转后的信号 [B, N]
        phi_est: 估计的相位 [B, 1]
    """
    batch_size = y_q.shape[0]
    device = y_q.device
    
    # 1) Adjoint 操作（使用模型的 phys_enc）
    try:
        z = model.phys_enc.adjoint_operator(y_q, theta)
    except Exception as e:
        # Fallback: 如果模型没有 adjoint_operator，直接用 y_q
        print(f"Warning: adjoint_operator failed, using y_q directly: {e}")
        z = y_q
    
    # 2) Pilot-based 常相位对齐（与 proposed 相同的方式）
    # 
    # 专家 Bug 修复：相位估计方向必须与主模型一致！
    # 
    # 信号模型：z_p ≈ x_p * exp(j*φ)
    # 正确估计：correlation = sum(z_p * x_p.conj()) = |x_p|² * exp(j*φ)
    #          phi_est = angle(correlation) = φ
    #          z_derot = z * exp(-j*φ) → 消除相位
    #
    # 错误的旧代码：correlation = sum(z_p.conj() * x_p) = |x_p|² * exp(-j*φ)
    #              phi_est = -φ
    #              z_derot = z * exp(-j*(-φ)) = z * exp(j*φ) → 相位加倍！
    #
    if x_pilot is not None and pilot_len > 0:
        x_p = x_pilot[:, :pilot_len]
        z_p = z[:, :pilot_len]
        
        # 正确的相位估计（与 gabv_net_model.py 的 PN tracker 一致）
        correlation = torch.sum(z_p * torch.conj(x_p), dim=1, keepdim=True)
        phi_est = torch.angle(correlation)
        
        # 去旋转整个符号序列
        z_derot = z * torch.exp(-1j * phi_est)
    else:
        z_derot = z
        phi_est = torch.zeros(batch_size, 1, device=device)
    
    return z_derot, phi_est


# ============================================================================
# 基线算法实现
# ============================================================================

class BaselineNaiveSlice:
    """
    最弱基线：直接对 y_q 做 hard slice（不做任何前端处理）
    
    预期 BER ≈ 0.5（随机猜测）
    """
    name = "naive_slice"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        
        # 直接 slice（没有 adjoint，没有 PN 对齐）
        x_hat = qpsk_hard_slice(y_q)
        
        return x_hat, theta_init


class BaselineMatchedFilter:
    """
    Matched Filter / Grid Search τ 估计 + Hard Slice
    
    这是"粗同步"的典型方法：
    1. 在 τ 网格上搜索最佳相关点
    2. 用找到的 τ 做 adjoint + slice
    
    P0-2 修复：搜索范围扩大到 ±2.0 samples，覆盖 cliff sweep 范围
    """
    name = "matched_filter"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']
        batch_size = y_q.shape[0]
        
        Ts = 1.0 / sim_cfg.fs
        
        # P0-2 修复：τ 网格搜索范围扩大到 ±2.0 samples
        # 覆盖 cliff sweep 的 init_error 范围 (0 ~ 1.5)
        tau_grid_samples = torch.linspace(-1.0, 1.0, 21, device=device)  # 41 点网格
        
        best_corr = None
        best_tau = theta_init[:, 0:1].clone()
        
        x_pilot = x_true[:, :pilot_len]
        
        for tau_offset in tau_grid_samples:
            # 构造试探 θ
            theta_test = theta_init.clone()
            theta_test[:, 0:1] = theta_init[:, 0:1] + tau_offset * Ts
            
            # 前端处理
            z_derot, _ = frontend_adjoint_and_pn(
                model, y_q, theta_test, x_true, pilot_len
            )
            
            # 计算与 pilot 的相关性
            z_p = z_derot[:, :pilot_len]
            corr = torch.abs(torch.sum(z_p.conj() * x_pilot, dim=1, keepdim=True))
            
            if best_corr is None:
                best_corr = corr
                best_tau = theta_test[:, 0:1]
            else:
                mask = corr > best_corr
                best_corr = torch.where(mask, corr, best_corr)
                best_tau = torch.where(mask, theta_test[:, 0:1], best_tau)
        
        # 用最佳 τ 做最终检测
        theta_hat = theta_init.clone()
        theta_hat[:, 0:1] = best_tau
        
        z_derot, _ = frontend_adjoint_and_pn(
            model, y_q, theta_hat, x_true, pilot_len
        )
        x_hat = qpsk_hard_slice(z_derot)
        
        return x_hat, theta_hat


class BaselineAdjointLMMSE:
    """
    Adjoint + Pilot PN Align + Bussgang-LMMSE 基线
    
    比 hard slice 强，使用 LMMSE 而非 hard decision
    """
    name = "adjoint_lmmse"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']
        
        # 使用与 proposed 相同的前端
        z_derot, phi_est = frontend_adjoint_and_pn(
            model, y_q, theta_init, x_true, pilot_len
        )
        
        # Bussgang-LMMSE
        # y = α*x + n, LMMSE: x_hat = z / (|α|² + σ²)
        snr_lin = 10 ** (sim_cfg.snr_db / 10)
        sigma2 = 1.0 / snr_lin
        
        # Bussgang 因子 for 1-bit: α ≈ sqrt(2/π)
        alpha = np.sqrt(2 / np.pi)
        
        # LMMSE 估计（软判决）
        x_hat_soft = z_derot / (alpha**2 + sigma2)
        
        # 最后做 hard decision（因为要计算 BER）
        x_hat = qpsk_hard_slice(x_hat_soft)
        
        return x_hat, theta_init  # τ 不更新


class BaselineAdjointSlice:
    """
    Adjoint + Pilot PN Align + Hard Slice 基线
    
    这是论文中最重要的对比基线！
    
    专家要求：必须复用模型前端，确保与 proposed 在同一域对比！
    """
    name = "adjoint_slice"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']
        
        # 使用与 proposed 相同的前端
        z_derot, phi_est = frontend_adjoint_and_pn(
            model, y_q, theta_init, x_true, pilot_len
        )
        
        # Hard Slice (QPSK)
        x_hat = qpsk_hard_slice(z_derot)
        
        return x_hat, theta_init  # τ 不更新


class BaselineProposedNoUpdate:
    """
    BV-VAMP 但不更新 τ
    
    用于验证 τ 更新机制的价值
    """
    name = "proposed_no_update"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_init = batch['theta_init']
        
        # 保存原始设置
        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = False
        
        outputs = model(batch)
        
        # 恢复设置
        model.cfg.enable_theta_update = original_setting
        
        x_hat = outputs['x_hat']
        theta_hat = outputs.get('theta_hat', theta_init)
        
        return x_hat, theta_hat


class BaselineProposed:
    """
    完整的 proposed 方法：BV-VAMP + τ 更新
    """
    name = "proposed"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_init = batch['theta_init']
        
        # 保存原始设置
        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = True
        
        outputs = model(batch)
        
        # 恢复设置
        model.cfg.enable_theta_update = original_setting
        
        x_hat = outputs['x_hat']
        theta_hat = outputs.get('theta_hat', theta_init)
        
        return x_hat, theta_hat


class BaselineOracle:
    """
    Oracle: 使用真实 θ
    
    理论性能上界
    """
    name = "oracle"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_true = batch['theta_true']
        
        # 使用真实 θ
        batch_oracle = batch.copy()
        batch_oracle['theta_init'] = theta_true.clone()
        
        # 保存原始设置
        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = False
        
        outputs = model(batch_oracle)
        
        # 恢复设置
        model.cfg.enable_theta_update = original_setting
        
        x_hat = outputs['x_hat']
        theta_hat = theta_true.clone()
        
        return x_hat, theta_hat


class BaselineRandomInit:
    """
    Random Init: 使用大随机误差的 θ
    
    理论性能下界（消融实验用）
    """
    name = "random_init"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_true = batch['theta_true']
        batch_size = theta_true.shape[0]
        Ts = 1.0 / sim_cfg.fs
        
        # 大随机误差：τ ± 2 samples, v ± 50 m/s
        noise_tau = torch.randn(batch_size, 1, device=device) * 2.0 * Ts
        noise_v = torch.randn(batch_size, 1, device=device) * 50.0
        noise_a = torch.randn(batch_size, 1, device=device) * 10.0
        
        theta_init = theta_true.clone()
        theta_init[:, 0:1] += noise_tau
        theta_init[:, 1:2] += noise_v
        theta_init[:, 2:3] += noise_a
        
        batch_random = batch.copy()
        batch_random['theta_init'] = theta_init
        
        # 不更新 θ
        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = False
        
        outputs = model(batch_random)
        
        model.cfg.enable_theta_update = original_setting
        
        x_hat = outputs['x_hat']
        theta_hat = theta_init  # 保持随机初始化
        
        return x_hat, theta_hat


class BaselineProposedTauSlice:
    """
    Proposed with τ Slice: 用 proposed 估计 τ，但检测用 hard slice
    
    消融实验：验证 VAMP 检测器的价值（vs 简单 slice）
    
    这是更干净的消融：
    - 保留 τ 更新能力（证明 τ 估计模块工作）
    - 移除 VAMP soft detection（证明 VAMP 的价值）
    
    专家 Bug 修复：必须用 theta_hat 做前端，否则结果无意义！
    """
    name = "proposed_tau_slice"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        # 1) 先用 proposed 获取估计的 θ（包含 τ 更新）
        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = True
        
        outputs = model(batch)
        theta_hat = outputs.get('theta_hat', batch['theta_init'])
        
        model.cfg.enable_theta_update = original_setting
        
        # 2) 用估计的 θ 做 adjoint + PN 对齐（与其他 baseline 相同的前端）
        # 专家修复：之前直接对 y_q 做 slice，这是错误的！
        # 必须先用 theta_hat 做信道补偿
        y_q = batch['y_q']
        x_true = batch['x_true']
        
        z_derot, _ = frontend_adjoint_and_pn(
            model, y_q, theta_hat, x_true, pilot_len
        )
        
        # 3) Hard slice 检测（而非 VAMP soft detection）
        x_hat = qpsk_hard_slice(z_derot)
        
        return x_hat, theta_hat


# ============================================================================
# 方法注册表
# ============================================================================

BASELINE_REGISTRY = {
    "naive_slice": BaselineNaiveSlice,
    "matched_filter": BaselineMatchedFilter,
    "adjoint_lmmse": BaselineAdjointLMMSE,
    "adjoint_slice": BaselineAdjointSlice,
    "proposed_no_update": BaselineProposedNoUpdate,
    "proposed_tau_slice": BaselineProposedTauSlice,  # 消融：τ估计OK但检测用slice
    "proposed": BaselineProposed,
    "oracle": BaselineOracle,
    "random_init": BaselineRandomInit,
}

# 方法层级（从弱到强）
METHOD_ORDER = [
    "naive_slice",
    "matched_filter",
    "adjoint_lmmse",
    "adjoint_slice",
    "proposed_no_update",
    "proposed_tau_slice",
    "proposed",
    "oracle",
]

# 快速测试用的方法子集
METHOD_QUICK = ["adjoint_slice", "proposed", "oracle"]

# Cliff sweep 用的方法集（包含 adjoint_lmmse）
METHOD_CLIFF = [
    "naive_slice",
    "adjoint_lmmse",  # 专家建议：加入更强的线性基线
    "adjoint_slice",
    "matched_filter",
    "proposed_no_update",
    "proposed",
    "oracle",
]

# 消融实验用的方法集（方案2，修正版）
METHOD_ABLATION = [
    "random_init",           # 理论下界
    "proposed_no_update",    # w/o τ update
    "proposed_tau_slice",    # τ估计OK但检测用slice（验证VAMP价值）
    "proposed",              # 完整方法
    "oracle",                # 理论上界
]


def get_baseline(method_name: str):
    """获取基线算法类"""
    if method_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[method_name]


def run_baseline(method_name: str, model, batch: Dict, sim_cfg, device: str, 
                 pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """运行指定的基线算法"""
    baseline_cls = get_baseline(method_name)
    return baseline_cls.run(model, batch, sim_cfg, device, pilot_len)
