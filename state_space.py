"""
state_space.py - Phase III State Space Model Definitions

本模块实现慢环跟踪器的状态空间模型，与Micro-DR报告对应如下：

【报告方案A (物理域)】           【当前实现 (延迟域)】
  x = [R, v, a]^T                 x = [τ, τ_dot, τ_ddot]^T
  单位: [m, m/s, m/s²]            单位: [s, -, 1/s]
  
  对应关系 (通过光速c转换):
    τ = R/c
    τ_dot = v/c  
    τ_ddot = a/c

【观测模型对应】
  Model-1 (仅时延):
    报告: z = R/c,  H = [1/c, 0, 0, 0, 0, 0]
    实现: z = τ,    H = [1, 0, 0]  (等价)
    
  Model-2 (时延+多普勒) - 待扩展:
    报告: z = [τ, f_D]^T
    f_D = -(f_c/c)*v + (1/2π)*φ_dot
    H[1,:] = [0, -f_c/c, 0, 0, 1/(2π), 0]

参考文档:
- Micro-DR: Phase-III Slow-Loop State Definition
- DR-III-T: 跨帧慢环理论报告
- DR-III-E: 工程实现指南

Author: Phase III MVP
Date: 2025-12
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import Enum


class MotionModel(Enum):
    """Motion model types."""
    CONSTANT_VELOCITY = "CV"      # 2-state: [tau, tau_dot]
    CONSTANT_ACCELERATION = "CA"  # 3-state: [tau, tau_dot, tau_ddot]


@dataclass
class StateSpaceConfig:
    """Configuration for state space model."""
    
    # Time step (frame interval) [s]
    dt: float = 1e-3  # 1 ms default
    
    # Motion model
    motion_model: MotionModel = MotionModel.CONSTANT_ACCELERATION
    
    # Process noise PSD for jerk [m²/s⁵]
    # Typical LEO values: 1e-6 to 1e-4
    S_jerk: float = 1e-5
    
    # Clock noise parameters (Allan variance coefficients)
    h0: float = 1e-22       # White FM coefficient
    h_minus2: float = 1e-24  # Random walk FM coefficient
    
    # Initial state uncertainty
    sigma_tau_init: float = 1e-9     # 1 ns (30 cm)
    sigma_tau_dot_init: float = 100.0  # 100 m/s equivalent
    sigma_tau_ddot_init: float = 10.0  # 10 m/s² equivalent
    
    # Physical constants
    c: float = 3e8  # Speed of light [m/s]
    
    @property
    def state_dim(self) -> int:
        """State dimension based on motion model."""
        if self.motion_model == MotionModel.CONSTANT_VELOCITY:
            return 2
        else:  # CA
            return 3


def compute_F_matrix(dt: float, motion_model: MotionModel = MotionModel.CONSTANT_ACCELERATION) -> np.ndarray:
    """
    Compute state transition matrix F.
    
    For Constant Acceleration model:
        x_{k+1} = F @ x_k + w_k
        
        F = [1, dt, 0.5*dt²]
            [0,  1,     dt  ]
            [0,  0,      1  ]
    
    Args:
        dt: Time step [s]
        motion_model: Motion model type
        
    Returns:
        F: State transition matrix [n_x, n_x]
    """
    if motion_model == MotionModel.CONSTANT_VELOCITY:
        F = np.array([
            [1.0, dt],
            [0.0, 1.0]
        ], dtype=np.float64)
    else:  # CA
        F = np.array([
            [1.0, dt, 0.5 * dt**2],
            [0.0, 1.0, dt],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
    
    return F


def compute_Q_kinematic(dt: float, S_jerk: float, 
                        motion_model: MotionModel = MotionModel.CONSTANT_ACCELERATION) -> np.ndarray:
    """
    Compute kinematic process noise covariance Q using Van Loan method.
    
    For CA model with jerk as white noise with PSD S_a:
        Q = S_a * [dt^5/20, dt^4/8, dt^3/6]
                  [dt^4/8,  dt^3/3, dt^2/2]
                  [dt^3/6,  dt^2/2,   dt  ]
    
    Physical basis:
        - S_jerk captures unmodeled accelerations (atmospheric drag variance, J2 residuals)
        - For LEO satellites, typical S_jerk ~ 1e-6 to 1e-4 m²/s⁵
        
    Args:
        dt: Time step [s]
        S_jerk: Jerk power spectral density [m²/s⁵]
        motion_model: Motion model type
        
    Returns:
        Q: Process noise covariance [n_x, n_x]
    """
    if motion_model == MotionModel.CONSTANT_VELOCITY:
        # CV model: acceleration as white noise
        Q = S_jerk * np.array([
            [dt**3 / 3, dt**2 / 2],
            [dt**2 / 2, dt]
        ], dtype=np.float64)
    else:  # CA
        # CA model: jerk as white noise (from Van Loan discretization)
        Q = S_jerk * np.array([
            [dt**5 / 20, dt**4 / 8, dt**3 / 6],
            [dt**4 / 8,  dt**3 / 3, dt**2 / 2],
            [dt**3 / 6,  dt**2 / 2, dt]
        ], dtype=np.float64)
    
    return Q


def compute_Q_clock(dt: float, h0: float, h_minus2: float) -> np.ndarray:
    """
    Compute clock noise covariance from Allan variance coefficients.
    
    Based on power-law spectrum model:
        Q_clock = [h0/2*dt + 2/3*pi²*h_{-2}*dt³,  pi²*h_{-2}*dt²]
                  [pi²*h_{-2}*dt²,                2*pi²*h_{-2}*dt]
    
    Physical basis:
        - h0: White frequency noise coefficient
        - h_{-2}: Random walk frequency noise coefficient
        - These come from Allan variance characterization of oscillators
        
    Args:
        dt: Time step [s]
        h0: White FM Allan variance coefficient
        h_minus2: Random walk FM Allan variance coefficient
        
    Returns:
        Q_clock: Clock noise covariance [2, 2] for [phase, freq_drift]
    """
    pi2 = np.pi**2
    
    Q_clock = np.array([
        [h0/2 * dt + 2/3 * pi2 * h_minus2 * dt**3,  pi2 * h_minus2 * dt**2],
        [pi2 * h_minus2 * dt**2,                     2 * pi2 * h_minus2 * dt]
    ], dtype=np.float64)
    
    return Q_clock


class ObservationModel(Enum):
    """观测模型类型，对应Micro-DR报告§3"""
    MODEL_1_DELAY_ONLY = "Model-1"      # 仅时延: z = [τ]
    MODEL_2_DELAY_DOPPLER = "Model-2"   # 时延+多普勒: z = [τ, f_D]


def compute_H_matrix(state_dim: int, obs_dim: int = 1, 
                     obs_model: ObservationModel = ObservationModel.MODEL_1_DELAY_ONLY,
                     fc: float = 300e9) -> np.ndarray:
    """
    计算观测矩阵H。对应Micro-DR报告§3观测方程与Jacobian。
    
    【Model-1 仅时延】 (报告§3.1)
      观测: z = [τ_hat]
      状态: x = [τ, τ_dot, τ_ddot]（delay-space）
      H = [1, 0, 0]
      
    【Model-2 时延+多普勒】 (报告§3.2)
      观测: z = [τ_hat, f_D_hat]
      f_D = -(f_c/c) * v = -f_c * τ_dot  (因为 τ_dot = v/c)
      
      Jacobian:
        ∂τ/∂τ = 1, ∂τ/∂τ_dot = 0, ∂τ/∂τ_ddot = 0
        ∂f_D/∂τ = 0, ∂f_D/∂τ_dot = -f_c, ∂f_D/∂τ_ddot = 0
        
      H = [[1,    0,    0],
           [0, -f_c,    0]]
           
      量纲分析 (报告§3.2):
        H[1,1] = -f_c [Hz / (s/s)] = -f_c [Hz]
        若 f_c = 300 GHz，则 H[1,1] = -3e11
        这意味着 τ_dot 变化 1e-9 会导致 f_D 变化 300 Hz
            
    Args:
        state_dim: 状态维度 (2 for CV, 3 for CA)
        obs_dim: 观测维度 (1 for Model-1, 2 for Model-2)
        obs_model: 观测模型类型
        fc: 载波频率 [Hz]，用于Model-2
        
    Returns:
        H: 观测矩阵 [n_z, n_x]
    """
    H = np.zeros((obs_dim, state_dim), dtype=np.float64)
    
    # 时延观测 (Model-1 & Model-2共有)
    H[0, 0] = 1.0
    
    # Model-2: 添加多普勒观测
    if obs_model == ObservationModel.MODEL_2_DELAY_DOPPLER and obs_dim >= 2 and state_dim >= 2:
        # f_D = -f_c * τ_dot
        # ∂f_D/∂τ_dot = -f_c
        H[1, 1] = -fc
        
    return H


def compute_initial_covariance(cfg: StateSpaceConfig) -> np.ndarray:
    """
    Compute initial state covariance P0.
    
    Note: State vector is in delay-space [τ, τ_dot, τ_ddot] where
    τ_dot = v/c and τ_ddot = a/c. The config values are in physical
    units and need to be converted.
    
    Args:
        cfg: State space configuration
        
    Returns:
        P0: Initial covariance [n_x, n_x]
    """
    n_x = cfg.state_dim
    c = cfg.c
    P0 = np.zeros((n_x, n_x), dtype=np.float64)
    
    # τ uncertainty (already in seconds)
    P0[0, 0] = cfg.sigma_tau_init**2
    
    if n_x >= 2:
        # Convert velocity uncertainty from m/s to τ_dot = v/c
        sigma_tau_dot = cfg.sigma_tau_dot_init / c
        P0[1, 1] = sigma_tau_dot**2
        
    if n_x >= 3:
        # Convert acceleration uncertainty from m/s² to τ_ddot = a/c
        sigma_tau_ddot = cfg.sigma_tau_ddot_init / c
        P0[2, 2] = sigma_tau_ddot**2
    
    return P0


def generate_kinematic_trajectory(
    K: int,
    dt: float,
    x0: Optional[np.ndarray] = None,
    Q: Optional[np.ndarray] = None,
    motion_model: MotionModel = MotionModel.CONSTANT_ACCELERATION,
    seed: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a kinematic trajectory using state space model.
    
    This creates a physically plausible trajectory for testing the tracker.
    
    Args:
        K: Number of frames
        dt: Time step [s]
        x0: Initial state [n_x]. If None, random initialization.
        Q: Process noise covariance. If None, use default.
        motion_model: Motion model type
        seed: Random seed for reproducibility
        
    Returns:
        x_seq: State sequence [K, n_x]
        w_seq: Process noise sequence [K, n_x]
    """
    rng = np.random.default_rng(seed)
    
    F = compute_F_matrix(dt, motion_model)
    n_x = F.shape[0]
    
    if Q is None:
        Q = compute_Q_kinematic(dt, S_jerk=1e-5, motion_model=motion_model)
    
    if x0 is None:
        # P0-4 Fix: 默认初始状态使用delay-space单位
        # 状态定义: [τ(s), τ_dot=v/c, τ_ddot=a/c]
        c = 3e8  # 光速
        if motion_model == MotionModel.CONSTANT_VELOCITY:
            # 默认: τ=0, v=100 m/s → τ_dot = 100/c
            x0 = np.array([0.0, 100.0 / c])
        else:
            # 默认: τ=0, v=100 m/s, a=10 m/s² → τ_dot=100/c, τ_ddot=10/c
            x0 = np.array([0.0, 100.0 / c, 10.0 / c])
    
    # Cholesky decomposition for sampling
    # Use relative regularization to not dominate small Q values
    Q_diag_min = np.min(np.diag(Q))
    eps = max(1e-50, Q_diag_min * 1e-6)  # Relative regularization
    try:
        L_Q = np.linalg.cholesky(Q + eps * np.eye(n_x))
    except np.linalg.LinAlgError:
        # Fallback to eigendecomposition
        eigvals, eigvecs = np.linalg.eigh(Q)
        eigvals = np.maximum(eigvals, 1e-12)
        L_Q = eigvecs @ np.diag(np.sqrt(eigvals))
    
    # Generate trajectory
    x_seq = np.zeros((K, n_x), dtype=np.float64)
    w_seq = np.zeros((K, n_x), dtype=np.float64)
    
    x_seq[0] = x0
    
    for k in range(1, K):
        # Process noise sample
        w_k = L_Q @ rng.standard_normal(n_x)
        w_seq[k] = w_k
        
        # State propagation
        x_seq[k] = F @ x_seq[k-1] + w_k
    
    return x_seq, w_seq


def tau_to_range(tau: float, c: float = 3e8) -> float:
    """Convert one-way delay to range."""
    return tau * c


def range_to_tau(R: float, c: float = 3e8) -> float:
    """Convert range to one-way delay."""
    return R / c


def velocity_to_doppler(v: float, fc: float, c: float = 3e8) -> float:
    """Convert radial velocity to Doppler frequency."""
    return -fc * v / c


def doppler_to_velocity(fd: float, fc: float, c: float = 3e8) -> float:
    """Convert Doppler frequency to radial velocity."""
    return -fd * c / fc


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("state_space.py - Self-Test")
    print("=" * 60)
    
    # Test configuration
    cfg = StateSpaceConfig(dt=1e-3, S_jerk=1e-5)
    print(f"\nConfig: dt={cfg.dt*1e3:.1f}ms, S_jerk={cfg.S_jerk:.1e}")
    print(f"State dim: {cfg.state_dim}")
    
    # Test F matrix
    print("\n[Test] F matrix (CA model):")
    F = compute_F_matrix(cfg.dt, MotionModel.CONSTANT_ACCELERATION)
    print(f"  Shape: {F.shape}")
    print(f"  F =\n{F}")
    
    # Test Q matrix
    print("\n[Test] Q matrix (kinematic):")
    Q = compute_Q_kinematic(cfg.dt, cfg.S_jerk)
    print(f"  Shape: {Q.shape}")
    print(f"  Q =\n{Q}")
    print(f"  Eigenvalues: {np.linalg.eigvalsh(Q)}")
    
    # Test H matrix
    print("\n[Test] H matrix:")
    H = compute_H_matrix(3, 2)
    print(f"  Shape: {H.shape}")
    print(f"  H =\n{H}")
    
    # Test trajectory generation
    print("\n[Test] Trajectory generation:")
    K = 100
    x_seq, w_seq = generate_kinematic_trajectory(K, cfg.dt, seed=42)
    print(f"  Trajectory shape: {x_seq.shape}")
    print(f"  Initial state: {x_seq[0]}")
    print(f"  Final state: {x_seq[-1]}")
    print(f"  Position drift (tau): {(x_seq[-1, 0] - x_seq[0, 0])*1e9:.2f} ns")
    
    # Verify F @ x propagation
    print("\n[Test] State propagation verification:")
    x_pred = F @ x_seq[0]
    x_actual = x_seq[1] - w_seq[1]
    print(f"  Prediction error: {np.linalg.norm(x_pred - x_actual):.2e}")
    
    print("\n" + "=" * 60)
    print("Self-Test Complete ✓")
    print("=" * 60)
