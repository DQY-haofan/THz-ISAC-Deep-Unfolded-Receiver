"""
tracker.py - Phase III Slow Loop: EKF and RTS Smoother

本模块实现慢环跟踪算法，与Micro-DR报告对应如下：

【报告§5 Implementation Notes】
  快环输出接口:
    z_meas  →  z (np.ndarray)     : 去归一化的物理观测
    R_proxy →  R (np.ndarray)     : 动态协方差矩阵
    valid_flag → valid_flag (bool): 显式有效标志
    risk_metric → risk_k (float)  : 风险分数 (可选)

  慢环响应逻辑:
    valid_flag == 1 → 标准EKF测量更新
    valid_flag == 0 → 惯性滑行 (Coasting)
      - K_k = 0
      - x_k|k = x_k|k-1
      - P_k|k = P_k|k-1

【报告§6 Failure Modes & Guards】
  ✓ P阵非正定性 → Joseph形式 + ensure_spd()
  ✓ 离群点锁定 → Innovation Gating (马氏距离)
  ✓ 滤波器自满 → Process Noise Floor (S_jerk)

Key Features:
- Joseph form covariance update for numerical stability
- Dynamic R_k from fast-loop FIM proxy  
- Gating mechanism for invalid observations
- Physics-based Q matrix support

参考文档:
- Micro-DR §5: Implementation Notes
- Micro-DR §6: Failure Modes & Guards
- DR-III-T Section 5: EKF/RTS algorithms

Author: Phase III MVP
Date: 2025-12
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Union
import warnings

from state_space import (
    StateSpaceConfig, MotionModel, ObservationModel,
    compute_F_matrix, compute_Q_kinematic, compute_H_matrix, compute_initial_covariance
)


@dataclass
class TrackerConfig:
    """Configuration for EKF/RTS tracker."""
    
    # State space config
    state_config: StateSpaceConfig = field(default_factory=StateSpaceConfig)
    
    # Observation dimension
    obs_dim: int = 1  # 1 for delay-only, 2 for delay+velocity
    
    # Default measurement noise (used when R_k not provided)
    default_R_diag: np.ndarray = field(default_factory=lambda: np.array([1e-18]))  # (1 ns)^2
    
    # Gating threshold (Mahalanobis distance squared)
    # For chi-square(1), 95% = 3.84, 99% = 6.63
    gating_threshold: float = 9.0  # ~99.7% for 1-DOF
    
    # SPD repair epsilon
    # Must be smaller than typical delay-space covariances (~1e-18 to 1e-24)
    spd_epsilon: float = 1e-50
    
    # Enable adaptive Q (future)
    enable_adaptive_Q: bool = False


@dataclass 
class FilterState:
    """State of the Kalman filter at time k."""
    
    k: int                          # Time index
    x: np.ndarray                   # State estimate [n_x]
    P: np.ndarray                   # State covariance [n_x, n_x]
    x_pred: Optional[np.ndarray]    # Predicted state (prior) [n_x]
    P_pred: Optional[np.ndarray]    # Predicted covariance (prior) [n_x, n_x]
    innovation: Optional[np.ndarray] = None  # Innovation (y - H@x_pred) [n_z]
    S: Optional[np.ndarray] = None  # Innovation covariance [n_z, n_z]
    K: Optional[np.ndarray] = None  # Kalman gain [n_x, n_z]
    valid_flag: bool = True         # Whether measurement was used
    nis: float = 0.0                # Normalized Innovation Squared
    nees: float = 0.0               # Normalized Estimation Error Squared (if truth known)


def ensure_spd(P: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    Ensure matrix P is symmetric positive definite.
    
    Strategy:
    1. Force symmetry: P = (P + P.T) / 2
    2. Eigendecomposition
    3. Clip negative eigenvalues to eps
    4. Reconstruct
    
    Args:
        P: Matrix to repair [n, n]
        eps: Minimum eigenvalue
        
    Returns:
        P_spd: Repaired SPD matrix [n, n]
    """
    # Force symmetry
    P_sym = 0.5 * (P + P.T)
    
    # Eigendecomposition
    eigvals, eigvecs = np.linalg.eigh(P_sym)
    
    # Check if repair needed
    if np.min(eigvals) >= eps:
        return P_sym
    
    # Clip negative eigenvalues
    eigvals_clipped = np.maximum(eigvals, eps)
    
    # Reconstruct
    P_spd = eigvecs @ np.diag(eigvals_clipped) @ eigvecs.T
    
    return P_spd


def joseph_form_update(P_pred: np.ndarray, K: np.ndarray, H: np.ndarray, 
                       R: np.ndarray) -> np.ndarray:
    """
    Joseph form covariance update for numerical stability.
    
    P_k|k = (I - K @ H) @ P_k|k-1 @ (I - K @ H).T + K @ R @ K.T
    
    This form is numerically stable and guarantees symmetry.
    
    Args:
        P_pred: Predicted (prior) covariance [n_x, n_x]
        K: Kalman gain [n_x, n_z]
        H: Observation matrix [n_z, n_x]
        R: Measurement noise covariance [n_z, n_z]
        
    Returns:
        P_post: Posterior covariance [n_x, n_x]
    """
    n_x = P_pred.shape[0]
    I = np.eye(n_x)
    
    IKH = I - K @ H  # [n_x, n_x]
    
    # Joseph form
    P_post = IKH @ P_pred @ IKH.T + K @ R @ K.T
    
    return P_post


def compute_nis(innovation: np.ndarray, S: np.ndarray) -> float:
    """
    Compute Normalized Innovation Squared (NIS).
    
    NIS = ν.T @ S^{-1} @ ν
    
    For consistent filter, NIS ~ chi-square(n_z)
    
    Args:
        innovation: Innovation vector [n_z]
        S: Innovation covariance [n_z, n_z]
        
    Returns:
        nis: NIS value (scalar)
    """
    try:
        S_inv = np.linalg.inv(S)
        nis = float(innovation.T @ S_inv @ innovation)
    except np.linalg.LinAlgError:
        nis = float('inf')
    
    return nis


def compute_nees(x_est: np.ndarray, x_true: np.ndarray, P: np.ndarray) -> float:
    """
    Compute Normalized Estimation Error Squared (NEES).
    
    NEES = ε.T @ P^{-1} @ ε, where ε = x_est - x_true
    
    For consistent filter, NEES ~ chi-square(n_x)
    
    Args:
        x_est: State estimate [n_x]
        x_true: True state [n_x]
        P: State covariance [n_x, n_x]
        
    Returns:
        nees: NEES value (scalar)
    """
    epsilon = x_est - x_true
    
    try:
        P_inv = np.linalg.inv(P)
        nees = float(epsilon.T @ P_inv @ epsilon)
    except np.linalg.LinAlgError:
        nees = float('inf')
    
    return nees


class EKFTracker:
    """
    Extended Kalman Filter with Joseph form and dynamic R.
    
    This is the core slow-loop algorithm for Phase III.
    
    Features:
    - Joseph form covariance update
    - SPD repair after each update
    - Gating for outlier rejection
    - Support for dynamic R_k from fast-loop
    """
    
    def __init__(self, cfg: TrackerConfig):
        """
        Initialize EKF tracker.
        
        Args:
            cfg: Tracker configuration
        """
        self.cfg = cfg
        self.state_cfg = cfg.state_config
        
        # Dimensions
        self.n_x = self.state_cfg.state_dim
        self.n_z = cfg.obs_dim
        
        # State space matrices
        self.F = compute_F_matrix(self.state_cfg.dt, self.state_cfg.motion_model)
        self.Q = compute_Q_kinematic(self.state_cfg.dt, self.state_cfg.S_jerk, 
                                     self.state_cfg.motion_model)
        self.H = compute_H_matrix(self.n_x, self.n_z)
        
        # Default R
        if cfg.default_R_diag.shape[0] != self.n_z:
            self.R_default = np.diag(np.full(self.n_z, cfg.default_R_diag[0]))
        else:
            self.R_default = np.diag(cfg.default_R_diag)
        
        # State storage
        self.x: Optional[np.ndarray] = None
        self.P: Optional[np.ndarray] = None
        self.k: int = 0
        
        # History for smoothing
        self.history: List[FilterState] = []
        
    def initialize(self, x0: np.ndarray, P0: Optional[np.ndarray] = None):
        """
        Initialize filter state.
        
        Args:
            x0: Initial state estimate [n_x]
            P0: Initial covariance [n_x, n_x]. If None, use default.
        """
        self.x = x0.copy()
        
        if P0 is None:
            P0 = compute_initial_covariance(self.state_cfg)
        self.P = P0.copy()
        
        self.k = 0
        self.history = []
        
        # Store initial state
        state = FilterState(
            k=0, x=self.x.copy(), P=self.P.copy(),
            x_pred=None, P_pred=None, valid_flag=True
        )
        self.history.append(state)
        
    def predict(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Time update (prediction) step.
        
        x_{k|k-1} = F @ x_{k-1|k-1}
        P_{k|k-1} = F @ P_{k-1|k-1} @ F.T + Q
        
        Returns:
            x_pred: Predicted state [n_x]
            P_pred: Predicted covariance [n_x, n_x]
        """
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        
        # Ensure SPD
        P_pred = ensure_spd(P_pred, self.cfg.spd_epsilon)
        
        return x_pred, P_pred
    
    def update(self, z: np.ndarray, R: Optional[np.ndarray] = None,
               valid_flag: bool = True, x_true: Optional[np.ndarray] = None
               ) -> FilterState:
        """
        Measurement update step with Joseph form.
        
        Args:
            z: Measurement vector [n_z]
            R: Measurement noise covariance [n_z, n_z]. If None, use default.
            valid_flag: If False, skip measurement update (coasting)
            x_true: True state for NEES computation (optional)
            
        Returns:
            state: FilterState with diagnostics
        """
        self.k += 1
        
        # Prediction step
        x_pred, P_pred = self.predict()
        
        # Use default R if not provided
        if R is None:
            R = self.R_default
        
        # Ensure R is properly shaped
        R = np.atleast_2d(R)
        if R.shape[0] != self.n_z:
            R = np.diag(np.full(self.n_z, R.flat[0]))
        
        if valid_flag:
            # Innovation
            innovation = z - self.H @ x_pred
            
            # Innovation covariance
            S = self.H @ P_pred @ self.H.T + R
            S = ensure_spd(S, self.cfg.spd_epsilon)
            
            # NIS for gating
            nis = compute_nis(innovation, S)
            
            # Gating check
            if nis > self.cfg.gating_threshold:
                # Outlier - skip measurement update
                warnings.warn(f"Measurement gated at k={self.k}: NIS={nis:.2f} > threshold={self.cfg.gating_threshold}")
                valid_flag = False
        
        if valid_flag:
            # Kalman gain
            try:
                S_inv = np.linalg.inv(S)
                K = P_pred @ self.H.T @ S_inv
            except np.linalg.LinAlgError:
                warnings.warn(f"S inversion failed at k={self.k}, skipping update")
                valid_flag = False
        
        if valid_flag:
            # State update
            x_post = x_pred + K @ innovation
            
            # Joseph form covariance update
            P_post = joseph_form_update(P_pred, K, self.H, R)
            P_post = ensure_spd(P_post, self.cfg.spd_epsilon)
            
            self.x = x_post
            self.P = P_post
        else:
            # Coasting - use prediction only
            self.x = x_pred
            self.P = P_pred
            innovation = np.zeros(self.n_z) if 'innovation' not in dir() else innovation
            S = R if 'S' not in dir() else S
            K = np.zeros((self.n_x, self.n_z))
            nis = 0.0 if 'nis' not in dir() else nis
        
        # Compute NEES if truth available
        nees = 0.0
        if x_true is not None:
            nees = compute_nees(self.x, x_true, self.P)
        
        # Create state record
        state = FilterState(
            k=self.k,
            x=self.x.copy(),
            P=self.P.copy(),
            x_pred=x_pred,
            P_pred=P_pred,
            innovation=innovation if valid_flag else None,
            S=S if valid_flag else None,
            K=K if valid_flag else None,
            valid_flag=valid_flag,
            nis=nis,
            nees=nees
        )
        
        self.history.append(state)
        
        return state
    
    def process_sequence(self, z_seq: np.ndarray, 
                         R_seq: Optional[np.ndarray] = None,
                         valid_flags: Optional[np.ndarray] = None,
                         x_true_seq: Optional[np.ndarray] = None
                         ) -> List[FilterState]:
        """
        Process a sequence of measurements.
        
        Args:
            z_seq: Measurement sequence [K, n_z]
            R_seq: Measurement noise sequence [K, n_z, n_z] or None
            valid_flags: Valid flag sequence [K] or None
            x_true_seq: True state sequence [K, n_x] or None
            
        Returns:
            states: List of FilterState for each time step
        """
        K = z_seq.shape[0]
        
        states = []
        for k in range(K):
            z_k = z_seq[k]
            R_k = R_seq[k] if R_seq is not None else None
            valid_k = valid_flags[k] if valid_flags is not None else True
            x_true_k = x_true_seq[k] if x_true_seq is not None else None
            
            state = self.update(z_k, R_k, valid_k, x_true_k)
            states.append(state)
        
        return states
    
    def get_estimates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get state estimates and covariances from history.
        
        Returns:
            x_est: State estimates [K+1, n_x]
            P_est: Covariances [K+1, n_x, n_x]
        """
        K = len(self.history)
        x_est = np.array([s.x for s in self.history])
        P_est = np.array([s.P for s in self.history])
        
        return x_est, P_est


class RTSSmoother:
    """
    Rauch-Tung-Striebel Smoother for offline refinement.
    
    The RTS smoother uses future information to improve past estimates.
    It requires a forward EKF pass first.
    """
    
    def __init__(self, ekf: EKFTracker):
        """
        Initialize RTS smoother from an EKF tracker.
        
        Args:
            ekf: EKF tracker with completed forward pass
        """
        self.ekf = ekf
        self.F = ekf.F
        self.Q = ekf.Q
        self.n_x = ekf.n_x
        
    def smooth(self, x_true_seq: Optional[np.ndarray] = None
               ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Perform RTS smoothing on the filter history.
        
        The backward recursion:
        C_k = P_{k|k} @ F.T @ P_{k+1|k}^{-1}
        x_{k|N} = x_{k|k} + C_k @ (x_{k+1|N} - x_{k+1|k})
        P_{k|N} = P_{k|k} + C_k @ (P_{k+1|N} - P_{k+1|k}) @ C_k.T
        
        Args:
            x_true_seq: True state sequence for NEES computation [K, n_x]
            
        Returns:
            x_smooth: Smoothed state estimates [K, n_x]
            P_smooth: Smoothed covariances [K, n_x, n_x]
            nees_smooth: Smoothed NEES values [K]
        """
        history = self.ekf.history
        K = len(history)
        
        # Initialize with final filtered estimates
        x_smooth = np.zeros((K, self.n_x))
        P_smooth = np.zeros((K, self.n_x, self.n_x))
        nees_smooth = np.zeros(K)
        
        # Last time step: smoothed = filtered
        x_smooth[K-1] = history[K-1].x
        P_smooth[K-1] = history[K-1].P
        
        if x_true_seq is not None:
            nees_smooth[K-1] = compute_nees(x_smooth[K-1], x_true_seq[K-1], P_smooth[K-1])
        
        # Backward recursion
        for k in range(K-2, -1, -1):
            x_kk = history[k].x
            P_kk = history[k].P
            
            # Get predicted values at k+1
            if history[k+1].P_pred is not None:
                P_pred = history[k+1].P_pred
                x_pred = history[k+1].x_pred
            else:
                # Recompute if not stored
                x_pred = self.F @ x_kk
                P_pred = self.F @ P_kk @ self.F.T + self.Q
            
            # Smoother gain
            try:
                P_pred_inv = np.linalg.inv(P_pred)
                C_k = P_kk @ self.F.T @ P_pred_inv
            except np.linalg.LinAlgError:
                # Fallback: pseudo-inverse
                C_k = P_kk @ self.F.T @ np.linalg.pinv(P_pred)
            
            # Smoothed estimates
            x_smooth[k] = x_kk + C_k @ (x_smooth[k+1] - x_pred)
            P_smooth[k] = P_kk + C_k @ (P_smooth[k+1] - P_pred) @ C_k.T
            # P0-1 Fix: 使用与EKF一致的spd_epsilon，避免1e-12默认值破坏delay-space尺度
            P_smooth[k] = ensure_spd(P_smooth[k], eps=self.ekf.cfg.spd_epsilon)
            
            # NEES
            if x_true_seq is not None:
                nees_smooth[k] = compute_nees(x_smooth[k], x_true_seq[k], P_smooth[k])
        
        return x_smooth, P_smooth, nees_smooth


def run_ekf_rts_pipeline(
    z_seq: np.ndarray,
    R_seq: Optional[np.ndarray] = None,
    valid_flags: Optional[np.ndarray] = None,
    x_true_seq: Optional[np.ndarray] = None,
    x0: Optional[np.ndarray] = None,
    P0: Optional[np.ndarray] = None,
    cfg: Optional[TrackerConfig] = None
) -> Dict[str, np.ndarray]:
    """
    Run complete EKF + RTS pipeline on a measurement sequence.
    
    Args:
        z_seq: Measurements [K, n_z]
        R_seq: Measurement noise [K, n_z, n_z] or None
        valid_flags: Valid flags [K] or None
        x_true_seq: True states [K, n_x] or None
        x0: Initial state [n_x] or None
        P0: Initial covariance [n_x, n_x] or None
        cfg: Tracker config or None (uses defaults)
        
    Returns:
        results: Dict with:
            - x_filt: Filtered estimates [K, n_x]
            - P_filt: Filtered covariances [K, n_x, n_x]
            - x_smooth: Smoothed estimates [K, n_x]
            - P_smooth: Smoothed covariances [K, n_x, n_x]
            - nees_filt: Filtered NEES [K]
            - nees_smooth: Smoothed NEES [K]
            - nis: NIS values [K]
    """
    if cfg is None:
        cfg = TrackerConfig()
    
    K = z_seq.shape[0]
    n_x = cfg.state_config.state_dim
    n_z = cfg.obs_dim
    
    # Initialize
    if x0 is None:
        # Use first measurement for initialization
        x0 = np.zeros(n_x)
        x0[0] = z_seq[0, 0]  # tau from measurement
        if n_z >= 2 and n_x >= 2:
            x0[1] = z_seq[0, 1]  # velocity if available
    
    # Create and initialize tracker
    ekf = EKFTracker(cfg)
    ekf.initialize(x0, P0)
    
    # Forward pass
    states = ekf.process_sequence(z_seq, R_seq, valid_flags, x_true_seq)
    
    # Get filtered results
    x_filt, P_filt = ekf.get_estimates()
    x_filt = x_filt[1:]  # Remove initial state
    P_filt = P_filt[1:]
    nees_filt = np.array([s.nees for s in states])
    nis = np.array([s.nis for s in states])
    
    # Backward pass (RTS smoothing)
    rts = RTSSmoother(ekf)
    
    # Need to include initial state for proper smoothing
    if x_true_seq is not None:
        x_true_with_init = np.vstack([x0[np.newaxis, :], x_true_seq])
    else:
        x_true_with_init = None
    
    x_smooth_full, P_smooth_full, nees_smooth_full = rts.smooth(x_true_with_init)
    x_smooth = x_smooth_full[1:]  # Remove initial state
    P_smooth = P_smooth_full[1:]
    nees_smooth = nees_smooth_full[1:]
    
    return {
        'x_filt': x_filt,
        'P_filt': P_filt,
        'x_smooth': x_smooth,
        'P_smooth': P_smooth,
        'nees_filt': nees_filt,
        'nees_smooth': nees_smooth,
        'nis': nis,
        'valid_flags': np.array([s.valid_flag for s in states])
    }


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("tracker.py - Self-Test")
    print("=" * 60)
    
    from state_space import generate_kinematic_trajectory
    
    # Generate test trajectory
    # State: [tau_res, tau_dot, tau_ddot] where tau is in seconds
    # tau_dot = v/c ≈ 100/3e8 ≈ 3.3e-7 (dimensionless rate)
    # tau_ddot = a/c ≈ 10/3e8 ≈ 3.3e-8 (rate of change)
    K = 50
    dt = 1e-3
    seed = 42
    c = 3e8
    
    # Convert physical v, a to delay rates
    v_phys = 100.0   # m/s
    a_phys = 5.0     # m/s²
    
    state_cfg = StateSpaceConfig(dt=dt, S_jerk=1e-20)  # Very small jerk for testing
    x0 = np.array([0.0, v_phys/c, a_phys/c])  # tau=0, tau_dot=v/c, tau_ddot=a/c
    
    x_true, _ = generate_kinematic_trajectory(
        K=K, dt=dt, x0=x0,
        motion_model=MotionModel.CONSTANT_ACCELERATION,
        seed=seed
    )
    
    print(f"\n[Setup] Generated trajectory: K={K}, dt={dt*1e3:.1f}ms")
    print(f"  Initial state: {x0}")
    print(f"  Final state: {x_true[-1]}")
    
    # Generate noisy measurements (delay-only)
    rng = np.random.default_rng(seed + 1)
    R_true = 1e-18  # (1 ns)^2
    z_seq = x_true[:, 0:1] + rng.normal(0, np.sqrt(R_true), (K, 1))
    
    print(f"\n[Measurements] Generated {K} noisy delay measurements")
    print(f"  Measurement noise std: {np.sqrt(R_true)*1e9:.2f} ns")
    
    # Configure tracker
    cfg = TrackerConfig(
        state_config=state_cfg,
        obs_dim=1,
        default_R_diag=np.array([R_true])
    )
    
    # Run pipeline
    print("\n[Running] EKF + RTS pipeline...")
    results = run_ekf_rts_pipeline(
        z_seq=z_seq,
        x_true_seq=x_true,
        x0=x0,
        cfg=cfg
    )
    
    # Compute metrics
    rmse_filt_tau = np.sqrt(np.mean((results['x_filt'][:, 0] - x_true[:, 0])**2))
    rmse_smooth_tau = np.sqrt(np.mean((results['x_smooth'][:, 0] - x_true[:, 0])**2))
    
    rmse_filt_v = np.sqrt(np.mean((results['x_filt'][:, 1] - x_true[:, 1])**2))
    rmse_smooth_v = np.sqrt(np.mean((results['x_smooth'][:, 1] - x_true[:, 1])**2))
    
    mean_nees_filt = np.mean(results['nees_filt'])
    mean_nees_smooth = np.mean(results['nees_smooth'])
    mean_nis = np.mean(results['nis'])
    
    print(f"\n[Results]")
    print(f"  RMSE tau (filtered):  {rmse_filt_tau*1e9:.3f} ns")
    print(f"  RMSE tau (smoothed):  {rmse_smooth_tau*1e9:.3f} ns")
    print(f"  RMSE v (filtered):    {rmse_filt_v:.3f} m/s")
    print(f"  RMSE v (smoothed):    {rmse_smooth_v:.3f} m/s")
    print(f"  Mean NEES (filtered): {mean_nees_filt:.2f} (expected ~{cfg.state_config.state_dim})")
    print(f"  Mean NEES (smoothed): {mean_nees_smooth:.2f}")
    print(f"  Mean NIS:             {mean_nis:.2f} (expected ~1)")
    
    # Verify improvement
    if rmse_smooth_tau < rmse_filt_tau:
        print(f"\n  ✓ Smoother improved tau RMSE by {(1-rmse_smooth_tau/rmse_filt_tau)*100:.1f}%")
    else:
        print(f"\n  ⚠ Smoother did not improve tau RMSE")
    
    # Protocol-0 check: RMSE should decrease with K
    print("\n[Protocol-0] Checking RMSE vs K trend...")
    rmse_vs_k = []
    for k_end in [10, 20, 30, 40, 50]:
        rmse_k = np.sqrt(np.mean((results['x_filt'][:k_end, 0] - x_true[:k_end, 0])**2))
        rmse_vs_k.append((k_end, rmse_k * 1e9))
        print(f"  K={k_end:2d}: RMSE_tau = {rmse_k*1e9:.3f} ns")
    
    # Check monotonic decrease
    is_decreasing = all(rmse_vs_k[i][1] >= rmse_vs_k[i+1][1] for i in range(len(rmse_vs_k)-1))
    if is_decreasing:
        print("  ✓ RMSE decreasing with K (Protocol-0 PASS)")
    else:
        print("  ⚠ RMSE not monotonically decreasing")
    
    print("\n" + "=" * 60)
    print("Self-Test Complete ✓")
    print("=" * 60)
