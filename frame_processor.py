"""
frame_processor.py - Phase III Fast Loop Frame Processor

本模块实现快环帧处理器，与Micro-DR报告对应如下：

【报告§4 R_k Proxy的计算与裁剪规则】

  R_k来源 (Plug-in FIM Proxy):
    R_raw = (χ(SNR_eff) · J_analog + εI)^{-1}
    
    其中:
    - J_analog: 基于模拟波形的理想FIM
    - χ(SNR): 1-Bit信息保留因子
      * 低SNR区: χ ≈ 2/π (线性损失 -1.96dB)
      * 高SNR死区: χ → 0 (信号削波，信息丧失)

  裁剪规则 (Clipping Rules):
    规则1: σ²_min ≤ σ² ≤ σ²_max
    规则2: 条件数门控 κ(F_k) > κ_max → valid_flag=0
    规则3: 1-Bit死区检测 χ < χ_th → valid_flag=0, R→∞

【报告§4.3 valid_flag定义】
  valid_flag = (DUN_Converged) ∧ (SNR > Cliff_Thresh) ∧ (κ < κ_max)

【快环输出接口】 (报告§5)
  z_meas     : 观测向量 [τ_hat] 或 [τ_hat, f_D_hat]
  R_proxy    : 动态协方差矩阵 (对称正定)
  valid_flag : 显式有效标志
  risk_metric: 归一化风险分数 (0-1)

Key Features:
- R_k computed from chi-scaled FIM (1-bit information factor)
- Automatic validity flagging when in "forbidden region"
- Support for both MF-based and DUN-based estimators

参考文档:
- Micro-DR §4: R_k Proxy计算与裁剪规则
- DR-III-T Section 3: CRLB-guided R_k dynamic mapping

Author: Phase III MVP
Date: 2025-12
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Union, Callable
from enum import Enum

# Import geometry metrics for chi computation
try:
    import sys
    sys.path.insert(0, '/mnt/user-data/uploads')
    from geometry_metrics import chi_from_rho, compute_sinr_eff, estimate_gamma_eff
    HAS_GM = True
except ImportError:
    HAS_GM = False
    print("[Warning] geometry_metrics not available, using fallback chi computation")


class EstimatorType(Enum):
    """Type of fast-loop estimator."""
    MATCHED_FILTER = "MF"           # Correlation-based
    DEEP_UNFOLDING = "DUN"          # GA-BV-Net or similar
    ORACLE = "ORACLE"               # Uses true values + noise


@dataclass
class FrameProcessorConfig:
    """Configuration for Frame Processor."""
    
    # System parameters
    fs: float = 10e9           # Sampling rate [Hz]
    fc: float = 300e9          # Carrier frequency [Hz]
    N: int = 1024              # Samples per frame
    c: float = 3e8             # Speed of light [m/s]
    
    # ADC configuration
    adc_bits: int = 1          # ADC resolution
    
    # Estimator type
    estimator_type: EstimatorType = EstimatorType.MATCHED_FILTER
    
    # R_k computation
    # For MF with integer sample resolution, minimum std is ~0.5 samples
    # At fs=10GHz, Ts=1e-10, so min_std = 0.5*Ts = 5e-11 s
    # min_var = (0.5*Ts)^2 ≈ 2.5e-21 s^2
    enable_dynamic_R: bool = True
    R_min_tau: float = 2.5e-21   # MF quantization floor (~0.5 sample at 10GHz)
    R_max_tau: float = 1e-16    # Maximum tau variance (forbidden region)
    R_inflation_factor: float = 2.0  # Safety margin for model mismatch
    
    # Validity thresholds
    chi_threshold: float = 0.01    # Below this, mark as invalid
    snr_threshold_db: float = -15  # Below this, mark as invalid
    
    @property
    def Ts(self) -> float:
        return 1.0 / self.fs
    
    @property
    def delay_resolution(self) -> float:
        """Delay resolution [s]."""
        return 1.0 / self.fs


def chi_fallback(rho: float) -> float:
    """
    Fallback chi computation when geometry_metrics not available.
    
    chi(rho) = (2/pi) / (1 + kappa * rho)
    where kappa = 1 - 2/pi ≈ 0.3634
    """
    chi_low = 2.0 / np.pi
    kappa = 1.0 - chi_low
    return chi_low / (1.0 + kappa * max(rho, 0))


def compute_fim_tau_analog(N: int, B: float, snr_linear: float) -> float:
    """
    Compute analog (unquantized) Fisher Information for tau.
    
    From wideband delay model:
    J_tau = 8 * pi^2 * B^2 * N * SNR
    
    Args:
        N: Number of samples
        B: Bandwidth [Hz]
        snr_linear: Linear SNR
        
    Returns:
        J_tau: Fisher information for tau
    """
    return 8 * np.pi**2 * B**2 * N * snr_linear


def compute_dynamic_R(
    snr_db: float,
    gamma_eff: float,
    N: int,
    B: float,
    adc_bits: int = 1,
    cfg: Optional[FrameProcessorConfig] = None
) -> Tuple[float, float, bool]:
    """
    Compute dynamic measurement noise variance R_k for tau.
    
    Strategy:
    1. Compute effective SINR: rho_eff = 1/(1/SNR + 1/gamma_eff)
    2. Compute chi factor for 1-bit: chi(rho_eff)
    3. Compute analog FIM: J_analog = 8*pi^2*B^2*N*SNR
    4. Apply chi scaling: J_eff = chi * J_analog
    5. R_k = 1/J_eff (with bounds)
    
    Args:
        snr_db: SNR in dB
        gamma_eff: Hardware distortion ratio (linear)
        N: Number of samples
        B: Bandwidth [Hz]
        adc_bits: ADC bits
        cfg: Configuration (optional)
        
    Returns:
        R_tau: Measurement noise variance for tau [s^2]
        chi: Information retention factor
        valid_flag: True if measurement is usable
    """
    if cfg is None:
        cfg = FrameProcessorConfig()
    
    snr_linear = 10 ** (snr_db / 10)
    
    # Compute effective SINR (harmonic mean)
    if HAS_GM:
        sinr_eff = compute_sinr_eff(snr_linear, gamma_eff)
        chi = chi_from_rho(sinr_eff)
    else:
        sinr_eff = 1.0 / ((1.0 / max(snr_linear, 1e-12)) + (1.0 / max(gamma_eff, 1e-12)))
        chi = chi_fallback(sinr_eff)
    
    # For multi-bit ADC, chi ≈ 1
    if adc_bits > 1:
        chi = min(chi + 0.3 * (adc_bits - 1), 1.0)  # Gradual improvement
    
    # P0-3 Fix: Analog FIM应使用sinr_eff而非snr_linear
    # 这样硬件失真(gamma_eff降低)会同时影响chi和J_analog，使R_k正确膨胀
    J_analog = compute_fim_tau_analog(N, B, sinr_eff)
    
    # Effective FIM with chi scaling
    J_eff = chi * J_analog
    
    # Convert to variance (CRLB)
    if J_eff > 1e-20:
        R_tau = 1.0 / J_eff
    else:
        R_tau = cfg.R_max_tau
    
    # P0-2 Fix: 先乘inflation factor，再clip，确保最终值不超过R_max
    R_tau = R_tau * cfg.R_inflation_factor
    R_tau = np.clip(R_tau, cfg.R_min_tau, cfg.R_max_tau)
    
    # Validity check
    valid_flag = (chi >= cfg.chi_threshold) and (snr_db >= cfg.snr_threshold_db)
    
    return R_tau, chi, valid_flag


@dataclass
class FrameOutput:
    """Output from processing a single frame."""
    
    # Measurements
    z: np.ndarray              # Measurement vector [n_z]
    
    # Uncertainty
    R: np.ndarray              # Measurement noise covariance [n_z, n_z]
    
    # Diagnostics
    valid_flag: bool           # True if measurement is usable
    chi: float                 # Information retention factor
    risk_k: float              # Risk metric (0=safe, 1=forbidden)
    snr_db: float              # Estimated/true SNR
    gamma_eff: float           # Hardware distortion ratio
    
    # Optional diagnostics
    residual_energy: Optional[float] = None
    alpha_raw_neg_ratio: Optional[float] = None  # For Bussgang diagnostics


class FrameProcessor:
    """
    Fast-loop frame processor for Phase III.
    
    Converts raw frame data to measurements with uncertainty quantification.
    """
    
    def __init__(self, cfg: FrameProcessorConfig):
        """
        Initialize Frame Processor.
        
        Args:
            cfg: Configuration
        """
        self.cfg = cfg
        
        # Placeholder for model (DUN) - to be loaded
        self.model = None
        
    def load_model(self, model_path: str):
        """
        Load a trained DUN model.
        
        Args:
            model_path: Path to checkpoint
        """
        # TODO: Implement model loading
        # For MVP, we use MF or Oracle
        pass
    
    def _estimate_mf(self, y_q: np.ndarray, x_ref: np.ndarray
                     ) -> Tuple[float, float]:
        """
        Matched filter estimation for tau.
        
        Correlation peak gives coarse tau estimate.
        
        Args:
            y_q: Quantized received signal [N]
            x_ref: Reference (pilot) signal [N]
            
        Returns:
            tau_hat: Delay estimate [s]
            peak_value: Correlation peak magnitude
        """
        # Cross-correlation
        corr = np.correlate(y_q, x_ref, mode='full')
        
        # Find peak
        N = len(x_ref)
        center = N - 1
        
        # Search window (±5 samples around center)
        search_range = 5
        start = max(0, center - search_range)
        end = min(len(corr), center + search_range + 1)
        
        local_corr = np.abs(corr[start:end])
        peak_idx_local = np.argmax(local_corr)
        peak_idx = start + peak_idx_local
        
        # Convert to delay
        delay_samples = peak_idx - center
        tau_hat = delay_samples * self.cfg.Ts
        
        peak_value = np.abs(corr[peak_idx])
        
        return tau_hat, peak_value
    
    def _estimate_oracle(self, tau_true: float, snr_db: float, 
                         gamma_eff: float) -> float:
        """
        Oracle estimation (true value + noise based on CRLB).
        
        This provides a performance upper bound for benchmarking.
        
        Args:
            tau_true: True delay [s]
            snr_db: SNR in dB
            gamma_eff: Hardware distortion ratio
            
        Returns:
            tau_hat: Noisy delay estimate [s]
        """
        R_tau, chi, _ = compute_dynamic_R(
            snr_db, gamma_eff, self.cfg.N, self.cfg.fs,
            self.cfg.adc_bits, self.cfg
        )
        
        # Add CRLB-scaled noise
        noise = np.random.randn() * np.sqrt(R_tau)
        tau_hat = tau_true + noise
        
        return tau_hat
    
    def process(self, frame_data: Dict) -> FrameOutput:
        """
        Process a single frame and output measurement with uncertainty.
        
        Args:
            frame_data: Dict containing:
                - y_q: Quantized received signal [N] or [B, N]
                - x_ref: Reference (pilot) signal [N] (for MF)
                - meta: Metadata dict with snr_db, gamma_eff, etc.
                - tau_true: (optional) True delay for Oracle mode
                
        Returns:
            output: FrameOutput with z, R, validity, diagnostics
        """
        y_q = frame_data['y_q']
        meta = frame_data.get('meta', {})
        
        # Handle batched input (take first sample)
        if y_q.ndim == 2:
            y_q = y_q[0]
        
        # Get metadata
        snr_db = meta.get('snr_db', 20.0)
        gamma_eff = meta.get('gamma_eff', 100.0)
        
        # Estimate tau based on estimator type
        if self.cfg.estimator_type == EstimatorType.ORACLE:
            tau_true = frame_data.get('tau_true', 0.0)
            tau_hat = self._estimate_oracle(tau_true, snr_db, gamma_eff)
            peak_value = 1.0  # Oracle doesn't have peak
            
        elif self.cfg.estimator_type == EstimatorType.MATCHED_FILTER:
            x_ref = frame_data.get('x_ref', frame_data.get('x_true'))
            if x_ref is None:
                raise ValueError("MF estimator requires x_ref or x_true")
            if x_ref.ndim == 2:
                x_ref = x_ref[0]
            tau_hat, peak_value = self._estimate_mf(y_q, x_ref)
            
        else:  # DUN
            # TODO: Implement DUN inference
            # For now, fall back to MF
            x_ref = frame_data.get('x_ref', frame_data.get('x_true'))
            if x_ref is not None:
                if x_ref.ndim == 2:
                    x_ref = x_ref[0]
                tau_hat, peak_value = self._estimate_mf(y_q, x_ref)
            else:
                tau_hat = 0.0
                peak_value = 0.0
        
        # Compute dynamic R
        R_tau, chi, valid_flag = compute_dynamic_R(
            snr_db, gamma_eff, self.cfg.N, self.cfg.fs,
            self.cfg.adc_bits, self.cfg
        )
        
        # Build output
        z = np.array([tau_hat])
        R = np.array([[R_tau]])
        
        # Risk metric: 0 = safe (high chi), 1 = forbidden (low chi)
        risk_k = 1.0 - chi / (2.0 / np.pi)  # Normalize by chi_max
        risk_k = np.clip(risk_k, 0.0, 1.0)
        
        return FrameOutput(
            z=z,
            R=R,
            valid_flag=valid_flag,
            chi=chi,
            risk_k=risk_k,
            snr_db=snr_db,
            gamma_eff=gamma_eff,
            residual_energy=peak_value
        )
    
    def process_sequence(self, seq_data: Dict) -> Dict:
        """
        Process a sequence of frames.
        
        Args:
            seq_data: Dict containing:
                - frames: List of frame dicts, or batched arrays
                - meta_seq: List of metadata dicts
                
        Returns:
            results: Dict with:
                - z_seq: Measurements [K, n_z]
                - R_seq: Noise covariances [K, n_z, n_z]
                - valid_flags: [K]
                - chi_seq: [K]
                - risk_seq: [K]
        """
        frames = seq_data.get('frames', [])
        K = len(frames)
        
        z_list = []
        R_list = []
        valid_list = []
        chi_list = []
        risk_list = []
        
        for k, frame in enumerate(frames):
            output = self.process(frame)
            z_list.append(output.z)
            R_list.append(output.R)
            valid_list.append(output.valid_flag)
            chi_list.append(output.chi)
            risk_list.append(output.risk_k)
        
        return {
            'z_seq': np.array(z_list),
            'R_seq': np.array(R_list),
            'valid_flags': np.array(valid_list),
            'chi_seq': np.array(chi_list),
            'risk_seq': np.array(risk_list)
        }


def create_frame_processor(
    estimator_type: str = "MF",
    adc_bits: int = 1,
    fs: float = 10e9,
    N: int = 1024
) -> FrameProcessor:
    """
    Factory function to create a FrameProcessor.
    
    Args:
        estimator_type: "MF", "DUN", or "ORACLE"
        adc_bits: ADC resolution
        fs: Sampling rate
        N: Samples per frame
        
    Returns:
        processor: Configured FrameProcessor
    """
    est_type = EstimatorType[estimator_type.upper()]
    
    cfg = FrameProcessorConfig(
        fs=fs,
        N=N,
        adc_bits=adc_bits,
        estimator_type=est_type
    )
    
    return FrameProcessor(cfg)


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("frame_processor.py - Self-Test")
    print("=" * 60)
    
    # Test configuration
    cfg = FrameProcessorConfig(
        fs=10e9,
        N=1024,
        adc_bits=1,
        estimator_type=EstimatorType.ORACLE
    )
    
    processor = FrameProcessor(cfg)
    
    print(f"\n[Config]")
    print(f"  fs = {cfg.fs/1e9:.1f} GHz")
    print(f"  N = {cfg.N}")
    print(f"  ADC bits = {cfg.adc_bits}")
    print(f"  Estimator = {cfg.estimator_type.value}")
    
    # Test dynamic R computation
    print("\n[Test] Dynamic R computation vs SNR:")
    for snr_db in [-10, 0, 10, 20, 30]:
        R_tau, chi, valid = compute_dynamic_R(
            snr_db=snr_db,
            gamma_eff=100.0,
            N=cfg.N,
            B=cfg.fs,
            adc_bits=cfg.adc_bits,
            cfg=cfg
        )
        print(f"  SNR={snr_db:3d}dB: R_tau={np.sqrt(R_tau)*1e9:.3f}ns (std), "
              f"chi={chi:.4f}, valid={valid}")
    
    # Test frame processing (Oracle mode)
    print("\n[Test] Frame processing (Oracle mode):")
    
    rng = np.random.default_rng(42)
    tau_true = 1e-9  # 1 ns
    
    frame_data = {
        'y_q': rng.standard_normal(cfg.N) + 1j * rng.standard_normal(cfg.N),  # Dummy
        'tau_true': tau_true,
        'meta': {
            'snr_db': 20.0,
            'gamma_eff': 100.0
        }
    }
    
    output = processor.process(frame_data)
    print(f"  True tau: {tau_true*1e9:.3f} ns")
    print(f"  Estimated tau: {output.z[0]*1e9:.3f} ns")
    print(f"  Error: {(output.z[0] - tau_true)*1e9:.3f} ns")
    print(f"  R_tau std: {np.sqrt(output.R[0,0])*1e9:.3f} ns")
    print(f"  chi: {output.chi:.4f}")
    print(f"  valid: {output.valid_flag}")
    print(f"  risk: {output.risk_k:.3f}")
    
    # Test MF mode
    print("\n[Test] Frame processing (MF mode):")
    cfg_mf = FrameProcessorConfig(
        fs=10e9,
        N=1024,
        adc_bits=1,
        estimator_type=EstimatorType.MATCHED_FILTER
    )
    processor_mf = FrameProcessor(cfg_mf)
    
    # Generate simple test signal
    N = 1024
    delay_samples = 3
    x_ref = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y_delayed = np.roll(x_ref, delay_samples)
    y_noisy = y_delayed + 0.1 * (rng.standard_normal(N) + 1j * rng.standard_normal(N))
    
    frame_mf = {
        'y_q': y_noisy,
        'x_ref': x_ref,
        'meta': {'snr_db': 20.0, 'gamma_eff': 100.0}
    }
    
    output_mf = processor_mf.process(frame_mf)
    expected_tau = delay_samples * cfg_mf.Ts
    print(f"  True delay: {delay_samples} samples = {expected_tau*1e9:.3f} ns")
    print(f"  Estimated tau: {output_mf.z[0]*1e9:.3f} ns")
    print(f"  Error: {(output_mf.z[0] - expected_tau)*1e9:.3f} ns")
    
    print("\n" + "=" * 60)
    print("Self-Test Complete ✓")
    print("=" * 60)
