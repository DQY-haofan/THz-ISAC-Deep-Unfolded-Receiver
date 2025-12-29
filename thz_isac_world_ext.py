"""
thz_isac_world_ext.py - Phase III Extension for Multi-Frame Simulation

This module extends thz_isac_world.py with multi-frame sequence generation:
1. simulate_sequence(): Generate K-frame sequences with state continuity
2. Phase continuity across frames
3. Kinematic state propagation with process noise
4. Hardware state memory (optional: PA drift, etc.)

Based on:
- DR-III-E Section 3: Multi-frame data generation
- DR-III-T Section 2: State space model

Author: Phase III MVP
Date: 2025-12
"""

import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

# Import base thz_isac_world
import sys
sys.path.insert(0, '/mnt/user-data/uploads')

try:
    from thz_isac_world import (
        SimConfig, simulate_batch, compute_bcrlb_diag,
        generate_symbols, apply_pa_saleh, apply_phase_noise,
        wideband_delay_operator, doppler_phase_operator,
        add_thermal_noise, quantize, compute_sim_stats
    )
    HAS_BASE = True
except ImportError:
    print("[Warning] Could not import thz_isac_world, using standalone implementation")
    HAS_BASE = False

from state_space import (
    StateSpaceConfig, MotionModel, compute_F_matrix, compute_Q_kinematic,
    generate_kinematic_trajectory
)


@dataclass
class SequenceConfig:
    """Configuration for sequence generation."""
    
    # Base simulation config
    sim_config: SimConfig = None
    
    # Sequence parameters
    K: int = 100                  # Number of frames
    dt: float = 1e-3              # Frame interval [s]
    
    # Motion model
    motion_model: MotionModel = MotionModel.CONSTANT_ACCELERATION
    S_jerk: float = 1e-10         # Jerk PSD [m²/s⁵] - reduced for realistic LEO
    
    # Initial state [tau_res, v, a]
    # tau_res in samples, v in m/s, a in m/s²
    # NOTE: v must be small enough that tau drift stays within MF search window
    # tau_dot = v/c, so v=1 m/s → Δτ = 1*dt/c = 3.33e-12 s/frame = 0.033 samples/frame
    init_tau_samples: float = 0.0
    init_v: float = 1.0           # m/s - reduced from 100 to keep tau in MF window
    init_a: float = 0.01          # m/s² - small acceleration
    
    # Initial state noise (for robustness testing)
    init_noise_tau_samples: float = 0.0
    init_noise_v: float = 0.0
    init_noise_a: float = 0.0
    
    # Phase continuity
    enable_phase_continuity: bool = True
    
    # Hardware drift (future)
    enable_hw_drift: bool = False
    
    def __post_init__(self):
        if self.sim_config is None:
            self.sim_config = SimConfig()


def simulate_sequence(
    config: SequenceConfig,
    batch_size: int = 1,
    seed: Optional[int] = None,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Generate a multi-frame ISAC signal sequence with physical continuity.
    
    This is the main function for Phase III data generation.
    
    Args:
        config: Sequence configuration
        batch_size: Number of parallel sequences (B)
        seed: Random seed for reproducibility
        verbose: Print progress
        
    Returns:
        seq: Dictionary containing:
            - frames: List of K frame data dicts
            - truth: Dict with 'tau', 'v', 'a' trajectories [K, B]
            - meta: Sequence-level metadata
    """
    if seed is not None:
        rng = np.random.default_rng(seed)
    else:
        rng = np.random.default_rng()
    
    K = config.K
    dt = config.dt
    sim_cfg = config.sim_config
    
    # Setup state space
    Ts = 1.0 / sim_cfg.fs
    c = 3e8  # speed of light
    
    # Generate kinematic trajectory
    state_cfg = StateSpaceConfig(
        dt=dt,
        S_jerk=config.S_jerk,
        motion_model=config.motion_model
    )
    
    # Convert from physical units to delay-space units
    # State vector is [τ (s), τ_dot (s/s), τ_ddot (s/s²)]
    # where τ_dot = v/c and τ_ddot = a/c
    x0 = np.array([
        config.init_tau_samples * Ts,  # τ in seconds
        config.init_v / c,              # τ_dot = v/c (dimensionless rate)
        config.init_a / c               # τ_ddot = a/c
    ])
    
    # Add init noise (in physical units, converted to delay-space)
    if config.init_noise_tau_samples > 0:
        x0[0] += rng.normal(0, config.init_noise_tau_samples * Ts)
    if config.init_noise_v > 0:
        x0[1] += rng.normal(0, config.init_noise_v / c)
    if config.init_noise_a > 0:
        x0[2] += rng.normal(0, config.init_noise_a / c)
    
    # Process noise Q must also be scaled to delay-space
    # S_jerk is in m²/s⁵, need to convert to s²/s⁵ = 1/s³
    # Q_delay = Q_physical / c²
    S_jerk_delay = config.S_jerk / (c**2)
    Q_kin = compute_Q_kinematic(dt, S_jerk_delay, config.motion_model)
    
    x_traj, _ = generate_kinematic_trajectory(
        K=K, dt=dt, x0=x0, Q=Q_kin,
        motion_model=config.motion_model,
        seed=seed
    )
    
    # Extract trajectories
    # Note: x_traj is in delay-space [τ, τ_dot, τ_ddot]
    # Convert back to physical units for simulation
    tau_res_seq = x_traj[:, 0]       # [K] - τ in seconds
    v_seq = x_traj[:, 1] * c         # [K] - convert τ_dot to v (m/s)
    a_seq = (x_traj[:, 2] if x_traj.shape[1] > 2 else np.zeros(K)) * c  # a (m/s²)
    
    # Phase memory for continuity
    phase_memory = np.zeros(batch_size)
    
    # Generate frames
    frames = []
    
    for k in range(K):
        if verbose and k % 10 == 0:
            print(f"  Generating frame {k}/{K}...")
        
        # Update sim_config with current kinematic state
        # Note: We override the config temporarily
        frame_cfg = SimConfig(
            fc=sim_cfg.fc,
            fs=sim_cfg.fs,
            N=sim_cfg.N,
            P_in_dBm=sim_cfg.P_in_dBm,
            enable_pa=sim_cfg.enable_pa,
            alpha_a=sim_cfg.alpha_a,
            beta_a=sim_cfg.beta_a,
            alpha_phi=sim_cfg.alpha_phi,
            beta_phi=sim_cfg.beta_phi,
            ibo_dB=sim_cfg.ibo_dB,
            enable_pn=sim_cfg.enable_pn,
            pn_linewidth=sim_cfg.pn_linewidth,
            enable_channel=sim_cfg.enable_channel,
            R=sim_cfg.R,  # Keep base R, tau_res is handled separately
            v_rel=v_seq[k],
            a_rel=a_seq[k],
            coarse_acquisition_error_samples=tau_res_seq[k] / Ts,  # Back to samples
            phi0_random=sim_cfg.phi0_random,
            enable_quantization=sim_cfg.enable_quantization,
            snr_db=sim_cfg.snr_db,
            adc_bits=sim_cfg.adc_bits
        )
        
        # Generate frame using base simulate_batch
        if HAS_BASE:
            frame_data = simulate_batch(frame_cfg, batch_size=batch_size, 
                                        seed=seed + k if seed else None)
        else:
            # Fallback: generate simplified frame
            frame_data = _simulate_batch_simple(frame_cfg, batch_size, rng, k)
        
        # P0-5 Fix: Handle phase continuity
        if config.enable_phase_continuity:
            if k > 0:
                # Apply accumulated phase from previous frame to current frame
                # This ensures PN doesn't reset at frame boundaries
                frame_data['phase_offset'] = phase_memory.copy()
            else:
                frame_data['phase_offset'] = np.zeros(batch_size)
            
            # Update phase memory with end-of-frame phase
            # Phase accumulates due to: (1) PN random walk, (2) Doppler shift
            N = sim_cfg.N
            dt_frame = N / sim_cfg.fs  # Frame duration
            
            # Doppler-induced phase accumulation: Δφ = 2π * f_D * T
            # f_D = -f_c * v / c
            doppler_phase = -2 * np.pi * sim_cfg.fc * v_seq[k] / c * dt_frame
            
            # PN random walk accumulation (approximation based on linewidth)
            # σ_φ = sqrt(2π * Δf * T) where Δf is linewidth
            if sim_cfg.enable_pn and sim_cfg.pn_linewidth > 0:
                pn_std = np.sqrt(2 * np.pi * sim_cfg.pn_linewidth * dt_frame)
                pn_increment = rng.normal(0, pn_std, size=batch_size)
            else:
                pn_increment = np.zeros(batch_size)
            
            # Accumulate phase memory
            phase_memory += doppler_phase + pn_increment
            frame_data['phase_accumulated'] = phase_memory.copy()
        
        # Add frame index
        frame_data['frame_idx'] = k
        frame_data['tau_res_true'] = tau_res_seq[k]
        frame_data['v_true'] = v_seq[k]
        frame_data['a_true'] = a_seq[k]
        
        frames.append(frame_data)
    
    # Build truth arrays [K, B] or [K]
    truth = {
        'tau_res': tau_res_seq,          # [K]
        'v': v_seq,                       # [K]
        'a': a_seq,                       # [K]
        'tau_res_samples': tau_res_seq / Ts,  # [K]
        'x_state': x_traj                 # [K, 3]
    }
    
    # Metadata
    meta = {
        'K': K,
        'dt': dt,
        'batch_size': batch_size,
        'seed': seed,
        'snr_db': sim_cfg.snr_db,
        'adc_bits': sim_cfg.adc_bits,
        'fs': sim_cfg.fs,
        'N': sim_cfg.N,
        'motion_model': config.motion_model.value,
        'S_jerk': config.S_jerk,
        'init_state': x0.tolist()
    }
    
    return {
        'frames': frames,
        'truth': truth,
        'meta': meta
    }


def _simulate_batch_simple(cfg: SimConfig, batch_size: int, 
                           rng: np.random.Generator, frame_idx: int) -> Dict:
    """
    Simplified batch simulation fallback when thz_isac_world not available.
    
    This generates minimal data for testing the tracking pipeline.
    """
    N = cfg.N
    
    # Generate symbols
    bits = rng.integers(0, 4, size=(batch_size, N))
    constellation = np.array([1+1j, 1-1j, -1+1j, -1-1j]) / np.sqrt(2)
    x_true = constellation[bits]
    
    # Simple delay and Doppler
    tau_res = cfg.coarse_acquisition_error_samples / cfg.fs
    
    # Apply delay
    y = wideband_delay_operator(x_true, tau_res, cfg.fs)
    
    # Apply Doppler
    p_t = doppler_phase_operator(N, cfg.fs, cfg.fc, cfg.v_rel, cfg.a_rel)
    y = y * p_t[np.newaxis, :]
    
    # Add noise
    snr_lin = 10 ** (cfg.snr_db / 10)
    noise_power = 1.0 / snr_lin
    noise = rng.normal(0, np.sqrt(noise_power/2), (batch_size, N)) + \
            1j * rng.normal(0, np.sqrt(noise_power/2), (batch_size, N))
    y_noisy = y + noise
    
    # Quantize
    if cfg.enable_quantization and cfg.adc_bits == 1:
        y_q = (np.sign(y_noisy.real) + 1j * np.sign(y_noisy.imag)) / np.sqrt(2)
    else:
        y_q = y_noisy
    
    return {
        'x_true': x_true,
        'y_raw': y_noisy,
        'y_q': y_q,
        'theta_true': np.tile([tau_res, cfg.v_rel, cfg.a_rel], (batch_size, 1)),
        'meta': {
            'tau_res': tau_res,
            'tau_res_samples': tau_res * cfg.fs,
            'snr_db': cfg.snr_db,
            'snr_linear': snr_lin,
            'gamma_eff': 100.0,  # Placeholder
            'chi': 0.5,  # Placeholder
            'adc_bits': cfg.adc_bits
        }
    }


def simulate_sequence_simple(
    K: int = 100,
    dt: float = 1e-3,
    snr_db: float = 20.0,
    adc_bits: int = 1,
    init_tau_samples: float = 0.0,
    init_v: float = 100.0,
    init_a: float = 10.0,
    S_jerk: float = 1e-5,
    seed: Optional[int] = None,
    batch_size: int = 1,
    verbose: bool = False
) -> Dict[str, Any]:
    """
    Simplified interface for sequence generation.
    
    Args:
        K: Number of frames
        dt: Frame interval [s]
        snr_db: SNR [dB]
        adc_bits: ADC bits
        init_tau_samples: Initial delay error [samples]
        init_v: Initial velocity [m/s]
        init_a: Initial acceleration [m/s²]
        S_jerk: Jerk PSD [m²/s⁵]
        seed: Random seed
        batch_size: Number of parallel sequences
        verbose: Print progress
        
    Returns:
        seq: Sequence data dictionary
    """
    sim_cfg = SimConfig(
        snr_db=snr_db,
        adc_bits=adc_bits,
        enable_pa=True,
        enable_pn=True,
        enable_quantization=(adc_bits <= 8)
    )
    
    seq_cfg = SequenceConfig(
        sim_config=sim_cfg,
        K=K,
        dt=dt,
        S_jerk=S_jerk,
        init_tau_samples=init_tau_samples,
        init_v=init_v,
        init_a=init_a
    )
    
    return simulate_sequence(seq_cfg, batch_size=batch_size, seed=seed, verbose=verbose)


def extract_measurements_from_sequence(seq: Dict, 
                                        add_noise: bool = True,
                                        R_tau: float = 1e-18
                                        ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract measurement sequence from simulated data.
    
    For testing tracker without fast-loop: use oracle measurements.
    
    Args:
        seq: Sequence data from simulate_sequence
        add_noise: Add measurement noise
        R_tau: Measurement noise variance [s²]
        
    Returns:
        z_seq: Measurements [K, 1]
        R_seq: Noise covariances [K, 1, 1]
        x_true: True states [K, 3]
    """
    truth = seq['truth']
    K = len(truth['tau_res'])
    
    # True tau
    tau_true = truth['tau_res']  # [K]
    
    if add_noise:
        noise = np.random.randn(K) * np.sqrt(R_tau)
        z_seq = (tau_true + noise).reshape(K, 1)
    else:
        z_seq = tau_true.reshape(K, 1)
    
    R_seq = np.full((K, 1, 1), R_tau)
    
    x_true = truth['x_state']  # [K, 3]
    
    return z_seq, R_seq, x_true


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("thz_isac_world_ext.py - Self-Test")
    print("=" * 60)
    
    # Test sequence generation
    print("\n[Test 1] Sequence generation (simplified interface)")
    seq = simulate_sequence_simple(
        K=50,
        dt=1e-3,
        snr_db=20.0,
        adc_bits=1,
        init_tau_samples=0.5,
        init_v=100.0,
        init_a=10.0,
        seed=42,
        verbose=True
    )
    
    print(f"\n  Generated {len(seq['frames'])} frames")
    print(f"  Truth shape: tau_res={seq['truth']['tau_res'].shape}")
    print(f"  Initial tau_res: {seq['truth']['tau_res'][0]*1e9:.3f} ns")
    print(f"  Final tau_res: {seq['truth']['tau_res'][-1]*1e9:.3f} ns")
    print(f"  Tau_res drift: {(seq['truth']['tau_res'][-1] - seq['truth']['tau_res'][0])*1e9:.3f} ns")
    
    # Check frame structure
    frame0 = seq['frames'][0]
    print(f"\n  Frame 0 keys: {list(frame0.keys())}")
    if 'y_q' in frame0:
        print(f"  y_q shape: {frame0['y_q'].shape}")
    if 'x_true' in frame0:
        print(f"  x_true shape: {frame0['x_true'].shape}")
    
    # Test measurement extraction
    print("\n[Test 2] Measurement extraction")
    z_seq, R_seq, x_true = extract_measurements_from_sequence(seq, add_noise=True)
    print(f"  z_seq shape: {z_seq.shape}")
    print(f"  R_seq shape: {R_seq.shape}")
    print(f"  x_true shape: {x_true.shape}")
    print(f"  Measurement noise std: {np.sqrt(R_seq[0,0,0])*1e9:.3f} ns")
    
    # Test with full config
    print("\n[Test 3] Full configuration")
    if HAS_BASE:
        sim_cfg = SimConfig(
            snr_db=15.0,
            adc_bits=1,
            enable_pa=True,
            enable_pn=True,
            pn_linewidth=100e3
        )
    else:
        print("  (Using fallback sim config)")
        sim_cfg = SimConfig(snr_db=15.0, adc_bits=1)
    
    seq_cfg = SequenceConfig(
        sim_config=sim_cfg,
        K=20,
        dt=1e-3,
        S_jerk=1e-6,
        init_tau_samples=1.0,
        init_v=150.0,
        init_a=5.0
    )
    
    seq2 = simulate_sequence(seq_cfg, batch_size=1, seed=123)
    print(f"  Generated {seq2['meta']['K']} frames")
    print(f"  SNR: {seq2['meta']['snr_db']} dB")
    print(f"  Motion model: {seq2['meta']['motion_model']}")
    
    print("\n" + "=" * 60)
    print("Self-Test Complete ✓")
    print("=" * 60)
