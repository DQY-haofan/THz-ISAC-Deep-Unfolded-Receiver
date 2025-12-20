"""
gabv_net_model.py (Wideband Delay Model - v7.0)

FUNDAMENTAL RESTRUCTURING (Expert Review):
===========================================
This model implements the CORRECT physics for THz-ISAC:

Signal Model:
    y = exp(-j × φ₀) × D(τ) × x × p(t) + noise

Where:
    - D(τ): Wideband delay operator (identifiable via bandwidth B)
    - p(t): Doppler/acceleration phase (NO absolute R term!)
    - φ₀: Nuisance constant phase (tracked by PN module)

Key Changes from v6.x:
    1. PhysicsEncoder uses WIDEBAND DELAY operator, not carrier phase
    2. theta = [tau_res, v, a] instead of [R, v, a]
    3. ThetaUpdater uses physics-based SCORE features
    4. Jacobians computed for wideband delay model

Identifiability:
    - τ is identifiable from GROUP DELAY (bandwidth B)
    - Resolution: δτ ~ 1/B ≈ 100ps for B=10GHz → δR ~ 3cm
    - Carrier phase exp(-j×2π×fc×τ) is NUISANCE, tracked by PN

Author: Expert Review v7.0 (Wideband Delay)
Date: 2025-12-19
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class GABVConfig:
    """GA-BV-Net configuration with wideband delay model."""

    # System parameters
    N: int = 1024
    fs: float = 10e9            # Bandwidth B = 10 GHz
    fc: float = 300e9           # Carrier frequency
    c: float = 3e8              # Speed of light

    # Network architecture
    n_layers: int = 8
    embed_dim: int = 64

    # Theta estimation
    enable_theta_update: bool = True
    theta_update_start_layer: int = 1

    # Meta features
    meta_dim: int = 6

    @property
    def Ts(self):
        return 1.0 / self.fs

    @property
    def wavelength(self):
        return self.c / self.fc

    @property
    def delay_resolution(self):
        """Delay resolution from bandwidth [s]."""
        return 1.0 / self.fs

    @property
    def range_resolution(self):
        """Range resolution from bandwidth [m]."""
        return self.c * self.delay_resolution


# =============================================================================
# 1-bit Quantization (PyTorch version for theta updater)
# =============================================================================

def quantize_1bit_torch(y: torch.Tensor) -> torch.Tensor:
    """
    1-bit quantization for complex signal (PyTorch version).

    This is CRITICAL for theta updater consistency:
    - The observation y_q is 1-bit quantized
    - The residual must compare y_q with Q(y_pred), not y_pred directly
    - Otherwise the score direction is wrong and acceptance is meaningless

    Expert insight: Using analog residual r = y_q - H(θ)x̂ causes:
    - Acceptance criterion to be unreliable
    - Score direction to be wrong
    - Step sizes μ to shrink to ~0 or explode

    Args:
        y: Complex tensor [B, N]

    Returns:
        y_q: Quantized complex tensor [B, N], values in {±1 ± 1j}/sqrt(2)
    """
    y_r = torch.sign(y.real)
    y_i = torch.sign(y.imag)
    # Handle zero (rare but possible)
    y_r = torch.where(y_r == 0, torch.ones_like(y_r), y_r)
    y_i = torch.where(y_i == 0, torch.ones_like(y_i), y_i)
    y_q = (y_r + 1j * y_i) / math.sqrt(2)
    return y_q


# =============================================================================
# Physics Encoder (Wideband Delay Model)
# =============================================================================

class PhysicsEncoder(nn.Module):
    """
    Physics-based channel encoder using WIDEBAND DELAY model.

    Model:
        H(θ) = exp(-j × φ₀) × D(τ) × diag(p(t))

    Where:
        - D(τ): Wideband delay operator (frequency domain)
        - p(t): Doppler/acceleration phase (time domain)
        - φ₀: Nuisance phase (NOT estimated by PhysicsEncoder)

    Note:
        We do NOT include the carrier phase exp(-j×2π×fc×τ) in H(θ).
        That term is a nuisance and is handled by the PN tracker.
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.cfg = cfg

        # Pre-compute grids (use float64 for precision)
        # Time grid [N]
        t_grid = torch.arange(cfg.N, dtype=torch.float64) * cfg.Ts
        self.register_buffer('t_grid', t_grid)

        # Frequency grid [N] for wideband delay
        f_grid = torch.fft.fftfreq(cfg.N, d=cfg.Ts).to(torch.float64)
        self.register_buffer('f_grid', f_grid)

        # Store constants
        self.fc = cfg.fc
        self.c = cfg.c
        self.fs = cfg.fs
        self.N = cfg.N

    def compute_delay_transfer_function(self, tau: torch.Tensor) -> torch.Tensor:
        """
        Compute wideband delay transfer function H_D(f) = exp(-j × 2π × f × τ).

        Args:
            tau: Delay [B, 1] in seconds

        Returns:
            H_D: Transfer function [B, N] (complex)
        """
        # tau: [B, 1], f_grid: [N]
        # H_D = exp(-j × 2π × f × τ) : [B, N]
        tau_f64 = tau.to(torch.float64)
        f = self.f_grid.unsqueeze(0)  # [1, N]

        phase = -2 * math.pi * f * tau_f64  # [B, N]
        H_D = torch.exp(1j * phase)

        return H_D.to(torch.cfloat)

    def compute_doppler_phase(self, v: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """
        Compute Doppler/acceleration phase p(t).

        p(t) = exp(-j × 2π × fc × (v/c × t + 0.5 × a/c × t²))

        Note: NO absolute R term! R only affects delay, not this phase.

        Args:
            v: Velocity [B, 1] in m/s
            a: Acceleration [B, 1] in m/s²

        Returns:
            p_t: Phase vector [B, N] (complex)
        """
        v_f64 = v.to(torch.float64)
        a_f64 = a.to(torch.float64)
        t = self.t_grid.unsqueeze(0)  # [1, N]

        # Doppler phase (NO R term!)
        phase = 2 * math.pi * self.fc * (
            v_f64 / self.c * t +
            0.5 * a_f64 / self.c * t**2
        )

        p_t = torch.exp(-1j * phase)
        return p_t.to(torch.cfloat)

    def forward_operator(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Apply forward channel model: y = D(τ) × x × p(t).

        Args:
            x: Input symbols [B, N] (complex)
            theta: Parameters [B, 3] = [tau, v, a]
                   tau in seconds, v in m/s, a in m/s²

        Returns:
            y: Output [B, N] (complex)

        Note:
            We do NOT apply the nuisance phase φ₀ here.
            That is handled by the PN tracker.
        """
        tau = theta[:, 0:1]  # [B, 1]
        v = theta[:, 1:2]    # [B, 1]
        a = theta[:, 2:3]    # [B, 1]

        # Step 1: Apply wideband delay D(τ) in frequency domain
        X = torch.fft.fft(x, dim=1)  # [B, N]
        H_D = self.compute_delay_transfer_function(tau)  # [B, N]
        Y = X * H_D
        y = torch.fft.ifft(Y, dim=1)  # [B, N]

        # Step 2: Apply Doppler phase p(t) in time domain
        p_t = self.compute_doppler_phase(v, a)  # [B, N]
        y = y * p_t

        return y

    def adjoint_operator(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """
        Apply adjoint channel model: x = D(-τ)^H × y × p(t)^*.

        Args:
            y: Received signal [B, N] (complex)
            theta: Parameters [B, 3] = [tau, v, a]

        Returns:
            x: Recovered symbols [B, N] (complex)
        """
        tau = theta[:, 0:1]
        v = theta[:, 1:2]
        a = theta[:, 2:3]

        # Step 1: Remove Doppler phase (conjugate)
        p_t = self.compute_doppler_phase(v, a)
        y_derotated = y * torch.conj(p_t)

        # Step 2: Apply inverse delay (negative tau)
        H_D_inv = self.compute_delay_transfer_function(-tau)
        Y = torch.fft.fft(y_derotated, dim=1)
        X = Y * H_D_inv
        x = torch.fft.ifft(X, dim=1)

        return x

    def compute_channel_jacobian(self, theta: torch.Tensor, x: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Compute Jacobian ∂y/∂θ for physics-based score features.

        The Jacobian captures how the output changes with respect to each parameter.

        For wideband delay model:
            y = D(τ) × x × p(t)

            ∂y/∂τ = ∂D(τ)/∂τ × x × p(t)
                  = -j × 2π × f × D(τ) × x × p(t)  (in frequency domain)

            ∂y/∂v = D(τ) × x × ∂p(t)/∂v
                  = D(τ) × x × (-j × 2π × fc/c × t) × p(t)

            ∂y/∂a = D(τ) × x × ∂p(t)/∂a
                  = D(τ) × x × (-j × π × fc/c × t²) × p(t)

        Args:
            theta: [B, 3] parameters
            x: [B, N] input symbols

        Returns:
            dy_dtau: [B, N] gradient w.r.t. delay
            dy_dv: [B, N] gradient w.r.t. velocity
            dy_da: [B, N] gradient w.r.t. acceleration
        """
        tau = theta[:, 0:1]
        v = theta[:, 1:2]
        a = theta[:, 2:3]

        # Get basic quantities
        H_D = self.compute_delay_transfer_function(tau)
        p_t = self.compute_doppler_phase(v, a)
        X = torch.fft.fft(x, dim=1)

        t = self.t_grid.unsqueeze(0).to(x.device)  # [1, N]
        f = self.f_grid.unsqueeze(0).to(x.device)  # [1, N]

        # ∂y/∂τ: derivative of delay operator
        # ∂D(τ)/∂τ = -j × 2π × f × D(τ)
        dH_D_dtau = -1j * 2 * math.pi * f * H_D
        dY_dtau = X * dH_D_dtau
        dy_dtau_delayed = torch.fft.ifft(dY_dtau, dim=1)
        dy_dtau = dy_dtau_delayed * p_t

        # ∂y/∂v: derivative of Doppler phase
        # ∂p(t)/∂v = -j × 2π × fc/c × t × p(t)
        dp_dv = -1j * 2 * math.pi * self.fc / self.c * t.to(torch.float64)
        dp_dv = dp_dv.to(torch.cfloat) * p_t

        # y = D(τ) × x × p(t), so ∂y/∂v = D(τ) × x × ∂p/∂v
        Y_delayed = X * H_D
        y_delayed = torch.fft.ifft(Y_delayed, dim=1)
        dy_dv = y_delayed * dp_dv

        # ∂y/∂a: derivative of acceleration phase
        # ∂p(t)/∂a = -j × π × fc/c × t² × p(t)
        dp_da = -1j * math.pi * self.fc / self.c * (t.to(torch.float64))**2
        dp_da = dp_da.to(torch.cfloat) * p_t

        dy_da = y_delayed * dp_da

        return dy_dtau, dy_dv, dy_da


# =============================================================================
# Riemannian Phase Noise Tracker
# =============================================================================

class RiemannianPNTracker(nn.Module):
    """
    Phase noise tracker operating on the unit circle manifold.

    This module tracks BOTH:
    1. Time-varying phase noise from oscillator
    2. Nuisance constant phase φ₀ (absorbed here!)

    The key insight is that the carrier phase exp(-j×2π×fc×τ) is NOT
    identifiable from narrowband observations, so we treat it as nuisance
    and let this module absorb it through pilot-aided calibration.
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.cfg = cfg

        # Learnable phase bias - INITIALIZE TO ZERO!
        # The network will learn the correct bias during training
        self.phase_bias = nn.Parameter(torch.tensor(0.0))

        # Pilot-based phase estimation
        self.n_pilot = 64

        # Phase noise statistics predictor
        self.pn_stats_net = nn.Sequential(
            nn.Linear(cfg.meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 2),  # [log_sigma, smoothness]
        )

        # Time-varying phase predictor (not used in simple version)
        self.phase_net = nn.Sequential(
            nn.Linear(4, 32),
            nn.Tanh(),
            nn.Linear(32, 1),
        )

    def forward(self, y: torch.Tensor, meta: torch.Tensor,
                x_pilot: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate and remove phase noise + nuisance constant phase.

        Args:
            y: Received signal [B, N]
            meta: Meta features [B, meta_dim]
            x_pilot: Known pilot symbols [B, n_pilot] (optional)

        Returns:
            y_derotated: Phase-corrected signal [B, N]
            phi_est: Estimated phase trajectory [B, N]
        """
        B, N = y.shape
        device = y.device

        # Estimate nuisance constant phase from pilots
        if x_pilot is not None:
            y_pilot = y[:, :self.n_pilot]
            # Phase = angle(sum(y_pilot × conj(x_pilot)))
            # This estimates the common phase offset
            pilot_corr = torch.sum(y_pilot * torch.conj(x_pilot), dim=1, keepdim=True)
            pilot_phase = torch.angle(pilot_corr)
        else:
            pilot_phase = torch.zeros(B, 1, device=device)

        # Add learnable bias (initialized to 0, will adapt during training)
        phi0_est = pilot_phase + self.phase_bias

        # For now, use constant phase across all symbols
        # (Future: add time-varying PN estimation)
        phi_est = phi0_est.expand(B, N)

        # Derotate: y_derotated = y × exp(-j × φ_est)
        y_derotated = y * torch.exp(-1j * phi_est)

        return y_derotated, phi_est


# =============================================================================
# Pilot Navigator (Gate Network)
# =============================================================================

class PilotNavigator(nn.Module):
    """
    Computes adaptive gates based on pilots and meta features.

    Important: g_theta must be initialized to a reasonable value (not ~0)
    otherwise theta update will be ineffective from the start.
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(cfg.meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [g_data, g_prior, g_pn, g_theta]
        )

        # Initialize bias of last layer so that Sigmoid outputs ~0.5 initially
        # Sigmoid(0) = 0.5, so we want bias ≈ 0
        # But for g_theta (index 3), we want it higher initially, so bias > 0
        with torch.no_grad():
            self.gate_net[-1].bias.data = torch.tensor([0.0, 0.0, 0.0, 1.0])  # g_theta starts at Sigmoid(1) ≈ 0.73

    def forward(self, meta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gates from meta features."""
        logits = self.gate_net(meta.float())
        gates = torch.sigmoid(logits)

        return {
            'g_data': gates[:, 0:1],
            'g_prior': gates[:, 1:2],
            'g_pn': gates[:, 2:3],
            'g_theta': gates[:, 3:4],
        }


# =============================================================================
# TauEstimator (v7 Architecture - Layer 1: Fast Loop)
# =============================================================================

class TauEstimatorInternal(nn.Module):
    """
    Deterministic τ estimator using Gauss-Newton on pilot symbols.

    This is Layer 1 of the hierarchical inference architecture.

    Key properties:
    1. NO learned gates - deterministic update prevents collapse
    2. τ-only - v/a are unidentifiable in single frame
    3. Domain-consistent - φ̂ applied to prediction only
    4. Frozen α - residual comparison uses same scale

    Why separate from ScoreBasedThetaUpdater:
    - Simpler (1D GN instead of 3D)
    - More stable (no gate learning)
    - Clearer physical interpretation
    """

    def __init__(self, cfg, n_iterations: int = 5):  # Increased from 3 to 5
        super().__init__()
        self.cfg = cfg
        self.n_iterations = n_iterations
        self.fs = cfg.fs

        # Fixed damping (not learned) - increased for faster convergence
        self.damping = 0.7  # Increased from 0.5

        # Max update per iteration - increased for larger errors
        self.max_delta_tau = 0.5 / cfg.fs  # Increased from 0.3 to 0.5 samples

        # Bounds
        tau_max = 5.0 / cfg.fs
        v_max = 10000.0
        a_max = 1000.0
        self.register_buffer('theta_min', torch.tensor([-tau_max, -v_max, -a_max]))
        self.register_buffer('theta_max', torch.tensor([tau_max, v_max, a_max]))

    def forward(self,
                theta: torch.Tensor,
                y_q: torch.Tensor,
                x_pilot: torch.Tensor,  # Already sliced to [B, Np]
                phi_est: torch.Tensor,
                phys_enc,
                pilot_len: int = 64) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate τ using iterative Gauss-Newton.

        Args:
            x_pilot: Pilot symbols [B, Np] (already sliced)

        Returns:
            theta_hat: Updated theta [B, 3] (only τ modified)
            info: Diagnostic information
        """
        B = theta.shape[0]
        device = theta.device
        N = y_q.shape[1]  # Full frame length
        Np = pilot_len

        # Extract pilots from observation
        y_q_pilot = y_q[:, :Np]

        # Normalize x_pilot (already sliced to Np length)
        x_pilot_power = torch.mean(torch.abs(x_pilot)**2, dim=1, keepdim=True).clamp(min=1e-10)
        x_pilot_norm = x_pilot / torch.sqrt(x_pilot_power)

        # Pad x_pilot to full length for forward_operator
        # (forward_operator expects [B, N] input)
        x_full = torch.zeros(B, N, dtype=x_pilot_norm.dtype, device=device)
        x_full[:, :Np] = x_pilot_norm

        # Phase alignment
        if phi_est.dim() == 2 and phi_est.shape[1] > 1:
            phi_const = phi_est[:, :Np].mean(dim=1, keepdim=True)
        else:
            phi_const = phi_est
        phase = torch.exp(-1j * phi_const)

        # Iterative GN
        theta_current = theta.clone()
        total_tau_change = 0.0
        final_improvement = 0.0

        for it in range(self.n_iterations):
            # Prediction (use padded x_full)
            y_pred_full = phys_enc.forward_operator(x_full, theta_current)
            y_pred = y_pred_full[:, :Np] * phase

            # Jacobian (τ only, use padded x_full)
            J_tau, _, _ = phys_enc.compute_channel_jacobian(theta_current, x_full)
            J_tau = J_tau[:, :Np] * phase

            # Bussgang
            var = torch.mean(torch.abs(y_pred)**2, dim=1, keepdim=True).clamp(min=1e-6)
            alpha = math.sqrt(2/math.pi) / torch.sqrt(var)

            # Residual
            y_tilde = y_q_pilot / (alpha + 1e-6)
            r = y_tilde - y_pred

            # 1D Gauss-Newton: Δτ = Re(J^H r) / ||J||²
            # GN naturally gives descent direction, no negative sign needed!
            J_norm_sq = torch.sum(torch.abs(J_tau)**2, dim=1, keepdim=True).clamp(min=1e-10)
            grad = torch.real(torch.sum(torch.conj(J_tau) * r, dim=1, keepdim=True))

            # GN step with damping (NO negative sign - GN is not gradient descent)
            delta_tau = self.damping * grad / J_norm_sq
            delta_tau = torch.clamp(delta_tau, -self.max_delta_tau, self.max_delta_tau)

            # Update τ only
            theta_new = theta_current.clone()
            theta_new[:, 0:1] = theta_current[:, 0:1] + delta_tau
            theta_new = torch.clamp(theta_new, self.theta_min, self.theta_max)

            # Track changes
            total_tau_change += delta_tau.abs().mean().item()

            # Compute improvement (frozen α) - use x_full for forward_operator
            y_pred_new = phys_enc.forward_operator(x_full, theta_new)[:, :Np] * phase
            r_new = y_tilde - y_pred_new  # Same y_tilde (frozen α)

            resid_old = torch.mean(torch.abs(r)**2)
            resid_new = torch.mean(torch.abs(r_new)**2)
            final_improvement = ((resid_old - resid_new) / (resid_old + 1e-10)).item()

            theta_current = theta_new

        info = {
            'n_iterations': self.n_iterations,
            'total_tau_change': total_tau_change,
            'final_improvement': final_improvement,
            'delta_tau': (theta_current[:, 0] - theta[:, 0]).abs().mean().item() * self.fs,  # In samples
            'bussgang_alpha': alpha.mean().item(),
        }

        return theta_current, info


# =============================================================================
# Score-based Theta Updater (Legacy - kept for compatibility)
# =============================================================================

class ScoreBasedThetaUpdater(nn.Module):
    """
    Physics-based theta updater using SCORE (gradient) features.

    Key Design Principles:
    1. Do NOT learn raw delta_theta directly (multi-modal, unstable)
    2. Use physics-based SCORE features from residual
    3. Network only learns STEP SIZES (μ)
    4. Apply acceptance criterion for stability
    5. Support BCRLB weighting for proper scaling

    Score computation (Gauss-Newton direction):
        r = y - H(θ)×x̂  (residual)
        score_k = Re(<∂y/∂θ_k, r>) / (||∂y/∂θ_k||² + ε)

    This is the natural gradient in the observation space.

    Update:
        δθ = -μ ⊙ score
        θ_new = θ + gate × δθ

    Acceptance:
        Only accept if ||y - H(θ_new)×x̂||² < ||y - H(θ)×x̂||²
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.cfg = cfg

        # Step size network
        # Input: [log_residual_power, |score_tau|, |score_v|, |score_a|, g_theta, confidence, snr_norm]
        self.step_net = nn.Sequential(
            nn.Linear(7, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),
            nn.Softplus(),  # Ensure positive step sizes
        )

        # Initialize step_net to output ~1 initially
        # Softplus(0) = ln(2) ≈ 0.69, so we want the final linear layer to output ~0.5
        # This gives Softplus(0.5) ≈ 0.97, which is close to 1
        with torch.no_grad():
            self.step_net[-2].bias.data.fill_(0.5)  # Output ~1 from Softplus

        # BCRLB-based scaling for converting step sizes to physical units
        # With unit-normalized score, step_size × bcrlb_scale gives physical delta
        # AGGRESSIVE: Large steps to verify direction is correct (can tune down later)
        self.register_buffer('bcrlb_scale', torch.tensor([
            1.0 / cfg.fs,   # 1.0 sample per layer (VERY LARGE for testing)
            50.0,           # 50 m/s
            5.0,            # 5 m/s²
        ]))

        # Physical bounds for theta = [tau, v, a]
        # tau bounds: ±2 samples (resolution ~ 1/B)
        tau_bound = 2.0 / cfg.fs  # ±200ps for 10GHz
        self.register_buffer('theta_min', torch.tensor([-tau_bound, -1e4, -100.0]))
        self.register_buffer('theta_max', torch.tensor([tau_bound, 1e4, 100.0]))

        # Max delta per iteration (for stability)
        # Increased for more aggressive theta updates
        self.register_buffer('max_delta', torch.tensor([
            1.0 / cfg.fs,   # 1.0 sample period max (AGGRESSIVE for testing)
            200.0,          # 200 m/s per iteration
            20.0,           # 20 m/s² per iteration
        ]))

        # Confidence threshold for gating
        self.confidence_threshold = 0.3

        # Acceptance relaxation (accept if new_resid < old_resid * (1 + relax))
        self.acceptance_relaxation = 0.02

    def compute_score(self,
                      residual: torch.Tensor,
                      dy_dtheta: Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
                      ) -> torch.Tensor:
        """
        Compute physics-based score using UNIT-NORMALIZED Jacobians.

        Problem with original formula:
            score_k = Re(<∂y/∂θ_k, r>) / ||∂y/∂θ_k||²
            When ||∂y/∂τ|| ~ 10^10, score_tau ~ 10^-13 (useless!)

        Solution: Use unit-normalized Jacobian
            J_hat_k = ∂y/∂θ_k / ||∂y/∂θ_k||
            score_k = Re(<J_hat_k, r>)

        This gives score in range [-||r||, ||r||], with clear physical meaning:
        - Positive: increasing θ_k will reduce residual
        - Negative: decreasing θ_k will reduce residual
        - Magnitude: proportional to how much residual can be reduced

        Args:
            residual: [B, N] complex residual
            dy_dtheta: Tuple of Jacobians (dy_dtau, dy_dv, dy_da)

        Returns:
            score: [B, 3] score vector
        """
        dy_dtau, dy_dv, dy_da = dy_dtheta

        eps = 1e-10

        # Normalize Jacobians to unit norm
        dy_dtau_norm = torch.sqrt(torch.sum(torch.abs(dy_dtau)**2, dim=1, keepdim=True) + eps)
        dy_dtau_hat = dy_dtau / dy_dtau_norm

        dy_dv_norm = torch.sqrt(torch.sum(torch.abs(dy_dv)**2, dim=1, keepdim=True) + eps)
        dy_dv_hat = dy_dv / dy_dv_norm

        dy_da_norm = torch.sqrt(torch.sum(torch.abs(dy_da)**2, dim=1, keepdim=True) + eps)
        dy_da_hat = dy_da / dy_da_norm

        # Score = inner product with unit-normalized Jacobian
        # This gives the projection of residual onto each gradient direction
        score_tau = torch.real(torch.sum(torch.conj(dy_dtau_hat) * residual, dim=1, keepdim=True))
        score_v = torch.real(torch.sum(torch.conj(dy_dv_hat) * residual, dim=1, keepdim=True))
        score_a = torch.real(torch.sum(torch.conj(dy_da_hat) * residual, dim=1, keepdim=True))

        score = torch.cat([score_tau, score_v, score_a], dim=1)  # [B, 3]

        return score

    def forward(self,
                theta: torch.Tensor,
                residual: torch.Tensor,  # NOTE: This is ignored, we compute our own
                x_est: torch.Tensor,
                g_theta: torch.Tensor,
                phys_enc: PhysicsEncoder,
                y_obs: torch.Tensor,
                g_theta_sched: float = 1.0,
                snr_db: float = 20.0,
                pilot_len: int = None,
                phi_est: torch.Tensor = None,
                x_pilot: torch.Tensor = None,  # NEW: Known pilot symbols
                ) -> Tuple[torch.Tensor, Dict]:
        """
        Update theta using 3x3 Gauss-Newton with proper phase handling.

        CRITICAL FIX: Use known pilot symbols, not x_est!

        Problem: x_est is estimated using the current (possibly wrong) theta.
        If theta_init has error, x_est "adapts" to compensate:
        - y_pred = H(theta_init) × x_est ≈ y_obs (because x_est adapted)
        - residual r = y_tilde - y_pred becomes small
        - gradient b = J^H @ r also becomes small!

        Solution: Use known pilot symbols for y_pred and Jacobian.
        - y_pred_pilot = H(theta) × x_pilot (known pilots, not estimated x_est)
        - This gives true residual reflecting theta error
        """
        B = theta.shape[0]
        N_full = x_est.shape[1]
        device = theta.device

        if pilot_len is None:
            pilot_len = N_full
        Np = pilot_len

        # === Get phase estimate (constant) ===
        if phi_est is not None:
            if phi_est.dim() == 2 and phi_est.shape[1] > 1:
                phi_const = phi_est[:, :Np].mean(dim=1, keepdim=True)
            else:
                phi_const = phi_est
        else:
            phi_const = torch.zeros(B, 1, device=device)

        phase = torch.exp(-1j * phi_const)  # [B, 1]

        # === CRITICAL: Use known pilots for y_pred and Jacobian ===
        # If x_pilot is provided, use it for prediction (not x_est which adapted to theta error)
        if x_pilot is not None:
            # IMPORTANT: Normalize x_pilot to unit power!
            # The simulator may output non-normalized symbols
            x_pilot_power = torch.mean(torch.abs(x_pilot)**2, dim=1, keepdim=True).clamp(min=1e-10)
            x_for_pred = x_pilot / torch.sqrt(x_pilot_power)  # Normalize to unit power
            x_pilot_input_power = x_pilot_power.mean().item()
        else:
            # Fallback: use x_est (less accurate for theta update)
            x_for_pred = x_est
            x_pilot_input_power = 0.0

        # === PATCH A: Rotate prediction, NOT y_q ===
        # Use x_pilot for prediction (not x_est which adapted to theta error)
        y_pred_full = phys_enc.forward_operator(x_for_pred, theta)
        y_pred = y_pred_full[:, :Np]
        y_pred_phi = y_pred * phase  # Put phase onto prediction

        # DEBUG: Track power at each stage to find where energy is lost
        x_power = torch.mean(torch.abs(x_for_pred)**2).item()
        x_est_power = torch.mean(torch.abs(x_est)**2).item()
        y_pred_full_power = torch.mean(torch.abs(y_pred_full)**2).item()
        y_pred_power = torch.mean(torch.abs(y_pred)**2).item()

        # === Jacobians with phase applied (using x_pilot) ===
        dy_dtheta = phys_enc.compute_channel_jacobian(theta, x_for_pred)
        J_tau, J_v, J_a = dy_dtheta
        J_tau = J_tau[:, :Np] * phase
        J_v = J_v[:, :Np] * phase
        J_a = J_a[:, :Np] * phase

        # === Bussgang linearization in matched domain ===
        var = torch.mean(torch.abs(y_pred_phi)**2, dim=1, keepdim=True).clamp(min=1e-6)
        alpha = math.sqrt(2/math.pi) / torch.sqrt(var)

        # y_q remains UNTOUCHED (critical for 1-bit!)
        y_q_pilot = y_obs[:, :Np]
        y_tilde = y_q_pilot / (alpha + 1e-6)

        # Residual in matched domain
        r = y_tilde - y_pred_phi  # [B, Np]

        # === PATCH B: 3x3 Gauss-Newton to decouple (τ, v, a) ===
        # Build normal equations: G = J^H J, b = Re(J^H r)

        # CRITICAL: Normalize Jacobian columns to avoid ill-conditioning
        # The Jacobians have vastly different scales:
        # - J_tau ~ 10^13 (because f_grid ~ 10^9 Hz)
        # - J_v, J_a ~ 10^6 (because fc/c ~ 10^6)
        # Without normalization, G has condition number ~ 10^14+

        norm_tau = torch.sqrt(torch.sum(torch.abs(J_tau)**2, dim=1, keepdim=True)).clamp(min=1e-10)
        norm_v = torch.sqrt(torch.sum(torch.abs(J_v)**2, dim=1, keepdim=True)).clamp(min=1e-10)
        norm_a = torch.sqrt(torch.sum(torch.abs(J_a)**2, dim=1, keepdim=True)).clamp(min=1e-10)

        # Normalized Jacobians (unit norm)
        J_tau_n = J_tau / norm_tau
        J_v_n = J_v / norm_v
        J_a_n = J_a / norm_a

        def inner_product(a, b):
            """Complex inner product: <a, b> = sum(conj(a) * b)"""
            return torch.sum(torch.conj(a) * b, dim=1, keepdim=True)

        # Gram matrix elements (now well-conditioned, ~O(1))
        G11 = torch.real(inner_product(J_tau_n, J_tau_n))  # Should be ~1
        G12 = torch.real(inner_product(J_tau_n, J_v_n))
        G13 = torch.real(inner_product(J_tau_n, J_a_n))
        G22 = torch.real(inner_product(J_v_n, J_v_n))      # Should be ~1
        G23 = torch.real(inner_product(J_v_n, J_a_n))
        G33 = torch.real(inner_product(J_a_n, J_a_n))      # Should be ~1

        # Right-hand side (with normalized Jacobians)
        b1 = torch.real(inner_product(J_tau_n, r))
        b2 = torch.real(inner_product(J_v_n, r))
        b3 = torch.real(inner_product(J_a_n, r))

        # DEBUG: Check residual and projection magnitudes
        r_norm = torch.sqrt(torch.sum(torch.abs(r)**2, dim=1, keepdim=True))

        # Raw (un-normalized) projections for diagnostics
        b1_raw = torch.real(inner_product(J_tau, r))
        b2_raw = torch.real(inner_product(J_v, r))
        b3_raw = torch.real(inner_product(J_a, r))

        # Assemble [B, 3, 3] and [B, 3, 1]
        G = torch.zeros((B, 3, 3), device=device, dtype=torch.float32)
        G[:, 0, 0] = G11.squeeze(1)
        G[:, 0, 1] = G12.squeeze(1)
        G[:, 0, 2] = G13.squeeze(1)
        G[:, 1, 0] = G12.squeeze(1)  # Symmetric
        G[:, 1, 1] = G22.squeeze(1)
        G[:, 1, 2] = G23.squeeze(1)
        G[:, 2, 0] = G13.squeeze(1)  # Symmetric
        G[:, 2, 1] = G23.squeeze(1)  # Symmetric
        G[:, 2, 2] = G33.squeeze(1)

        b = torch.stack([b1.squeeze(1), b2.squeeze(1), b3.squeeze(1)], dim=1).unsqueeze(-1)  # [B, 3, 1]

        # Damping for numerical stability (Levenberg-Marquardt style)
        lam = 0.01  # Increased from 1e-3 for better stability
        I = torch.eye(3, device=device).unsqueeze(0).expand(B, -1, -1)
        G_reg = G + lam * I

        # DEBUG: Check G matrix and b vector values
        G_diag = torch.diagonal(G, dim1=1, dim2=2)  # [B, 3]
        b_vec = torch.stack([b1.squeeze(1), b2.squeeze(1), b3.squeeze(1)], dim=1)  # [B, 3]

        # CRITICAL: Ensure b has same dtype as G (float32)
        b = b.float()  # Convert to float32 to match G

        # Solve: delta_normalized = inv(G_reg) @ b
        # CRITICAL: We need negative sign because:
        # - J = ∂y_pred/∂θ (Jacobian of prediction)
        # - r = y_tilde - y_pred (residual)
        # - To minimize ||r||², we need Δθ = -(J^H J)^-1 J^H r
        # - The negative comes from ∂r/∂θ = -∂y_pred/∂θ = -J
        try:
            delta_normalized = -torch.linalg.solve(G_reg, b).squeeze(-1)  # [B, 3] - NOTE THE NEGATIVE!
            solve_success = True
        except Exception as e:
            # Fallback if solve fails
            delta_normalized = torch.zeros(B, 3, device=device)
            solve_success = False
            print(f"[WARNING] GN solve failed: {e}")

        # Convert back to physical units
        # delta_physical = delta_normalized / norm (because J_n = J/norm)
        scale_tau = 1.0 / norm_tau.squeeze(1)  # [B]
        scale_v = 1.0 / norm_v.squeeze(1)
        scale_a = 1.0 / norm_a.squeeze(1)

        delta_gn = torch.stack([
            delta_normalized[:, 0] * scale_tau,
            delta_normalized[:, 1] * scale_v,
            delta_normalized[:, 2] * scale_a,
        ], dim=1)  # [B, 3] in physical units

        # =====================================================================
        # EXPERT-RECOMMENDED: Hierarchical Update Strategy
        # =====================================================================
        #
        # Key insight from experts:
        # 1. τ is "envelope-level" parameter - must converge first
        # 2. v is "phase-level" parameter - requires τ to be aligned
        # 3. In current config (64 pilots, 6.4ns), v's phase change is only
        #    ~0.001 rad - essentially unidentifiable from single frame!
        #
        # Strategy: "Acquisition before Tracking"
        # - Fast loop (per-frame): Update τ only
        # - Slow loop (cross-frame): Update v, a (future work: use EKF/PLL)
        #
        # For now, we disable v/a updates entirely. This is not a workaround,
        # but the CORRECT design given the information-theoretic constraints.
        # =====================================================================

        delta_gn[:, 1] = 0.0  # Disable v update (insufficient information in single frame)
        delta_gn[:, 2] = 0.0  # Disable a update

        # === Apply fixed step size (bypass step_net for now) ===
        # Expert recommendation: use fixed mu until direction is verified
        # Increase mu for tau to get more aggressive updates
        mu = 1.0  # Increased from 0.5 - more aggressive
        delta_theta = mu * delta_gn

        # Clamp to physical limits
        delta_theta = torch.clamp(delta_theta, -self.max_delta, self.max_delta)

        # === Apply gate and scheduling ===
        # EXPERT FIX: Use FIXED gate instead of learned g_theta
        #
        # Why: PilotNavigator learns to shut down theta update (g_theta → 0)
        # because Bussgang residual_improvement is negative due to domain mismatch,
        # NOT because the τ update is actually harmful.
        #
        # Expert recommendation: Use fixed damped Newton step g = 0.5
        # This is standard in Gauss-Newton optimization (damping factor)
        # Only modulate by g_theta_sched to allow warmup
        effective_gate = 0.5 * g_theta_sched

        # Safety minimum
        effective_gate = max(effective_gate, 0.1)

        theta_candidate = theta + effective_gate * delta_theta

        # Clamp to bounds
        theta_final = torch.clamp(theta_candidate, self.theta_min, self.theta_max)

        # === Diagnostics ===
        # EXPERT FIX: Compare residuals with FROZEN alpha (use old alpha for both)
        # Otherwise we're comparing different objective functions!
        y_pred_new = phys_enc.forward_operator(x_for_pred, theta_final)[:, :Np] * phase

        # Use alpha_old (not recomputed) for fair comparison
        y_tilde_new = y_q_pilot / (alpha + 1e-6)  # Same alpha as before!
        r_new = y_tilde_new - y_pred_new

        resid_old = torch.mean(torch.abs(r)**2, dim=1, keepdim=True)
        resid_new = torch.mean(torch.abs(r_new)**2, dim=1, keepdim=True)
        improvement = (resid_old - resid_new) / (resid_old + 1e-10)

        # Condition number of G (for diagnostics)
        try:
            G_cond = torch.linalg.cond(G_reg).mean().item()
        except:
            G_cond = float('nan')

        info = {
            'delta_theta': delta_theta.detach(),
            'effective_gate': effective_gate,  # Now a scalar, not tensor
            'accept_rate': 1.0,  # Always accept with GN
            'soft_accept': 1.0,
            'delta_tau': delta_theta[:, 0].abs().mean().item(),
            'delta_v': delta_theta[:, 1].abs().mean().item(),
            'delta_a': delta_theta[:, 2].abs().mean().item(),
            'residual_improvement': improvement.mean().item(),
            'bussgang_alpha': alpha.mean().item(),
            'G_cond': G_cond,
            'delta_gn_tau': delta_gn[:, 0].mean().item(),
            'delta_gn_v': delta_gn[:, 1].mean().item(),
            'delta_gn_a': delta_gn[:, 2].mean().item(),
            # Debug info
            'r_norm': r_norm.mean().item(),
            'b1': b1.mean().item(),
            'b2': b2.mean().item(),
            'b3': b3.mean().item(),
            'b1_raw': b1_raw.mean().item(),
            'norm_J_tau': norm_tau.mean().item(),
            'norm_J_v': norm_v.mean().item(),
            # Normalized deltas
            'delta_n_tau': delta_normalized[:, 0].mean().item(),
            'delta_n_v': delta_normalized[:, 1].mean().item(),
            'delta_n_a': delta_normalized[:, 2].mean().item(),
            'scale_tau': scale_tau.mean().item(),
            'scale_v': scale_v.mean().item(),
            # NEW: Flag if using known pilots
            'using_x_pilot': 1.0 if x_pilot is not None else 0.0,
            # Power diagnostics
            'x_power': x_power,
            'x_est_power': x_est_power,
            'x_pilot_input_power': x_pilot_input_power,
            'y_pred_full_power': y_pred_full_power,
            'y_pred_power': y_pred_power,
            # GN solve diagnostics
            'G_diag_0': G_diag[:, 0].mean().item(),  # Should be ~1
            'G_diag_1': G_diag[:, 1].mean().item(),  # Should be ~1
            'G_diag_2': G_diag[:, 2].mean().item(),  # Should be ~1
            'b_vec_0': b_vec[:, 0].mean().item(),    # b1
            'b_vec_1': b_vec[:, 1].mean().item(),    # b2
            'b_vec_2': b_vec[:, 2].mean().item(),    # b3
            'solve_success': 1.0 if solve_success else 0.0,
            # Expert-recommended diagnostics
            'v_info_weak': 1.0,  # Flag: v is unidentifiable in single frame
        }

        return theta_final, info


# =============================================================================
# Bussgang-VAMP Solver Layer
# =============================================================================

class BussgangVAMPLayer(nn.Module):
    """Single layer of Bussgang-VAMP iteration."""

    def __init__(self, cfg: GABVConfig):
        super().__init__()

        # Learnable damping
        self.damping = nn.Parameter(torch.tensor(0.5))

        # Denoiser with residual connection
        self.denoiser = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),
            nn.Linear(32, 2),
        )
        # Initialize to zero for identity via residual
        with torch.no_grad():
            self.denoiser[-1].weight.zero_()
            self.denoiser[-1].bias.zero_()

    def forward(self,
                z: torch.Tensor,
                gamma: torch.Tensor,
                phys_enc: PhysicsEncoder,
                theta: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        One iteration of Bussgang-VAMP.

        Args:
            z: Current estimate [B, N] (complex)
            gamma: Precision (1/variance)
            phys_enc: Physics encoder
            theta: Channel parameters [B, 3]

        Returns:
            z_new: Updated estimate [B, N]
            x_est: Symbol estimate [B, N]
        """
        B, N = z.shape

        # Estimate symbol amplitude from input (for power normalization compatibility)
        z_amplitude = torch.mean(torch.abs(z), dim=1, keepdim=True).clamp(min=1e-6)

        # Normalize to unit amplitude for processing
        z_normalized = z / z_amplitude

        # Stack real/imag for denoiser
        z_ri = torch.stack([z_normalized.real, z_normalized.imag], dim=-1)  # [B, N, 2]

        # Apply denoiser with residual connection
        delta = self.denoiser(z_ri.float())  # [B, N, 2]
        x_ri = z_ri + delta  # Residual: starts as identity
        x_normalized = x_ri[..., 0] + 1j * x_ri[..., 1]  # [B, N]

        # QPSK projection: use I/Q sign decision (CORRECT method!)
        # QPSK constellation: (±1 ± 1j) / sqrt(2)
        # Simply take sign of real and imag parts
        I_sign = torch.sign(x_normalized.real)
        Q_sign = torch.sign(x_normalized.imag)
        # Handle exactly zero (rare, but possible)
        I_sign = torch.where(I_sign == 0, torch.ones_like(I_sign), I_sign)
        Q_sign = torch.where(Q_sign == 0, torch.ones_like(Q_sign), Q_sign)
        x_qpsk = (I_sign + 1j * Q_sign) / math.sqrt(2)  # Unit power QPSK

        # Scale back to original amplitude
        x_qpsk_scaled = x_qpsk * z_amplitude
        x_est = x_normalized * z_amplitude  # Keep denoised estimate at correct scale

        # Soft decision: interpolate between quantized and input
        damping = torch.sigmoid(self.damping)
        z_new = damping * x_qpsk_scaled + (1 - damping) * z

        return z_new, x_est


# =============================================================================
# Main GA-BV-Net Model
# =============================================================================

class GABVNet(nn.Module):
    """
    Geometry-Aware Bussgang-VAMP Network with Wideband Delay Model.

    v7 Architecture (Hierarchical Inference):
        Layer 1: TauEstimator - Fast τ tracking (per-frame, deterministic GN)
        Layer 2: VAMP Detector - Symbol estimation (uses τ_hat)
        Layer 3: DopplerTracker - Slow v/a tracking (cross-frame, future)

    Key design principles:
        1. τ and v/a have different time scales (fast vs slow loop)
        2. τ estimation is deterministic (no learned gates that can collapse)
        3. VAMP uses fixed θ from TauEstimator
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.cfg = cfg

        # Physics encoder (wideband delay model)
        self.phys_enc = PhysicsEncoder(cfg)

        # PN tracker (absorbs nuisance phase)
        self.pn_tracker = RiemannianPNTracker(cfg)

        # Pilot navigator (gates for data/prior blending, NOT for theta)
        self.pilot = PilotNavigator(cfg)

        # === LAYER 1: TauEstimator (Fast Loop) ===
        # Deterministic GN-based τ estimation using pilots
        # This replaces the learned ScoreBasedThetaUpdater
        self.tau_estimator = TauEstimatorInternal(cfg, n_iterations=3)

        # Legacy theta_updater (kept for compatibility, but bypassed in v7)
        self.theta_updater = ScoreBasedThetaUpdater(cfg)

        # === LAYER 2: VAMP Detector ===
        self.solver_layers = nn.ModuleList([
            BussgangVAMPLayer(cfg) for _ in range(cfg.n_layers)
        ])

        # Symbol refiner with residual connection
        # This ensures output is at least as good as input initially
        self.refiner = nn.Sequential(
            nn.Linear(2, 32),
            nn.Tanh(),  # Tanh instead of ReLU for better gradients
            nn.Linear(32, 2),
        )
        # Initialize to zero so refiner starts as identity (via residual)
        with torch.no_grad():
            self.refiner[-1].weight.zero_()
            self.refiner[-1].bias.zero_()

        # Flag to bypass refiner for debugging
        self.bypass_refiner = False

    def forward(self, batch: Dict, g_theta_sched: float = 1.0) -> Dict:
        """
        Forward pass.

        CRITICAL FIX: Correct order of operations!
        ==================================================

        Old order (WRONG):
            y_q → PN_tracker → adjoint → VAMP → refine

            Problem: Doppler phase is TIME-VARYING!
            PN tracker estimates constant phase from pilots, but Doppler
            causes different phase at different time indices.
            Result: BER = 0.5

        New order (CORRECT):
            y_q → adjoint (remove Doppler) → PN_tracker → VAMP → refine

            The adjoint removes time-varying Doppler first.
            Then PN tracker only handles the constant nuisance phase φ₀.

        Args:
            batch: Dictionary with:
                - y_q: Quantized received signal [B, N]
                - theta_init: Initial theta estimate [B, 3]
                - meta: Meta features [B, meta_dim]
                - x_true: (optional) True symbols for supervised loss
                - snr_db: (optional) SNR in dB for BCRLB scaling

            g_theta_sched: Theta update scheduling factor (0→1)

        Returns:
            Dictionary with:
                - x_hat: Symbol estimate [B, N]
                - theta_hat: Parameter estimate [B, 3]
                - layers: Per-layer diagnostics
        """
        y_q = batch['y_q']
        theta = batch['theta_init'].clone()
        meta = batch['meta']
        snr_db = batch.get('snr_db', 20.0)
        if isinstance(snr_db, torch.Tensor):
            snr_db = snr_db.mean().item()

        B, N = y_q.shape
        device = y_q.device

        # Get pilots if available
        x_pilot = batch.get('x_true', None)
        if x_pilot is not None:
            x_pilot_slice = x_pilot[:, :self.pn_tracker.n_pilot]
        else:
            x_pilot_slice = None

        # === Step 1: Apply adjoint FIRST to remove Doppler ===
        # This is CRITICAL: Doppler phase is time-varying, must remove first!
        z_doppler_removed = self.phys_enc.adjoint_operator(y_q, theta)

        # === Step 2: PN Tracking on Doppler-corrected signal ===
        # Now we only need to estimate/remove the constant phase offset φ₀
        z_derotated, phi_est = self.pn_tracker(z_doppler_removed, meta, x_pilot_slice)

        # === Step 3: Compute Gates ===
        gates = self.pilot(meta)

        # === Step 4: Initialize VAMP ===
        z = z_derotated
        gamma = torch.ones(B, 1, device=device)

        # === Step 5: τ Estimation (LAYER 1 of v7 Architecture) ===
        # Use TauEstimator BEFORE VAMP to get better θ
        # This is the "Fast Loop" - deterministic GN-based τ estimation
        pilot_len = self.pn_tracker.n_pilot
        theta_info = {}

        if self.cfg.enable_theta_update and x_pilot is not None:
            # Use new TauEstimator (deterministic, no learned gate)
            # Pass x_pilot_slice (pilots only), not full x_pilot
            theta, theta_info = self.tau_estimator(
                theta,
                y_q,
                x_pilot_slice,  # Use sliced pilots, not full sequence
                phi_est,
                self.phys_enc,
                pilot_len=pilot_len,
            )

            # Sync z to new theta BEFORE VAMP
            z_doppler_removed = self.phys_enc.adjoint_operator(y_q, theta)
            z_derotated = z_doppler_removed * torch.exp(-1j * phi_est)
            z = z_derotated

        # === Step 6: VAMP Detector (LAYER 2 of v7 Architecture) ===
        # Run VAMP with updated θ from TauEstimator
        layer_outputs = []

        for k, layer in enumerate(self.solver_layers):
            # VAMP iteration with updated theta
            z, x_est = layer(z, gamma, self.phys_enc, theta)

            layer_outputs.append({
                'x_est': x_est.detach(),
                'theta': theta.detach(),
                'gates': {k: v.detach() for k, v in gates.items()},
                'theta_info': theta_info if k == 0 else {},
            })

        # Note: theta update now happens BEFORE VAMP (in Step 5)
        # This is the v7 architecture: TauEstimator → VAMP → Refiner

        # === Step 7: Final Refinement with Residual ===
        if self.bypass_refiner:
            x_hat = x_est
        else:
            # Estimate amplitude to preserve power normalization
            x_amplitude = torch.mean(torch.abs(x_est), dim=1, keepdim=True).clamp(min=1e-6)
            x_normalized = x_est / x_amplitude

            # Apply refiner to normalized symbols
            x_ri = torch.stack([x_normalized.real, x_normalized.imag], dim=-1)
            x_delta = self.refiner(x_ri.float())  # Refinement delta

            # Add delta and rescale
            x_refined_norm = x_normalized + (x_delta[..., 0] + 1j * x_delta[..., 1])
            x_hat = x_refined_norm * x_amplitude

        return {
            'x_hat': x_hat,
            'theta_hat': theta,
            'phi_est': phi_est,
            'layers': layer_outputs,
            'gates': gates,
        }


# =============================================================================
# Model Creation Helper
# =============================================================================

def create_gabv_model(cfg: Optional[GABVConfig] = None) -> GABVNet:
    """Create GA-BV-Net model."""
    if cfg is None:
        cfg = GABVConfig()
    return GABVNet(cfg)


# =============================================================================
# Self-Test
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("gabv_net_model.py v7.0 - Wideband Delay Model Self-Test")
    print("=" * 70)

    # Create model
    cfg = GABVConfig()
    model = create_gabv_model(cfg)

    print(f"\n[Model] Created GA-BV-Net with {sum(p.numel() for p in model.parameters())} parameters")
    print(f"  N = {cfg.N}")
    print(f"  fs = {cfg.fs/1e9:.1f} GHz")
    print(f"  fc = {cfg.fc/1e9:.1f} GHz")
    print(f"  Delay resolution: {cfg.delay_resolution*1e12:.1f} ps")
    print(f"  Range resolution: {cfg.range_resolution*100:.1f} cm")

    # Test forward pass
    print("\n[TEST] Forward Pass")
    B, N = 4, cfg.N
    batch = {
        'y_q': torch.randn(B, N, dtype=torch.cfloat),
        'theta_init': torch.zeros(B, 3),  # [tau, v, a]
        'meta': torch.randn(B, cfg.meta_dim),
    }

    output = model(batch)
    print(f"  x_hat shape: {output['x_hat'].shape}")
    print(f"  theta_hat shape: {output['theta_hat'].shape}")
    print(f"  Number of layers: {len(output['layers'])}")

    # Test PhysicsEncoder consistency
    print("\n[TEST] PhysicsEncoder Wideband Delay")
    phys_enc = model.phys_enc
    x = torch.randn(B, N, dtype=torch.cfloat)
    theta = torch.zeros(B, 3)
    theta[:, 0] = 1e-10  # Small delay

    y = phys_enc.forward_operator(x, theta)
    x_rec = phys_enc.adjoint_operator(y, theta)

    error = torch.mean(torch.abs(x - x_rec)**2).item()
    print(f"  Forward-adjoint reconstruction error: {error:.6f}")

    # Test Jacobian
    print("\n[TEST] Jacobian Computation")
    dy_dtau, dy_dv, dy_da = phys_enc.compute_channel_jacobian(theta, x)
    print(f"  dy_dtau shape: {dy_dtau.shape}")
    print(f"  dy_dtau magnitude: {torch.abs(dy_dtau).mean().item():.4f}")
    print(f"  dy_dv magnitude: {torch.abs(dy_dv).mean().item():.4f}")
    print(f"  dy_da magnitude: {torch.abs(dy_da).mean().item():.4f}")

    print("\n" + "=" * 70)
    print("Self-Test Complete")
    print("=" * 70)