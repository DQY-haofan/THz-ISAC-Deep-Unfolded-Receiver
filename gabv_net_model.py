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
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()

        self.gate_net = nn.Sequential(
            nn.Linear(cfg.meta_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 4),  # [g_data, g_prior, g_pn, g_theta]
            nn.Sigmoid(),
        )

    def forward(self, meta: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute gates from meta features."""
        gates = self.gate_net(meta.float())

        return {
            'g_data': gates[:, 0:1],
            'g_prior': gates[:, 1:2],
            'g_pn': gates[:, 2:3],
            'g_theta': gates[:, 3:4],
        }


# =============================================================================
# Score-based Theta Updater
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

        # BCRLB-based scaling (learnable, initialized to sqrt(BCRLB))
        # These will be updated based on SNR during forward
        self.register_buffer('bcrlb_scale', torch.tensor([
            1e-11,   # sqrt(BCRLB_tau) ~ 10ps
            10.0,    # sqrt(BCRLB_v) ~ 10 m/s
            1.0,     # sqrt(BCRLB_a) ~ 1 m/s²
        ]))

        # Physical bounds for theta = [tau, v, a]
        # tau bounds: ±2 samples (resolution ~ 1/B)
        tau_bound = 2.0 / cfg.fs  # ±200ps for 10GHz
        self.register_buffer('theta_min', torch.tensor([-tau_bound, -1e4, -100.0]))
        self.register_buffer('theta_max', torch.tensor([tau_bound, 1e4, 100.0]))

        # Max delta per iteration (for stability)
        # Tied to physical resolution: ~0.1 sample for tau
        self.register_buffer('max_delta', torch.tensor([
            0.1 / cfg.fs,   # 0.1 sample period (10ps for 10GHz)
            100.0,          # 100 m/s per iteration
            10.0,           # 10 m/s² per iteration
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
        Compute physics-based score (gradient direction).

        score_k = Re(<∂y/∂θ_k, r>) / (||∂y/∂θ_k||² + ε)

        This is essentially a Gauss-Newton step direction.

        Args:
            residual: [B, N] complex residual
            dy_dtheta: Tuple of Jacobians (dy_dtau, dy_dv, dy_da)

        Returns:
            score: [B, 3] score vector
        """
        dy_dtau, dy_dv, dy_da = dy_dtheta

        eps = 1e-8

        # Score for tau
        inner_tau = torch.real(torch.sum(torch.conj(dy_dtau) * residual, dim=1, keepdim=True))
        norm_tau = torch.sum(torch.abs(dy_dtau)**2, dim=1, keepdim=True) + eps
        score_tau = inner_tau / norm_tau

        # Score for v
        inner_v = torch.real(torch.sum(torch.conj(dy_dv) * residual, dim=1, keepdim=True))
        norm_v = torch.sum(torch.abs(dy_dv)**2, dim=1, keepdim=True) + eps
        score_v = inner_v / norm_v

        # Score for a
        inner_a = torch.real(torch.sum(torch.conj(dy_da) * residual, dim=1, keepdim=True))
        norm_a = torch.sum(torch.abs(dy_da)**2, dim=1, keepdim=True) + eps
        score_a = inner_a / norm_a

        score = torch.cat([score_tau, score_v, score_a], dim=1)  # [B, 3]

        return score

    def forward(self,
                theta: torch.Tensor,
                residual: torch.Tensor,
                x_est: torch.Tensor,
                g_theta: torch.Tensor,
                phys_enc: PhysicsEncoder,
                y_obs: torch.Tensor,
                g_theta_sched: float = 1.0,
                snr_db: float = 20.0,
                ) -> Tuple[torch.Tensor, Dict]:
        """
        Update theta using score-based descent with BCRLB-aware scaling.

        Args:
            theta: Current estimate [B, 3]
            residual: [B, N] residual (y - H(θ)×x̂)
            x_est: [B, N] symbol estimate
            g_theta: [B, 1] gate from pilot navigator
            phys_enc: Physics encoder for Jacobian
            y_obs: [B, N] observed signal for acceptance test
            g_theta_sched: Scheduling factor (0→1 during warmup)
            snr_db: SNR in dB for BCRLB scaling

        Returns:
            theta_new: Updated theta [B, 3]
            info: Dictionary with diagnostics
        """
        B = theta.shape[0]
        device = theta.device

        # === Step 1: Compute Jacobian ===
        dy_dtheta = phys_enc.compute_channel_jacobian(theta, x_est)

        # === Step 2: Compute Score (Gauss-Newton direction) ===
        score = self.compute_score(residual, dy_dtheta)  # [B, 3]

        # === Step 3: Build Features ===
        residual_power = torch.mean(torch.abs(residual)**2, dim=1, keepdim=True)
        log_power = torch.log10(residual_power + 1e-10)

        score_magnitude = torch.abs(score)  # [B, 3]

        # Symbol confidence (how close to constellation points)
        # For QPSK: ideal normalized symbols have |real| = |imag| = 1/sqrt(2)
        # We need to normalize first, then check
        x_est_amplitude = torch.mean(torch.abs(x_est), dim=1, keepdim=True).clamp(min=1e-6)
        x_est_normalized = x_est / x_est_amplitude

        confidence = 1.0 - torch.mean(
            torch.abs(torch.abs(x_est_normalized.real) - 1/math.sqrt(2))**2 +
            torch.abs(torch.abs(x_est_normalized.imag) - 1/math.sqrt(2))**2,
            dim=1, keepdim=True
        )
        confidence = torch.clamp(confidence, 0, 1)

        # SNR normalization
        snr_norm = torch.tensor([[(snr_db - 15) / 15]], device=device).expand(B, 1)

        feat = torch.cat([
            log_power,           # [B, 1]
            score_magnitude,     # [B, 3]
            g_theta,             # [B, 1]
            confidence,          # [B, 1]
            snr_norm,            # [B, 1]
        ], dim=1).float()  # [B, 7]

        # === Step 4: Predict Step Sizes ===
        step_sizes = self.step_net(feat)  # [B, 3]

        # Scale step sizes by BCRLB (helps balance different units)
        step_sizes = step_sizes * self.bcrlb_scale.unsqueeze(0)

        # === Step 5: Compute Delta ===
        # delta_theta = -μ × score (negative because we want to minimize residual)
        delta_theta = -step_sizes * score

        # Clamp delta to physical limits
        delta_theta = torch.clamp(delta_theta, -self.max_delta, self.max_delta)

        # === Step 6: Apply Gate and Scheduling ===
        effective_gate = g_theta * g_theta_sched
        confidence_gate = torch.sigmoid((confidence - self.confidence_threshold) * 10)
        combined_gate = effective_gate * confidence_gate

        theta_candidate = theta + combined_gate * delta_theta

        # === Step 7: Clamp to Bounds ===
        theta_clamped = torch.clamp(theta_candidate, self.theta_min, self.theta_max)

        # === Step 8: Acceptance Test ===
        # Only accept if residual power decreases (with small relaxation)
        y_pred_old = phys_enc.forward_operator(x_est, theta)
        y_pred_new = phys_enc.forward_operator(x_est, theta_clamped)

        resid_old = torch.mean(torch.abs(y_obs - y_pred_old)**2, dim=1, keepdim=True)
        resid_new = torch.mean(torch.abs(y_obs - y_pred_new)**2, dim=1, keepdim=True)

        # Accept if new residual is better (with small tolerance)
        accept = (resid_new < resid_old * (1 + self.acceptance_relaxation)).float()

        # Soft acceptance: blend based on improvement ratio
        improvement = (resid_old - resid_new) / (resid_old + 1e-10)
        soft_accept = torch.sigmoid(improvement * 20)  # Soft version

        # Final theta: use soft acceptance for smoother gradients
        theta_final = soft_accept * theta_clamped + (1 - soft_accept) * theta

        # === Diagnostics ===
        info = {
            'score': score.detach(),
            'step_sizes': step_sizes.detach(),
            'delta_theta': delta_theta.detach(),
            'effective_gate': combined_gate.mean().item(),
            'accept_rate': accept.mean().item(),
            'soft_accept': soft_accept.mean().item(),
            'delta_tau': delta_theta[:, 0].abs().mean().item(),
            'delta_v': delta_theta[:, 1].abs().mean().item(),
            'delta_a': delta_theta[:, 2].abs().mean().item(),
            'confidence': confidence.mean().item(),
            'residual_improvement': improvement.mean().item(),
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

        # QPSK projection: snap to nearest constellation point (unit amplitude)
        phase = torch.angle(x_normalized)
        # QPSK phases: π/4, 3π/4, -3π/4, -π/4
        phase_quantized = torch.round(phase / (math.pi / 2)) * (math.pi / 2) + math.pi / 4
        x_qpsk = torch.exp(1j * phase_quantized)  # Unit amplitude QPSK

        # Scale back to original amplitude
        # FIX: No /sqrt(2)! Both x_qpsk and z_normalized are unit amplitude.
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

    Architecture:
        1. PN Tracker: Removes nuisance phase (including carrier phase)
        2. Pilot Navigator: Computes adaptive gates
        3. VAMP Layers: Iterative symbol estimation
        4. Theta Updater: Refines channel parameters using score-based descent
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.cfg = cfg

        # Physics encoder (wideband delay model)
        self.phys_enc = PhysicsEncoder(cfg)

        # PN tracker (absorbs nuisance phase)
        self.pn_tracker = RiemannianPNTracker(cfg)

        # Pilot navigator (gates)
        self.pilot = PilotNavigator(cfg)

        # VAMP solver layers
        self.solver_layers = nn.ModuleList([
            BussgangVAMPLayer(cfg) for _ in range(cfg.n_layers)
        ])

        # Theta updater (score-based)
        self.theta_updater = ScoreBasedThetaUpdater(cfg)

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

        # === Step 5: Iterative Refinement ===
        layer_outputs = []

        for k, layer in enumerate(self.solver_layers):
            # VAMP iteration
            z, x_est = layer(z, gamma, self.phys_enc, theta)

            # Theta update (if enabled)
            theta_info = {}
            if self.cfg.enable_theta_update and k >= self.cfg.theta_update_start_layer:
                # Re-compute Doppler-removed signal with current theta
                z_with_current_theta = self.phys_enc.adjoint_operator(y_q, theta)

                # Compute residual in Doppler-corrected space
                # x_est should match z_with_current_theta after PN correction
                residual = z_derotated - x_est

                # Update theta (with SNR for BCRLB scaling)
                theta, theta_info = self.theta_updater(
                    theta, residual, x_est,
                    gates['g_theta'], self.phys_enc,
                    y_q, g_theta_sched, snr_db
                )

                # Update z_derotated with new theta for consistency
                z_doppler_removed = self.phys_enc.adjoint_operator(y_q, theta)
                z_derotated = z_doppler_removed * torch.exp(-1j * phi_est)

            layer_outputs.append({
                'x_est': x_est.detach(),
                'theta': theta.detach(),
                'gates': {k: v.detach() for k, v in gates.items()},
                'theta_info': theta_info,
            })

        # === Step 6: Final Refinement with Residual ===
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