# tau_estimator.py
"""
TauEstimator: Fast-loop delay estimation module

This is Layer 1 of the hierarchical inference architecture:
- Uses ONLY pilot symbols (known)
- Deterministic Gauss-Newton τ estimation
- Outputs: τ_hat (per-frame)

Key design principles:
1. Domain-consistent: φ̂ applied to prediction, not observation
2. Frozen α: Compare residuals with same scale
3. τ-only update: v/a disabled (unidentifiable in single frame)
"""

import torch
import torch.nn as nn
import math
from typing import Dict, Tuple, Optional


class TauEstimator(nn.Module):
    """
    Deterministic τ estimator using Gauss-Newton on pilot symbols.

    This module is designed to be:
    1. Stable: No learned gates that can collapse
    2. Physics-consistent: Respects 1-bit quantization constraints
    3. Fast: Single-pass estimation per frame

    Architecture:
        Input: y_q (quantized obs), x_pilot (known pilots), θ_init, φ̂
        Output: θ_hat with updated τ (v, a unchanged)
    """

    def __init__(self, cfg, n_iterations: int = 3):
        """
        Args:
            cfg: Configuration object with fs, fc, etc.
            n_iterations: Number of GN iterations per frame
        """
        super().__init__()
        self.cfg = cfg
        self.n_iterations = n_iterations

        # Physical constants
        self.fs = cfg.fs
        self.fc = cfg.fc
        self.c = 3e8

        # Damping factor (fixed, not learned)
        self.damping = 0.5  # Standard GN damping

        # Max update per iteration for stability
        self.max_delta_tau = 0.5 / cfg.fs  # 0.5 samples max

        # Bounds
        tau_max = 5.0 / cfg.fs  # 5 samples
        v_max = 10000.0
        a_max = 1000.0
        self.register_buffer('theta_min', torch.tensor([-tau_max, -v_max, -a_max]))
        self.register_buffer('theta_max', torch.tensor([tau_max, v_max, a_max]))

    def forward(self,
                theta: torch.Tensor,
                y_q: torch.Tensor,
                x_pilot: torch.Tensor,
                phi_est: torch.Tensor,
                phys_enc,  # PhysicsEncoder for H(θ) and Jacobian
                pilot_len: int = 64) -> Tuple[torch.Tensor, Dict]:
        """
        Estimate τ using iterative Gauss-Newton.

        Args:
            theta: Initial theta [B, 3] = [τ, v, a]
            y_q: Quantized observation [B, N] complex
            x_pilot: Known pilot symbols [B, N] complex
            phi_est: Phase estimate [B, N] or [B, 1]
            phys_enc: Physics encoder for forward model
            pilot_len: Number of pilot symbols

        Returns:
            theta_hat: Updated theta [B, 3] (only τ is modified)
            info: Diagnostic information
        """
        B = theta.shape[0]
        device = theta.device
        Np = pilot_len

        # Extract pilot observations
        y_q_pilot = y_q[:, :Np]

        # Normalize x_pilot to unit power
        x_pilot_power = torch.mean(torch.abs(x_pilot) ** 2, dim=1, keepdim=True).clamp(min=1e-10)
        x_pilot_norm = x_pilot / torch.sqrt(x_pilot_power)

        # Get constant phase for prediction alignment
        if phi_est.dim() == 2 and phi_est.shape[1] > 1:
            phi_const = phi_est[:, :Np].mean(dim=1, keepdim=True)
        else:
            phi_const = phi_est
        phase = torch.exp(-1j * phi_const)

        # Iterative GN updates
        theta_current = theta.clone()
        total_improvement = 0.0

        for iteration in range(self.n_iterations):
            # Compute prediction with current θ
            y_pred_full = phys_enc.forward_operator(x_pilot_norm, theta_current)
            y_pred = y_pred_full[:, :Np] * phase  # Apply phase to prediction

            # Compute Jacobian (only τ component needed)
            J_tau, J_v, J_a = phys_enc.compute_channel_jacobian(theta_current, x_pilot_norm)
            J_tau = J_tau[:, :Np] * phase  # Apply phase to Jacobian

            # Bussgang linearization
            var = torch.mean(torch.abs(y_pred) ** 2, dim=1, keepdim=True).clamp(min=1e-6)
            alpha = math.sqrt(2 / math.pi) / torch.sqrt(var)

            # Residual (y_q unchanged, prediction rotated)
            y_tilde = y_q_pilot / (alpha + 1e-6)
            r = y_tilde - y_pred

            # Gauss-Newton for τ only (1D problem, no coupling issues)
            # Normal equation: ||J_τ||² × Δτ = Re(J_τ^H × r)
            J_tau_norm_sq = torch.sum(torch.abs(J_tau) ** 2, dim=1, keepdim=True).clamp(min=1e-10)
            grad_tau = torch.real(torch.sum(torch.conj(J_tau) * r, dim=1, keepdim=True))

            # GN step: Δτ = grad / ||J||²
            delta_tau = grad_tau / J_tau_norm_sq

            # Apply damping and clamp
            delta_tau = self.damping * delta_tau
            delta_tau = torch.clamp(delta_tau, -self.max_delta_tau, self.max_delta_tau)

            # Update τ only (v, a unchanged)
            theta_new = theta_current.clone()
            theta_new[:, 0:1] = theta_current[:, 0:1] - delta_tau  # Negative because ∂r/∂θ = -∂y_pred/∂θ

            # Clamp to bounds
            theta_new = torch.clamp(theta_new, self.theta_min, self.theta_max)

            # Compute improvement (with frozen alpha)
            y_pred_new = phys_enc.forward_operator(x_pilot_norm, theta_new)[:, :Np] * phase
            y_tilde_new = y_q_pilot / (alpha + 1e-6)  # Same alpha!
            r_new = y_tilde_new - y_pred_new

            resid_old = torch.mean(torch.abs(r) ** 2, dim=1)
            resid_new = torch.mean(torch.abs(r_new) ** 2, dim=1)
            improvement = (resid_old - resid_new) / (resid_old + 1e-10)
            total_improvement += improvement.mean().item()

            # Update for next iteration
            theta_current = theta_new

        # Prepare diagnostic info
        info = {
            'n_iterations': self.n_iterations,
            'total_improvement': total_improvement,
            'delta_tau_final': delta_tau.abs().mean().item(),
            'tau_change': (theta_current[:, 0] - theta[:, 0]).abs().mean().item(),
            'bussgang_alpha': alpha.mean().item(),
        }

        return theta_current, info


class DopplerTracker(nn.Module):
    """
    Slow-loop Doppler tracking using cross-frame Kalman filtering.

    This is Layer 3 of the hierarchical architecture.
    Uses accumulated τ estimates and phase drift to estimate v, a.

    Note: This is a placeholder for future implementation.
    Currently, v and a are kept at their initial (prior) values.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # EKF state: [v, a]
        # Process noise covariance (learnable in future)
        self.register_buffer('Q', torch.diag(torch.tensor([10.0, 1.0])))  # Process noise
        self.register_buffer('R', torch.diag(torch.tensor([0.01])))  # Measurement noise

        # State estimate (per-batch tracking would need external state management)
        self.v_est = None
        self.a_est = None
        self.P = None  # Covariance

    def reset(self, batch_size: int, device: torch.device, v_init: float = 7500.0, a_init: float = 0.0):
        """Reset tracker state for new sequence."""
        self.v_est = torch.full((batch_size, 1), v_init, device=device)
        self.a_est = torch.full((batch_size, 1), a_init, device=device)
        self.P = torch.eye(2, device=device).unsqueeze(0).expand(batch_size, -1, -1) * 100.0

    def predict(self, dt: float = 1e-7):
        """Predict step: propagate state using kinematic model."""
        if self.v_est is None:
            return

        # State transition: v_new = v + a*dt, a_new = a
        self.v_est = self.v_est + self.a_est * dt

        # Covariance prediction (simplified)
        # P = F @ P @ F.T + Q
        # For now, just add process noise
        self.P = self.P + self.Q.unsqueeze(0)

    def update(self, tau_hat: torch.Tensor, tau_prev: torch.Tensor, dt: float = 1e-7):
        """
        Update step: use τ change to infer velocity.

        Measurement model: Δτ ≈ (2v/c) × dt × fc / fs (in samples)

        This is a simplified update. Full EKF would be more complex.
        """
        if self.v_est is None:
            return

        # Δτ in seconds
        delta_tau = (tau_hat[:, 0:1] - tau_prev[:, 0:1])

        # Inferred velocity from Δτ
        # v = Δτ × c × fs / (2 × fc × dt)
        c = 3e8
        fs = self.cfg.fs
        fc = self.cfg.fc

        v_measured = delta_tau * c * fs / (2 * fc * dt + 1e-10)

        # Simple exponential smoothing (not full EKF for now)
        alpha_smooth = 0.1  # Smoothing factor
        self.v_est = (1 - alpha_smooth) * self.v_est + alpha_smooth * v_measured

    def get_theta(self, tau_hat: torch.Tensor) -> torch.Tensor:
        """Combine τ estimate with v, a estimates."""
        if self.v_est is None:
            return tau_hat

        theta = torch.zeros_like(tau_hat)
        theta[:, 0:1] = tau_hat[:, 0:1]  # τ from TauEstimator
        theta[:, 1:2] = self.v_est  # v from DopplerTracker
        theta[:, 2:3] = self.a_est  # a from DopplerTracker
        return theta