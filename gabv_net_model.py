"""
gabv_net_model.py (Expert Review v6.2 - Critical Boundary Fix)

Description:
    PyTorch Implementation of Geometry-Aware Bussgang-VAMP Network (GA-BV-Net).

    **CRITICAL FIXES APPLIED (Expert Review v6.2):**
    - [FIX 1-8] All previous fixes preserved
    - [FIX 13] ThetaUpdater: Replaced unstable angle stats with physics-based score features
    - [FIX 14] ThetaUpdater: Outputs step sizes (μ) instead of raw delta_theta
    - [FIX 15] ThetaUpdater: Consistent residual domain for acceptance criterion
    - [FIX 16] ThetaUpdater: Log-scale power feature for stability
    - [FIX 17] ThetaUpdater: Force float32 dtype for nn.Linear compatibility
    - [FIX 17] compute_channel_jacobian: Use tensor constant for dtype consistency
    - [FIX 18] **CRITICAL** theta_min/theta_max bounds were 1000x too large!
              Was: [4.9e8, 5.1e8] for R (490,000-510,000 km)
              Now: [1e5, 1e6] for R (100-1000 km)
              This caused theta_hat.R to clamp to 4.9e8 = 490,000,000 m!
    - [NEW] g_theta scheduling support from training loop
    - [NEW] Score-based Gauss-Newton direction features
    - [DEBUG] Temporary debug outputs for verification (remove in production)

    Meta Feature Schema (FROZEN - must match train_gabv_net.py):
        meta[:, 0] = snr_db_norm      = (snr_db - 15) / 15
        meta[:, 1] = gamma_eff_db_norm = (10*log10(gamma_eff) - 10) / 20
        meta[:, 2] = chi              (raw, range [0, 2/π])
        meta[:, 3] = sigma_eta_norm   = sigma_eta / 0.1
        meta[:, 4] = pn_linewidth_norm = log10(pn_linewidth + 1) / log10(1e6)
        meta[:, 5] = ibo_db_norm      = (ibo_dB - 3) / 3

Author: Expert Review Fixed v6.2 (Critical Boundary Fix)
Date: 2025-12-19
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

# --- 1. Configuration ---

@dataclass
class GABVConfig:
    """Hyperparameters for GA-BV-Net."""
    n_layers: int = 8
    hidden_dim_pilot: int = 64
    hidden_dim_pn: int = 64
    cg_steps: int = 5
    use_bidirectional_pn: bool = True

    # [FIX 1] fs MUST match SimConfig.fs = 10e9
    N_sym: int = 1024
    fs: float = 10e9   # FIXED: was 25e9, now matches SimConfig
    fc: float = 300e9

    share_weights: bool = True
    enable_geom_loss: bool = True
    block_size_fim: int = 32

    # [FIX 4] Enable theta update
    enable_theta_update: bool = True
    theta_update_start_layer: int = 1  # Start updating from layer 1


# --- Meta Feature Normalization Constants (FROZEN) ---

META_CONSTANTS = {
    'snr_db_center': 15.0,
    'snr_db_scale': 15.0,
    'gamma_eff_db_center': 10.0,
    'gamma_eff_db_scale': 20.0,
    'sigma_eta_scale': 0.1,
    'pn_linewidth_scale': 1e6,
    'ibo_db_center': 3.0,
    'ibo_db_scale': 3.0,
}


# --- 2. Sub-Modules ---

class PilotNavigator(nn.Module):
    def __init__(self, cfg: GABVConfig, feature_dim: int = 6):
        super().__init__()
        h_dim = cfg.hidden_dim_pilot

        self.net = nn.Sequential(
            nn.Linear(feature_dim, h_dim),
            nn.LayerNorm(h_dim),
            nn.ReLU(),
            nn.Linear(h_dim, h_dim // 2),
            nn.ReLU(),
            nn.Linear(h_dim // 2, 4)  # Added g_theta output
        )
        self.register_buffer('brake_threshold', torch.tensor(-15.0))

    def forward(self, meta_features: torch.Tensor, log_vol: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logits = self.net(meta_features)

        g_PN = torch.sigmoid(logits[:, 0:1])
        g_NL = F.softplus(logits[:, 1:2]) + 0.5
        learned_brake = torch.sigmoid(logits[:, 2:3])
        g_theta = torch.sigmoid(logits[:, 3:4])  # [FIX 4] Gate for theta update

        if log_vol is not None:
            physics_brake = torch.sigmoid((log_vol - self.brake_threshold) * 5.0)
            g_grad = learned_brake * physics_brake
        else:
            g_grad = learned_brake

        return {"g_PN": g_PN, "g_NL": g_NL, "g_grad": g_grad, "g_theta": g_theta}


class PhysicsEncoder(nn.Module):
    """
    [FIX 2 & 3 & 7] Physics Encoder - NOW MATCHES thz_isac_world.apply_channel_squint EXACTLY

    Channel model (SINGLE-TRIP, matches simulator):
        tau = R / c  (NOT 2R/c!)
        tau_dot = v / c
        tau_ddot = a / c
        phase = 2π * fc * (tau + tau_dot * t + 0.5 * tau_ddot * t²)
        h[n] = exp(-j * phase[n])

    [FIX 7] CRITICAL: Phase calculation uses Float64 to avoid precision loss!
        At R=500km, fc=300GHz: phase ≈ 3e9 rad (500M cycles)
        Float32 precision: ~7 digits → ~300 rad error → BER ≈ 0.35
        Float64 precision: ~15 digits → negligible error → BER ≈ 0.00
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.N = cfg.N_sym
        self.fs = cfg.fs  # Now 10e9, matches SimConfig
        self.fc = cfg.fc
        self.c = 3e8

        # [FIX 7] Time grid in Float64 for precision!
        t_grid = torch.arange(self.N, dtype=torch.float64) / self.fs
        self.register_buffer('t_grid', t_grid)

        # Frequency grid (for potential wideband extensions)
        f_grid = torch.fft.fftfreq(self.N, d=1/self.fs)
        self.register_buffer('f_grid', f_grid)

        # Regularization network for FIM
        self.reg_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus()
        )

        # [FIX 6] Learnable Bussgang gain compensation
        self.bussgang_gain = nn.Parameter(torch.tensor(1.0))

    def compute_channel_diag(self, theta: torch.Tensor) -> torch.Tensor:
        """
        Compute diagonal channel response h_diag.

        EXACTLY matches thz_isac_world.apply_channel_squint:
            tau = R / c  (SINGLE-TRIP!)
            phase = 2π * fc * (tau + (v/c)*t + 0.5*(a/c)*t²)
            h = exp(-j * phase)

        [FIX 7] Uses Float64 internally for phase calculation to avoid
        catastrophic precision loss at large R values!

        Args:
            theta: [B, 3] parameters (R, v, a)

        Returns:
            h_diag: [B, N] diagonal channel (complex64 for efficiency)
        """
        # [FIX 7] Convert to Float64 for precise phase calculation
        theta_f64 = theta.to(torch.float64)

        R = theta_f64[:, 0:1]  # [B, 1]
        v = theta_f64[:, 1:2]  # [B, 1]
        a = theta_f64[:, 2:3]  # [B, 1]

        t = self.t_grid.unsqueeze(0)  # [1, N], already float64

        # [FIX 2] SINGLE-TRIP delay parameters (matches simulator!)
        tau = R / self.c           # τ = R/c (NOT 2R/c!)
        tau_dot = v / self.c       # τ̇ = v/c
        tau_ddot = a / self.c      # τ̈ = a/c

        # Phase calculation in Float64 - EXACTLY matches simulator
        phase = 2 * math.pi * self.fc * (
            tau + tau_dot * t + 0.5 * tau_ddot * (t ** 2)
        )  # [B, N], float64

        # Compute h in complex128, then convert to complex64 for efficiency
        h_diag_f64 = torch.exp(-1j * phase)
        h_diag = h_diag_f64.to(torch.cfloat)  # complex64

        return h_diag

    def compute_channel_jacobian(self, theta: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        [NEW] Compute channel Jacobian ∂h/∂θ for physics-based features.

        Returns derivatives with respect to R, v, a:
            ∂h/∂R = h * (-j * 2π * fc / c)
            ∂h/∂v = h * (-j * 2π * fc / c * t)
            ∂h/∂a = h * (-j * 2π * fc / c * 0.5 * t²)

        Args:
            theta: [B, 3] parameters (R, v, a)

        Returns:
            dh_dR: [B, N] complex
            dh_dv: [B, N] complex
            dh_da: [B, N] complex
        """
        h_diag = self.compute_channel_diag(theta)  # [B, N]
        t = self.t_grid.unsqueeze(0).to(theta.device)  # [1, N]

        # Common factor: -j * 2π * fc / c
        # [FIX 17] Use tensor constant to maintain dtype consistency
        factor = torch.tensor(-1j * 2 * 3.141592653589793 * self.fc / self.c,
                              dtype=h_diag.dtype, device=theta.device)

        # Jacobians
        dh_dR = h_diag * factor  # [B, N]
        dh_dv = h_diag * (factor * t)  # [B, N]
        dh_da = h_diag * (factor * 0.5 * t ** 2)  # [B, N]

        return dh_dR, dh_dv, dh_da

    def forward_operator(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Forward operator: y = H @ x (diagonal channel = element-wise multiply)"""
        h_diag = self.compute_channel_diag(theta)
        # Apply Bussgang gain compensation
        h_eff = h_diag * self.bussgang_gain
        return x * h_eff

    def adjoint_operator(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        """Adjoint operator: x = H^H @ y"""
        h_diag = self.compute_channel_diag(theta)
        h_eff = h_diag * self.bussgang_gain
        return y * torch.conj(h_eff)

    def get_approx_fim_info(self, theta: torch.Tensor, gamma_eff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Approximate FIM information for natural gradient."""
        reg = self.reg_net(gamma_eff) + 1e-6
        log_vol = -torch.log(reg) * 3
        fim_inv_diag = torch.ones_like(theta) * reg
        return fim_inv_diag, log_vol


class RiemannianPNTracker(nn.Module):
    """
    [B-lite] Phase Noise Tracker with Wiener Prior Regularization

    Key Features:
    - Explicit (cos, sin) output for numerical stability
    - [FIX 8] Normalized interpolation to preserve unit magnitude
    - [FIX 12] Learnable phase_bias for phase ambiguity resolution
    - [NEW B-lite] Wiener prior loss: penalizes large phase jumps

    The Wiener prior enforces: φ[n] ≈ φ[n-1]
    This aligns with BCRLB theory and improves phase tracking.
    """

    def __init__(self, cfg):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=2,
            hidden_size=cfg.hidden_dim_pn,
            bidirectional=cfg.use_bidirectional_pn,
            batch_first=True
        )
        out_dim = cfg.hidden_dim_pn * 2 if cfg.use_bidirectional_pn else cfg.hidden_dim_pn

        # Output (cos, sin) for stability
        self.head = nn.Sequential(
            nn.Linear(out_dim, cfg.hidden_dim_pn // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim_pn // 2, 2)  # (cos, sin)
        )

        # [FIX 12] Learnable phase bias for phase ambiguity
        self.phase_bias = nn.Parameter(torch.tensor(-1.5708))  # Initialize to -90°

        # [NEW B-lite] Wiener prior strength (learnable)
        # Higher value = smoother phase trajectory = stronger regularization
        self.wiener_prior_strength = nn.Parameter(torch.tensor(1.0))

    def forward(self, y_q: torch.Tensor, g_PN: torch.Tensor,
                x_est: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Estimate phase noise and return de-rotation factor + Wiener prior loss.

        Args:
            y_q: Quantized observation [B, N]
            g_PN: PN gate [B, 1]
            x_est: Optional symbol estimate (used if reliable)

        Returns:
            derotator: exp(-jφ) de-rotation factor [B, N]
            wiener_loss: Wiener prior regularization term (scalar)
        """
        B, N = y_q.shape
        device = y_q.device

        # Use residual if x_est is reliable, else use y_q directly
        if x_est is not None and x_est.abs().mean() > 0.1:
            # Normalize x_est to unit magnitude for phase extraction
            x_norm = x_est / (x_est.abs() + 1e-6)
            resid = y_q * torch.conj(x_norm)
            feats = torch.stack([resid.real, resid.imag], dim=-1)
        else:
            # Early layers: use y_q directly
            feats = torch.stack([y_q.real, y_q.imag], dim=-1)  # [B, N, 2]

        rnn_out, _ = self.rnn(feats)  # [B, N, hidden*2]

        # Output (cos, sin)
        cos_sin = self.head(rnn_out)  # [B, N, 2]
        cos_phi = cos_sin[..., 0]
        sin_phi = cos_sin[..., 1]

        # Normalize to unit circle
        mag = torch.sqrt(cos_phi ** 2 + sin_phi ** 2 + 1e-8)
        cos_phi = cos_phi / mag
        sin_phi = sin_phi / mag

        # Compute phase for Wiener prior loss
        phi_hat = torch.atan2(sin_phi, cos_phi)  # [B, N]

        # [NEW B-lite] Wiener prior loss: penalize large phase jumps
        # L_wiener = E[(φ[n] - φ[n-1])^2] * strength
        # This is the "prior FIM" structure from BCRLB theory
        if N > 1:
            phi_diff = phi_hat[:, 1:] - phi_hat[:, :-1]  # [B, N-1]
            # Wrap phase difference to [-π, π] to handle wrapping
            phi_diff = torch.atan2(torch.sin(phi_diff), torch.cos(phi_diff))
            # Mean squared phase difference weighted by learnable strength
            wiener_loss = torch.mean(phi_diff ** 2) * torch.abs(self.wiener_prior_strength)
        else:
            wiener_loss = torch.tensor(0.0, device=device)

        # De-rotation factor: exp(-jφ) = cos(φ) - j*sin(φ)
        derotator = torch.complex(cos_phi, -sin_phi)  # [B, N]

        # [FIX 12] Apply learnable phase bias
        bias_correction = torch.exp(-1j * self.phase_bias)
        derotator = derotator * bias_correction

        # Apply gate: g_PN=0 means no de-rotation
        # [FIX 8] CRITICAL: Must normalize after linear interpolation!
        # Linear interpolation of complex numbers doesn't preserve unit magnitude
        eff_derotator = (1 - g_PN) * torch.ones_like(derotator) + g_PN * derotator
        eff_derotator = eff_derotator / (eff_derotator.abs() + 1e-8)  # Normalize!

        return eff_derotator, wiener_loss


class BussgangRefiner(nn.Module):
    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.scale_slope = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_in: torch.Tensor, g_NL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        beta = self.scale_slope * g_NL

        z_out_real = torch.tanh(beta * z_in.real)
        z_out_imag = torch.tanh(beta * z_in.imag)
        z_out = torch.complex(z_out_real, z_out_imag)

        d_real = beta * (1 - z_out_real**2)
        d_imag = beta * (1 - z_out_imag**2)

        onsager = torch.mean(d_real + d_imag, dim=1) * 0.5

        return z_out, onsager


class UnrolledCGSolver(nn.Module):
    """CG Solver for diagonal channel (optimized)."""

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.steps = cfg.cg_steps
        self.step_scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, y: torch.Tensor, x_init: torch.Tensor,
                phys_enc: 'PhysicsEncoder', theta: torch.Tensor,
                snr_linear: torch.Tensor) -> torch.Tensor:
        """
        CG solver for: min_x ||y - Hx||² + λ||x||²

        For diagonal H, this has closed-form solution:
            x = H^H y / (|H|² + λ)
        """
        # Get channel
        h_diag = phys_enc.compute_channel_diag(theta)
        h_eff = h_diag * phys_enc.bussgang_gain

        # Regularization
        lambda_reg = 1.0 / (snr_linear + 1e-6)  # [B, 1]

        # For diagonal system, use efficient closed-form
        h_sq = h_eff.abs() ** 2  # [B, N]
        rhs = torch.conj(h_eff) * y  # H^H y

        # x = H^H y / (|H|² + λ)
        x = rhs / (h_sq + lambda_reg)

        return x


class ThetaUpdater(nn.Module):
    """
    [Expert v6.0] Physics-Aware Theta Updater with Score-Based Features

    Key Changes from v5:
    - [FIX 13] Replaced unstable angle(resid) stats with physics-based score features
    - [FIX 14] Network outputs step sizes (μ) instead of raw delta_theta
    - [FIX 15] Consistent residual domain for acceptance criterion
    - [FIX 16] Log-scale power feature for numerical stability

    The score features are derived from the Gauss-Newton direction:
        s_k = Re(⟨∂(h·x)/∂θ_k, r⟩) / ||∂(h·x)/∂θ_k||²

    This is physics-consistent and much more stable than learning raw deltas.
    """

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # [FIX 14] Network outputs step sizes (μ), not raw delta_theta
        # Input: [log_power, |s_R|, |s_v|, |s_a|, g_theta, confidence] = 6 features
        self.step_net = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 3),  # Output: step sizes for (R, v, a)
            nn.Softplus()  # Ensure non-negative step sizes
        )

        # Base step size scaling (learnable)
        self.base_step_scale = nn.Parameter(torch.tensor([1.0, 1.0, 1.0]))

        # Physical limits [FIX 9]
        self.max_delta_R = 20.0  # meters per layer
        self.max_delta_v = 5.0   # m/s per layer
        self.max_delta_a = 1.0   # m/s^2 per layer

        # Confidence threshold [FIX 9]
        self.confidence_threshold = 0.5

        # Parameter bounds [FIX 18] CORRECTED - was 1000x too large!
        # R typical: 500 km = 5e5 m, range: [100 km, 1000 km]
        # v typical: 7500 m/s (satellite), range: [-10 km/s, +10 km/s]
        # a typical: 10 m/s², range: [-100, +100]
        self.register_buffer('theta_min', torch.tensor([1e5, -1e4, -100.0]))
        self.register_buffer('theta_max', torch.tensor([1e6, 1e4, 100.0]))

        # [NEW B-lite] Enable acceptance criterion
        self.use_acceptance_criterion = True

    def compute_score_features(self, resid: torch.Tensor, x_est: torch.Tensor,
                               phys_enc: 'PhysicsEncoder', theta: torch.Tensor) -> torch.Tensor:
        """
        [FIX 13] Compute physics-based score features (Gauss-Newton direction).

        Score for parameter k:
            s_k = Re(⟨∂y_pred/∂θ_k, r⟩) / ||∂y_pred/∂θ_k||²

        where y_pred = h * x_est, r = residual

        This gives the steepest descent direction in the θ space.

        Args:
            resid: Residual signal [B, N]
            x_est: Current symbol estimate [B, N]
            phys_enc: Physics encoder (for Jacobian computation)
            theta: Current parameters [B, 3]

        Returns:
            scores: [B, 3] score features (s_R, s_v, s_a)
        """
        # Get channel Jacobians: ∂h/∂R, ∂h/∂v, ∂h/∂a
        dh_dR, dh_dv, dh_da = phys_enc.compute_channel_jacobian(theta)

        # [DEBUG] Print Jacobian stats
        if hasattr(self, '_debug_counter') and self._debug_counter < 3:
            print(f"  [compute_score_features DEBUG]")
            print(f"    dh_dR: mean_abs={dh_dR.abs().mean().item():.2f}, max_abs={dh_dR.abs().max().item():.2f}")

        # Compute ∂y_pred/∂θ = (∂h/∂θ) * x_est
        dy_dR = dh_dR * x_est  # [B, N]
        dy_dv = dh_dv * x_est  # [B, N]
        dy_da = dh_da * x_est  # [B, N]

        # [DEBUG] Print dy stats
        if hasattr(self, '_debug_counter') and self._debug_counter < 3:
            print(f"    dy_dR: mean_abs={dy_dR.abs().mean().item():.2f}, max_abs={dy_dR.abs().max().item():.2f}")

        # Score: s_k = Re(⟨∂y/∂θ_k, r⟩) / ||∂y/∂θ_k||²
        def compute_score(dy_dtheta, r):
            # Inner product: ⟨∂y/∂θ_k, r⟩
            inner = torch.sum(torch.conj(dy_dtheta) * r, dim=1)  # [B]
            inner_real = inner.real

            # Norm squared: ||∂y/∂θ_k||²
            norm_sq = torch.sum(dy_dtheta.abs() ** 2, dim=1) + 1e-12  # [B]

            # Score
            score = inner_real / norm_sq  # [B]
            return score

        s_R = compute_score(dy_dR, resid)  # [B]
        s_v = compute_score(dy_dv, resid)  # [B]
        s_a = compute_score(dy_da, resid)  # [B]

        # Stack into [B, 3]
        scores = torch.stack([s_R, s_v, s_a], dim=1)

        return scores

    def forward(self, theta: torch.Tensor, resid: torch.Tensor, x_est: torch.Tensor,
                g_theta: torch.Tensor, fim_inv: Optional[torch.Tensor] = None,
                phys_enc: Optional['PhysicsEncoder'] = None,
                z_denoised: Optional[torch.Tensor] = None,
                g_theta_sched: float = 1.0) -> Tuple[torch.Tensor, Dict]:
        """
        Update theta with physics-based score features.

        Args:
            theta: Current parameters [B, 3] (R, v, a)
            resid: Residual signal [B, N] (in z_denoised domain)
            x_est: Current symbol estimate [B, N]
            g_theta: Update gate from PilotNavigator [B, 1]
            fim_inv: Optional FIM inverse for natural gradient [B, 3]
            phys_enc: Physics encoder (required for score computation)
            z_denoised: Denoised observation (for acceptance criterion)
            g_theta_sched: Scheduling factor from training loop [0, 1]

        Returns:
            theta_new: Updated parameters [B, 3]
            info: Debug information dict
        """
        B = theta.shape[0]
        device = theta.device

        # [DEBUG] Print input stats
        if not hasattr(self, '_debug_counter'):
            self._debug_counter = 0

        if self._debug_counter < 5:
            print(f"\n[ThetaUpdater DEBUG {self._debug_counter}] Input stats:")
            print(f"  resid: mean={resid.abs().mean().item():.6f}, max={resid.abs().max().item():.6f}")
            print(f"  x_est: mean={x_est.abs().mean().item():.6f}, max={x_est.abs().max().item():.6f}")

        # [FIX 16] Log-scale power feature for stability
        resid_power = torch.mean(resid.abs() ** 2, dim=1, keepdim=True)  # [B, 1]
        log_power = torch.log10(resid_power + 1e-9)  # Log scale for stability

        # [FIX 13] Compute physics-based score features
        if phys_enc is not None:
            scores = self.compute_score_features(resid, x_est, phys_enc, theta)  # [B, 3]
            # Use absolute values as features (direction comes from sign)
            score_magnitudes = scores.abs()  # [B, 3]
        else:
            # Fallback: use zero scores (no physics-based direction)
            scores = torch.zeros(B, 3, device=device)
            score_magnitudes = torch.zeros(B, 3, device=device)

        # Confidence based on x_est quality
        confidence = torch.mean(x_est.abs(), dim=1, keepdim=True)  # [B, 1]

        # [FIX 14] Feature vector for step size network
        feat = torch.cat([
            log_power,                    # [B, 1] - log scale power
            score_magnitudes,             # [B, 3] - score magnitudes
            g_theta,                      # [B, 1] - gate from pilot
            confidence,                   # [B, 1] - symbol confidence
        ], dim=1).float()  # [B, 6], [FIX 17] Force float32 for nn.Linear compatibility

        # [DEBUG] Print intermediate values for first batch element
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0

        if self._debug_counter < 5:
            print(f"\n[ThetaUpdater DEBUG {self._debug_counter}]")
            print(f"  theta_in[0]: R={theta[0, 0].item():.1f}, v={theta[0, 1].item():.2f}, a={theta[0, 2].item():.4f}")
            print(f"  scores[0]: R={scores[0, 0].item():.6e}, v={scores[0, 1].item():.6e}, a={scores[0, 2].item():.6e}")
            print(f"  score_magnitudes[0]: {score_magnitudes[0].tolist()}")
            print(f"  log_power[0]: {log_power[0].item():.4f}")
            print(f"  g_theta[0]: {g_theta[0].item():.4f}")
            print(f"  confidence[0]: {confidence[0].item():.4f}")
            print(f"  feat[0]: {feat[0].tolist()}")

        # [FIX 14] Predict step sizes (μ), not raw delta_theta
        step_sizes = self.step_net(feat)  # [B, 3], non-negative via Softplus

        # [DEBUG] Continue debug output
        if self._debug_counter < 5:
            print(f"  step_sizes[0]: {step_sizes[0].tolist()}")

        # Scale by base step size
        step_sizes = step_sizes * torch.abs(self.base_step_scale)

        # [FIX 14] Compute update direction from scores: Δθ = -μ ⊙ s
        # Negative sign: we want to minimize residual, scores point uphill
        delta_theta = -step_sizes * scores  # [B, 3]

        # Physical clamping per component
        delta_R = torch.clamp(delta_theta[:, 0:1], -self.max_delta_R, self.max_delta_R)
        delta_v = torch.clamp(delta_theta[:, 1:2], -self.max_delta_v, self.max_delta_v)
        delta_a = torch.clamp(delta_theta[:, 2:3], -self.max_delta_a, self.max_delta_a)
        delta_theta_clipped = torch.cat([delta_R, delta_v, delta_a], dim=1)

        # [DEBUG] Continue debug output
        if self._debug_counter < 5:
            print(f"  delta_theta[0] (before clamp): R={delta_theta[0, 0].item():.6e}, v={delta_theta[0, 1].item():.6e}, a={delta_theta[0, 2].item():.6e}")
            print(f"  delta_theta_clipped[0]: R={delta_R[0].item():.4f}, v={delta_v[0].item():.4f}, a={delta_a[0].item():.6f}")

        # Confidence gating
        confidence_gate = torch.sigmoid((confidence - self.confidence_threshold) * 10)

        # [NEW] Apply g_theta scheduling from training loop
        g_theta_scheduled = g_theta * g_theta_sched

        # Combined gate
        effective_gate = g_theta_scheduled * confidence_gate

        # Compute candidate theta
        theta_candidate = theta + effective_gate * delta_theta_clipped

        # [DEBUG] Final debug output
        if self._debug_counter < 5:
            print(f"  effective_gate[0]: {effective_gate[0].item():.4f}")
            print(f"  theta_candidate[0]: R={theta_candidate[0, 0].item():.1f}, v={theta_candidate[0, 1].item():.2f}, a={theta_candidate[0, 2].item():.4f}")

        # [FIX 15] Acceptance criterion with CONSISTENT residual domain
        accept_rate = 1.0
        if self.use_acceptance_criterion and z_denoised is not None and phys_enc is not None:
            with torch.no_grad():
                # Current residual norm (already computed in resid)
                resid_norm_before = torch.mean(resid.abs() ** 2, dim=1)  # [B]

                # Compute new channel and residual with candidate theta
                # [FIX 15] Use z_denoised domain (same as resid), NOT y_q
                h_new = phys_enc.compute_channel_diag(theta_candidate)
                y_reconstructed_new = h_new * x_est
                resid_new = z_denoised - y_reconstructed_new
                resid_norm_after = torch.mean(resid_new.abs() ** 2, dim=1)  # [B]

                # Accept only if residual decreased (with small tolerance)
                accept_mask = (resid_norm_after < resid_norm_before * 1.01).float()  # [B]
                accept_rate = accept_mask.mean().item()

                # Apply acceptance: use candidate if accepted, else keep original
                accept_mask = accept_mask.unsqueeze(-1)  # [B, 1]
                theta_new = accept_mask * theta_candidate + (1 - accept_mask) * theta
        else:
            theta_new = theta_candidate

        # Final clamping to reasonable range
        theta_new = torch.clamp(theta_new, self.theta_min, self.theta_max)

        # Debug info
        info = {
            'delta_R': delta_R.mean().item(),
            'delta_v': delta_v.mean().item(),
            'delta_a': delta_a.mean().item(),
            'g_theta_effective': effective_gate.mean().item(),
            'confidence': confidence.mean().item(),
            'accept_rate': accept_rate,
            'score_R': scores[:, 0].mean().item() if scores is not None else 0.0,
            'score_v': scores[:, 1].mean().item() if scores is not None else 0.0,
            'score_a': scores[:, 2].mean().item() if scores is not None else 0.0,
            'step_size_R': step_sizes[:, 0].mean().item(),
            'step_size_v': step_sizes[:, 1].mean().item(),
            'step_size_a': step_sizes[:, 2].mean().item(),
        }

        return theta_new, info


class FisherUpdater(nn.Module):
    """Legacy updater - kept for compatibility."""
    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.lr = nn.Parameter(torch.tensor(0.01))

    def forward(self, theta: torch.Tensor, task_grad: Optional[torch.Tensor],
                fim_inv: torch.Tensor, g_grad: torch.Tensor) -> torch.Tensor:
        if task_grad is None:
            return theta
        nat_grad = fim_inv * task_grad
        delta = self.lr * g_grad * nat_grad
        return theta - delta


# --- 3. Top-Level Model ---

class GABVNet(nn.Module):
    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.cfg = cfg

        if cfg.share_weights:
            self.pilot = PilotNavigator(cfg)
            self.phys_enc = PhysicsEncoder(cfg)
            self.pn_tracker = RiemannianPNTracker(cfg)  # B-lite version
            self.refiner = BussgangRefiner(cfg)
            self.solver = UnrolledCGSolver(cfg)
            self.theta_updater = ThetaUpdater(cfg)  # v6.0 physics-based version
        else:
            raise NotImplementedError("Per-layer weights not yet implemented.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GA-BV-Net with B-lite enhancements.

        B-lite Features:
        - Accumulates wiener_loss from PN tracker across layers
        - ThetaUpdater with physics-based score features
        - Full support for FIX 7-16
        - g_theta scheduling support

        Args:
            batch: Dictionary containing:
                - 'y_q': Quantized observation [B, N]
                - 'meta': Meta features [B, 6]
                - 'theta_init': Initial theta estimate [B, 3]
                - 'g_theta_sched': (optional) Scheduling factor [0, 1]

        Returns:
            Dictionary containing:
                - 'x_hat': Symbol estimate [B, N]
                - 'theta_hat': Final theta estimate [B, 3]
                - 'phi_hat': Final de-rotation factor [B, N]
                - 'layers': Per-layer outputs (with theta_info)
                - 'geom_cache': FIM information
                - 'wiener_loss': Total Wiener prior loss (for training)
        """
        y_q = batch['y_q']
        meta = batch['meta']
        theta = batch['theta_init'].clone()

        # [NEW] g_theta scheduling from training loop
        g_theta_sched = batch.get('g_theta_sched', 1.0)

        B, N = y_q.shape
        device = y_q.device

        # Denormalize meta features
        snr_db_norm = meta[:, 0:1]
        gamma_eff_db_norm = meta[:, 1:2]

        snr_db = snr_db_norm * META_CONSTANTS['snr_db_scale'] + META_CONSTANTS['snr_db_center']
        gamma_eff_db = gamma_eff_db_norm * META_CONSTANTS['gamma_eff_db_scale'] + META_CONSTANTS['gamma_eff_db_center']

        snr_linear = 10.0 ** (snr_db / 10.0)
        gamma_eff_linear = 10.0 ** (gamma_eff_db / 10.0)

        # Initialize
        x_est = torch.zeros_like(y_q)

        layer_outputs = []
        geom_cache = {'log_vols': [], 'fim_invs': []}

        # [B-lite] Accumulate Wiener prior loss across layers
        total_wiener_loss = torch.tensor(0.0, device=device, dtype=torch.float32)

        for k in range(self.cfg.n_layers):
            # 1. Physics - get FIM info
            fim_inv, log_vol = self.phys_enc.get_approx_fim_info(theta, gamma_eff_linear)
            if self.cfg.enable_geom_loss:
                geom_cache['log_vols'].append(log_vol)
                geom_cache['fim_invs'].append(fim_inv)

            # 2. Pilot - get gates
            gates = self.pilot(meta, log_vol=log_vol)

            # [FIX 10] Adjust g_PN based on pn_linewidth_norm in meta
            # When pn_linewidth_norm < 0.1, there's no PN, so g_PN should be 0
            pn_norm = meta[:, 4:5]  # pn_linewidth_norm is at index 4
            gates['g_PN'] = gates['g_PN'] * (pn_norm > 0.1).float()

            # 3. PN Tracking - [B-lite] Now returns (derotator, wiener_loss)
            pn_output = self.pn_tracker(y_q, gates['g_PN'], x_est if k > 0 else None)

            # Handle both old (single return) and new (tuple return) versions
            if isinstance(pn_output, tuple):
                derotator, wiener_loss = pn_output
                total_wiener_loss = total_wiener_loss + wiener_loss
            else:
                derotator = pn_output
                wiener_loss = torch.tensor(0.0, device=device)

            y_corr = y_q * derotator

            # 4. Refiner
            z_denoised, onsager = self.refiner(y_corr, gates['g_NL'])

            # 5. Solver
            x_est = self.solver(z_denoised, x_est, self.phys_enc, theta, snr_linear)

            # 6. [FIX 4] Theta Update with physics-based features
            theta_info = {
                'delta_R': 0.0,
                'delta_v': 0.0,
                'delta_a': 0.0,
                'g_theta_effective': gates['g_theta'].mean().item(),
                'accept_rate': 1.0,
            }

            if self.cfg.enable_theta_update and k >= self.cfg.theta_update_start_layer:
                # [DEBUG] Print theta before update
                if k == self.cfg.theta_update_start_layer:
                    print(f"\n[GABVNet DEBUG] Layer {k}, theta BEFORE update[0]: R={theta[0, 0].item():.1f}")

                # Compute channel and residual
                h_diag = self.phys_enc.compute_channel_diag(theta)
                resid = z_denoised - h_diag * x_est

                # [FIX 15] Update theta with consistent residual domain
                theta, theta_info = self.theta_updater(
                    theta, resid, x_est,
                    gates['g_theta'], fim_inv,
                    phys_enc=self.phys_enc,
                    z_denoised=z_denoised,  # Pass for acceptance criterion
                    g_theta_sched=g_theta_sched,  # Pass scheduling factor
                )

                # [DEBUG] Print theta after update
                if k == self.cfg.theta_update_start_layer:
                    print(f"[GABVNet DEBUG] Layer {k}, theta AFTER update[0]: R={theta[0, 0].item():.1f}")

            layer_outputs.append({
                'x_est': x_est.clone(),
                'theta': theta.clone(),
                'gates': gates,
                'derotator': derotator,
                'theta_info': theta_info,  # [B-lite] Debug info
            })

        return {
            'x_hat': x_est,
            'theta_hat': theta,
            'phi_hat': layer_outputs[-1]['derotator'],
            'layers': layer_outputs,
            'geom_cache': geom_cache,
            'wiener_loss': total_wiener_loss,  # [B-lite] For training
        }


def create_gabv_model(cfg: Optional[GABVConfig] = None) -> GABVNet:
    """Factory function to create GA-BV-Net model."""
    if cfg is None:
        cfg = GABVConfig()
    return GABVNet(cfg)


# --- 4. Utility Functions ---

def denormalize_meta(meta: torch.Tensor) -> Dict[str, torch.Tensor]:
    """Utility function to denormalize meta features."""
    snr_db_norm = meta[:, 0:1]
    gamma_eff_db_norm = meta[:, 1:2]
    chi = meta[:, 2:3]
    sigma_eta_norm = meta[:, 3:4]
    pn_linewidth_norm = meta[:, 4:5]
    ibo_db_norm = meta[:, 5:6]

    snr_db = snr_db_norm * META_CONSTANTS['snr_db_scale'] + META_CONSTANTS['snr_db_center']
    gamma_eff_db = gamma_eff_db_norm * META_CONSTANTS['gamma_eff_db_scale'] + META_CONSTANTS['gamma_eff_db_center']
    sigma_eta = sigma_eta_norm * META_CONSTANTS['sigma_eta_scale']
    pn_linewidth = 10.0 ** (pn_linewidth_norm * math.log10(META_CONSTANTS['pn_linewidth_scale'])) - 1.0
    ibo_db = ibo_db_norm * META_CONSTANTS['ibo_db_scale'] + META_CONSTANTS['ibo_db_center']

    return {
        'snr_db': snr_db,
        'snr_linear': 10.0 ** (snr_db / 10.0),
        'gamma_eff_db': gamma_eff_db,
        'gamma_eff_linear': 10.0 ** (gamma_eff_db / 10.0),
        'chi': chi,
        'sigma_eta': sigma_eta,
        'pn_linewidth': pn_linewidth,
        'ibo_db': ibo_db,
    }


# --- 5. Verification ---

if __name__ == "__main__":
    print("=" * 60)
    print("GABVNet Model - Expert Fixed Version v6.0 Self-Test")
    print("=" * 60)

    # Check fs consistency
    cfg = GABVConfig()
    print(f"\n[Config Check]")
    print(f"  fs = {cfg.fs/1e9:.1f} GHz (should be 10.0)")
    print(f"  fc = {cfg.fc/1e9:.1f} GHz")
    print(f"  N  = {cfg.N_sym}")
    print(f"  enable_theta_update = {cfg.enable_theta_update}")

    # Test model creation
    print(f"\n[Model Creation Test]")
    model = create_gabv_model(cfg)
    print(f"  Model created successfully")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    theta_params = sum(p.numel() for n, p in model.named_parameters() if 'theta_updater' in n)
    print(f"  Total parameters: {total_params:,}")
    print(f"  ThetaUpdater parameters: {theta_params:,}")

    # Test forward pass
    print(f"\n[Forward Pass Test]")
    B, N = 4, 1024
    device = 'cpu'

    batch = {
        'y_q': torch.randn(B, N, dtype=torch.cfloat, device=device),
        'meta': torch.randn(B, 6, device=device),
        'theta_init': torch.tensor([[5e5, 7500.0, 10.0]] * B, device=device),
        'g_theta_sched': 0.5,
    }

    with torch.no_grad():
        outputs = model(batch)

    print(f"  x_hat shape: {outputs['x_hat'].shape}")
    print(f"  theta_hat shape: {outputs['theta_hat'].shape}")
    print(f"  wiener_loss: {outputs['wiener_loss'].item():.4f}")

    # Check theta_info
    last_layer = outputs['layers'][-1]
    theta_info = last_layer['theta_info']
    print(f"  theta_info keys: {list(theta_info.keys())}")
    print(f"  delta_R: {theta_info['delta_R']:.4f}")
    print(f"  accept_rate: {theta_info['accept_rate']:.4f}")

    print("\n" + "=" * 60)
    print("Self-Test Complete")
    print("=" * 60)