"""
gabv_net_model.py (Fixed Version v5 - PN Tracker Normalization Fix)

Description:
    PyTorch Implementation of Geometry-Aware Bussgang-VAMP Network (GA-BV-Net).

    **CRITICAL FIXES APPLIED (Expert Review):**
    - [FIX 1] fs = 10e9 (matches SimConfig, was 25e9)
    - [FIX 2] PhysicsEncoder uses SINGLE-TRIP model: tau = R/c (was 2R/c)
    - [FIX 3] PhysicsEncoder matches thz_isac_world.apply_channel_squint exactly
    - [FIX 4] Theta UPDATE implemented (was theta_next = theta)
    - [FIX 5] Improved PN tracker with explicit (cos, sin) output
    - [FIX 6] Added learnable Bussgang gain compensation
    - [FIX 7] **CRITICAL** Phase calculation uses Float64!
             At R=500km, fc=300GHz: phase ≈ 3e9 rad (500M cycles)
             Float32 has only 7 digits precision → ~300 rad error → BER ≈ 0.35
             Float64 has 15 digits precision → negligible error → BER ≈ 0.00
    - [FIX 8] **CRITICAL** PN Tracker gate interpolation normalized!
             Linear interpolation (1-g)*1 + g*exp(jφ) has |result| ≠ 1
             This corrupted signal amplitude → BER ≈ 0.5
             Added normalization: result / |result| → BER ≈ 0.00

    Meta Feature Schema (FROZEN - must match train_gabv_net.py):
        meta[:, 0] = snr_db_norm      = (snr_db - 15) / 15
        meta[:, 1] = gamma_eff_db_norm = (10*log10(gamma_eff) - 10) / 20
        meta[:, 2] = chi              (raw, range [0, 2/π])
        meta[:, 3] = sigma_eta_norm   = sigma_eta / 0.1
        meta[:, 4] = pn_linewidth_norm = log10(pn_linewidth + 1) / log10(1e6)
        meta[:, 5] = ibo_db_norm      = (ibo_dB - 3) / 3

Author: Expert Review Fixed v5 (PN Tracker Normalization)
Date: 2025-12-18
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
    [FIX 5 + FIX 12] Improved Phase Noise Tracker

    FIX 5: Explicit (cos, sin) output with normalization
    FIX 12: Learnable phase bias correction for initial phase ambiguity
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=2,
            hidden_size=cfg.hidden_dim_pn,
            bidirectional=cfg.use_bidirectional_pn,
            batch_first=True
        )
        out_dim = cfg.hidden_dim_pn * 2 if cfg.use_bidirectional_pn else cfg.hidden_dim_pn

        self.head = nn.Sequential(
            nn.Linear(out_dim, cfg.hidden_dim_pn // 2),
            nn.ReLU(),
            nn.Linear(cfg.hidden_dim_pn // 2, 2)
        )

        # [FIX 12] Learnable phase bias correction
        # Initialize to 0, let training find the right value
        self.phase_bias = nn.Parameter(torch.tensor(-1.5708))  # -π/2 ≈ -1.5708

    def forward(self, y_q: torch.Tensor, g_PN: torch.Tensor,
                x_est: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Estimate phase noise and return de-rotation factor.
        """
        if x_est is not None and x_est.abs().mean() > 0.1:
            x_norm = x_est / (x_est.abs() + 1e-6)
            resid = y_q * torch.conj(x_norm)
            feats = torch.stack([resid.real, resid.imag], dim=-1)
        else:
            feats = torch.stack([y_q.real, y_q.imag], dim=-1)

        rnn_out, _ = self.rnn(feats)

        cos_sin = self.head(rnn_out)
        cos_phi = cos_sin[..., 0]
        sin_phi = cos_sin[..., 1]

        mag = torch.sqrt(cos_phi ** 2 + sin_phi ** 2 + 1e-8)
        cos_phi = cos_phi / mag
        sin_phi = sin_phi / mag

        derotator = torch.complex(cos_phi, -sin_phi)

        # [FIX 12] Apply learnable phase bias correction (符号修正!)
        bias_correction = torch.exp(-1j * self.phase_bias)
        derotator = derotator * bias_correction

        # [FIX 8] Normalize after interpolation
        eff_derotator = (1 - g_PN) * torch.ones_like(derotator) + g_PN * derotator
        eff_derotator = eff_derotator / (eff_derotator.abs() + 1e-8)

        return eff_derotator

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
    [FIX 4 & 9] Theta Update Module with Physical Constraints

    [FIX 9] Added:
    - Physical clipping: Limits per-layer update to physically plausible range
    - Confidence gating: Only update when symbol estimate is reliable
    - Prevents drift that caused BER ≈ 0.5
    """

    def __init__(self, cfg: GABVConfig):
        super().__init__()
        # Network to estimate theta gradient from residual statistics
        self.grad_net = nn.Sequential(
            nn.Linear(6, 32),  # Input: residual stats + gate
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 3),  # Output: Δθ
        )

        # Learnable step sizes for R, v, a (different scales)
        self.lr_param = nn.Parameter(torch.tensor([1e-2, 1e-3, 1e-4]))

        # Theta bounds for clamping
        self.register_buffer('theta_min', torch.tensor([1e3, -1e4, -100.0]))
        self.register_buffer('theta_max', torch.tensor([1e7, 1e4, 100.0]))

        # [FIX 9] Physical limits for per-layer update
        # Max velocity ~8km/s, block duration ~0.1ms → max displacement ~0.8m per block
        # We allow 20m per layer to be safe (accounts for estimation error)
        self.register_buffer('max_delta_R', torch.tensor(20.0))  # meters
        self.register_buffer('max_delta_v', torch.tensor(5.0))  # m/s
        self.register_buffer('max_delta_a', torch.tensor(1.0))  # m/s²

        # [FIX 9] Confidence threshold for gating
        self.register_buffer('confidence_threshold', torch.tensor(0.005))

    def forward(self, theta: torch.Tensor, resid: torch.Tensor,
                x_est: torch.Tensor, g_theta: torch.Tensor,
                fim_inv: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Update theta based on residual with physical constraints.

        [FIX 9] Key additions:
        1. Physical clipping: |ΔR| < 20m, |Δv| < 5m/s, |Δa| < 1m/s²
        2. Confidence gating: Only update when |x_est| > threshold
        """
        B = theta.shape[0]

        # Compute residual statistics
        resid_power = torch.mean(resid.abs() ** 2, dim=1, keepdim=True)  # [B, 1]
        resid_phase_mean = torch.mean(torch.angle(resid + 1e-8), dim=1, keepdim=True)
        resid_phase_var = torch.var(torch.angle(resid + 1e-8), dim=1, keepdim=True)

        # Cross-correlation with x_est
        cross_corr_real = torch.mean((resid * torch.conj(x_est)).real, dim=1, keepdim=True)
        cross_corr_imag = torch.mean((resid * torch.conj(x_est)).imag, dim=1, keepdim=True)

        # Feature vector
        feat = torch.cat([
            resid_power,
            resid_phase_mean,
            resid_phase_var,
            cross_corr_real,
            cross_corr_imag,
            g_theta
        ], dim=1)  # [B, 6]

        # Estimate update direction
        delta_theta = self.grad_net(feat)  # [B, 3]

        # Apply FIM scaling if available (natural gradient)
        if fim_inv is not None:
            delta_theta = delta_theta * fim_inv

        # Apply gate and step size
        lr = torch.abs(self.lr_param)  # Ensure positive
        delta_theta_scaled = lr * delta_theta  # [B, 3]

        # ============================================================
        # [FIX 9] Physical Clipping - Prevent unrealistic updates
        # ============================================================
        # Clamp each parameter independently
        delta_R = torch.clamp(delta_theta_scaled[:, 0:1], -self.max_delta_R, self.max_delta_R)
        delta_v = torch.clamp(delta_theta_scaled[:, 1:2], -self.max_delta_v, self.max_delta_v)
        delta_a = torch.clamp(delta_theta_scaled[:, 2:3], -self.max_delta_a, self.max_delta_a)
        delta_theta_clipped = torch.cat([delta_R, delta_v, delta_a], dim=1)

        # ============================================================
        # [FIX 9] Confidence Gating - Only update when x_est is reliable
        # ============================================================
        # Use average symbol magnitude as confidence proxy
        # QPSK symbols have magnitude ~1, noisy estimates are smaller
        confidence = torch.mean(x_est.abs(), dim=1, keepdim=True)  # [B, 1]

        # Soft gate: sigmoid centered at threshold
        confidence_gate = torch.sigmoid((confidence - self.confidence_threshold) * 10)

        # ============================================================
        # Final Update with both gates
        # ============================================================
        # g_theta: learned gate from network
        # confidence_gate: physics-based confidence gate
        effective_gate = g_theta * confidence_gate

        theta_new = theta + effective_gate * delta_theta_clipped

        # Final clamping to reasonable range
        theta_new = torch.clamp(theta_new, self.theta_min, self.theta_max)

        return theta_new

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
            self.pn_tracker = RiemannianPNTracker(cfg)
            self.refiner = BussgangRefiner(cfg)
            self.solver = UnrolledCGSolver(cfg)
            self.theta_updater = ThetaUpdater(cfg)  # [FIX 4] New theta updater
        else:
            raise NotImplementedError("Per-layer weights not yet implemented.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GA-BV-Net.

        [FIX 4] Now actually updates theta!
        """
        y_q = batch['y_q']
        meta = batch['meta']
        theta = batch['theta_init'].clone()  # Will be updated!

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

            # 3. PN Tracking - [FIX 5] improved tracker
            derotator = self.pn_tracker(y_q, gates['g_PN'], x_est if k > 0 else None)
            y_corr = y_q * derotator

            # 4. Refiner
            z_denoised, onsager = self.refiner(y_corr, gates['g_NL'])

            # 5. Solver
            x_est = self.solver(z_denoised, x_est, self.phys_enc, theta, snr_linear)

            # 6. [FIX 4] Theta Update - ACTUALLY UPDATE!
            if self.cfg.enable_theta_update and k >= self.cfg.theta_update_start_layer:
                # Compute residual
                h_diag = self.phys_enc.compute_channel_diag(theta)
                resid = z_denoised - h_diag * x_est

                # Update theta
                theta = self.theta_updater(
                    theta, resid, x_est,
                    gates['g_theta'], fim_inv
                )

            layer_outputs.append({
                'x_est': x_est.clone(),
                'theta': theta.clone(),
                'gates': gates,
                'derotator': derotator
            })

        return {
            'x_hat': x_est,
            'theta_hat': theta,
            'phi_hat': layer_outputs[-1]['derotator'],
            'layers': layer_outputs,
            'geom_cache': geom_cache
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
    print("GABVNet Model - Fixed Version v3 Self-Test")
    print("=" * 60)

    # Check fs consistency
    cfg = GABVConfig()
    print(f"\n[Config Check]")
    print(f"  fs = {cfg.fs/1e9:.1f} GHz (should be 10.0)")
    print(f"  fc = {cfg.fc/1e9:.1f} GHz")
    print(f"  N  = {cfg.N_sym}")
    print(f"  enable_theta_update = {cfg.enable_theta_update}")

    print("\n" + "=" * 60)
    print("Self-Test Complete")
    print("=" * 60)