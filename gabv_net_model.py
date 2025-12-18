"""
gabv_net_model.py (Fixed Version v2)

Description:
    PyTorch Implementation of Geometry-Aware Bussgang-VAMP Network (GA-BV-Net).

    Fixes Applied:
    - [v1] Removed incorrect 'unsqueeze' calls that caused dimensionality explosion (4D tensors).
    - [v1] Ensures correct broadcasting for [B, 1] * [B, N] operations.
    - [v2] CRITICAL FIX: SNR denormalization in forward() - converts normalized meta features
           back to physical values before use in CG solver regularization.

    Meta Feature Schema (FROZEN - must match train_gabv_net.py):
        meta[:, 0] = snr_db_norm      = (snr_db - 15) / 15
        meta[:, 1] = gamma_eff_db_norm = (10*log10(gamma_eff) - 10) / 20
        meta[:, 2] = chi              (raw, range [0, 2/π])
        meta[:, 3] = sigma_eta_norm   = sigma_eta / 0.1
        meta[:, 4] = pn_linewidth_norm = log10(pn_linewidth + 1) / log10(1e6)
        meta[:, 5] = ibo_db_norm      = (ibo_dB - 3) / 3
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

    N_sym: int = 1024
    fs: float = 25e9
    fc: float = 300e9

    share_weights: bool = True
    enable_geom_loss: bool = True
    block_size_fim: int = 32


# --- Meta Feature Normalization Constants (FROZEN) ---
# These MUST match the constants in train_gabv_net.py exactly!

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
            nn.Linear(h_dim // 2, 3)
        )
        self.register_buffer('brake_threshold', torch.tensor(-15.0))

    def forward(self, meta_features: torch.Tensor, log_vol: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        logits = self.net(meta_features)

        # All outputs are [B, 1]
        g_PN = torch.sigmoid(logits[:, 0:1])
        g_NL = F.softplus(logits[:, 1:2]) + 0.5
        learned_brake = torch.sigmoid(logits[:, 2:3])

        if log_vol is not None:
            physics_brake = torch.sigmoid((log_vol - self.brake_threshold) * 5.0)
            g_grad = learned_brake * physics_brake
        else:
            g_grad = learned_brake

        return {"g_PN": g_PN, "g_NL": g_NL, "g_grad": g_grad}


class PhysicsEncoder(nn.Module):
    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.N, self.fs, self.fc = cfg.N_sym, cfg.fs, cfg.fc

        self.register_buffer('f_grid', torch.fft.fftfreq(self.N, d=1/self.fs))
        self.register_buffer('t_grid', torch.arange(self.N, dtype=torch.float32) / self.fs)

        self.reg_net = nn.Sequential(
            nn.Linear(1, 16), nn.ReLU(),
            nn.Linear(16, 1), nn.Softplus()
        )

    def forward_operator(self, x: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, v, a = theta[:, 0], theta[:, 1], theta[:, 2]

        tau = 2 * R / 3e8
        # [1, N] * [B, 1] -> [B, N]
        phase_shift = torch.exp(-1j * 2 * math.pi * self.f_grid.unsqueeze(0) * tau.unsqueeze(1))

        x_freq = torch.fft.fft(x, dim=1)
        x_delayed = torch.fft.ifft(x_freq * phase_shift, dim=1)

        k_fact = 4 * math.pi * self.fc / 3e8
        t = self.t_grid.unsqueeze(0)
        v = v.unsqueeze(1); a = a.unsqueeze(1)

        doppler_phase = 1j * k_fact * (v * t + 0.5 * a * (t**2))

        return x_delayed * torch.exp(doppler_phase)

    def adjoint_operator(self, y: torch.Tensor, theta: torch.Tensor) -> torch.Tensor:
        R, v, a = theta[:, 0], theta[:, 1], theta[:, 2]

        k_fact = 4 * math.pi * self.fc / 3e8
        t = self.t_grid.unsqueeze(0)
        v = v.unsqueeze(1); a = a.unsqueeze(1)
        doppler_phase = 1j * k_fact * (v * t + 0.5 * a * (t**2))

        y_derot = y * torch.exp(-doppler_phase)

        tau = 2 * R / 3e8
        phase_shift = torch.exp(-1j * 2 * math.pi * self.f_grid.unsqueeze(0) * tau.unsqueeze(1))

        y_freq = torch.fft.fft(y_derot, dim=1)
        y_out = torch.fft.ifft(y_freq * torch.conj(phase_shift), dim=1)

        return y_out

    def get_approx_fim_info(self, theta: torch.Tensor, gamma_eff: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        reg = self.reg_net(gamma_eff) + 1e-6
        log_vol = -torch.log(reg) * 3
        fim_inv_diag = torch.ones_like(theta) * reg
        return fim_inv_diag, log_vol


class RiemannianPNTracker(nn.Module):
    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.rnn = nn.GRU(
            input_size=2,
            hidden_size=cfg.hidden_dim_pn,
            bidirectional=cfg.use_bidirectional_pn,
            batch_first=True
        )
        out_dim = cfg.hidden_dim_pn * 2 if cfg.use_bidirectional_pn else cfg.hidden_dim_pn
        self.head = nn.Linear(out_dim, 1)

    def forward(self, resid_complex: torch.Tensor, g_PN: torch.Tensor) -> torch.Tensor:
        # resid_complex: [B, N]
        # feats: [B, N, 2]
        feats = torch.stack([resid_complex.real, resid_complex.imag], dim=-1)

        rnn_out, _ = self.rnn(feats)
        delta_phi = self.head(rnn_out).squeeze(-1) # [B, N]

        phi_cum = torch.cumsum(delta_phi, dim=1)
        rotator = torch.exp(-1j * phi_cum)

        # g_PN is [B, 1], rotator is [B, N]. Broadcasts correctly.
        eff_rotator = (1 - g_PN) + g_PN * rotator

        return eff_rotator


class BussgangRefiner(nn.Module):
    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.scale_slope = nn.Parameter(torch.tensor(1.0))

    def forward(self, z_in: torch.Tensor, g_NL: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # FIX: g_NL is already [B, 1]. Do NOT unsqueeze.
        # [B, 1] * [B, N] -> [B, N]
        beta = self.scale_slope * g_NL

        z_out_real = torch.tanh(beta * z_in.real)
        z_out_imag = torch.tanh(beta * z_in.imag)
        z_out = torch.complex(z_out_real, z_out_imag)

        d_real = beta * (1 - z_out_real**2)
        d_imag = beta * (1 - z_out_imag**2)

        onsager = torch.mean(d_real + d_imag, dim=1) * 0.5

        return z_out, onsager


class UnrolledCGSolver(nn.Module):
    def __init__(self, cfg: GABVConfig):
        super().__init__()
        self.steps = cfg.cg_steps
        self.step_scale = nn.Parameter(torch.tensor(1.0))

    def matvec(self, x, phys_enc, theta, lambda_reg):
        Hx = phys_enc.forward_operator(x, theta)
        HnHx = phys_enc.adjoint_operator(Hx, theta)
        # FIX: lambda_reg is [B, 1]. Do NOT unsqueeze.
        return HnHx + lambda_reg * x

    def forward(self, rhs: torch.Tensor, x_init: torch.Tensor,
                phys_enc: PhysicsEncoder, theta: torch.Tensor, snr_linear: torch.Tensor) -> torch.Tensor:
        """
        CG Solver for (H^H H + λI) x = rhs

        Args:
            rhs: Right-hand side [B, N]
            x_init: Initial estimate [B, N]
            phys_enc: Physics encoder module
            theta: Target parameters [B, 3]
            snr_linear: LINEAR SNR values [B, 1] (NOT normalized!)
                       This should be 10^(snr_db/10), typically in range [0.1, 1000]

        Returns:
            x: Solution [B, N]
        """
        # Regularization: λ = 1/SNR (Tikhonov regularization)
        # For SNR=15dB (linear ~31.6), λ ≈ 0.03
        # For SNR=0dB (linear ~1), λ ≈ 1.0
        lambda_reg = 1.0 / (snr_linear + 1e-6)  # [B, 1]

        x = x_init.clone()
        r = rhs - self.matvec(x, phys_enc, theta, lambda_reg)
        p = r.clone()
        rsold = torch.sum(r.abs()**2, dim=1)  # [B]

        for _ in range(self.steps):
            Ap = self.matvec(p, phys_enc, theta, lambda_reg)

            denom = torch.sum(torch.conj(p) * Ap, dim=1).real
            # rsold is [B], denom is [B]. Div is [B]. unsqueeze -> [B, 1].
            # This unsqueeze IS correct because rsold is 1D.
            alpha = (rsold / (denom + 1e-9)).unsqueeze(1) * self.step_scale

            x = x + alpha * p
            r = r - alpha * Ap

            rsnew = torch.sum(r.abs()**2, dim=1)
            beta = (rsnew / (rsold + 1e-9)).unsqueeze(1)
            p = r + beta * p
            rsold = rsnew

        return x


class FisherUpdater(nn.Module):
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
            self.updater = FisherUpdater(cfg)
        else:
            raise NotImplementedError("Per-layer weights not yet implemented.")

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Forward pass of GA-BV-Net.

        Args:
            batch: Dictionary containing:
                - 'y_q': Quantized received signal [B, N]
                - 'meta': Normalized meta features [B, 6]
                - 'theta_init': Initial target parameter estimate [B, 3]
                - 'x_init' (optional): Initial signal estimate [B, N]

        Returns:
            Dictionary containing:
                - 'x_hat': Estimated transmitted signal [B, N]
                - 'theta_hat': Estimated target parameters [B, 3]
                - 'phi_hat': Estimated phase noise [B, N]
                - 'layers': Per-layer outputs
                - 'geom_cache': Geometric information for loss computation
        """
        y_q = batch['y_q']
        meta = batch['meta']
        theta = batch['theta_init']

        # =====================================================================
        # CRITICAL FIX (v2): Denormalize meta features to physical values
        # =====================================================================
        # Meta features are normalized in train_gabv_net.py as:
        #   meta[:, 0] = snr_db_norm = (snr_db - 15) / 15
        #   meta[:, 1] = gamma_eff_db_norm = (10*log10(gamma_eff) - 10) / 20
        #
        # We need to convert back to physical values for the CG solver:
        #   snr_linear = 10^(snr_db / 10)
        #
        # Example:
        #   snr_db = 15 dB → snr_db_norm = 0 → snr_linear = 31.6
        #   snr_db = 0 dB  → snr_db_norm = -1 → snr_linear = 1.0
        #   snr_db = 30 dB → snr_db_norm = 1 → snr_linear = 1000
        # =====================================================================

        # Step 1: Extract normalized values
        snr_db_norm = meta[:, 0:1]       # Normalized SNR (dB)
        gamma_eff_db_norm = meta[:, 1:2] # Normalized Gamma_eff (dB)

        # Step 2: Denormalize to dB values
        snr_db = snr_db_norm * META_CONSTANTS['snr_db_scale'] + META_CONSTANTS['snr_db_center']
        gamma_eff_db = gamma_eff_db_norm * META_CONSTANTS['gamma_eff_db_scale'] + META_CONSTANTS['gamma_eff_db_center']

        # Step 3: Convert dB to linear scale
        snr_linear = 10.0 ** (snr_db / 10.0)           # For CG solver regularization
        gamma_eff_linear = 10.0 ** (gamma_eff_db / 10.0)  # For FIM computation

        # =====================================================================
        # End of CRITICAL FIX
        # =====================================================================

        if 'x_init' in batch:
            x_est = batch['x_init']
        else:
            x_est = torch.zeros_like(y_q)

        layer_outputs = []
        geom_cache = {'log_vols': [], 'fim_invs': []}

        for k in range(self.cfg.n_layers):
            # 1. Physics - use LINEAR gamma_eff for FIM
            fim_inv, log_vol = self.phys_enc.get_approx_fim_info(theta, gamma_eff_linear)
            if self.cfg.enable_geom_loss:
                geom_cache['log_vols'].append(log_vol)
                geom_cache['fim_invs'].append(fim_inv)

            # 2. Pilot - uses normalized meta (network learns from normalized features)
            gates = self.pilot(meta, log_vol=log_vol)

            # 3. PN Tracking
            resid_complex = y_q * torch.conj(x_est + 1e-6)
            rotator = self.pn_tracker(resid_complex, gates['g_PN'])
            y_corr = y_q * torch.conj(rotator)

            # 4. Refiner
            z_denoised, onsager = self.refiner(y_corr, gates['g_NL'])

            # 5. Solver - use LINEAR snr for regularization
            rhs = self.phys_enc.adjoint_operator(z_denoised, theta)
            x_est = self.solver(rhs, x_est, self.phys_enc, theta, snr_linear)

            # 6. Update
            theta_next = theta

            layer_outputs.append({
                'x_est': x_est,
                'theta': theta_next,
                'gates': gates,
                'rotator': rotator
            })
            theta = theta_next

        return {
            'x_hat': x_est,
            'theta_hat': theta,
            'phi_hat': layer_outputs[-1]['rotator'],
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
    """
    Utility function to denormalize meta features for debugging/logging.

    Args:
        meta: Normalized meta features [B, 6]

    Returns:
        Dictionary with denormalized physical values
    """
    snr_db_norm = meta[:, 0:1]
    gamma_eff_db_norm = meta[:, 1:2]
    chi = meta[:, 2:3]
    sigma_eta_norm = meta[:, 3:4]
    pn_linewidth_norm = meta[:, 4:5]
    ibo_db_norm = meta[:, 5:6]

    # Denormalize
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