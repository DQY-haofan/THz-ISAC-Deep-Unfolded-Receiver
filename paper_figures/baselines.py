"""
baselines.py - Baseline algorithms (Expert v2.0 - Top Journal Ready)

Expert Requirements Implemented:
- P0-2: Dual Oracle naming (oracle_sync = Oracle-A, oracle_ideal = Oracle-B)
- P0-3: Paper method sets (METHOD_PAPER_CORE, METHOD_PAPER_FULL, METHOD_DEBUG)
- Fair comparison: All baselines share same frontend

Method hierarchy (weak → strong):
1. naive_slice      - Direct slice (no frontend)
2. matched_filter   - Grid Search τ + Slice (expensive upper bound)
3. adjoint_lmmse    - Adjoint + PN Align + LMMSE
4. adjoint_slice    - Adjoint + PN Align + Hard Slice
5. proposed_no_update - BV-VAMP without τ update
6. proposed_tau_slice - Proposed τ estimation + Slice (ablation)
7. proposed         - Full method (BV-VAMP + τ update)
8. oracle_sync      - Oracle-A: Same hardware, true θ (main comparison)
9. oracle_ideal     - Oracle-B: Ideal hardware (optional, appendix only)
"""

import torch
import numpy as np
import math
from typing import Dict, Tuple, Optional, List


# ============================================================================
# Helper Functions
# ============================================================================

def qpsk_hard_slice(z: torch.Tensor) -> torch.Tensor:
    """QPSK hard decision in complex plane."""
    xr = torch.sign(z.real)
    xi = torch.sign(z.imag)
    xr = torch.where(xr == 0, torch.ones_like(xr), xr)
    xi = torch.where(xi == 0, torch.ones_like(xi), xi)
    return (xr + 1j * xi) / np.sqrt(2)


def frontend_adjoint_and_pn(model, y_q: torch.Tensor, theta: torch.Tensor,
                            x_pilot: torch.Tensor, pilot_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Shared frontend for all methods (ensures fair comparison).

    Flow:
      z = H*(theta) y_q          # Adjoint operation
      z_derot = pn_derotation(z) # Pilot-based constant phase alignment

    Args:
        model: GABVNet model
        y_q: Quantized received signal [B, N]
        theta: Channel parameters [B, 3] = [τ, v, a]
        x_pilot: True symbols (with pilot) [B, N]
        pilot_len: Pilot length

    Returns:
        z_derot: De-rotated signal [B, N]
        phi_est: Estimated phase [B, 1]
    """
    batch_size = y_q.shape[0]
    device = y_q.device

    # 1) Adjoint operation
    try:
        z = model.phys_enc.adjoint_operator(y_q, theta)
    except Exception as e:
        print(f"Warning: adjoint_operator failed: {e}")
        z = y_q

    # 2) Pilot-based constant phase alignment
    if x_pilot is not None and pilot_len > 0:
        x_p = x_pilot[:, :pilot_len]
        z_p = z[:, :pilot_len]

        # Correct phase estimation
        correlation = torch.sum(z_p * torch.conj(x_p), dim=1, keepdim=True)
        phi_est = torch.angle(correlation)
        z_derot = z * torch.exp(-1j * phi_est)
    else:
        z_derot = z
        phi_est = torch.zeros(batch_size, 1, device=device)

    return z_derot, phi_est


# ============================================================================
# Baseline Implementations
# ============================================================================

class BaselineNaiveSlice:
    """Weakest baseline: direct slice without any frontend."""
    name = "naive_slice"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_hat = qpsk_hard_slice(y_q)
        return x_hat, theta_init


class BaselineMatchedFilter:
    """
    Matched Filter / Grid Search τ estimation + Hard Slice.

    Note: This is an EXPENSIVE upper bound (41× FFTs).
    Visualize as semi-transparent dashed line to show computational cost.
    """
    name = "matched_filter"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']
        batch_size = y_q.shape[0]

        Ts = 1.0 / sim_cfg.fs

        # τ grid search range: ±2.0 samples (41 points)
        tau_grid_samples = torch.linspace(-2.0, 2.0, 41, device=device)

        best_corr = None
        best_tau = theta_init[:, 0:1].clone()
        x_pilot = x_true[:, :pilot_len]

        for tau_offset in tau_grid_samples:
            theta_test = theta_init.clone()
            theta_test[:, 0:1] = theta_init[:, 0:1] + tau_offset * Ts

            z_derot, _ = frontend_adjoint_and_pn(model, y_q, theta_test, x_true, pilot_len)

            z_p = z_derot[:, :pilot_len]
            corr = torch.abs(torch.sum(z_p.conj() * x_pilot, dim=1, keepdim=True))

            if best_corr is None:
                best_corr = corr
                best_tau = theta_test[:, 0:1]
            else:
                mask = corr > best_corr
                best_corr = torch.where(mask, corr, best_corr)
                best_tau = torch.where(mask, theta_test[:, 0:1], best_tau)

        theta_hat = theta_init.clone()
        theta_hat[:, 0:1] = best_tau

        z_derot, _ = frontend_adjoint_and_pn(model, y_q, theta_hat, x_true, pilot_len)
        x_hat = qpsk_hard_slice(z_derot)

        return x_hat, theta_hat


class BaselineAdjointLMMSE:
    """Adjoint + Pilot PN Align + Bussgang-LMMSE."""
    name = "adjoint_lmmse"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']

        z_derot, phi_est = frontend_adjoint_and_pn(model, y_q, theta_init, x_true, pilot_len)

        snr_lin = 10 ** (sim_cfg.snr_db / 10)
        sigma2 = 1.0 / snr_lin
        alpha = np.sqrt(2 / np.pi)

        x_hat_soft = z_derot / (alpha**2 + sigma2)
        x_hat = qpsk_hard_slice(x_hat_soft)

        return x_hat, theta_init


class BaselineAdjointSlice:
    """
    Adjoint + Pilot PN Align + Hard Slice.

    Key comparison baseline for the paper.
    """
    name = "adjoint_slice"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']

        z_derot, phi_est = frontend_adjoint_and_pn(model, y_q, theta_init, x_true, pilot_len)
        x_hat = qpsk_hard_slice(z_derot)

        return x_hat, theta_init


class BaselineProposedNoUpdate:
    """BV-VAMP without τ update (ablation: verify τ update value)."""
    name = "proposed_no_update"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_init = batch['theta_init']

        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = False

        outputs = model(batch)

        model.cfg.enable_theta_update = original_setting

        x_hat = outputs['x_hat']
        theta_hat = outputs.get('theta_hat', theta_init)

        return x_hat, theta_hat


class BaselineProposedTauSlice:
    """
    Proposed τ estimation + Hard Slice (ablation: verify VAMP value).

    Uses proposed for τ estimation but slice for detection.
    """
    name = "proposed_tau_slice"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = True

        outputs = model(batch)
        theta_hat = outputs.get('theta_hat', batch['theta_init'])

        model.cfg.enable_theta_update = original_setting

        y_q = batch['y_q']
        x_true = batch['x_true']

        z_derot, _ = frontend_adjoint_and_pn(model, y_q, theta_hat, x_true, pilot_len)
        x_hat = qpsk_hard_slice(z_derot)

        return x_hat, theta_hat


class BaselineProposed:
    """Full proposed method: BV-VAMP + τ update."""
    name = "proposed"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_init = batch['theta_init']

        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = True

        outputs = model(batch)

        model.cfg.enable_theta_update = original_setting

        x_hat = outputs['x_hat']
        theta_hat = outputs.get('theta_hat', theta_init)

        return x_hat, theta_hat


class BaselineOracleSync:
    """
    Oracle-A (Sync Oracle): Same 1-bit, same dirty hardware, only true θ given.

    This is the PRIMARY oracle for gap-to-oracle computation.
    Used to measure the "inference efficiency" of the estimator.
    """
    name = "oracle_sync"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_true = batch['theta_true']

        batch_oracle = batch.copy()
        batch_oracle['theta_init'] = theta_true.clone()

        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = False

        outputs = model(batch_oracle)

        model.cfg.enable_theta_update = original_setting

        x_hat = outputs['x_hat']
        theta_hat = theta_true.clone()

        return x_hat, theta_hat


# Legacy alias for backward compatibility
class BaselineOracle(BaselineOracleSync):
    """Legacy alias: oracle → oracle_sync."""
    name = "oracle"


class BaselineRandomInit:
    """Random Init: Large random θ error (theoretical lower bound)."""
    name = "random_init"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_true = batch['theta_true']
        batch_size = theta_true.shape[0]
        Ts = 1.0 / sim_cfg.fs

        noise_tau = torch.randn(batch_size, 1, device=device) * 2.0 * Ts
        noise_v = torch.randn(batch_size, 1, device=device) * 50.0
        noise_a = torch.randn(batch_size, 1, device=device) * 10.0

        theta_init = theta_true.clone()
        theta_init[:, 0:1] += noise_tau
        theta_init[:, 1:2] += noise_v
        theta_init[:, 2:3] += noise_a

        batch_random = batch.copy()
        batch_random['theta_init'] = theta_init

        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = False

        outputs = model(batch_random)

        model.cfg.enable_theta_update = original_setting

        x_hat = outputs['x_hat']
        theta_hat = theta_init

        return x_hat, theta_hat


# ============================================================================
# Method Registry
# ============================================================================

BASELINE_REGISTRY = {
    "naive_slice": BaselineNaiveSlice,
    "matched_filter": BaselineMatchedFilter,
    "adjoint_lmmse": BaselineAdjointLMMSE,
    "adjoint_slice": BaselineAdjointSlice,
    "proposed_no_update": BaselineProposedNoUpdate,
    "proposed_tau_slice": BaselineProposedTauSlice,
    "proposed": BaselineProposed,
    "oracle_sync": BaselineOracleSync,
    "oracle": BaselineOracle,  # Legacy alias
    "random_init": BaselineRandomInit,
}


# ============================================================================
# Paper Method Sets (P0-3)
# ============================================================================

# Paper Core: Main text figures (minimal set for core story)
METHOD_PAPER_CORE = [
    "adjoint_slice",
    "proposed_no_update",
    "proposed",
    "oracle_sync",
]

# Paper Full: Main text + appendix (complete comparison)
METHOD_PAPER_FULL = [
    "naive_slice",
    "matched_filter",
    "adjoint_lmmse",
    "adjoint_slice",
    "proposed_no_update",
    "proposed_tau_slice",
    "proposed",
    "oracle_sync",
]

# Debug: Quick testing only
METHOD_DEBUG = [
    "adjoint_slice",
    "proposed",
    "oracle_sync",
]

# Cliff sweep methods (core contribution figure)
METHOD_CLIFF = [
    "naive_slice",
    "adjoint_lmmse",
    "adjoint_slice",
    "matched_filter",
    "proposed_no_update",
    "proposed",
    "oracle_sync",
]

# Ablation methods
METHOD_ABLATION = [
    "random_init",
    "proposed_no_update",
    "proposed_tau_slice",
    "proposed",
    "oracle_sync",
]

# SNR sweep methods
METHOD_SNR_SWEEP = [
    "matched_filter",
    "adjoint_slice",
    "proposed_no_update",
    "proposed",
    "oracle_sync",
]

# Robustness sweep methods (PN, Pilot)
METHOD_ROBUSTNESS = [
    "adjoint_slice",
    "proposed_no_update",
    "proposed",
    "oracle_sync",
]

# Legacy aliases
METHOD_ORDER = METHOD_PAPER_FULL
METHOD_QUICK = METHOD_DEBUG


# ============================================================================
# API Functions
# ============================================================================

def get_baseline(method_name: str):
    """Get baseline class by name."""
    # Handle legacy "oracle" name
    if method_name == "oracle":
        method_name = "oracle_sync"

    if method_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[method_name]


def run_baseline(method_name: str, model, batch: Dict, sim_cfg, device: str,
                 pilot_len: int = 64) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run specified baseline algorithm."""
    baseline_cls = get_baseline(method_name)
    return baseline_cls.run(model, batch, sim_cfg, device, pilot_len)


def validate_method_set(methods: List[str], required_methods: List[str],
                        context: str = "") -> bool:
    """Validate that all required methods are present."""
    missing = set(required_methods) - set(methods)
    if missing:
        print(f"⚠️ {context}: Missing methods {missing}")
        return False
    return True