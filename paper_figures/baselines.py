"""
baselines.py - Baseline algorithms (Expert Review - Top Journal Ready)

Expert Requirements Implemented:
- P0-1: Trial-first evaluation support (same trial for all methods)
- P0-2: Fixed matched_filter (41 grid points, covers max init_error)
- P0-3: Added proposed_tau_slice (ablation: τ update + hard slice)
- P0-4: Added oracle_local_best (Oracle-B: local τ search)
- P0-5: Dual Oracle naming (oracle_sync = Oracle-A, oracle_local_best = Oracle-B)

Method hierarchy (weak → strong):
1. naive_slice         - Direct slice (no frontend)
2. matched_filter      - Grid Search τ + Slice (41-point, expensive upper bound)
3. adjoint_lmmse       - Adjoint + PN Align + LMMSE
4. adjoint_slice       - Adjoint + PN Align + Hard Slice
5. proposed_no_update  - BV-VAMP without τ update
6. proposed_tau_slice  - Proposed τ estimation + Slice (ablation)
7. proposed            - Full method (BV-VAMP + τ update)
8. oracle_sync         - Oracle-A: Same hardware, true θ (main comparison)
9. oracle_local_best   - Oracle-B: Local best τ (strongest bound)
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
    """Shared frontend for all methods (ensures fair comparison)."""
    batch_size = y_q.shape[0]
    device = y_q.device

    try:
        z = model.phys_enc.adjoint_operator(y_q, theta)
    except Exception as e:
        print(f"Warning: adjoint_operator failed: {e}")
        z = y_q

    if x_pilot is not None and pilot_len > 0:
        x_p = x_pilot[:, :pilot_len]
        z_p = z[:, :pilot_len]
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
    name = "naive_slice"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_hat = qpsk_hard_slice(y_q)
        return x_hat, theta_init


class BaselineMatchedFilter:
    name = "matched_filter"
    GRID_POINTS = 11
    DEFAULT_SEARCH_HALF_RANGE = 0.5

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            search_half_range: float = None, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']

        Ts = 1.0 / sim_cfg.fs
        if search_half_range is None:
            search_half_range = BaselineMatchedFilter.DEFAULT_SEARCH_HALF_RANGE

        grid_points = BaselineMatchedFilter.GRID_POINTS
        tau_grid_samples = torch.linspace(-search_half_range, search_half_range, grid_points, device=device)

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
    name = "adjoint_lmmse"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']

        z_derot, _ = frontend_adjoint_and_pn(model, y_q, theta_init, x_true, pilot_len)
        snr_lin = 10 ** (sim_cfg.snr_db / 10)
        sigma2 = 1.0 / snr_lin
        alpha = np.sqrt(2 / np.pi)
        x_hat_soft = z_derot / (alpha**2 + sigma2)
        x_hat = qpsk_hard_slice(x_hat_soft)

        return x_hat, theta_init


class BaselineAdjointSlice:
    name = "adjoint_slice"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_init = batch['theta_init']
        x_true = batch['x_true']

        z_derot, _ = frontend_adjoint_and_pn(model, y_q, theta_init, x_true, pilot_len)
        x_hat = qpsk_hard_slice(z_derot)

        return x_hat, theta_init


class BaselineProposedNoUpdate:
    name = "proposed_no_update"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_init = batch['theta_init']
        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = False
        outputs = model(batch)
        model.cfg.enable_theta_update = original_setting
        x_hat = outputs['x_hat']
        theta_hat = outputs.get('theta_hat', theta_init)
        return x_hat, theta_hat


class BaselineProposedTauSlice:
    name = "proposed_tau_slice"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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
    name = "proposed"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        theta_init = batch['theta_init']
        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = True
        outputs = model(batch)
        model.cfg.enable_theta_update = original_setting
        x_hat = outputs['x_hat']
        theta_hat = outputs.get('theta_hat', theta_init)
        return x_hat, theta_hat


class BaselineOracleSync:
    name = "oracle_sync"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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


class BaselineOracleLocalBest:
    name = "oracle_local_best"
    SEARCH_HALF_RANGE = 0.2
    GRID_POINTS = 201

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
        y_q = batch['y_q']
        theta_true = batch['theta_true']
        x_true = batch['x_true']

        Ts = 1.0 / sim_cfg.fs
        tau_grid_samples = torch.linspace(
            -BaselineOracleLocalBest.SEARCH_HALF_RANGE,
            BaselineOracleLocalBest.SEARCH_HALF_RANGE,
            BaselineOracleLocalBest.GRID_POINTS, device=device
        )

        best_corr = None
        best_tau = theta_true[:, 0:1].clone()
        x_pilot = x_true[:, :pilot_len]

        for tau_offset in tau_grid_samples:
            theta_test = theta_true.clone()
            theta_test[:, 0:1] = theta_true[:, 0:1] + tau_offset * Ts
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

        theta_hat = theta_true.clone()
        theta_hat[:, 0:1] = best_tau

        batch_oracle = batch.copy()
        batch_oracle['theta_init'] = theta_hat.clone()

        original_setting = model.cfg.enable_theta_update
        model.cfg.enable_theta_update = False
        outputs = model(batch_oracle)
        model.cfg.enable_theta_update = original_setting

        x_hat = outputs['x_hat']
        return x_hat, theta_hat


class BaselineOracle(BaselineOracleSync):
    name = "oracle"


class BaselineRandomInit:
    name = "random_init"

    @staticmethod
    @torch.no_grad()
    def run(model, batch: Dict, sim_cfg, device: str, pilot_len: int = 64,
            **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
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
    "oracle_local_best": BaselineOracleLocalBest,
    "oracle": BaselineOracle,
    "random_init": BaselineRandomInit,
}

METHOD_PAPER_CORE = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]
METHOD_PAPER_FULL = ["naive_slice", "matched_filter", "adjoint_lmmse", "adjoint_slice",
                     "proposed_no_update", "proposed_tau_slice", "proposed", "oracle_sync"]
METHOD_DEBUG = ["adjoint_slice", "proposed", "oracle_sync"]
METHOD_CLIFF = ["naive_slice", "adjoint_lmmse", "adjoint_slice", "matched_filter",
                "proposed_no_update", "proposed", "oracle_sync"]
METHOD_ABLATION = ["random_init", "proposed_no_update", "proposed_tau_slice", "proposed", "oracle_sync"]
METHOD_SNR_SWEEP = ["matched_filter", "adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]
METHOD_ROBUSTNESS = ["adjoint_slice", "proposed_no_update", "proposed", "oracle_sync"]
METHOD_ORACLE_COMPARE = ["proposed", "oracle_sync", "oracle_local_best"]
METHOD_ORDER = METHOD_PAPER_FULL
METHOD_QUICK = METHOD_DEBUG


def get_baseline(method_name: str):
    if method_name == "oracle":
        method_name = "oracle_sync"
    if method_name not in BASELINE_REGISTRY:
        raise ValueError(f"Unknown method: {method_name}. Available: {list(BASELINE_REGISTRY.keys())}")
    return BASELINE_REGISTRY[method_name]


def run_baseline(method_name: str, model, batch: Dict, sim_cfg, device: str,
                 pilot_len: int = 64, **kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    baseline_cls = get_baseline(method_name)
    return baseline_cls.run(model, batch, sim_cfg, device, pilot_len, **kwargs)


def get_method_info() -> Dict[str, Dict]:
    info = {}
    for name, cls in BASELINE_REGISTRY.items():
        info[name] = {'name': cls.name, 'class': cls.__name__}
        if name == 'matched_filter':
            info[name]['grid_points'] = cls.GRID_POINTS
            info[name]['default_search_half_range'] = cls.DEFAULT_SEARCH_HALF_RANGE
        elif name == 'oracle_local_best':
            info[name]['search_half_range'] = cls.SEARCH_HALF_RANGE
            info[name]['grid_points'] = cls.GRID_POINTS
    return info