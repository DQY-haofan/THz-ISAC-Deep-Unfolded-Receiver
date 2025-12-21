"""
baselines_receivers.py (GPU Version)

Description:
    PyTorch implementation of LMMSE and 1-bit GAMP baselines.
    Supports Batch processing on GPU for high-speed simulation.

    Algorithms:
    1. Bussgang-LMMSE (Diagonal approximation for speed)
    2. 1-bit GAMP (Vectorized)
"""

import torch
import torch.nn.functional as F
import math

def qpsk_demod_torch(x_hat):
    """Hard slicer for QPSK (Differentiable-ish for sign, hard for BER)."""
    # x_hat: [B, N] complex
    xr = torch.sign(x_hat.real)
    xi = torch.sign(x_hat.imag)
    # Handle zeros
    xr[xr == 0] = 1
    xi[xi == 0] = 1
    return (xr + 1j * xi) / math.sqrt(2)

# --- 1. GPU LMMSE (Bussgang) ---

def detector_lmmse_bussgang_torch(y_q, H_diag, noise_var, P_signal=1.0):
    """
    Element-wise LMMSE for Diagonal Channels on GPU.

    Args:
        y_q: [B, N] Complex tensor (Observation)
        H_diag: [B, N] Complex tensor (Diagonal Channel Response)
        noise_var: float or [B, 1] (Effective Noise Variance)
        P_signal: float (Signal Power)

    Returns:
        x_hat: [B, N] Estimated symbols
    """
    # W = (P * H') / (P * |H|^2 + noise)
    # This is the scalar Wiener filter applied per subcarrier/sample

    # Numerator: P * H*
    num = P_signal * torch.conj(H_diag)

    # Denominator: P * |H|^2 + sigma^2
    # Ensure real-valued denominator
    h_sq = H_diag.abs() ** 2
    den = P_signal * h_sq + noise_var

    # Filter
    W = num / (den + 1e-9)

    # Apply: x = W * y
    x_hat = W * y_q

    return x_hat

# --- 2. GPU 1-bit GAMP ---

def phi_norm_cdf(x):
    """Standard Normal CDF (GPU friendly)."""
    return 0.5 * (1 + torch.erf(x / math.sqrt(2)))

def phi_norm_pdf(x):
    """Standard Normal PDF (GPU friendly)."""
    return torch.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)

def output_update_1bit_torch(p, v_p, y, noise_var):
    """GAMP Output Node (1-bit Quantizer) on GPU."""
    # p, v_p: [B, N] (Mean and Var of linear mix)
    # y: [B, N] (Observed bits, +/- 1)

    # Effective noise std dev (Complex split -> /2)
    tau = torch.sqrt(v_p + noise_var/2 + 1e-9)

    # Process Real and Imaginary separately
    def compute_g(p_c, y_c, tau_c):
        # argument: (y * p) / tau
        # y_c is +/- 1
        xi = (y_c * p_c) / tau_c

        # Mill's ratio: pdf(xi) / cdf(xi)
        cdf = phi_norm_cdf(xi)
        cdf = torch.clamp(cdf, min=1e-9) # Avoid div zero
        pdf = phi_norm_pdf(xi)
        ratio = pdf / cdf

        # g = (y / tau) * ratio
        g = (y_c / tau_c) * ratio

        # -g' = (ratio * (xi + ratio)) / tau^2
        dg = (ratio * (xi + ratio)) / (tau_c**2)

        return g, dg

    g_re, dg_re = compute_g(p.real, y.real, tau)
    g_im, dg_im = compute_g(p.imag, y.imag, tau)

    g = torch.complex(g_re, g_im)
    dg = dg_re + dg_im # Approximate complex variance update

    # z_hat = p + v_p * g
    z_hat = p + v_p * g

    # v_z = v_p - v_p^2 * dg
    v_z = v_p - (v_p**2) * dg
    v_z = torch.clamp(v_z, min=1e-9)

    return z_hat, v_z

def detector_gamp_1bit_torch(y_q, H_diag, noise_var, max_iter=20, damping=0.5):
    """
    1-bit GAMP on GPU (Diagonal H optimized).
    """
    B, N = y_q.shape
    device = y_q.device

    # Init
    x_hat = torch.zeros_like(y_q)
    v_x = torch.ones(B, N, device=device) # Prior variance = 1
    s_hat = torch.zeros_like(y_q)

    # Precompute |H|^2
    S = H_diag.abs()**2 # [B, N]

    for _ in range(max_iter):
        # 1. Linear Step (Output)
        # For diagonal H: Matrix mul becomes element-wise mul
        # v_p = |H|^2 * v_x
        v_p = S * v_x

        # p = H * x - v_p * s (Onsager)
        p = H_diag * x_hat - v_p * s_hat

        # 2. Non-linear Step (Output)
        z_hat, v_z = output_update_1bit_torch(p, v_p, y_q, noise_var)

        # Residual s
        # g_out = (z - p) / v_p
        g_out = (z_hat - p) / (v_p + 1e-9)
        dg_out = (1.0 - v_z / (v_p + 1e-9)) / (v_p + 1e-9)

        s_hat = g_out
        v_s = dg_out

        # 3. Linear Step (Input)
        # inv_v_r = |H|^2 * v_s
        inv_v_r = S * v_s
        v_r = 1.0 / (inv_v_r + 1e-9)

        # r = x + v_r * (H' * s)
        r = x_hat + v_r * (torch.conj(H_diag) * s_hat)

        # 4. Input Estimator (Prior: Gaussian(0,1))
        # MMSE Denoise: x_new = r / (1 + v_r)
        v_x_new = 1.0 / (1.0 + 1.0/v_r)
        x_hat_new = v_x_new * (r / v_r)

        # Damping
        x_hat = damping * x_hat + (1 - damping) * x_hat_new
        v_x = damping * v_x + (1 - damping) * v_x_new

    return x_hat