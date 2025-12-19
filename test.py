#!/usr/bin/env python3
"""
debug_model.py - Debug GA-BV-Net model flow

This script tests each component to find where BER becomes 0.5
"""

import numpy as np
import sys

# Check torch
try:
    import torch

    HAS_TORCH = True
except ImportError:
    print("PyTorch not available!")
    HAS_TORCH = False
    sys.exit(1)

from thz_isac_world import SimConfig, simulate_batch


def compute_ber(x_hat, x_true):
    """Compute QPSK BER correctly."""
    if isinstance(x_hat, torch.Tensor):
        x_hat = x_hat.detach().numpy()
    if isinstance(x_true, torch.Tensor):
        x_true = x_true.detach().numpy()

    # I bits
    ber_I = np.mean((np.real(x_hat) > 0) != (np.real(x_true) > 0))
    # Q bits
    ber_Q = np.mean((np.imag(x_hat) > 0) != (np.imag(x_true) > 0))
    return (ber_I + ber_Q) / 2


# Test simulator output
print("=" * 60)
print("DEBUG: GA-BV-Net Model Flow")
print("=" * 60)

# Generate test data
cfg = SimConfig(
    snr_db=30.0,
    enable_pa=False,
    enable_pn=False,
    enable_channel=True,
    enable_quantization=False,  # No quantization for clarity
    phi0_random=False,
    coarse_acquisition_error_samples=0.0,
)
data = simulate_batch(cfg, batch_size=4, seed=42)

# Convert to torch
y = torch.from_numpy(data['y_raw']).to(torch.cfloat)
x_true = torch.from_numpy(data['x_true']).to(torch.cfloat)
theta_true = torch.from_numpy(data['theta_true']).float()

print(f"\n[Data Info]")
print(f"  y shape: {y.shape}")
print(f"  x_true shape: {x_true.shape}")
print(f"  theta_true[0]: {theta_true[0].tolist()}")
print(
    f"  theta_true format: [tau_res, v, a] = [{theta_true[0, 0]:.2e}s, {theta_true[0, 1]:.0f}m/s, {theta_true[0, 2]:.0f}m/s²]")

# Create PhysicsEncoder
from gabv_net_model import GABVConfig, PhysicsEncoder

pcfg = GABVConfig()
phys_enc = PhysicsEncoder(pcfg)

print(f"\n[Test 1: PhysicsEncoder.adjoint_operator]")
# If theta is exact, adjoint should recover x
x_recovered = phys_enc.adjoint_operator(y, theta_true)

ber_recovered = compute_ber(x_recovered, x_true)
print(f"  BER after adjoint (exact theta): {ber_recovered:.4f}")

if ber_recovered < 0.01:
    print("  ✓ PhysicsEncoder.adjoint works correctly!")
else:
    print("  ✗ PhysicsEncoder.adjoint has a problem!")

    # Debug: check Doppler phase
    v = theta_true[:, 1:2]
    a = theta_true[:, 2:3]
    p_t = phys_enc.compute_doppler_phase(v, a)

    # Manual derotation
    y_derotated_manual = y * torch.conj(p_t)
    ber_manual = compute_ber(y_derotated_manual, x_true)
    print(f"  BER after manual Doppler removal: {ber_manual:.4f}")

print(f"\n[Test 2: Forward then Adjoint]")
# Forward: x -> y_pred
# Then Adjoint: y_pred -> x_recovered
y_pred = phys_enc.forward_operator(x_true, theta_true)
x_roundtrip = phys_enc.adjoint_operator(y_pred, theta_true)

roundtrip_error = torch.mean(torch.abs(x_true - x_roundtrip) ** 2).item()
print(f"  Round-trip MSE: {roundtrip_error:.6f}")
if roundtrip_error < 0.001:
    print("  ✓ Forward-Adjoint is consistent!")
else:
    print("  ✗ Forward-Adjoint mismatch!")

print(f"\n[Test 3: Model Forward Pass]")
from gabv_net_model import GABVNet, GABVConfig

model_cfg = GABVConfig(enable_theta_update=False)
model = GABVNet(model_cfg)
model.eval()

# Prepare batch
meta = torch.zeros(4, model_cfg.meta_dim)
batch = {
    'y_q': y,
    'x_true': x_true,
    'theta_init': theta_true,
    'meta': meta,
}

with torch.no_grad():
    outputs = model(batch)

x_hat = outputs['x_hat']
ber_model = compute_ber(x_hat, x_true)
print(f"  BER from model: {ber_model:.4f}")

if ber_model < 0.2:
    print("  ✓ Model can estimate symbols!")
else:
    print("  ✗ Model fails to estimate symbols!")

    # Debug: check intermediate values
    print(f"\n  [Debugging model internals]")
    print(f"  x_hat[0,:5]: {x_hat[0, :5].tolist()}")
    print(f"  x_true[0,:5]: {x_true[0, :5].tolist()}")

    # Check if PN tracker is the problem
    y_derotated, phi_est = model.pn_tracker(y, meta, x_true[:, :64])
    ber_after_pn = compute_ber(y_derotated, x_true)
    print(f"  BER after PN tracker only: {ber_after_pn:.4f}")
    print(f"  phi_est mean: {phi_est.detach().mean().item():.4f}")

    # Check after adjoint
    z = phys_enc.adjoint_operator(y_derotated.detach(), theta_true)
    ber_after_adj = compute_ber(z, x_true)
    print(f"  BER after adjoint (post-PN): {ber_after_adj:.4f}")

print("\n" + "=" * 60)
print("DEBUG COMPLETE")
print("=" * 60)