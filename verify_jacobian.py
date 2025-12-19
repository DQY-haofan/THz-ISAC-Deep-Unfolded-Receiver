#!/usr/bin/env python3
"""
verify_jacobian.py - Numerical gradient check for Jacobian direction

Expert 2's diagnostic: Compare analytical Jacobian vs finite difference
to verify the gradient direction is correct.

Expected result:
- Correlation (Real) ≈ +1.0 → Jacobian is CORRECT
- Correlation (Real) ≈ -1.0 → Jacobian sign is FLIPPED
- Correlation (Real) ≈ 0.0 → Jacobian is WRONG (f_grid error?)
"""

import torch
import torch.fft
import math


def test_jacobian_direction():
    """Test if analytical Jacobian matches finite difference."""
    print("=" * 70)
    print("JACOBIAN DIRECTION VERIFICATION")
    print("=" * 70)

    # Config
    N = 1024
    fs = 10e9
    Ts = 1 / fs

    # Correct frequency grid (must use fftfreq, NOT linspace!)
    f_grid = torch.fft.fftfreq(N, d=Ts)

    print(f"\n[Config]")
    print(f"  N = {N}, fs = {fs / 1e9} GHz, Ts = {Ts * 1e12:.1f} ps")
    print(f"  f_grid range: [{f_grid.min() / 1e9:.2f}, {f_grid.max() / 1e9:.2f}] GHz")

    # Generate QPSK signal
    torch.manual_seed(42)
    x_real = torch.sign(torch.randn(1, N))
    x_imag = torch.sign(torch.randn(1, N))
    x = (x_real + 1j * x_imag) / math.sqrt(2)
    X = torch.fft.fft(x)

    # Forward operator
    def forward(tau):
        H_D = torch.exp(-1j * 2 * math.pi * f_grid * tau)
        y = torch.fft.ifft(X * H_D, dim=1)
        return y

    # Analytical Jacobian
    def jacobian_analytical(tau):
        H_D = torch.exp(-1j * 2 * math.pi * f_grid * tau)
        # d/dτ[exp(-j2πfτ)] = -j2πf × exp(-j2πfτ)
        dH_dtau = -1j * 2 * math.pi * f_grid * H_D
        dY = X * dH_dtau
        dy = torch.fft.ifft(dY, dim=1)
        return dy

    # Test scenario
    tau_true = 0.0
    tau_init = -0.5 * Ts  # -0.5 samples

    print(f"\n[Scenario]")
    print(f"  tau_true = {tau_true / Ts:.3f} samples")
    print(f"  tau_init = {tau_init / Ts:.3f} samples")
    print(f"  tau_error = {(tau_init - tau_true) / Ts:.3f} samples")

    # Compute
    y_true = forward(tau_true)
    y_pred = forward(tau_init)
    residual = y_true - y_pred

    # Analytical Jacobian
    dy_dtau_ana = jacobian_analytical(tau_init)

    # Numerical Jacobian (finite difference)
    epsilon = 1e-14  # Very small for numerical derivative
    y_plus = forward(tau_init + epsilon)
    dy_dtau_num = (y_plus - y_pred) / epsilon

    print(f"\n[Test 1: Jacobian vs Finite Difference]")

    # Correlation check
    dot_prod = torch.sum(torch.conj(dy_dtau_ana) * dy_dtau_num)
    norm_ana = torch.sqrt(torch.sum(torch.abs(dy_dtau_ana) ** 2))
    norm_num = torch.sqrt(torch.sum(torch.abs(dy_dtau_num) ** 2))
    correlation = dot_prod / (norm_ana * norm_num + 1e-12)

    print(f"  Correlation (Real): {correlation.real.item():.6f} (expect +1.0)")
    print(f"  Correlation (Imag): {correlation.imag.item():.6f} (expect 0.0)")

    if correlation.real > 0.99:
        print("  ✓ Analytical Jacobian matches Finite Difference!")
    elif correlation.real < -0.5:
        print("  ✗ CRITICAL: Jacobian SIGN is FLIPPED!")
    else:
        print("  ✗ CRITICAL: Jacobian is WRONG (f_grid mismatch?)")

    # Test 2: Score direction
    print(f"\n[Test 2: Score Direction]")

    # Score with conjugate (correct)
    score_correct = torch.sum(torch.conj(dy_dtau_ana) * residual).real

    # Score without conjugate (wrong)
    score_wrong = torch.sum(dy_dtau_ana * residual).real

    print(f"  residual energy: {torch.sum(torch.abs(residual) ** 2).item():.2f}")
    print(f"  score (with conj):    {score_correct.item():.2e}")
    print(f"  score (without conj): {score_wrong.item():.2e}")

    # Check direction
    # If tau_init < tau_true, we want score > 0 (increase tau)
    # If tau_init > tau_true, we want score < 0 (decrease tau)
    expected_sign = 1 if tau_init < tau_true else -1
    actual_sign = 1 if score_correct > 0 else -1

    print(f"\n  tau_init ({tau_init / Ts:.3f}) {'<' if tau_init < tau_true else '>'} tau_true ({tau_true / Ts:.3f})")
    print(f"  Expected score sign: {'POSITIVE' if expected_sign > 0 else 'NEGATIVE'}")
    print(f"  Actual score sign:   {'POSITIVE' if actual_sign > 0 else 'NEGATIVE'}")

    if expected_sign == actual_sign:
        print("  ✓ Score direction is CORRECT!")
    else:
        print("  ✗ Score direction is WRONG!")

    # Test 3: Verify update improves residual
    print(f"\n[Test 3: Update Improves Residual?]")

    step_size = 0.1 * Ts  # Conservative step
    tau_updated = tau_init + step_size * (1 if score_correct > 0 else -1)

    y_updated = forward(tau_updated)
    residual_old = torch.sum(torch.abs(y_true - y_pred) ** 2).item()
    residual_new = torch.sum(torch.abs(y_true - y_updated) ** 2).item()

    print(f"  tau_init:    {tau_init / Ts:.4f} samples")
    print(f"  tau_updated: {tau_updated / Ts:.4f} samples")
    print(f"  tau_true:    {tau_true / Ts:.4f} samples")
    print(f"  residual_old: {residual_old:.2f}")
    print(f"  residual_new: {residual_new:.2f}")

    if residual_new < residual_old:
        print("  ✓ Update REDUCES residual!")
    else:
        print("  ✗ Update INCREASES residual!")

    # Summary
    print(f"\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    all_pass = (
            correlation.real > 0.99 and
            expected_sign == actual_sign and
            residual_new < residual_old
    )

    if all_pass:
        print("  ✓ All tests PASSED - Jacobian and score are correct!")
        print("  → Problem is likely in phi0 alignment or domain mismatch")
    else:
        print("  ✗ Some tests FAILED - Check Jacobian implementation")


def test_with_phase_offset():
    """Test score direction when phi0 is present but not accounted for."""
    print("\n" + "=" * 70)
    print("PHASE OFFSET IMPACT TEST")
    print("=" * 70)

    N = 1024
    fs = 10e9
    Ts = 1 / fs
    f_grid = torch.fft.fftfreq(N, d=Ts)

    torch.manual_seed(42)
    x = (torch.sign(torch.randn(1, N)) + 1j * torch.sign(torch.randn(1, N))) / math.sqrt(2)
    X = torch.fft.fft(x)

    def forward(tau, phi0=None):
        H_D = torch.exp(-1j * 2 * math.pi * f_grid * tau)
        y = torch.fft.ifft(X * H_D, dim=1)
        if phi0 is not None:
            return y * torch.exp(1j * phi0)  # Apply constant phase
        return y

    def jacobian(tau):
        H_D = torch.exp(-1j * 2 * math.pi * f_grid * tau)
        dH_dtau = -1j * 2 * math.pi * f_grid * H_D
        dy = torch.fft.ifft(X * dH_dtau, dim=1)
        return dy  # Note: No phi0 in Jacobian!

    tau_true = 0.0
    tau_init = -0.5 * Ts  # -0.5 samples
    phi0 = torch.tensor(0.5)  # Random phase offset (as tensor)

    print(f"\n[Scenario with phi0]")
    print(f"  tau_init = {tau_init / Ts:.3f} samples")
    print(f"  phi0 = {phi0.item():.3f} rad ({phi0.item() * 180 / math.pi:.1f}°)")

    # Observation includes phi0
    y_obs = forward(tau_true, phi0)

    # Prediction does NOT include phi0 (common bug!)
    y_pred_wrong = forward(tau_init, phi0=None)  # No phi0

    # Prediction WITH phi0 (correct)
    y_pred_correct = forward(tau_init, phi0)

    # Jacobian (doesn't know about phi0)
    dy_dtau = jacobian(tau_init)

    # Score with wrong prediction (phi0 mismatch)
    residual_wrong = y_obs - y_pred_wrong
    score_wrong = torch.sum(torch.conj(dy_dtau) * residual_wrong).real

    # Score with correct prediction (phi0 aligned)
    residual_correct = y_obs - y_pred_correct
    score_correct = torch.sum(torch.conj(dy_dtau) * residual_correct).real

    print(f"\n[Score Comparison]")
    print(f"  Score (phi0 mismatch): {score_wrong.item():.2e}")
    print(f"  Score (phi0 aligned):  {score_correct.item():.2e}")

    expected_sign = 1  # tau should increase

    print(f"\n[Direction Check]")
    if score_wrong > 0 and score_correct > 0:
        print("  ✓ Both scores point in correct direction")
    elif score_wrong < 0 and score_correct > 0:
        print("  ✗ phi0 mismatch FLIPS the score direction!")
        print("  → This is likely your ROOT CAUSE!")
    else:
        print(f"  ? Unexpected: wrong={score_wrong.item():.2e}, correct={score_correct.item():.2e}")


if __name__ == "__main__":
    test_jacobian_direction()
    test_with_phase_offset()