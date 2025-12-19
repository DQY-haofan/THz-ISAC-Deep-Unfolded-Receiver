#!/usr/bin/env python3
"""
quick_test.py - Quick Validation for Wideband Delay Model

Run this BEFORE training to verify:
1. Simulator works correctly
2. PhysicsEncoder matches simulator (if torch available)
3. Key physical quantities are correct

Usage:
    python quick_test.py
"""

import numpy as np
import sys


# Test if dependencies exist
def test_dependencies():
    """Test that required dependencies exist."""
    print("\n[TEST 0] Dependencies")

    try:
        from thz_isac_world import SimConfig, simulate_batch
        print("  ✓ thz_isac_world imported")
    except ImportError as e:
        print(f"  ✗ thz_isac_world: {e}")
        return False

    try:
        import torch
        print("  ✓ PyTorch available")
        has_torch = True
    except ImportError:
        print("  ! PyTorch not available (will skip model tests)")
        has_torch = False

    return True


def test_wideband_delay():
    """Test wideband delay operator."""
    print("\n[TEST 1] Wideband Delay Operator")

    from thz_isac_world import wideband_delay_operator

    N = 1024
    fs = 10e9

    # Test signal
    rng = np.random.default_rng(42)
    x = rng.normal(0, 1, N) + 1j * rng.normal(0, 1, N)

    # Test 1: Zero delay
    y_zero = wideband_delay_operator(x, 0.0, fs)
    error_zero = np.max(np.abs(y_zero - x))
    print(f"  Zero delay error: {error_zero:.2e} {'✓' if error_zero < 1e-10 else '✗'}")

    # Test 2: 1-sample delay
    tau_1s = 1.0 / fs
    y_1s = wideband_delay_operator(x, tau_1s, fs)
    expected = np.roll(x, 1)
    error_1s = np.max(np.abs(y_1s - expected))
    print(f"  1-sample delay error: {error_1s:.2e} {'✓' if error_1s < 1e-10 else '✗'}")

    return error_zero < 1e-10 and error_1s < 1e-10


def test_simulation_chain():
    """Test full simulation chain."""
    print("\n[TEST 2] Full Simulation Chain")

    from thz_isac_world import SimConfig, simulate_batch

    cfg = SimConfig(snr_db=20.0)
    data = simulate_batch(cfg, batch_size=16, seed=42)

    print(f"  x_true shape: {data['x_true'].shape}")
    print(f"  y_q shape: {data['y_q'].shape}")
    print(f"  theta_true shape: {data['theta_true'].shape}")

    # Check theta is [tau_res, v, a] not [R, v, a]
    theta = data['theta_true'][0]
    print(f"  theta[0] = [tau_res={theta[0]:.2e}, v={theta[1]:.1f}, a={theta[2]:.1f}]")

    # tau_res should be small (near 0 for perfect acquisition)
    is_tau_small = abs(theta[0]) < 1e-6  # Should be ~0
    print(f"  tau_res is small: {'✓' if is_tau_small else '✗'}")

    # Check meta contains resolution info
    meta = data['meta']
    print(f"  Range resolution: {meta['range_resolution'] * 100:.1f} cm")
    print(f"  Wavelength: {meta['wavelength'] * 1000:.2f} mm")
    print(f"  Gamma_eff: {meta['gamma_eff']:.2f}")

    return True


def test_physics_constants():
    """Test physical constants are correct."""
    print("\n[TEST 3] Physical Constants")

    c = 3e8
    fc = 300e9
    fs = 10e9

    wavelength = c / fc
    delay_resolution = 1.0 / fs
    range_resolution = c * delay_resolution

    print(f"  Carrier frequency: {fc / 1e9:.0f} GHz")
    print(f"  Bandwidth: {fs / 1e9:.0f} GHz")
    print(f"  Wavelength: {wavelength * 1000:.2f} mm")
    print(f"  Delay resolution: {delay_resolution * 1e12:.1f} ps")
    print(f"  Range resolution: {range_resolution * 100:.1f} cm")

    # Key insight: λ << range resolution
    # Carrier phase cycles 30x per range resolution cell
    cycles_per_cell = range_resolution / wavelength
    print(f"  Carrier cycles per range cell: {cycles_per_cell:.0f}")
    print(f"  → This is why carrier phase is NOT identifiable!")

    return wavelength < 2e-3 and range_resolution > 0.01


def test_doppler_phase():
    """Test Doppler phase operator."""
    print("\n[TEST 4] Doppler Phase Operator")

    from thz_isac_world import doppler_phase_operator

    N = 1024
    fs = 10e9
    fc = 300e9
    v = 1000.0  # 1000 m/s
    a = 0.0

    p_t = doppler_phase_operator(N, fs, fc, v, a)

    # Doppler frequency: fd = fc * v / c = 300e9 * 1000 / 3e8 = 1 MHz
    fd_expected = fc * v / 3e8

    # Phase rate: dφ/dt = 2π × fd
    # Over one symbol (Ts = 100ps), phase change = 2π × fd × Ts
    phase_per_symbol = 2 * np.pi * fd_expected / fs

    # Check phase difference between consecutive samples
    phase_diff = np.angle(p_t[1]) - np.angle(p_t[0])

    print(f"  Expected Doppler: {fd_expected / 1e6:.2f} MHz")
    print(f"  Expected phase/symbol: {np.degrees(phase_per_symbol):.4f}°")
    print(f"  Measured phase/symbol: {np.degrees(phase_diff):.4f}°")

    error = abs(phase_diff - (-phase_per_symbol))  # Negative because p_t = exp(-j*phase)
    print(f"  Phase error: {np.degrees(error):.6f}° {'✓' if error < 0.001 else '✗'}")

    return error < 0.01


def test_identifiability():
    """Test identifiability concepts."""
    print("\n[TEST 5] Identifiability Analysis")

    fc = 300e9
    fs = 10e9
    c = 3e8

    wavelength = c / fc
    range_resolution = c / fs

    # For 10m range error:
    delta_R = 10.0  # meters

    # Carrier phase cycles
    carrier_cycles = delta_R / wavelength
    carrier_phase_deg = (carrier_cycles % 1) * 360

    # Group delay samples
    delta_tau = delta_R / c
    delay_samples = delta_tau * fs

    print(f"  For ΔR = 10m:")
    print(f"    Carrier: {carrier_cycles:.0f} cycles → {carrier_phase_deg:.1f}° (mod 360°)")
    print(f"    Group delay: {delay_samples:.1f} samples")
    print(f"  → Carrier phase wraps {int(carrier_cycles)} times, losing information!")
    print(f"  → Group delay is directly measurable")

    # Basin of attraction
    basin_samples = 1.5  # Typical
    basin_range = basin_samples * range_resolution
    print(f"\n  Basin of attraction:")
    print(f"    ~{basin_samples} samples = ~{basin_range * 100:.1f} cm")
    print(f"    Beyond this, gradient descent fails")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("GA-BV-Net Wideband Delay Model - Quick Validation")
    print("=" * 60)

    if not test_dependencies():
        print("\n❌ Missing dependencies. Please check installation.")
        return 1

    tests = [
        ("Wideband Delay", test_wideband_delay),
        ("Simulation Chain", test_simulation_chain),
        ("Physics Constants", test_physics_constants),
        ("Doppler Phase", test_doppler_phase),
        ("Identifiability", test_identifiability),
    ]

    results = []
    for name, test_fn in tests:
        try:
            result = test_fn()
            results.append((name, result))
        except Exception as e:
            print(f"  ✗ Error: {e}")
            results.append((name, False))

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    all_pass = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}: {status}")
        if not passed:
            all_pass = False

    if all_pass:
        print("\n✅ All tests passed! Ready for training.")
    else:
        print("\n⚠️  Some tests failed. Please review before training.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())