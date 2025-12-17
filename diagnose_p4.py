"""
diagnose_complete.py - 完整诊断脚本

检查：
1. 模块导入
2. 场景配置和 Γ_eff 值
3. 模型加载和架构
4. SNR 转换是否正确
5. 模型前向传播
6. BER 计算

Usage:
    python diagnose_complete.py
"""

import numpy as np
import torch
import sys
import os
import glob
import inspect

print("=" * 70)
print("THz-ISAC Complete Diagnostic")
print("=" * 70)

# =============================================================================
# 1. Module Import Check
# =============================================================================
print("\n[1/6] Module Import Check")
print("-" * 50)

try:
    from thz_isac_world import SimConfig, simulate_batch
    print("  ✓ thz_isac_world imported")
except ImportError as e:
    print(f"  ✗ thz_isac_world failed: {e}")
    sys.exit(1)

try:
    from gabv_net_model import GABVNet, GABVConfig
    HAS_MODEL = True
    print("  ✓ gabv_net_model imported")
except ImportError as e:
    print(f"  ✗ gabv_net_model failed: {e}")
    HAS_MODEL = False

# =============================================================================
# 2. Scene Configuration Check
# =============================================================================
print("\n[2/6] Scene Configuration Check")
print("-" * 50)

scenes = [
    ("S4_ideal", {"enable_pa": False, "enable_pn": False, "enable_quantization": False}),
    ("S1_full_hw", {"enable_pa": True, "enable_pn": True, "enable_quantization": True}),
    ("S2_pa_only", {"enable_pa": True, "enable_pn": False, "enable_quantization": False}),
    ("S3_pn_only", {"enable_pa": False, "enable_pn": True, "enable_quantization": True}),
]

print(f"  {'Scene':<15} {'Γ_eff (dB)':<12} {'χ':<10} {'Status'}")
print(f"  {'-'*15} {'-'*12} {'-'*10} {'-'*10}")

scene_results = {}
for name, params in scenes:
    cfg = SimConfig()
    for k, v in params.items():
        setattr(cfg, k, v)
    cfg.snr_db = 15.0
    cfg.ibo_dB = 3.0
    cfg.pn_linewidth = 100e3

    data = simulate_batch(cfg, batch_size=32, seed=42)
    gamma_eff = data['meta']['gamma_eff']
    chi = data['meta']['chi']
    gamma_db = 10 * np.log10(gamma_eff + 1e-12)

    scene_results[name] = {'gamma_eff': gamma_eff, 'gamma_db': gamma_db, 'chi': chi}

    if name == "S4_ideal" and gamma_eff > 1e8:
        status = "✓ OK (ideal)"
    elif name != "S4_ideal" and gamma_eff < 1e8:
        status = "✓ OK (hw)"
    elif name != "S4_ideal" and gamma_eff > 1e8:
        status = "⚠️ WRONG!"
    else:
        status = "?"

    print(f"  {name:<15} {gamma_db:<12.1f} {chi:<10.4f} {status}")

# =============================================================================
# 3. Model Loading Check
# =============================================================================
print("\n[3/6] Model Loading Check")
print("-" * 50)

checkpoints = glob.glob("results/checkpoints/Stage*/final.pth")
if not checkpoints:
    print("  ✗ No checkpoint found in results/checkpoints/Stage*/")
    print("    Run: python train_gabv_net.py --curriculum --steps 1000")
    model = None
else:
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"  Found checkpoint: {latest_ckpt}")

    if HAS_MODEL:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"  Device: {device}")

        try:
            checkpoint = torch.load(latest_ckpt, map_location=device)
            print(f"  Checkpoint keys: {list(checkpoint.keys())}")

            if 'config' in checkpoint:
                print(f"  Config: {checkpoint['config']}")

            n_layers = checkpoint['config'].get('n_layers', 8) if 'config' in checkpoint else 8
            cfg = GABVConfig(n_layers=n_layers)
            model = GABVNet(cfg)
            model.load_state_dict(checkpoint['model_state'])
            model.to(device)
            model.eval()
            print(f"  ✓ Model loaded (n_layers={n_layers})")
        except Exception as e:
            print(f"  ✗ Model loading failed: {e}")
            model = None
    else:
        model = None

# =============================================================================
# 4. SNR Conversion Check (CRITICAL!)
# =============================================================================
print("\n[4/6] SNR Conversion Check (CRITICAL)")
print("-" * 50)

if HAS_MODEL:
    # 检查模型代码中的 SNR 处理
    print("  Checking GABVNet.forward() for SNR handling...")

    source_code = inspect.getsource(GABVNet.forward)

    # 检查是否有正确的 SNR 转换
    has_snr_denorm = "snr_db_norm * 15.0 + 15.0" in source_code or \
                     "snr_db_norm * 15 + 15" in source_code
    has_snr_linear_conv = "10 **" in source_code and "snr_db" in source_code

    if has_snr_denorm and has_snr_linear_conv:
        print("  ✓ SNR denormalization found")
        print("  ✓ SNR dB-to-linear conversion found")
    else:
        print("  ⚠️ WARNING: SNR conversion may be incorrect!")
        print("  ")
        print("  Looking for 'snr = meta[:, 0:1]' pattern...")

        if "snr = meta[:, 0:1]" in source_code:
            print("  🔴 CRITICAL BUG FOUND!")
            print("     Model uses normalized SNR directly as linear SNR!")
            print("     This causes lambda_reg = 1e6 instead of ~0.03")
            print("     ")
            print("  FIX NEEDED in gabv_net_model.py:")
            print("  Replace:")
            print("      snr = meta[:, 0:1]")
            print("  With:")
            print("      snr_db_norm = meta[:, 0:1]")
            print("      snr_db = snr_db_norm * 15.0 + 15.0")
            print("      snr = 10 ** (snr_db / 10.0)")

    # 模拟 SNR 转换
    print("\n  SNR Conversion Simulation:")
    for snr_db in [0, 10, 15, 20, 30]:
        snr_db_norm = (snr_db - 15.0) / 15.0

        # 错误的方式
        lambda_wrong = 1.0 / (snr_db_norm + 1e-6)

        # 正确的方式
        snr_db_recovered = snr_db_norm * 15.0 + 15.0
        snr_linear = 10 ** (snr_db_recovered / 10.0)
        lambda_correct = 1.0 / (snr_linear + 1e-6)

        print(f"    SNR={snr_db}dB: norm={snr_db_norm:.2f}, λ_wrong={lambda_wrong:.2e}, λ_correct={lambda_correct:.4f}")

# =============================================================================
# 5. Forward Pass Test
# =============================================================================
print("\n[5/6] Forward Pass Test")
print("-" * 50)

if model is not None:
    # 使用 S1_full_hw 场景
    sim_cfg = SimConfig()
    sim_cfg.snr_db = 15.0
    sim_cfg.enable_pa = True
    sim_cfg.enable_pn = True
    sim_cfg.enable_quantization = True
    sim_cfg.ibo_dB = 3.0
    sim_cfg.pn_linewidth = 100e3

    print(f"  Testing with S1_full_hw at SNR=15dB...")

    data = simulate_batch(sim_cfg, batch_size=32, seed=42)
    meta = data['meta']

    print(f"  Simulation output:")
    print(f"    Γ_eff = {10*np.log10(meta['gamma_eff']+1e-12):.1f} dB")
    print(f"    χ = {meta['chi']:.4f}")

    # 构造 meta 特征
    snr_db_center, snr_db_scale = 15.0, 15.0
    gamma_eff_db_center, gamma_eff_db_scale = 10.0, 20.0
    sigma_eta_scale = 0.1
    pn_linewidth_scale = 1e6
    ibo_db_center, ibo_db_scale = 3.0, 3.0

    snr_db_norm = (sim_cfg.snr_db - snr_db_center) / snr_db_scale
    gamma_eff_db = 10 * np.log10(meta['gamma_eff'] + 1e-12)
    gamma_eff_db_norm = (gamma_eff_db - gamma_eff_db_center) / gamma_eff_db_scale
    chi_raw = meta['chi']
    sigma_eta_norm = meta.get('sigma_eta', 0.0) / sigma_eta_scale
    pn_linewidth_norm = np.log10(sim_cfg.pn_linewidth + 1) / np.log10(pn_linewidth_scale)
    ibo_db_norm = (sim_cfg.ibo_dB - ibo_db_center) / ibo_db_scale

    print(f"\n  Meta features:")
    print(f"    [0] snr_db_norm = {snr_db_norm:.4f}")
    print(f"    [1] gamma_eff_db_norm = {gamma_eff_db_norm:.4f}")
    print(f"    [2] chi = {chi_raw:.4f}")
    print(f"    [3] sigma_eta_norm = {sigma_eta_norm:.4f}")
    print(f"    [4] pn_linewidth_norm = {pn_linewidth_norm:.4f}")
    print(f"    [5] ibo_db_norm = {ibo_db_norm:.4f}")

    features = torch.tensor([
        snr_db_norm, gamma_eff_db_norm, chi_raw,
        sigma_eta_norm, pn_linewidth_norm, ibo_db_norm
    ], dtype=torch.float32)
    meta_t = features.unsqueeze(0).expand(32, -1).clone().to(device)

    y_q_t = torch.from_numpy(data['y_q']).cfloat().to(device)
    x_true_t = torch.from_numpy(data['x_true']).cfloat().to(device)
    theta_true_t = torch.from_numpy(data['theta_true']).float().to(device)
    theta_init_t = theta_true_t + torch.randn_like(theta_true_t) * torch.tensor([100., 10., 0.5], device=device)

    batch = {
        'y_q': y_q_t,
        'x_true': x_true_t,
        'theta_init': theta_init_t,
        'meta': meta_t
    }

    try:
        with torch.no_grad():
            outputs = model(batch)

        x_hat = outputs['x_hat'].cpu().numpy()
        x_true = data['x_true']

        print(f"\n  Model output:")
        print(f"    x_hat shape: {x_hat.shape}")
        print(f"    x_hat mean abs: {np.mean(np.abs(x_hat)):.6e}")
        print(f"    x_true mean abs: {np.mean(np.abs(x_true)):.6e}")
        print(f"    Ratio x_hat/x_true: {np.mean(np.abs(x_hat))/np.mean(np.abs(x_true)):.4f}")

        if np.mean(np.abs(x_hat)) < 1e-6:
            print(f"\n  🔴 x_hat ≈ 0! This indicates the SNR bug!")

    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        import traceback
        traceback.print_exc()

# =============================================================================
# 6. BER Calculation Test
# =============================================================================
print("\n[6/6] BER Calculation Test")
print("-" * 50)

if model is not None and 'x_hat' in dir():
    # 计算 BER
    bit_I_true = (np.real(x_true) > 0).astype(int)
    bit_Q_true = (np.imag(x_true) > 0).astype(int)
    bit_I_hat = (np.real(x_hat) > 0).astype(int)
    bit_Q_hat = (np.imag(x_hat) > 0).astype(int)

    errors_I = np.sum(bit_I_true != bit_I_hat)
    errors_Q = np.sum(bit_Q_true != bit_Q_hat)
    total_bits = 2 * x_true.size
    ber = (errors_I + errors_Q) / total_bits

    print(f"  BER Calculation:")
    print(f"    I errors: {errors_I}/{x_true.size} ({100*errors_I/x_true.size:.1f}%)")
    print(f"    Q errors: {errors_Q}/{x_true.size} ({100*errors_Q/x_true.size:.1f}%)")
    print(f"    Total BER: {ber:.4f}")

    if ber > 0.4:
        print(f"\n  🔴 BER ≈ 0.5 indicates model is not working!")
        print(f"     Likely cause: SNR bug in gabv_net_model.py")

        # 相关性检查
        corr_real = np.corrcoef(np.real(x_hat).flatten(), np.real(x_true).flatten())[0,1]
        corr_imag = np.corrcoef(np.imag(x_hat).flatten(), np.imag(x_true).flatten())[0,1]
        print(f"\n  Correlation check:")
        print(f"    Real correlation: {corr_real:.4f}")
        print(f"    Imag correlation: {corr_imag:.4f}")

        if abs(corr_real) < 0.1 and abs(corr_imag) < 0.1:
            print(f"    ⚠️ No correlation - model output is essentially random")
    else:
        print(f"\n  ✓ BER looks reasonable!")

# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 70)
print("DIAGNOSTIC SUMMARY")
print("=" * 70)

issues = []

# Check for SNR bug
if HAS_MODEL:
    source_code = inspect.getsource(GABVNet.forward)
    if "snr = meta[:, 0:1]" in source_code:
        issues.append("🔴 CRITICAL: SNR bug in gabv_net_model.py - must fix and retrain!")

# Check for BER
if model is not None and 'ber' in dir() and ber > 0.4:
    issues.append("🔴 CRITICAL: BER ≈ 0.5 - model is not working")

# Check Γ_eff
if scene_results['S1_full_hw']['gamma_eff'] > 1e8:
    issues.append("⚠️ S1_full_hw has Γ_eff = 1e9 - scene config may be wrong")

if issues:
    print("\nISSUES FOUND:")
    for issue in issues:
        print(f"  {issue}")

    print("\n" + "-" * 50)
    print("RECOMMENDED ACTIONS:")
    print("-" * 50)
    print("""
1. Download the fixed gabv_net_model.py:
   - Replace your current gabv_net_model.py with gabv_net_model_fixed.py
   
2. Delete old checkpoints:
   rm -rf results/checkpoints/Stage*
   
3. Retrain the model:
   python train_gabv_net.py --curriculum --steps 1000
   
4. Run this diagnostic again to verify the fix

5. Then run the experiment:
   python run_p4_all_scenes.py --ckpt results/checkpoints/Stage3_xxx/final.pth
""")
else:
    print("\n✓ No critical issues found!")
    print("\nYou can run:")
    print("  python run_p4_all_scenes.py --ckpt results/checkpoints/Stage3_xxx/final.pth")

print("\n" + "=" * 70)