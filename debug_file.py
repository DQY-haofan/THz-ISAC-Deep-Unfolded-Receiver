# 在项目根目录运行
# !/usr/bin/env python3
"""
diagnose_gabvnet.py - 完整诊断脚本
保存到你的项目根目录，然后运行: python diagnose_gabvnet.py
"""

import numpy as np
import torch
import sys

print("=" * 70)
print("GA-BV-Net 完整诊断")
print("=" * 70)

# === 1. 环境检查 ===
print("\n[1] 环境检查")
print("-" * 50)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# === 2. 模块导入检查 ===
print("\n[2] 模块导入")
print("-" * 50)
try:
    from thz_isac_world import SimConfig, simulate_batch
    from gabv_net_model import GABVNet, GABVConfig, create_gabv_model
    from baselines_receivers import detector_lmmse_bussgang_torch
    from run_p4_experiments import construct_meta_features

    print("✅ 所有模块导入成功")
except ImportError as e:
    print(f"❌ 导入失败: {e}")
    sys.exit(1)

# === 3. FIX 验证 ===
print("\n[3] FIX 验证")
print("-" * 50)

# FIX 11: Meta 构造检查
print("\n[FIX 11] Meta 构造 - enable_pn 检查:")
for enable_pn in [False, True]:
    test_meta = {
        'snr_db': 20.0,
        'gamma_eff': 1e6,
        'chi': 0.5,
        'sigma_eta': 0.01,
        'enable_pn': enable_pn,
        'pn_linewidth': 100e3,  # 即使有值，enable_pn=False 时应该被忽略
        'ibo_dB': 3.0
    }
    meta_t = construct_meta_features(test_meta, 1)
    pn_norm = meta_t[0, 4].item()
    status = "✅" if (not enable_pn and pn_norm < 0.01) or (enable_pn and pn_norm > 0.5) else "❌"
    print(f"  enable_pn={enable_pn}: pn_linewidth_norm = {pn_norm:.4f} {status}")

# FIX 10: g_PN 门控检查
print("\n[FIX 10] g_PN 门控检查:")
model = create_gabv_model(GABVConfig()).to(device)
model.eval()

for pn_norm_val in [0.0, 0.5]:
    batch = {
        'y_q': torch.randn(4, 1024, dtype=torch.cfloat).to(device),
        'meta': torch.zeros(4, 6).to(device),
        'theta_init': torch.tensor([[500e3, 7500.0, 10.0]]).expand(4, 3).float().to(device)
    }
    batch['meta'][:, 4] = pn_norm_val

    with torch.no_grad():
        outputs = model(batch)
    g_pn = outputs['layers'][-1]['gates']['g_PN'].mean().item()
    status = "✅" if (pn_norm_val < 0.1 and g_pn < 0.05) or (pn_norm_val >= 0.1 and g_pn > 0) else "❌"
    print(f"  pn_norm={pn_norm_val}: g_PN = {g_pn:.4f} {status}")

# === 4. 场景测试 ===
print("\n[4] 场景测试 (SNR=20dB)")
print("-" * 50)


def compute_ber(x_hat, x_true):
    bits_hat_r = (np.real(x_hat) < 0).astype(int)
    bits_hat_i = (np.imag(x_hat) < 0).astype(int)
    bits_true_r = (np.real(x_true) < 0).astype(int)
    bits_true_i = (np.imag(x_true) < 0).astype(int)
    return 0.5 * (np.mean(bits_hat_r != bits_true_r) + np.mean(bits_hat_i != bits_true_i))


scenarios = {
    'S4_ideal': {'enable_pa': False, 'enable_pn': False, 'enable_quantization': False},
    'S2_pa_only': {'enable_pa': True, 'ibo_dB': 3.0, 'enable_pn': False, 'enable_quantization': True},
    'S3_pn_only': {'enable_pa': False, 'enable_pn': True, 'pn_linewidth': 100e3, 'enable_quantization': True},
    'S1_full_hw': {'enable_pa': True, 'ibo_dB': 3.0, 'enable_pn': True, 'pn_linewidth': 100e3,
                   'enable_quantization': True},
}

print(f"\n{'场景':<12} {'BER_Net':<10} {'BER_LMMSE':<10} {'g_PN':<8} {'pn_norm':<8} {'状态'}")
print("-" * 60)

for name, cfg_override in scenarios.items():
    sim_cfg = SimConfig()
    sim_cfg.snr_db = 20.0
    for k, v in cfg_override.items():
        setattr(sim_cfg, k, v)

    # 生成数据
    data = simulate_batch(sim_cfg, batch_size=64, seed=42)
    meta = data['meta']

    # 构建 meta features
    meta_t = construct_meta_features(meta, 64).to(device)
    pn_norm = meta_t[0, 4].item()

    # 准备输入
    y_q_t = torch.from_numpy(data['y_q']).cfloat().to(device)
    x_true_t = torch.from_numpy(data['x_true']).cfloat().to(device)
    theta_true_t = torch.from_numpy(data['theta_true']).float().to(device)

    batch = {
        'y_q': y_q_t,
        'theta_init': theta_true_t.clone(),
        'meta': meta_t
    }

    # 运行模型
    with torch.no_grad():
        outputs = model(batch)

    x_hat = outputs['x_hat'].cpu().numpy()
    g_pn = outputs['layers'][-1]['gates']['g_PN'].mean().item()
    ber_net = compute_ber(x_hat, data['x_true'])

    # LMMSE baseline
    h_diag = torch.from_numpy(meta['h_diag']).cfloat().to(device)
    snr_lin = 10 ** (sim_cfg.snr_db / 10)
    sig_pwr = torch.mean(torch.abs(x_true_t) ** 2).item()
    noise_var = sig_pwr / snr_lin
    x_lmmse = detector_lmmse_bussgang_torch(y_q_t, h_diag, noise_var)
    ber_lmmse = compute_ber(x_lmmse.cpu().numpy(), data['x_true'])

    # 判断状态
    if name == 'S4_ideal':
        status = "✅" if ber_net < 0.01 else f"❌ 应该≈0"
    elif ber_net > 0.45:
        status = "❌ Phase问题"
    elif ber_net > ber_lmmse * 3:
        status = "⚠️ 需改进"
    else:
        status = "✅"

    print(f"{name:<12} {ber_net:<10.4f} {ber_lmmse:<10.4f} {g_pn:<8.4f} {pn_norm:<8.4f} {status}")

# === 5. Phase 诊断 (仅对 S4_ideal) ===
print("\n[5] S4_ideal Phase 诊断")
print("-" * 50)
sim_cfg = SimConfig()
sim_cfg.snr_db = 20.0
sim_cfg.enable_pa = False
sim_cfg.enable_pn = False
sim_cfg.enable_quantization = False

data = simulate_batch(sim_cfg, batch_size=64, seed=42)
meta_t = construct_meta_features(data['meta'], 64).to(device)

batch = {
    'y_q': torch.from_numpy(data['y_q']).cfloat().to(device),
    'theta_init': torch.from_numpy(data['theta_true']).float().to(device),
    'meta': meta_t
}

with torch.no_grad():
    outputs = model(batch)

x_hat = outputs['x_hat']
x_true = torch.from_numpy(data['x_true']).cfloat().to(device)
derotator = outputs['phi_hat']

# 相位分析
phase_diff = torch.angle(x_hat * torch.conj(x_true))
mean_phase = phase_diff.mean().item()
std_phase = phase_diff.std().item()
derotator_phase = torch.angle(derotator).mean().item()

print(f"  x_hat vs x_true 平均相位差: {np.degrees(mean_phase):.2f}°")
print(f"  相位差标准差: {np.degrees(std_phase):.2f}°")
print(f"  Derotator 平均相位: {np.degrees(derotator_phase):.2f}°")

if abs(mean_phase) > 0.1:  # > 5.7°
    print(f"  ⚠️ 存在固定相位偏移，可能是 Ghost Phase 问题")
else:
    print(f"  ✅ 相位正常")

# === 6. 总结 ===
print("\n" + "=" * 70)
print("诊断完成 - 请将以上输出发送给我进行分析")
print("=" * 70)