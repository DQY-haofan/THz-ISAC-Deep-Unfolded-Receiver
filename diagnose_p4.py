"""
诊断脚本：检查 P4 实验问题根源

运行方式：
    python diagnose_p4.py

将检测：
1. 模块导入
2. 不同场景的 Γ_eff 值
3. 模型加载
4. 前向传播和 BER 计算
5. Meta 特征对齐问题
"""

import numpy as np
import torch
import sys
import os

print("=" * 60)
print("P4 实验问题诊断")
print("=" * 60)

# 1. 检查模块导入
print("\n[1] 检查模块导入...")
try:
    from thz_isac_world import SimConfig, simulate_batch

    print("  ✓ thz_isac_world 导入成功")
except ImportError as e:
    print(f"  ✗ thz_isac_world 导入失败: {e}")
    sys.exit(1)

try:
    from gabv_net_model import GABVNet, GABVConfig

    print("  ✓ gabv_net_model 导入成功")
    HAS_MODEL = True
except ImportError as e:
    print(f"  ✗ gabv_net_model 导入失败: {e}")
    HAS_MODEL = False

# 2. 检查场景配置
print("\n[2] 测试不同场景的 Γ_eff...")

scenes = [
    ("S4_ideal", {"enable_pa": False, "enable_pn": False, "enable_quantization": False}),
    ("S1_full_hw", {"enable_pa": True, "enable_pn": True, "enable_quantization": True}),
]

for name, params in scenes:
    cfg = SimConfig()
    for k, v in params.items():
        setattr(cfg, k, v)
    cfg.snr_db = 10

    data = simulate_batch(cfg, batch_size=32, seed=42)
    gamma_eff = data['meta']['gamma_eff']
    chi = data['meta']['chi']

    gamma_db = 10 * np.log10(gamma_eff + 1e-12)
    print(f"  {name}: Γ_eff = {gamma_db:.1f} dB, χ = {chi:.4f}")

    if gamma_eff > 1e8:
        print(f"    ⚠️ Γ_eff = 1e9 表示这是理想场景（无硬件损伤）")

# 3. 检查模型加载
print("\n[3] 检查模型加载...")

import glob

checkpoints = glob.glob("results/checkpoints/Stage*/final.pth")
if not checkpoints:
    print("  ✗ 未找到 checkpoint 文件")
    print("  请先运行训练: python train_gabv_net.py --curriculum")
else:
    latest_ckpt = max(checkpoints, key=os.path.getctime)
    print(f"  找到 checkpoint: {latest_ckpt}")

    if HAS_MODEL:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            print(f"  ✓ 模型加载成功 (n_layers={n_layers})")

            # 4. 测试模型前向传播
            print("\n[4] 测试模型前向传播...")

            # 使用 S1_full_hw 场景测试
            sim_cfg = SimConfig()
            sim_cfg.snr_db = 15
            sim_cfg.enable_pa = True
            sim_cfg.ibo_dB = 3.0
            sim_cfg.enable_pn = True
            sim_cfg.pn_linewidth = 100e3
            sim_cfg.enable_quantization = True

            data = simulate_batch(sim_cfg, batch_size=32, seed=42)

            print(f"  场景: Full HW (PA+PN+1bit)")
            print(f"  SNR: {sim_cfg.snr_db} dB")
            print(f"  Γ_eff: {10 * np.log10(data['meta']['gamma_eff'] + 1e-12):.1f} dB")
            print(f"  χ: {data['meta']['chi']:.4f}")

            y_q_t = torch.from_numpy(data['y_q']).cfloat().to(device)
            x_true_t = torch.from_numpy(data['x_true']).cfloat().to(device)
            theta_true_t = torch.from_numpy(data['theta_true']).float().to(device)
            theta_init_t = theta_true_t + torch.randn_like(theta_true_t) * torch.tensor([100., 10., 0.5], device=device)

            # 构造 meta 特征 (使用与 train_gabv_net.py 完全一致的归一化)
            meta = data['meta']

            # 归一化常量 (必须与 TrainConfig 一致!)
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

            print(f"\n  [Meta 特征诊断]")
            print(f"    snr_db_norm: {snr_db_norm:.4f}")
            print(f"    gamma_eff_db_norm: {gamma_eff_db_norm:.4f} (原始: {gamma_eff_db:.1f} dB)")
            print(f"    chi_raw: {chi_raw:.4f}")
            print(f"    sigma_eta_norm: {sigma_eta_norm:.4f}")
            print(f"    pn_linewidth_norm: {pn_linewidth_norm:.4f}")
            print(f"    ibo_db_norm: {ibo_db_norm:.4f}")

            features = torch.tensor([
                snr_db_norm, gamma_eff_db_norm, chi_raw,
                sigma_eta_norm, pn_linewidth_norm, ibo_db_norm
            ], dtype=torch.float32)
            meta_t = features.unsqueeze(0).expand(32, -1).clone().to(device)

            batch = {
                'y_q': y_q_t,
                'x_true': x_true_t,
                'theta_init': theta_init_t,
                'meta': meta_t
            }

            with torch.no_grad():
                outputs = model(batch)

            print(f"\n  [模型输出诊断]")
            print(f"    Output keys: {list(outputs.keys())}")

            x_hat = outputs['x_hat'].cpu().numpy()
            x_true = data['x_true']

            print(f"    x_hat shape: {x_hat.shape}")
            print(f"    x_hat dtype: {x_hat.dtype}")
            print(f"    x_true shape: {x_true.shape}")

            print(f"\n    x_hat 前5个值: {x_hat[0, :5]}")
            print(f"    x_true 前5个值: {x_true[0, :5]}")

            print(f"\n    x_hat real 统计: mean={np.mean(np.real(x_hat)):.4f}, std={np.std(np.real(x_hat)):.4f}")
            print(f"    x_hat imag 统计: mean={np.mean(np.imag(x_hat)):.4f}, std={np.std(np.imag(x_hat)):.4f}")
            print(f"    x_true real 统计: mean={np.mean(np.real(x_true)):.4f}, std={np.std(np.real(x_true)):.4f}")
            print(f"    x_true imag 统计: mean={np.mean(np.imag(x_true)):.4f}, std={np.std(np.imag(x_true)):.4f}")

            # 计算 BER
            bit_I_true = (np.real(x_true) > 0).astype(int)
            bit_Q_true = (np.imag(x_true) > 0).astype(int)
            bit_I_hat = (np.real(x_hat) > 0).astype(int)
            bit_Q_hat = (np.imag(x_hat) > 0).astype(int)

            errors_I = np.sum(bit_I_true != bit_I_hat)
            errors_Q = np.sum(bit_Q_true != bit_Q_hat)
            total_bits = 2 * x_true.size
            ber = (errors_I + errors_Q) / total_bits

            print(f"\n  [BER 诊断]")
            print(f"    I 位错误: {errors_I}/{x_true.size} ({100 * errors_I / x_true.size:.1f}%)")
            print(f"    Q 位错误: {errors_Q}/{x_true.size} ({100 * errors_Q / x_true.size:.1f}%)")
            print(f"    总 BER: {ber:.4f}")

            if ber > 0.4:
                print("\n  🔴 BER ≈ 0.5 问题诊断:")

                # 检查 x_hat 是否全是某个常数
                unique_real = len(np.unique(np.sign(np.real(x_hat))))
                unique_imag = len(np.unique(np.sign(np.imag(x_hat))))
                print(f"    x_hat real 符号种类: {unique_real} (应为 2)")
                print(f"    x_hat imag 符号种类: {unique_imag} (应为 2)")

                if unique_real == 1 or unique_imag == 1:
                    print("    ⚠️ x_hat 输出全是同一个符号！模型没有正确解码")

                # 检查相关性
                corr_real = np.corrcoef(np.real(x_hat).flatten(), np.real(x_true).flatten())[0, 1]
                corr_imag = np.corrcoef(np.imag(x_hat).flatten(), np.imag(x_true).flatten())[0, 1]
                print(f"    Real 部分相关系数: {corr_real:.4f}")
                print(f"    Imag 部分相关系数: {corr_imag:.4f}")

                if abs(corr_real) < 0.1 and abs(corr_imag) < 0.1:
                    print("    ⚠️ x_hat 与 x_true 几乎无相关性")
                    print("    可能原因:")
                    print("    1. 模型未正确训练")
                    print("    2. Meta 特征归一化不匹配")
                    print("    3. 输入数据格式问题")

            elif ber < 0.1:
                print(f"\n  ✓ BER = {ber:.4f}，模型工作正常!")
            else:
                print(f"\n  ⚠️ BER = {ber:.4f}，性能一般")

        except Exception as e:
            print(f"  ✗ 模型测试失败: {e}")
            import traceback

            traceback.print_exc()

print("\n" + "=" * 60)
print("诊断完成")
print("=" * 60)