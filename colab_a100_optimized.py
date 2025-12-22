# =============================================================================
# 🚀 GA-BV-Net A100 极限训练配置
# =============================================================================
#
# A100 优势：
# - 40GB/80GB 显存 → 超大 batch size
# - Tensor Core → 混合精度 2x 加速
# - 高带宽 → 更快数据传输
#
# 预计训练时间：10000 步 ≈ 10-15 分钟 (A100)
# =============================================================================

# %%
# =============================================================================
# CELL 1: 环境检查
# =============================================================================

import torch

print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    gpu_name = torch.cuda.get_device_name(0)
    gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"GPU: {gpu_name}")
    print(f"Memory: {gpu_mem:.1f} GB")

    # 检测 A100 类型
    if "A100" in gpu_name:
        if gpu_mem > 45:
            print("✅ A100-80GB detected!")
            A100_TYPE = "80GB"
        else:
            print("✅ A100-40GB detected!")
            A100_TYPE = "40GB"
    else:
        print(f"⚠️ Not A100, but {gpu_name}")
        A100_TYPE = "OTHER"

# %%
# =============================================================================
# CELL 2: A100 优化配置
# =============================================================================

# 根据 A100 类型选择配置
if A100_TYPE == "80GB":
    A100_CONFIG = {
        # === 超大 Batch Size ===
        "batch_size": 512,  # 80GB 可以用 512
        "grad_accumulation": 1,  # 不需要累积

        # === 训练步数 ===
        "n_steps": 15000,  # 更多步数

        # === 学习率 (大batch需要更高lr) ===
        "lr_main": 1e-3,  # 3x 提高
        "lr_denoiser": 3e-3,  # Denoiser 更激进
        "warmup_steps": 500,

        # === 模型容量 (可选扩大) ===
        "vamp_layers": 8,  # 更多层
        "denoiser_hidden": 512,  # 更宽
    }
elif A100_TYPE == "40GB":
    A100_CONFIG = {
        "batch_size": 256,  # 40GB 用 256
        "grad_accumulation": 2,  # 累积到有效 512
        "n_steps": 15000,
        "lr_main": 8e-4,
        "lr_denoiser": 2e-3,
        "warmup_steps": 500,
        "vamp_layers": 8,
        "denoiser_hidden": 384,
    }
else:
    # 其他 GPU 的保守配置
    A100_CONFIG = {
        "batch_size": 64,
        "grad_accumulation": 4,
        "n_steps": 10000,
        "lr_main": 3e-4,
        "lr_denoiser": 1e-3,
        "warmup_steps": 300,
        "vamp_layers": 6,
        "denoiser_hidden": 256,
    }

# 通用配置
A100_CONFIG.update({
    # === 混合精度 (关键加速!) ===
    "use_amp": True,  # 自动混合精度

    # === 损失权重 ===
    "denoiser_loss_weight": 5.0,
    "tau_loss_weight": 0.1,

    # === 课程学习 ===
    "snr_start": 28,  # 从更高 SNR 开始
    "snr_end": 8,  # 降到更低
    "curriculum_steps": 5000,

    # === 正则化 ===
    "weight_decay": 1e-4,
    "grad_clip": 5.0,
    "dropout": 0.1,

    # === 数据加载 ===
    "num_workers": 4,
    "pin_memory": True,

    # === 保存 ===
    "save_every": 2500,
    "log_every": 50,
})

print("\n" + "=" * 60)
print("🔧 A100 优化配置")
print("=" * 60)
for k, v in A100_CONFIG.items():
    print(f"  {k}: {v}")

# %%
# =============================================================================
# CELL 3: 导入和设备设置
# =============================================================================

import os
import sys
import time
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler

# 设置 CUDA 优化
torch.backends.cudnn.benchmark = True  # 自动寻找最优算法
torch.backends.cuda.matmul.allow_tf32 = True  # 启用 TF32
torch.backends.cudnn.allow_tf32 = True

device = torch.device("cuda")
print(f"\n✅ Using device: {device}")
print(f"  cuDNN benchmark: enabled")
print(f"  TF32: enabled")

# %%
# =============================================================================
# CELL 4: 导入项目模块
# =============================================================================

# 确保项目在路径中
# sys.path.insert(0, '/content/project')

from gabv_net_model import GABVNet, GABVNetConfig
from thz_isac_world import SimConfig, THzISACWorld

print("✅ 项目模块导入成功")

# %%
# =============================================================================
# CELL 5: 创建模型 (A100 优化版)
# =============================================================================

cfg = A100_CONFIG

# 模型配置 - 可以更大
model_config = GABVNetConfig(
    N=1024,
    pilot_len=128,
    vamp_layers=cfg['vamp_layers'],
    tau_gn_iters=3,
    denoiser_hidden=cfg['denoiser_hidden'],
)

model = GABVNet(model_config).to(device)

# 编译模型 (PyTorch 2.0+, 额外 10-30% 加速)
if hasattr(torch, 'compile'):
    print("🔥 使用 torch.compile() 加速...")
    model = torch.compile(model, mode="reduce-overhead")

# 统计参数
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\n📊 模型参数:")
print(f"  Total: {total_params:,}")
print(f"  Trainable: {trainable_params:,}")

# 加载现有 checkpoint (可选)
existing_ckpt = "results/checkpoints/Stage3_FullTrak_1766361144/final.pth"
if os.path.exists(existing_ckpt):
    print(f"\n📦 加载现有 checkpoint: {existing_ckpt}")
    ckpt = torch.load(existing_ckpt, map_location=device)
    # 处理可能的 compile 包装
    state_dict = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state_dict, strict=False)
    print("✅ 权重加载完成")

# %%
# =============================================================================
# CELL 6: 优化器设置 (A100 优化)
# =============================================================================

# 分组参数
denoiser_params = []
other_params = []

for name, param in model.named_parameters():
    if 'denoiser' in name.lower() or 'vamp' in name.lower():
        denoiser_params.append(param)
    else:
        other_params.append(param)

print(f"\n⚙️ 参数分组:")
print(f"  Denoiser: {sum(p.numel() for p in denoiser_params):,} params @ lr={cfg['lr_denoiser']}")
print(f"  Other: {sum(p.numel() for p in other_params):,} params @ lr={cfg['lr_main']}")

# 使用 AdamW with fused=True (A100 优化)
optimizer = optim.AdamW([
    {'params': other_params, 'lr': cfg['lr_main']},
    {'params': denoiser_params, 'lr': cfg['lr_denoiser']},
], weight_decay=cfg['weight_decay'], fused=True)  # fused=True 更快!


# 学习率调度
def lr_lambda(step):
    if step < cfg['warmup_steps']:
        return step / cfg['warmup_steps']
    progress = (step - cfg['warmup_steps']) / (cfg['n_steps'] - cfg['warmup_steps'])
    return 0.01 + 0.99 * 0.5 * (1 + np.cos(np.pi * progress))


scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

# 混合精度 Scaler
scaler = GradScaler() if cfg['use_amp'] else None

print(f"\n✅ 优化器设置完成")
print(f"  Mixed Precision (AMP): {'Enabled' if cfg['use_amp'] else 'Disabled'}")

# %%
# =============================================================================
# CELL 7: 数据生成器 (优化版)
# =============================================================================

sim_config = SimConfig(
    fc=300e9,
    fs=10e9,
    N=1024,
    enable_pa=True,
    enable_pn=True,
    enable_quantization=True,
    snr_db=20.0,
)

world = THzISACWorld(sim_config)


def generate_batch_cuda(batch_size, snr_db):
    """直接生成 CUDA tensor 的批次"""
    sim_config.snr_db = snr_db
    batch = world.generate_batch(batch_size)

    return {
        'y': torch.from_numpy(batch['y']).to(device, non_blocking=True),
        'x': torch.from_numpy(batch['x']).to(device, non_blocking=True),
        'tau': torch.from_numpy(batch['tau']).to(device, non_blocking=True),
        'pilots': torch.from_numpy(batch['pilots']).to(device, non_blocking=True),
        'pilot_idx': torch.from_numpy(batch['pilot_idx']).to(device, non_blocking=True),
    }


print("✅ 数据生成器就绪")

# %%
# =============================================================================
# CELL 8: 训练循环 (A100 极限版)
# =============================================================================

print("\n" + "=" * 60)
print("🚀 A100 极限训练开始")
print("=" * 60)
print(f"  Batch size: {cfg['batch_size']}")
print(f"  Grad accumulation: {cfg['grad_accumulation']}")
print(f"  Effective batch: {cfg['batch_size'] * cfg['grad_accumulation']}")
print(f"  Total steps: {cfg['n_steps']}")
print(f"  Mixed Precision: {cfg['use_amp']}")
print("=" * 60)

# 输出目录
output_dir = Path("results/a100_intensive")
output_dir.mkdir(parents=True, exist_ok=True)

# 训练记录
history = {
    'step': [], 'loss': [], 'det_loss': [], 'denoiser_loss': [],
    'lr': [], 'snr': [], 'gpu_mem': [], 'throughput': [],
}

best_loss = float('inf')
start_time = time.time()
tokens_processed = 0

model.train()

# 预热 GPU
print("\n🔥 预热 GPU...")
for _ in range(3):
    dummy_batch = generate_batch_cuda(cfg['batch_size'], 20.0)
    with autocast(enabled=cfg['use_amp']):
        _ = model(dummy_batch['y'], dummy_batch['pilots'], dummy_batch['pilot_idx'])
torch.cuda.synchronize()
print("✅ GPU 预热完成")

# 主训练循环
pbar = tqdm(range(1, cfg['n_steps'] + 1), desc="Training", ncols=120)

for step in pbar:
    step_start = time.time()

    # === 课程学习 SNR ===
    if step < cfg['curriculum_steps']:
        progress = step / cfg['curriculum_steps']
        current_snr = cfg['snr_start'] - progress * (cfg['snr_start'] - cfg['snr_end'])
    else:
        current_snr = np.random.uniform(cfg['snr_end'], cfg['snr_start'])

    # === 梯度累积循环 ===
    optimizer.zero_grad(set_to_none=True)  # 更高效的清零

    accumulated_loss = 0.0
    accumulated_det = 0.0
    accumulated_den = 0.0

    for micro_step in range(cfg['grad_accumulation']):
        # 生成数据
        batch = generate_batch_cuda(cfg['batch_size'], current_snr)

        # 混合精度前向
        with autocast(enabled=cfg['use_amp']):
            outputs = model(batch['y'], batch['pilots'], batch['pilot_idx'])

            # 检测损失
            det_loss = nn.functional.mse_loss(outputs['x_hat'], batch['x'])

            # τ 损失
            tau_loss = nn.functional.mse_loss(outputs['tau_hat'], batch['tau'])

            # Denoiser 中间层损失
            denoiser_loss = torch.tensor(0.0, device=device)
            if 'intermediate_x' in outputs and outputs['intermediate_x']:
                for i, x_int in enumerate(outputs['intermediate_x']):
                    weight = 0.5 ** (len(outputs['intermediate_x']) - i - 1)
                    denoiser_loss = denoiser_loss + weight * nn.functional.mse_loss(x_int, batch['x'])

            # 总损失 (除以累积步数)
            loss = (det_loss + cfg['tau_loss_weight'] * tau_loss +
                    cfg['denoiser_loss_weight'] * denoiser_loss) / cfg['grad_accumulation']

        # 反向传播 (混合精度)
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        accumulated_loss += loss.item() * cfg['grad_accumulation']
        accumulated_det += det_loss.item()
        accumulated_den += denoiser_loss.item() if isinstance(denoiser_loss, torch.Tensor) else denoiser_loss

    # === 优化器步进 ===
    if scaler:
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        scaler.step(optimizer)
        scaler.update()
    else:
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg['grad_clip'])
        optimizer.step()

    scheduler.step()

    # === 统计 ===
    step_time = time.time() - step_start
    tokens_processed += cfg['batch_size'] * cfg['grad_accumulation'] * 1024
    throughput = cfg['batch_size'] * cfg['grad_accumulation'] / step_time
    gpu_mem = torch.cuda.max_memory_allocated() / 1e9

    # 记录
    history['step'].append(step)
    history['loss'].append(accumulated_loss)
    history['det_loss'].append(accumulated_det / cfg['grad_accumulation'])
    history['denoiser_loss'].append(accumulated_den / cfg['grad_accumulation'])
    history['lr'].append(scheduler.get_last_lr()[1])
    history['snr'].append(current_snr)
    history['gpu_mem'].append(gpu_mem)
    history['throughput'].append(throughput)

    # === 进度条更新 ===
    pbar.set_postfix({
        'loss': f"{accumulated_loss:.4f}",
        'det': f"{accumulated_det / cfg['grad_accumulation']:.4f}",
        'den': f"{accumulated_den / cfg['grad_accumulation']:.4f}",
        'lr': f"{scheduler.get_last_lr()[1]:.1e}",
        'SNR': f"{current_snr:.0f}",
        'mem': f"{gpu_mem:.1f}G",
        'spd': f"{throughput:.0f}/s",
    })

    # === 详细日志 ===
    if step % cfg['log_every'] == 0:
        elapsed = time.time() - start_time
        eta = elapsed / step * (cfg['n_steps'] - step)

        tqdm.write(
            f"\n[{step:5d}/{cfg['n_steps']}] "
            f"Loss: {accumulated_loss:.4f} | "
            f"Det: {accumulated_det / cfg['grad_accumulation']:.4f} | "
            f"Den: {accumulated_den / cfg['grad_accumulation']:.4f} | "
            f"LR: {scheduler.get_last_lr()[1]:.2e} | "
            f"SNR: {current_snr:.1f} | "
            f"GPU: {gpu_mem:.1f}GB | "
            f"Speed: {throughput:.0f} samples/s | "
            f"ETA: {eta / 60:.1f}min"
        )

        # Denoiser 权重监控
        denoiser_norm = sum(p.data.norm().item() for p in denoiser_params)
        tqdm.write(f"  Denoiser weight norm: {denoiser_norm:.4f}")

    # === 保存 ===
    if step % cfg['save_every'] == 0 or step == cfg['n_steps']:
        checkpoint = {
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'config': cfg,
            'history': history,
            'loss': accumulated_loss,
        }

        save_path = output_dir / f"step_{step}.pth"
        torch.save(checkpoint, save_path)

        if accumulated_loss < best_loss:
            best_loss = accumulated_loss
            torch.save(checkpoint, output_dir / "best.pth")
            tqdm.write(f"  💾 New best! Loss: {best_loss:.4f}")

# %%
# =============================================================================
# CELL 9: 训练完成统计
# =============================================================================

total_time = time.time() - start_time
avg_throughput = np.mean(history['throughput'])

print("\n" + "=" * 60)
print("✅ 训练完成!")
print("=" * 60)
print(f"  总时间: {total_time / 60:.1f} 分钟")
print(f"  平均速度: {avg_throughput:.0f} samples/s")
print(f"  GPU 峰值内存: {max(history['gpu_mem']):.1f} GB")
print(f"  最佳 Loss: {best_loss:.4f}")
print(f"  最终 Loss: {history['loss'][-1]:.4f}")

# 保存最终模型
final_ckpt = {
    'step': cfg['n_steps'],
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'config': cfg,
    'history': history,
    'training_time': total_time,
}
torch.save(final_ckpt, output_dir / "final.pth")
print(f"\n📦 最终模型保存到: {output_dir / 'final.pth'}")

# %%
# =============================================================================
# CELL 10: Denoiser 权重诊断
# =============================================================================

print("\n" + "=" * 60)
print("🔬 Denoiser 权重诊断 (训练后)")
print("=" * 60)

model.eval()

for name, param in model.named_parameters():
    if 'denoiser' in name.lower() or 'vamp' in name.lower():
        norm = param.data.norm().item()
        std = param.data.std().item()
        mean = param.data.mean().item()
        max_val = param.data.abs().max().item()
        print(f"{name}:")
        print(f"  norm={norm:.4f}, std={std:.6f}, mean={mean:.6f}, max={max_val:.4f}")

# %%
# =============================================================================
# CELL 11: 快速验证
# =============================================================================

print("\n" + "=" * 60)
print("🧪 快速验证")
print("=" * 60)

model.eval()
with torch.no_grad():
    for test_snr in [10, 15, 20, 25]:
        test_batch = generate_batch_cuda(128, test_snr)

        with autocast(enabled=cfg['use_amp']):
            outputs = model(test_batch['y'], test_batch['pilots'], test_batch['pilot_idx'])

        # 计算 MSE
        mse = nn.functional.mse_loss(outputs['x_hat'], test_batch['x']).item()
        tau_rmse = torch.sqrt(nn.functional.mse_loss(outputs['tau_hat'], test_batch['tau'])).item()

        # 计算 SER (近似)
        x_hat = outputs['x_hat']
        x_true = test_batch['x']
        x_hat_hard = torch.sign(x_hat.real) + 1j * torch.sign(x_hat.imag)
        x_hat_hard = x_hat_hard / np.sqrt(2)
        x_true_hard = torch.sign(x_true.real) + 1j * torch.sign(x_true.imag)
        x_true_hard = x_true_hard / np.sqrt(2)
        ser = (x_hat_hard != x_true_hard).float().mean().item()

        print(f"  SNR={test_snr:2d} dB: MSE={mse:.4f}, τ_RMSE={tau_rmse:.4f}, SER≈{ser:.4f}")

# %%
# =============================================================================
# CELL 12: 绘制训练曲线
# =============================================================================

import matplotlib.pyplot as plt

fig, axes = plt.subplots(2, 3, figsize=(15, 8))

# Loss
ax = axes[0, 0]
ax.semilogy(history['step'], history['loss'], 'b-', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Total Loss')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)

# Detection Loss
ax = axes[0, 1]
ax.semilogy(history['step'], history['det_loss'], 'g-', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Detection Loss')
ax.set_title('Detection Loss')
ax.grid(True, alpha=0.3)

# Denoiser Loss
ax = axes[0, 2]
ax.semilogy(history['step'], history['denoiser_loss'], 'r-', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Denoiser Loss')
ax.set_title('Denoiser Loss')
ax.grid(True, alpha=0.3)

# Learning Rate
ax = axes[1, 0]
ax.semilogy(history['step'], history['lr'], 'purple', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Learning Rate')
ax.set_title('Learning Rate Schedule')
ax.grid(True, alpha=0.3)

# SNR Curriculum
ax = axes[1, 1]
ax.plot(history['step'], history['snr'], 'orange', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('SNR (dB)')
ax.set_title('SNR Curriculum')
ax.grid(True, alpha=0.3)

# Throughput
ax = axes[1, 2]
ax.plot(history['step'], history['throughput'], 'cyan', alpha=0.7)
ax.set_xlabel('Step')
ax.set_ylabel('Samples/s')
ax.set_title('Training Throughput')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(output_dir / "training_curves.png", dpi=150)
plt.show()

print(f"\n📊 训练曲线保存到: {output_dir / 'training_curves.png'}")

# %%
# =============================================================================
# CELL 13: 下载结果
# =============================================================================

import shutil
from google.colab import files

# 打包
archive_name = "a100_training_results"
shutil.make_archive(archive_name, 'zip', str(output_dir))

# 下载
files.download(f"{archive_name}.zip")

print("✅ 下载完成!")
print(f"\n解压后，将 final.pth 复制到本地项目:")
print(f"  cp final.pth results/checkpoints/Stage3_FullTrak_1766361144/final.pth")