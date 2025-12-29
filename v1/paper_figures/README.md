# 📊 Paper Figures 模块

重构后的论文图表生成系统，代码清晰、职责分离、易于维护。

## 📁 文件结构

```
paper_figures/
├── __init__.py          # 模块导出
├── baselines.py         # 基线算法实现（~300行）
├── evaluator.py         # 数据采集和评估（~500行）
├── visualization.py     # 纯绘图函数（~450行）
├── run_experiments.py   # 主入口脚本（~200行）
└── README.md            # 本文档
```

**对比原来**：`visualization_v3.py` 有 2000+ 行，职责混乱

## 🎯 职责划分

| 文件 | 职责 | 依赖 |
|------|------|------|
| `baselines.py` | 基线算法实现 | `torch`, `numpy` |
| `evaluator.py` | 数据采集、模型加载、sweep 函数 | `baselines.py`, model |
| `visualization.py` | 从 CSV 生成图表 | `pandas`, `matplotlib` |
| `run_experiments.py` | 协调各模块的主入口 | 全部 |

## 🚀 使用方法

### 完整运行（数据采集 + 可视化）

```bash
cd paper_figures
python run_experiments.py --ckpt path/to/checkpoint.pth --n_mc 20
```

### 快速测试

```bash
python run_experiments.py --ckpt path/to/checkpoint.pth --quick
```

### 仅可视化（从已有 CSV 数据）

```bash
# 如果已经有 CSV 数据，可以跳过数据采集
python run_experiments.py --visualize_only --data_dir results/paper_figs
```

或者直接调用可视化模块：

```bash
python visualization.py --data_dir results/paper_figs
```

### 仅运行基线测试

```python
from baselines import run_baseline, METHOD_ORDER

# 测试单个基线
x_hat, theta_hat = run_baseline("adjoint_slice", model, batch, sim_cfg, device)

# 查看所有可用方法
print(METHOD_ORDER)
# ['naive_slice', 'matched_filter', 'adjoint_lmmse', 'adjoint_slice', 
#  'proposed_no_update', 'proposed', 'oracle']
```

## 📋 方法层级

从弱到强排列：

| 方法 | 描述 | 预期 BER |
|------|------|----------|
| `naive_slice` | 直接 slice（不做前端处理） | ≈ 0.5 |
| `matched_filter` | Grid Search τ + Slice | < 0.5 |
| `adjoint_lmmse` | Adjoint + PN Align + LMMSE | < 0.3 |
| `adjoint_slice` | Adjoint + PN Align + Slice | < 0.3 |
| `proposed_no_update` | BV-VAMP 无 τ 更新 | < 0.2 |
| `proposed` | 完整方法 | ≈ 0.1 |
| `oracle` | 使用真实 θ | ≈ 0.1 |

## 📈 输出图表

| 图表 | 描述 | 专家方案 |
|------|------|----------|
| Fig 1 | BER vs SNR | - |
| Fig 2 | RMSE_τ vs SNR | - |
| Fig 3 | Success Rate vs SNR | - |
| Fig 4 | **Cliff with ALL methods** | **方案1（核心图）** |
| Fig 5 | SNR @ multi init_error | **方案3** |
| Fig 6 | Jacobian Condition Number | - |
| Fig 7 | Gap-to-Oracle | - |
| Fig 8 | Robustness (PN & Pilot) | - |
| Fig 9 | Latency | - |

## 🔧 扩展指南

### 添加新的基线算法

1. 在 `baselines.py` 中添加新类：

```python
class BaselineNewMethod:
    name = "new_method"
    
    @staticmethod
    @torch.no_grad()
    def run(model, batch, sim_cfg, device, pilot_len=64):
        # 实现算法
        return x_hat, theta_hat
```

2. 注册到 `BASELINE_REGISTRY`：

```python
BASELINE_REGISTRY["new_method"] = BaselineNewMethod
```

### 添加新的图表

1. 在 `visualization.py` 中添加函数：

```python
def fig_new_plot(df: pd.DataFrame, out_dir: str):
    # 绑图逻辑
    pass
```

2. 在 `generate_all_figures()` 中调用

### 添加新的 Sweep

在 `evaluator.py` 中添加函数：

```python
def run_new_sweep(model, gabv_cfg, eval_cfg) -> pd.DataFrame:
    records = []
    # sweep 逻辑
    return pd.DataFrame(records)
```

## 📝 专家建议总结

核心叙事：

> "在 1-bit 量化与脏硬件 THz-ISAC 链路中，初始同步误差会触发检测
> '悬崖式失效'；本文提出的 pilot-only 几何一致 τ 快环跟踪将接收机
> 重新拉回可跟踪盆地，使检测性能在该盆地内逼近 oracle 上界。"

关键证据：
- init_error=0 时所有方法都接近 oracle → baseline 实现正确
- init_error=0.3 时 baseline 失效，proposed 仍工作 → τ 更新是关键
- basin 边界约 0.3-0.5 samples → 物理可解释

## ⚠️ 旧文件处理

以下旧文件可以**删除**：
- `visualization.py`（旧版）
- `visualization_v2.py`
- `visualization_v3.py`
- `visualization_v3_expert.py`

以下文件**保留**：
- `gabv_net_model.py`（模型定义）
- `thz_isac_world.py`（仿真器）
- `train_gabv_net.py`（训练脚本）
- Checkpoint 文件
