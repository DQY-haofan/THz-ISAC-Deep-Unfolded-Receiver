"""
results_manager.py - 统一结果保存与可视化模块

功能:
    1. 统一输出文件夹结构管理
    2. 每个图都保存 CSV (数据) + PNG + PDF
    3. 多场景对比图 (将4个阶段合并到一个图中)

目录结构:
    results/
    ├── all_outputs/           # 统一输出文件夹
    │   ├── data/              # 所有CSV数据
    │   │   ├── S1_metrics.csv
    │   │   ├── S2_metrics.csv
    │   │   ├── combined_metrics.csv
    │   │   └── fig_ber_data.csv  # 每个图的数据
    │   ├── figures_individual/  # 单场景图
    │   │   ├── S1_ber.png/pdf
    │   │   ├── S2_ber.png/pdf
    │   │   └── ...
    │   ├── figures_combined/    # 多场景对比图
    │   │   ├── combined_ber.png/pdf/csv
    │   │   ├── combined_rmse.png/pdf/csv
    │   │   └── ...
    │   └── config/             # 配置文件
    │       └── experiment_config.json

Author: Claude Assistant
Date: 2025-12-18
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict
import time

# =============================================================================
# 配置
# =============================================================================

# 出版级别绘图配置
plt.style.use('seaborn-v0_8-whitegrid')
PLOT_CONFIG = {
    'font.family': 'serif',
    'font.size': 11,
    'axes.labelsize': 13,
    'axes.titlesize': 13,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.figsize': (8, 6),
    'lines.linewidth': 2.0,
    'lines.markersize': 8,
    'figure.dpi': 150,
    'savefig.dpi': 300,
}
plt.rcParams.update(PLOT_CONFIG)

# 场景颜色和标记配置
SCENE_STYLES = {
    'S1_full_hw': {'color': '#1f77b4', 'marker': 'o', 'linestyle': '-', 'label': 'Full HW (PA+PN+1bit)'},
    'S2_pa_only': {'color': '#ff7f0e', 'marker': 's', 'linestyle': '--', 'label': 'PA Only'},
    'S3_pn_only': {'color': '#2ca02c', 'marker': '^', 'linestyle': '-.', 'label': 'PN Only'},
    'S4_ideal': {'color': '#d62728', 'marker': 'd', 'linestyle': ':', 'label': 'Ideal'},
}

# 指标配置
METRIC_CONFIG = {
    'BER': {
        'ylabel': 'Bit Error Rate (BER)',
        'scale': 'log',
        'ylim': [1e-5, 1],
        'title': 'BER vs SNR Comparison',
    },
    'NMSE': {
        'ylabel': 'NMSE (dB)',
        'scale': 'linear',
        'ylim': None,
        'title': 'NMSE vs SNR Comparison',
    },
    'RMSE_R': {
        'ylabel': 'Range RMSE (m)',
        'scale': 'log',
        'ylim': None,
        'title': 'Range RMSE vs SNR Comparison',
    },
    'RMSE_v': {
        'ylabel': 'Velocity RMSE (m/s)',
        'scale': 'log',
        'ylim': None,
        'title': 'Velocity RMSE vs SNR Comparison',
    },
    'gamma_eff': {
        'ylabel': r'$\Gamma_{\mathrm{eff}}$ (dB)',
        'scale': 'linear',
        'ylim': None,
        'title': r'Hardware Distortion $\Gamma_{\mathrm{eff}}$ vs SNR',
    },
    'chi': {
        'ylabel': r'$\chi$ (Information Retention)',
        'scale': 'linear',
        'ylim': [0, 0.7],
        'title': r'Chi Factor $\chi$ vs SNR',
    },
}


@dataclass
class OutputConfig:
    """输出配置"""
    base_dir: str = "results/all_outputs"
    data_subdir: str = "data"
    figures_individual_subdir: str = "figures_individual"
    figures_combined_subdir: str = "figures_combined"
    config_subdir: str = "config"

    save_png: bool = True
    save_pdf: bool = True
    save_csv: bool = True

    png_dpi: int = 300
    pdf_format: str = 'pdf'


# =============================================================================
# 结果管理器类
# =============================================================================

class ResultsManager:
    """
    统一的结果保存与可视化管理器

    使用方法:
        manager = ResultsManager(base_dir="results/all_outputs")

        # 添加场景数据
        manager.add_scene_data("S1_full_hw", df_s1, config_s1)
        manager.add_scene_data("S2_pa_only", df_s2, config_s2)

        # 保存所有单场景图
        manager.save_all_individual_plots()

        # 生成并保存多场景对比图
        manager.save_all_combined_plots()

        # 保存汇总CSV
        manager.save_combined_csv()
    """

    def __init__(self, base_dir: str = "results/all_outputs"):
        self.config = OutputConfig(base_dir=base_dir)
        self.base_path = Path(base_dir)
        self.scene_data: Dict[str, pd.DataFrame] = {}
        self.scene_configs: Dict[str, dict] = {}

        # 创建目录结构
        self._create_directories()

    def _create_directories(self):
        """创建输出目录结构"""
        dirs = [
            self.base_path / self.config.data_subdir,
            self.base_path / self.config.figures_individual_subdir,
            self.base_path / self.config.figures_combined_subdir,
            self.base_path / self.config.config_subdir,
        ]
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    @property
    def data_dir(self) -> Path:
        return self.base_path / self.config.data_subdir

    @property
    def figures_individual_dir(self) -> Path:
        return self.base_path / self.config.figures_individual_subdir

    @property
    def figures_combined_dir(self) -> Path:
        return self.base_path / self.config.figures_combined_subdir

    @property
    def config_dir(self) -> Path:
        return self.base_path / self.config.config_subdir

    # =========================================================================
    # 数据管理
    # =========================================================================

    def add_scene_data(self, scene_id: str, df: pd.DataFrame,
                       scene_config: Optional[dict] = None):
        """添加场景数据"""
        self.scene_data[scene_id] = df.copy()
        if scene_config:
            self.scene_configs[scene_id] = scene_config

        # 保存单场景CSV
        csv_path = self.data_dir / f"{scene_id}_metrics.csv"
        df.to_csv(csv_path, index=False)
        print(f"  [Data] Saved: {csv_path}")

    def load_scene_data(self, scene_id: str, csv_path: str):
        """从CSV加载场景数据"""
        df = pd.read_csv(csv_path)
        self.scene_data[scene_id] = df
        print(f"  [Data] Loaded: {csv_path}")

    def save_combined_csv(self):
        """保存所有场景合并的CSV"""
        if not self.scene_data:
            print("  [Warning] No scene data to combine")
            return

        combined_rows = []
        for scene_id, df in self.scene_data.items():
            df_copy = df.copy()
            df_copy.insert(0, 'scene_id', scene_id)
            combined_rows.append(df_copy)

        combined_df = pd.concat(combined_rows, ignore_index=True)
        csv_path = self.data_dir / "combined_all_scenes.csv"
        combined_df.to_csv(csv_path, index=False)
        print(f"  [Data] Saved combined CSV: {csv_path}")

        return combined_df

    # =========================================================================
    # 单场景绘图
    # =========================================================================

    def _save_figure(self, fig: plt.Figure, base_name: str,
                     output_dir: Path, plot_data: Optional[pd.DataFrame] = None):
        """
        保存图形为 PNG + PDF + CSV

        Args:
            fig: matplotlib Figure对象
            base_name: 基础文件名 (不含扩展名)
            output_dir: 输出目录
            plot_data: 用于生成图的数据DataFrame
        """
        # PNG
        if self.config.save_png:
            png_path = output_dir / f"{base_name}.png"
            fig.savefig(png_path, dpi=self.config.png_dpi, bbox_inches='tight')
            print(f"    [PNG] {png_path}")

        # PDF
        if self.config.save_pdf:
            pdf_path = output_dir / f"{base_name}.pdf"
            fig.savefig(pdf_path, format='pdf', bbox_inches='tight')
            print(f"    [PDF] {pdf_path}")

        # CSV (图的数据)
        if self.config.save_csv and plot_data is not None:
            csv_path = output_dir / f"{base_name}_data.csv"
            plot_data.to_csv(csv_path, index=False)
            print(f"    [CSV] {csv_path}")

    def plot_individual_ber(self, scene_id: str, include_baselines: bool = True):
        """绘制单场景BER图"""
        if scene_id not in self.scene_data:
            print(f"  [Warning] Scene {scene_id} not found")
            return

        df = self.scene_data[scene_id]
        style = SCENE_STYLES.get(scene_id, {'color': 'blue', 'marker': 'o', 'linestyle': '-'})

        fig, ax = plt.subplots(figsize=(8, 6))

        # 准备导出数据
        plot_data = {'snr_db': df['snr_db'].values}

        # GA-BV-Net
        if 'BER_Net' in df.columns and not df['BER_Net'].isna().all():
            ax.semilogy(df['snr_db'], df['BER_Net'],
                        color='#1f77b4', marker='o', linestyle='-',
                        linewidth=2, markersize=8, label='GA-BV-Net')
            plot_data['BER_Net'] = df['BER_Net'].values

        # Baselines
        if include_baselines:
            if 'BER_LMMSE' in df.columns and not df['BER_LMMSE'].isna().all():
                ax.semilogy(df['snr_db'], df['BER_LMMSE'],
                            color='#ff7f0e', marker='s', linestyle='--',
                            linewidth=2, markersize=7, label='Bussgang-LMMSE')
                plot_data['BER_LMMSE'] = df['BER_LMMSE'].values

            if 'BER_GAMP' in df.columns and not df['BER_GAMP'].isna().all():
                ax.semilogy(df['snr_db'], df['BER_GAMP'],
                            color='#2ca02c', marker='^', linestyle='-.',
                            linewidth=2, markersize=7, label='1-bit GAMP')
                plot_data['BER_GAMP'] = df['BER_GAMP'].values

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Bit Error Rate (BER)')
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='-', alpha=0.3)
        ax.set_ylim([1e-5, 1])

        fig.tight_layout()

        # 保存
        self._save_figure(fig, f"{scene_id}_ber",
                          self.figures_individual_dir,
                          pd.DataFrame(plot_data))
        plt.close(fig)

    def plot_individual_rmse(self, scene_id: str, metric: str = 'RMSE_R'):
        """绘制单场景RMSE图"""
        if scene_id not in self.scene_data:
            return

        df = self.scene_data[scene_id]

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_data = {'snr_db': df['snr_db'].values}

        # RMSE
        if metric in df.columns and not df[metric].isna().all():
            ax.semilogy(df['snr_db'], df[metric],
                        'b-o', linewidth=2, markersize=8, label=f'GA-BV-Net {metric}')
            plot_data[metric] = df[metric].values

        # BCRLB参考
        bcrlb_col = f'BCRLB_{metric.split("_")[1]}_ref' if '_' in metric else 'BCRLB_R_ref'
        if bcrlb_col in df.columns and not df[bcrlb_col].isna().all():
            lb_ref = np.sqrt(df[bcrlb_col])
            ax.semilogy(df['snr_db'], lb_ref, 'k-', linewidth=2,
                        label=r'$\sqrt{\mathrm{BCRLB}_\mathrm{ref}}$ (χ-scaled)')
            plot_data['BCRLB_sqrt'] = lb_ref.values

        # Analog CRLB
        bcrlb_analog_col = bcrlb_col.replace('_ref', '_analog')
        if bcrlb_analog_col in df.columns and not df[bcrlb_analog_col].isna().all():
            lb_analog = np.sqrt(df[bcrlb_analog_col])
            ax.semilogy(df['snr_db'], lb_analog, 'k:', linewidth=1.5,
                        label=r'$\sqrt{\mathrm{CRLB}_\mathrm{analog}}$')
            plot_data['CRLB_analog_sqrt'] = lb_analog.values

        config = METRIC_CONFIG.get(metric, {})
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(config.get('ylabel', f'{metric}'))
        ax.legend(loc='upper right')
        ax.grid(True, which='both', linestyle='-', alpha=0.3)

        fig.tight_layout()
        self._save_figure(fig, f"{scene_id}_{metric.lower()}",
                          self.figures_individual_dir,
                          pd.DataFrame(plot_data))
        plt.close(fig)

    def plot_individual_gamma_chi(self, scene_id: str):
        """绘制单场景 Gamma_eff 和 Chi 双轴图"""
        if scene_id not in self.scene_data:
            return

        df = self.scene_data[scene_id]

        fig, ax1 = plt.subplots(figsize=(8, 6))
        plot_data = {'snr_db': df['snr_db'].values}

        # Gamma_eff (左轴)
        if 'gamma_eff' in df.columns and not df['gamma_eff'].isna().all():
            gamma_db = 10 * np.log10(df['gamma_eff'].values + 1e-12)
            ax1.plot(df['snr_db'], gamma_db, 'b-o', linewidth=2, markersize=8,
                     label=r'$\Gamma_{\mathrm{eff}}$ (dB)')
            ax1.set_ylabel(r'$\Gamma_{\mathrm{eff}}$ (dB)', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            plot_data['gamma_eff_db'] = gamma_db

        # Chi (右轴)
        ax2 = ax1.twinx()
        if 'chi' in df.columns and not df['chi'].isna().all():
            ax2.plot(df['snr_db'], df['chi'], 'r--s', linewidth=2, markersize=7,
                     label=r'$\chi$')
            ax2.axhline(y=2 / np.pi, color='r', linestyle=':', alpha=0.5,
                        label=r'$\chi_{\mathrm{max}} = 2/\pi$')
            ax2.set_ylabel(r'$\chi$ (Information Retention)', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim([0, 0.7])
            plot_data['chi'] = df['chi'].values

        ax1.set_xlabel('SNR (dB)')
        ax1.grid(True, linestyle='-', alpha=0.3)

        # 合并图例
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')

        fig.tight_layout()
        self._save_figure(fig, f"{scene_id}_gamma_chi",
                          self.figures_individual_dir,
                          pd.DataFrame(plot_data))
        plt.close(fig)

    def plot_individual_nmse(self, scene_id: str):
        """绘制单场景NMSE图"""
        if scene_id not in self.scene_data:
            return

        df = self.scene_data[scene_id]

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_data = {'snr_db': df['snr_db'].values}

        if 'NMSE_Net' in df.columns and not df['NMSE_Net'].isna().all():
            ax.plot(df['snr_db'], df['NMSE_Net'], 'b-o',
                    linewidth=2, markersize=8, label='GA-BV-Net')
            plot_data['NMSE_Net'] = df['NMSE_Net'].values

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('NMSE (dB)')
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='-', alpha=0.3)

        fig.tight_layout()
        self._save_figure(fig, f"{scene_id}_nmse",
                          self.figures_individual_dir,
                          pd.DataFrame(plot_data))
        plt.close(fig)

    def save_all_individual_plots(self):
        """保存所有单场景图"""
        print("\n[Saving Individual Plots]")
        for scene_id in self.scene_data.keys():
            print(f"  Scene: {scene_id}")
            self.plot_individual_ber(scene_id)
            self.plot_individual_rmse(scene_id, 'RMSE_R')
            self.plot_individual_nmse(scene_id)
            self.plot_individual_gamma_chi(scene_id)

    # =========================================================================
    # 多场景对比图
    # =========================================================================

    def plot_combined_metric(self, metric_column: str,
                             ylabel: str = None,
                             yscale: str = 'linear',
                             ylim: Optional[Tuple[float, float]] = None,
                             title: str = None,
                             transform_fn=None):
        """
        绘制多场景对比图

        Args:
            metric_column: 数据列名
            ylabel: Y轴标签
            yscale: 'linear' 或 'log'
            ylim: Y轴范围
            title: 图标题 (不显示但记录)
            transform_fn: 数据转换函数 (例如 dB转换)
        """
        if not self.scene_data:
            print("  [Warning] No scene data available")
            return

        fig, ax = plt.subplots(figsize=(9, 6))
        plot_data = {'snr_db': None}

        for scene_id, df in self.scene_data.items():
            if metric_column not in df.columns:
                continue
            if df[metric_column].isna().all():
                continue

            style = SCENE_STYLES.get(scene_id, {
                'color': 'gray', 'marker': 'x', 'linestyle': '-', 'label': scene_id
            })

            # 获取数据
            y_data = df[metric_column].values
            if transform_fn:
                y_data = transform_fn(y_data)

            # 绘制
            if yscale == 'log':
                ax.semilogy(df['snr_db'], y_data,
                            color=style['color'], marker=style['marker'],
                            linestyle=style['linestyle'], linewidth=2, markersize=8,
                            label=style['label'])
            else:
                ax.plot(df['snr_db'], y_data,
                        color=style['color'], marker=style['marker'],
                        linestyle=style['linestyle'], linewidth=2, markersize=8,
                        label=style['label'])

            # 记录数据
            if plot_data['snr_db'] is None:
                plot_data['snr_db'] = df['snr_db'].values
            plot_data[f'{scene_id}_{metric_column}'] = y_data

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(ylabel or metric_column)
        ax.legend(loc='best')
        ax.grid(True, which='both', linestyle='-', alpha=0.3)

        if ylim:
            ax.set_ylim(ylim)

        fig.tight_layout()

        # 保存
        base_name = f"combined_{metric_column.lower()}"
        self._save_figure(fig, base_name, self.figures_combined_dir,
                          pd.DataFrame(plot_data))
        plt.close(fig)
        print(f"  [Combined Plot] {metric_column}")

    def plot_combined_ber(self):
        """绘制多场景BER对比图"""
        self.plot_combined_metric(
            metric_column='BER_Net',
            ylabel='Bit Error Rate (BER)',
            yscale='log',
            ylim=[1e-5, 1],
            title='BER Comparison Across Scenarios'
        )

    def plot_combined_nmse(self):
        """绘制多场景NMSE对比图"""
        self.plot_combined_metric(
            metric_column='NMSE_Net',
            ylabel='NMSE (dB)',
            yscale='linear',
            title='NMSE Comparison Across Scenarios'
        )

    def plot_combined_rmse_range(self):
        """绘制多场景Range RMSE对比图"""
        self.plot_combined_metric(
            metric_column='RMSE_R',
            ylabel='Range RMSE (m)',
            yscale='log',
            title='Range RMSE Comparison Across Scenarios'
        )

    def plot_combined_gamma_eff(self):
        """绘制多场景Gamma_eff对比图"""
        self.plot_combined_metric(
            metric_column='gamma_eff',
            ylabel=r'$\Gamma_{\mathrm{eff}}$ (dB)',
            yscale='linear',
            title='Hardware Distortion Comparison',
            transform_fn=lambda x: 10 * np.log10(np.maximum(x, 1e-12))
        )

    def plot_combined_chi(self):
        """绘制多场景Chi对比图"""
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_data = {'snr_db': None}

        for scene_id, df in self.scene_data.items():
            if 'chi' not in df.columns or df['chi'].isna().all():
                continue

            style = SCENE_STYLES.get(scene_id, {
                'color': 'gray', 'marker': 'x', 'linestyle': '-', 'label': scene_id
            })

            ax.plot(df['snr_db'], df['chi'],
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], linewidth=2, markersize=8,
                    label=style['label'])

            if plot_data['snr_db'] is None:
                plot_data['snr_db'] = df['snr_db'].values
            plot_data[f'{scene_id}_chi'] = df['chi'].values

        # 添加理论极限线
        ax.axhline(y=2 / np.pi, color='black', linestyle=':', linewidth=1.5,
                   alpha=0.7, label=r'$\chi_{\max} = 2/\pi$')

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(r'$\chi$ (Information Retention Factor)')
        ax.legend(loc='best')
        ax.grid(True, linestyle='-', alpha=0.3)
        ax.set_ylim([0, 0.7])

        fig.tight_layout()
        self._save_figure(fig, "combined_chi", self.figures_combined_dir,
                          pd.DataFrame(plot_data))
        plt.close(fig)
        print(f"  [Combined Plot] chi")

    def plot_combined_gap_to_bcrlb(self):
        """绘制多场景与BCRLB差距对比图"""
        fig, ax = plt.subplots(figsize=(9, 6))
        plot_data = {'snr_db': None}

        for scene_id, df in self.scene_data.items():
            if 'RMSE_R' not in df.columns or 'BCRLB_R_ref' not in df.columns:
                continue
            if df['RMSE_R'].isna().all() or df['BCRLB_R_ref'].isna().all():
                continue

            style = SCENE_STYLES.get(scene_id, {
                'color': 'gray', 'marker': 'x', 'linestyle': '-', 'label': scene_id
            })

            rmse = df['RMSE_R'].values
            bcrlb = np.sqrt(df['BCRLB_R_ref'].values)
            gap_db = 20 * np.log10(rmse / (bcrlb + 1e-12) + 1e-12)

            ax.plot(df['snr_db'], gap_db,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], linewidth=2, markersize=8,
                    label=style['label'])

            if plot_data['snr_db'] is None:
                plot_data['snr_db'] = df['snr_db'].values
            plot_data[f'{scene_id}_gap_db'] = gap_db

        # 添加0 dB参考线
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1.5, label='BCRLB (0 dB)')

        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Gap to BCRLB (dB)')
        ax.legend(loc='best')
        ax.grid(True, linestyle='-', alpha=0.3)

        fig.tight_layout()
        self._save_figure(fig, "combined_gap_bcrlb", self.figures_combined_dir,
                          pd.DataFrame(plot_data))
        plt.close(fig)
        print(f"  [Combined Plot] gap_to_bcrlb")

    def plot_combined_all_in_one(self):
        """
        绘制2x2子图，将四个主要指标合并到一个图中
        """
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        all_plot_data = {}

        # (0,0) BER
        ax = axes[0, 0]
        for scene_id, df in self.scene_data.items():
            if 'BER_Net' not in df.columns or df['BER_Net'].isna().all():
                continue
            style = SCENE_STYLES.get(scene_id, {'color': 'gray', 'marker': 'x', 'linestyle': '-', 'label': scene_id})
            ax.semilogy(df['snr_db'], df['BER_Net'],
                        color=style['color'], marker=style['marker'],
                        linestyle=style['linestyle'], linewidth=2, markersize=7,
                        label=style['label'])
            all_plot_data[f'{scene_id}_BER'] = df['BER_Net'].values
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('BER')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_ylim([1e-5, 1])
        ax.set_title('(a) Bit Error Rate', fontsize=11, fontweight='bold')

        # (0,1) RMSE_R
        ax = axes[0, 1]
        for scene_id, df in self.scene_data.items():
            if 'RMSE_R' not in df.columns or df['RMSE_R'].isna().all():
                continue
            style = SCENE_STYLES.get(scene_id, {'color': 'gray', 'marker': 'x', 'linestyle': '-', 'label': scene_id})
            ax.semilogy(df['snr_db'], df['RMSE_R'],
                        color=style['color'], marker=style['marker'],
                        linestyle=style['linestyle'], linewidth=2, markersize=7,
                        label=style['label'])
            all_plot_data[f'{scene_id}_RMSE_R'] = df['RMSE_R'].values
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel('Range RMSE (m)')
        ax.legend(loc='upper right', fontsize=8)
        ax.grid(True, which='both', alpha=0.3)
        ax.set_title('(b) Range RMSE', fontsize=11, fontweight='bold')

        # (1,0) Gamma_eff
        ax = axes[1, 0]
        for scene_id, df in self.scene_data.items():
            if 'gamma_eff' not in df.columns or df['gamma_eff'].isna().all():
                continue
            style = SCENE_STYLES.get(scene_id, {'color': 'gray', 'marker': 'x', 'linestyle': '-', 'label': scene_id})
            gamma_db = 10 * np.log10(df['gamma_eff'].values + 1e-12)
            ax.plot(df['snr_db'], gamma_db,
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], linewidth=2, markersize=7,
                    label=style['label'])
            all_plot_data[f'{scene_id}_gamma_db'] = gamma_db
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(r'$\Gamma_{\mathrm{eff}}$ (dB)')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_title(r'(c) Hardware Distortion $\Gamma_{\mathrm{eff}}$', fontsize=11, fontweight='bold')

        # (1,1) Chi
        ax = axes[1, 1]
        for scene_id, df in self.scene_data.items():
            if 'chi' not in df.columns or df['chi'].isna().all():
                continue
            style = SCENE_STYLES.get(scene_id, {'color': 'gray', 'marker': 'x', 'linestyle': '-', 'label': scene_id})
            ax.plot(df['snr_db'], df['chi'],
                    color=style['color'], marker=style['marker'],
                    linestyle=style['linestyle'], linewidth=2, markersize=7,
                    label=style['label'])
            all_plot_data[f'{scene_id}_chi'] = df['chi'].values
        ax.axhline(y=2 / np.pi, color='black', linestyle=':', alpha=0.7, label=r'$\chi_{\max}$')
        ax.set_xlabel('SNR (dB)')
        ax.set_ylabel(r'$\chi$')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 0.7])
        ax.set_title(r'(d) Information Retention $\chi$', fontsize=11, fontweight='bold')

        # 添加SNR数据
        for scene_id, df in self.scene_data.items():
            all_plot_data['snr_db'] = df['snr_db'].values
            break

        fig.tight_layout()
        self._save_figure(fig, "combined_all_metrics_2x2", self.figures_combined_dir,
                          pd.DataFrame(all_plot_data))
        plt.close(fig)
        print(f"  [Combined Plot] all_metrics_2x2")

    def save_all_combined_plots(self):
        """保存所有多场景对比图"""
        print("\n[Saving Combined Plots]")
        self.plot_combined_ber()
        self.plot_combined_nmse()
        self.plot_combined_rmse_range()
        self.plot_combined_gamma_eff()
        self.plot_combined_chi()
        self.plot_combined_gap_to_bcrlb()
        self.plot_combined_all_in_one()

    # =========================================================================
    # 配置保存
    # =========================================================================

    def save_experiment_config(self, additional_info: Optional[dict] = None):
        """保存实验配置"""
        config_data = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'scenes': list(self.scene_data.keys()),
            'scene_configs': self.scene_configs,
            'output_config': asdict(self.config),
        }
        if additional_info:
            config_data['additional_info'] = additional_info

        config_path = self.config_dir / "experiment_config.json"
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2, default=str)
        print(f"\n[Config] Saved: {config_path}")

    # =========================================================================
    # 完整保存流程
    # =========================================================================

    def save_all(self, additional_config: Optional[dict] = None):
        """完整保存流程"""
        print("\n" + "=" * 60)
        print("ResultsManager - Saving All Outputs")
        print("=" * 60)

        # 1. 保存合并CSV
        self.save_combined_csv()

        # 2. 保存单场景图
        self.save_all_individual_plots()

        # 3. 保存多场景对比图
        self.save_all_combined_plots()

        # 4. 保存配置
        self.save_experiment_config(additional_config)

        print("\n" + "=" * 60)
        print(f"All outputs saved to: {self.base_path}")
        print("=" * 60)


# =============================================================================
# 辅助函数：与现有代码集成
# =============================================================================

def integrate_with_run_p4(results_dir: str = "results/p4_experiments",
                          output_dir: str = "results/all_outputs"):
    """
    集成现有的run_p4_experiments.py输出

    Args:
        results_dir: 现有结果目录 (包含 S1_full_hw, S2_pa_only 等子目录)
        output_dir: 新的统一输出目录
    """
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"[Error] Results directory not found: {results_dir}")
        return None

    manager = ResultsManager(base_dir=output_dir)

    # 查找所有场景目录
    scene_dirs = [d for d in results_path.iterdir() if d.is_dir()]

    for scene_dir in scene_dirs:
        scene_id = scene_dir.name
        metrics_file = scene_dir / "metrics_mean.csv"
        config_file = scene_dir / "config.json"

        if metrics_file.exists():
            df = pd.read_csv(metrics_file)

            # 加载配置
            scene_config = None
            if config_file.exists():
                with open(config_file, 'r') as f:
                    scene_config = json.load(f)

            manager.add_scene_data(scene_id, df, scene_config)
            print(f"  [Loaded] {scene_id}: {len(df)} SNR points")

    if not manager.scene_data:
        print("[Warning] No scene data found")
        return None

    return manager


# =============================================================================
# Main - 演示和测试
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Results Manager Demo")
    parser.add_argument('--input', type=str, default='results/p4_experiments',
                        help='Input results directory')
    parser.add_argument('--output', type=str, default='results/all_outputs',
                        help='Output directory')
    parser.add_argument('--demo', action='store_true',
                        help='Run demo with synthetic data')
    args = parser.parse_args()

    if args.demo:
        # 演示模式：生成合成数据
        print("\n[Demo Mode] Generating synthetic data...")

        manager = ResultsManager(base_dir=args.output)

        snr_grid = np.arange(-5, 26, 2)

        # 生成4个场景的模拟数据
        for scene_id, scene_info in SCENE_STYLES.items():
            np.random.seed(hash(scene_id) % 2 ** 32)

            # 模拟不同场景的性能
            base_ber = 0.5 * np.exp(-snr_grid / 10)
            base_rmse = 100 * np.exp(-snr_grid / 15)

            if 'ideal' in scene_id:
                ber_factor = 0.3
                rmse_factor = 0.5
            elif 'pa_only' in scene_id:
                ber_factor = 0.8
                rmse_factor = 0.9
            elif 'pn_only' in scene_id:
                ber_factor = 0.7
                rmse_factor = 0.8
            else:  # full_hw
                ber_factor = 1.0
                rmse_factor = 1.0

            df = pd.DataFrame({
                'snr_db': snr_grid,
                'BER_Net': base_ber * ber_factor + np.random.normal(0, 0.01, len(snr_grid)),
                'NMSE_Net': -snr_grid + np.random.normal(0, 1, len(snr_grid)),
                'RMSE_R': base_rmse * rmse_factor,
                'gamma_eff': 10 ** ((snr_grid * 0.5 + 5) / 10),
                'chi': 0.6 - 0.3 * (1 - np.exp(-snr_grid / 20)),
                'BCRLB_R_ref': (base_rmse * 0.3) ** 2,
                'BCRLB_R_analog': (base_rmse * 0.1) ** 2,
            })

            # 确保BER在合理范围
            df['BER_Net'] = df['BER_Net'].clip(1e-5, 0.5)

            manager.add_scene_data(scene_id, df, {'description': scene_info['label']})

        # 保存所有
        manager.save_all(additional_config={'mode': 'demo', 'note': 'Synthetic data for testing'})

    else:
        # 集成模式：从现有结果目录加载
        print(f"\n[Integration Mode] Loading from: {args.input}")

        manager = integrate_with_run_p4(args.input, args.output)

        if manager:
            manager.save_all()
        else:
            print("[Error] Failed to load data. Try --demo for synthetic data.")