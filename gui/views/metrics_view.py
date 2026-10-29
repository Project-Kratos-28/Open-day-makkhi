"""
Metrics view displaying live run KPIs, model architecture specs, and historical benchmark baselines.
"""
import tkinter as tk
from tkinter import ttk
from typing import Optional

from gui.config import MODELS, THEME
from gui.data_manager import CheckpointInfo
from gui.runner import RunSummary


class MetricsView(ttk.Frame):
    """Card-based dashboard displaying performance statistics and model metadata."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, padding=8)
        self.columnconfigure(0, weight=1)
        self.columnconfigure(1, weight=1)
        self.columnconfigure(2, weight=1)
        self._build_ui()

    def _build_ui(self) -> None:
        # --- Card 1: Live Run Performance (KPIs) ---
        self.card_live = ttk.LabelFrame(self, text=" Live Run Performance ", padding=10)
        self.card_live.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)
        self.card_live.columnconfigure(0, weight=1)
        self.card_live.columnconfigure(1, weight=1)

        ttk.Label(self.card_live, text="Success Rate:", font=(THEME["font_family"], 9, "bold")).grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.lbl_success = ttk.Label(
            self.card_live,
            text="-- % (0/0)",
            font=(THEME["font_family"], 11, "bold"),
            foreground="#059669",
        )
        self.lbl_success.grid(row=0, column=1, sticky="e", pady=2)

        ttk.Label(self.card_live, text="Avg Time to Goal:", font=(THEME["font_family"], 9)).grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.lbl_avg_time = ttk.Label(
            self.card_live,
            text="-- s",
            font=(THEME["font_family"], 10, "bold"),
            foreground="#0284c7",
        )
        self.lbl_avg_time.grid(row=1, column=1, sticky="e", pady=2)

        ttk.Label(self.card_live, text="Fastest / Slowest:", font=(THEME["font_family"], 9)).grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.lbl_min_max_time = ttk.Label(self.card_live, text="-- / --", font=(THEME["font_family"], 9))
        self.lbl_min_max_time.grid(row=2, column=1, sticky="e", pady=2)

        ttk.Label(self.card_live, text="Crashes / Timeouts:", font=(THEME["font_family"], 9)).grid(
            row=3, column=0, sticky="w", pady=2
        )
        self.lbl_breakdown = ttk.Label(self.card_live, text="0 / 0", font=(THEME["font_family"], 9))
        self.lbl_breakdown.grid(row=3, column=1, sticky="e", pady=2)

        # --- Card 2: Model Architecture Specs ---
        self.card_model = ttk.LabelFrame(self, text=" Model Specifications ", padding=10)
        self.card_model.grid(row=0, column=1, sticky="nsew", padx=4, pady=4)
        self.card_model.columnconfigure(0, weight=1)
        self.card_model.columnconfigure(1, weight=1)

        ttk.Label(self.card_model, text="Model Type:", font=(THEME["font_family"], 9, "bold")).grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.lbl_model_name = ttk.Label(
            self.card_model, text="Real Fly Connectome", font=(THEME["font_family"], 10, "bold")
        )
        self.lbl_model_name.grid(row=0, column=1, sticky="e", pady=2)

        ttk.Label(self.card_model, text="Neurons / Wiring:", font=(THEME["font_family"], 9)).grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.lbl_neurons = ttk.Label(self.card_model, text="922 neurons (sparse)", font=(THEME["font_family"], 9))
        self.lbl_neurons.grid(row=1, column=1, sticky="e", pady=2)

        ttk.Label(self.card_model, text="Trainable Weights:", font=(THEME["font_family"], 9)).grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.lbl_params = ttk.Label(self.card_model, text="~75,083 parameters", font=(THEME["font_family"], 9))
        self.lbl_params.grid(row=2, column=1, sticky="e", pady=2)

        ttk.Label(self.card_model, text="Training Steps:", font=(THEME["font_family"], 9)).grid(
            row=3, column=0, sticky="w", pady=2
        )
        self.lbl_steps = ttk.Label(self.card_model, text="~5,000,000 steps", font=(THEME["font_family"], 9))
        self.lbl_steps.grid(row=3, column=1, sticky="e", pady=2)

        # --- Card 3: Historical Evaluation Benchmark (eval_results.csv) ---
        self.card_bench = ttk.LabelFrame(self, text=" Benchmark Baselines (100 Test Mazes) ", padding=10)
        self.card_bench.grid(row=0, column=2, sticky="nsew", padx=4, pady=4)
        self.card_bench.columnconfigure(0, weight=1)
        self.card_bench.columnconfigure(1, weight=1)

        ttk.Label(self.card_bench, text="Standard Mazes:", font=(THEME["font_family"], 9)).grid(
            row=0, column=0, sticky="w", pady=2
        )
        self.lbl_bench_std = ttk.Label(
            self.card_bench, text="-- %", font=(THEME["font_family"], 9, "bold"), foreground="#059669"
        )
        self.lbl_bench_std.grid(row=0, column=1, sticky="e", pady=2)

        ttk.Label(self.card_bench, text="Hard Mazes (Unseen):", font=(THEME["font_family"], 9)).grid(
            row=1, column=0, sticky="w", pady=2
        )
        self.lbl_bench_hard = ttk.Label(self.card_bench, text="-- %", font=(THEME["font_family"], 9))
        self.lbl_bench_hard.grid(row=1, column=1, sticky="e", pady=2)

        ttk.Label(self.card_bench, text="Very Hard Mazes:", font=(THEME["font_family"], 9)).grid(
            row=2, column=0, sticky="w", pady=2
        )
        self.lbl_bench_vhard = ttk.Label(self.card_bench, text="-- %", font=(THEME["font_family"], 9))
        self.lbl_bench_vhard.grid(row=2, column=1, sticky="e", pady=2)

        ttk.Label(self.card_bench, text="Benchmark Mean Time:", font=(THEME["font_family"], 9)).grid(
            row=3, column=0, sticky="w", pady=2
        )
        self.lbl_bench_time = ttk.Label(self.card_bench, text="-- s", font=(THEME["font_family"], 9))
        self.lbl_bench_time.grid(row=3, column=1, sticky="e", pady=2)

    def update_model_info(self, info: CheckpointInfo) -> None:
        """Updates the Model and Benchmark cards when user alters model/seed."""
        meta = MODELS.get(info.model_key)
        if meta:
            self.lbl_model_name.config(text=meta.display_name)
            self.lbl_neurons.config(text=f"{meta.neurons} neurons")
            self.lbl_params.config(text=f"~{meta.params:,} parameters")

        if info.exists and info.training_steps > 0:
            self.lbl_steps.config(text=f"~{info.training_steps:,} steps (upd {info.training_updates})")
        else:
            self.lbl_steps.config(text="--")

        # Benchmark stats
        if info.benchmark_standard_success is not None:
            self.lbl_bench_std.config(text=f"{info.benchmark_standard_success * 100:.1f} %")
        else:
            self.lbl_bench_std.config(text="N/A")

        if info.benchmark_hard_success is not None:
            self.lbl_bench_hard.config(text=f"{info.benchmark_hard_success * 100:.1f} %")
        else:
            self.lbl_bench_hard.config(text="N/A")

        if info.benchmark_very_hard_success is not None:
            self.lbl_bench_vhard.config(text=f"{info.benchmark_very_hard_success * 100:.1f} %")
        else:
            self.lbl_bench_vhard.config(text="N/A")

        if info.benchmark_mean_time_s is not None and not str(info.benchmark_mean_time_s).lower() == "nan":
            self.lbl_bench_time.config(text=f"{info.benchmark_mean_time_s:.2f} s")
        else:
            self.lbl_bench_time.config(text="--")

    def update_live_summary(self, summary: RunSummary) -> None:
        """Updates the Live Performance card as episodes finish."""
        self.lbl_success.config(
            text=f"{summary.success_rate:.1f}% ({summary.goals}/{summary.total_episodes})"
        )
        if summary.avg_time_s > 0:
            self.lbl_avg_time.config(text=f"{summary.avg_time_s:.2f} s")
            self.lbl_min_max_time.config(text=f"{summary.min_time_s:.1f}s / {summary.max_time_s:.1f}s")
        else:
            self.lbl_avg_time.config(text="-- s")
            self.lbl_min_max_time.config(text="-- / --")

        self.lbl_breakdown.config(
            text=f"{summary.crashes} crashes / {summary.timeouts} timeouts"
        )

    def reset_live_summary(self) -> None:
        """Resets the Live Performance card to initial state."""
        self.lbl_success.config(text="-- % (0/0)")
        self.lbl_avg_time.config(text="-- s")
        self.lbl_min_max_time.config(text="-- / --")
        self.lbl_breakdown.config(text="0 / 0")
