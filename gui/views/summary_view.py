"""
Summary view displaying research context, model comparison, and key findings for Open Day.
"""
import tkinter as tk
from tkinter import ttk

from gui.config import THEME


class SummaryView(ttk.Frame):
    """Informational view summarizing the 4 models and scientific hypothesis."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, padding=8)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self._build_ui()

    def _build_ui(self) -> None:
        text_frame = ttk.Frame(self)
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.txt = tk.Text(
            text_frame,
            wrap="word",
            bg="#0f172a",
            fg="#f8fafc",
            font=(THEME["font_family"], 9),
            padx=10,
            pady=10,
            relief="flat",
        )
        self.txt.grid(row=0, column=0, sticky="nsew")

        scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.txt.yview)
        scroll.grid(row=0, column=1, sticky="ns")
        self.txt.configure(yscrollcommand=scroll.set)

        summary_content = (
            "OPEN DAY RESEARCH DEMO: Fruit Fly Central Complex → Drone Navigation\n"
            "====================================================================\n\n"
            "1. THE RESEARCH GOAL:\n"
            "   Can the biological wiring of an insect's navigation centre fly a drone through mazes?\n"
            "   We extracted 922 neurons and 44,723 connections from the Drosophila male CNS connectome\n"
            "   (Hulse et al. 2021, eLife). The topology and Dale's law signs (excitatory vs inhibitory)\n"
            "   are frozen from biology; only connection strengths are learned via Reinforcement Learning (PPO).\n\n"
            "2. THE 4 SIZE-MATCHED MODELS (~75k trainable parameters each):\n"
            "   • Real Fly Connectome (conn): 922 biological neurons. Fixed biological wiring & Dale's law signs.\n"
            "   • Shuffled Fly Control (shuf): 922 neurons. Exact same sparsity & degree distribution, scrambled pairs.\n"
            "   • Standard GRU (gru): 139 neurons. Conventional deep learning memory network with gating.\n"
            "   • Dense RNN Baseline (base): 254 neurons. Unconstrained all-to-all RNN with same leaky neuron math.\n\n"
            "3. KEY FINDINGS:\n"
            "   • Real fly wiring is the most reliable: 0 seed collapses across all 5 seeds.\n"
            "   • Learns significantly faster in early training (~85% at 2.2M steps vs 58% for GRU and 42% for Dense).\n"
            "   • Outperforms the parameter-matched Dense network at every difficulty level.\n"
            "   • Standard GRU reaches comparable asymptotic performance on familiar mazes, but fly wiring avoids\n"
            "     the local-minima 'hover trap' and requires simpler, ungated biological neurons.\n"
        )
        self.txt.insert("1.0", summary_content)
        self.txt.configure(state="disabled")
