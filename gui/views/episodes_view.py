"""
Episodes table view displaying individual maze completion times, outcomes, and progress.
"""
import csv
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import List

from gui.config import THEME
from gui.runner import EpisodeResult


class EpisodesView(ttk.Frame):
    """Table showing each completed maze episode, its outcome, and completion time."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, padding=4)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self._episodes: List[EpisodeResult] = []
        self._build_ui()

    def _build_ui(self) -> None:
        table_frame = ttk.Frame(self)
        table_frame.grid(row=0, column=0, sticky="nsew")
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)

        cols = ("episode", "outcome", "time", "rate", "notes")
        self.tree = ttk.Treeview(
            table_frame,
            columns=cols,
            show="headings",
            selectmode="browse",
            height=8,
        )

        self.tree.heading("episode", text="Ep #")
        self.tree.heading("outcome", text="Outcome")
        self.tree.heading("time", text="Time (s)")
        self.tree.heading("rate", text="Success")
        self.tree.heading("notes", text="Flight Notes")

        self.tree.column("episode", width=60, minwidth=45, anchor="center")
        self.tree.column("outcome", width=105, minwidth=85, anchor="center")
        self.tree.column("time", width=80, minwidth=65, anchor="center")
        self.tree.column("rate", width=85, minwidth=70, anchor="center")
        self.tree.column("notes", width=140, minwidth=100, anchor="w")

        v_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        h_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=v_scroll.set, xscrollcommand=h_scroll.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        v_scroll.grid(row=0, column=1, sticky="ns")
        h_scroll.grid(row=1, column=0, sticky="ew")

        # Color tags
        self.tree.tag_configure("goal", foreground="#047857", font=(THEME["font_family"], 9, "bold"))
        self.tree.tag_configure("crash", foreground="#b91c1c")
        self.tree.tag_configure("timeout", foreground="#b45309")

        # Action bar under table
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="ew", pady=(4, 0))

        self.lbl_count = ttk.Label(bar, text="Mazes: 0", font=(THEME["font_family"], 8))
        self.lbl_count.pack(side="left")

        btn_export = ttk.Button(bar, text="Export CSV", width=10, command=self._export_csv)
        btn_export.pack(side="right", padx=(2, 0))

        btn_clear = ttk.Button(bar, text="Clear", width=6, command=self.clear)
        btn_clear.pack(side="right")

    def add_episode(self, result: EpisodeResult) -> None:
        """Appends a new episode row and updates session success count."""
        self._episodes.append(result)

        total = len(self._episodes)
        goals = sum(1 for e in self._episodes if e.outcome == "goal")
        rate_str = f"{(goals / total) * 100:.0f}% ({goals}/{total})"

        if result.outcome == "goal":
            outcome_text = "GOAL"
            tag = "goal"
            note = f"Navigated in {result.time_s:.2f}s"
        elif result.outcome == "crash":
            outcome_text = "CRASH"
            tag = "crash"
            note = f"Collided at {result.time_s:.2f}s"
        else:
            outcome_text = "TIMEOUT"
            tag = "timeout"
            note = f"Max time exceeded ({result.time_s:.2f}s)"

        self.tree.insert(
            "",
            "end",
            values=(
                f"#{result.episode}",
                outcome_text,
                f"{result.time_s:.2f} s",
                rate_str,
                note,
            ),
            tags=(tag,),
        )

        children = self.tree.get_children()
        if children:
            self.tree.see(children[-1])

        self.lbl_count.config(text=f"Mazes: {total} ({goals} goals)")

    def clear(self) -> None:
        """Clears all rows from the table."""
        self._episodes.clear()
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.lbl_count.config(text="Mazes: 0")

    def _export_csv(self) -> None:
        if not self._episodes:
            messagebox.showinfo("Export CSV", "No episode records to export.")
            return

        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            title="Save Maze Run Results",
        )
        if not path:
            return

        try:
            with open(path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Episode", "Outcome", "Time_Taken_Seconds", "Timestamp"])
                for r in self._episodes:
                    writer.writerow([r.episode, r.outcome, r.time_s, r.timestamp])
            messagebox.showinfo("Export CSV", f"Results exported to {path}!")
        except Exception as e:
            messagebox.showerror("Export Error", f"Failed to save CSV: {e}")
