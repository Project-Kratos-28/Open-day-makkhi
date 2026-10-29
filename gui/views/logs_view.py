"""
Console log viewer for real-time simulator output inspection.
"""
import tkinter as tk
from tkinter import ttk, messagebox
from typing import Optional

from gui.config import THEME


class LogsView(ttk.Frame):
    """Terminal output viewer with auto-scroll and clipboard copy."""

    def __init__(self, parent: tk.Widget):
        super().__init__(parent, padding=6)
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)

        self.var_autoscroll = tk.BooleanVar(value=True)
        self._build_ui()

    def _build_ui(self) -> None:
        text_frame = ttk.Frame(self)
        text_frame.grid(row=0, column=0, sticky="nsew")
        text_frame.columnconfigure(0, weight=1)
        text_frame.rowconfigure(0, weight=1)

        self.txt_log = tk.Text(
            text_frame,
            wrap="none",
            bg="#0f172a",
            fg="#f8fafc",
            insertbackground="#38bdf8",
            font=(THEME["font_mono"], 9),
            padx=8,
            pady=8,
            height=10,
        )
        self.txt_log.grid(row=0, column=0, sticky="nsew")

        v_scroll = ttk.Scrollbar(text_frame, orient="vertical", command=self.txt_log.yview)
        v_scroll.grid(row=0, column=1, sticky="ns")

        h_scroll = ttk.Scrollbar(text_frame, orient="horizontal", command=self.txt_log.xview)
        h_scroll.grid(row=1, column=0, sticky="ew")

        self.txt_log.configure(xscrollcommand=h_scroll.set, yscrollcommand=v_scroll.set)

        # Bottom bar
        bar = ttk.Frame(self)
        bar.grid(row=1, column=0, sticky="ew", pady=(6, 0))

        chk_scroll = ttk.Checkbutton(bar, text="Auto-scroll to latest output", variable=self.var_autoscroll)
        chk_scroll.pack(side="left")

        btn_copy = ttk.Button(bar, text="Copy Logs", width=12, command=self._copy_logs)
        btn_copy.pack(side="right", padx=(4, 0))

        btn_clear = ttk.Button(bar, text="Clear Logs", width=12, command=self.clear)
        btn_clear.pack(side="right")

    def append_line(self, line: str) -> None:
        """Appends a new line of text to the console display."""
        self.txt_log.insert("end", line + "\n")
        if self.var_autoscroll.get():
            self.txt_log.see("end")

    def clear(self) -> None:
        """Clears the console text."""
        self.txt_log.delete("1.0", "end")

    def _copy_logs(self) -> None:
        content = self.txt_log.get("1.0", "end-1c")
        if not content:
            messagebox.showinfo("Copy Logs", "Log console is empty.")
            return
        self.clipboard_clear()
        self.clipboard_append(content)
        messagebox.showinfo("Copy Logs", "All console logs copied to clipboard!")
