"""
Control panel view for parameter configuration, command generation, and launch controls.
"""
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, Optional

from gui.config import (
    AVAILABLE_SEEDS,
    AVAILABLE_SIZES,
    DEFAULT_EPISODES,
    DEFAULT_MAZE_SIZE,
    DEFAULT_MODEL_KEY,
    DEFAULT_SEED,
    DEFAULT_SPEED,
    MODELS,
    SPEED_OPTIONS,
    THEME,
)
from gui.data_manager import DataManager


class ControlsView(ttk.LabelFrame):
    """Configuration controls: Model, Seed (0..4), Maze Size, Speed, Episodes, and Launch/Stop."""

    def __init__(
        self,
        parent: tk.Widget,
        data_manager: DataManager,
        on_launch: Callable[..., None],
        on_stop: Callable[[], None],
        on_change: Optional[Callable[[], None]] = None,
    ):
        super().__init__(parent, text=" Simulation Parameters ", padding=8)
        self.data_manager = data_manager
        self.on_launch = on_launch
        self.on_stop = on_stop
        self.on_change = on_change

        # Observable Tk variables
        self.var_model = tk.StringVar(value=DEFAULT_MODEL_KEY)
        self.var_seed = tk.IntVar(value=DEFAULT_SEED)
        self.var_size = tk.IntVar(value=DEFAULT_MAZE_SIZE)
        self.var_episodes = tk.IntVar(value=DEFAULT_EPISODES)
        self.var_speed = tk.DoubleVar(value=DEFAULT_SPEED)
        self.var_embedded = tk.BooleanVar(value=True)
        self.var_command = tk.StringVar()

        self._build_ui()
        self._update_command_preview()

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)

        # --- Section 1: Model Selection ---
        lbl_model = ttk.Label(self, text="Select Model Architecture:", font=(THEME["font_family"], 9, "bold"))
        lbl_model.grid(row=0, column=0, sticky="w", pady=(0, 2))

        model_frame = ttk.Frame(self)
        model_frame.grid(row=1, column=0, sticky="ew", pady=(0, 4))

        self.model_radios = {}
        row_idx = 0
        for key, meta in MODELS.items():
            r = ttk.Radiobutton(
                model_frame,
                text=f"{meta.display_name} ({meta.neurons} neurons, ~{meta.params:,} wts)",
                value=key,
                variable=self.var_model,
                command=self._on_input_changed,
            )
            r.grid(row=row_idx, column=0, sticky="w", pady=1)
            self.model_radios[key] = r
            row_idx += 1

        # --- Section 2: Seed & Size Selectors ---
        param_grid = ttk.Frame(self)
        param_grid.grid(row=2, column=0, sticky="ew", pady=(0, 4))
        param_grid.columnconfigure(1, weight=1)

        # Seed Selector (0 to 4)
        lbl_seed = ttk.Label(param_grid, text="Seed:", font=(THEME["font_family"], 9, "bold"))
        lbl_seed.grid(row=0, column=0, sticky="w", padx=(0, 4), pady=2)

        seed_frame = ttk.Frame(param_grid)
        seed_frame.grid(row=0, column=1, sticky="w", pady=2)
        for s in AVAILABLE_SEEDS:
            rb = ttk.Radiobutton(
                seed_frame,
                text=f"s{s}",
                value=s,
                variable=self.var_seed,
                command=self._on_input_changed,
            )
            rb.pack(side="left", padx=2)

        # Maze Size Selector (4 to 8)
        lbl_size = ttk.Label(param_grid, text="Maze Size:", font=(THEME["font_family"], 9, "bold"))
        lbl_size.grid(row=1, column=0, sticky="w", padx=(0, 4), pady=2)

        size_frame = ttk.Frame(param_grid)
        size_frame.grid(row=1, column=1, sticky="w", pady=2)
        for sz in AVAILABLE_SIZES:
            rb = ttk.Radiobutton(
                size_frame,
                text=f"{sz}x{sz}",
                value=sz,
                variable=self.var_size,
                command=self._on_input_changed,
            )
            rb.pack(side="left", padx=2)

        # --- Section 3: Runtime Options (Episodes, Speed, Viewer Mode) ---
        opts_frame = ttk.Frame(self)
        opts_frame.grid(row=3, column=0, sticky="ew", pady=(0, 4))

        lbl_ep = ttk.Label(opts_frame, text="Episodes:")
        lbl_ep.pack(side="left", padx=(0, 2))
        sp_ep = ttk.Spinbox(
            opts_frame,
            from_=1,
            to=50,
            width=4,
            textvariable=self.var_episodes,
            command=self._on_input_changed,
        )
        sp_ep.pack(side="left", padx=(0, 8))

        lbl_spd = ttk.Label(opts_frame, text="Speed:")
        lbl_spd.pack(side="left", padx=(0, 2))
        cb_spd = ttk.Combobox(
            opts_frame,
            values=[f"{s}x" for s in SPEED_OPTIONS],
            width=5,
            state="readonly",
        )
        cb_spd.set(f"{DEFAULT_SPEED}x")
        cb_spd.pack(side="left", padx=(0, 8))
        cb_spd.bind("<<ComboboxSelected>>", lambda e: self._on_speed_selected(cb_spd.get()))

        chk_embed = ttk.Checkbutton(
            opts_frame,
            text="Embed in GUI",
            variable=self.var_embedded,
            command=self._on_input_changed,
        )
        chk_embed.pack(side="left")

        # --- Section 4: Command Preview ---
        cmd_box = ttk.LabelFrame(self, text=" Generated Command Preview ", padding=4)
        cmd_box.grid(row=4, column=0, sticky="ew", pady=(0, 4))
        cmd_box.columnconfigure(0, weight=1)

        ent_cmd = ttk.Entry(
            cmd_box, textvariable=self.var_command, state="readonly", font=(THEME["font_mono"], 8)
        )
        ent_cmd.grid(row=0, column=0, sticky="ew", padx=(0, 4))

        btn_copy = ttk.Button(cmd_box, text="Copy", width=5, command=self._copy_command)
        btn_copy.grid(row=0, column=1)

        self.lbl_ckpt_status = ttk.Label(
            self,
            text="Checking checkpoint...",
            font=(THEME["font_family"], 8, "italic"),
        )
        self.lbl_ckpt_status.grid(row=5, column=0, sticky="w", pady=(0, 4))

        # --- Section 5: Launch & Stop Action Buttons ---
        btn_bar = ttk.Frame(self)
        btn_bar.grid(row=6, column=0, sticky="ew")
        btn_bar.columnconfigure(0, weight=1)
        btn_bar.columnconfigure(1, weight=1)

        self.btn_launch = tk.Button(
            btn_bar,
            text="▶  Launch Maze Simulation",
            font=(THEME["font_family"], 10, "bold"),
            bg="#059669",
            fg="#ffffff",
            activebackground="#047857",
            activeforeground="#ffffff",
            relief="raised",
            bd=2,
            padx=8,
            pady=6,
            cursor="hand2",
            command=self._on_launch_clicked,
        )
        self.btn_launch.grid(row=0, column=0, sticky="ew", padx=(0, 3))

        self.btn_stop = tk.Button(
            btn_bar,
            text="⏹  Stop",
            font=(THEME["font_family"], 10, "bold"),
            bg="#dc2626",
            fg="#ffffff",
            activebackground="#b91c1c",
            activeforeground="#ffffff",
            relief="raised",
            bd=2,
            padx=8,
            pady=6,
            cursor="hand2",
            state="disabled",
            command=self.on_stop,
        )
        self.btn_stop.grid(row=0, column=1, sticky="ew", padx=(3, 0))

    def _on_speed_selected(self, val_str: str) -> None:
        val = float(val_str.rstrip("x"))
        self.var_speed.set(val)
        self._on_input_changed()

    def _on_input_changed(self) -> None:
        self._update_command_preview()
        if self.on_change:
            self.on_change()

    def _update_command_preview(self) -> None:
        model = self.var_model.get()
        seed = self.var_seed.get()
        size = self.var_size.get()
        ep = self.var_episodes.get()
        speed = self.var_speed.get()
        embed = self.var_embedded.get()

        info = self.data_manager.get_checkpoint_info(model, seed)
        run_name = info.run_name

        cmd_parts = [f"python src/watch.py {run_name}", f"--size {size}"]
        if ep != DEFAULT_EPISODES:
            cmd_parts.append(f"--episodes {ep}")
        if speed != 1.0:
            cmd_parts.append(f"--speed {speed}")
        if not embed:
            cmd_parts.append("(External PyBullet Window)")

        self.var_command.set(" ".join(cmd_parts))

        if info.exists:
            status_text = f"✔ Found: {info.run_name} (Updates: {info.training_updates:,}, Steps: ~{info.training_steps:,})"
            self.lbl_ckpt_status.config(text=status_text, foreground="#059669")
            self.btn_launch.config(state="normal")
        else:
            status_text = f"✖ Checkpoint not found: {info.run_name}"
            self.lbl_ckpt_status.config(text=status_text, foreground="#dc2626")
            self.btn_launch.config(state="disabled")

    def _copy_command(self) -> None:
        self.clipboard_clear()
        self.clipboard_append(self.var_command.get())
        messagebox.showinfo("Copied", "Command copied to clipboard!")

    def _on_launch_clicked(self) -> None:
        model = self.var_model.get()
        seed = self.var_seed.get()
        size = self.var_size.get()
        ep = max(1, self.var_episodes.get())
        speed = self.var_speed.get()
        embed = self.var_embedded.get()

        info = self.data_manager.get_checkpoint_info(model, seed)
        self.on_launch(
            run_name=info.run_name,
            size=size,
            episodes=ep,
            speed=speed,
            embedded=embed,
        )

    def set_running_state(self, running: bool) -> None:
        """Toggles controls availability based on simulation running state."""
        if running:
            self.btn_launch.config(state="disabled", bg="#6b7280")
            self.btn_stop.config(state="normal", bg="#dc2626")
            for r in self.model_radios.values():
                r.config(state="disabled")
        else:
            info = self.data_manager.get_checkpoint_info(self.var_model.get(), self.var_seed.get())
            self.btn_launch.config(state="normal" if info.exists else "disabled", bg="#059669")
            self.btn_stop.config(state="disabled", bg="#6b7280")
            for r in self.model_radios.values():
                r.config(state="normal")
