"""
High-resolution 3D simulation viewer rendering PyBullet camera frames directly into a Tkinter canvas.
Supports custom HD resolutions, 3D shadows, lighting, and aspect-ratio-preserving crisp scaling.
"""
import tkinter as tk
from tkinter import ttk
from typing import Any, Callable, Dict, Optional, Tuple

from PIL import Image, ImageTk

from gui.config import THEME

RESOLUTION_PRESETS = {
    "High (800x600)": (800, 600),
    "Ultra HD (1024x768)": (1024, 768),
    "HD Wide (960x540)": (960, 540),
    "Standard (640x480)": (640, 480),
}
DEFAULT_RESOLUTION_NAME = "High (800x600)"


class ViewerView(ttk.LabelFrame):
    """High-definition 3D viewport displaying real-time simulation graphics and HUD telemetry."""

    def __init__(
        self,
        parent: tk.Widget,
        on_camera_change: Optional[Callable[[str], None]] = None,
        on_resolution_change: Optional[Callable[[int, int], None]] = None,
        on_shadows_change: Optional[Callable[[bool], None]] = None,
    ):
        super().__init__(parent, text=" 3D Simulation Viewer (High-Resolution Live Viewport) ", padding=6)
        self.on_camera_change = on_camera_change
        self.on_resolution_change = on_resolution_change
        self.on_shadows_change = on_shadows_change

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        self._photo_image: Optional[ImageTk.PhotoImage] = None
        self._canvas_width = 800
        self._canvas_height = 540

        self.var_camera = tk.StringVar(value="overview")
        self.var_resolution = tk.StringVar(value=DEFAULT_RESOLUTION_NAME)
        self.var_shadows = tk.BooleanVar(value=True)

        self._build_ui()
        self._show_idle_placeholder()

    def _build_ui(self) -> None:
        # Top toolbar above canvas
        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        toolbar.columnconfigure(6, weight=1)

        # 1. Camera View Selector
        lbl_cam = ttk.Label(toolbar, text="Camera:", font=(THEME["font_family"], 9, "bold"))
        lbl_cam.grid(row=0, column=0, sticky="w", padx=(0, 4))

        cb_cam = ttk.Combobox(
            toolbar,
            textvariable=self.var_camera,
            values=["overview", "chase"],
            state="readonly",
            width=10,
        )
        cb_cam.grid(row=0, column=1, sticky="w", padx=(0, 10))
        cb_cam.bind("<<ComboboxSelected>>", self._on_cam_selected)

        # 2. Resolution Selector (HD / Ultra)
        lbl_res = ttk.Label(toolbar, text="Resolution:", font=(THEME["font_family"], 9, "bold"))
        lbl_res.grid(row=0, column=2, sticky="w", padx=(0, 4))

        cb_res = ttk.Combobox(
            toolbar,
            textvariable=self.var_resolution,
            values=list(RESOLUTION_PRESETS.keys()),
            state="readonly",
            width=18,
        )
        cb_res.grid(row=0, column=3, sticky="w", padx=(0, 10))
        cb_res.bind("<<ComboboxSelected>>", self._on_res_selected)

        # 3. Shadows & 3D Lighting Toggle
        chk_shadow = ttk.Checkbutton(
            toolbar,
            text="3D Shadows",
            variable=self.var_shadows,
            command=self._on_shadow_toggle,
        )
        chk_shadow.grid(row=0, column=4, sticky="w", padx=(0, 10))

        # 4. Live Status Badge
        self.lbl_view_status = tk.Label(
            toolbar,
            text="● STANDBY",
            font=(THEME["font_family"], 9, "bold"),
            bg="#1e293b",
            fg="#94a3b8",
            padx=8,
            pady=2,
            relief="groove",
        )
        self.lbl_view_status.grid(row=0, column=6, sticky="e")

        # Canvas for displaying rendered high-definition RGB frames
        self.canvas = tk.Canvas(
            self,
            width=self._canvas_width,
            height=self._canvas_height,
            bg="#0b1120",  # Deep navy black
            highlightthickness=1,
            highlightbackground="#334155",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew")
        self.canvas.bind("<Configure>", self._on_canvas_resize)

        # Bottom HUD Telemetry Strip
        hud_bar = tk.Frame(self, bg="#1e293b", padx=10, pady=4)
        hud_bar.grid(row=2, column=0, sticky="ew", pady=(4, 0))
        hud_bar.columnconfigure(0, weight=1)
        hud_bar.columnconfigure(1, weight=1)
        hud_bar.columnconfigure(2, weight=1)
        hud_bar.columnconfigure(3, weight=1)
        hud_bar.columnconfigure(4, weight=1)

        self.lbl_hud_ep = tk.Label(
            hud_bar, text="Episode: --", bg="#1e293b", fg="#f8fafc", font=(THEME["font_mono"], 9)
        )
        self.lbl_hud_ep.grid(row=0, column=0, sticky="w")

        self.lbl_hud_time = tk.Label(
            hud_bar, text="Time: 0.0 s", bg="#1e293b", fg="#38bdf8", font=(THEME["font_mono"], 9, "bold")
        )
        self.lbl_hud_time.grid(row=0, column=1, sticky="w")

        self.lbl_hud_pos = tk.Label(
            hud_bar, text="Pos: [0.0, 0.0, 0.0]", bg="#1e293b", fg="#94a3b8", font=(THEME["font_mono"], 9)
        )
        self.lbl_hud_pos.grid(row=0, column=2, sticky="w")

        self.lbl_hud_spd = tk.Label(
            hud_bar, text="Speed: 0.0 m/s", bg="#1e293b", fg="#94a3b8", font=(THEME["font_mono"], 9)
        )
        self.lbl_hud_spd.grid(row=0, column=3, sticky="w")

        self.lbl_hud_res = tk.Label(
            hud_bar, text="Res: 800x600", bg="#1e293b", fg="#64748b", font=(THEME["font_mono"], 8)
        )
        self.lbl_hud_res.grid(row=0, column=4, sticky="e")

    def _on_cam_selected(self, event=None) -> None:
        if self.on_camera_change:
            self.on_camera_change(self.var_camera.get())

    def _on_res_selected(self, event=None) -> None:
        preset_name = self.var_resolution.get()
        w, h = RESOLUTION_PRESETS.get(preset_name, (800, 600))
        self.lbl_hud_res.config(text=f"Res: {w}x{h}")
        if self.on_resolution_change:
            self.on_resolution_change(w, h)

    def _on_shadow_toggle(self) -> None:
        if self.on_shadows_change:
            self.on_shadows_change(self.var_shadows.get())

    def _on_canvas_resize(self, event: tk.Event) -> None:
        if event.width > 100 and event.height > 100:
            self._canvas_width = event.width
            self._canvas_height = event.height

    def update_frame(self, pil_image: Image.Image, telemetry: Dict[str, Any]) -> None:
        """Draws incoming camera frame preserving aspect ratio with crisp Lanczos scaling."""
        cw = max(self.canvas.winfo_width(), 100)
        ch = max(self.canvas.winfo_height(), 100)
        img_w, img_h = pil_image.size

        # Compute aspect-preserving scale
        scale = min(float(cw) / float(img_w), float(ch) / float(img_h))
        target_w = max(1, int(img_w * scale))
        target_h = max(1, int(img_h * scale))

        if (target_w, target_h) == pil_image.size:
            display_img = pil_image
        else:
            # Use high-quality Lanczos filter for razor-sharp downscaling/upscaling
            display_img = pil_image.resize((target_w, target_h), Image.Resampling.LANCZOS)

        self._photo_image = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(cw // 2, ch // 2, image=self._photo_image, anchor="center")

        # Update HUD telemetry
        ep = telemetry.get("episode", 1)
        flight_time = telemetry.get("time_s", 0.0)
        pos = telemetry.get("pos", [0.0, 0.0, 0.0])
        spd = telemetry.get("speed", 0.0)
        raw_status = telemetry.get("status")
        status = str(raw_status).upper() if raw_status else "FLYING"

        self.lbl_hud_ep.config(text=f"Episode: #{ep}")
        self.lbl_hud_time.config(text=f"Time: {flight_time:.1f} s")
        self.lbl_hud_pos.config(text=f"Pos: [{pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}]")
        self.lbl_hud_spd.config(text=f"Speed: {spd:.2f} m/s")
        self.lbl_hud_res.config(text=f"Render: {img_w}x{img_h}")

        if status == "GOAL":
            self.lbl_view_status.config(text="● GOAL REACHED", fg="#10b981")
        elif status == "CRASH":
            self.lbl_view_status.config(text="● CRASHED", fg="#ef4444")
        elif status == "TIMEOUT":
            self.lbl_view_status.config(text="● TIMEOUT", fg="#f59e0b")
        else:
            self.lbl_view_status.config(text="● FLYING", fg="#38bdf8")

    def show_completed(self, outcome: str, flight_time: float) -> None:
        """Displays completion status overlay when an episode concludes."""
        if outcome == "goal":
            self.lbl_view_status.config(text=f"🎯 GOAL REACHED ({flight_time:.1f}s)", fg="#10b981")
        elif outcome == "crash":
            self.lbl_view_status.config(text=f"💥 CRASHED ({flight_time:.1f}s)", fg="#ef4444")
        else:
            self.lbl_view_status.config(text=f"⏱️ TIMEOUT ({flight_time:.1f}s)", fg="#f59e0b")

    def _show_idle_placeholder(self) -> None:
        """Displays aesthetic placeholder graphics when idle."""
        self.canvas.delete("all")
        w = self._canvas_width
        h = self._canvas_height
        self.canvas.create_rectangle(0, 0, w, h, fill="#0b1120", outline="")
        self.canvas.create_text(
            w // 2,
            h // 2 - 24,
            text="🪰 High-Definition 3D Viewport",
            font=(THEME["font_family"], 14, "bold"),
            fill="#38bdf8",
        )
        self.canvas.create_text(
            w // 2,
            h // 2 + 12,
            text="Select your model and click '▶ Launch Maze Simulation' to begin.",
            font=(THEME["font_family"], 10),
            fill="#94a3b8",
        )
        self.canvas.create_text(
            w // 2,
            h // 2 + 36,
            text="Preset: High Definition (800x600) with 3D Shadows & Lighting enabled.",
            font=(THEME["font_family"], 9, "italic"),
            fill="#64748b",
        )
        self.lbl_view_status.config(text="● STANDBY", fg="#94a3b8")
