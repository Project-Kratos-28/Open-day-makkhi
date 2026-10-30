"""
Main Application window for the Fly Connectome Drone Maze GUI.
Left Column: Parameters, Launch/Stop buttons, and Tabbed Notebook (Completion Times, Output Log, Summary).
Right Column: Metrics Cards and Dedicated Full-Size 3D Simulation Viewer.
"""
import queue
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Any, Dict, Optional

from PIL import Image

from gui.config import PROJECT_ROOT, THEME
from gui.data_manager import DataManager
from gui.runner import EpisodeResult, RunSummary, SimulationRunner
from gui.views import (
    ControlsView,
    EpisodesView,
    LogsView,
    MetricsView,
    SummaryView,
    ViewerView,
)


class DroneMazeApp(tk.Tk):
    """Main interactive application window with streamlined side-by-side layout."""

    def __init__(self):
        super().__init__()
        self.title("Fruit Fly Connectome Drone Navigation | Makkhi GUI")
        self.geometry("1200x820")
        self.minsize(1040, 700)

        self._apply_styles()

        self.data_manager = DataManager()
        self.msg_queue: queue.Queue = queue.Queue()
        self._frame_lock = threading.Lock()
        self._latest_frame = None

        self._popout_window: Optional[tk.Toplevel] = None
        self._popout_viewer: Optional[ViewerView] = None
        self._placeholder_frame: Optional[tk.Frame] = None

        self.runner = SimulationRunner(
            on_frame=self._queue_frame,
            on_line=self._queue_line,
            on_episode=self._queue_episode,
            on_summary=self._queue_summary,
            on_status=self._queue_status,
            on_finished=self._queue_finished,
        )

        self._build_layout()
        self._poll_queue()

        self.protocol("WM_DELETE_WINDOW", self._on_close_app)

        # Initial synchronization
        self._on_parameters_changed()

    def _apply_styles(self) -> None:
        style = ttk.Style(self)
        if "clam" in style.theme_names():
            style.theme_use("clam")

        style.configure(".", font=(THEME["font_family"], 9))
        style.configure("TLabelFrame", relief="solid", borderwidth=1)
        style.configure("TLabelFrame.Label", font=(THEME["font_family"], 9, "bold"))
        style.configure("Treeview.Heading", font=(THEME["font_family"], 9, "bold"))
        style.configure("Treeview", rowheight=24)

    def _build_layout(self) -> None:
        # Header banner
        header = tk.Frame(self, bg="#0f172a", padx=16, pady=10)
        header.pack(side="top", fill="x")

        lbl_title = tk.Label(
            header,
            text="Fruit Fly Connectome - Drone Maze Navigation",
            font=(THEME["font_family"], 13, "bold"),
            bg="#0f172a",
            fg="#38bdf8",
        )
        lbl_title.pack(side="left")

        self.lbl_head_status = tk.Label(
            header,
            text="● Ready",
            font=(THEME["font_family"], 9, "bold"),
            bg="#0f172a",
            fg="#10b981",
        )
        self.lbl_head_status.pack(side="right")

        # Main workspace (Two columns: Left = Controls & Tabbed Results, Right = Metrics & 3D Viewer)
        workspace = ttk.Frame(self, padding=8)
        workspace.pack(side="top", fill="both", expand=True)
        workspace.columnconfigure(0, weight=0, minsize=440)
        workspace.columnconfigure(1, weight=1)
        workspace.rowconfigure(0, weight=1)

        # ==========================================
        # LEFT COLUMN: Parameters + Tabbed Notebook
        # ==========================================
        left_col = ttk.Frame(workspace)
        left_col.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        left_col.columnconfigure(0, weight=1)
        left_col.rowconfigure(0, weight=0)
        left_col.rowconfigure(1, weight=1)

        # 1. Top of Left Column: Parameters & Launch/Stop Buttons
        self.controls_view = ControlsView(
            left_col,
            data_manager=self.data_manager,
            on_launch=self._launch_simulation,
            on_stop=self._stop_simulation,
            on_change=self._on_parameters_changed,
            on_viewer_toggle=self._on_viewer_mode_changed,
        )
        self.controls_view.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        # 2. Directly below Launch and Stop: Tabbed Notebook (Times, Logs, Summary)
        self.left_notebook = ttk.Notebook(left_col)
        self.left_notebook.grid(row=1, column=0, sticky="nsew")

        # Tab 1: Completion Times & Episodes Table
        self.episodes_view = EpisodesView(self.left_notebook)
        self.left_notebook.add(self.episodes_view, text="Completion Times  ")

        # Tab 2: Simulation Output Log
        self.logs_view = LogsView(self.left_notebook)
        self.left_notebook.add(self.logs_view, text="Output Log  ")

        # Tab 3: Model Summary & Research Context
        self.summary_view = SummaryView(self.left_notebook)
        self.left_notebook.add(self.summary_view, text="Research Summary  ")

        # ==========================================
        # RIGHT COLUMN: Metrics Cards + 3D Viewport
        # ==========================================
        right_col = ttk.Frame(workspace)
        right_col.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
        right_col.columnconfigure(0, weight=1)
        right_col.rowconfigure(0, weight=0)
        right_col.rowconfigure(1, weight=1)
        self.right_col = right_col

        # Top: KPI Metrics Cards
        self.metrics_view = MetricsView(right_col)
        self.metrics_view.grid(row=0, column=0, sticky="ew", pady=(0, 6))

        # Main: Full-size Dedicated 3D Simulation Viewport
        self.embedded_viewer = ViewerView(
            right_col,
            on_camera_change=self.runner.set_camera_mode,
            on_resolution_change=self.runner.set_render_resolution,
            on_dock_toggle=self._undock_viewer_to_popout,
            is_popout=False,
        )
        self.embedded_viewer.grid(row=1, column=0, sticky="nsew")
        self.viewer_view = self.embedded_viewer  # Backward compatibility
        self.active_viewer = self.embedded_viewer

        # Placeholder frame for when Sim Viewer is displayed in a different view
        self._placeholder_frame = tk.Frame(right_col, bg="#0b1120", bd=1, relief="solid")
        self._placeholder_frame.columnconfigure(0, weight=1)
        self._placeholder_frame.rowconfigure(0, weight=1)

        ph_inner = tk.Frame(self._placeholder_frame, bg="#0b1120", padx=20, pady=20)
        ph_inner.grid(row=0, column=0)

        self.lbl_ph_title = tk.Label(
            ph_inner,
            text="🖥️ Sim Viewer: Running in Different View",
            font=(THEME["font_family"], 13, "bold"),
            bg="#0b1120",
            fg="#38bdf8",
        )
        self.lbl_ph_title.pack(pady=(0, 10))

        self.lbl_ph_desc = tk.Label(
            ph_inner,
            text="",
            font=(THEME["font_family"], 9),
            bg="#0b1120",
            fg="#94a3b8",
            justify="center",
        )
        self.lbl_ph_desc.pack(pady=(0, 16))

        ph_btn_bar = tk.Frame(ph_inner, bg="#0b1120")
        ph_btn_bar.pack()

        self.btn_ph_front = tk.Button(
            ph_btn_bar,
            text="⤢ Bring Viewer to Front",
            font=(THEME["font_family"], 9, "bold"),
            bg="#0284c7",
            fg="#ffffff",
            activebackground="#0369a1",
            activeforeground="#ffffff",
            padx=10,
            pady=5,
            cursor="hand2",
            command=self._focus_popout_window,
        )
        self.btn_ph_front.pack(side="left", padx=6)

        self.btn_ph_dock = tk.Button(
            ph_btn_bar,
            text="🗗 Dock Sim Viewer to Main GUI",
            font=(THEME["font_family"], 9, "bold"),
            bg="#059669",
            fg="#ffffff",
            activebackground="#047857",
            activeforeground="#ffffff",
            padx=10,
            pady=5,
            cursor="hand2",
            command=self._dock_viewer_to_gui,
        )
        self.btn_ph_dock.pack(side="left", padx=6)

        # Bottom Status Bar
        status_bar = ttk.Frame(self, relief="sunken", padding=(8, 4))
        status_bar.pack(side="bottom", fill="x")

        self.lbl_status = ttk.Label(status_bar, text="Ready. Select parameters and click '▶ Launch Maze Simulation'.")
        self.lbl_status.pack(side="left")

        lbl_ws = ttk.Label(status_bar, text=f"Workspace: {PROJECT_ROOT.name}", foreground="#64748b")
        lbl_ws.pack(side="right")

    def _on_parameters_changed(self) -> None:
        model = self.controls_view.var_model.get()
        seed = self.controls_view.var_seed.get()
        info = self.data_manager.get_checkpoint_info(model, seed)
        self.metrics_view.update_model_info(info)

    def _on_viewer_mode_changed(self, on_gui: bool, diff_mode: str) -> None:
        """Handles switching simulation viewer between on-GUI and separate/external view."""
        if on_gui:
            if self._popout_window is not None:
                try:
                    self._popout_window.destroy()
                except Exception:
                    pass
                self._popout_window = None
                self._popout_viewer = None

            if self._placeholder_frame is not None:
                self._placeholder_frame.grid_remove()

            self.embedded_viewer.grid(row=1, column=0, sticky="nsew")
            self.active_viewer = self.embedded_viewer
            self.lbl_status.config(text="Sim Viewer display: Embedded on GUI main window.")
        else:
            self.embedded_viewer.grid_remove()
            if diff_mode == "popout":
                self.lbl_ph_title.config(text="🖥️ Sim Viewer: Pop-out Window (Separate View)")
                self.lbl_ph_desc.config(
                    text="The 3D Simulation Viewer is active in a separate window.\nYou can move it to another monitor or resize it to full screen."
                )
                self.btn_ph_front.pack(side="left", padx=6)
                self._placeholder_frame.grid(row=1, column=0, sticky="nsew")
                self._ensure_popout_window()
                self.active_viewer = self._popout_viewer
                self.lbl_status.config(text="Sim Viewer display: Separate Pop-out Window.")
            else:
                if self._popout_window is not None:
                    try:
                        self._popout_window.destroy()
                    except Exception:
                        pass
                    self._popout_window = None
                    self._popout_viewer = None

                self.lbl_ph_title.config(text="🖥️ Sim Viewer: External PyBullet 3D Window")
                self.lbl_ph_desc.config(
                    text="When simulation is launched, PyBullet's native OpenGL 3D window will open.\n\n• Left-click + drag: Rotate camera in 3D\n• Right-click + drag: Zoom in / out\n• Middle-click + drag: Pan camera"
                )
                self.btn_ph_front.pack_forget()
                self._placeholder_frame.grid(row=1, column=0, sticky="nsew")
                self.active_viewer = self.embedded_viewer
                self.lbl_status.config(text="Sim Viewer display: Native PyBullet 3D Window (External).")

    def _dock_viewer_to_gui(self) -> None:
        self.controls_view.var_viewer_on_gui.set(True)
        self.controls_view._on_viewer_checkbox_changed()

    def _undock_viewer_to_popout(self) -> None:
        self.controls_view.var_viewer_on_gui.set(False)
        self.controls_view.var_diff_view_mode.set("popout")
        self.controls_view.cb_diff_type.set("Separate Pop-out Window")
        self.controls_view._on_viewer_checkbox_changed()

    def _ensure_popout_window(self) -> None:
        if self._popout_window is None or not self._popout_window.winfo_exists():
            self._popout_window = tk.Toplevel(self)
            self._popout_window.title("Fruit Fly 3D Simulation Viewer (Separate View)")
            self._popout_window.geometry("960x650")
            self._popout_window.minsize(640, 480)
            self._popout_window.protocol("WM_DELETE_WINDOW", self._dock_viewer_to_gui)

            self._popout_viewer = ViewerView(
                self._popout_window,
                on_camera_change=self.runner.set_camera_mode,
                on_resolution_change=self.runner.set_render_resolution,
                on_dock_toggle=self._dock_viewer_to_gui,
                is_popout=True,
            )
            self._popout_viewer.pack(fill="both", expand=True, padx=6, pady=6)
            self._popout_viewer.var_camera.set(self.embedded_viewer.var_camera.get())
            self._popout_viewer.var_resolution.set(self.embedded_viewer.var_resolution.get())
            self._popout_viewer.var_shadows.set(self.embedded_viewer.var_shadows.get())
        else:
            self._focus_popout_window()

    def _focus_popout_window(self) -> None:
        if self._popout_window is not None and self._popout_window.winfo_exists():
            self._popout_window.deiconify()
            self._popout_window.lift()
            self._popout_window.focus_force()

    def _on_close_app(self) -> None:
        try:
            self.runner.stop()
        except Exception:
            pass
        if self._popout_window is not None:
            try:
                self._popout_window.destroy()
            except Exception:
                pass
        self.destroy()

    def _launch_simulation(
        self,
        run_name: str,
        size: int,
        episodes: int,
        speed: float,
        on_gui: bool = True,
        diff_mode: str = "native",
        embedded: Optional[bool] = None,
    ) -> None:
        if embedded is not None:
            on_gui = embedded

        self.metrics_view.reset_live_summary()
        self.controls_view.set_running_state(True)
        self.lbl_head_status.config(text="● Simulation Running", fg="#38bdf8")

        # Automatically show the Completion Times tab so user can watch times accumulate
        self.left_notebook.select(0)

        try:
            if on_gui:
                use_embedded = True
                mode_desc = "Embedded Viewport (On GUI)"
            elif diff_mode == "popout":
                use_embedded = True
                self._ensure_popout_window()
                mode_desc = "Pop-out Viewport (Separate Window)"
            else:
                use_embedded = False
                mode_desc = "External PyBullet Window"

            cmd = self.runner.start(
                run_name=run_name,
                size=size,
                episodes=episodes,
                speed=speed,
                embedded=use_embedded,
            )
            self.logs_view.append_line(f"=== [EXEC ({mode_desc})] {' '.join(cmd)} ===")
        except Exception as e:
            messagebox.showerror("Launch Error", f"Could not start simulation:\n{e}")
            self.controls_view.set_running_state(False)
            self.lbl_head_status.config(text="● Error", fg="#ef4444")

    def _stop_simulation(self) -> None:
        self.runner.stop()
        self.lbl_head_status.config(text="● Stopping...", fg="#f59e0b")

    # --- Thread-Safe Queue Handlers ---
    def _queue_frame(self, pil_img: Image.Image, telemetry: Dict[str, Any]) -> None:
        # Keep only the newest simulation frame. Old frames are discarded so a slow GUI
        # never builds up a backlog and starts showing delayed or jumpy video.
        with self._frame_lock:
            self._latest_frame = (pil_img, telemetry)

    def _queue_line(self, line: str) -> None:
        self.msg_queue.put(("line", line))

    def _queue_episode(self, res: EpisodeResult) -> None:
        self.msg_queue.put(("episode", res))

    def _queue_summary(self, summary: RunSummary) -> None:
        self.msg_queue.put(("summary", summary))

    def _queue_status(self, msg: str) -> None:
        self.msg_queue.put(("status", msg))

    def _queue_finished(self, exit_code: int) -> None:
        self.msg_queue.put(("finished", exit_code))

    def _poll_queue(self) -> None:
        """Periodically processes updates from background simulation worker."""
        try:
            with self._frame_lock:
                latest_frame = self._latest_frame
                self._latest_frame = None
            if latest_frame is not None:
                img, tel = latest_frame
                if self.active_viewer is not None:
                    self.active_viewer.update_frame(img, tel)

            while True:
                item = self.msg_queue.get_nowait()
                kind, payload = item
                if kind == "line":
                    self.logs_view.append_line(payload)
                elif kind == "episode":
                    self.episodes_view.add_episode(payload)
                    if self.active_viewer is not None:
                        self.active_viewer.show_completed(payload.outcome, payload.time_s)
                elif kind == "summary":
                    self.metrics_view.update_live_summary(payload)
                elif kind == "status":
                    self.lbl_status.config(text=payload)
                elif kind == "finished":
                    self.controls_view.set_running_state(False)
                    self.lbl_head_status.config(
                        text="● Ready" if payload == 0 else f"● Stopped ({payload})",
                        fg="#10b981" if payload == 0 else "#f59e0b",
                    )
        except queue.Empty:
            pass
        finally:
            self.after(16, self._poll_queue)


def main() -> None:
    app = DroneMazeApp()
    app.mainloop()


if __name__ == "__main__":
    main()
