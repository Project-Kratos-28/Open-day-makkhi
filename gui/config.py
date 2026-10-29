"""
Configuration and constants for the Open Day Makkhi Drone Maze GUI.
"""
from dataclasses import dataclass
from pathlib import Path
import sys

# Directory Paths
GUI_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = GUI_DIR.parent
RUNS_DIR = PROJECT_ROOT / "runs"
SRC_DIR = PROJECT_ROOT / "src"
WATCH_SCRIPT = SRC_DIR / "watch.py"
EVAL_RESULTS_CSV = RUNS_DIR / "eval_results.csv"

# Resolve python interpreter (prioritize venv if present)
VENV_PYTHON = PROJECT_ROOT / "venv" / "bin" / "python"
PYTHON_EXE = str(VENV_PYTHON) if VENV_PYTHON.exists() else sys.executable


@dataclass(frozen=True)
class ModelMetadata:
    key: str
    display_name: str
    short_name: str
    prefix: str
    neurons: int
    params: int
    badge_color: str
    description: str


# Model definitions
MODELS = {
    "conn": ModelMetadata(
        key="conn",
        display_name="Real Fly Connectome",
        short_name="Connectome",
        prefix="maze_conn",
        neurons=922,
        params=75083,
        badge_color="#10b981",  # Emerald
        description="Fruit fly central-complex biological wiring with fixed Dale's law signs.",
    ),
    "shuf": ModelMetadata(
        key="shuf",
        display_name="Shuffled Fly Control",
        short_name="Shuffled Fly",
        prefix="maze_shuf",
        neurons=922,
        params=75083,
        badge_color="#f59e0b",  # Amber
        description="Topological control: identical neurons and sparsity, with randomly scrambled wiring.",
    ),
    "gru": ModelMetadata(
        key="gru",
        display_name="Standard GRU",
        short_name="GRU",
        prefix="maze_gru",
        neurons=139,
        params=75757,
        badge_color="#6366f1",  # Indigo
        description="Industry standard gated recurrent unit sized to the exact parameter budget.",
    ),
    "base": ModelMetadata(
        key="base",
        display_name="Dense Baseline RNN",
        short_name="Dense RNN",
        prefix="maze_base",
        neurons=254,
        params=75440,
        badge_color="#ec4899",  # Pink
        description="Unconstrained fully connected RNN with matching leaky-rate neuron dynamics.",
    ),
}

DEFAULT_MODEL_KEY = "conn"
DEFAULT_SEED = 0
DEFAULT_MAZE_SIZE = 4
DEFAULT_EPISODES = 5
DEFAULT_SPEED = 1.0
DEFAULT_LAYOUT_SEED = 1000

AVAILABLE_SEEDS = [0, 1, 2, 3, 4]
AVAILABLE_SIZES = [4, 5, 6, 7, 8]
SPEED_OPTIONS = [0.5, 1.0, 1.5, 2.0, 3.0]

# UI Palette - Modern Slate Theme
THEME = {
    "bg_dark": "#0f172a",       # Slate 900
    "bg_card": "#1e293b",       # Slate 800
    "bg_subtle": "#334155",     # Slate 700
    "border": "#475569",        # Slate 600
    "text_primary": "#f8fafc",  # Slate 50
    "text_secondary": "#94a3b8",# Slate 400
    "text_muted": "#64748b",    # Slate 500
    "accent": "#38bdf8",        # Sky 400
    "accent_hover": "#0284c7",  # Sky 600
    "success": "#10b981",       # Emerald 500
    "warning": "#f59e0b",       # Amber 500
    "danger": "#ef4444",        # Red 500
    "info": "#6366f1",          # Indigo 500
    "font_family": "DejaVu Sans",
    "font_mono": "DejaVu Sans Mono",
}
