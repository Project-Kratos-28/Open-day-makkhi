"""
Data manager for loading run metadata, checkpoint information, and benchmark evaluations.
"""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional
import csv

from gui.config import EVAL_RESULTS_CSV, MODELS, RUNS_DIR, ModelMetadata


@dataclass
class CheckpointInfo:
    run_name: str
    model_key: str
    seed: int
    exists: bool
    checkpoint_path: Optional[Path] = None
    has_best: bool = False
    training_updates: int = 0
    training_steps: int = 0
    final_train_success: float = 0.0
    final_train_crash: float = 0.0
    final_train_timeout: float = 0.0
    # Test benchmark from eval_results.csv
    benchmark_standard_success: Optional[float] = None
    benchmark_hard_success: Optional[float] = None
    benchmark_very_hard_success: Optional[float] = None
    benchmark_mean_time_s: Optional[float] = None


class DataManager:
    """Manages filesystem checks, metadata parsing, and benchmark querying."""

    def __init__(self, runs_dir: Path = RUNS_DIR, eval_csv: Path = EVAL_RESULTS_CSV):
        self.runs_dir = runs_dir
        self.eval_csv = eval_csv
        self._eval_cache: Dict[str, Dict[str, Dict[str, float]]] = {}
        self.reload_benchmarks()

    def reload_benchmarks(self) -> None:
        """Parses eval_results.csv if present into memory."""
        self._eval_cache.clear()
        if not self.eval_csv.exists():
            return

        try:
            with open(self.eval_csv, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    run_name = row.get("run", "").strip()
                    level = row.get("level", "").strip()
                    if not run_name or not level:
                        continue
                    if run_name not in self._eval_cache:
                        self._eval_cache[run_name] = {}

                    try:
                        succ = float(row.get("success_rate", 0.0))
                    except ValueError:
                        succ = 0.0

                    try:
                        mean_time = float(row.get("mean_time_to_goal_s", "nan"))
                    except ValueError:
                        mean_time = float("nan")

                    self._eval_cache[run_name][level] = {
                        "success_rate": succ,
                        "mean_time_s": mean_time,
                    }
        except Exception:
            pass

    def get_run_name(self, model_key: str, seed: int, task: str = "grid") -> str:
        """Constructs the canonical folder name, e.g. maze_conn_s2."""
        meta = MODELS.get(model_key)
        prefix = meta.prefix if meta else f"maze_{model_key}"
        if task != "grid":
            prefix = meta.key if meta else model_key
        return f"{prefix}_s{seed}"

    def get_checkpoint_info(self, model_key: str, seed: int, task: str = "grid") -> CheckpointInfo:
        """Inspects disk for a given run and returns its status and metrics."""
        run_name = self.get_run_name(model_key, seed, task)
        run_dir = self.runs_dir / run_name

        if not run_dir.exists():
            # Check fallback to non-grid if grid requested, or vice-versa
            alt_name = f"{model_key}_s{seed}" if task == "grid" else f"maze_{model_key}_s{seed}"
            if (self.runs_dir / alt_name).exists():
                run_name = alt_name
                run_dir = self.runs_dir / alt_name

        best_pt = run_dir / "model_best.pt"
        std_pt = run_dir / "model.pt"
        exists = best_pt.exists() or std_pt.exists()
        ckpt_path = best_pt if best_pt.exists() else (std_pt if std_pt.exists() else None)

        info = CheckpointInfo(
            run_name=run_name,
            model_key=model_key,
            seed=seed,
            exists=exists,
            checkpoint_path=ckpt_path,
            has_best=best_pt.exists(),
        )

        # Parse training log if present
        log_csv = run_dir / "log.csv"
        if log_csv.exists():
            self._enrich_from_log(log_csv, info)

        # Parse benchmark stats if present
        self._enrich_from_benchmark(run_name, model_key, seed, info)

        return info

    def _enrich_from_log(self, log_path: Path, info: CheckpointInfo) -> None:
        """Reads the last row of log.csv for final training stats."""
        try:
            with open(log_path, mode="r", encoding="utf-8") as f:
                reader = list(csv.DictReader(f))
                if not reader:
                    return
                last_row = reader[-1]
                info.training_updates = int(last_row.get("update", 0))
                info.training_steps = int(last_row.get("env_steps", 0))
                info.final_train_success = float(last_row.get("success_rate", 0.0))
                info.final_train_crash = float(last_row.get("crash_rate", 0.0))
                info.final_train_timeout = float(last_row.get("timeout_rate", 0.0))
        except Exception:
            pass

    def _enrich_from_benchmark(self, run_name: str, model_key: str, seed: int, info: CheckpointInfo) -> None:
        """Looks up historical benchmark scores in eval_results.csv."""
        # Check by run_name or short name (e.g. conn_s0)
        candidates = [run_name, f"{model_key}_s{seed}"]
        for cand in candidates:
            if cand in self._eval_cache:
                data = self._eval_cache[cand]
                if "standard" in data:
                    info.benchmark_standard_success = data["standard"]["success_rate"]
                    info.benchmark_mean_time_s = data["standard"]["mean_time_s"]
                if "hard" in data:
                    info.benchmark_hard_success = data["hard"]["success_rate"]
                if "very_hard" in data:
                    info.benchmark_very_hard_success = data["very_hard"]["success_rate"]
                break
