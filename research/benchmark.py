"""Reproducible Phase-0 benchmarks for Lonelybot.

This module intentionally measures the current engine before introducing a
new belief-state agent. It benchmarks:
  * Rust/PyO3 MCTS decision latency,
  * policy/value CPU/GPU batch inference,
  * the exact CLI solver over deterministic seeds,
  * Gymnasium environment smoke execution.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON_DIR = ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from policy_value_net import BOARD_SIZE, NB_ACTIONS, PolicyValueNet


SOLVED_RE = re.compile(r"Solvable in\s+(\d+)\s+moves", re.IGNORECASE)
RUN_MS_RE = re.compile(r"Run in\s+([0-9.]+)\s+ms", re.IGNORECASE)


def system_info() -> Dict[str, object]:
    gpu_name = None
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "gpu_name": gpu_name,
        "board_size": BOARD_SIZE,
        "action_size": NB_ACTIONS,
    }


def benchmark_policy_batches(
    batch_sizes: Sequence[int] = (1, 32, 256, 2048, 8192),
    repeats: int = 30,
    warmup: int = 5,
    device: str | None = None,
) -> pd.DataFrame:
    model = PolicyValueNet(device=device)
    rng = np.random.default_rng(0)
    rows: List[Dict[str, object]] = []

    for batch_size in batch_sizes:
        x = rng.integers(0, 53, size=(batch_size, BOARD_SIZE), dtype=np.int16).astype(np.float32)
        for _ in range(warmup):
            model.predict_batch(x)
        if model.device.type == "cuda":
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(repeats):
            policy, value = model.predict_batch(x)
        if model.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - start

        rows.append(
            {
                "device": str(model.device),
                "batch_size": batch_size,
                "repeats": repeats,
                "elapsed_s": elapsed,
                "states_per_s": (batch_size * repeats) / elapsed,
                "policy_shape": str(tuple(policy.shape)),
                "value_shape": str(tuple(value.shape)),
            }
        )
    return pd.DataFrame(rows)


def benchmark_mcts_decisions(
    n_states: int = 10,
    n_playouts: int = 64,
    max_depth: int = 64,
) -> pd.DataFrame:
    from lonelybot_py import best_move_mcts_py, generate_random_state_py

    rows: List[Dict[str, object]] = []
    for i in range(n_states):
        state = generate_random_state_py()
        start = time.perf_counter()
        move = best_move_mcts_py(state, "neutral", n_playouts, max_depth, None)
        elapsed = time.perf_counter() - start
        rows.append(
            {
                "sample": i,
                "n_playouts": n_playouts,
                "max_depth": max_depth,
                "elapsed_s": elapsed,
                "move_found": move is not None,
                "move": repr(move) if move is not None else None,
            }
        )
    return pd.DataFrame(rows)


def benchmark_oracle_cli(
    lonecli_path: str | Path,
    seeds: Iterable[int],
    draw_step: int = 3,
    timeout_s: float = 15.0,
) -> pd.DataFrame:
    lonecli_path = str(lonecli_path)
    rows: List[Dict[str, object]] = []

    for seed in seeds:
        start = time.perf_counter()
        try:
            proc = subprocess.run(
                [lonecli_path, "solve", "default", str(int(seed)), str(int(draw_step))],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout_s,
                check=False,
            )
            wall = time.perf_counter() - start
            output = proc.stdout
            solved_match = SOLVED_RE.search(output)
            run_match = RUN_MS_RE.search(output)
            status = "solved" if solved_match else ("impossible" if "Impossible" in output else "unknown")
            rows.append(
                {
                    "seed": int(seed),
                    "draw_step": int(draw_step),
                    "status": status,
                    "solution_moves": int(solved_match.group(1)) if solved_match else None,
                    "engine_ms": float(run_match.group(1)) if run_match else None,
                    "wall_s": wall,
                    "returncode": proc.returncode,
                }
            )
        except subprocess.TimeoutExpired:
            rows.append(
                {
                    "seed": int(seed),
                    "draw_step": int(draw_step),
                    "status": "timeout",
                    "solution_moves": None,
                    "engine_ms": None,
                    "wall_s": time.perf_counter() - start,
                    "returncode": None,
                }
            )
    return pd.DataFrame(rows)


def smoke_test_env(max_steps: int = 20) -> Dict[str, object]:
    from klondike_env import KlondikeEnv

    env = KlondikeEnv()
    obs, info = env.reset()
    rewards: List[float] = []
    steps = 0
    terminated = False

    while steps < max_steps and not terminated:
        valid = info.get("valid_actions", [])
        if not valid:
            break
        action = int(valid[0])
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        steps += 1
        if truncated:
            break

    return {
        "observation_shape": tuple(obs.shape),
        "steps": steps,
        "terminated": terminated,
        "reward_sum": float(sum(rewards)),
        "remaining_valid_actions": len(info.get("valid_actions", [])),
    }


def save_outputs(output_dir: Path, tables: Dict[str, pd.DataFrame], metadata: Dict[str, object]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        frame.to_csv(output_dir / f"{name}.csv", index=False)
    with (output_dir / "system_info.json").open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lonecli", default=str(ROOT / "target" / "release" / "lonecli"))
    parser.add_argument("--oracle-seeds", type=int, default=20)
    parser.add_argument("--draw-step", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--mcts-states", type=int, default=10)
    parser.add_argument("--mcts-playouts", type=int, default=64)
    parser.add_argument("--mcts-depth", type=int, default=64)
    parser.add_argument("--output-dir", default=str(ROOT / "research" / "outputs"))
    args = parser.parse_args()

    info = system_info()
    print(json.dumps(info, indent=2))
    print("Environment smoke test:", smoke_test_env())

    gpu_df = benchmark_policy_batches()
    mcts_df = benchmark_mcts_decisions(args.mcts_states, args.mcts_playouts, args.mcts_depth)
    oracle_df = benchmark_oracle_cli(
        args.lonecli,
        range(args.oracle_seeds),
        draw_step=args.draw_step,
        timeout_s=args.timeout,
    )

    tables = {"gpu_policy": gpu_df, "mcts_decisions": mcts_df, "oracle": oracle_df}
    save_outputs(Path(args.output_dir), tables, info)

    print("\nGPU / policy benchmark")
    print(gpu_df.to_string(index=False))
    print("\nMCTS benchmark")
    print(mcts_df.to_string(index=False))
    print("\nOracle benchmark")
    print(oracle_df.to_string(index=False))


if __name__ == "__main__":
    main()
