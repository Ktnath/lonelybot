"""Reproducible research benchmarks for Lonelybot.

Phase 0.1 focuses on trustworthy comparisons before the full belief-state
solver is introduced. It provides:
  * exact Oracle solving over deterministic seeds,
  * Oracle first-move extraction and solver-search statistics,
  * deterministic partial-information states derived from the same seeds,
  * Heuristic vs MCTS comparisons on identical deals,
  * CPU vs CUDA policy/value throughput,
  * Gymnasium smoke execution.

The current MCTS remains a determinization baseline. It is deliberately kept
separate from the future information-set / belief-state agent.
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
from typing import Dict, Iterable, List, Sequence, Tuple

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
TOTAL_VISIT_RE = re.compile(r"Total visit:\s*(\d+)", re.IGNORECASE)
TRANS_HIT_RE = re.compile(r"Transposition hit:\s*(\d+)", re.IGNORECASE)
MISS_STATE_RE = re.compile(r"Miss state:\s*(\d+)", re.IGNORECASE)
MAX_DEPTH_RE = re.compile(r"Max depth search:\s*(\d+)", re.IGNORECASE)


def system_info() -> Dict[str, object]:
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
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
                "batch_size": int(batch_size),
                "repeats": int(repeats),
                "elapsed_s": elapsed,
                "states_per_s": (batch_size * repeats) / elapsed,
                "batch_latency_ms": elapsed * 1000.0 / repeats,
                "policy_shape": str(tuple(policy.shape)),
                "value_shape": str(tuple(value.shape)),
            }
        )
    return pd.DataFrame(rows)


def benchmark_policy_devices(
    batch_sizes: Sequence[int] = (1, 32, 256, 2048, 8192),
    repeats: int = 20,
    warmup: int = 5,
) -> pd.DataFrame:
    frames = [benchmark_policy_batches(batch_sizes, repeats, warmup, device="cpu")]
    if torch.cuda.is_available():
        frames.append(benchmark_policy_batches(batch_sizes, repeats, warmup, device="cuda"))
    result = pd.concat(frames, ignore_index=True)

    cpu = result[result["device"] == "cpu"].set_index("batch_size")["states_per_s"]
    result["speedup_vs_cpu"] = result.apply(
        lambda row: row["states_per_s"] / cpu.get(row["batch_size"], np.nan), axis=1
    )
    return result


def _first_oracle_move(output: str, solved_match: re.Match[str] | None) -> str | None:
    if solved_match is None:
        return None
    tail = output[solved_match.end():]
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        # The first non-empty line after "Solvable in ... moves" is the
        # engine's specialized move list: "PS A♦, R 5♠, ...".
        first = line.split(",", 1)[0].strip()
        return first or None
    return None


def _match_int(pattern: re.Pattern[str], output: str) -> int | None:
    m = pattern.search(output)
    return int(m.group(1)) if m else None


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
            output = proc.stdout or ""
            solved_match = SOLVED_RE.search(output)
            run_match = RUN_MS_RE.search(output)

            if solved_match:
                status = "solved"
            elif "Impossible" in output:
                status = "impossible"
            elif "Terminated" in output:
                status = "terminated"
            elif "Crashed" in output:
                status = "crashed"
            elif proc.returncode != 0:
                status = "error"
            else:
                status = "unknown"

            rows.append(
                {
                    "seed": int(seed),
                    "draw_step": int(draw_step),
                    "status": status,
                    "solution_moves": int(solved_match.group(1)) if solved_match else None,
                    "oracle_first_move": _first_oracle_move(output, solved_match),
                    "engine_ms": float(run_match.group(1)) if run_match else None,
                    "wall_s": wall,
                    "total_visit": _match_int(TOTAL_VISIT_RE, output),
                    "transposition_hits": _match_int(TRANS_HIT_RE, output),
                    "miss_states": _match_int(MISS_STATE_RE, output),
                    "max_depth_search": _match_int(MAX_DEPTH_RE, output),
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
                    "oracle_first_move": None,
                    "engine_ms": None,
                    "wall_s": time.perf_counter() - start,
                    "total_visit": None,
                    "transposition_hits": None,
                    "miss_states": None,
                    "max_depth_search": None,
                    "returncode": None,
                }
            )
    return pd.DataFrame(rows)


def oracle_summary(oracle_df: pd.DataFrame) -> Dict[str, object]:
    counts = oracle_df["status"].value_counts(dropna=False).to_dict()
    solved = oracle_df[oracle_df["status"] == "solved"]
    engine = oracle_df["engine_ms"].dropna()

    summary: Dict[str, object] = {
        "oracle_total": int(len(oracle_df)),
        "oracle_solved": int(counts.get("solved", 0)),
        "oracle_impossible": int(counts.get("impossible", 0)),
        "oracle_timeout": int(counts.get("timeout", 0)),
        "oracle_terminated": int(counts.get("terminated", 0)),
        "oracle_crashed": int(counts.get("crashed", 0)),
        "oracle_error": int(counts.get("error", 0)),
        "oracle_unknown": int(counts.get("unknown", 0)),
        "oracle_mean_moves_solved": float(solved["solution_moves"].mean()) if len(solved) else None,
    }
    if len(engine):
        for q, name in ((0.50, "p50"), (0.90, "p90"), (0.95, "p95"), (0.99, "p99")):
            summary[f"oracle_engine_ms_{name}"] = float(engine.quantile(q))
        summary["oracle_engine_ms_max"] = float(engine.max())
    return summary


def _extract_json_object(text: str) -> dict:
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("Could not find JSON object in lonecli print output")
    return json.loads(text[start : end + 1])


def partial_state_from_seed_cli(
    lonecli_path: str | Path,
    seed: int,
    draw_step: int = 3,
):
    """Build the legitimate initial information state for a deterministic deal.

    `lonecli print` reveals the full generated deal for reproducibility. We then
    deliberately mask every face-down tableau card and every stock card before
    constructing `GameState`. Thus Heuristic/MCTS receive the same visible
    initial information for the same seed, while Oracle keeps the full deal.
    """
    from lonelybot_py import GameState

    proc = subprocess.run(
        [str(lonecli_path), "print", "default", str(int(seed))],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    raw = _extract_json_object(proc.stdout)
    tableau = raw.get("tableau piles", [])
    stock = raw.get("stock", [])

    columns = []
    for pile in tableau[:7]:
        hidden_count = sum(1 for card in pile if isinstance(card, str) and card[-1:].islower())
        visible = [card for card in pile if isinstance(card, str) and card[-1:].isupper()]
        columns.append({"hidden": ["unknown"] * hidden_count, "visible": visible})
    while len(columns) < 7:
        columns.append({"hidden": [], "visible": []})

    partial = {
        "draw_step": int(draw_step),
        "columns": columns,
        "deck": ["unknown"] * len(stock),
    }
    return GameState.from_json(json.dumps(partial)), partial


def _move_string(move_obj: object | None) -> str | None:
    if move_obj is None:
        return None
    text = repr(move_obj).strip()
    return text or None


def benchmark_agents_on_seeds(
    lonecli_path: str | Path,
    oracle_df: pd.DataFrame,
    draw_step: int = 3,
    mcts_grid: Sequence[Tuple[int, int]] = ((64, 64), (256, 96), (1024, 128)),
) -> pd.DataFrame:
    """Compare Heuristic and dense-evaluation MCTS on the exact same deals."""
    from lonelybot_py import best_move_mcts_py, best_move_py

    oracle_map = oracle_df.set_index("seed")["oracle_first_move"].to_dict()
    rows: List[Dict[str, object]] = []

    for seed in oracle_df["seed"].astype(int).tolist():
        state, partial = partial_state_from_seed_cli(lonecli_path, seed, draw_step)
        unknown_cards = sum(len(col["hidden"]) for col in partial["columns"]) + len(partial["deck"])
        oracle_move = oracle_map.get(seed)

        start = time.perf_counter()
        heuristic = best_move_py(state, "neutral", None)
        elapsed = time.perf_counter() - start
        heuristic_move = _move_string(heuristic)
        rows.append(
            {
                "seed": seed,
                "agent": "heuristic",
                "n_playouts": 0,
                "max_depth": 0,
                "elapsed_s": elapsed,
                "move": heuristic_move,
                "simulation_score": None,
                "win_rate": None,
                "unknown_cards": unknown_cards,
                "oracle_first_move": oracle_move,
                "oracle_agreement": bool(oracle_move and heuristic_move == oracle_move),
            }
        )

        for n_playouts, max_depth in mcts_grid:
            start = time.perf_counter()
            result = best_move_mcts_py(
                state,
                "neutral",
                int(n_playouts),
                int(max_depth),
                None,
            )
            elapsed = time.perf_counter() - start
            move = _move_string(result["move"]) if result is not None else None
            rows.append(
                {
                    "seed": seed,
                    "agent": "mcts",
                    "n_playouts": int(n_playouts),
                    "max_depth": int(max_depth),
                    "elapsed_s": elapsed,
                    "move": move,
                    "simulation_score": int(result["simulation_score"]) if result is not None else None,
                    "win_rate": float(result["win_rate"]) if result is not None else None,
                    "unknown_cards": unknown_cards,
                    "oracle_first_move": oracle_move,
                    "oracle_agreement": bool(oracle_move and move == oracle_move),
                }
            )

    return pd.DataFrame(rows)


def agent_summary(agent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()
    group_cols = ["agent", "n_playouts", "max_depth"]
    rows = []
    for key, frame in agent_df.groupby(group_cols, dropna=False):
        oracle_known = frame[frame["oracle_first_move"].notna()]
        rows.append(
            {
                "agent": key[0],
                "n_playouts": int(key[1]),
                "max_depth": int(key[2]),
                "n": int(len(frame)),
                "decision_ms_mean": float(frame["elapsed_s"].mean() * 1000.0),
                "decision_ms_p95": float(frame["elapsed_s"].quantile(0.95) * 1000.0),
                "oracle_agreement_rate": float(oracle_known["oracle_agreement"].mean()) if len(oracle_known) else None,
                "mean_simulation_score": float(frame["simulation_score"].dropna().mean()) if frame["simulation_score"].notna().any() else None,
                "nonzero_simulation_rate": float((frame["simulation_score"].fillna(0) != 0).mean()),
                "mean_rollout_win_rate": float(frame["win_rate"].dropna().mean()) if frame["win_rate"].notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def smoke_test_env(max_steps: int = 20) -> Dict[str, object]:
    from klondike_env import KlondikeEnv

    env = KlondikeEnv()
    obs, info = env.reset(seed=0)
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
    parser.add_argument("--oracle-seeds", type=int, default=100)
    parser.add_argument("--draw-step", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--output-dir", default=str(ROOT / "research" / "outputs"))
    args = parser.parse_args()

    info = system_info()
    print(json.dumps(info, indent=2))
    print("Environment smoke test:", smoke_test_env())

    policy_df = benchmark_policy_devices()
    oracle_df = benchmark_oracle_cli(
        args.lonecli,
        range(args.oracle_seeds),
        draw_step=args.draw_step,
        timeout_s=args.timeout,
    )
    agent_df = benchmark_agents_on_seeds(args.lonecli, oracle_df, draw_step=args.draw_step)
    agent_summary_df = agent_summary(agent_df)

    tables = {
        "policy_devices": policy_df,
        "oracle": oracle_df,
        "agents": agent_df,
        "agent_summary": agent_summary_df,
    }
    save_outputs(Path(args.output_dir), tables, info)

    print("\nPolicy CPU/GPU benchmark")
    print(policy_df.to_string(index=False))
    print("\nOracle summary")
    print(json.dumps(oracle_summary(oracle_df), indent=2))
    print("\nAgent summary")
    print(agent_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
