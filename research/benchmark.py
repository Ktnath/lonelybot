"""Reproducible research benchmarks for Lonelybot.

Phase 0.2 keeps the exact Oracle and deterministic-deal infrastructure from
Phase 0.1, but focuses the comparison on difficult decisions rather than the
mostly-trivial first move. It provides:
  * exact Oracle solving over deterministic seeds,
  * complete Oracle move-path extraction,
  * masked partial-information snapshots along the Oracle path,
  * critical-decision discovery where the heuristic disagrees with Oracle,
  * guided-MCTS vs heuristic comparisons on those same snapshots,
  * CPU vs CUDA policy/value throughput utilities.

The current MCTS remains a determinization baseline. It is deliberately kept
separate from the future information-set / particle-belief agent.
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


def _oracle_moves(output: str, solved_match: re.Match[str] | None) -> List[str]:
    if solved_match is None:
        return []
    tail = output[solved_match.end():]
    for line in tail.splitlines():
        line = line.strip()
        if not line:
            continue
        return [move.strip() for move in line.split(",") if move.strip()]
    return []


def _first_oracle_move(output: str, solved_match: re.Match[str] | None) -> str | None:
    moves = _oracle_moves(output, solved_match)
    return moves[0] if moves else None


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

            moves = _oracle_moves(output, solved_match)
            rows.append(
                {
                    "seed": int(seed),
                    "draw_step": int(draw_step),
                    "status": status,
                    "solution_moves": int(solved_match.group(1)) if solved_match else None,
                    "oracle_first_move": moves[0] if moves else None,
                    "oracle_moves_json": json.dumps(moves, ensure_ascii=False) if moves else None,
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
                    "oracle_moves_json": None,
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


def _move_type(move: str | None) -> str | None:
    if not move:
        return None
    return move.split(None, 1)[0]


def trace_oracle_solution(
    trace_path: str | Path,
    seed: int,
    draw_step: int,
    moves: Sequence[str],
) -> List[dict]:
    proc = subprocess.run(
        [str(trace_path), str(int(seed)), str(int(draw_step)), json.dumps(list(moves), ensure_ascii=False)],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def collect_critical_decisions(
    trace_path: str | Path,
    oracle_df: pd.DataFrame,
    draw_step: int = 3,
    max_per_seed: int = 2,
    max_points: int = 120,
    min_step: int = 1,
) -> pd.DataFrame:
    """Collect difficult states along Oracle solutions.

    A point is considered critical when the masked-state heuristic disagrees
    with the next Oracle action and the true state has at least two legal moves.
    We also record whether the Oracle action is representable among the current
    agent's legal moves; this exposes action-space mismatches explicitly instead
    of counting them as ordinary decision errors.
    """
    from lonelybot_py import GameState, best_move_py, legal_actions_py

    rows: List[Dict[str, object]] = []
    solved = oracle_df[oracle_df["status"] == "solved"]

    for _, oracle_row in solved.iterrows():
        if len(rows) >= max_points:
            break
        seed = int(oracle_row["seed"])
        moves_json = oracle_row.get("oracle_moves_json")
        if not isinstance(moves_json, str) or not moves_json:
            continue
        moves = json.loads(moves_json)
        snapshots = trace_oracle_solution(trace_path, seed, draw_step, moves)
        kept_for_seed = 0

        for snap in snapshots:
            step = int(snap["step"])
            if step < min_step:
                continue
            oracle_move = str(snap["oracle_move"])
            true_legal = list(snap.get("true_legal_moves", []))
            if len(true_legal) < 2:
                continue

            partial = snap["partial"]
            state = GameState.from_json(json.dumps(partial, ensure_ascii=False))
            heuristic = best_move_py(state, "neutral", None)
            heuristic_move = _move_string(heuristic)
            if heuristic_move == oracle_move:
                continue

            agent_legal = list(legal_actions_py(state))
            unknown_cards = sum(len(col["hidden"]) for col in partial["columns"]) + len(partial["deck"])
            rows.append(
                {
                    "seed": seed,
                    "step": step,
                    "oracle_move": oracle_move,
                    "oracle_move_type": _move_type(oracle_move),
                    "heuristic_move": heuristic_move,
                    "heuristic_move_type": _move_type(heuristic_move),
                    "oracle_representable": oracle_move in agent_legal,
                    "agent_legal_count": len(agent_legal),
                    "true_legal_count": len(true_legal),
                    "unknown_cards": unknown_cards,
                    "foundation_cards": int(snap.get("foundation_cards", 0)),
                    "hidden_down_cards": int(snap.get("hidden_down_cards", 0)),
                    "partial_json": json.dumps(partial, ensure_ascii=False),
                    "agent_legal_json": json.dumps(agent_legal, ensure_ascii=False),
                    "true_legal_json": json.dumps(true_legal, ensure_ascii=False),
                }
            )
            kept_for_seed += 1
            if kept_for_seed >= max_per_seed or len(rows) >= max_points:
                break

    return pd.DataFrame(rows)


def benchmark_agents_on_critical(
    critical_df: pd.DataFrame,
    mcts_grid: Sequence[Tuple[int, int]] = ((64, 96), (256, 96), (1024, 96)),
) -> pd.DataFrame:
    from lonelybot_py import GameState, best_move_mcts_py, best_move_py

    rows: List[Dict[str, object]] = []
    for _, point in critical_df.iterrows():
        state = GameState.from_json(str(point["partial_json"]))
        oracle_move = str(point["oracle_move"])

        start = time.perf_counter()
        heuristic = best_move_py(state, "neutral", None)
        elapsed = time.perf_counter() - start
        heuristic_move = _move_string(heuristic)
        heuristic_correct = heuristic_move == oracle_move
        rows.append(
            {
                "seed": int(point["seed"]),
                "step": int(point["step"]),
                "agent": "heuristic",
                "n_playouts": 0,
                "max_depth": 0,
                "elapsed_s": elapsed,
                "move": heuristic_move,
                "simulation_score": None,
                "win_rate": None,
                "oracle_move": oracle_move,
                "oracle_move_type": point["oracle_move_type"],
                "oracle_representable": bool(point["oracle_representable"]),
                "oracle_agreement": heuristic_correct,
                "changed_from_heuristic": False,
                "improved_vs_heuristic": False,
                "degraded_vs_heuristic": False,
            }
        )

        for n_playouts, max_depth in mcts_grid:
            start = time.perf_counter()
            result = best_move_mcts_py(state, "neutral", int(n_playouts), int(max_depth), None)
            elapsed = time.perf_counter() - start
            move = _move_string(result["move"]) if result is not None else None
            correct = move == oracle_move
            rows.append(
                {
                    "seed": int(point["seed"]),
                    "step": int(point["step"]),
                    "agent": "mcts",
                    "n_playouts": int(n_playouts),
                    "max_depth": int(max_depth),
                    "elapsed_s": elapsed,
                    "move": move,
                    "simulation_score": int(result["simulation_score"]) if result is not None else None,
                    "win_rate": float(result["win_rate"]) if result is not None else None,
                    "oracle_move": oracle_move,
                    "oracle_move_type": point["oracle_move_type"],
                    "oracle_representable": bool(point["oracle_representable"]),
                    "oracle_agreement": correct,
                    "changed_from_heuristic": move != heuristic_move,
                    "improved_vs_heuristic": bool(correct and not heuristic_correct),
                    "degraded_vs_heuristic": bool(heuristic_correct and not correct),
                }
            )
    return pd.DataFrame(rows)


def critical_agent_summary(agent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["agent", "n_playouts", "max_depth"]
    for key, frame in agent_df.groupby(group_cols, dropna=False):
        representable = frame[frame["oracle_representable"] == True]
        rows.append(
            {
                "agent": key[0],
                "n_playouts": int(key[1]),
                "max_depth": int(key[2]),
                "n": int(len(frame)),
                "n_oracle_representable": int(len(representable)),
                "oracle_representable_rate": float(frame["oracle_representable"].mean()),
                "oracle_agreement_all": float(frame["oracle_agreement"].mean()),
                "oracle_agreement_representable": float(representable["oracle_agreement"].mean()) if len(representable) else None,
                "changed_from_heuristic_rate": float(frame["changed_from_heuristic"].mean()),
                "improved_vs_heuristic_count": int(frame["improved_vs_heuristic"].sum()),
                "degraded_vs_heuristic_count": int(frame["degraded_vs_heuristic"].sum()),
                "decision_ms_mean": float(frame["elapsed_s"].mean() * 1000.0),
                "decision_ms_p95": float(frame["elapsed_s"].quantile(0.95) * 1000.0),
                "mean_simulation_score": float(frame["simulation_score"].dropna().mean()) if frame["simulation_score"].notna().any() else None,
                "mean_rollout_win_rate": float(frame["win_rate"].dropna().mean()) if frame["win_rate"].notna().any() else None,
            }
        )
    return pd.DataFrame(rows)


def benchmark_agents_on_seeds(
    lonecli_path: str | Path,
    oracle_df: pd.DataFrame,
    draw_step: int = 3,
    mcts_grid: Sequence[Tuple[int, int]] = ((64, 64), (256, 96), (1024, 128)),
) -> pd.DataFrame:
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
        rows.append({
            "seed": seed, "agent": "heuristic", "n_playouts": 0, "max_depth": 0,
            "elapsed_s": elapsed, "move": heuristic_move, "simulation_score": None,
            "win_rate": None, "unknown_cards": unknown_cards, "oracle_first_move": oracle_move,
            "oracle_agreement": bool(oracle_move and heuristic_move == oracle_move),
        })

        for n_playouts, max_depth in mcts_grid:
            start = time.perf_counter()
            result = best_move_mcts_py(state, "neutral", int(n_playouts), int(max_depth), None)
            elapsed = time.perf_counter() - start
            move = _move_string(result["move"]) if result is not None else None
            rows.append({
                "seed": seed, "agent": "mcts", "n_playouts": int(n_playouts),
                "max_depth": int(max_depth), "elapsed_s": elapsed, "move": move,
                "simulation_score": int(result["simulation_score"]) if result is not None else None,
                "win_rate": float(result["win_rate"]) if result is not None else None,
                "unknown_cards": unknown_cards, "oracle_first_move": oracle_move,
                "oracle_agreement": bool(oracle_move and move == oracle_move),
            })
    return pd.DataFrame(rows)


def agent_summary(agent_df: pd.DataFrame) -> pd.DataFrame:
    if agent_df.empty:
        return pd.DataFrame()
    rows = []
    group_cols = ["agent", "n_playouts", "max_depth"]
    for key, frame in agent_df.groupby(group_cols, dropna=False):
        oracle_known = frame[frame["oracle_first_move"].notna()]
        rows.append({
            "agent": key[0], "n_playouts": int(key[1]), "max_depth": int(key[2]),
            "n": int(len(frame)),
            "decision_ms_mean": float(frame["elapsed_s"].mean() * 1000.0),
            "decision_ms_p95": float(frame["elapsed_s"].quantile(0.95) * 1000.0),
            "oracle_agreement_rate": float(oracle_known["oracle_agreement"].mean()) if len(oracle_known) else None,
            "mean_simulation_score": float(frame["simulation_score"].dropna().mean()) if frame["simulation_score"].notna().any() else None,
            "nonzero_simulation_rate": float((frame["simulation_score"].fillna(0) != 0).mean()),
            "mean_rollout_win_rate": float(frame["win_rate"].dropna().mean()) if frame["win_rate"].notna().any() else None,
        })
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
    parser.add_argument("--trace", default=str(ROOT / "target" / "release" / "research_trace"))
    parser.add_argument("--oracle-seeds", type=int, default=100)
    parser.add_argument("--draw-step", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=15.0)
    parser.add_argument("--critical-points", type=int, default=120)
    parser.add_argument("--output-dir", default=str(ROOT / "research" / "outputs"))
    args = parser.parse_args()

    info = system_info()
    print(json.dumps(info, indent=2))
    oracle_df = benchmark_oracle_cli(args.lonecli, range(args.oracle_seeds), args.draw_step, args.timeout)
    critical_df = collect_critical_decisions(
        args.trace, oracle_df, draw_step=args.draw_step, max_points=args.critical_points
    )
    critical_agents = benchmark_agents_on_critical(critical_df)
    critical_summary = critical_agent_summary(critical_agents)

    tables = {
        "oracle": oracle_df,
        "critical_decisions": critical_df,
        "critical_agents": critical_agents,
        "critical_summary": critical_summary,
    }
    save_outputs(Path(args.output_dir), tables, info)

    print("\nOracle summary")
    print(json.dumps(oracle_summary(oracle_df), indent=2))
    print("\nCritical decision summary")
    print(critical_summary.to_string(index=False))


if __name__ == "__main__":
    main()
