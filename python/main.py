"""Lonelybot interactive CLI.

Commands include:
    best        - heuristic best move
    mcts <n> <d> - MCTS best move using ``n`` playouts and depth ``d``
    prob        - show column probabilities
    custom      - load a custom state
    weights     - load heuristic weights
    set         - set heuristic field
    style       - choose style
    quit        - exit
"""
import json
from lonelybot_py import (
    GameState,
    HeuristicConfigPy,
    ranked_moves_py,
    best_move_mcts_py,
    column_probabilities_py,
)
from utils import parse_hidden


def main():
    game = GameState()
    style = "neutral"
    cfg = HeuristicConfigPy(
        None, None, None, None, None,
        None, None,  # long_column_bonus, chain_bonus
        None, None, None  # aggressive_coef, conservative_coef, neutral_coef
    )

    while True:
        cmd = input("lonelybot> ").strip()
        if cmd == "quit":
            break

        elif cmd == "best":
            moves = ranked_moves_py(game, style, cfg)
            if moves:
                print(moves[0])
            else:
                print("No moves available.")
            continue

        elif cmd.startswith("mcts"):
            parts = cmd.split()
            if len(parts) == 1:
                mv = best_move_mcts_py(game, style, cfg)
            elif len(parts) == 3:
                try:
                    _, n_playouts, depth = parts
                    n_playouts = int(n_playouts)
                    depth = int(depth)
                    mv = best_move_mcts_py(game, style, cfg, n_playouts, depth)
                except ValueError:
                    print("Usage: mcts <playouts:int> <depth:int>")
                    continue
            else:
                print("Usage: mcts [<playouts> <depth>]")
                continue

            if mv:
                print(mv)
            else:
                print("No move found.")
            continue

        elif cmd == "prob":
            cols = column_probabilities_py(game)
            for i, col in enumerate(cols, 1):
                print(f"Column {i}:")
                for card, prob in col:
                    print(f"  {card}: {prob:.2%}")
            continue

        elif cmd.startswith("custom"):
            try:
                _, path = cmd.split(maxsplit=1)
            except ValueError:
                print("Usage: custom <file>")
                continue

            with open(path) as f:
                data = json.load(f)
            if "columns" in data:
                for col in data["columns"]:
                    col["hidden"] = parse_hidden(col.get("hidden", []))
            if "deck" in data:
                data["deck"] = parse_hidden(data["deck"])
            game = GameState.from_json(json.dumps(data))
            print("loaded", path)
            continue

        elif cmd.startswith("weights"):
            try:
                _, path = cmd.split(maxsplit=1)
            except ValueError:
                print("Usage: weights <file>")
                continue

            with open(path) as f:
                weights = json.load(f)

            cfg = HeuristicConfigPy(
                weights.get("reveal_bonus"),
                weights.get("empty_column_bonus"),
                weights.get("early_foundation_penalty"),
                weights.get("keep_king_bonus"),
                weights.get("deadlock_penalty"),
                weights.get("long_column_bonus"),
                weights.get("chain_bonus"),
                weights.get("aggressive_coef"),
                weights.get("conservative_coef"),
                weights.get("neutral_coef"),
            )
            print("heuristics loaded", path)
            continue

        elif cmd.startswith("set"):
            try:
                _, name, value = cmd.split(maxsplit=2)
            except ValueError:
                print("Usage: set <field> <value>")
                continue

            if not hasattr(cfg, name):
                print("Unknown field:", name)
                continue
            try:
                setattr(cfg, name, int(value))
            except Exception as e:
                print(f"Error setting field: {e}")
                continue
            print(f"{name} set to {value}")
            continue

        elif cmd.startswith("style"):
            try:
                _, st = cmd.split(maxsplit=1)
            except ValueError:
                print("Usage: style <aggressive|conservative|neutral>")
                continue
            if st not in ["aggressive", "conservative", "neutral"]:
                print("Unknown style. Choose aggressive, conservative or neutral")
                continue
            style = st
            print("style set to", style)
            continue

        elif cmd == "help":
            print(
                "commands: best, mcts <playouts> <depth>, prob, custom <file>, weights <file>, set <field> <value>, style <type>, quit"
            )
            continue

        else:
            print("Unknown command. Type 'help' for list.")
            continue


if __name__ == "__main__":
    main()
