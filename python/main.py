"""Lonelybot interactive CLI"""
import json
from lonelybot_py import (
    GameState,
    HeuristicConfigPy,
    ranked_moves_py,
    column_probabilities_py,
)
from utils import parse_hidden


def main():
    game = GameState()
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
            moves = ranked_moves_py(game, "neutral", cfg)
            if moves:
                print(moves[0])
            else:
                print("No moves available.")
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

        elif cmd == "help":
            print(
                "commands: best, prob, custom <file>, weights <file>, set <field> <value>, quit"
            )
            continue

        else:
            print("Unknown command. Type 'help' for list.")
            continue


if __name__ == "__main__":
    main()
