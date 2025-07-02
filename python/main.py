"""Lonelybot interactive CLI"""
import json
from lonelybot_py import GameState, ranked_moves_py, column_probabilities_py
from utils import parse_hidden


def main():
    game = GameState()
    while True:
        cmd = input("lonelybot> ").strip()
        if cmd == "quit":
            break
        if cmd == "best":
            moves = ranked_moves_py(game, "neutral")
            if moves:
                print(moves[0])
            continue
        if cmd == "prob":
            cols = column_probabilities_py(game)
            for i, col in enumerate(cols, 1):
                print(f"Column {i}:")
                for card, prob in col:
                    print(f"  {card}: {prob:.2%}")
            continue
        if cmd.startswith("custom"):
            _, path = cmd.split(maxsplit=1)
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
        if cmd == "help":
            print("commands: best, prob, custom <file>, quit")
            continue


if __name__ == "__main__":
    main()
