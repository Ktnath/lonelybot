"""Lonelybot interactive CLI"""
import json
from lonelybot_py import GameState, ranked_moves_py
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
            print("probability feature not implemented in python stub")
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
