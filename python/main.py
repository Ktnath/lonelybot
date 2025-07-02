"""Lonelybot interactive CLI"""
import json
from lonelybot_py import GameState, py_ranked_moves


def main():
    game = GameState()
    while True:
        cmd = input("lonelybot> ").strip()
        if cmd == "quit":
            break
        if cmd == "best":
            moves = py_ranked_moves(game, "neutral")
            if moves:
                print(moves[0])
            continue
        if cmd == "prob":
            print("probability feature not implemented in python stub")
            continue
        if cmd.startswith("custom"):
            _, path = cmd.split(maxsplit=1)
            with open(path) as f:
                state = json.load(f)
            print("loaded", state)
            continue
        if cmd == "help":
            print("commands: best, prob, custom <file>, quit")
            continue


if __name__ == "__main__":
    main()
