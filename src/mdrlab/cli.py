import argparse

from .core import load, normalize, to_csv


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("command", choices=["validate", "normalize"])
    p.add_argument("path")
    args = p.parse_args()
    rows = normalize(load(args.path))
    print("OK" if args.command == "validate" else to_csv(rows), end="\n")


if __name__ == "__main__":
    main()
