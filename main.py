#!/usr/bin/env python3
"""Price Predictor Pro - Entry Point. Built by Patricio Tirado (Hyperium IA)."""

from src.cli import build_parser, run


def main():
    ap = build_parser()
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
