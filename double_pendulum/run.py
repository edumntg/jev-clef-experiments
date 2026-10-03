"""Double inverted pendulum on a cart. Same loop and flags as `uv run pendulum`, harder plant.

Examples:
  uv run double-pendulum --policy lqr
  uv run double-pendulum --policy jev
  uv run double-pendulum --policy clef --text-state
"""
import os

from pendulum.run import run
from . import sim


def main():
    run(sim, "double-pendulum", os.path.join(os.path.dirname(os.path.abspath(__file__)), "outputs"))


if __name__ == "__main__":
    main()
