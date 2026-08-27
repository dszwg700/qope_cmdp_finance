"""Regenerate sample-size and horizon sensitivity figures from published summaries."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.make_figures import (
    DEFAULT_RESULTS,
    _configure_style,
    horizon_ess_sensitivity,
    sample_size_sensitivity,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--figures-dir", type=Path, default=DEFAULT_RESULTS / "figures")
    parser.add_argument("--dpi", type=int, default=220)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    _configure_style()
    args.figures_dir.mkdir(parents=True, exist_ok=True)
    sample_size_sensitivity(args.results_dir, args.figures_dir, args.dpi)
    horizon_ess_sensitivity(args.results_dir, args.figures_dir, args.dpi)
    print(f"Wrote sensitivity figures to {args.figures_dir}")
