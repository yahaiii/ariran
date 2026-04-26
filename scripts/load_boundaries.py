"""
Load Nigerian administrative boundaries into public.admin_boundaries.

This script is intentionally conservative for setup stability:
- If no source file is provided, it exits successfully with guidance.
- If a source file is provided, it currently validates the path and exits with
  a clear not-implemented message to avoid silent data corruption.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load admin boundaries")
    parser.add_argument(
        "--source",
        default="",
        help="Path to boundary dataset (GeoJSON/Shapefile), optional",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()

    if not args.source:
        print("No boundary source provided. Skipping load.")
        print("Use --source /path/to/boundaries when dataset is available.")
        return 0

    source_path = Path(args.source)
    if not source_path.exists():
        print(f"Boundary source not found: {source_path}")
        return 1

    print("Boundary loader is scaffolded but not implemented yet.")
    print("Source path validated:", source_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
