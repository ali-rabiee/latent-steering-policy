"""Convert kinova-isaac data_collection sessions (JSONL+PNG) to a zarr dataset.

Usage:
    python scripts/convert_to_zarr.py \
        --logs-root ../kinova-isaac/logs/boxes_v0 \
        --out data/boxes_v0.zarr [--include-failures] [--cameras front,wrist]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsteer.data import schema
from lsteer.data.convert import convert


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--include-failures", action="store_true")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument(
        "--cameras",
        default=",".join(schema.CAMERA_NAMES),
        help="comma-separated camera names; each needs an images/<cam>/ dir per episode",
    )
    args = parser.parse_args()

    convert(
        args.logs_root,
        args.out,
        require_success=not args.include_failures,
        img_size=args.img_size,
        camera_names=tuple(c for c in args.cameras.split(",") if c),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
