"""Train the frozen diffusion policy.

Usage:
    python scripts/train.py --config configs/boxes_v0.yaml \
        [--override optim.lr=3e-5 log.logger=none ...]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from lsteer.training.config import load_config
from lsteer.training.trainer import Trainer


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--override", nargs="*", default=[], help="dotted overrides, e.g. optim.lr=3e-5")
    args = parser.parse_args()

    cfg = load_config(args.config, args.override)
    trainer = Trainer(cfg)
    ckpt = trainer.fit()
    print(f"done -> {ckpt}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
