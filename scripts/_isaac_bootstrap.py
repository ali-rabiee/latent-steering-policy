"""sys.path bootstrap for importing kinova-isaac packages under Isaac Lab's Kit.

Replicates kinova-isaac/data_collection/collect_data.py:9-27. MUST be called
AFTER AppLauncher has started Kit (Kit mutates sys.path and preloads modules
that shadow kinova-isaac's `environments` and `utils` packages) and BEFORE any
`import environments...` / `import data_collection...` / `import controllers...`.

Root resolution order: explicit argument -> $KINOVA_ISAAC_ROOT -> ../kinova-isaac
relative to this repo.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

DEFAULT_KINOVA_ROOT = Path(__file__).resolve().parents[2] / "kinova-isaac"


def bootstrap_kinova(root: str | Path | None = None) -> Path:
    root = Path(root or os.environ.get("KINOVA_ISAAC_ROOT") or DEFAULT_KINOVA_ROOT).resolve()
    if not (root / "environments").is_dir():
        raise FileNotFoundError(f"kinova-isaac root not found or invalid: {root}")
    root_str = str(root)

    # kinova-isaac root must be FIRST on sys.path
    if root_str in sys.path:
        sys.path.remove(root_str)
    sys.path.insert(0, root_str)

    # purge a shadowed non-package `environments` module (Isaac side-effect)
    env_mod = sys.modules.get("environments")
    if env_mod is not None and not hasattr(env_mod, "__path__"):
        del sys.modules["environments"]

    # Isaac's bundled cv2 may import `cv2.utils` early, shadowing kinova's `utils/`
    utils_mod = sys.modules.get("utils")
    if utils_mod is not None:
        utils_file = str(getattr(utils_mod, "__file__", "") or "")
        if utils_file and root_str not in utils_file:
            for key in list(sys.modules.keys()):
                if key == "utils" or key.startswith("utils."):
                    del sys.modules[key]

    return root


def add_lsteer_to_path() -> None:
    src = Path(__file__).resolve().parents[1] / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
