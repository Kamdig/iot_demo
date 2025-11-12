from __future__ import annotations

import sys
from pathlib import Path


if __package__ in (None, ""):
    package_root = Path(__file__).resolve().parent
    sys.path.insert(0, str(package_root.parent))
    from thumbs_pi.thumbs import main  # type: ignore WPS433 - runtime import for script usage
else:  # pragma: no cover - executed when package is imported as module
    from .thumbs import main


if __name__ == "__main__":
    main()
