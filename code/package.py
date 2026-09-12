"""Build code.zip for submission.

Includes the runnable solution, the README and the required evaluation folder.
Excludes caches, virtualenvs, the dataset and anything that could carry a
secret.
"""

from __future__ import annotations

import os
import zipfile

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(HERE)
OUT = os.path.join(REPO_ROOT, "code.zip")

SKIP_DIRS = {"__pycache__", ".cache", ".venv", "venv", ".pytest_cache", ".git"}
SKIP_NAMES = {".env", "log.txt", "code.zip"}
SKIP_SUFFIXES = (".pyc", ".pyo", ".log")


def build(out: str = OUT) -> str:
    written = 0
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(HERE):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for name in sorted(files):
                if name in SKIP_NAMES or name.endswith(SKIP_SUFFIXES):
                    continue
                path = os.path.join(root, name)
                # code/main.py is archived as main.py so evaluation/ sits at
                # the zip root, which is where the brief expects it.
                zf.write(path, os.path.relpath(path, HERE))
                written += 1
    print(f"wrote {out} ({written} files, {os.path.getsize(out) / 1024:.0f} KB)")
    return out


if __name__ == "__main__":
    build()
