"""Regenerate every notebook in notebooks/ from the cell definitions in scripts/nb/.

    python scripts/build_notebooks.py

Notebooks are generated (never hand-edited JSON) and committed with outputs cleared.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.nb import nb00, nb01, nb02, nb03, nb04, nb05  # noqa: E402
from scripts.nb.tools import write_notebook  # noqa: E402


def main():
    out_dir = REPO / "notebooks"
    for mod in [nb00, nb01, nb02, nb03, nb04, nb05]:
        path = write_notebook(mod.cells(), out_dir / f"{mod.TITLE}.ipynb", mod.TITLE)
        print("wrote", path.relative_to(REPO), f"({len(mod.cells())} cells)")


if __name__ == "__main__":
    main()
