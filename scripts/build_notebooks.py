"""Regenerate every notebook in notebooks/ from the cell definitions in scripts/nb/.

    python scripts/build_notebooks.py

Notebooks are generated (never hand-edited JSON) and committed with outputs cleared.
"""

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.nb import nb00, nb01, nb02, nb03, nb04, nb05, nb_eda, nb_finetune  # noqa: E402
from scripts.nb.tools import write_notebook  # noqa: E402


def check_code_cells(cells, title: str) -> None:
    """Fail the build if any code cell has a syntax error (IPython magics are stripped first)."""
    import ast

    for i, cell in enumerate(cells):
        if cell.cell_type != "code":
            continue
        src = "\n".join(l for l in cell.source.splitlines() if not l.lstrip().startswith(("%", "!")))
        try:
            ast.parse(src)
        except SyntaxError as exc:
            raise SystemExit(f"{title}: code cell {i} does not parse: {exc}\n{cell.source}")


def main():
    # The two notebooks of the Kaggle workflow, plus the earlier Colab/Drive series kept for reference.
    targets = [(REPO / "notebooks", [nb_eda, nb_finetune]), (REPO / "notebooks" / "legacy_colab", [nb00, nb01, nb02, nb03, nb04, nb05])]
    for out_dir, mods in targets:
        for mod in mods:
            cells = mod.cells()
            check_code_cells(cells, mod.TITLE)
            path = write_notebook(cells, out_dir / f"{mod.TITLE}.ipynb", mod.TITLE)
            print("wrote", path.relative_to(REPO), f"({len(cells)} cells)")


if __name__ == "__main__":
    main()
