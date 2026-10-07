"""Tiny helpers to build the notebooks with nbformat (never hand-written JSON)."""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent

import nbformat as nbf

REPO_URL = "https://github.com/SattamAltwaim/rsna-pe-ct.git"


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip())


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip())


def plot_note(title: str, looking_at: str, why: str, look_for: str):
    """The mandatory three-part explanation that precedes every plot."""
    return md(f"""
    ### {title}

    **What you're looking at.** {looking_at}

    **Why it matters.** {why}

    **What to look for.** {look_for}
    """)


def setup_cell(branch: str = "main"):
    return code(f'''
    import os, sys, subprocess

    BRANCH = "{branch}"
    REPO = "{REPO_URL}"
    DEST = "/content/rsna-pe-ct"

    if os.path.exists("/content"):                       # on Colab
        if not os.path.exists(DEST):
            subprocess.run(["git", "clone", "--branch", BRANCH, REPO, DEST], check=True)
        else:
            subprocess.run(["git", "-C", DEST, "fetch", "origin", BRANCH], check=True)
            subprocess.run(["git", "-C", DEST, "reset", "--hard", f"origin/{{BRANCH}}"], check=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-e", DEST], check=True)
        if DEST not in sys.path:
            sys.path.insert(0, DEST)
        from google.colab import drive; drive.mount("/content/drive")
    else:                                                # local fallback (Mac)
        parent = os.path.abspath(os.path.join(os.getcwd(), ".."))
        if os.path.exists(os.path.join(parent, "pe_ct")) and parent not in sys.path:
            sys.path.insert(0, parent)

    %load_ext autoreload
    %autoreload 2
    import pe_ct
    print("pe_ct", pe_ct.__version__, "| commit:",
          subprocess.run(["git", "-C", DEST if os.path.exists(DEST) else "..", "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip())
    ''')


def config_cell(notebook: str, extra_lines: str = ""):
    """CONFIG cell: one Config object, optional smoke-test shrinkage, figure dir."""
    return code(f'''
    import os
    from pe_ct.config import default_config
    from pe_ct import viz

    NOTEBOOK = "{notebook}"
    cfg = default_config()                       # Drive paths on Colab, ./data locally
    SMOKE = os.environ.get("PE_CT_SMOKE") == "1" # tiny sizes for a quick end-to-end check
    if SMOKE:
        cfg.n_eda, cfg.n_dev, cfg.n_test_sample, cfg.shard_size = 4, 60, 20, 10
    {extra_lines}
    cfg.ensure_dirs()
    viz.use_notebook_style()
    FIG = cfg.figures_dir
    print(cfg)
    ''')


def learned_cell(bullets: list):
    body = "\n".join(f"- {b}" for b in bullets)
    return md(f"""
    ## What we learned

    {body}
    """)


def write_notebook(cells: list, path, title: str):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "colab": {"name": title, "provenance": []},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, str(path))
    return path
