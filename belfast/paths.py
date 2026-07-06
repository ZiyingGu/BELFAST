# belfast/paths.py

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def find_repo_root(start=None):
    """
    Find the git/project root from the installed package location.

    Works well for editable installs:
        pip install -e .

    For normal installs, this may fall back to PACKAGE_ROOT.
    """
    if start is None:
        start = PACKAGE_ROOT

    start = Path(start).resolve()

    for parent in [start, *start.parents]:
        if (parent / ".git").exists():
            return parent

        if (parent / "pyproject.toml").exists():
            return parent

    return PACKAGE_ROOT


REPO_ROOT = find_repo_root()

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
NN_DIR = REPO_ROOT / "nn"
SCRIPTS_DIR = REPO_ROOT / "scripts"