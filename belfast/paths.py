# belfast/paths.py

from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parent


def find_repo_root(start=None):
    if start is None:
        start = PACKAGE_ROOT

    start = Path(start).resolve()

    for parent in [start, *start.parents]:
        if (parent / ".git").exists() or (parent / "pyproject.toml").exists():
            return parent

    return PACKAGE_ROOT


REPO_ROOT = find_repo_root()


# ============================================================
# User settings
# Change these if data/results are stored elsewhere.
# ============================================================

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = REPO_ROOT / "results"
NN_DIR = REPO_ROOT / "nn"

# Example:
# DATA_DIR = Path("/Volumes/MySSD/BELFAST_data")
# RESULTS_DIR = Path("/Volumes/MySSD/BELFAST_results")


# ============================================================
# Data directories
# ============================================================

ISOCHRONE_DIR = DATA_DIR / "isochrones"
KLD_DIR = DATA_DIR / "Kurucz_limb_darkening"
PARS_FILTER_DIR = DATA_DIR / "PARS_filters"


# ============================================================
# Project directories
# ============================================================

SCRIPTS_DIR = REPO_ROOT / "scripts"