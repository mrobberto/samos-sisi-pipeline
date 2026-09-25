import os
from importlib import import_module

from .pipeline_config import *

_modname = os.environ.get(
    "SISI_ACTIVE_REDUCTION",
    "config.reductions.sisi_Dolidze25_T00",
)

_mod = import_module(_modname)

for _k in dir(_mod):
    if not _k.startswith("_"):
        globals()[_k] = getattr(_mod, _k)

def build_sisi_paths():
    """
    Compatibility hook for pipeline scripts.

    The active reduction files now define paths directly, so there is
    nothing to build here.
    """
    return None

def ensure_dirs():
    """
    Create the standard output directories if the active reduction defines them.
    """
    for name in [
        "SISI_REDUCED",
        "SISI_ST01_BIAS",
        "SISI_ST02_PREPROCESS",
        "SISI_DIR_02_PREPROCESS",
        "SISI_ST03_COADD",
        "SISI_ST04_FLAT_PREPROC",
        "SISI_ST04_MASTERFLATS",
        "SISI_ST05_FLATFIELD",
        "SISI_ST06_ORIENT",
        "SISI_ST07_WCS",
    ]:
        p = globals().get(name)
        if p is not None:
            p.mkdir(parents=True, exist_ok=True)