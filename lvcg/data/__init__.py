"""LVCG data processing modules.

This directory was missing from the public LVCG release: the author's ``.gitignore``
lists ``data/``, which also matches this source package, so ``lvcg/models`` could not be
imported. ``angle.py`` and ``beat_segmentation.py`` are reconstructed from the paper
(Appendix A.3 Table 7 and A.4) and from the call sites in ``lvcg/models``; the function
names follow the ones the release's own code graph records. The MIMIC-IV pretraining
loaders (``pipeline.py``, ``mimic.py``) are not reconstructed, because the supervised
PTB-XL protocol used here does not pretrain on MIMIC.
"""

from .angle import (
    LEAD_DIRECTIONS_MIMIC,
    LEAD_DIRECTIONS_PTBXL,
    LEAD_NAMES,
    LEAD_ORDERS,
    compute_lead_directions,
    get_lead_directions,
    reorder_leads,
)
from .beat_segmentation import BeatSegmenter

__all__ = [
    "BeatSegmenter",
    "LEAD_DIRECTIONS_MIMIC",
    "LEAD_DIRECTIONS_PTBXL",
    "LEAD_NAMES",
    "LEAD_ORDERS",
    "compute_lead_directions",
    "get_lead_directions",
    "reorder_leads",
]
