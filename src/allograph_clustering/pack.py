"""A pack: every crop of one character, as one `.npz` file.

    code_point           "U+5019"
    revision             the classifier clustering the pack was cut from
    ids                  (n,) crop identifiers; the part before the first colon names the source dataset
    grey                 (n, 160, 160) uint8, the crop in its top-left corner on white
    size                 (n, 2) height and width of the crop inside `grey`; 0 when its pixels are missing
    embedding            (n, d) float32, the character classifier's penultimate layer
    classifier_cluster   (n,) int16, the cluster of that embedding the crop belongs to, from 1, largest first
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

FIELDS = ("code_point", "revision", "ids", "grey", "size", "embedding", "classifier_cluster")


def load(path: Path) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"{path}: no pack; download it with `gh release download packs -p '{path.name}' -D {path.parent}`")
    with np.load(path) as data:
        missing = [f for f in FIELDS if f not in data]
        if missing:
            raise ValueError(f"{path}: pack lacks {missing}")
        return {f: data[f] for f in FIELDS}
