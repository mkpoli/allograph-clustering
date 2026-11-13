"""Allograph induction for one character: its crops clustered by stroke shape, so its allographs can be named.

A character has allographs, recurring ways of writing it, and some have no code point of their own, such as
the ん-like and zigzag abbreviations of 候 in Japanese manuscripts. A character classifier cannot separate
them: it is trained to give every way of writing a character the same answer, so what is left between
clusters of its embedding is partly shape and partly paper, crop box and ink weight. This package works on
one character at a time, from a pack of its crops:

1. `pack`: the character's crops, at most 160 px on the long side, each with its classifier embedding and
   the classifier cluster it came from (`packs/<code>.npz`).
2. `ink`: each crop framed at 96 px with its ink mask and a quality score (`framed.npz`).
3. `restore`: a U-Net trained on the crisp crops, made worse on purpose, returns every crop as clean ink
   (`restored.npy`), which is trimmed and refitted (`ink.npy`).
4. `encoder`: two contrastive encoders trained on the restored ink alone, so that position, size, slant,
   ink weight and texture stop counting (`features.npy`).
5. `cluster`: UMAP to ten dimensions and a Ward tree over every crop, cut into shape clusters (`tree.npz`).
6. `classify`: the categories a person named in `categories/<code>.yaml`, each made of shape clusters. The
   classifier embedding, which reads the character well, decides written character or abbreviation; the
   shape clusters sort the abbreviations. Every crop gets one category's label (`labels.tsv`).
7. `report`: one HTML page with the categories, the tree, a gallery, the classifier clusters against the
   categories crop by crop, a 3-D map and the method.

Steps 2 to 5 are the clustering, and with the naming of categories they make up the induction; step 6 is
the classification. Every step writes below `runs/<code>/` and is skipped when its output is there, so a
report can be rebuilt without training again. Training is random on the GPU: shape cluster numbers hold
for one run, and the categories file names the run it was written for.
"""
from __future__ import annotations

from pathlib import Path

#: One file of crops per character.
PACKS = Path("packs")
#: Where each character's run is written.
RUNS = Path("runs")
#: The categories a person named, one file per character.
CATEGORIES = Path("categories")
#: Shape clusters the tree is cut into.
CLUSTERS = 24


def run_dir(code: str, base: Path = RUNS) -> Path:
    return base / code
