"""The categories a person named, and the classification of every crop into one of them.

`categories/<code>.yaml` names each category and the shape clusters it is made of, for one run:

    code_point: U+5019
    run: 3f2a…            # the run its shape cluster numbers belong to (`run.json`)
    categories:
      - label: 候
        written: true     # the character written out, in any hand
        clusters: [1, 2, 3, 4, 5]
        text: …
      - label: ん / h shape
        drawings: [n-shape]   # drawings/<name>.svg beside the file
        clusters: [18, 19]
        text: …
    checks: [27, 3, 6]    # classifier clusters to highlight in the report
    notes: [ … ]          # the reading of the evidence, shown as a list
    sources: [ … ]        # anything else the categories rest on, cited in the report

The shape encoder judges stroke shape well and the character well only when the crop is clean; the
character classifier reads the character well. So when a category is `written`, whether a crop is written
out is decided from the classifier embedding: a classifier cluster counts as written when most of its
crops fall in the written shape clusters; a logistic regression on the embedding learns from the crops
where the classifier cluster and the shape cluster agree, and decides the crops where they do not. A crop
it moves out of the written category takes the majority category of its 15 nearest abbreviated crops in
the encoder's space.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

#: Least crops of each side (written, abbreviated) the regression needs among the agreed crops.
MIN_CLASS = 10
#: Abbreviated neighbours that vote on the category of a crop moved out of the written category.
NEIGHBOURS = 15


def load(path: Path, clusters: int) -> dict:
    data = yaml.safe_load(path.read_text())
    seen: dict[int, str] = {}
    for category in data["categories"]:
        for c in category["clusters"]:
            if c in seen:
                raise ValueError(f"{path}: shape cluster {c} is in both {seen[c]!r} and {category['label']!r}")
            seen[c] = category["label"]
    missing = sorted(set(range(1, clusters + 1)) - set(seen))
    if missing:
        raise ValueError(f"{path}: shape clusters {missing} belong to no category")
    if sum(bool(c.get("written")) for c in data["categories"]) > 1:
        raise ValueError(f"{path}: more than one category is the written character")
    return data


def proposal(code: str, run: str, groups: list[list[int]]) -> dict:
    """A starting categories file: one unnamed category per group of shape clusters."""
    return {"code_point": code, "run": run,
            "categories": [{"label": f"Group {i + 1}", "clusters": g, "text": ""} for i, g in enumerate(groups)],
            "checks": [], "notes": [], "sources": []}


def assign(categories: list[dict], shape: np.ndarray, classifier: np.ndarray, emb: np.ndarray,
           features: np.ndarray) -> tuple[np.ndarray, dict]:
    """Category index of every crop, and how the written-or-abbreviation call came out.

    `shape` and `classifier` are each crop's shape cluster and classifier cluster, `emb` its classifier
    embedding and `features` its encoder features.
    """
    of_cluster = {c: i for i, category in enumerate(categories) for c in category["clusters"]}
    by_shape = np.array([of_cluster[int(c)] for c in shape])
    written = next((i for i, c in enumerate(categories) if c.get("written")), None)
    if written is None:
        return by_shape, {}
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.neighbors import KNeighborsClassifier

    in_shape = by_shape == written
    clusters = [c for c in np.unique(classifier) if in_shape[classifier == c].mean() > 0.5]
    in_classifier = np.isin(classifier, clusters)
    agree = in_classifier == in_shape
    if min(in_classifier[agree].sum(), (~in_classifier[agree]).sum()) < MIN_CLASS:
        # The classifier clusters give too few examples of one side to learn from: the shape clusters decide.
        return by_shape, {"written_clusters": [int(c) for c in clusters], "skipped": True}
    x = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    clf = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced")
    held = cross_val_predict(clf, x[agree], in_classifier[agree], cv=5, method="predict_proba")[:, 1]
    clf.fit(x[agree], in_classifier[agree])
    is_written = np.where(agree, in_classifier, clf.predict_proba(x)[:, 1] > 0.5)
    category = np.where(is_written, written, by_shape)
    moved = in_shape & ~is_written
    if moved.any():
        abbreviated = ~in_shape
        if not abbreviated.any():
            raise ValueError("crops leave the written category, but no shape cluster is an abbreviation to give them")
        knn = KNeighborsClassifier(min(NEIGHBOURS, int(abbreviated.sum())), metric="cosine")
        category[moved] = knn.fit(features[abbreviated], by_shape[abbreviated]).predict(features[moved])
    return category, {
        "written_clusters": [int(c) for c in clusters],
        "agree": int(agree.sum()),
        "held_out_accuracy": float(((held > 0.5) == in_classifier[agree]).mean()),
        "disagree": int((~agree).sum()),
        "to_written": int((~agree & is_written).sum()),
        "to_abbreviation": int((~agree & ~is_written).sum()),
    }
