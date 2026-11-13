"""The clustering of one character's pack, each step skipped when its output is already in the run directory,
and the classification of a finished run under a categories file."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import CATEGORIES, CLUSTERS, classify, cluster, encoder, ink, pack, restore

#: Least usable crops a character needs: enough for the encoder's batches and for 24 shape clusters of some size.
MIN_CROPS = 500


def run(code: str, out: Path, *, packs: Path, steps: int = 6000, epochs: int = 100,
        device: str = "cuda", log=print) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    data = pack.load(packs / f"{code}.npz")
    size = data["size"]

    if not (out / "framed.npz").exists():
        log("framing")
        grey, mask, quality = ink.frame(data["grey"], size)
        np.savez(out / "framed.npz", grey=grey, mask=mask, quality=quality)
    framed = np.load(out / "framed.npz")

    if not (out / "restored.npy").exists():
        log("restoration")
        restored, crisp = restore.train(framed["grey"], framed["mask"], framed["quality"], steps=steps, device=device)
        np.save(out / "restored.npy", restored)
        (out / "restore.json").write_text(json.dumps({"crisp": crisp, "steps": steps}))
    if not (out / "ink.npy").exists():
        usable = (size[:, 0] > 0) & (framed["mask"].max(axis=(1, 2)) > 0)
        np.save(out / "ink.npy", ink.refit(np.load(out / "restored.npy"), usable))
    refit = np.load(out / "ink.npy")

    if not (out / "features.npy").exists():
        idx = np.nonzero(refit.max(axis=(1, 2)) > 0)[0]
        if len(idx) < MIN_CROPS:
            raise ValueError(f"{code}: {len(idx)} usable crops; shape clustering needs at least {MIN_CROPS}")
        log(f"encoders on {len(idx)} crops")
        light = encoder.train(refit[idx], epochs=epochs, seed=0, device=device)
        heavy = encoder.train(refit[idx], epochs=epochs, seed=1, heavy=True, device=device)
        np.save(out / "features.npy", np.concatenate([light, heavy], 1) / np.sqrt(2))
        np.save(out / "idx.npy", idx)
    features = np.load(out / "features.npy")

    if not (out / "tree.npz").exists():
        log("tree")
        y, z = cluster.build(features)
        np.savez(out / "tree.npz", y=y, z=z)

    summary = {
        "code_point": code,
        "run": hashlib.sha256(features.tobytes()).hexdigest()[:12],
        "revision": str(data["revision"]),
        "crops": len(size),
        "usable": len(np.load(out / "idx.npy")),
        **json.loads((out / "restore.json").read_text()),
        "epochs": epochs,
    }
    old = json.loads((out / "run.json").read_text()) if (out / "run.json").exists() else {}
    summary["created"] = old.get("created") if old.get("run") == summary["run"] else datetime.now(UTC).isoformat(timespec="seconds")
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def categories_path(code: str, run: Path, given: Path | None = None, log=print) -> Path:
    """The categories file of `code`: `given`, else `categories/<code>.yaml`, else a proposal written into the run.

    The proposal is six unnamed categories, the top branches of the tree."""
    import yaml

    path = given or CATEGORIES / f"{code}.yaml"
    if path.exists():
        return path
    z = np.load(run / "tree.npz")["z"]
    shape, top = cluster.cut(z, CLUSTERS), cluster.cut(z, 6)
    groups = [sorted({int(c) for c in np.unique(shape[top == g])}) for g in range(1, 7)]
    meta = json.loads((run / "run.json").read_text())
    path = run / "proposal.yaml"
    path.write_text(yaml.safe_dump(classify.proposal(code, meta["run"], groups), allow_unicode=True, sort_keys=False))
    log(f"no categories for {code}; proposal written to {path}")
    return path


def label(run: Path, data: dict, path: Path) -> tuple[dict, np.ndarray, np.ndarray, dict]:
    """The categories of `path`, and each usable crop's shape cluster and category index, with the classification's figures."""
    chosen = classify.load(path, CLUSTERS)
    idx = np.load(run / "idx.npy")
    shape = cluster.cut(np.load(run / "tree.npz")["z"], CLUSTERS)
    category, stats = classify.assign(chosen["categories"], shape, data["classifier_cluster"][idx].astype(int),
                                      data["embedding"][idx], np.load(run / "features.npy"))
    return chosen, shape, category, stats
