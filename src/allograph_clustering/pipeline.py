"""The clustering of one character's pack, each step skipped when its output is already in the run directory."""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from . import cluster, encoder, ink, pack, restore

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
