"""`allograph-clustering cluster` and `allograph-clustering label`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from . import CATEGORIES, CLUSTERS, PACKS, RUNS, run_dir

app = typer.Typer(help="Allograph induction for one character: cluster its crops by stroke shape, then label them.",
                  no_args_is_help=True)

Code = Annotated[str, typer.Argument(help="code point of the character, such as U+5019")]
Packs = Annotated[Path, typer.Option(help="directory of the packs, one per character")]
Runs = Annotated[Path, typer.Option(help="directory of the runs, one per character")]


@app.command()
def cluster(
    code: Code,
    packs: Packs = PACKS,
    out: Runs = RUNS,
    steps: Annotated[int, typer.Option(help="restoration training steps")] = 6000,
    epochs: Annotated[int, typer.Option(help="encoder training epochs")] = 100,
    device: Annotated[str, typer.Option(help="torch device")] = "cuda",
) -> None:
    """Restore, encode and cluster one character's crops (15 to 30 minutes for 12,000 crops on a GPU)."""
    from . import pipeline

    typer.echo(json.dumps(pipeline.run(code, run_dir(code, out), packs=packs, steps=steps, epochs=epochs, device=device,
                                       log=lambda m: typer.echo(m, err=True)), ensure_ascii=False))


@app.command()
def label(
    code: Code,
    categories: Annotated[Path | None, typer.Option(help="categories file (default categories/<code>.yaml)")] = None,
    packs: Packs = PACKS,
    out: Runs = RUNS,
) -> None:
    """Classify every crop into the named categories and write `labels.tsv` and `report.html` into the run.

    Without a categories file, a proposal of six unnamed categories, the tree's top branches, is written as
    `proposal.yaml` in the run and used, to be edited and moved to `categories/`.
    """
    import numpy as np
    import yaml

    from . import classify, pack, report
    from . import cluster as clustering

    run = run_dir(code, out)
    meta = json.loads((run / "run.json").read_text())
    z = np.load(run / "tree.npz")["z"]
    shape = clustering.cut(z, CLUSTERS)
    path = categories or CATEGORIES / f"{code}.yaml"
    if not path.exists():
        top = clustering.cut(z, 6)
        groups = [sorted({int(c) for c in np.unique(shape[top == g])}) for g in range(1, 7)]
        path = run / "proposal.yaml"
        path.write_text(yaml.safe_dump(classify.proposal(code, meta["run"], groups), allow_unicode=True, sort_keys=False))
        typer.echo(f"no categories for {code}; proposal written to {path}", err=True)
    chosen = classify.load(path, CLUSTERS)
    data = pack.load(packs / f"{code}.npz")
    idx = np.load(run / "idx.npy")
    category, stats = classify.assign(chosen["categories"], shape, data["classifier_cluster"][idx].astype(int),
                                      data["embedding"][idx], np.load(run / "features.npy"))
    labels = [c["label"] for c in chosen["categories"]]
    with (run / "labels.tsv").open("w") as tsv:
        tsv.write("id\tclassifier_cluster\tshape_cluster\tlabel\n")
        for i, s, c in zip(idx, shape, category, strict=True):
            tsv.write(f"{data['ids'][i]}\t{int(data['classifier_cluster'][i])}\t{int(s)}\t{labels[c]}\n")
    char = chr(int(code.removeprefix("U+"), 16))
    page = report.render(run, chosen, path.parent / "drawings", report.page_data(run, data, chosen, category, stats), char)
    (run / "report.html").write_text(page)
    typer.echo(json.dumps({"labels": dict(zip(labels, np.bincount(category, minlength=len(labels)).tolist(), strict=True)),
                           **stats, "report": str(run / "report.html")}, ensure_ascii=False))
