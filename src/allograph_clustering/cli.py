"""`allograph-clustering cluster`, `label` and `desk`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer

from . import PACKS, RUNS, run_dir

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

    from . import pack, pipeline, report

    run = run_dir(code, out)
    path = pipeline.categories_path(code, run, categories, log=lambda m: typer.echo(m, err=True))
    data = pack.load(packs / f"{code}.npz")
    chosen, shape, category, stats = pipeline.label(run, data, path)
    idx = np.load(run / "idx.npy")
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


@app.command()
def desk(
    code: Code,
    categories: Annotated[Path | None, typer.Option(help="categories file (default categories/<code>.yaml)")] = None,
    packs: Packs = PACKS,
    out: Runs = RUNS,
    port: Annotated[int, typer.Option(help="port on 127.0.0.1")] = 8760,
) -> None:
    """Serve one run's report on this machine, classified again whenever the categories file changes."""
    from . import desk as desks
    from . import pack, pipeline

    run = run_dir(code, out)

    def echo(message: str) -> None:
        typer.echo(message, err=True)

    path = pipeline.categories_path(code, run, categories, log=echo)
    desks.serve(desks.Desk(code, run, pack.load(packs / f"{code}.npz"), path), port, log=echo)
