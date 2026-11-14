import json

import numpy as np
import yaml
from scipy.cluster.hierarchy import linkage

from allograph_clustering import pipeline


def run_with_tree(tmp_path):
    rng = np.random.default_rng(0)
    np.savez(tmp_path / "tree.npz", z=linkage(rng.normal(size=(60, 3)), "ward"))
    (tmp_path / "run.json").write_text(json.dumps({"run": "abc"}))
    return tmp_path


def test_a_missing_categories_file_gets_a_proposal_of_six_covering_every_shape_cluster(tmp_path):
    path = pipeline.categories_path("U+5019", run_with_tree(tmp_path), tmp_path / "none.yaml", log=lambda m: None)
    proposal = yaml.safe_load(path.read_text())
    assert path == tmp_path / "proposal.yaml" and proposal["run"] == "abc"
    assert sorted(c for category in proposal["categories"] for c in category["clusters"]) == list(range(1, 25))
    assert len(proposal["categories"]) == 6


def test_an_edited_proposal_of_the_same_run_is_kept(tmp_path):
    run = run_with_tree(tmp_path)
    path = pipeline.categories_path("U+5019", run, tmp_path / "none.yaml", log=lambda m: None)
    path.write_text(path.read_text().replace("Group 1", "Zigzag"))
    assert "Zigzag" in pipeline.categories_path("U+5019", run, tmp_path / "none.yaml", log=lambda m: None).read_text()
    (run / "run.json").write_text(json.dumps({"run": "def"}))
    assert "Zigzag" not in pipeline.categories_path("U+5019", run, tmp_path / "none.yaml", log=lambda m: None).read_text()
