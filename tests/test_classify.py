from pathlib import Path

import numpy as np
import pytest
import yaml

from allograph_clustering import classify


def write(tmp_path: Path, categories: list[dict]) -> Path:
    path = tmp_path / "U+5019.yaml"
    path.write_text(yaml.safe_dump({"code_point": "U+5019", "run": "abc", "categories": categories}, allow_unicode=True))
    return path


def test_load_accepts_categories_covering_every_shape_cluster(tmp_path):
    path = write(tmp_path, [{"label": "候", "written": True, "clusters": [1, 2]}, {"label": "ん", "clusters": [3, 4]}])
    assert [c["label"] for c in classify.load(path, 4)["categories"]] == ["候", "ん"]


@pytest.mark.parametrize(("categories", "message"), [
    ([{"label": "a", "clusters": [1, 2]}, {"label": "b", "clusters": [2, 3, 4]}], "shape cluster 2"),
    ([{"label": "a", "clusters": [1, 2]}], r"shape clusters \[3, 4\]"),
    ([{"label": "a", "written": True, "clusters": [1, 2]}, {"label": "b", "written": True, "clusters": [3, 4]}], "more than one"),
])
def test_load_rejects_overlap_gaps_and_two_written_categories(tmp_path, categories, message):
    with pytest.raises(ValueError, match=message):
        classify.load(write(tmp_path, categories), 4)


def test_proposal_names_one_category_per_group():
    proposal = classify.proposal("U+5019", "abc", [[1, 2], [3]])
    assert [c["label"] for c in proposal["categories"]] == ["Group 1", "Group 2"]
    assert proposal["categories"][1]["clusters"] == [3]


def test_assign_without_a_written_category_follows_the_shape_clusters():
    categories = [{"label": "a", "clusters": [1, 2]}, {"label": "b", "clusters": [3]}]
    shape = np.array([1, 3, 2, 3])
    category, stats = classify.assign(categories, shape, np.ones(4, int), np.ones((4, 2)), np.ones((4, 2)))
    assert category.tolist() == [0, 1, 0, 1]
    assert stats == {}


def test_assign_lets_the_classifier_embedding_decide_crops_the_clusterings_disagree_on():
    rng = np.random.default_rng(0)
    categories = [{"label": "候", "written": True, "clusters": [1]}, {"label": "ん", "clusters": [2]}, {"label": "ゝ", "clusters": [3]}]
    # classifier cluster 1 is written, 2 abbreviated; the embedding separates them along its first axis
    classifier = np.repeat([1, 2], 60)
    emb = np.c_[np.where(classifier == 1, 1.0, -1.0), rng.normal(0, 0.1, (120, 3))]
    shape = np.where(classifier == 1, 1, rng.choice([2, 3], 120))
    shape[0] = 2       # a written crop the shape tree put among the abbreviations
    shape[70] = 1      # an abbreviation the shape tree put among the written
    features = np.c_[(shape == 3).astype(float), rng.normal(0, 0.1, (120, 3))]
    category, stats = classify.assign(categories, shape, classifier, emb, features)
    assert category[0] == 0
    assert category[70] in (1, 2)
    assert stats["disagree"] == 2 and stats["to_written"] == 1 and stats["to_abbreviation"] == 1


def test_assign_falls_back_to_shape_when_no_classifier_cluster_is_mostly_written():
    categories = [{"label": "候", "written": True, "clusters": [1]}, {"label": "ん", "clusters": [2]}]
    shape = np.array([1] * 20 + [2] * 80)
    category, stats = classify.assign(categories, shape, np.ones(100, int), np.ones((100, 3)), np.ones((100, 3)))
    assert category.tolist() == [0] * 20 + [1] * 80
    assert stats["skipped"] and stats["written_clusters"] == []


def test_the_committed_categories_load_and_match_the_committed_labels():
    root = Path(__file__).parent.parent
    for path in sorted((root / "categories").glob("U+*.yaml")):
        data = classify.load(path, 24)
        labels = {row.split("\t")[3] for row in (root / "labels" / f"{path.stem}.tsv").read_text().splitlines()[1:]}
        assert labels == {c["label"] for c in data["categories"]}
        for category in data["categories"]:
            for name in category.get("drawings", []):
                assert (path.parent / "drawings" / f"{name}.svg").exists()
