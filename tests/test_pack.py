import numpy as np
import pytest

from allograph_clustering import pack


def test_load_names_the_download_when_the_pack_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match="gh release download packs"):
        pack.load(tmp_path / "U+5019.npz")


def test_load_rejects_a_pack_without_every_field(tmp_path):
    np.savez(tmp_path / "U+5019.npz", ids=np.array(["a:1"]))
    with pytest.raises(ValueError, match="classifier_cluster"):
        pack.load(tmp_path / "U+5019.npz")
