from pathlib import Path

import numpy as np

from mambafold.data.dataset import RCSBDataset
from mambafold.data.sequence_cache import sequence_digest, sequence_embedding_path


def test_sequence_path_is_content_addressed_by_full_sequence(tmp_path):
    sequence = "ACDEFG"
    path_a = sequence_embedding_path(tmp_path, sequence)
    path_b = sequence_embedding_path(tmp_path, sequence)

    assert path_a == path_b
    assert path_a == (
        tmp_path
        / "by_sequence"
        / sequence_digest(sequence)[:2]
        / f"{sequence_digest(sequence)}.npy"
    )
    assert sequence_embedding_path(tmp_path, sequence + "H") != path_a
    shared_prefix = "A" * 1024
    assert sequence_embedding_path(tmp_path, shared_prefix + "C") != sequence_embedding_path(
        tmp_path, shared_prefix + "D"
    )


def test_rcsb_cache_resolver_prefers_sequence_and_falls_back(tmp_path):
    dataset = RCSBDataset.__new__(RCSBDataset)
    dataset.esm_dir = Path(tmp_path)
    sequence = "ACDEFG"
    legacy_path = tmp_path / "1abc_ch0.npy"
    np.save(legacy_path, np.zeros((6, 4), dtype=np.float16))

    assert dataset._esm_embedding_path("1abc", 0, sequence) == legacy_path

    sequence_path = sequence_embedding_path(tmp_path, sequence)
    sequence_path.parent.mkdir(parents=True)
    np.save(sequence_path, np.ones((6, 4), dtype=np.float16))

    assert dataset._esm_embedding_path("1abc", 0, sequence) == sequence_path
