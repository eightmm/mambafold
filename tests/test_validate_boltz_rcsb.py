import json

from scripts.validate_boltz_rcsb import iter_json_array


def test_iter_json_array_across_small_chunks(tmp_path):
    records = [
        {"id": "1abc", "chains": [{"chain_name": "A"}]},
        {"id": "2def", "chains": []},
    ]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(records))

    assert list(iter_json_array(path, chunk_size=7)) == records
