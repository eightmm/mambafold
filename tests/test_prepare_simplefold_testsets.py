import zipfile

from scripts.prepare_simplefold_testsets import (
    PRIMARY_REFERENCE_MIN_IDENTITY,
    TASKS,
    archive_index,
    split_target_id,
)


def test_archive_index_enforces_models_targets_and_samples(tmp_path):
    archive_path = tmp_path / "apo_predictions.zip"
    task = TASKS["apo"]
    with zipfile.ZipFile(archive_path, "w") as archive:
        for model in (
            "simplefold_100M",
            "simplefold_360M",
            "simplefold_700M",
            "simplefold_1.1B",
            "simplefold_1.6B",
            "simplefold_3B",
        ):
            for target_index in range(task.expected_targets):
                for sample in range(task.expected_samples):
                    archive.writestr(
                        f"apo_predictions/{model}/T{target_index}_sampled_{sample}.cif",
                        "data_x\n",
                    )
        archive.writestr(
            "__MACOSX/apo_predictions/simplefold_3B/._T0_sampled_0.cif", "metadata"
        )

    index = archive_index(archive_path, task)
    assert len(index["simplefold_3B"]) == 90
    assert set(index["simplefold_3B"]["T0"]) == set(range(5))


def test_split_target_id_normalizes_eigenfold_name():
    assert split_target_id("7skh.B.pdb") == "7skh_b"


def test_primary_reference_identity_contract_is_explicit():
    assert PRIMARY_REFERENCE_MIN_IDENTITY == 0.95
