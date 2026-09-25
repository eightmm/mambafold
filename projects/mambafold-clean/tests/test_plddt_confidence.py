import math

import pytest
import torch
import torch.nn.functional as F

from mambafold.confidence import (
    PLDDT_CHECKPOINT_ARTIFACT_TYPE,
    PLDDTCheckpointError,
    PLDDTHead,
    PLDDTHeadConfig,
    expected_lddt,
    expected_plddt,
    hard_lddt_ca,
    lddt_bin_centers,
    load_plddt_checkpoint,
    macro_per_protein_cross_entropy,
    save_plddt_checkpoint,
    soft_adjacent_bin_labels,
)


def _small_config() -> PLDDTHeadConfig:
    return PLDDTHeadConfig(
        d_model=16,
        n_bins=8,
        n_layers=2,
        n_heads=4,
        ff_mult=2,
        dropout=0.0,
    )


def test_plddt_config_defaults_and_validation():
    assert PLDDTHeadConfig() == PLDDTHeadConfig(
        d_model=1024,
        n_bins=50,
        n_layers=4,
        n_heads=16,
        ff_mult=4,
        dropout=0.0,
    )
    with pytest.raises(ValueError, match="divisible"):
        PLDDTHeadConfig(d_model=15, n_heads=4)
    with pytest.raises(ValueError, match="n_bins"):
        PLDDTHeadConfig(n_bins=1)
    with pytest.raises(ValueError, match="dropout"):
        PLDDTHeadConfig(dropout=1.0)


def test_plddt_head_forward_masks_padding_and_empty_rows():
    torch.manual_seed(7)
    head = PLDDTHead(_small_config())
    latent = torch.randn(2, 4, 16, requires_grad=True)
    res_mask = torch.tensor([[True, True, False, False], [False, False, False, False]])

    logits = head(latent, res_mask)
    assert logits.shape == (2, 4, 8)
    assert torch.isfinite(logits).all()
    torch.testing.assert_close(logits[0, 2:], torch.zeros_like(logits[0, 2:]))
    torch.testing.assert_close(logits[1], torch.zeros_like(logits[1]))

    logits[0, :2].sum().backward()
    assert latent.grad is not None
    assert torch.isfinite(latent.grad).all()


def test_hard_lddt_ca_matches_hand_calculation_and_valid_mask():
    true_ca = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [20.0, 0.0, 0.0], [35.0, 0.0, 0.0]]]
    )
    pred_ca = torch.tensor(
        [[[0.0, 0.0, 0.0], [10.4, 0.0, 0.0], [22.0, 0.0, 0.0], [35.0, 0.0, 0.0]]]
    )
    mask = torch.ones(1, 4, dtype=torch.bool)

    score, valid = hard_lddt_ca(pred_ca, true_ca, mask)

    # True neighbors are 0<->1 and 1<->2.  The exact 15 Å pair 2<->3 is
    # excluded by the strict reference-neighbor cutoff.
    torch.testing.assert_close(score, torch.tensor([[1.0, 0.75, 0.5, 0.0]]))
    assert valid.tolist() == [[True, True, True, False]]


def test_hard_lddt_ca_uses_strict_error_thresholds_and_angstrom_units():
    true_ca = torch.tensor([[[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]]])
    pred_ca = torch.tensor([[[0.0, 0.0, 0.0], [10.5, 0.0, 0.0]]])
    score, valid = hard_lddt_ca(pred_ca, true_ca, torch.ones(1, 2, dtype=torch.bool))

    # Error == 0.5 fails the 0.5 Å threshold and passes 1/2/4 Å.
    torch.testing.assert_close(score, torch.full((1, 2), 0.75))
    assert valid.all()


@pytest.mark.parametrize("which", ["pred", "true"])
def test_hard_lddt_ca_rejects_nonfinite_valid_coordinates(which):
    pred_ca = torch.zeros(1, 2, 3)
    true_ca = torch.zeros(1, 2, 3)
    target = pred_ca if which == "pred" else true_ca
    target[0, 1, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        hard_lddt_ca(pred_ca, true_ca, torch.ones(1, 2, dtype=torch.bool))


def test_hard_lddt_ca_allows_nonfinite_padding_coordinates():
    pred_ca = torch.tensor([[[0.0, 0.0, 0.0], [float("nan"), 0.0, 0.0]]])
    true_ca = pred_ca.clone()
    scores, valid = hard_lddt_ca(
        pred_ca,
        true_ca,
        torch.tensor([[True, False]]),
    )
    torch.testing.assert_close(scores, torch.zeros_like(scores))
    assert not valid.any()


def test_bin_centers_and_soft_adjacent_labels():
    centers = lddt_bin_centers()
    assert centers.shape == (50,)
    assert centers[0].item() == pytest.approx(0.01)
    assert centers[-1].item() == pytest.approx(0.99)

    scores = torch.tensor([[0.0, 0.25, 0.5, 1.0]])
    mask = torch.tensor([[True, True, True, False]])
    labels = soft_adjacent_bin_labels(scores, mask, n_bins=4)

    torch.testing.assert_close(labels[0, 0], torch.tensor([1.0, 0.0, 0.0, 0.0]))
    torch.testing.assert_close(labels[0, 1], torch.tensor([0.5, 0.5, 0.0, 0.0]))
    torch.testing.assert_close(labels[0, 2], torch.tensor([0.0, 0.5, 0.5, 0.0]))
    torch.testing.assert_close(labels[0, 3], torch.zeros(4))


def test_macro_cross_entropy_weights_proteins_equally():
    logits = torch.tensor(
        [
            [[3.0, -1.0], [0.0, 0.0], [9.0, -9.0]],
            [[-2.0, 2.0], [4.0, -4.0], [1.0, -1.0]],
        ],
        requires_grad=True,
    )
    labels = torch.tensor(
        [
            [[1.0, 0.0], [0.5, 0.5], [0.0, 0.0]],
            [[0.0, 1.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, True, False], [True, False, False]])

    actual = macro_per_protein_cross_entropy(logits, labels, mask)
    residue_ce = -(labels * F.log_softmax(logits, dim=-1)).sum(dim=-1)
    expected = ((residue_ce[0, 0] + residue_ce[0, 1]) / 2 + residue_ce[1, 0]) / 2
    torch.testing.assert_close(actual, expected)
    actual.backward()
    assert logits.grad is not None


def test_macro_cross_entropy_empty_batch_is_differentiable_zero():
    logits = torch.randn(2, 3, 5, requires_grad=True)
    labels = torch.zeros_like(logits)
    mask = torch.zeros(2, 3, dtype=torch.bool)
    loss = macro_per_protein_cross_entropy(logits, labels, mask)
    assert loss.item() == 0.0
    loss.backward()
    torch.testing.assert_close(logits.grad, torch.zeros_like(logits))


def test_expected_scores_use_bin_centers_and_temperature():
    logits = torch.tensor([[[-100.0, -100.0, -100.0, 100.0]]])
    torch.testing.assert_close(expected_lddt(logits), torch.tensor([[0.875]]))
    torch.testing.assert_close(expected_plddt(logits), torch.tensor([[87.5]]))
    uniform = expected_lddt(torch.zeros(2, 3, 50), temperature=2.0)
    torch.testing.assert_close(uniform, torch.full((2, 3), 0.5))
    for bad_temperature in (0.0, -1.0, math.inf, math.nan, True):
        with pytest.raises(ValueError, match="temperature"):
            expected_lddt(torch.zeros(1, 2), temperature=bad_temperature)


def test_checkpoint_round_trip_is_separate_safe_and_strict(tmp_path):
    torch.manual_seed(19)
    config = _small_config()
    head = PLDDTHead(config).eval()
    latent = torch.randn(2, 3, config.d_model)
    mask = torch.tensor([[True, True, True], [True, False, False]])
    expected = head(latent, mask)

    path = tmp_path / "plddt_head.pt"
    save_plddt_checkpoint(
        path,
        head,
        temperature=1.25,
        step=3200,
        metrics={"val_macro_ce": 2.75},
        provenance={
            "folding_checkpoint_sha256": "abc123",
            "config": "configs/plddt_head.yaml",
            "datasets": ["confidence_train_v1", "confidence_val_v1"],
        },
    )
    raw = torch.load(path, map_location="cpu", weights_only=True)
    assert set(raw) == {
        "schema_version",
        "artifact_type",
        "head_config",
        "state_dict",
        "temperature",
        "step",
        "metrics",
        "provenance",
    }
    assert raw["artifact_type"] == PLDDT_CHECKPOINT_ARTIFACT_TYPE
    assert raw["temperature"] == 1.25
    assert all(isinstance(key, str) for key in raw["state_dict"])
    assert not any("path" in key.lower() for key in raw)

    restored, temperature, metadata = load_plddt_checkpoint(path, expected_config=config)
    assert temperature == 1.25
    assert metadata == {
        "step": 3200,
        "metrics": {"val_macro_ce": 2.75},
        "provenance": {
            "folding_checkpoint_sha256": "abc123",
            "config": "configs/plddt_head.yaml",
            "datasets": ["confidence_train_v1", "confidence_val_v1"],
        },
    }
    assert restored.training is False
    torch.testing.assert_close(restored(latent, mask), expected)

    raw["private_path"] = "/private/training/checkpoint.pt"
    tampered = tmp_path / "tampered.pt"
    torch.save(raw, tampered)
    with pytest.raises(PLDDTCheckpointError, match="invalid checkpoint keys"):
        load_plddt_checkpoint(tampered)


def test_checkpoint_rejects_incompatible_state_and_nonfinite_temperature(tmp_path):
    head = PLDDTHead(_small_config())
    path = tmp_path / "head.pt"
    save_plddt_checkpoint(path, head)
    raw = torch.load(path, map_location="cpu", weights_only=True)
    raw["state_dict"].pop("to_logits.bias")
    broken = tmp_path / "broken.pt"
    torch.save(raw, broken)
    with pytest.raises(PLDDTCheckpointError, match="incompatible"):
        load_plddt_checkpoint(broken)
    with pytest.raises(PLDDTCheckpointError, match="temperature"):
        save_plddt_checkpoint(tmp_path / "nan.pt", head, temperature=math.nan)
    with pytest.raises(PLDDTCheckpointError, match="private absolute path"):
        save_plddt_checkpoint(
            tmp_path / "private.pt",
            head,
            provenance={"source_checkpoint": "/home/user/private/ckpt.pt"},
        )
