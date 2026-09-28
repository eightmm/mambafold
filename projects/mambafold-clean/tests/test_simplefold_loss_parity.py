from __future__ import annotations

import torch

from mambafold.data.constants import CA_ATOM_ID, COORD_SCALE
from mambafold.data.transforms import _sample_t, flow_corrupt
from mambafold.data.types import ProteinBatch
from mambafold.losses.lddt import soft_lddt_all_atom_loss
from mambafold.train import engine
from mambafold.utils.geometry import weighted_rigid_align


def _dense_simplefold_lddt(
    pred: torch.Tensor,
    true: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    losses = []
    for batch_index in range(pred.shape[0]):
        mask = valid_mask[batch_index].reshape(-1)
        pred_atoms = pred[batch_index].reshape(-1, 3)[mask] * COORD_SCALE
        true_atoms = true[batch_index].reshape(-1, 3)[mask] * COORD_SCALE
        pred_distance = torch.cdist(pred_atoms, pred_atoms)
        true_distance = torch.cdist(true_atoms, true_atoms)
        pair_mask = true_distance < 15.0
        pair_mask &= ~torch.eye(
            true_atoms.shape[0], dtype=torch.bool, device=true_atoms.device
        )
        error = (pred_distance - true_distance).abs()
        score = sum(torch.sigmoid(threshold - error) for threshold in (0.5, 1.0, 2.0, 4.0)) / 4
        mask_f = pair_mask.to(score.dtype)
        losses.append(1.0 - (score * mask_f).sum() / mask_f.sum().clamp(min=1))
    return torch.stack(losses)


def _small_batch(batch_size: int = 2) -> ProteinBatch:
    length = 1
    atoms = 2
    mask = torch.ones(batch_size, length, atoms, dtype=torch.bool)
    ca_mask = mask[..., CA_ATOM_ID]
    zeros_la3 = torch.zeros(batch_size, length, atoms, 3)
    zeros_l = torch.zeros(batch_size, length, dtype=torch.long)
    return ProteinBatch(
        res_type=zeros_l,
        res_seq_nums=zeros_l,
        atom_type=torch.zeros(batch_size, length, atoms, dtype=torch.long),
        pair_type=torch.zeros(batch_size, length, atoms, dtype=torch.long),
        res_mask=torch.ones(batch_size, length, dtype=torch.bool),
        atom_mask=mask,
        valid_mask=mask,
        ca_mask=ca_mask,
        chain_id=zeros_l,
        entity_id=zeros_l,
        sym_id=zeros_l,
        is_nterm=torch.zeros(batch_size, length, dtype=torch.bool),
        is_cterm=torch.zeros(batch_size, length, dtype=torch.bool),
        x_clean=zeros_la3,
        x_t=zeros_la3,
        eps=zeros_la3,
        t=torch.tensor([0.0, 1.0]).reshape(batch_size, 1, 1, 1),
        esm=None,
    )


def _baseline_loss_kwargs() -> dict[str, float | int | bool | str]:
    return {
        "alpha_mode": "ramp",
        "use_rigid_align": False,
        "w_fm": 1.0,
        "w_lddt_atom": 1.0,
        "lddt_cutoff_A": 15.0,
        "lddt_pair_chunk_size": 2,
    }


def test_chunked_exact_all_atom_lddt_matches_dense_value_and_gradient() -> None:
    torch.manual_seed(7)
    true = torch.randn(2, 3, 4, 3, dtype=torch.float64) * 0.15
    valid_mask = torch.tensor(
        [
            [[1, 1, 1, 0], [1, 1, 0, 0], [1, 1, 1, 1]],
            [[1, 1, 0, 0], [1, 1, 1, 0], [1, 0, 0, 0]],
        ],
        dtype=torch.bool,
    )
    pred_dense = (true + 0.025 * torch.randn_like(true)).requires_grad_(True)
    pred_chunked = pred_dense.detach().clone().requires_grad_(True)

    expected = _dense_simplefold_lddt(pred_dense, true, valid_mask)
    actual = soft_lddt_all_atom_loss(
        pred_chunked,
        true,
        valid_mask,
        pair_chunk_size=2,
        reduction="none",
    )
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)

    expected.mean().backward()
    actual.mean().backward()
    torch.testing.assert_close(pred_chunked.grad, pred_dense.grad, rtol=1e-9, atol=1e-9)


def test_exact_all_atom_lddt_is_invariant_to_chunk_size() -> None:
    torch.manual_seed(11)
    true = torch.randn(1, 3, 3, 3) * 0.1
    pred = true + 0.03 * torch.randn_like(true)
    mask = torch.ones(1, 3, 3, dtype=torch.bool)
    losses = [
        soft_lddt_all_atom_loss(pred, true, mask, pair_chunk_size=size)
        for size in (1, 2, 64)
    ]
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(losses[0], losses[2])


def test_per_example_alpha_is_applied_before_batch_reduction(monkeypatch) -> None:
    batch = _small_batch()
    output = {"v_atom": torch.zeros_like(batch.x_clean)}

    def fake_atom_lddt(*_args, reduction: str, **_kwargs) -> torch.Tensor:
        assert reduction == "none"
        return torch.tensor([1.0, 0.0])

    monkeypatch.setattr(engine, "soft_lddt_all_atom_loss", fake_atom_lddt)
    total, metrics = engine.allatom_loss_surface(
        output,
        batch,
        **_baseline_loss_kwargs(),
    )

    # alpha=[1, 5], losses=[1, 0] -> mean(alpha_i * loss_i)=0.5.
    torch.testing.assert_close(total, torch.tensor(0.5))
    assert metrics["lddt_atom"] == 0.5
    assert metrics["lddt_atom_weighted"] == 0.5
    assert metrics["alpha"] == 3.0


def test_composite_baseline_loss_has_finite_gradient() -> None:
    torch.manual_seed(23)
    batch_size, length, atoms = 2, 2, 3
    clean = torch.randn(batch_size, length, atoms, 3) * 0.1
    eps = torch.randn_like(clean)
    t = torch.tensor([0.3, 0.8]).reshape(batch_size, 1, 1, 1)
    mask = torch.ones(batch_size, length, atoms, dtype=torch.bool)
    zeros_l = torch.zeros(batch_size, length, dtype=torch.long)
    batch = ProteinBatch(
        res_type=zeros_l,
        res_seq_nums=zeros_l,
        atom_type=torch.zeros(batch_size, length, atoms, dtype=torch.long),
        pair_type=torch.zeros(batch_size, length, atoms, dtype=torch.long),
        res_mask=torch.ones(batch_size, length, dtype=torch.bool),
        atom_mask=mask,
        valid_mask=mask,
        ca_mask=mask[..., CA_ATOM_ID],
        chain_id=zeros_l,
        entity_id=zeros_l,
        sym_id=zeros_l,
        is_nterm=torch.zeros(batch_size, length, dtype=torch.bool),
        is_cterm=torch.zeros(batch_size, length, dtype=torch.bool),
        x_clean=clean,
        x_t=t * clean + (1.0 - t) * eps,
        eps=eps,
        t=t,
        esm=None,
    )
    velocity = torch.randn_like(clean, requires_grad=True)
    output = {"v_atom": velocity}
    kwargs = _baseline_loss_kwargs()
    kwargs["use_rigid_align"] = True
    total, _ = engine.allatom_loss_surface(output, batch, **kwargs)
    total.backward()

    assert torch.isfinite(total)
    assert velocity.grad is not None
    assert torch.isfinite(velocity.grad).all()


def test_fm_mse_averages_xyz_and_then_proteins_equally() -> None:
    pred = torch.zeros(2, 1, 2, 3)
    target = torch.zeros_like(pred)
    mask = torch.tensor([[[1, 0]], [[1, 1]]], dtype=torch.bool)
    pred[0, 0, 0] = 1.0
    pred[1, 0, 0] = 2.0
    pred[1, 0, 1] = 4.0

    per_example = engine._masked_mse_per_example(pred, target, mask)
    # Protein 0: mean([1,1,1])=1. Protein 1: mean([4,4,4,16,16,16])=10.
    torch.testing.assert_close(per_example, torch.tensor([1.0, 10.0]))
    torch.testing.assert_close(per_example.mean(), torch.tensor(5.5))


def test_weighted_rigid_align_recovers_rotation_and_translation() -> None:
    true = torch.tensor(
        [[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]]
    )
    rotation = torch.tensor([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    pred = true @ rotation.T + torch.tensor([2.0, -3.0, 5.0])
    aligned = weighted_rigid_align(true, pred, torch.ones(1, 4, dtype=torch.bool))
    torch.testing.assert_close(aligned, pred, rtol=1e-5, atol=1e-5)


def test_logit_normal_sampler_matches_simplefold_formula(monkeypatch) -> None:
    monkeypatch.setattr(torch, "randn", lambda *_args, **_kwargs: torch.tensor([0.0]))
    monkeypatch.setattr(torch, "rand", lambda *_args, **_kwargs: torch.tensor([0.25]))
    logit_normal = float(torch.sigmoid(torch.tensor(0.8)))
    expected = (0.98 * logit_normal + 0.02 * 0.25) * (1.0 - 2.0e-4) + 1.0e-4
    assert abs(_sample_t("logit_normal") - expected) < 1e-7


def test_flow_corruption_keeps_uncentered_gaussian_noise(monkeypatch) -> None:
    noise = torch.tensor([[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]])
    monkeypatch.setattr(torch, "randn_like", lambda _coords: noise.clone())
    monkeypatch.setattr(
        "mambafold.data.transforms._sample_t",
        lambda _schedule, _uniform_weight=0.02: 0.25,
    )
    coords = torch.zeros_like(noise)
    mask = torch.tensor([[True, True]])
    x_t, eps, t = flow_corrupt(coords, mask, "logit_normal")
    torch.testing.assert_close(eps, noise)
    torch.testing.assert_close(x_t, 0.75 * noise)
    assert t == 0.25
