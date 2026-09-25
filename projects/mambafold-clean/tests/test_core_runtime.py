from __future__ import annotations

import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from mambafold.data.collate import ProteinCollator
from mambafold.data.constants import CA_ATOM_ID, MAX_ATOMS_PER_RES, PAIR_PAD_ID
from mambafold.data.types import ProteinBatch, ProteinExample
from mambafold.model.fold import MambaFoldAllAtom
from mambafold.sampling import prepare_inference_batch, sample
from mambafold.train.config import parse_args
from mambafold.train.engine import allatom_loss_surface


def _batch(batch_size: int = 2, length: int = 2) -> ProteinBatch:
    atoms = MAX_ATOMS_PER_RES
    atom_mask = torch.zeros(batch_size, length, atoms, dtype=torch.bool)
    atom_mask[..., :5] = True
    clean = torch.randn(batch_size, length, atoms, 3) * 0.1
    eps = torch.randn_like(clean)
    t = torch.linspace(0.25, 0.75, batch_size).reshape(batch_size, 1, 1, 1)
    zeros_l = torch.zeros(batch_size, length, dtype=torch.long)
    pair_type = torch.full((batch_size, length, atoms), PAIR_PAD_ID, dtype=torch.long)
    pair_type[atom_mask] = 0
    return ProteinBatch(
        res_type=zeros_l,
        res_seq_nums=torch.arange(length).expand(batch_size, -1),
        atom_type=torch.zeros(batch_size, length, atoms, dtype=torch.long),
        pair_type=pair_type,
        res_mask=torch.ones(batch_size, length, dtype=torch.bool),
        atom_mask=atom_mask,
        valid_mask=atom_mask,
        ca_mask=atom_mask[..., CA_ATOM_ID],
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


def _tiny_model(*, use_plm: bool = False, d_plm: int = 8) -> MambaFoldAllAtom:
    # Zero SSM layers keeps the smoke test CPU-only while exercising every
    # embedding, pooling, conditioning, and folding-output parameter.
    return MambaFoldAllAtom(
        d_res=32,
        n_trunk=0,
        d_res_type=8,
        d_res_pos=8,
        d_plm=d_plm,
        d_plm_proj=8,
        d_ca_emb=16,
        use_plm=use_plm,
        mimo_rank=1,
        d_state=16,
        expand=1,
        headdim=8,
        d_atom=16,
        n_atom_layers=0,
        # The cross-residue backbone mixer is an SSM too, so it has to go with
        # the rest of them for this to stay CPU-only. It is exercised on GPU
        # instead — nothing here covers it.
        n_atom_cross_layers=0,
        self_conditioning=True,
    )


def test_folding_model_has_only_core_outputs_and_all_parameters_receive_gradients() -> None:
    torch.manual_seed(5)
    model = _tiny_model()
    assert not any(
        token in name
        for name, _ in model.named_parameters()
        for token in ("conf", "pcb", "distogram", "contact", "aux_pair")
    )

    batch = _batch()
    output = model(batch)
    assert set(output) == {"v_atom", "trunk_latent"}
    loss, _ = allatom_loss_surface(
        output,
        batch,
        use_rigid_align=False,
        w_lddt_atom=0.0,
    )
    loss.backward()

    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert missing == []
    assert all(
        torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.grad is not None
    )


@pytest.mark.skipif(not dist.is_available(), reason="torch.distributed is unavailable")
def test_two_ddp_iterations_need_no_unused_parameter_detection(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    rendezvous = tmp_path / "ddp-rendezvous"
    try:
        dist.init_process_group(
            "gloo",
            init_method=f"file://{rendezvous}",
            rank=0,
            world_size=1,
        )
    except RuntimeError as exc:
        if "Operation not permitted" in str(exc) or "Cannot resolve" in str(exc):
            pytest.skip(f"sandbox does not permit a local Gloo socket: {exc}")
        raise
    try:
        model = DDP(_tiny_model(), find_unused_parameters=False)
        optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
        batch = _batch()
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            output = model(batch)
            loss, _ = allatom_loss_surface(
                output,
                batch,
                use_rigid_align=False,
                w_lddt_atom=0.0,
            )
            loss.backward()
            optimizer.step()
    finally:
        dist.destroy_process_group()


def _example(length: int) -> ProteinExample:
    atoms = MAX_ATOMS_PER_RES
    atom_mask = torch.zeros(length, atoms, dtype=torch.bool)
    atom_mask[:, :5] = True
    pair_type = torch.full((length, atoms), PAIR_PAD_ID, dtype=torch.long)
    pair_type[atom_mask] = 0
    return ProteinExample(
        res_type=torch.zeros(length, dtype=torch.long),
        atom_type=torch.zeros(length, atoms, dtype=torch.long),
        pair_type=pair_type,
        coords=torch.zeros(length, atoms, 3),
        atom_mask=atom_mask,
        observed_mask=atom_mask.clone(),
        res_seq_nums=torch.arange(length),
        seq_len=length,
    )


class _DummyVelocity(torch.nn.Module):
    def forward(self, batch: ProteinBatch) -> dict[str, torch.Tensor]:
        return {
            "v_atom": -0.25 * batch.x_t,
            "trunk_latent": batch.res_type.float().unsqueeze(-1),
        }


@pytest.mark.parametrize("method", ["ode", "sde"])
def test_batch_one_sampler_is_the_same_batch_first_path(method: str) -> None:
    model = _DummyVelocity()
    short, long = _example(2), _example(3)
    batched = sample(
        model,
        prepare_inference_batch([short, long], "cpu"),
        [3, 7],
        n_steps=3,
        method=method,
    )
    single = sample(
        model,
        prepare_inference_batch([short], "cpu"),
        [3],
        n_steps=3,
        method=method,
    )

    torch.testing.assert_close(batched.final_aa[0, :2], single.final_aa[0])
    torch.testing.assert_close(batched.final_ca[0, :2], single.final_ca[0])
    assert batched.residue_mask[0].tolist() == [True, True, False]
    assert not hasattr(batched, "confidence")


def test_noise_copies_share_one_plm_row_and_model_broadcasts_it() -> None:
    torch.manual_seed(17)
    example = _example(2)
    example.esm = torch.randn(2, 8)
    batch = ProteinCollator(
        augment=False,
        copies_per_protein=2,
        max_length=2,
    )([example])
    assert batch is not None
    assert batch.batch_size == 2
    assert batch.esm is not None and batch.esm.shape == (1, 2, 8)
    assert not torch.equal(batch.eps[0], batch.eps[1])

    model = _tiny_model(use_plm=True, d_plm=8)
    output = model(batch)
    assert output["v_atom"].shape[0] == 2
    output["v_atom"].sum().backward()
    assert model.plm_proj.weight.grad is not None


def test_yaml_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config = tmp_path / "typo.yaml"
    config.write_text("w_confidence_typo: 1.0\n")
    with pytest.raises(SystemExit):
        parse_args(["--config", str(config), "--out_dir", str(tmp_path / "out")])


def test_length_bucketed_sampler_draws_uniformly_and_groups_by_length() -> None:
    from mambafold.data.length_sampler import LengthBucketedDistributedBatchSampler

    # 200 short chains and 200 long ones. The weighting this sampler used to
    # apply unconditionally, w = clip((L/200)**0.5, 1.0, 1.5), scores 1.0 at
    # L=100 and 1.5 at L=900, so it would put the long half at 0.60. Uniform is
    # 0.50, and 4000 draws give a standard error of 0.008 — the two are ~12
    # sigma apart, so this band separates them without being flaky.
    index_lengths = {i: (100 if i < 200 else 900) for i in range(400)}
    sampler = LengthBucketedDistributedBatchSampler(
        index_lengths=index_lengths, batch_size=4, rank=0, world_size=1, seed=0,
        num_samples_per_rank=4000,
    )
    drawn = [i for batch in sampler for i in batch]
    long_share = sum(i >= 200 for i in drawn) / len(drawn)
    assert 0.47 < long_share < 0.53, f"draw is length-weighted: {long_share:.3f} long"

    # Batches are cut from length-sorted megabatches, so exactly one batch per
    # megabatch can straddle the short/long boundary. Everything else must be
    # length-pure, which is the property the collator's padding depends on.
    mixed = sum(1 for batch in sampler if len({index_lengths[i] for i in batch}) > 1)
    assert mixed <= math.ceil(len(sampler) / sampler.megabatch_mult), (
        f"{mixed} of {len(sampler)} batches mix lengths; bucketing is not grouping"
    )


def test_chain_index_cache_key_ignores_how_the_path_was_spelled(tmp_path: Path) -> None:
    """The prebuilt index must be findable by the loader that consumes it.

    `pipeline/13_prebuild_loader_caches.py` constructs its dataset from absolute
    paths (`ROOT / source["data_dir"]`) while the training loader passes the
    config's relative string through unchanged. The cache key hashes the
    directory and the file list, so the two spellings produced different keys:
    the stage spent an hour building an index training could never find, and the
    only symptom was that training rebuilt it silently.
    """
    from mambafold.data.dataset import RCSBDataset
    from mambafold.data.length_cache import _DEFAULT_CACHE_DIR, _cache_path

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name in ("a.npz", "b.npz"):
        (corpus / name).touch()

    def key(data_dir: str) -> str:
        dataset = RCSBDataset(
            data_dir=data_dir, max_length=1024, extract_monomer_chains=False
        )
        return _cache_path(dataset, Path(_DEFAULT_CACHE_DIR)).name

    relative = os.path.relpath(corpus, Path.cwd())
    assert key(str(corpus)) == key(relative), (
        "absolute and relative spellings of one corpus hash differently, so a "
        "prebuilt cache cannot be found by the training loader"
    )


def test_single_rank_accumulation_over_one_protein_is_runnable() -> None:
    """A one-GPU run must be able to start.

    `accum_same_protein` repeats indices through the batch sampler, and the
    batch sampler only existed when `batch_size * world_size > 1`. On one rank
    at batch_size 1 that product is 1, so the sampler was absent and the loader
    raised — the configuration this project trains could not be run on a single
    GPU at all, which is exactly the configuration anyone debugging would reach
    for. Measured as a hard failure of the one-rank stage of the DDP smoke.
    """
    from mambafold.data.loader import LengthBucketedDistributedBatchSampler, RepeatBatchSampler

    base = LengthBucketedDistributedBatchSampler(
        index_lengths={i: 100 + i for i in range(32)},
        batch_size=1, rank=0, world_size=1, seed=0,
    )
    repeated = RepeatBatchSampler(base, 2)
    batches = list(repeated)
    assert len(batches) == 2 * len(base)
    # Consecutive pairs must be the same indices: that identity is what makes
    # accumulation over a protein equal to one pass with twice the copies.
    assert all(batches[i] == batches[i + 1] for i in range(0, len(batches) - 1, 2))
