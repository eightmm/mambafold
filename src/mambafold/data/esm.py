"""On-the-fly protein language model embeddings via EvolutionaryScale ESM."""

from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
from torch import Tensor

from mambafold.data.constants import AA_3TO1, ID_TO_AA

SUPPORTED_PLM_MODES = {"esm3", "esmc", "blend", "concat"}


def residue_ids_to_sequences(res_type: Tensor, res_mask: Tensor) -> list[str]:
    """Convert residue type IDs into one-letter protein sequences."""
    sequences: list[str] = []
    for ids, mask in zip(res_type, res_mask):
        tokens = []
        for aa_id in ids[mask].tolist():
            aa3 = ID_TO_AA.get(int(aa_id), "UNK")
            tokens.append(AA_3TO1.get(aa3, "X"))
        sequences.append("".join(tokens))
    return sequences


class EvolutionaryScalePLM(nn.Module):
    """Frozen ESM3/ESMC embedder with selectable or mixed outputs."""

    def __init__(
        self,
        d_out: int = 1024,
        mode: str = "blend",
        esm3_model_name: str = "esm3-open",
        esmc_model_name: str = "esmc_600m",
    ):
        super().__init__()
        if mode not in SUPPORTED_PLM_MODES:
            raise ValueError(f"Unsupported PLM mode: {mode}")

        self.d_out = d_out
        self.mode = mode
        self.esm3_model_name = esm3_model_name
        self.esmc_model_name = esmc_model_name

        self.esm3_proj = nn.LazyLinear(d_out)
        self.esmc_proj = nn.LazyLinear(d_out)
        self.mix_gate = nn.Sequential(
            nn.Linear(2 * d_out, d_out),
            nn.SiLU(),
            nn.Linear(d_out, d_out),
        )
        self.concat_proj = nn.Linear(2 * d_out, d_out)

        self._esm3_client = None
        self._esmc_client = None
        self._esm_api = None

    def forward(self, res_type: Tensor, res_mask: Tensor) -> Tensor:
        """Return [B, L, d_out] PLM embeddings padded to batch length."""
        sequences = residue_ids_to_sequences(res_type, res_mask)
        device = res_type.device

        if self.mode == "esm3":
            esm3 = self.esm3_proj(self._embed_esm3(sequences, device))
            return esm3 * res_mask.unsqueeze(-1).to(esm3.dtype)

        if self.mode == "esmc":
            esmc = self.esmc_proj(self._embed_esmc(sequences, device))
            return esmc * res_mask.unsqueeze(-1).to(esmc.dtype)

        esm3 = self.esm3_proj(self._embed_esm3(sequences, device))
        esmc = self.esmc_proj(self._embed_esmc(sequences, device))

        if self.mode == "blend":
            gate = torch.sigmoid(self.mix_gate(torch.cat([esm3, esmc], dim=-1)))
            mixed = gate * esm3 + (1.0 - gate) * esmc
        else:
            mixed = self.concat_proj(torch.cat([esm3, esmc], dim=-1))

        return mixed * res_mask.unsqueeze(-1).to(mixed.dtype)

    def _embed_esm3(self, sequences: Iterable[str], device: torch.device) -> Tensor:
        client = self._get_esm3_client(device)
        return self._embed_sequences(client, sequences, device)

    def _embed_esmc(self, sequences: Iterable[str], device: torch.device) -> Tensor:
        client = self._get_esmc_client(device)
        return self._embed_sequences(client, sequences, device)

    def _embed_sequences(self, client, sequences: Iterable[str], device: torch.device) -> Tensor:
        _, _, ESMProtein, LogitsConfig = self._load_esm_api()
        embeddings = []
        max_len = 0
        feature_dim = None

        for sequence in sequences:
            if not sequence:
                seq_embed = torch.zeros(0, self.d_out, device=device)
                embeddings.append(seq_embed)
                continue

            with torch.no_grad():
                protein = ESMProtein(sequence=sequence)
                protein_tensor = client.encode(protein)
                logits_output = client.logits(
                    protein_tensor,
                    LogitsConfig(sequence=True, return_embeddings=True),
                )
                seq_embed = self._trim_special_tokens(logits_output.embeddings, len(sequence))
            embeddings.append(seq_embed.to(device=device, dtype=torch.float32))
            max_len = max(max_len, seq_embed.shape[0])
            feature_dim = seq_embed.shape[-1]

        feature_dim = feature_dim or self.d_out
        padded = torch.zeros(len(embeddings), max_len, feature_dim, device=device)
        for i, seq_embed in enumerate(embeddings):
            if seq_embed.numel() > 0:
                padded[i, : seq_embed.shape[0]] = seq_embed
        return padded

    def _trim_special_tokens(self, embeddings: Tensor | None, seq_len: int) -> Tensor:
        if embeddings is None:
            raise RuntimeError("EvolutionaryScale ESM did not return embeddings.")
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.shape[0] == seq_len + 2:
            return embeddings[1:-1]
        if embeddings.shape[0] == seq_len:
            return embeddings
        raise RuntimeError(
            f"Unexpected embedding length: got {embeddings.shape[0]}, expected {seq_len} or {seq_len + 2}."
        )

    def _get_esm3_client(self, device: torch.device):
        if self._esm3_client is None:
            ESM3, _, _, _ = self._load_esm_api()
            self._esm3_client = ESM3.from_pretrained(self.esm3_model_name).to(device)
            self._freeze(self._esm3_client)
        return self._move_client(self._esm3_client, device)

    def _get_esmc_client(self, device: torch.device):
        if self._esmc_client is None:
            _, ESMC, _, _ = self._load_esm_api()
            self._esmc_client = ESMC.from_pretrained(self.esmc_model_name).to(device)
            self._freeze(self._esmc_client)
        return self._move_client(self._esmc_client, device)

    def _move_client(self, client, device: torch.device):
        if hasattr(client, "device") and client.device != device:
            client = client.to(device)
        return client

    def _freeze(self, client) -> None:
        client.eval()
        for param in client.parameters():
            param.requires_grad_(False)

    def _load_esm_api(self):
        if self._esm_api is not None:
            return self._esm_api

        try:
            from esm.models.esm3 import ESM3
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as exc:
            raise RuntimeError(
                "EvolutionaryScale ESM is required for ESM3/ESMC support. "
                "Install `esm` from https://github.com/evolutionaryscale/esm "
                "and remove the legacy `fair-esm` package."
            ) from exc

        self._esm_api = (ESM3, ESMC, ESMProtein, LogitsConfig)
        return self._esm_api
