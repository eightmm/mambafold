"""Standalone ESM embedding extractor (not part of model weights).

Usage:
    embedder = ESMEmbedder("esm3-open")
    # In collator or inference:
    esm_out = embedder(sequences, max_length=384)  # [B, max_length, d_esm]
"""

from __future__ import annotations

import torch
from torch import Tensor


class ESMEmbedder:
    """Standalone ESM embedding extractor.

    Loads an ESM model (ESMC or ESM3) and extracts per-residue embeddings.
    Not an nn.Module — ESM weights are never part of the training model.
    """

    def __init__(self, model_name: str = "esm3-open", device: str = "cuda"):
        self.model_name = model_name
        self.device = torch.device(device)
        self._client = None
        self._api = None
        self.d_esm: int | None = None

    @torch.no_grad()
    def __call__(
        self, sequences: list[str], max_length: int | None = None,
    ) -> Tensor:
        """Extract ESM embeddings for a list of protein sequences.

        Args:
            sequences: List of one-letter amino acid sequences.
            max_length: Pad/truncate to this length. If None, use max seq len.

        Returns:
            Tensor of shape [B, L, d_esm] (float32, on self.device).
        """
        client = self._get_client()
        _, _, ESMProtein, LogitsConfig = self._get_api()

        embeddings = []
        for seq in sequences:
            if not seq:
                embeddings.append(None)
                continue
            protein = ESMProtein(sequence=seq)
            protein_tensor = client.encode(protein)
            logits_output = client.logits(
                protein_tensor,
                LogitsConfig(sequence=True, return_embeddings=True),
            )
            emb = self._trim_special_tokens(logits_output.embeddings, len(seq))
            embeddings.append(emb.to(device=self.device, dtype=torch.float32))

        # Determine dimensions
        real = [e for e in embeddings if e is not None]
        if not real:
            d = self.d_esm or 1
            L = max_length or 1
            return torch.zeros(len(sequences), L, d, device=self.device)

        d = real[0].shape[-1]
        self.d_esm = d
        L = max_length or max(e.shape[0] for e in real)

        # Pad to [B, L, d]
        out = torch.zeros(len(embeddings), L, d, device=self.device)
        for i, emb in enumerate(embeddings):
            if emb is not None:
                seq_len = min(emb.shape[0], L)
                out[i, :seq_len] = emb[:seq_len]
        return out

    def _trim_special_tokens(self, embeddings: Tensor | None, seq_len: int) -> Tensor:
        if embeddings is None:
            raise RuntimeError("ESM did not return embeddings.")
        if embeddings.dim() == 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.shape[0] == seq_len + 2:
            return embeddings[1:-1]
        if embeddings.shape[0] == seq_len:
            return embeddings
        raise RuntimeError(
            f"Unexpected embedding length: got {embeddings.shape[0]}, "
            f"expected {seq_len} or {seq_len + 2}."
        )

    def _get_client(self):
        if self._client is not None:
            return self._client
        api = self._get_api()
        name = self.model_name
        if name.startswith("esmc"):
            _, ESMC, _, _ = api
            self._client = ESMC.from_pretrained(name).to(self.device)
        else:
            ESM3, _, _, _ = api
            self._client = ESM3.from_pretrained(name).to(self.device)
        self._client.eval()
        for p in self._client.parameters():
            p.requires_grad_(False)
        return self._client

    def _get_api(self):
        if self._api is not None:
            return self._api
        try:
            from esm.models.esm3 import ESM3
            from esm.models.esmc import ESMC
            from esm.sdk.api import ESMProtein, LogitsConfig
        except ImportError as exc:
            raise RuntimeError(
                "EvolutionaryScale ESM is required. "
                "Install from https://github.com/evolutionaryscale/esm"
            ) from exc
        self._api = (ESM3, ESMC, ESMProtein, LogitsConfig)
        return self._api
