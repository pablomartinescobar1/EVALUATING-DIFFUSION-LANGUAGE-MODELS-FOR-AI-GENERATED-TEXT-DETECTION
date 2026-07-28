"""Diffusion backbones (LLaDA-8B-Base, LLaDA-8B-VRPO a.k.a. "LLaDA 1.5", Dream-LLaDA).

Strategy 1 (score) and Strategy 3 (embed) treat the shifted logits as if they were an
autoregressive next-token distribution. This is a simplification inherited from the
original notebooks, NOT the masked/ELBO estimator the paper text (Section 3.1)
describes for diffusion log-likelihood. It is left unchanged here deliberately:
"fixing" it would change the paper's reported numbers, which is a research decision
for the author to make separately, not a side effect of a refactor. See NOTES.md.

Strategy 2 (pawn_metrics) DOES implement the real diffusion-specific adaptation
described in the paper (Section 3.2): Monte Carlo masking at a fixed, per-model
`mask_ratio` (grid-searched offline -- see scripts/tune_diffusion_masking.py),
averaged over `num_samples` independent masked reconstructions per text. Note this
also differs slightly from the paper's literal "t ~ U[0,1]" description (a fixed,
tuned ratio is used instead of a fresh random ratio per sample) -- also documented
in NOTES.md, also left as-is for the same reason.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig

from aitext.metrics.pawn import METRIC_NAMES, compute_pawn_metrics, sequence_log_likelihood
from aitext.metrics.zero_shot import (
    NO_CANDIDATES,
    YES_CANDIDATES,
    build_cloze_input_ids,
    resolve_single_token_id,
    yes_no_log_odds,
)
from aitext.models.base import ModelWrapper, autocast_context, iter_batches


class DiffusionModel(ModelWrapper):
    def __init__(
        self,
        name: str,
        hf_id: str,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        quantize_4bit: bool = True,
        max_length: int = 512,
        batch_size: int = 4,
        mask_ratio: float = 0.35,
        num_samples: int = 10,
        accepts_2d_attention_mask: bool = True,
    ):
        self.name = name
        self.max_length = max_length
        self.batch_size = batch_size
        self.mask_ratio = mask_ratio
        self.num_samples = num_samples
        self.accepts_2d_attention_mask = accepts_2d_attention_mask
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id, trust_remote_code=True)

        quant_config = BitsAndBytesConfig(load_in_4bit=True) if quantize_4bit else None
        self.model = AutoModel.from_pretrained(
            hf_id, quantization_config=quant_config, trust_remote_code=True, dtype=self.dtype
        ).eval()
        if hasattr(self.model, "tie_weights"):
            self.model.tie_weights()
        self.device = str(next(self.model.parameters()).device)

        self.mask_token_id = (
            self.tokenizer.mask_token_id
            if getattr(self.tokenizer, "mask_token_id", None) is not None
            else self.tokenizer.vocab_size - 1
        )

        self._yes_id = resolve_single_token_id(self.tokenizer, YES_CANDIDATES)
        self._no_id = resolve_single_token_id(self.tokenizer, NO_CANDIDATES)

    def _tokenize(self, batch_texts: list[str]):
        return self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

    def _run_model(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        kwargs = {"input_ids": input_ids}
        if self.accepts_2d_attention_mask:
            kwargs["attention_mask"] = attention_mask
        with torch.no_grad(), autocast_context(self.device, self.dtype):
            return self.model(**kwargs, output_hidden_states=True)

    def score(self, texts: list[str]) -> list[float]:
        scores = []
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} score"):
            inputs = self._tokenize(batch)
            outputs = self._run_model(inputs["input_ids"], inputs["attention_mask"])
            logits = outputs.logits[:, :-1, :]
            labels = inputs["input_ids"][:, 1:]
            attention_mask = inputs["attention_mask"][:, 1:]
            scores.extend(sequence_log_likelihood(logits, labels, attention_mask))
        return scores

    def pawn_metrics(self, texts: list[str]) -> dict[str, list[float]]:
        columns: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} pawn"):
            batch_values = self._diffusion_pawn_batch(batch)
            for name, vals in zip(METRIC_NAMES, batch_values):
                columns[name].extend(vals)
        return columns

    def _diffusion_pawn_batch(self, batch_texts: list[str]) -> list[list[float]]:
        """Monte Carlo masked reconstruction: mask a random `mask_ratio` fraction of
        non-special tokens, ask the model to reconstruct them, score only those
        positions, and average over `num_samples` independent draws."""
        accumulated = [[] for _ in METRIC_NAMES]

        for _ in range(self.num_samples):
            inputs = self._tokenize(batch_texts)
            input_ids = inputs["input_ids"]
            attention_mask = inputs["attention_mask"]
            batch_size, seq_len = input_ids.shape

            mask_probs = (
                torch.full((batch_size, seq_len), self.mask_ratio, device=input_ids.device)
                * attention_mask.float()
            )
            for special_id in self.tokenizer.all_special_ids:
                mask_probs[input_ids == special_id] = 0.0
            mask_positions = torch.bernoulli(mask_probs).bool()

            # Guarantee at least one masked token per sequence.
            for row in range(batch_size):
                if not mask_positions[row].any():
                    valid = attention_mask[row].nonzero(as_tuple=True)[0]
                    if len(valid) > 0:
                        pick = valid[torch.randint(0, len(valid), (1,))]
                        mask_positions[row, pick] = True

            corrupted_ids = input_ids.clone()
            corrupted_ids[mask_positions] = self.mask_token_id

            outputs = self._run_model(corrupted_ids, attention_mask)
            values = compute_pawn_metrics(outputs.logits, input_ids, mask_positions)
            for i, vals in enumerate(values):
                accumulated[i].append(vals)

        return [np.mean(np.array(sample_values), axis=0).tolist() for sample_values in accumulated]

    def embed(self, texts: list[str]) -> np.ndarray:
        """Mean pooling over non-pad tokens (paper Section 3.3) -- diffusion models
        have no [CLS]-equivalent, so the pooled representation captures full-sequence
        bidirectional context instead of a single summary token."""
        all_embeddings = []
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} embed"):
            inputs = self._tokenize(batch)
            outputs = self._run_model(inputs["input_ids"], inputs["attention_mask"])
            hidden_states = outputs.hidden_states[-1]
            mask = inputs["attention_mask"].unsqueeze(-1).float()
            summed = (hidden_states * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1)
            all_embeddings.extend((summed / counts).float().cpu().numpy())
        return np.array(all_embeddings)

    def zero_shot_score(self, texts: list[str]) -> list[float]:
        """Single deterministic forward pass at one fixed mask position -- unlike
        pawn_metrics's Monte Carlo averaging over randomly chosen positions, there is
        nothing stochastic to average over here (the answer slot is the same position
        every time)."""
        scores = []
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} zero_shot"):
            ids_list = [
                build_cloze_input_ids(self.tokenizer, text, self.mask_token_id, self.max_length)
                for text in batch
            ]
            inputs = self.tokenizer.pad({"input_ids": ids_list}, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self._run_model(inputs["input_ids"], inputs["attention_mask"])
            rows, positions = (inputs["input_ids"] == self.mask_token_id).nonzero(as_tuple=True)
            logits_at_answer = outputs.logits[rows, positions, :]
            scores.extend(yes_no_log_odds(logits_at_answer, self._yes_id, self._no_id))
        return scores
