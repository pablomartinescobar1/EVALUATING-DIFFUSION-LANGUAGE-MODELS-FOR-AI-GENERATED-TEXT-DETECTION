"""Autoregressive backbones (GPT-2, LLaMA, GPT-3-proxy).

Fixes a bug present in the original notebooks: `batch_autoregressive_scores` called
the model with `labels=input_ids` and read back `outputs.loss`, which Hugging Face
already reduces (mean) across the *entire batch* into one scalar -- so every text in
a batch of 4 received an identical score, destroying the per-example signal for
GPT-2/LLaMA/GPT-3-proxy specifically. Here `score()` computes the same Eq. 1 quantity
per example via `aitext.metrics.pawn.sequence_log_likelihood`, exactly like the
masked_lm/diffusion wrappers already did correctly. See NOTES.md.
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from aitext.metrics.pawn import METRIC_NAMES, compute_pawn_metrics, sequence_log_likelihood
from aitext.models.base import ModelWrapper, autocast_context, iter_batches


class AutoregressiveModel(ModelWrapper):
    def __init__(
        self,
        name: str,
        hf_id: str,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        quantize_4bit: bool = False,
        max_length: int = 512,
        batch_size: int = 4,
    ):
        self.name = name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if quantize_4bit:
            quant_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
            self.model = AutoModelForCausalLM.from_pretrained(
                hf_id, quantization_config=quant_config, dtype=self.dtype
            ).eval()
            self.device = str(next(self.model.parameters()).device)
        else:
            self.model = AutoModelForCausalLM.from_pretrained(hf_id, dtype=self.dtype).to(self.device).eval()

        if self.tokenizer.pad_token_id >= self.model.config.vocab_size:
            self.model.resize_token_embeddings(len(self.tokenizer))

    def _tokenize(self, batch_texts: list[str]):
        return self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self.max_length,
        ).to(self.device)

    def _run_model(self, inputs):
        with torch.no_grad(), autocast_context(self.device, self.dtype):
            return self.model(**inputs, output_hidden_states=True)

    def score(self, texts: list[str]) -> list[float]:
        scores = []
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} score"):
            inputs = self._tokenize(batch)
            outputs = self._run_model(inputs)
            logits = outputs.logits[:, :-1, :]
            labels = inputs["input_ids"][:, 1:]
            attention_mask = inputs["attention_mask"][:, 1:]
            scores.extend(sequence_log_likelihood(logits, labels, attention_mask))
        return scores

    def pawn_metrics(self, texts: list[str]) -> dict[str, list[float]]:
        columns: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} pawn"):
            inputs = self._tokenize(batch)
            outputs = self._run_model(inputs)
            logits = outputs.logits[:, :-1, :]
            labels = inputs["input_ids"][:, 1:]
            attention_mask = inputs["attention_mask"][:, 1:]
            values = compute_pawn_metrics(logits, labels, attention_mask)
            for name, vals in zip(METRIC_NAMES, values):
                columns[name].extend(vals)
        return columns

    def embed(self, texts: list[str]) -> np.ndarray:
        """Last non-pad token's hidden state -- causal attention means it's the
        only position that has attended to the whole sequence (paper Section 3.3)."""
        all_embeddings = []
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} embed"):
            inputs = self._tokenize(batch)
            outputs = self._run_model(inputs)
            hidden_states = outputs.hidden_states[-1]
            seq_lengths = inputs["attention_mask"].sum(dim=1) - 1
            for row, seq_len in enumerate(seq_lengths):
                all_embeddings.append(hidden_states[row, seq_len, :].float().cpu().numpy())
        return np.array(all_embeddings)
