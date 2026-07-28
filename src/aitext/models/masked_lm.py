"""Masked-language-model backbones (BERT, RoBERTa).

Unlike autoregressive models, MLM logits already cover every position in one forward
pass, so score()/pawn_metrics() evaluate directly against `input_ids` (no shift).
"""
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from aitext.metrics.pawn import METRIC_NAMES, compute_pawn_metrics, sequence_log_likelihood
from aitext.metrics.zero_shot import (
    NO_CANDIDATES,
    YES_CANDIDATES,
    build_cloze_input_ids,
    resolve_single_token_id,
    yes_no_log_odds,
)
from aitext.models.base import ModelWrapper, autocast_context, iter_batches


class MaskedLMModel(ModelWrapper):
    def __init__(
        self,
        name: str,
        hf_id: str,
        device: str = "cuda",
        dtype: torch.dtype | None = None,
        max_length: int = 512,
        batch_size: int = 4,
    ):
        self.name = name
        self.max_length = max_length
        self.batch_size = batch_size
        self.device = device if torch.cuda.is_available() else "cpu"
        self.dtype = dtype or (torch.bfloat16 if self.device == "cuda" else torch.float32)

        self.tokenizer = AutoTokenizer.from_pretrained(hf_id)
        self.model = AutoModelForMaskedLM.from_pretrained(hf_id, torch_dtype=self.dtype).to(self.device).eval()

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

    def _run_model(self, inputs):
        with torch.no_grad(), autocast_context(self.device, self.dtype):
            return self.model(**inputs, output_hidden_states=True)

    def score(self, texts: list[str]) -> list[float]:
        scores = []
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} score"):
            inputs = self._tokenize(batch)
            outputs = self._run_model(inputs)
            scores.extend(
                sequence_log_likelihood(outputs.logits, inputs["input_ids"], inputs["attention_mask"])
            )
        return scores

    def pawn_metrics(self, texts: list[str]) -> dict[str, list[float]]:
        columns: dict[str, list[float]] = {name: [] for name in METRIC_NAMES}
        vocab_size = self.model.config.vocab_size
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} pawn"):
            inputs = self._tokenize(batch)
            outputs = self._run_model(inputs)
            input_ids = inputs["input_ids"].clamp(0, vocab_size - 1)
            values = compute_pawn_metrics(outputs.logits, input_ids, inputs["attention_mask"])
            for name, vals in zip(METRIC_NAMES, values):
                columns[name].extend(vals)
        return columns

    def embed(self, texts: list[str]) -> np.ndarray:
        """[CLS] token (position 0) -- paper Section 3.3."""
        all_embeddings = []
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} embed"):
            inputs = self._tokenize(batch)
            outputs = self._run_model(inputs)
            hidden_states = outputs.hidden_states[-1]
            all_embeddings.extend(hidden_states[:, 0, :].float().cpu().numpy())
        return np.array(all_embeddings)

    def zero_shot_score(self, texts: list[str]) -> list[float]:
        scores = []
        mask_token_id = self.tokenizer.mask_token_id
        for batch in iter_batches(texts, self.batch_size, desc=f"{self.name} zero_shot"):
            ids_list = [
                build_cloze_input_ids(self.tokenizer, text, mask_token_id, self.max_length) for text in batch
            ]
            inputs = self.tokenizer.pad({"input_ids": ids_list}, return_tensors="pt", padding=True).to(
                self.device
            )
            outputs = self._run_model(inputs)
            rows, positions = (inputs["input_ids"] == mask_token_id).nonzero(as_tuple=True)
            logits_at_answer = outputs.logits[rows, positions, :]
            scores.extend(yes_no_log_odds(logits_at_answer, self._yes_id, self._no_id))
        return scores
