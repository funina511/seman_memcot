"""Pluggable scoring backends for assistant token confidence extraction."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class ScoringBackend(Protocol):
    """Backend contract for assistant-token scoring."""

    def score_assistant_tokens(
        self,
        *,
        system_prompt: str,
        question: str,
        assistant_text: str,
        assistant_window_size: int = 4096,
    ) -> tuple[list[int], list[tuple[int, int]], list[float]]:
        """Return assistant token ids, offsets, and confidences."""


def _validate_scoring_output(
    token_ids,
    offsets,
    confidences,
):
    if len(token_ids) != len(confidences):
        raise ValueError(
            "Assistant token and confidence lengths do not align: "
            f"{len(token_ids)} tokens vs {len(confidences)} confidences."
        )
    if len(token_ids) != len(offsets):
        raise ValueError(
            "Assistant token and offset lengths do not align: "
            f"{len(token_ids)} tokens vs {len(offsets)} offsets."
        )


@dataclass(frozen=True)
class HFScoringBackend:
    """Reference Hugging Face backend using the existing sliding-window scorer."""

    model: object
    tokenizer: object

    def score_assistant_tokens(
        self,
        *,
        system_prompt,
        question,
        assistant_text,
        assistant_window_size=4096,
    ):
        from semantic_aware.scoring import (
            compute_token_confidences_windowed,
            tokenize_prompt_and_assistant,
        )

        prefix_ids, assistant_token_ids, assistant_offsets = tokenize_prompt_and_assistant(
            self.tokenizer,
            system_prompt=system_prompt,
            question=question,
            assistant_text=assistant_text,
        )
        confidences = compute_token_confidences_windowed(
            self.model,
            prefix_ids=prefix_ids,
            assistant_ids=assistant_token_ids,
            assistant_window_size=assistant_window_size,
        )
        _validate_scoring_output(
            assistant_token_ids,
            assistant_offsets,
            confidences,
        )
        return assistant_token_ids, assistant_offsets, confidences


@dataclass
class SGLangScoringBackend:
    """Optional SGLang-backed scorer.

    The backend accepts either a client-like object in ``model`` exposing
    ``score_assistant_tokens(...)`` or, if available, a module-level
    ``sglang.score_assistant_tokens(...)`` helper.
    """

    model: object
    tokenizer: object | None = None

    def _score_payload(
        self,
        *,
        system_prompt,
        question,
        assistant_text,
        assistant_window_size,
    ):
        scorer = getattr(self.model, "score_assistant_tokens", None)
        if callable(scorer):
            return scorer(
                system_prompt=system_prompt,
                question=question,
                assistant_text=assistant_text,
                assistant_window_size=assistant_window_size,
                tokenizer=self.tokenizer,
            )

        # Prefer an attached client scorer when available; only import sglang for
        # the module-level fallback path so local stubs stay usable in tests.
        try:
            import sglang
        except ImportError as exc:
            raise RuntimeError(
                "sglang backend requires the 'sglang' package to be installed."
            ) from exc

        module_scorer = getattr(sglang, "score_assistant_tokens", None)
        if callable(module_scorer):
            return module_scorer(
                model=self.model,
                tokenizer=self.tokenizer,
                system_prompt=system_prompt,
                question=question,
                assistant_text=assistant_text,
                assistant_window_size=assistant_window_size,
            )

        raise RuntimeError(
            "sglang backend could not find a supported scoring entrypoint."
        )

    def score_assistant_tokens(
        self,
        *,
        system_prompt,
        question,
        assistant_text,
        assistant_window_size=4096,
    ):
        payload = self._score_payload(
            system_prompt=system_prompt,
            question=question,
            assistant_text=assistant_text,
            assistant_window_size=assistant_window_size,
        )
        token_ids = list(payload["token_ids"])
        offsets = [tuple(offset) for offset in payload["offsets"]]
        confidences = list(payload["confidences"])
        _validate_scoring_output(token_ids, offsets, confidences)
        return token_ids, offsets, confidences


def get_scoring_backend(*, backend_name, model, tokenizer):
    """Build a scoring backend by name."""
    if backend_name == "hf":
        return HFScoringBackend(model=model, tokenizer=tokenizer)
    if backend_name == "sglang":
        return SGLangScoringBackend(model=model, tokenizer=tokenizer)
    raise ValueError(f"Unknown scoring backend: {backend_name}")
