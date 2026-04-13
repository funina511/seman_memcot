"""Pluggable scoring backends for assistant token confidence extraction."""

from __future__ import annotations

import asyncio
import atexit
from dataclasses import dataclass
import math
import os
import threading
from typing import Protocol


MAX_NEG_LOGPROB = -100.0


class ScoringBackend(Protocol):
    """Backend contract for assistant-token scoring."""

    def score_assistant_tokens(
        self,
        *,
        system_prompt: str,
        question: str,
        assistant_text: str,
        assistant_window_size: int = 4096,
        assistant_stride: int | None = None,
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


def _unpack_payload(payload):
    """Normalize backend payloads to (token_ids, offsets, confidences)."""
    if isinstance(payload, dict):
        return payload.get("token_ids", []), payload.get("offsets", []), payload.get("confidences", [])
    if isinstance(payload, (list, tuple)) and len(payload) == 3:
        return payload[0], payload[1], payload[2]
    raise TypeError(
        "Unsupported scoring payload type. Expected dict with token_ids/offsets/confidences "
        "or a 3-tuple/list (token_ids, offsets, confidences), got "
        f"{type(payload).__name__}."
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
        assistant_stride=None,
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
            assistant_stride=assistant_stride,
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
        assistant_stride=None,
    ):
        scorer = getattr(self.model, "score_assistant_tokens", None)
        if callable(scorer):
            scorer_kwargs = dict(
                system_prompt=system_prompt,
                question=question,
                assistant_text=assistant_text,
                assistant_window_size=assistant_window_size,
                tokenizer=self.tokenizer,
            )
            if assistant_stride is not None:
                scorer_kwargs["assistant_stride"] = assistant_stride
            return scorer(**scorer_kwargs)

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
            scorer_kwargs = dict(
                model=self.model,
                tokenizer=self.tokenizer,
                system_prompt=system_prompt,
                question=question,
                assistant_text=assistant_text,
                assistant_window_size=assistant_window_size,
            )
            if assistant_stride is not None:
                scorer_kwargs["assistant_stride"] = assistant_stride
            return module_scorer(**scorer_kwargs)

        model_type = type(self.model).__name__
        raise RuntimeError(
            "sglang backend could not find a supported scoring entrypoint. "
            "Provide either a model object exposing score_assistant_tokens(...) "
            "or an installed sglang package exposing sglang.score_assistant_tokens(...). "
            f"Current model handle type: {model_type}."
        )

    def score_assistant_tokens(
        self,
        *,
        system_prompt,
        question,
        assistant_text,
        assistant_window_size=4096,
        assistant_stride=None,
    ):
        payload_kwargs = dict(
            system_prompt=system_prompt,
            question=question,
            assistant_text=assistant_text,
            assistant_window_size=assistant_window_size,
        )
        if assistant_stride is not None:
            payload_kwargs["assistant_stride"] = assistant_stride
        payload = self._score_payload(**payload_kwargs)
        raw_token_ids, raw_offsets, raw_confidences = _unpack_payload(payload)
        token_ids = list(raw_token_ids)
        offsets = [tuple(offset) for offset in raw_offsets]
        confidences = list(raw_confidences)
        _validate_scoring_output(token_ids, offsets, confidences)
        return token_ids, offsets, confidences


def _coerce_outputs(outputs):
    if isinstance(outputs, dict):
        return [outputs]
    if isinstance(outputs, list):
        return outputs
    return list(outputs)


def _extract_prompt_logprobs(output_obj):
    if not isinstance(output_obj, dict):
        return []
    meta_info = output_obj.get("meta_info", {})
    if not isinstance(meta_info, dict):
        return []
    prompt_logprobs = meta_info.get("input_token_logprobs", [])
    return prompt_logprobs if isinstance(prompt_logprobs, list) else []


def _parse_prompt_logprob_item(item):
    if item is None:
        return None
    if isinstance(item, (int, float)):
        return float(item)
    if isinstance(item, dict):
        for key in ("logprob", "log_prob", "token_logprob", "value"):
            value = item.get(key)
            if isinstance(value, (int, float)):
                return float(value)
        return None
    if isinstance(item, (list, tuple)) and item:
        head = item[0]
        if isinstance(head, (int, float)):
            return float(head)
    return None


def _get_prompt_logprob_item(prompt_logprobs, *, position, full_input_len, logprob_start_len):
    """Fetch one prompt logprob from absolute or truncated SGLang payloads.

    Some SGLang runtimes return ``input_token_logprobs`` aligned to the full
    prompt length, while others return a truncated slice that starts at
    ``logprob_start_len``. This helper keeps the adapter agnostic to either
    layout.
    """
    if not isinstance(prompt_logprobs, list):
        return None

    truncated_len = max(0, full_input_len - logprob_start_len)
    if logprob_start_len >= 0 and len(prompt_logprobs) == truncated_len:
        truncated_index = position - logprob_start_len
        if 0 <= truncated_index < len(prompt_logprobs):
            return prompt_logprobs[truncated_index]

    if 0 <= position < len(prompt_logprobs):
        return prompt_logprobs[position]

    truncated_index = position - logprob_start_len
    if 0 <= truncated_index < len(prompt_logprobs):
        return prompt_logprobs[truncated_index]
    return None


class _AsyncLoopRunner:
    """Run async coroutines on one persistent background event loop."""

    def __init__(self):
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="sglang-loop-runner",
            daemon=True,
        )
        self._closed = False
        self._thread.start()

    def _run_loop(self):
        asyncio.set_event_loop(self._loop)
        self._loop.run_forever()

    def run(self, coroutine):
        if self._closed:
            raise RuntimeError("Async loop runner has been closed.")
        future = asyncio.run_coroutine_threadsafe(coroutine, self._loop)
        return future.result()

    def close(self):
        if self._closed:
            return
        self._closed = True
        if self._loop.is_running():
            try:
                async def _cancel_pending_tasks():
                    current = asyncio.current_task()
                    pending = [
                        task for task in asyncio.all_tasks()
                        if task is not current and not task.done()
                    ]
                    for task in pending:
                        task.cancel()
                    if pending:
                        await asyncio.gather(*pending, return_exceptions=True)

                future = asyncio.run_coroutine_threadsafe(_cancel_pending_tasks(), self._loop)
                future.result(timeout=1.0)
            except Exception:
                pass
        if self._loop.is_running():
            self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=2.0)
        self._loop.close()


class _SGLangEngineAdapter:
    """Model-like adapter that exposes score_assistant_tokens for SGLang."""

    def __init__(
        self,
        *,
        model_name,
        tokenizer,
        dtype="bfloat16",
        trust_remote_code=True,
    ):
        try:
            import sglang as sgl
        except ImportError as exc:
            raise RuntimeError(
                "sglang backend requires the 'sglang' package. "
                "Activate your sglang environment first."
            ) from exc

        tp_size = int(os.environ.get("SGLANG_TP_SIZE", "1"))
        engine_kwargs = {
            "model_path": model_name,
            "tp_size": tp_size,
            "trust_remote_code": bool(trust_remote_code),
            "dtype": dtype,
        }
        mem_fraction_static = os.environ.get("SGLANG_MEM_FRACTION_STATIC")
        if mem_fraction_static:
            try:
                engine_kwargs["mem_fraction_static"] = float(mem_fraction_static)
            except ValueError:
                pass
        chunked_prefill_size = os.environ.get("SGLANG_CHUNKED_PREFILL_SIZE")
        if chunked_prefill_size:
            try:
                engine_kwargs["chunked_prefill_size"] = int(chunked_prefill_size)
            except ValueError:
                pass
        if os.environ.get("SGLANG_DISABLE_RADIX_CACHE", "0") == "1":
            engine_kwargs["disable_radix_cache"] = True
        cuda_graph_max_bs = os.environ.get("SGLANG_CUDA_GRAPH_MAX_BS")
        if cuda_graph_max_bs:
            try:
                engine_kwargs["cuda_graph_max_bs"] = int(cuda_graph_max_bs)
            except ValueError:
                pass

        self._engine = sgl.Engine(**engine_kwargs)
        self._tokenizer = tokenizer
        self._runner = _AsyncLoopRunner()

    def _build_sampling_params(self):
        return {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": 0,
            "skip_special_tokens": False,
        }

    async def _async_generate_with_logprobs(self, *, input_ids, logprob_start_len):
        sampling_params = self._build_sampling_params()

        async_generate = getattr(self._engine, "async_generate", None)
        if callable(async_generate):
            try:
                outputs = await async_generate(
                    input_ids=[input_ids],
                    sampling_params=sampling_params,
                    return_logprob=True,
                    logprob_start_len=logprob_start_len,
                    top_logprobs_num=0,
                )
            except TypeError:
                fallback_sampling = dict(sampling_params)
                fallback_sampling.update(
                    {
                        "return_logprob": True,
                        "logprob_start_len": logprob_start_len,
                        "top_logprobs_num": 0,
                    }
                )
                outputs = await async_generate(
                    input_ids=[input_ids],
                    sampling_params=fallback_sampling,
                )
            return _coerce_outputs(outputs)

        raise RuntimeError(
            "sglang engine does not expose async_generate; unsupported sglang runtime API."
        )

    def _generate_with_logprobs(self, *, input_ids, logprob_start_len):
        sampling_params = self._build_sampling_params()
        generate = getattr(self._engine, "generate", None)
        if callable(generate):
            try:
                return _coerce_outputs(
                    generate(
                        input_ids=[input_ids],
                        sampling_params=sampling_params,
                        return_logprob=True,
                        logprob_start_len=logprob_start_len,
                        top_logprobs_num=0,
                    )
                )
            except TypeError:
                compatible_sampling = dict(sampling_params)
                compatible_sampling.update(
                    {
                        "return_logprob": True,
                        "logprob_start_len": logprob_start_len,
                        "top_logprobs_num": 0,
                    }
                )
                return _coerce_outputs(
                    generate(
                        input_ids=[input_ids],
                        sampling_params=compatible_sampling,
                    )
                )

        async_generate = getattr(self._engine, "async_generate", None)
        if callable(async_generate):
            return self._runner.run(
                self._async_generate_with_logprobs(
                    input_ids=input_ids,
                    logprob_start_len=logprob_start_len,
                )
            )

        raise RuntimeError(
            "sglang engine exposes neither generate nor async_generate; unsupported runtime API."
        )

    def score_assistant_tokens(
        self,
        *,
        system_prompt,
        question,
        assistant_text,
        assistant_window_size=4096,
        assistant_stride=None,
        tokenizer=None,
    ):
        tokenizer = tokenizer or self._tokenizer
        if tokenizer is None:
            raise RuntimeError(
                "sglang scoring requires a tokenizer to build assistant offsets."
            )

        from semantic_aware.scoring import (
            build_scoring_windows,
            tokenize_prompt_and_assistant,
        )

        prefix_ids, assistant_token_ids, assistant_offsets = tokenize_prompt_and_assistant(
            tokenizer,
            system_prompt=system_prompt,
            question=question,
            assistant_text=assistant_text,
        )
        if not assistant_token_ids:
            return [], [], []

        windows = build_scoring_windows(
            prefix_ids=prefix_ids,
            assistant_ids=assistant_token_ids,
            assistant_window_size=assistant_window_size,
            assistant_stride=assistant_stride,
        )

        confidences = []

        # Mirror HF windowed scoring semantics: each pass scores only the fresh tail.
        for window in windows:
            full_input_ids = window.prefix_ids + window.assistant_ids
            fresh_start_position = len(window.prefix_ids) + window.assistant_start_index
            logprob_start_len = max(0, fresh_start_position - 1)
            outputs = self._generate_with_logprobs(
                input_ids=full_input_ids,
                logprob_start_len=logprob_start_len,
            )
            if not outputs:
                raise RuntimeError("sglang scoring returned no outputs for prompt logprobs.")

            prompt_logprobs = _extract_prompt_logprobs(outputs[0])
            fresh_positions = range(
                len(window.prefix_ids) + window.assistant_start_index,
                len(full_input_ids),
            )
            for position in fresh_positions:
                item = _get_prompt_logprob_item(
                    prompt_logprobs,
                    position=position,
                    full_input_len=len(full_input_ids),
                    logprob_start_len=logprob_start_len,
                )
                logp = _parse_prompt_logprob_item(item)
                if logp is None:
                    logp = MAX_NEG_LOGPROB
                confidences.append(math.exp(max(float(logp), MAX_NEG_LOGPROB)))

        # Keep contract strict even when runtime returns sparse/incomplete logprobs.
        if len(confidences) < len(assistant_token_ids):
            confidences.extend(
                [math.exp(MAX_NEG_LOGPROB)] * (len(assistant_token_ids) - len(confidences))
            )
        elif len(confidences) > len(assistant_token_ids):
            confidences = confidences[: len(assistant_token_ids)]

        return assistant_token_ids, assistant_offsets, confidences

    def shutdown(self):
        self._runner.close()
        shutdown = getattr(self._engine, "shutdown", None)
        if callable(shutdown):
            shutdown()


_SGLANG_ADAPTERS = []
_SGLANG_CLEANUP_REGISTERED = False


def _cleanup_sglang_adapters():
    while _SGLANG_ADAPTERS:
        adapter = _SGLANG_ADAPTERS.pop()
        try:
            adapter.shutdown()
        except Exception:
            pass


def _register_sglang_adapter(adapter):
    global _SGLANG_CLEANUP_REGISTERED
    _SGLANG_ADAPTERS.append(adapter)
    if not _SGLANG_CLEANUP_REGISTERED:
        atexit.register(_cleanup_sglang_adapters)
        _SGLANG_CLEANUP_REGISTERED = True


def get_scoring_backend(*, backend_name, model, tokenizer):
    """Build a scoring backend by name."""
    if backend_name == "hf":
        return HFScoringBackend(model=model, tokenizer=tokenizer)
    if backend_name == "sglang":
        return SGLangScoringBackend(model=model, tokenizer=tokenizer)
    raise ValueError(f"Unknown scoring backend: {backend_name}")


def _try_load_tokenizer(model_name, *, trust_remote_code):
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None

    try:
        return AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=bool(trust_remote_code),
        )
    except Exception:
        return None


def init_scoring_backend(
    *,
    backend_name,
    model_name,
    dtype="bfloat16",
    trust_remote_code=True,
):
    """Create a backend and the tokenizer used for protected-token masking.

    - hf: load HF model + tokenizer.
    - sglang: keep a lightweight model handle and avoid HF model loading.
    """
    if backend_name == "hf":
        from semantic_aware.scoring import load_model_and_tokenizer

        model, tokenizer = load_model_and_tokenizer(
            model_name,
            dtype=dtype,
            trust_remote_code=bool(trust_remote_code),
        )
        return get_scoring_backend(
            backend_name=backend_name,
            model=model,
            tokenizer=tokenizer,
        ), tokenizer

    if backend_name == "sglang":
        tokenizer = _try_load_tokenizer(
            model_name,
            trust_remote_code=trust_remote_code,
        )
        if tokenizer is None:
            raise RuntimeError(
                "Failed to load tokenizer for sglang backend. "
                "Tokenizer is required to map assistant token offsets."
            )
        model = _SGLangEngineAdapter(
            model_name=model_name,
            tokenizer=tokenizer,
            dtype=dtype,
            trust_remote_code=bool(trust_remote_code),
        )
        _register_sglang_adapter(model)
        return get_scoring_backend(
            backend_name=backend_name,
            model=model,
            tokenizer=tokenizer,
        ), tokenizer

    raise ValueError(f"Unknown scoring backend: {backend_name}")
