"""Model scoring helpers for AdaptiveStep token confidence extraction."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScoringWindow:
    """One teacher-forcing pass over the full prefix and an assistant slice."""

    prefix_ids: list[int]
    assistant_ids: list[int]
    # Local offset where this pass should start emitting fresh-token scores.
    assistant_start_index: int


def parse_dtype(dtype_name):
    """Resolve torch dtype by name."""
    import torch

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    return mapping[dtype_name]


def load_model_and_tokenizer(model_name, dtype="bfloat16", trust_remote_code=True):
    """Load a Hugging Face causal LM and tokenizer."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    torch_dtype = parse_dtype(dtype)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=bool(trust_remote_code),
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch_dtype,
        trust_remote_code=bool(trust_remote_code),
        device_map="auto",
    )
    model.eval()
    return model, tokenizer


def render_prompt_and_full_text(tokenizer, system_prompt, question, assistant_text):
    """Render chat-formatted prefix and full text strings."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": question})
    prefix_messages = list(messages)
    full_messages = list(messages)
    prefix_messages.append({"role": "assistant", "content": ""})
    full_messages.append({"role": "assistant", "content": assistant_text})

    prefix_text = tokenizer.apply_chat_template(
        prefix_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    full_text = tokenizer.apply_chat_template(
        full_messages,
        tokenize=False,
        add_generation_prompt=False,
    )
    return prefix_text, full_text


def build_scoring_windows(prefix_ids, assistant_ids, assistant_window_size):
    """Build bounded sliding windows over assistant tokens while keeping the prefix."""
    if assistant_window_size <= 0:
        raise ValueError("assistant_window_size must be positive.")

    if not assistant_ids:
        return []

    if len(assistant_ids) <= assistant_window_size:
        return [
            ScoringWindow(
                prefix_ids=list(prefix_ids),
                assistant_ids=list(assistant_ids),
                assistant_start_index=0,
            )
        ]

    windows = []
    initial_window = assistant_ids[:assistant_window_size]
    windows.append(
        ScoringWindow(
            prefix_ids=list(prefix_ids),
            assistant_ids=list(initial_window),
            assistant_start_index=0,
        )
    )

    for end in range(assistant_window_size + 1, len(assistant_ids) + 1):
        start = end - assistant_window_size
        windows.append(
            ScoringWindow(
                prefix_ids=list(prefix_ids),
                assistant_ids=list(assistant_ids[start:end]),
                assistant_start_index=assistant_window_size - 1,
            )
        )
    return windows


def tokenize_assistant_in_context(tokenizer, prefix_text, full_text):
    """Tokenize full text and extract assistant token metadata in context."""
    try:
        encoded = tokenizer(
            full_text,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
    except (NotImplementedError, TypeError, ValueError) as exc:
        raise ValueError(
            "Tokenizer must support offset mapping for scoring. "
            "Use a fast tokenizer or one that returns offset_mapping."
        ) from exc

    full_input_ids = encoded["input_ids"]
    full_offsets = encoded.get("offset_mapping")
    if full_offsets is None:
        raise ValueError(
            "Tokenizer must support offset mapping for scoring. "
            "Use a fast tokenizer or one that returns offset_mapping."
        )
    assistant_start = len(prefix_text)

    assistant_token_ids = []
    assistant_offsets = []
    assistant_positions = []
    for position, (token_id, (start, end)) in enumerate(zip(full_input_ids, full_offsets)):
        if end <= assistant_start:
            continue
        if start < assistant_start:
            relative_offset = (0, end - assistant_start)
        else:
            relative_offset = (start - assistant_start, end - assistant_start)
        assistant_token_ids.append(token_id)
        assistant_offsets.append(relative_offset)
        assistant_positions.append(position)
    return full_input_ids, assistant_token_ids, assistant_offsets, assistant_positions


def tokenize_prompt_and_assistant(tokenizer, system_prompt, question, assistant_text):
    """Render chat text once and return prefix ids plus assistant ids/offsets."""
    prefix_text, full_text = render_prompt_and_full_text(
        tokenizer,
        system_prompt=system_prompt,
        question=question,
        assistant_text=assistant_text,
    )
    full_input_ids, assistant_token_ids, assistant_offsets, assistant_positions = (
        tokenize_assistant_in_context(tokenizer, prefix_text, full_text)
    )
    prefix_cutoff = assistant_positions[0] if assistant_positions else len(full_input_ids)
    return (
        full_input_ids[:prefix_cutoff],
        assistant_token_ids,
        assistant_offsets,
    )


def resolve_model_input_device(model):
    """Pick the embedding/input device instead of assuming a single model.device."""
    get_input_embeddings = getattr(model, "get_input_embeddings", None)
    if callable(get_input_embeddings):
        embeddings = get_input_embeddings()
        weight = getattr(embeddings, "weight", None)
        device = getattr(weight, "device", None)
        if device is not None:
            return device

    parameters = getattr(model, "parameters", None)
    if callable(parameters):
        try:
            parameter = next(parameters())
        except (StopIteration, TypeError):
            parameter = None
        if parameter is not None and getattr(parameter, "device", None) is not None:
            return parameter.device

    buffers = getattr(model, "buffers", None)
    if callable(buffers):
        try:
            buffer = next(buffers())
        except (StopIteration, TypeError):
            buffer = None
        if buffer is not None and getattr(buffer, "device", None) is not None:
            return buffer.device

    device = getattr(model, "device", None)
    if device is not None:
        return device

    raise ValueError("Unable to determine an input device for the model.")


def compute_token_confidences(model, full_input_ids, assistant_positions):
    """Compute token probabilities for assistant tokens under teacher forcing."""
    if not full_input_ids:
        return []

    import torch

    input_ids = torch.tensor(
        [full_input_ids],
        device=resolve_model_input_device(model),
    )
    with torch.no_grad():
        logits = model(input_ids=input_ids).logits
        log_probs = logits.log_softmax(dim=-1)

    confidences = []
    for position in assistant_positions:
        if position == 0:
            continue
        gold_token_id = full_input_ids[position]
        token_log_prob = log_probs[0, position - 1, gold_token_id].item()
        confidences.append(math.exp(token_log_prob))
    return confidences


def compute_token_confidences_windowed(
    model,
    prefix_ids,
    assistant_ids,
    assistant_window_size=4096,
):
    """Score only the fresh assistant tail while keeping a bounded recent history."""
    confidences = []
    for window in build_scoring_windows(
        prefix_ids=prefix_ids,
        assistant_ids=assistant_ids,
        assistant_window_size=assistant_window_size,
    ):
        full_input_ids = window.prefix_ids + window.assistant_ids
        # Each overlapped pass keeps only recent assistant context, but scores the new tail.
        assistant_positions = list(
            range(
                len(window.prefix_ids) + window.assistant_start_index,
                len(full_input_ids),
            )
        )
        confidences.extend(
            compute_token_confidences(model, full_input_ids, assistant_positions)
        )
    return confidences


def score_assistant_tokens(
    model=None,
    tokenizer=None,
    *,
    backend=None,
    system_prompt,
    question,
    assistant_text,
    assistant_window_size=4096,
):
    """Return assistant token ids, offsets, and confidences.

    ``backend`` is the preferred entrypoint. ``model`` and ``tokenizer`` are still
    accepted so existing HF callers can transition without changing call sites all at once.
    """
    if assistant_window_size <= 0:
        raise ValueError("assistant_window_size must be positive.")

    if backend is None:
        if model is None or tokenizer is None:
            raise TypeError(
                "score_assistant_tokens requires either a backend or both model and tokenizer."
            )
        from semantic_aware.scoring_backends import HFScoringBackend

        backend = HFScoringBackend(model=model, tokenizer=tokenizer)

    return backend.score_assistant_tokens(
        system_prompt=system_prompt,
        question=question,
        assistant_text=assistant_text,
        assistant_window_size=assistant_window_size,
    )
