"""Protected token handling for safe boundary selection."""

PROTECTED_STRINGS = [
    "<｜begin▁of▁sentence｜>",
    "<｜User｜>",
    "<｜Assistant｜>",
    "<think>",
    "</think>",
    "<|im_start|>",
    "<|im_end|>",
    "<|splitter|>",
    "<|continue|>",
    "<|recover|>",
    "<|begin_of_thought|>",
    "<|end_of_thought|>",
    "<|begin_of_solution|>",
    "<|end_of_solution|>",
]


def find_protected_spans(text, protected_strings=None):
    """Return sorted spans for protected strings found in text."""
    if protected_strings is None:
        protected_strings = PROTECTED_STRINGS
    spans = []
    for token in protected_strings:
        start = 0
        while True:
            idx = text.find(token, start)
            if idx == -1:
                break
            spans.append((idx, idx + len(token), token))
            start = idx + len(token)
    spans.sort()
    return spans


def _overlaps(span_a, span_b):
    a_start, a_end = span_a
    b_start, b_end = span_b
    return not (a_end <= b_start or a_start >= b_end)


def build_cuttable_mask(token_ids, offsets, special_ids, protected_spans):
    """Mark which assistant tokens are safe to use as boundaries."""
    if len(token_ids) != len(offsets):
        raise ValueError("token_ids and offsets must have the same length.")

    mask = []
    special_ids = set(special_ids)
    for token_id, offset in zip(token_ids, offsets):
        start, end = offset
        cuttable = True
        if token_id in special_ids:
            cuttable = False
        if start == end:
            cuttable = False
        if cuttable:
            for protected_start, protected_end, _ in protected_spans:
                if _overlaps((start, end), (protected_start, protected_end)):
                    cuttable = False
                    break
        mask.append(cuttable)
    return mask
