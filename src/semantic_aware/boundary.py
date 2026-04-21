"""AdaptiveStep boundary selection helpers."""

import re


_FORMULA_FRAGMENT_RE = re.compile(r"^\s*[A-Za-z0-9]+(?:\s*[-+*/=<>^%:]+\s*[A-Za-z0-9]+)+\s*$")


def _is_cjk_char(char):
    return "\u4e00" <= char <= "\u9fff"


def _is_word_connector(char):
    return char in {"_", "-"}


def _has_word_char_left(text, index):
    while index >= 0 and _is_word_connector(text[index]):
        index -= 1
    return index >= 0 and text[index].isalnum()


def _has_word_char_right(text, index):
    while index < len(text) and _is_word_connector(text[index]):
        index += 1
    return index < len(text) and text[index].isalnum()


def _is_inside_word(text, cut_char):
    if cut_char <= 0 or cut_char >= len(text):
        return False
    left = text[cut_char - 1]
    right = text[cut_char]

    # Allow cuts between two CJK characters.
    left_is_cjk = _is_cjk_char(left)
    right_is_cjk = _is_cjk_char(right)
    if left_is_cjk and right_is_cjk:
        return False

    if left.isalnum() and right.isalnum():
        return True

    if _is_word_connector(left) or _is_word_connector(right):
        return _has_word_char_left(text, cut_char - 1) and _has_word_char_right(text, cut_char)

    return False


def is_valid_segment_text(text, min_meaningful_chars=3):
    """Return whether text looks substantial enough to stand alone."""
    stripped = text.strip()
    if not stripped:
        return False

    spans = []
    current = []
    for char in stripped:
        if char.isalnum():
            current.append(char)
            continue
        if current:
            spans.append("".join(current))
            current = []
    if current:
        spans.append("".join(current))

    if not spans:
        return False
    if max(len(span) for span in spans) >= min_meaningful_chars:
        return True

    # Preserve short word-like or numeric fragments while still rejecting pure punctuation shards.
    if len(spans) == 1 and len(spans[0]) >= 2 and spans[0].isalnum():
        return True

    # Preserve compact formula-like fragments such as x+y=2, n->n+1, and a/b.
    return bool(_FORMULA_FRAGMENT_RE.fullmatch(stripped))


def _relocate_boundary_index(
    *,
    candidate_index,
    cuttable_mask,
    min_step_tokens,
    last_boundary,
    text,
    offsets,
    shift_window,
):
    """Find a nearby valid boundary when the current candidate falls inside a word."""
    if text is None or offsets is None:
        return None

    search_limit = max(0, int(shift_window))

    def is_usable(index):
        if index < 0 or index >= len(cuttable_mask):
            return False
        if not cuttable_mask[index]:
            return False
        step_size = index + 1 if last_boundary < 0 else index - last_boundary
        if step_size < min_step_tokens:
            return False
        _, end_char = offsets[index]
        return not _is_inside_word(text, end_char)

    if is_usable(candidate_index):
        return candidate_index

    for distance in range(1, search_limit + 1):
        right_index = candidate_index + distance
        if is_usable(right_index):
            return right_index
        left_index = candidate_index - distance
        if is_usable(left_index):
            return left_index

    return None


def pick_boundaries(
    confidences,
    cuttable_mask,
    tau,
    min_step_tokens,
    text=None,
    offsets=None,
    word_relocation_window=3,
    max_boundary_shift_tokens=None,
):
    """Pick low-confidence boundary tokens, keeping local minima only.

    When *text* and *offsets* are provided, the function also avoids selecting
    boundaries that would fall inside a word. When a low-confidence candidate
    lands inside a word, the selector can shift to a nearby cuttable boundary
    within *word_relocation_window* tokens instead of consuming the step budget
    with an unresolved cut.
    """
    if len(confidences) != len(cuttable_mask):
        raise ValueError("confidences and cuttable_mask must have the same length.")

    # Keep the newer interface readable while still accepting the audit-plan
    # naming used by callers/tests that talk about "max boundary shift".
    if max_boundary_shift_tokens is not None:
        word_relocation_window = max_boundary_shift_tokens

    low_indices = [
        idx
        for idx, (confidence, is_cuttable) in enumerate(zip(confidences, cuttable_mask))
        if is_cuttable and confidence < tau
    ]
    if not low_indices:
        return []

    clusters = []
    current = [low_indices[0]]
    for idx in low_indices[1:]:
        if idx == current[-1] + 1:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)

    boundaries = []
    last_boundary = -1
    for cluster in clusters:
        candidates = sorted(cluster, key=lambda i: confidences[i])
        chosen = None
        for idx in candidates:
            step_size = idx + 1 if not boundaries else idx - last_boundary
            if step_size < min_step_tokens:
                continue
            if text is not None and offsets is not None:
                _, end_char = offsets[idx]
                if _is_inside_word(text, end_char):
                    continue
            chosen = idx
            break

        if chosen is None and text is not None and offsets is not None:
            for idx in candidates:
                relocated = _relocate_boundary_index(
                    candidate_index=idx,
                    cuttable_mask=cuttable_mask,
                    min_step_tokens=min_step_tokens,
                    last_boundary=last_boundary,
                    text=text,
                    offsets=offsets,
                    shift_window=word_relocation_window,
                )
                if relocated is not None:
                    chosen = relocated
                    break

        if chosen is not None:
            boundaries.append(chosen)
            last_boundary = chosen
    return boundaries


def _validate_boundaries(boundaries, offsets):
    if any(boundary < 0 or boundary >= len(offsets) for boundary in boundaries):
        raise ValueError("boundaries must be within offsets range.")
    if boundaries != sorted(boundaries):
        raise ValueError("boundaries must be sorted.")
    if len(boundaries) != len(set(boundaries)):
        raise ValueError("boundaries must be unique.")


def split_text_by_token_boundaries(text, offsets, boundaries, min_step_chars):
    """Split after each boundary token, merging or dropping tiny/invalid fragments."""
    if not text:
        return []
    if not boundaries:
        return [text] if is_valid_segment_text(text) else []

    _validate_boundaries(boundaries, offsets)

    raw_segments = []
    start_char = 0
    for boundary in boundaries:
        _, end_char = offsets[boundary]
        if _is_inside_word(text, end_char):
            continue
        if end_char - start_char < min_step_chars:
            continue
        piece = text[start_char:end_char]
        if piece.strip():
            raw_segments.append(piece)
        start_char = end_char
    tail = text[start_char:]
    if tail.strip() and len(tail) < min_step_chars:
        if raw_segments:
            raw_segments[-1] += tail
        else:
            raw_segments.append(tail)
    elif tail.strip():
        raw_segments.append(tail)

    def should_merge_short_alpha_fragment(piece, *, has_neighbor):
        stripped = piece.strip()
        return has_neighbor and stripped.isalpha() and len(stripped) < 3

    segments = []
    pending_prefix = ""
    for piece in raw_segments:
        piece = pending_prefix + piece
        pending_prefix = ""
        if should_merge_short_alpha_fragment(piece, has_neighbor=bool(segments) or len(raw_segments) > 1):
            if segments:
                segments[-1] += piece
            else:
                pending_prefix = piece
            continue
        if is_valid_segment_text(piece) or segments:
            if is_valid_segment_text(piece):
                segments.append(piece)
            else:
                segments[-1] += piece
            continue

        # Hold onto an invalid leading fragment so it can attach to the next real segment.
        pending_prefix = piece

    if pending_prefix:
        if segments:
            segments[-1] += pending_prefix

    return segments
