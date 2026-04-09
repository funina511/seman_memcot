"""AdaptiveStep boundary selection helpers."""


def is_valid_segment_text(text, min_meaningful_chars=3):
    """Return whether text looks substantial enough to stand alone."""
    stripped = text.strip()
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

    # Preserve short word-like fragments while still rejecting pure punctuation shards.
    return len(spans) == 1 and len(spans[0]) >= 2 and any(
        char.isalpha() for char in spans[0]
    )


def pick_boundaries(confidences, cuttable_mask, tau, min_step_tokens):
    """Pick low-confidence boundary tokens, keeping local minima only."""
    if len(confidences) != len(cuttable_mask):
        raise ValueError("confidences and cuttable_mask must have the same length.")

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
    last_boundary = -10**9
    for cluster in clusters:
        best_idx = min(cluster, key=lambda i: confidences[i])
        step_size = best_idx + 1 if not boundaries else best_idx - last_boundary
        if step_size >= min_step_tokens:
            boundaries.append(best_idx)
            last_boundary = best_idx
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
