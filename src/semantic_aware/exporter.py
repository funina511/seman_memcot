"""Output record formatting helpers."""


def split_double_newline(text):
    """Split text on double newlines while dropping empty chunks."""
    if not text:
        return []
    return [chunk.strip() for chunk in text.split("\n\n") if chunk.strip()]


def build_output_record(source_idx, system_prompt, question, gt_output, thoughts_list):
    """Build one LightThinker training record."""
    return {
        "source_idx": source_idx,
        "system_prompt": system_prompt,
        "question": question,
        "gt_output": gt_output,
        "question_list": split_double_newline(question),
        "thoughts_list": thoughts_list,
        "system_list": split_double_newline(system_prompt),
        "add_eos": True,
    }


def build_output_record_from_reference(reference_record, thoughts_list, source_idx):
    """Inherit all fields from reference record and only replace thoughts_list."""
    output_record = dict(reference_record)
    output_record["thoughts_list"] = thoughts_list
    output_record.setdefault("source_idx", source_idx)
    return output_record
