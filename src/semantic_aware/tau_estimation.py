"""Tau estimation helpers."""


def compute_quantile(values, quantile):
    """Compute a simple inclusive quantile for sorted probabilities."""
    if not values:
        raise ValueError("Cannot compute quantile on empty values.")
    if not 0 <= quantile <= 1:
        raise ValueError("Quantile must be in [0, 1].")
    values = sorted(values)
    index = int(round((len(values) - 1) * quantile))
    return values[index]


def estimate_tau_from_records(sample_records, quantiles):
    """Estimate tau candidates from sampled confidence records."""
    usable = []
    for record in sample_records:
        confidences = record["confidences"]
        cuttable_mask = record["cuttable_mask"]
        if len(confidences) != len(cuttable_mask):
            raise ValueError("confidences and cuttable_mask must have the same length.")
        for confidence, is_cuttable in zip(confidences, cuttable_mask):
            if is_cuttable:
                usable.append(confidence)
    if not usable:
        raise ValueError("No cuttable confidences found in sampled records.")
    return {
        f"q_{quantile:.4f}": compute_quantile(usable, quantile)
        for quantile in quantiles
    }
