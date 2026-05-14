from __future__ import annotations

import pandas as pd


def compute_qc_summary(counts: pd.DataFrame) -> pd.DataFrame:
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    summaries = []

    for sample in sample_cols:
        values = pd.to_numeric(counts[sample], errors="coerce").fillna(0)
        summaries.append(
            {
                "sample_id": sample,
                "library_size": int(values.sum()),
                "detected_probes": int((values > 0).sum()),
                "positive_control_sum": int(
                    values[counts["CodeClass"].str.lower().eq("positive")].sum()
                ),
                "negative_control_mean": float(
                    values[counts["CodeClass"].str.lower().eq("negative")].mean()
                ),
            }
        )

    return pd.DataFrame(summaries)

