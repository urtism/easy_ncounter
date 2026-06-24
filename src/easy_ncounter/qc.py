from __future__ import annotations

import math

import pandas as pd


DEFAULT_QC_PARAMS = {
    "background_sd_multiplier": 2.0,
    "low_library_fraction": 0.5,
    "low_detected_fraction": 0.5,
    "low_positive_fraction": 0.5,
    "negative_control_mad_multiplier": 3.0,
    "min_fov_counted_fraction": 0.75,
    "binding_density_min": 0.05,
    "binding_density_max": 2.25,
    "fail_library_fraction": 0.25,
    "fail_detected_fraction": 0.25,
    "filter_min_count": 10,
    "filter_min_samples": 2,
    "filter_min_group_samples": 1,
    "qc_mode": "easy_strict",
    "bruker_hk_geomean_min": 100.0,
    "positive_control_linearity_min": 0.95,
    "strict_exclusion": False,
}


def compute_qc_summary(
    counts: pd.DataFrame,
    params: dict | None = None,
    rcc_metrics: pd.DataFrame | None = None,
) -> pd.DataFrame:
    qc_params = {**DEFAULT_QC_PARAMS, **(params or {})}
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    code_class = counts["CodeClass"].astype(str).str.lower()
    positive_mask = code_class.eq("positive")
    negative_mask = code_class.eq("negative")
    endogenous_mask = code_class.eq("endogenous")
    housekeeper_mask = code_class.str.contains("housekeep|housekeeping|reference", regex=True)
    summaries = []

    for sample in sample_cols:
        values = pd.to_numeric(counts[sample], errors="coerce").fillna(0)
        positive_values = values[positive_mask]
        negative_values = values[negative_mask]
        endogenous_values = values[endogenous_mask]
        housekeeper_values = values[housekeeper_mask]
        negative_mean = _safe_mean(negative_values)
        negative_sd = _safe_sd(negative_values)
        background_threshold = negative_mean + (
            float(qc_params["background_sd_multiplier"]) * negative_sd
        )
        summaries.append(
            {
                "sample_id": sample,
                "library_size": int(values.sum()),
                "endogenous_library_size": int(endogenous_values.sum()),
                "detected_probes": int((values > 0).sum()),
                "detected_endogenous_probes": int((endogenous_values > 0).sum()),
                "positive_control_sum": int(positive_values.sum()),
                "positive_control_cv": _safe_cv(positive_values),
                "positive_control_linearity": _positive_control_linearity(
                    positive_values,
                    counts.loc[positive_mask, "Name"] if positive_mask.any() else pd.Series(dtype=str),
                ),
                "negative_control_mean": negative_mean,
                "negative_control_sd": negative_sd,
                "background_threshold": background_threshold,
                "endogenous_above_background": int((endogenous_values > background_threshold).sum()),
                "hk_geomean": _geometric_mean(housekeeper_values),
            }
        )

    summary = pd.DataFrame(summaries)
    if rcc_metrics is not None and not rcc_metrics.empty:
        summary = summary.merge(rcc_metrics, on="sample_id", how="left")
    return add_qc_flags(summary, qc_params)


def add_qc_flags(summary: pd.DataFrame, params: dict | None = None) -> pd.DataFrame:
    if summary.empty:
        return summary.assign(qc_warnings="", qc_status="")

    qc_params = {**DEFAULT_QC_PARAMS, **(params or {})}
    qc_mode = str(qc_params.get("qc_mode", "easy_strict"))
    flagged = summary.copy()
    if qc_mode == "bruker_like":
        return _add_bruker_like_qc_flags(flagged, qc_params)

    med_library = flagged["library_size"].median()
    med_detected = flagged["detected_endogenous_probes"].median()
    med_positive = flagged["positive_control_sum"].median()
    neg_limit = _median_plus_mad(
        flagged["negative_control_mean"],
        multiplier=float(qc_params["negative_control_mad_multiplier"]),
    )

    warnings = []
    failures = []
    for _, row in flagged.iterrows():
        sample_warnings = []
        sample_failures = []
        if med_library > 0 and row["library_size"] < med_library * float(
            qc_params["low_library_fraction"]
        ):
            sample_warnings.append("low_library_size")
        if med_library > 0 and row["library_size"] < med_library * float(
            qc_params["fail_library_fraction"]
        ):
            sample_failures.append("very_low_library_size")
        if med_detected > 0 and row["detected_endogenous_probes"] < med_detected * float(
            qc_params["low_detected_fraction"]
        ):
            sample_warnings.append("low_detected_endogenous_probes")
        if med_detected > 0 and row["detected_endogenous_probes"] < med_detected * float(
            qc_params["fail_detected_fraction"]
        ):
            sample_failures.append("very_low_detected_endogenous_probes")
        if med_positive > 0 and row["positive_control_sum"] < med_positive * float(
            qc_params["low_positive_fraction"]
        ):
            sample_warnings.append("low_positive_controls")
        if pd.notna(neg_limit) and row["negative_control_mean"] > neg_limit:
            sample_warnings.append("high_negative_controls")
        if "fov_counted_fraction" in row and pd.notna(row["fov_counted_fraction"]):
            if row["fov_counted_fraction"] < float(qc_params["min_fov_counted_fraction"]):
                sample_failures.append("low_fov_counted_fraction")
        if "bindingdensity" in row and pd.notna(row["bindingdensity"]):
            if row["bindingdensity"] < float(qc_params["binding_density_min"]):
                sample_warnings.append("low_binding_density")
            if row["bindingdensity"] > float(qc_params["binding_density_max"]):
                sample_warnings.append("high_binding_density")
        warnings.append(sample_warnings)
        failures.append(sample_failures)

    flagged["qc_warnings"] = [";".join(items) for items in warnings]
    flagged["qc_fail_reasons"] = [";".join(items) for items in failures]
    flagged["qc_status"] = [
        "FAIL" if failure_items else ("WARN" if warning_items else "PASS")
        for warning_items, failure_items in zip(warnings, failures)
    ]
    return _add_common_qc_output_columns(flagged, qc_mode="easy_strict")


def background_correct_counts(counts: pd.DataFrame, qc_summary: pd.DataFrame) -> pd.DataFrame:
    corrected = counts.copy()
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    thresholds = qc_summary.set_index("sample_id")["background_threshold"].to_dict()

    for sample in sample_cols:
        threshold = float(thresholds.get(sample, 0))
        values = pd.to_numeric(corrected[sample], errors="coerce").fillna(0)
        corrected[sample] = (values - threshold).clip(lower=0)

    return corrected


def filter_endogenous_counts(
    counts: pd.DataFrame,
    qc_summary: pd.DataFrame,
    metadata: pd.DataFrame,
    group_column: str,
    params: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    qc_params = {**DEFAULT_QC_PARAMS, **(params or {})}
    sample_cols = [col for col in counts.columns if col not in {"CodeClass", "Name"}]
    status_column = "exclusion_status" if "exclusion_status" in qc_summary.columns else "qc_status"
    passing_samples = qc_summary.loc[qc_summary[status_column].ne("FAIL"), "sample_id"].tolist()
    usable_samples = [sample for sample in sample_cols if sample in passing_samples]

    code_class = counts["CodeClass"].astype(str).str.lower()
    endogenous = code_class.eq("endogenous")
    values = counts[usable_samples].apply(pd.to_numeric, errors="coerce").fillna(0)
    min_count = float(qc_params["filter_min_count"])
    min_samples = int(qc_params["filter_min_samples"])
    detected = values >= min_count

    keep_global = detected.sum(axis=1) >= min_samples
    keep_group = pd.Series(False, index=counts.index)
    if group_column in metadata.columns:
        for _, group_metadata in metadata.groupby(group_column):
            group_samples = [sample for sample in group_metadata["sample_id"] if sample in usable_samples]
            if group_samples:
                keep_group = keep_group | (
                    detected[group_samples].sum(axis=1) >= int(qc_params["filter_min_group_samples"])
                )

    keep = endogenous & (keep_global | keep_group)
    filtered = counts.loc[keep, ["CodeClass", "Name", *usable_samples]].copy()
    decisions = pd.DataFrame(
        {
            "Name": counts["Name"],
            "CodeClass": counts["CodeClass"],
            "kept": keep,
            "detected_samples": detected.sum(axis=1) if usable_samples else 0,
            "reason": [
                "kept" if is_kept else ("control_probe" if not is_endo else "low_expression")
                for is_kept, is_endo in zip(keep, endogenous)
            ],
        }
    )
    return filtered, decisions


def _safe_mean(values: pd.Series) -> float:
    return float(values.mean()) if len(values) else 0.0


def _safe_sd(values: pd.Series) -> float:
    return float(values.std(ddof=0)) if len(values) else 0.0


def _safe_cv(values: pd.Series) -> float:
    mean = _safe_mean(values)
    if mean == 0:
        return 0.0
    return float(_safe_sd(values) / mean)


def _add_bruker_like_qc_flags(flagged: pd.DataFrame, qc_params: dict) -> pd.DataFrame:
    output = flagged.copy()
    warnings = []
    failures = []
    for _, row in output.iterrows():
        sample_warnings = []
        sample_failures = []
        hk_geomean = row.get("hk_geomean")
        if pd.notna(hk_geomean) and float(hk_geomean) < float(qc_params["bruker_hk_geomean_min"]):
            sample_warnings.append("low_housekeeper_geomean")
            if bool(qc_params.get("strict_exclusion")):
                sample_failures.append("low_housekeeper_geomean")
        fov = row.get("fov_counted_fraction")
        if pd.notna(fov) and float(fov) < float(qc_params["min_fov_counted_fraction"]):
            sample_warnings.append("low_fov_registration_rate")
        binding = row.get("bindingdensity")
        if pd.notna(binding):
            if float(binding) < float(qc_params["binding_density_min"]):
                sample_warnings.append("low_binding_density")
            if float(binding) > float(qc_params["binding_density_max"]):
                sample_warnings.append("high_binding_density")
        linearity = row.get("positive_control_linearity")
        if pd.notna(linearity) and float(linearity) < float(qc_params["positive_control_linearity_min"]):
            sample_warnings.append("low_positive_control_linearity")
        warnings.append(sample_warnings)
        failures.append(sample_failures)

    output["qc_warnings"] = [";".join(items) for items in warnings]
    output["qc_fail_reasons"] = [";".join(items) for items in failures]
    output["qc_status"] = [
        "FAIL" if failure_items else ("WARN" if warning_items else "PASS")
        for warning_items, failure_items in zip(warnings, failures)
    ]
    return _add_common_qc_output_columns(output, qc_mode="bruker_like")


def _add_common_qc_output_columns(flagged: pd.DataFrame, qc_mode: str) -> pd.DataFrame:
    output = flagged.copy()
    warning_text = output["qc_warnings"].astype(str) if "qc_warnings" in output else pd.Series([""] * len(output))
    output["qc_mode"] = qc_mode
    output["hk_geomean_flag"] = warning_text.str.contains("housekeeper").map(
        lambda value: "WARN" if value else "PASS"
    )
    output["positive_control_linearity_flag"] = warning_text.str.contains(
        "positive_control_linearity"
    ).map(lambda value: "WARN" if value else "PASS")
    output["fov_registration_rate"] = output.get("fov_counted_fraction", pd.Series([pd.NA] * len(output)))
    output["fov_registration_flag"] = warning_text.str.contains("fov").map(
        lambda value: "WARN" if value else "PASS"
    )
    output["binding_density"] = output.get("bindingdensity", pd.Series([pd.NA] * len(output)))
    output["binding_density_flag"] = warning_text.str.contains("binding_density").map(
        lambda value: "WARN" if value else "PASS"
    )
    output["exclusion_status"] = output["qc_status"].where(output["qc_status"].eq("FAIL"), "INCLUDED")
    output["exclusion_reason"] = output.get("qc_fail_reasons", "")
    return output


def _geometric_mean(values: pd.Series) -> float:
    positive = pd.to_numeric(values, errors="coerce").dropna()
    positive = positive[positive > 0]
    if len(positive) == 0:
        return 0.0
    return float(math.exp(positive.apply(math.log).mean()))


def _positive_control_linearity(values: pd.Series, names: pd.Series) -> float | None:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if len(numeric) < 3 or numeric.nunique() < 2:
        return None
    ranks = pd.Series(range(len(numeric), 0, -1), index=numeric.index, dtype=float)
    corr = _pearson_correlation(numeric.rank(), ranks)
    return float(corr) if pd.notna(corr) else None


def _pearson_correlation(left: pd.Series, right: pd.Series) -> float | None:
    aligned = pd.concat([left, right], axis=1).dropna()
    if len(aligned) < 2:
        return None
    left_values = aligned.iloc[:, 0].astype(float)
    right_values = aligned.iloc[:, 1].astype(float)
    left_centered = left_values - left_values.mean()
    right_centered = right_values - right_values.mean()
    denominator = math.sqrt(float((left_centered**2).sum() * (right_centered**2).sum()))
    if denominator == 0:
        return None
    return float((left_centered * right_centered).sum() / denominator)


def _median_plus_mad(values: pd.Series, multiplier: float) -> float:
    median = values.median()
    mad = (values - median).abs().median()
    return float(median + multiplier * mad)
