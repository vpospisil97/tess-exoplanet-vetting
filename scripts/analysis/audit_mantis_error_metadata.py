
"""
MODEL B - CORRECTED METADATA AUDIT
==================================

Inspect eight persistent validation errors using
the actual column names in signals.csv.

Checks:
    - Signal UID
    - TIC identifier
    - Binary label versus original signal_label
    - Orbital period in days
    - Transit duration in hours
    - Transit depth in ppm
    - Signal-to-noise ratio
    - Original astronomical dispositions

Does not train models or use held-out test data.

Run:
    python -u scripts/analysis/audit_mantis_error_metadata.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]

SIGNALS = ROOT / "signals.csv"

ERRORS = (
    ROOT
    / "full_dataset"
    / "mantis_model_b"
    / "error_analysis"
    / "validation_error_details.csv"
)

OUTPUT = (
    ROOT
    / "full_dataset"
    / "mantis_model_b"
    / "error_analysis"
    / "selected_metadata_audit.csv"
)

REPORT = (
    ROOT
    / "docs"
    / "model_b_metadata_audit.md"
)

NUMERIC_FIELDS = (
    "period_days",
    "duration_hours",
    "depth_ppm",
    "planet_snr",
    "epoch_bjd",
)

TEXT_FIELDS = (
    "signal_source",
    "signal_label",
    "signal_label_reason",
    "tess_disposition",
    "tfopwg_disposition",
    "sectors",
    "source_pipeline",
    "training_ready_minimum",
    "n_fits_for_this_tic",
)


def normalize_tic(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    if text.upper().startswith("TIC"):
        text = text[3:].strip()

    if text.endswith(".0"):
        text = text[:-2]

    return text


def load_errors():
    if not ERRORS.is_file():
        raise FileNotFoundError(ERRORS)

    errors = pd.read_csv(
        ERRORS,
        dtype={
            "signal_uid": str,
            "tic_key": str,
        },
    )

    if len(errors) != 293:
        raise ValueError(
            f"Expected 293 validation signals, "
            f"found {len(errors)}"
        )

    if errors["signal_uid"].duplicated().any():
        raise ValueError(
            "Duplicate signal UID in error report"
        )

    required = (
        "signal_uid",
        "tic_key",
        "true_label",
        "tuned_mean_probability",
        "tuned_errors_out_of_3",
    )

    for field in required:
        if field not in errors.columns:
            raise KeyError(
                f"Missing error field: {field}"
            )

    return errors


def select_errors(errors):
    fp = errors[
        (errors["true_label"] == 0)
        & (errors["tuned_errors_out_of_3"] == 3)
    ].sort_values(
        ["tuned_mean_probability", "signal_uid"],
        ascending=[False, True],
    ).head(4).copy()

    fn = errors[
        (errors["true_label"] == 1)
        & (errors["tuned_errors_out_of_3"] == 3)
    ].sort_values(
        ["tuned_mean_probability", "signal_uid"],
        ascending=[True, True],
    ).head(4).copy()

    if len(fp) != 4 or len(fn) != 4:
        raise RuntimeError(
            "Expected four examples from each error class"
        )

    fp["error_type"] = "false_positive"
    fn["error_type"] = "false_negative"

    selected = pd.concat(
        [fp, fn],
        ignore_index=True,
    )

    return selected


def load_signals():
    if not SIGNALS.is_file():
        raise FileNotFoundError(
            f"Metadata file not found: {SIGNALS}"
        )

    signals = pd.read_csv(
        SIGNALS,
        dtype={
            "signal_uid": str,
            "tic_id": str,
        },
        low_memory=False,
    )

    required = (
        "signal_uid",
        "tic_id",
        *NUMERIC_FIELDS,
        *TEXT_FIELDS,
    )

    missing = [
        field
        for field in required
        if field not in signals.columns
    ]

    if missing:
        raise KeyError(
            f"Missing metadata columns: {missing}"
        )

    signals["signal_uid"] = (
        signals["signal_uid"].astype(str).str.strip()
    )

    if signals["signal_uid"].duplicated().any():
        raise ValueError(
            "Duplicate signal UID in signals.csv"
        )

    for field in NUMERIC_FIELDS:
        signals[field] = pd.to_numeric(
            signals[field],
            errors="coerce",
        )

        signals[field] = signals[field].replace(
            [np.inf, -np.inf],
            np.nan,
        )

    return signals.set_index(
        "signal_uid",
        drop=False,
    )


def audit(selected, signals):
    records = []

    for _, example in selected.iterrows():
        uid = str(example["signal_uid"]).strip()

        if uid not in signals.index:
            raise ValueError(
                f"Signal not found in metadata: {uid}"
            )

        signal = signals.loc[uid]

        expected_tic = normalize_tic(
            example["tic_key"]
        )

        metadata_tic = normalize_tic(
            signal["tic_id"]
        )

        if expected_tic != metadata_tic:
            raise RuntimeError(
                f"TIC mismatch: {uid}: "
                f"{expected_tic} != {metadata_tic}"
            )

        binary_label = int(
            example["true_label"]
        )

        expected_label = (
            "confirmed"
            if binary_label == 1
            else "false_positive"
        )

        metadata_label = str(
            signal["signal_label"]
        ).strip().lower()

        if metadata_label != expected_label:
            raise RuntimeError(
                f"Label mismatch: {uid}: "
                f"{metadata_label} != {expected_label}"
            )

        score = float(
            example["tuned_mean_probability"]
        )

        if not np.isfinite(score) or not 0 <= score <= 1:
            raise ValueError(
                f"Invalid model score: {uid}"
            )

        record = {
            "error_type": example["error_type"],
            "signal_uid": uid,
            "tic_id": expected_tic,
            "true_label": binary_label,
            "model_score": score,
        }

        for field in NUMERIC_FIELDS:
            record[field] = signal[field]

        for field in TEXT_FIELDS:
            value = signal[field]

            record[field] = (
                ""
                if pd.isna(value)
                else str(value)
            )

        records.append(record)

    return pd.DataFrame(records)


def save_report(result):
    REPORT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    columns = [
        "error_type",
        "signal_uid",
        "tic_id",
        "model_score",
        "signal_label",
        "period_days",
        "duration_hours",
        "depth_ppm",
        "planet_snr",
        "tess_disposition",
        "tfopwg_disposition",
    ]

    table = result[columns].copy()

    lines = [
        "# Model B - Corrected Metadata Audit",
        "",
        "## Purpose",
        "",
        "Inspect eight persistent validation errors",
        "using original TESS signal metadata.",
        "",
        "The sample deliberately contains four",
        "confident false positives and four",
        "confident false negatives.",
        "",
        "## Verification",
        "",
        "- All eight signal UIDs matched.",
        "- TIC identifiers matched for all signals.",
        "- Metadata labels matched the prepared binary labels.",
        "- No held-out test examples were used.",
        "",
        "## Selected signals",
        "",
        table.to_markdown(
            index=False,
            floatfmt=".4f",
        ),
        "",
        "## Units",
        "",
        "- `period_days`: days",
        "- `duration_hours`: hours",
        "- `depth_ppm`: parts per million",
        "- `planet_snr`: signal-to-noise ratio",
        "- `epoch_bjd`: Barycentric Julian Date",
        "",
        "## Data quality",
        "",
    ]

    for field in NUMERIC_FIELDS:
        missing = int(
            result[field].isna().sum()
        )

        lines.append(
            f"- Missing {field}: {missing}/8"
        )

    lines.extend([
        "",
        "## Interpretation cautions",
        "",
        "- Model scores are not calibrated probabilities.",
        "- Deep transits need not originate from planets.",
        "- Labels and dispositions are taken from",
        "  the source metadata and may reflect",
        "  different classification conventions.",
        "- The eight signals were selected because",
        "  the classifier made persistent errors.",
        "- This sample is not representative",
        "  of the complete validation dataset.",
        "- Identifying a physical false-positive",
        "  mechanism requires additional evidence.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python -u .\\scripts\\analysis\\audit_mantis_error_metadata.py",
        "```",
        "",
    ])

    REPORT.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )


def main():
    print("=" * 72)
    print("MODEL B - CORRECTED METADATA AUDIT")
    print("=" * 72)

    errors = load_errors()

    selected = select_errors(errors)

    signals = load_signals()

    print("Metadata:", SIGNALS)
    print("Metadata records:", len(signals))
    print("Selected validation signals:", len(selected))

    result = audit(
        selected,
        signals,
    )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    result.to_csv(
        OUTPUT,
        index=False,
        encoding="utf-8-sig",
    )

    print()
    print("=" * 72)
    print("CORRECTED ASTRONOMICAL METADATA")
    print("=" * 72)

    columns = [
        "error_type",
        "signal_uid",
        "tic_id",
        "model_score",
        "period_days",
        "duration_hours",
        "depth_ppm",
        "planet_snr",
    ]

    print(
        result[columns].to_string(
            index=False,
            na_rep="MISSING",
            float_format=lambda value: f"{value:.4f}",
        )
    )

    print()
    print("=" * 72)
    print("ASTRONOMICAL DISPOSITIONS")
    print("=" * 72)

    print(
        result[
            [
                "signal_uid",
                "signal_label",
                "tess_disposition",
                "tfopwg_disposition",
                "signal_label_reason",
            ]
        ].to_string(
            index=False,
            na_rep="MISSING",
        )
    )

    print()
    print("=" * 72)
    print("MISSING NUMERIC METADATA")
    print("=" * 72)

    for field in NUMERIC_FIELDS:
        print(
            f"{field}: "
            f"{int(result[field].isna().sum())}/8"
        )

    save_report(result)

    print()
    print("TIC mismatches: 0")
    print("Label mismatches: 0")
    print("CSV:", OUTPUT)
    print("Report:", REPORT)
    print("TEST SET NOT USED")
    print("CORRECTED METADATA AUDIT COMPLETE")


if __name__ == "__main__":
    main()
