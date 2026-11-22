
"""
Model B: repeat the grid-selected Mantis-8M classification head
with the same three seeds as the original Mantis MLP.

Reuses train_mantis_head.py and its existing data checks.
Does not load or evaluate the held-out test set.
Does not overwrite the original Mantis head experiments.

Run from the project root:
    python -u scripts/training/train_mantis_head_tuned.py
"""

import hashlib
import json
import re
import statistics
from pathlib import Path

import torch
import train_mantis_head as head

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "full_dataset" / "mantis_model_b"
GRID_DIR = FEATURE_DIR / "head_grid_v1"
SELECTION_FILE = GRID_DIR / "best_head_config.json"
ORIGINAL_DIR = FEATURE_DIR / "mlp_head_v1"
SUMMARY_FILE = GRID_DIR / "tuned_three_seed_summary.json"
REPORT_FILE = ROOT / "docs" / "model_b_tuned_three_seeds.md"

SEEDS = (42, 123, 2026)

METRICS = (
    "average_precision",
    "roc_auc",
    "balanced_accuracy",
    "f1",
)


def file_hash(path):
    """Vypočítá SHA-256 otisk vstupního souboru."""
    digest = hashlib.sha256()

    with path.open("rb") as stream:
        for chunk in iter(
            lambda: stream.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)

    return digest.hexdigest()


def load_selection():
    """Načte nejlepší konfiguraci a ověří otisky příznaků."""
    if not SELECTION_FILE.is_file():
        raise FileNotFoundError(SELECTION_FILE)

    selection = json.loads(
        SELECTION_FILE.read_text(
            encoding="utf-8"
        )
    )

    name = selection.get("selected_trial")
    config = selection.get("config")

    if not isinstance(name, str) or not re.fullmatch(
        r"hidden_(64|128|256)__drop_0p[135]",
        name,
    ):
        raise ValueError(
            f"Unexpected selected trial: {name!r}"
        )

    if (
        not isinstance(config, dict)
        or config.get("trial_name") != name
    ):
        raise ValueError(
            "Saved selection and classifier config disagree"
        )

    if selection.get("selection_seed") != 42:
        raise ValueError(
            "Expected grid-search seed 42"
        )

    if selection.get("test_set_used") is not False:
        raise ValueError(
            "Selection has unexpected test-set status"
        )

    expected_name = (
        f"hidden_{config.get('hidden_size')}__drop_"
        f"{str(config.get('dropout')).replace('.', 'p')}"
    )

    if name != expected_name:
        raise ValueError(
            "Selected trial name does not match hyperparameters"
        )

    saved_hashes = config.get("feature_hashes")

    if not isinstance(saved_hashes, dict):
        raise ValueError(
            "Missing input hashes in grid-search config"
        )

    for split in ("train", "validation"):
        path = FEATURE_DIR / f"features_{split}.npz"

        if file_hash(path) != saved_hashes.get(split):
            raise RuntimeError(
                f"{split} feature data changed since grid search"
            )

    return name, config


def read_original(seed):
    """Načte výsledky původního klasifikátoru pro daný seed."""
    path = (
        ORIGINAL_DIR
        / f"seed_{seed}"
        / "metrics.json"
    )

    if not path.is_file():
        raise FileNotFoundError(path)

    result = json.loads(
        path.read_text(encoding="utf-8")
    )

    if (
        result.get("seed") != seed
        or result.get("test_set_used") is not False
    ):
        raise ValueError(
            f"Invalid original result for seed {seed}"
        )

    return result


def stat(results, metric):
    """Vypočítá průměr a výběrovou směrodatnou odchylku."""
    values = [
        r["validation"][metric]
        for r in results
    ]

    return (
        statistics.mean(values),
        statistics.stdev(values),
    )


def fmt_stats(results, metric):
    """Naformátuje souhrn metrik pro dokumentaci."""
    mean, std = stat(results, metric)
    return f"{mean:.4f} ± {std:.4f}"


def write_outputs(name, config, tuned, original):
    """Uloží porovnání vyladěné a původní klasifikační hlavy."""
    summary = {
        "model": "frozen_Mantis-8M_plus_MLP",
        "selected_trial": name,
        "hyperparameter_selection_seed": 42,
        "seeds": list(SEEDS),
        "config": config,
        "test_set_used": False,
        "tuned_runs": tuned,
        "original_runs": original,
        "comparison": {},
    }

    for metric in METRICS:
        tuned_mean, tuned_sd = stat(
            tuned, metric
        )

        original_mean, original_sd = stat(
            original, metric
        )

        summary["comparison"][metric] = {
            "tuned_mean": tuned_mean,
            "tuned_sample_sd": tuned_sd,
            "original_mean": original_mean,
            "original_sample_sd": original_sd,
            "difference_tuned_minus_original": (
                tuned_mean - original_mean
            ),
        }

    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Model B — Selected Mantis head, three random seeds",
        "",
        "## Selected hyperparameters",
        "",
        f"- Selected by validation AP in the seed-42 nine-trial grid: `{name}`",
        f"- Hidden width: {config['hidden_size']}",
        f"- Dropout: {config['dropout']}",
        "- Pretrained Mantis-8M encoder frozen; only MLP head trained",
        "- TIC-grouped train and validation data identical to original Model B",
        "- Feature scaling fitted on training data only",
        "- Seeds: 42, 123, 2026",
        "- Held-out test data not loaded or evaluated",
        "",
        "## Per-seed validation results",
        "",
        "| Seed | Original AP | Tuned AP | Original ROC-AUC | Tuned ROC-AUC | Tuned F1 | Best epoch |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for base, selected in zip(original, tuned):
        lines.append(
            f"| {selected['seed']} "
            f"| {base['validation']['average_precision']:.4f} "
            f"| {selected['validation']['average_precision']:.4f} "
            f"| {base['validation']['roc_auc']:.4f} "
            f"| {selected['validation']['roc_auc']:.4f} "
            f"| {selected['validation']['f1']:.4f} "
            f"| {selected['best_epoch']} |"
        )

    lines += [
        "",
        "## Three-seed validation mean ± sample standard deviation",
        "",
        "| Metric | Original Mantis head | Grid-selected head | Difference in means |",
        "|---|---:|---:|---:|",
    ]

    for metric in METRICS:
        delta = summary["comparison"][metric][
            "difference_tuned_minus_original"
        ]

        lines.append(
            f"| {metric} "
            f"| {fmt_stats(original, metric)} "
            f"| {fmt_stats(tuned, metric)} "
            f"| {delta:+.4f} |"
        )

    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Seed 42 was used both for grid selection and in this repeat; its result is not independent of selection.",
        "- All metrics come from the same validation stars used for model selection.",
        "- Smaller standard deviation across initializations is not proof of generalization.",
        "- Final comparison requires one-time evaluation on the held-out TIC-grouped test set.",
        "- Feature extraction time is excluded from MLP training time.",
        "",
        "## Reproduction",
        "",
        "```powershell",
        "python -u .\\scripts\\training\\train_mantis_head_tuned.py",
        "```",
        "",
    ]

    REPORT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    REPORT_FILE.write_text(
        "\n".join(lines),
        encoding="utf-8",
    )

    return summary


def main():
    """Ověří a porovná tři běhy vybrané MLP konfigurace."""
    print("=" * 72)
    print("MODEL B - GRID-SELECTED HEAD, THREE SEEDS")
    print("=" * 72)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable"
        )

    name, config = load_selection()

    print("Selected configuration:", name)
    print("Hidden size:", config["hidden_size"])
    print("Dropout:", config["dropout"])
    print("Feature hash verification: OK")

    head.CONFIG = config
    head.OUTPUT = GRID_DIR / name

    train, val, scaler = head.load_data()

    device = torch.device("cuda")

    tuned = []
    original = []

    for seed in SEEDS:
        print(
            f"\n{'=' * 72}\n"
            f"SEED {seed}\n"
            f"{'=' * 72}"
        )

        existing = head.read_completed(seed)

        if existing is not None:
            print(
                "SKIP: already trained and validated"
            )
            result = existing
        else:
            result = head.train_seed(
                seed,
                train,
                val,
                scaler,
                device,
            )

        tuned.append(result)
        original.append(read_original(seed))

        print(
            f"Seed {seed}: "
            f"original AP="
            f"{original[-1]['validation']['average_precision']:.4f} "
            f"| tuned AP="
            f"{result['validation']['average_precision']:.4f}"
        )

    summary = write_outputs(
        name,
        config,
        tuned,
        original,
    )

    print("\n" + "=" * 72)
    print("FINAL COMPARISON: VALIDATION MEAN ± SAMPLE SD")
    print("=" * 72)

    for metric in METRICS:
        delta = summary["comparison"][metric][
            "difference_tuned_minus_original"
        ]

        print(
            f"{metric:20s} "
            f"Original {fmt_stats(original, metric)} "
            f"| Tuned {fmt_stats(tuned, metric)} "
            f"| Difference {delta:+.4f}"
        )

    print("\nReport:", REPORT_FILE)
    print("Summary:", SUMMARY_FILE)
    print("TEST SET NOT USED")
    print("THREE-SEED TUNED MODEL B COMPLETE")


if __name__ == "__main__":
    main()
