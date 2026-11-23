"""Vytvoří úplnou závěrečnou zprávu E3, včetně E1 a E2.

Zpráva používá průměr přes tři inicializace v každém foldu
pak průměr a výběrovou směrodatnou odchylku přes pět foldů.
Všechny porovnávané varianty jsou párované foldem a seedem.

Spouštění:
    python -u scripts/training/report_e3_full.py
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prepare_e3_folds as cv

MODELS = ("baseline", "cnn", "mantis", "e1_local_only", "e1_global_only", "e2_without_mask")
METRICS = ("average_precision", "roc_auc", "balanced_accuracy", "f1")
DESTINATION = cv.ROOT / "docs" / "e3_cross_validation.md"


def read_all():
    """Načte pouze úplnou sadu uložených validačních výsledků."""
    data = cv.read_source()
    folds = cv.load_folds(data)
    results = {}
    for model in MODELS:
        rows = {}
        seeds = (None,) if model == "baseline" else cv.SEEDS
        for fold in range(1, 6):
            for seed in seeds:
                run = cv.load_complete(model, fold, seed)
                if run is None:
                    raise RuntimeError(f"Chybí výsledek {model}, fold {fold}, seed {seed}")
                if run["outer_validation_signals"] != len(folds[fold-1][1]):
                    raise RuntimeError(f"Neplatný počet validačních signálů {model}, fold {fold}")
                for metric in METRICS:
                    if metric not in run["metrics"]:
                        raise RuntimeError(f"Chybí metrika {model}, fold {fold}: {metric}")
                rows[(fold, seed)] = run
        results[model] = rows
    return results


def fold_means(results, metric):
    """U neuronové sítě nejprve zprůměruje seedy v každém foldu."""
    values = []
    for fold in range(1, 6):
        fold_values = [r["metrics"][metric] for (k, _), r in results.items() if k == fold]
        values.append(statistics.mean(fold_values))
    return values


def format_mean_sd(values):
    """Vrátí průměr a výběrovou směrodatnou odchylku pěti foldů."""
    return f"{statistics.mean(values):.4f} ± {statistics.stdev(values):.4f}"


def comparison_row(label, results):
    """Vykreslí tabulkový souhrn čtyř metrik přes pět foldů."""
    cells = [format_mean_sd(fold_means(results, m)) for m in METRICS]
    return f"| {label} | " + " | ".join(cells) + " |"


def paired_differences(a, b, metric):
    """Spočítá párované rozdíly pro stejný fold a seed."""
    differences = {}
    for fold in range(1, 6):
        values = []
        for seed in cv.SEEDS:
            values.append(a[(fold, seed)]["metrics"][metric] - b[(fold, seed)]["metrics"][metric])
        differences[fold] = values
    return differences


def append_difference(lines, title, a, b):
    """Přidá párovaný rozdíl AP včetně výsledků jednotlivých foldů."""
    diff = paired_differences(a, b, "average_precision")
    fold_averages = [statistics.mean(diff[fold]) for fold in range(1, 6)]
    all_values = [value for vals in diff.values() for value in vals]
    lines.extend([
        f"**{title}:** ΔAP = {format_mean_sd(fold_averages)} "
        f"(pět průměrů foldů; kladných párů {sum(x > 0 for x in all_values)}/15).",
        "",
    ])
    return fold_averages


def write_report():
    """Ověří kompletnost výsledků a bezpečně uloží sumarizaci E3."""
    all_results = read_all()
    lines = [
        "# E3 — pětinásobná skupinová cross-validation včetně E1/E2",
        "",
        "## Metodika a hranice platnosti",
        "",
        "- Výchozí dataset: 1 918 označených signálů. CV používá pouze 1 628 původních train+validation signálů; 290 původních testovacích příkladů zůstává odděleno.",
        "- Pět vnějších foldů metodou `StratifiedGroupKFold` seskupuje signály podle hvězdy TIC. Každý vývojový signál je jednou ve vnější validaci.",
        "- Baseline se vyhodnotí jednou v každém foldu, CNN/Mantis a ablace E1/E2 ve třech seedech 42, 123, 2026 v každém foldu.",
        "- Pro neuronové sítě se počet epoch vybírá oddělenou **vnitřní** TIC-validací. Po výběru se model inicializuje znovu a natrénuje na celé vnější trénovací části.",
        "- Stejné výchozí hyperparametry jako v dokončeném projektu; nedělá se nový grid search. Normalizace baseline a Mantis se učí jen na aktuálním trénovacím foldu.",
        "- E1: nepoužitá větev CNN se nuluje **až po enkodéru**. E2: vstupní kanál masky se nuluje, zatímco flux zůstává stejný.",
        "- Výsledky CNN `both` a E2 `with_mask` jsou **stejné uložené CNN běhy**, nikoli nové nezávislé modely. Podmínky jsou tedy párované foldem a seedem.",
        "- Hyperparametry byly původně vybrány na validačních příkladech, které nyní patří do CV poolu; jde proto o **dodatečné ověření robustnosti**, ne zcela nezávislý, nestranný odhad výkonu po výběru modelu.",
        "- Uváděná SD je výběrová SD z pěti průměrů foldů. Nejde o 15 nezávislých datových splitů ani o interval spolehlivosti.",
        "",
        "## Hlavní modely – validace v pěti vnějších foldech",
        "",
        "| Model | AP | ROC-AUC | Balanced accuracy | F1 |",
        "|---|---:|---:|---:|---:|",
    ]
    for model, title in (("baseline", "Logistická regrese"), ("cnn", "CNN"), ("mantis", "Mantis-8M + MLP")):
        lines.append(comparison_row(title, all_results[model]))

    lines.extend([
        "", "## E1 – ablace lokálního a globálního pohledu", "",
        "| Varianta | AP | ROC-AUC | Balanced accuracy | F1 |",
        "|---|---:|---:|---:|---:|",
        comparison_row("Oba pohledy", all_results["cnn"]),
        comparison_row("Pouze lokální", all_results["e1_local_only"]),
        comparison_row("Pouze globální", all_results["e1_global_only"]),
        "",
    ])
    append_difference(lines, "Oba minus lokální", all_results["cnn"], all_results["e1_local_only"])
    append_difference(lines, "Oba minus globální", all_results["cnn"], all_results["e1_global_only"])

    lines.extend([
        "## E2 – ablace explicitní masky pokrytí", "",
        "| Varianta | AP | ROC-AUC | Balanced accuracy | F1 |",
        "|---|---:|---:|---:|---:|",
        comparison_row("S maskou", all_results["cnn"]),
        comparison_row("Bez masky", all_results["e2_without_mask"]),
        "",
    ])
    append_difference(lines, "S maskou minus bez masky", all_results["cnn"], all_results["e2_without_mask"])

    lines.extend([
        "## AP podle vnějšího foldu (průměr přes seedy)", "",
        "| Fold | Baseline | CNN | Mantis | E1 lokální | E1 globální | E2 bez masky |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for fold in range(1, 6):
        values = [statistics.mean(
            run["metrics"]["average_precision"]
            for (row_fold, _), run in all_results[model].items() if row_fold == fold
        ) for model in MODELS]
        lines.append(f"| {fold} | " + " | ".join(f"{v:.4f}" for v in values) + " |")

    lines.extend([
        "", "## Interpretace a omezení", "",
        "- E1/E2 představují citlivost na konkrétní hvězdy a náhodnou inicializaci při pěti skupinových rozděleních, nikoli důkaz kauzality či záruku výkonu na neověřených kandidátech.",
        "- Rozdíl mezi dvěma variantami je vhodnější hodnotit **párově v rámci stejných foldů a seedů** než srovnáním nesouvisejících průměrů.",
        "- Neaktivní větev v E1 nemá stejnou efektivní kapacitu jako dvouvětvová CNN a nulový flux může i bez explicitní masky zčásti indikovat chybějící pozorování.",
        "- Výsledky tohoto E3 nejsou náhradou dříve zamrazeného jednorázového testu; původní metriky se nepřepisují.",
        "- CV SD neodstraňuje riziko biasu katalogových štítků, reprezentativnosti dat ani výběru hyperparametrů před CV.",
        "", "## Reprodukce", "",
        "```powershell",
        "python -u .\\scripts\\training\\prepare_e3_folds.py",
        "python -u .\\scripts\\training\\cross_validate_baseline.py",
        "python -u .\\scripts\\training\\cross_validate_cnn.py",
        "python -u .\\scripts\\training\\cross_validate_mantis.py",
        "python -u .\\scripts\\training\\cross_validate_ablations.py --experiment all",
        "python -u .\\scripts\\training\\report_e3_full.py",
        "```", "",
    ])
    DESTINATION.parent.mkdir(exist_ok=True)
    DESTINATION.write_text("\n".join(lines), encoding="utf-8")
    print(f"Souhrnný report: {DESTINATION}")
    print("\n".join(lines[:23]))


if __name__ == "__main__":
    write_report()
