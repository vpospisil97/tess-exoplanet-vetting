
$ErrorActionPreference = "Stop"

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location -LiteralPath $Root

Write-Host "`n========================================" -ForegroundColor Cyan
Write-Host "TESS - REORGANIZACE PROJEKTU"
Write-Host "========================================" -ForegroundColor Cyan

# Verify that this is the correct project.
$required = @(
    "full_dataset\full_cnn_views.npz",
    "full_dataset\cnn_split_indices.npz",
    "full_dataset\training_manifest.csv"
)

foreach ($item in $required) {
    if (-not (Test-Path -LiteralPath $item)) {
        throw "Chybi dulezity soubor: $item"
    }
}

if (Test-Path -LiteralPath ".\run.py") {
    throw "run.py jiz existuje. Organizaci nespoustej opakovane."
}

# Every known Python script is assigned to one category.
$Groups = [ordered]@{
    "audit" = @(
        "analyze_fits_quality.py",
        "audit_dataset_selection.py",
        "audit_fits_quality.py",
        "audit_training_ready.py",
        "audit_transit_coverage.py",
        "check_dataset_capacity.py",
        "check_duplicates.py",
        "check_fits.py",
        "check_quality_flags.py",
        "review_transit_coverage.py",
        "validate_metadata.py"
    )

    "download" = @(
        "diagnose_failed_downloads.py",
        "diagnose_http_headers.py",
        "download_full_dataset.py"
    )

    "preprocessing" = @(
        "build_full_manifest.py",
        "build_tess_inputs.py",
        "build_training_manifest.py",
        "diagnose_preprocessing.py",
        "prepare_lightcurves.py",
        "test_preprocessing.py"
    )

    "analysis" = @(
        "analyze_all_transits.py",
        "analyze_training_classes.py",
        "inspect_lightcurves.py",
        "inspect_transits.py",
        "plot_phasefold_examples.py",
        "plot_preprocessing.py"
    )

    "training" = @(
        "split_tess_dataset.py",
        "train_baseline.py"
    )
}

# Preflight checks before making any changes.
$plan = @()

foreach ($category in $Groups.Keys) {
    foreach ($filename in $Groups[$category]) {
        $source = Join-Path $Root $filename
        $destination = Join-Path $Root (
            "scripts\$category\$filename"
        )

        if (-not (Test-Path -LiteralPath $source)) {
            throw "Zdrojovy soubor chybi: $filename"
        }

        if (Test-Path -LiteralPath $destination) {
            throw "Cilovy soubor jiz existuje: $destination"
        }

        $plan += [PSCustomObject]@{
            Name = $filename
            Source = $source
            Destination = $destination
        }
    }
}

Write-Host "`nSkriptu k presunu: $($plan.Count)" -ForegroundColor Green

# Create directories.
$directories = @(
    "scripts\audit",
    "scripts\download",
    "scripts\preprocessing",
    "scripts\analysis",
    "scripts\training",
    "src\tess_vetting",
    "data\raw",
    "data\interim",
    "data\processed",
    "reports\figures",
    "reports\tables",
    "models",
    "docs",
    "backups"
)

foreach ($dir in $directories) {
    New-Item -ItemType Directory -Force -Path (
        Join-Path $Root $dir
    ) | Out-Null
}

# Back up all scripts before moving anything.
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$backup = Join-Path $Root "backups\before_structure_$stamp"

New-Item -ItemType Directory -Path $backup | Out-Null

Write-Host "`nVytvarim zalohu..." -ForegroundColor Cyan

foreach ($item in $plan) {
    $copy = Join-Path $backup $item.Name

    Copy-Item -LiteralPath $item.Source -Destination $copy

    $hash1 = (Get-FileHash -LiteralPath $item.Source -Algorithm SHA256).Hash
    $hash2 = (Get-FileHash -LiteralPath $copy -Algorithm SHA256).Hash

    if ($hash1 -ne $hash2) {
        throw "Zaloha nesouhlasi: $($item.Name)"
    }
}

Write-Host "Zaloha OK: $backup" -ForegroundColor Green

# Generate a common launcher.
# It preserves the original root-based __file__
# convention used by existing scripts.
$runner = @'
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def available_scripts():
    result = {}

    for path in SCRIPTS.rglob("*.py"):
        result.setdefault(path.stem, []).append(path)

    return result


def main():
    scripts = available_scripts()

    if len(sys.argv) < 2 or sys.argv[1] in (
        "--list", "--help", "-h"
    ):
        print("TESS PROJECT - AVAILABLE SCRIPTS")
        print()

        for name in sorted(scripts):
            paths = scripts[name]
            if len(paths) == 1:
                relative = paths[0].relative_to(ROOT)
                print(f"  {name:32s} {relative}")

        print()
        print("Usage:")
        print("  python run.py SCRIPT_NAME [arguments]")
        return

    name = sys.argv[1].removesuffix(".py")

    if name not in scripts:
        raise SystemExit(f"Unknown script: {name}")

    matches = scripts[name]

    if len(matches) != 1:
        raise SystemExit(
            f"Ambiguous script name: {name}"
        )

    script = matches[0]

    # Keep legacy root-relative file paths working.
    legacy_file = ROOT / script.name

    sys.argv = [
        str(legacy_file),
        *sys.argv[2:],
    ]

    # Support imports between existing scripts.
    sys.path.insert(0, str(ROOT))

    for folder in SCRIPTS.iterdir():
        if folder.is_dir():
            sys.path.insert(0, str(folder))

    namespace = {
        "__name__": "__main__",
        "__file__": str(legacy_file),
        "__package__": None,
    }

    source = script.read_bytes()

    exec(
        compile(source, str(script), "exec"),
        namespace,
    )


if __name__ == "__main__":
    main()
'@

$runnerPath = Join-Path $Root "run.py"
$runner | Set-Content -LiteralPath $runnerPath -Encoding UTF8

# Move scripts. If verification fails, restore them.
$moved = @()

try {
    Write-Host "`nPresouvam Python skripty..." -ForegroundColor Cyan

    foreach ($item in $plan) {
        Move-Item -LiteralPath $item.Source `
                  -Destination $item.Destination

        $moved += $item

        Write-Host "  $($item.Name)"
    }

    Write-Host "`nOveruji Python syntaxi..." -ForegroundColor Cyan

    $pythonFiles = @($runnerPath) + @(
        $plan | ForEach-Object { $_.Destination }
    )

    & python -m py_compile @pythonFiles

    if ($LASTEXITCODE -ne 0) {
        throw "Kontrola Python syntaxe selhala."
    }

    Write-Host "`nOveruji spoustec..." -ForegroundColor Cyan

    & python $runnerPath --list

    if ($LASTEXITCODE -ne 0) {
        throw "Spoustec nefunguje."
    }

    # Help mode checks the main scripts without training.
    foreach ($script in @(
        "build_tess_inputs",
        "split_tess_dataset",
        "train_baseline"
    )) {
        & python $runnerPath $script --help | Out-Null

        if ($LASTEXITCODE -ne 0) {
            throw "Neprosla kontrola: $script"
        }
    }

} catch {
    Write-Host "`nCHYBA - OBNOVUJI SCRIPTS" -ForegroundColor Red

    for ($i = $moved.Count - 1; $i -ge 0; $i--) {
        $item = $moved[$i]

        if (Test-Path -LiteralPath $item.Destination) {
            Move-Item -LiteralPath $item.Destination `
                      -Destination $item.Source
        }
    }

    throw
}

# Prepare Git exclusions.
$gitignore = @'
__pycache__/
*.py[cod]
.venv/
.env
.vscode/

# Original astronomical data
full_dataset/
TESS_sample_150/
lightcurve_inspection/
prepared_lightcurves/
qlp_diagnostics/
transit_analysis_150/
transit_inspection/

# Generated data and reports
*.fits
*.csv
*.npz
*.joblib
*.png
*.bak
*.bak2

# Local backups
backups/
'@

if (-not (Test-Path ".gitignore")) {
    $gitignore | Set-Content ".gitignore" -Encoding UTF8
}

# Basic dependencies.
$requirements = @'
numpy>=2.0,<3
pandas>=2.0,<3
astropy>=6.0,<8
matplotlib>=3.8,<4
scikit-learn>=1.4,<2
joblib>=1.3,<2
'@

if (-not (Test-Path "requirements.txt")) {
    $requirements | Set-Content "requirements.txt" -Encoding UTF8
}

# Project documentation.
$readme = @'
# TESS Exoplanet Transit Vetting

Deep Learning semester project focused on classification of
TESS transit candidates using astronomical light curves.

## Project structure

- scripts/audit - data validation and quality control
- scripts/download - data acquisition and diagnostics
- scripts/preprocessing - manifests and CNN input creation
- scripts/analysis - exploratory analysis and visualization
- scripts/training - dataset splitting and model training
- src/tess_vetting - reusable Python modules (planned)
- data - future normalized data layout
- reports - figures and experimental tables
- models - trained models and checkpoints
- docs - documentation and LaTeX report

## Current data location

Existing data are intentionally preserved:

- full_dataset/
- signals.csv
- lightcurves.csv
- targets.csv

Data migration is a separate planned step.

## Running scripts

Use the root-level launcher to preserve legacy paths.

List scripts:

    python run.py --list

Create CNN inputs:

    python run.py build_tess_inputs --per-class 0 --seed 42

Create TIC-based split:

    python run.py split_tess_dataset --seed 42

Train baseline:

    python run.py train_baseline --seed 42

## Labels

- 1 = confirmed planet
- 0 = false positive

Train/validation/test are grouped by TIC to prevent
the same star appearing across dataset splits.

## Status

Initial CNN input dataset and leakage-safe split are ready.
The preprocessing pipeline is still experimental and
requires further scientific validation.
'@

if (-not (Test-Path "README.md")) {
    $readme | Set-Content "README.md" -Encoding UTF8
}

Write-Host "`n========================================" -ForegroundColor Green
Write-Host "REORGANIZACE DOKONCENA" -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green

Write-Host "Presunuto Python souboru: $($moved.Count)"
Write-Host "Zaloha: $backup"
Write-Host "FITS soubory zustaly na puvodnim miste."
Write-Host "Dataset a modely nebyly presunuty."

Write-Host "`nNove spousteni:" -ForegroundColor Cyan
Write-Host "python .\run.py --list"
Write-Host "python .\run.py train_baseline --help"
