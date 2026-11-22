"""Spouštěč projektových skriptů se zachováním původních cest."""

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
SCRIPTS = ROOT / "scripts"


def available_scripts():
    """Zmapuje dostupné Python skripty podle názvu souboru."""
    result = {}

    for path in SCRIPTS.rglob("*.py"):
        result.setdefault(path.stem, []).append(path)

    return result


def main():
    """Spustí vybraný projektový skript se zachováním starších relativních cest."""
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
