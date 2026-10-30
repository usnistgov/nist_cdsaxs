"""Generate a runnable Python entry point with external JSON settings."""
from pathlib import Path

from .config import write_json


SCRIPT = '''#!/usr/bin/env python3
"""Generated CD-SAXS reduction script.

Edit settings.json to tune constants, integration, slices, and exports.
Examples:
    python reduce_sample.py --stage phi
    python reduce_sample.py --accept phi --stage chi
    python reduce_sample.py --stage reduce
    python reduce_sample.py --config resolved_settings.json

Requires the nist_cdsaxs package (pip install -e /path/to/nist_cdsaxs).
"""
from pathlib import Path
from cdsaxs.harness.__main__ import main

SETTINGS_FILE = Path(__file__).resolve().with_name("settings.json")


def reduce(argv=None):
    """Run staged calibration, corrections, integration, slicing, and export.

    Every rebuild reloads raw detector data and reapplies intensity corrections.
    Stage functions are implemented in cdsaxs.harness.runner and can also be
    imported for custom workflows. All scientific parameters live in JSON.
    """
    return main(argv, default_config=SETTINGS_FILE,
                default_resume=Path(__file__).resolve().parent)


if __name__ == "__main__":
    raise SystemExit(reduce())
'''


def generate(directory, settings):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    settings_path = directory / "settings.json"
    script_path = directory / "reduce_sample.py"
    if not settings_path.exists():
        write_json(settings_path, settings)
    if not script_path.exists():
        script_path.write_text(SCRIPT, encoding="utf-8")
    return script_path
