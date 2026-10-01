from pathlib import Path

from functools import lru_cache
from flowtangent.utils.io import _ft_root

from flowtangent.components import Airfoil
import shutil

# ----------------------------------------------------------------------------------------------------------------------
#  Airfoil Directory
# ----------------------------------------------------------------------------------------------------------------------

_AF_DIR = _ft_root() / "data/airfoils"
STUB_FILE = Path(__file__).resolve().parent / "_data.pyi"

@lru_cache(maxsize=None)
def _load_map_from_disk(name: str):
    filename = name.replace('_', '-')
    """Hidden helper that does the disk I/O, safely cached, and routes by type."""
    file_path = _AF_DIR / f"{name}.txt"
    if not file_path.exists():
        raise AttributeError(f"Map '{name}' not found in FlowTangent library ({_AF_DIR}).")

    return Airfoil.from_file(file_path)


def __getattr__(name: str):
    """Intercepts module-level attribute access."""
    if name.startswith("_"):
        raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
    return _load_map_from_disk(name)


def __dir__():
    """Allows IDEs and the `dir()` command to see the available airfoils."""
    if _AF_DIR.exists():
        return [f.stem for f in _AF_DIR.glob("*.dat")]
    return []

def generate_stub():
    lines = [
        "from typing import Any",
        "from ._classes import Airfoil",
        "",
    ]

    for file in _AF_DIR.glob("*.dat"):
        shutil.move(file, str(file).replace('-', '_'))
        # Write the attribute to the stub file
        lines.append(f"{file.stem.replace('-', '_')}: Airfoil")

    STUB_FILE.write_text("\n".join(lines))
    print(f"Generated {STUB_FILE.name} with {len(lines) - 3} airfoils.")

if __name__ == "__main__":
    generate_stub()