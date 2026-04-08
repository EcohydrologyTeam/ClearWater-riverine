from pathlib import Path
from typing import Optional
import inspect

def resolve_path(path: Path, root_path: Optional[Path] = None):
    """Resolves filepath from configuration file"""

    if path.is_absolute():
        absolute_path = path
    else:
        if root_path is None:
            root_path = Path.cwd()
        absolute_path = root_path / path
    
    validate_path(absolute_path)
    return absolute_path
    

def validate_path(path: Path):
    """Validate if path exists"""
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    