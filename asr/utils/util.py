from itertools import repeat
from pathlib import Path

import pandas as pd

ROOT_PATH = Path(__file__).absolute().resolve().parent.parent.parent


def import_class_by_path(path: str):
    """
    Recursive import of class by path string.
    """
    paths = path.split('.')
    path = ".".join(paths[:-1])
    class_name = paths[-1]
    mod = __import__(path, fromlist=[class_name])
    mod = getattr(mod, class_name)
    return mod


def inf_loop(data_loader):
    """wrapper function for endless data loader."""
    for loader in repeat(data_loader):
        yield from loader
