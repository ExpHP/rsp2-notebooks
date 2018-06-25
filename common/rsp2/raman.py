
import tempfile
import shutil
from pathlib import Path
from . import io

__all__ = [
    'raman',
]

import subprocess

def raman(structure_path, esols):
    with temp_dir() as temp:
        io.eigensols.to_path(temp / 'eigensols.json', esols)
        subprocess.run([
                'rsp2-sparse-analysis',
                structure_path,
                temp / 'eigensols.json',
                '--output', temp,
        ]).check_returncode()
        return io.dwim.from_path(temp / 'raman.json')

class temp_dir:
    def __init__(self):
        self.dir = Path(tempfile.mkdtemp('-rsp2-nb'))

    def __enter__(self):
        return self.dir

    def __exit__(self, *exc_info):
        if exc_info[0]:
            import warnings
            warnings.warn(f"leaked temp dir: {self.dir}")
        else:
            shutil.rmtree(str(self.dir))
