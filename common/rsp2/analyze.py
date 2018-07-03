
import tempfile
import shutil
from pathlib import Path
from . import io

__all__ = [
    'to_dir',
]

import subprocess

def to_dir(structure_path, esols):
    temp = temp_dir()
    io.eigensols.to_path(temp / 'eigensols.json', esols)
    subprocess.run([
        'rsp2-sparse-analysis',
        structure_path,
        temp / 'eigensols.json',
        '--output', temp.dir,
    ]).check_returncode()
    return Rsp2Output(temp)

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

    def __truediv__(self, other):
        return self.dir / other

class Rsp2Output:
    def __init__(self, dir):
        self.dir = dir

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        if hasattr(self.dir, '__exit__'):
            self.dir.__exit__(*exc_info)

    def raman(self):
        return io.dwim.from_path(self.dir / 'raman.json')

    def unfold_data(self):
        return io.dwim.from_path(self.dir / 'unfold.json')
