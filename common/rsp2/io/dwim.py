from pathlib import Path
import io

def from_path(path, file=None, fullpath=None):
    if isinstance(path, Path):
        path = str(path)

    if fullpath is None:
        fullpath = path
    if file is None:
        with open(path, 'rb') as file:
            return from_path(path, file)

    if _endswith_nocase(path, '.json'):
        import json
        return json.load(wrap_text(file))

    elif _endswith_nocase(path, '.npy'):
        import numpy
        return numpy.load(file)

    elif _endswith_nocase(path, '.npz'):
        import scipy.sparse
        return scipy.sparse.load_npz(file)

    elif _endswith_nocase(path, '.gz'):
        import gzip
        return from_path(path[:-len('.gz')], gzip.GzipFile(path, fileobj=file))

    else:
        raise ValueError(f'unknown extension in {repr(fullpath)}')

def to_path(path, obj, file=None, fullpath=None):
    if isinstance(path, Path):
        path = str(path)

    if fullpath is None:
        fullpath = path
    if file is None:
        with open(path, 'wb+') as file:
            return to_path(path, obj, file, fullpath)

    if _endswith_nocase(path, '.json'):
        import json
        json.dump(obj, wrap_text(file))

    elif _endswith_nocase(path, '.npy'):
        import numpy
        numpy.save(file, obj)

    elif _endswith_nocase(path, '.npz'):
        import scipy.sparse
        from io import BytesIO
        if scipy.sparse.issparse(obj):
            # (even if you say compressed=False, save_npz goes through the
            #  zipfile interface, which attempts to seek, which GZip )
            buf = BytesIO()
            scipy.sparse.save_npz(buf, obj, compressed=False)
            file.write(buf.getvalue())
        else:
            raise TypeError('dwim .npz is only supported for sparse')

    elif _endswith_nocase(path, '.gz'):
        import gzip
        to_path(path[:-len('.gz')], obj, gzip.GzipFile(path, mode='xb', fileobj=file))

    else:
        raise ValueError(f'unknown extension in {repr(fullpath)}')

def _endswith_nocase(s, suffix):
    return s[len(s) - len(suffix):].upper() == suffix.upper()

def wrap_text(f):
    if hasattr(f, 'encoding'):
        return f
    else:
        return io.TextIOWrapper(f)
