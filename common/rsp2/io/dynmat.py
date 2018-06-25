import numpy as np

import scipy.sparse

from . import dwim

def from_path(path):
    d = dwim.from_path(path)
    if isinstance(d, dict):
        return _from_dict(d)
    elif isinstance(d, np.ndarray) or scipy.sparse.issparse(d):
        return d
    else:
        raise TypeError

def to_path(path, obj):
    dwim.to_path(path, obj)

def _from_npy(m): return m
def _to_npy(m): return m

def _from_dict(m):
    data = np.array(m['complex-blocks'])

    assert data.shape[1] == 2

    # scipy will use eigs for dtype=complex even if imag is all zero.
    # Unfortunately this leads to small but nonzero imaginary components
    # appearing in the output eigenkets at gamma.
    # Hence we force dtype=float when possible.
    if np.absolute(data[:, 1]).max() == 0:
        data = data[:, 0] * 1.0
    else:
        data = data[:, 0] + 1.0j * data[:, 1]

    assert data.ndim == 3
    assert data.shape[1] == data.shape[2] == 3
    return scipy.sparse.bsr_matrix(
        (data, m['col'], m['row-ptr']),
        shape=tuple(3*x for x in m['dim']),
    )
