from typing import *

T = TypeVar('T')

def zip_eq(*seqs: Tuple[Sequence, ...]) -> List[Tuple]:
    """
    zip alternative designed for use on finite-length sequences, not iterators.

    This means it can **actually check the length.**
    """
    seqs = list(seqs)
    expect_all_same((len(s) for s in seqs),
                    'cannot zip 0 lists!',
                    'mismatched lengths')
    return list(zip(*seqs))

def dict_unzip_eq(d: Dict[Any, Sequence]) -> List[Dict]:
    """ Turns a dict of sequences into a list of dicts. """
    keys = d.keys()

    out = []
    for tup in zip_eq(*d.values()):
        out.append(dict(zip(keys, tup)))
    return out

def dict_zip_eq(*ds: Dict) -> Dict[Any, List]:
    """ Turns many dicts with matching keys into a dict of lists. """
    if not ds:
        raise TypeError('cannot zip 0 dicts!')

    keys = expect_all_same((d.keys() for d in ds),
                           'cannot zip 0 dicts!',
                           'mismatched key sets')

    out = { k: [] for k in keys }
    for d in ds:
        for (k, x) in d.items():
            out[k].append(x)

    return out

def expect_all_same(it: Iterable[T],
                    msg0: str = 'empty sequence',
                    msg2: str = 'conflicting values',
                    ) -> T:
    it = iter(it)

    try:
        x0 = next(it)
    except StopIteration:
        raise TypeError(msg0)

    for x in it:
        if x != x0:
            raise ValueError(f'{msg2}:\n A: {repr(x0)}\n B: {repr(x)}')

    return x0
