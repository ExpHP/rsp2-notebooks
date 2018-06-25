
import scipy.sparse.linalg as spla
import numpy as np
import logging as log
import sys
import math
import random

__all__ = [
    'try_shift_invert',
    'collect_by_shift_invert',
    'eigsh_T',
    'eigenvalue',
    'wavenumber',
]
# The default tolerance for eigsh is machine precision, which I feel is
# overkill. Hopefully a lighter tolerance will save some time.
#
# TODO maybe: experiment with this
TOL = 1e-10

# absolute cosines greater than this are deemed non-orthogonal
OVERLAP_THRESH = 1e-6

SQRT_EIGENVALUE_TO_THZ = 15.6333043006705
THZ_TO_WAVENUMBER = 33.3564095198152
SQRT_EIGENVALUE_TO_WAVENUMBER = SQRT_EIGENVALUE_TO_THZ * THZ_TO_WAVENUMBER

iife = lambda f: f()

class Struct(): pass

def eigsh_T(*args,
            allow_fewer=False,
            **kw):
    try:
        evals, evecs = spla.eigsh(*args, **kw)
        return evals, evecs.T

    except spla.eigen.arpack.ArpackNoConvergence as e:
        if allow_fewer:
            return e.eigenvalues, e.eigenvectors.T
        else:
            raise


# a tree of counts based on direct field assignment so that static
# linters can catch typos. (CLion handles it very impressively!)
class Count:
    def total(self): return int(self)
    def __int__(self): return sum(map(int, self.__dict__.values()))


# method originally used by rsp2 to search for negative modes.
# The idea is to repeatedly call shift-invert with a small maxiter
# and sigma=0.  This often produces incorrect solutions (and thus we must
# carefully vet the output of eigsh), but sometimes reveals some legitimate
# bad modes faster than non-shift-invert mode would.
def try_shift_invert(m,
                     sigma,
                     shift_invert_attempts,
                     tol=0,
                     OPinv=None,
                     M=None,
                     **kw):

    # A heavy computational step at the beginning of shift-invert mode is
    # factorizing the matrix; do that ahead of time.
    log.debug('precomputing OPinv')
    if OPinv is None:
        OPinv = get_OPinv(m, M=M, sigma=sigma, tol=tol)

    found_evals = []
    found_evecs = []

    # debug info
    counts = []

    for call_i in range(shift_invert_attempts):
        log.debug('shift-invert call', call_i + 1)
        (evals, evecs) = eigsh_T(
            m,
            sigma=sigma,
            OPinv=OPinv,
            M=M,
            allow_fewer=True,
            **kw
            # TODO: play around with ncv.
            #       Larger ncv is slower, but what do we get in return?
        )
        evecs = np.array(list(map(normalize, evecs)))

        count = Count() # total solutions found
        count.good = 0 # total solutions kept
        count.bad = Count() # total solutions rejected
        count.bad.repeat = 0 # linearly dependent with prior solutions
        count.bad.wrong = 0  # non-eigenvector solutions
        count.bad.ortho_bad = 0 # tried to orthogonalize, got a non-eigenvector
        count.bad.ortho_fail = 0 # tried to orthogonalize, and failed

        for (eval, ev) in zip(evals, evecs):
            # Is it ACTUALLY an eigenvector?
            # (solutions produced by shift-invert at sigma=0 often fail this.)
            if not is_valid_esol(m, eval, ev, tol=tol):
                count.bad.wrong += 1
                continue

            # Linearly dependent with existing solutions?
            if sum(np.abs(np.vdot(ev, other))**2 for other in found_evecs) > 0.95:
                count.bad.repeat += 1
                continue

            # Prepare it for possible insertion.
            ortho_ev = mgs_step(ev, found_evecs)

            # We didn't ruin it, did we?
            if not is_valid_esol(m, eval, ortho_ev, tol=tol):
                count.bad.ortho_bad += 1
                continue

            if sum(np.abs(np.vdot(ortho_ev, other))**2 for other in found_evecs) > 1e-6:
                count.bad.ortho_fail += 1
                continue

            # ship it
            count.good += 1
            found_evecs.append(ortho_ev)
            found_evals.append(eval)

        counts.append(count)

    log.debug(" Good -- Bad (Old Wrong OrthoFail OrthoBad)")
    for count in counts:
        log.debug(
            " {:^4} -- {:^3} ({:^3} {:^5} {:^9} {:^8})".format(
                count.good,
                count.bad.total(),
                count.bad.repeat,
                count.bad.wrong,
                count.bad.ortho_fail,
                count.bad.ortho_bad,
            ),
        )

    perm = np.argsort(found_evals)
    evals = np.array(found_evals)[perm]
    evecs = np.array(found_evecs)[perm]
    for val, v in zip(evals, evecs):
        assert is_valid_esol(m, val, v, tol=tol)
    for i in range(len(evecs)):
        for k in range(i):
            assert not is_overlapping(evecs[i], evecs[k])
    return evals, evecs

def eigenvalue(freqs): return np.sign(freqs) * np.square(freqs / SQRT_EIGENVALUE_TO_WAVENUMBER)
def wavenumber(evals): return np.sign(evals) * np.sqrt(evals) * SQRT_EIGENVALUE_TO_WAVENUMBER

def collect_by_shift_invert(m,
                            *,
                            group_size,
                            wavenumber_step,
                            wavenumber_stop,
                            maxiter,
                            max_substeps,
                            callback=None,
                            ev_history_steps=1,
                            start=eigenvalue(1e-2),
                            tol=0,
                            **kw):
    """
    :param m:
    :param start:
        Sigma for the first step.
    :param group_size:
        Diagonalize this many per step. This needs to be tuned for
        optimal speed; if set too high you will suffer from linear memory costs
        (each step only retains vectors from the current and prior step)
        and quadratic time costs (due to orthogonalization); if set too low, you
        will suffer from many expensive shift-inverse matrix factorizations.
        (one is performed each step)
    :param wavenumber_step:
        Each step, advance the starting point of the search space by this amount,
        permanently cutting off all solutions below.
    :param callback:
        A function to be called on the new esols discovered each step (in the
        tuple form produced by the other functions in this module).
        It should return a list the length of the esols.
    :param ev_history_steps:
        Remember eigenvectors from up to this many steps ago, for the purpose
        of detecting repeat solutions.
    :param max_substeps:
        Number of times to repeat running eigsh at a frequency before moving on.
    :param maxiter:
        eigsh parameter.  You should set it explicitly to something MUCH MUCH
        MUCH smaller than scipy's default, because shift-invert at nonzero sigma
        usually converges faster than lightning, and you don't want to be stuck
        waiting for hundreds of thousands of iterations for the final few runs
        at the top of the spectrum.

        As more eigensolutions are requested (``group_size``), fewer iterations
        may be necessary. (FIXME: why? is this due to the fact that scipy's
        default number of Lanczos vectors increases with ``k``?)

        (on a system of 2524 atoms, I've seen shift-invert mode gather 40
        eigensolutions in 3 iters, and 100 solutions in 1 iter. Contrast with
        the default maxiter of 25240!)
    :param tol:
        eigsh parameter
    :param kw:
        eigsh parameters.
    :return:
        ``evals``, or ``(evals, func_outputs)`` if a callback was supplied
    """

    # # Search ahead of time for an estimate on max eigenvalue.
    # # (or more specifically, a number that's "just a bit less")
    # @iife
    # def upper_eval_guesstimate():
    #     (large_evals, _) = eigsh_T(m,
    #                                maxiter=10 * np.sqrt(m.shape[0]),
    #                                allow_fewer=True,
    #                                k=40,
    #                                which='LA',
    #                                **kw)
    #     maximum = max(large_evals)
    #     return maximum - (maximum - min(large_evals)) * 1e-2

    # An orthonormal basis of vectors, each stored with the number of steps
    # since the last time it has had a sizable overlap with a step's eigsh
    # output. Vectors are evicted when this number hits the history cap.
    #
    # ---
    #
    # This might sound overly complicated at first. One might ask, "why not
    # just store a list of lists; one for each recent step?" And I can only
    # ask in response: what would you put in these lists so that the
    # orthogonalization process is numerically stable?
    #
    # Motivation for the chosen layout can be found even in the simplest case
    # where `ev_history_size == 1`. Suppose that on step 1 we find `a` and `b`,
    # and on step 2 we find `a + b`. On step 3, then, what vectors from the
    # subspace spanned by `a` and `b` ought to be used to orthogonalize new
    # vectors?
    #
    # Well, for usage of `mgs_step` to be numerically stable, the set of
    # basis kets we orthogonalize new kets against must always be itself
    # orthogonal, produced incrementally through successive calls to the
    # function.  This is easily true if we keep the original `a` and `b`, but
    # if we were to somehow replace them with `a + b` then I couldn't be so
    # certain!
    #
    # This is the solution that naturally arises from aiming to preserve
    # the original `a` and `b` for as long as they are relevant.
    history_basis = []

    found_evals = []
    found_func_out = []

    sigma = start

    log.debug(" Good -- Bad (Old Wrong OrthoBad LowFreq)")
    def log_count(count):
        log.debug(
            " {:^4} -- {:^3} ({:^3} {:^5} {:^8} {:^7})".format(
                count.good,
                count.bad.total(),
                count.bad.repeat,
                count.bad.wrong,
                count.bad.ortho_bad,
                count.bad.low_freq,
            ),
        )

    _log_time = None
    def time_log(s):
        nonlocal _log_time
        import time
        if _log_time:
            log.debug("finished in {}".format(time.time() - _log_time))
        _log_time = time.time()
        log.debug(s)

    time_log = lambda _s: None # disable

    while True:
        time_log("precomputing OPinv")
        OPinv = get_OPinv(m, M=None, sigma=sigma, tol=tol)

        # All new solutions added to the eigenbasis this step.
        new_good_evals = []
        new_good_evecs = []

        # For each ket in the history, contains the total overlap (probability)
        # against all kets computed across all substeps of this step.
        #
        # When it is too small, the ket ages towards eviction.
        #
        # (generally a float from 0 to max_substeps)
        history_overlap_probs = np.zeros((len(history_basis),))

        time_log("diagonalizing")
        for _ in range(max_substeps):
            (substep_evals, substep_evecs) = eigsh_T(
                m,
                k=group_size,
                sigma=sigma,
                allow_fewer=True,
                maxiter=maxiter,
                # shift invert with mode='normal' and which='LA' will quite reliably
                # produce the eigenvalues that are just above sigma.
                which='LA',
                tol=tol,
                OPinv=OPinv,
                **kw
            )
            substep_evecs = np.array(list(map(normalize, substep_evecs)))

            # (iifes to limit scope and catch stupid-but-deadly naming bugs)
            @iife
            def _inspection_loop():
                time_log("inspecting")
                nonlocal new_good_evals
                nonlocal new_good_evecs
                nonlocal history_overlap_probs

                count = Count()
                count.good = 0
                count.bad = Count()
                count.bad.wrong = 0
                count.bad.repeat = 0
                count.bad.low_freq = 0
                count.bad.ortho_bad = 0

                # Simply throw away stuff below sigma. It's quite possible
                # that it is linearly dependent with vectors that we have
                # long since dropped from the history.
                substep_valid_esols = (substep_evals, substep_evecs)
                substep_valid_esols = list(zip(*[
                    (eval, evec)
                    for (eval, evec) in zip(*substep_valid_esols)
                    if eval >= sigma
                ])) or [[], []]

                count.bad.low_freq += len(substep_evals) - len(substep_valid_esols[0])

                # Some results may be garbage (if so, usually more than one...)
                substep_valid_esols = list(zip(*[
                    (eval, evec)
                    for (eval, evec) in zip(*substep_valid_esols)
                    if is_valid_esol(m, eval, evec, tol=tol)
                ])) or [[], []]
                count.bad.wrong += len(substep_evals) - len(substep_valid_esols[0])
                count.bad.wrong -= count.bad.low_freq

                for (eval, evec) in zip(*substep_valid_esols):
                    # Prepare it for possible insertion.
                    evec, overlap_probs = mgs_step(evec,
                                                   (h.evec for h in history_basis),
                                                   return_overlaps=True)

                    # Record overlaps to ensure we don't evict the original kets
                    # that were degenerate with anything discovered this round.
                    history_overlap_probs += overlap_probs

                    # Linearly dependent with existing solutions?
                    if sum(overlap_probs) > 0.95:
                        # Too much overlap, the result might be garbage.
                        count.bad.repeat += 1
                        continue

                    # We didn't ruin it, did we?
                    if not is_valid_esol(m, eval, evec, tol=tol):
                        count.bad.ortho_bad += 1
                        continue

                    # It's a good one.
                    # Log it for calling the callback later.
                    count.good += 1
                    new_good_evals.append(eval)
                    new_good_evecs.append(evec)

                    # Put it in the history pronto to preserve the correctness
                    # of mgs_step.
                    hist = Struct()
                    hist.evec = evec
                    hist.age = 0
                    history_basis.append(hist)

                    # The new vector obviously overlaps with the eigensols
                    # we computed this substep, so start it out with prob 1.0.
                    # (this ensures it will not be deleted this step)
                    history_overlap_probs = np.concatenate((history_overlap_probs, [1.0]))

                log_count(count)

        time_log("updating")
        @iife
        def _update_history():
            nonlocal history_basis

            for i in range(len(history_basis)):
                history_basis[i].age += 1

            for (i, overlap) in enumerate(history_overlap_probs):
                if abs(overlap)**2 > 1e-3:
                    history_basis[i].age = 0

            delete_where(
                history_basis,
                [h.age >= ev_history_steps for h in history_basis],
            )

        sigma = eigenvalue(wavenumber(sigma) + wavenumber_step)
        if wavenumber(sigma) >= wavenumber_stop:
            break

        if new_good_evals:
            found_evals.extend(new_good_evals)
            if callback:
                time_log("calling callback")
                found_func_out.extend(callback((new_good_evals, new_good_evecs)))


    perm = np.argsort(found_evals)
    evals = np.array(found_evals)[perm]
    if callback:
        return evals, permute_python_list(found_func_out, perm)
    else:
        return evals

def log_uniform(min_value, max_value):
    """
    Draw a real number in the specified range from a distribution that
    is uniform in log space.
    """
    from math import log, exp
    from random import uniform
    return exp(uniform(log(min_value), log(max_value)))

def permute_python_list(lst, perm):
    # blocks ndarray dimension inference
    class Opaque:
        def __init__(self, value):
            self.value = value

    lst = list(map(Opaque, lst))
    permuted = np.array(lst)[perm]
    return [x.value for x in permuted]

def delete_where(lst, mask):
    assert isinstance(lst, list)
    indices, = np.where(list(mask))
    for i in reversed(indices.tolist()):
        del lst[i]

# precompute OPinv for faster repeated shift-invert calls
def get_OPinv(A, M, sigma, tol=0):
    # FIXME usage of scipy implementation detail
    matvec = spla.eigen.arpack.get_OPinv_matvec(A, M=M, symmetric=True, sigma=sigma, tol=tol)
    return spla.LinearOperator(A.shape, matvec=matvec, dtype=A.dtype)

def mgs_step(a, b_hats, return_overlaps=False):
    """
    This is the function such that

    >>> def mgs(original_vecs):
    >>>     out = []
    >>>     for vec in original_vecs:
    >>>         out.append(mgs_step(vec, out))
    >>>     return out

    is a correct implementation of Modified Gram Schmidt method.
    """
    overlaps = []
    for b_hat in b_hats:
        ab = par(a, b_hat)
        a -= ab
        if return_overlaps:
            overlaps.append(np.vdot(ab, ab))

    if return_overlaps:
        return normalize(a), overlaps
    else:
        return normalize(a)

# The part of `a` that points along `b_hat`.
def par(a, b_hat):
    return np.vdot(b_hat, a) * b_hat

def acousticness(v_hat):
    sum = np.reshape(v_hat, (-1, 3)).sum(axis=0)
    return abs(np.vdot(sum, sum))

def normalize(v):
    return v / np.sqrt(np.vdot(v, v))

def is_overlapping(a_hat, b_hat, thresh=OVERLAP_THRESH):
    return abs(np.vdot(a_hat, b_hat)) > thresh

def is_valid_esol(m, eval, evec, tol): # (tol should be the one given to eigsh)
    tol = tol or sys.float_info.epsilon

    assert abs(abs(np.vdot(evec, evec)) - 1) < 1e-12
    return lazy_any([
        lambda: acousticness(evec) > 1. - 1e-3,
        lambda: lazy_all([
            lambda: abs(abs(np.vdot(normalize(m @ evec), evec)) - 1.0) < 1e-2,
            lambda: (np.abs(m @ evec - eval * evec) < tol * 10**5).all(),
        ])
    ])

def lazy_any(it): return any(pred() for pred in it)
def lazy_all(it): return all(pred() for pred in it)
