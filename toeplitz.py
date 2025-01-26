import numpy as np
from scipy.linalg._solve_toeplitz import levinson
from collections import namedtuple


ToeplitzDeterminant = namedtuple('ToeplitzDeterminant', ['sign', 'logabsdet'])

def _prepare_toeplitz_inputs(c_or_cr):
    """Prepare inputs for Toeplitz determinant calculation."""
    if isinstance(c_or_cr, tuple):
        c, r = c_or_cr
        sym = False
    else:
        c = c_or_cr
        r = np.conj(c)
        sym = True
        
    if c[0] != r[0]:
        raise ValueError("c[0] must equal r[0] (diagonal element)")
    
    return c, r, sym

def slogdet_toeplitz(c_or_cr):
    """
    Compute sign and log absolute value of Toeplitz matrix determinant.
    """
    c, r, sym = _prepare_toeplitz_inputs(c_or_cr)
    n = len(c)
    powers = np.arange(n - 1, 0, -1)
    
    _, kf = levinson(np.concatenate((c[-2:0:-1], r[:-1])), r[1:])
    kf = kf[1:]

    if sym:
        kf_prod = np.abs(kf)**2
    else:
        _, kb = levinson(np.concatenate((r[-2:0:-1], c[:-1])), c[1:])
        kf_prod = kf * kb[1:]

    # Compute log abs determinant
    mask = np.abs(kf_prod) < 1
    log_terms = np.empty_like(kf_prod, dtype='complex')
    log_terms[mask] = np.log1p(-kf_prod[mask])
    log_terms[~mask] = np.log(kf_prod[~mask] - 1)
    
    if np.isreal(kf_prod).all():
        log_terms = log_terms.real
    
    log_det = n * np.log(np.abs(c[0])) + np.dot(powers, log_terms)

    # Compute sign
    signs = np.sign(2 * (1 > kf_prod) - 1)
    sign = np.prod(signs**powers) * np.sign(c[0])

    return ToeplitzDeterminant(sign, log_det)

def log_det_toeplitz(c_or_cr):
    """
    Compute log determinant of Toeplitz matrix.
    Only works for positive determinants.
    """
    c, r, sym = _prepare_toeplitz_inputs(c_or_cr)
    n = len(c)
    powers = np.arange(n - 1, 0, -1)
    
    _, kf = levinson(np.concatenate((c[-2:0:-1], r[:-1])), r[1:])
    
    if sym:
        log_prod = np.dot(powers, np.log1p(-np.abs(kf[1:])**2))
    else:
        _, kb = levinson(np.concatenate((r[-2:0:-1], c[:-1])), c[1:])
        log_prod = np.dot(powers, np.log1p(-kf[1:] * kb[1:]))
    
    return n * np.log(c[0]) + log_prod

def det_toeplitz(c_or_cr):
    """
    Compute determinant of Toeplitz matrix.
    May under- or over-flow.
    """
    c, r, sym = _prepare_toeplitz_inputs(c_or_cr)
    n = len(c)
    powers = np.arange(n - 1, 0, -1)
    
    _, kf = levinson(np.concatenate((c[-2:0:-1], r[:-1])), r[1:])
    
    if sym:
        k_prod = (1 - np.abs(kf[1:])**2)**powers
    else:
        _, kb = levinson(np.concatenate((r[-2:0:-1], c[:-1])), c[1:])
        k_prod = (1 - kf[1:]*kb[1:])**powers
    
    return c[0]**n * np.prod(k_prod)