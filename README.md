# det-toeplitz
Example code for computing the determinant of a Toeplitz matrix with Levinson recursion + SciPy. Sharing this because I haven't seen similar code elsewhere (perhaps for good reason), but I don't recommend using it.

This isn't the fastest but it's better than forming the full Toeplitz matrix and using `linalg.det()`. It leverages the undocumented `levinson()` function in `scipy.linalg._solve_toeplitz` for the reflection coefficients so it should be faster than pure Python. For moderately sized matrices I've gotten > 10x better performance vs `det()`.

# Usage

All functions follow the [SciPy `solve_toeplitz()` function](https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.solve_toeplitz.html) API and accept either the first column of a symmetric Toeplitz matrix or a tuple of the first column and first row for the non-symmetric case.

* `slogdet_toeplitz()`: sign + log determinant (positive and negative determinants)
* `log_det_toeplitz()`: log determinant (positive determinants only)
* `det_toeplitz()`: determinant (may under- or over-flow)

Some accommodation is made for complex inputs/outputs but this hasn't been tested much.

# Example

You can compute the loglikelihood of a multivariate Gaussian give it's autocovariance (i.e. the first column of the covariance matrix) like this:

```python
import numpy as np
from toeplitz import slogdet_toeplitz, log_det_toeplitz
from scipy.linalg import solve_toeplitz


def loglikelihood(y, auto_cov):
    """Compute log likelihood of Gaussian samples y with auto-covariance auto_cov and zero mean"""

    # compute quadratic term using efficient form for toeplitz matrix
    maha_dist_sq = solve_toeplitz(auto_cov, y).squeeze() @ y

    # compute log determinant using efficient form for toeplitz matrix
    ln_det = log_det_toeplitz(auto_cov)       

    # compute log likelihood
    loglike = -0.5 * (ln_det + maha_dist_sq + len(y) * np.log(2 * np.pi))

    return loglike
```

Note that using the FFT + PCG instead of `solve_toeplitz()` would be faster.
