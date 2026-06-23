"""Module to compute parameter uncertainties for the log-likelihood funtion.

This module adapts the parameter uncertainty estimation method used in lmfit
for a non-linear least squares fit to a log-likelihood fit."""
import warnings
import numpy as np
from scipy.linalg import LinAlgError, inv

# check for numdifftools
try:
    import numdifftools as ndt
    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False

"""
Compute the parameter uncertainties of the log-likelihood function.

The covariance matrix is calculated as the inverse of a numerically
approximated Hessian. The covariance is used to create the
parameter uncertainties.

Args:
    result: fitting result from lmfit minimize. The result is assumed
        to contain the optimised parameters in a params attribute.
    fcn: User function. This function must have the signature::
        fcn(result.params, *args)
    args: Optional positional arguments to pass to `fcn`.
"""
def compute_uncertainties(result, fcn, args=None):
    if not HAS_NUMDIFFTOOLS:
        return
    if args is None:
      args = []

    # Wrap the input function to f(x) for computation of the covariance
    def fun(x):
        # copy x values to the params
        for name, val in zip(result.var_names, x):
            result.params[name].value = float(val)
        return np.sum(fcn(result.params, *args))

    # Extract the parameters to an array
    x = np.array([result.params[name].value for name in result.var_names])

    covar = _calculate_covariance_matrix(fun, x)
    result.covar = covar

    # restore original values
    for i, name in enumerate(result.var_names):
        result.params[name].value = float(x[i])

    if covar is not None:
        # Adapted from lmfit.minimizer.Minimizer._calculate_uncertainties_correlations
        # Changes have been made to:
        # - reduce checks as the diagonal of the covariance matrix is known to be positive.
        # - pre-compute stderr for all params
        # - compute correlations for all params
        result.errorbars = True

        # pre-compute standard errors
        for ivar, name in enumerate(result.var_names):
            par = result.params[name]
            par.stderr = float(np.sqrt(covar[ivar, ivar]))
            par.correl = {}

        # Compute all correlations
        for ivar, name in enumerate(result.var_names):
            par = result.params[name]
            result.errorbars = result.errorbars and (par.stderr > 0.0)
            for jvar, varn2 in enumerate(result.var_names):
                if jvar != ivar:
                    try:
                        par.correl[varn2] = float(covar[ivar, jvar] /
                                                  (par.stderr * result.params[name].stderr))
                    except ZeroDivisionError:
                        result.errorbars = False
        if result.errorbars:
            if result.uvars is not None:
              print('Parameters:\n' +
                '\n'.join(f'  {k}={repr(v)}' for k, v in result.uvars.items()))

            result.uvars = result.params.create_uvars(covar=result.covar)


def _calculate_covariance_matrix(fun, x):
    """Calculate the covariance matrix.

    The ``numdiftoools`` package is used to estimate the Hessian
    matrix, and the covariance matrix is calculated as the inverse
    of the Hessian. This is valid for log-likelihood functions.

    Args:
        fun: Function accepting an array of parameters.
        x: Parameters.

    Returns:
        Covariance matrix if successful, otherwise None.
    """
    # Adapted from lmfit.minimizer.Minimizer._calculate_covariance_matrix
    # Changes have been made to accept the function to estimate (which was
    # originally a member of the Minimizer class).
    warnings.filterwarnings(action="ignore", module="scipy",
                            message="^internal gelsd")

    try:
        Hfun = ndt.Hessian(fun, step=1.e-4)
        hessian_ndt = Hfun(x)
        cov_x = inv(hessian_ndt)

        if cov_x.diagonal().min() < 0:
            # we know the calculated covariance is incorrect, so we set the covariance to None
            cov_x = None
    except (LinAlgError, ValueError):
        cov_x = None

    return cov_x
