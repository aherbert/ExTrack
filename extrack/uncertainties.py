"""Module to compute parameter uncertainties for the log-likelihood funtion.

This module adapts the parameter uncertainty estimation method used in lmfit
for a non-linear least squares fit to a log-likelihood fit."""
import warnings
import numpy as np
from scipy.linalg import LinAlgError, inv
from scipy.differentiate import hessian

# check for numdifftools
try:
    import numdifftools as ndt
    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False

"""
Compute the parameter uncertainties of the minimised negative log-likelihood function.

The covariance matrix is calculated as the inverse of a numerically
approximated Hessian. The covariance is used to create the
parameter uncertainties.

Args:
    result: fitting result from lmfit minimize. The result is assumed
        to contain the optimised parameters in a params attribute.
    fcn: User function. This function must have the signature::
        fcn(result.params, *args)
    args: Optional positional arguments to pass to `fcn`.
    cb: User callback function invoked when gradient estimation discovered a new optimum.
        This function must have the signature: cb(params, log_likelihood).
    kwargs: Keyword options to the function computing the Hessian.
"""
def compute_uncertainties(result, fcn, args=None, cb=None, **kwargs):
    if args is None:
        args = []

    # Wrap the input function to f(x) for computation of the covariance.
    # Store the min of the negative log-likelihood function.
    minv = np.inf
    optv = np.inf
    xx = np.array([])
    def fun(x):
        # copy x values to the params
        for name, val in zip(result.var_names, x):
            result.params[name].value = float(val)
        v = np.sum(fcn(result.params, *args))
        nonlocal minv
        nonlocal optv
        nonlocal xx
        # Store optimal value from first evaluation
        if optv == np.inf:
            optv = v
        if minv > v:
            minv = v
            xx = x
        return v

    # Extract the parameters to an array
    x = np.array([result.params[name].value for name in result.var_names])

    covar = _calculate_covariance_matrix(fun, x, **kwargs)
    result.covar = covar

    # warn for non-optimal value
    if not np.array_equal(x, xx):
        print(f"WARNING: computing uncertainties of non-optimal log likelihood function: {-optv} -> {-minv}")
        for i, name in enumerate(result.var_names):
            dx = xx[i] - x[i]
            if dx:
                print(f"  {name}={xx[i]}  delta={dx}  ({(dx / x[i])})")
            else:
                print(f"  {name}={xx[i]}")
        if cb is not None:
            # Populate values
            for i, name in enumerate(result.var_names):
                result.params[name].value = float(xx[i])
            cb(result.params, -minv)

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
            result.uvars = result.params.create_uvars(covar=result.covar)


def _calculate_covariance_matrix(fun, x, step=1e-4, rel_step=False, num_steps=1,
    richardson_terms=2,
    dd_method=0, order=8, maxiter=10, rtol=None, verbose=0, forward=False):
    """Calculate the covariance matrix.

    Use a numerical estimation of the Hessian
    matrix, and the covariance matrix is calculated as the inverse
    of the Hessian. This is valid for log-likelihood functions.

    Args:
        fun: Function accepting an array of parameters.
        x: Parameters.
        step: Step for the numerical differentiation.
        rel_step: Use relative step size.
        num_steps: Number of steps for differentiation (numdifftools).
        richardson_terms: Number of terms used in the Richardson extrapolation (numdifftools).
        dd_method: 0: numdifftools; 1: scipy.differentiate.hessian.
        order: order of the finite difference formula to be used (scipy hessian).
        maxiter: maximum iterations (scipy hessian).
        rtol: relative tolerance (scipy hessian).
        verbose: Verbosity.
        forward: Use forward differences; default is central (numdifftools).

    Returns:
        Covariance matrix if successful, otherwise None.
    """
    if dd_method == 1:
        # use scipy.differentiate.hessian
        print(f"calculate_covariance_matrix: {x}. {step} rel={rel_step} order={order} maxiter={maxiter} rtol={rtol}")
        # The function is repeatedly called with the same array value
        # so cache the results.
        cache = {}
        def ff(x):
            b = x.tobytes()
            if (v := cache.get(b)) is not None:
                return v
            v = fun(x)
            cache[b] = v
            return v
        # vectorized for fun(x_m) to f(m, ...) -> (...)
        def f(y):
            s = y.shape
            # Iterate over arrays of size m
            y = y.T.reshape((-1, s[0]))
            a = np.array([ff(yy) for yy in y])
            return a.reshape(tuple(reversed(s))[0:-1]).T
        # XXX: This can result in a broadcast error.
        # It is a bug in scipy.
        if rel_step:
            step = step*x
        tolerances = {}
        if rtol is not None:
            tolerances['rtol'] = rtol
        res = hessian(f, x, initial_step=step, order=order, maxiter=maxiter, tolerances=tolerances)
        if verbose or not np.all(res.success):
            print('status', list(res.status))
            print('hessian', list(res.ddf))
            print('error', list(res.error))
        h = res.ddf
    else:
        # use numdifftools
        if not HAS_NUMDIFFTOOLS:
            return None

        # Adapted from lmfit.minimizer.Minimizer._calculate_covariance_matrix
        # Changes have been made to accept the function to estimate (which was
        # originally a member of the Minimizer class).
        warnings.filterwarnings(action="ignore", module="scipy",
                                message="^internal gelsd")

        print(f"calculate_covariance_matrix: {x}. {step} rel={rel_step} num={num_steps} forward={forward} richardson_terms={richardson_terms}")
        if rel_step:
            step = step*x
        method = 'forward' if forward else 'central'
        Hfun = ndt.Hessian(fun,
          step=ndt.step_generators.MaxStepGenerator(base_step=step, num_steps=num_steps),
          richardson_terms=richardson_terms,
          method=method)
        h = Hfun(x)

    try:
        cov_x = inv(h)
        if cov_x.diagonal().min() < 0:
            print(f"bad covariance: {cov_x.diagonal()}")
            # we know the calculated covariance is incorrect, so we set the covariance to None
            cov_x = None
    except (LinAlgError, ValueError) as e:
        print("bad hessian^-1", e)
        cov_x = None

    return cov_x
