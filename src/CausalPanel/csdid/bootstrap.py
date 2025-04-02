import numpy as np
import pandas as pd
import logging
from scipy.stats import norm
from joblib import Parallel, delayed

def bootstrap_influence_function(inf_func, DIDparams, pl=False, cores=1):
    """
    Implements multiplier bootstrap for variance estimation.

    Parameters:
    - inf_func: Influence function matrix (NumPy array of shape n x k)
    - DIDparams: Dictionary with DiD config and data
    - pl: Enable parallel processing (bool)
    - cores: Number of cores (int)

    Returns:
    - dict with:
        - bres: Bootstrap influence draws
        - V: Bootstrap variance-covariance matrix
        - se: Standard errors
        - crit_val: Uniform confidence band critical value
    """
    # Unpack settings
    data = DIDparams['data']
    idname = DIDparams['idname']
    clustervars = DIDparams['clustervars']
    biters = DIDparams['biters']
    tname = DIDparams['tname']
    alp = DIDparams['alp']

     # Convert tlist to a sorted NumPy array
    try:
        tlist = np.sort(data[tname].unique())
    except:
        tlist = np.sort(data[tname].unique().to_numpy())

    # Get first-period dataset based on panel structure
    dta = data[data[tname] == tlist[0]] 

    # Convert IF to array and count units
    inf_func = np.asarray(inf_func)
    n = inf_func.shape[0]

    # Drop idname from clustervars if needed
    if clustervars and idname in clustervars:
        clustervars = [v for v in clustervars if v != idname]

    # Validate clustering
    if clustervars:
        if isinstance(clustervars, list) and isinstance(clustervars[0], str):
            raise ValueError("clustervars should be a variable name, not a list of strings.")
        if len(clustervars) > 1:
            raise ValueError("Cannot handle more than one cluster variable.")
        cluster_var = clustervars[0]

        # Check time-invariance
        cluster_tv = dta.groupby(idname)[cluster_var].nunique() == 1
        if not cluster_tv.all():
            raise ValueError("Cluster variable must not vary over time within units.")
    
    # Run multiplier bootstrap
    if not clustervars:
        n_clusters = n
        bres = np.sqrt(n) * run_multiplier_bootstrap(inf_func, biters, pl, cores)
    else:
        cluster = dta[[idname, cluster_var]].drop_duplicates().values[:, 1]
        cluster_sizes = dta.groupby(cluster).size().values
        n_clusters = len(np.unique(cluster))

        cluster_mean_if = pd.DataFrame(inf_func).groupby(cluster).sum().values / cluster_sizes[:, None]
        bres = np.sqrt(n_clusters) * run_multiplier_bootstrap(cluster_mean_if, biters, pl, cores)

    # Ensure shape consistency
    if bres.ndim == 1:
        bres = np.expand_dims(bres, axis=0)
    elif bres.ndim > 2:
        bres = bres.transpose()

    # Filter out degenerate columns
    ndg_mask = np.logical_and(~np.isnan(np.sum(bres, axis=0)),
                              np.sum(bres**2, axis=0) > np.sqrt(np.finfo(float).eps) * 10)
    bres = bres[:, ndg_mask]

    # Compute variance-covariance matrix
    V = np.cov(bres, rowvar=False) if bres.shape[1] > 1 else np.var(bres, axis=0, keepdims=True)

    # Quantile-based SE estimate
    q75, q25 = np.quantile(bres, [0.75, 0.25], axis=0, method="inverted_cdf")
    qnorm75, qnorm25 = norm.ppf(0.75), norm.ppf(0.25)
    bSigma = (q75 - q25) / (qnorm75 - qnorm25)

    # Uniform critical value
    bT = np.nanmax(np.abs(bres / bSigma), axis=1)
    bT = bT[np.isfinite(bT)]
    cval = np.quantile(bT, 1 - alp, method="inverted_cdf")

    # Final standard errors
    se = np.full(ndg_mask.shape, np.nan)
    se[ndg_mask] = bSigma / np.sqrt(n_clusters)

    return {
        "bres": bres,
        "V": V,
        "se": se,
        "cval": cval
    }


def multiplier_bootstrap(inf_func, biters):
    """
    Performs multiplier bootstrap using Rademacher weights.

    Parameters:
    - inf_func: Influence function array (n,) or (n, k)
    - biters: Number of bootstrap iterations

    Returns:
    - Bootstrapped means (biters x k)
    """
    inf_func = np.atleast_2d(inf_func)
    n, k = inf_func.shape
    outMat = np.zeros((biters, k))

    for b in range(biters):
        # Rademacher weights: ±1
        weights = np.random.choice([1, -1], size=(n, 1))
        # Apply to each column of influence function
        boot_sample = inf_func * weights
        outMat[b] = np.mean(boot_sample, axis=0)

    return outMat


def run_multiplier_bootstrap(inf_func, biters, pl=False, cores=1):
    """
    Wrapper to run multiplier bootstrap, with optional parallel execution.

    Parameters:
    - inf_func: Influence function array (n,) or (n, k)
    - biters: Number of bootstrap replications
    - pl: Use parallel processing (default: False)
    - cores: Number of parallel cores (default: 1)

    Returns:
    - Bootstrapped ATT estimates (biters x k)
    """
    inf_func = np.atleast_2d(inf_func)
    n = inf_func.shape[0]

    if pl and cores > 1 and n > 2500:
        # Distribute biters across cores
        base = biters // cores
        chunk_sizes = [base] * cores
        chunk_sizes[0] += biters - sum(chunk_sizes)  # balance remainder

        def parallel_function(b):
            return multiplier_bootstrap(inf_func, b)

        results = Parallel(n_jobs=cores)(
            delayed(parallel_function)(chunk) for chunk in chunk_sizes
        )
        return np.vstack(results)

    else:
        return multiplier_bootstrap(inf_func, biters)