import numpy as np
import pandas as pd
import logging
from scipy.stats import norm
from joblib import Parallel, delayed


def bootstrap_influence_function(inf_func, DIDparams, pl=False, cores=1):
    """
    Implements multiplier bootstrap for variance estimation.

    Parameters:
    - inf_func: Influence function matrix (NumPy array).
    - DIDparams: Dictionary containing DID estimation parameters.
    - pl (bool): Whether to use parallel processing (default: False).
    - cores (int): Number of cores for parallel processing (default: 1).

    Returns:
    - Dictionary with:
        - "bres": Bootstrapped influence function values.
        - "V": Bootstrap variance matrix.
        - "se": Bootstrap standard errors.
        - "crit_val": Critical value for uniform confidence band.
    """

    # Extract necessary parameters
    data = DIDparams["data"]
    idname = DIDparams["idname"]
    clustervars = DIDparams["clustervars"]
    biters = DIDparams["biters"]
    tname = DIDparams["tname"]
    alp = DIDparams["alp"]

    # Convert `tlist` to a sorted NumPy array
    try:
        tlist = np.sort(data[tname].unique())
    except:
        tlist = np.sort(data[tname].unique().to_numpy())

    # Get first-period dataset based on panel structure
    dta = data[data[tname] == tlist[0]] 

    # Ensure `inf_func` is a NumPy array
    inf_func = np.asarray(inf_func)

    # Number of observations (for clustering below)
    n = inf_func.shape[0]

    # Drop `idname` if it is in `clustervars`
    if clustervars and idname in clustervars:
        clustervars.remove(idname)

    # Validate `clustervars`
    if clustervars:
        if isinstance(clustervars, list) and isinstance(clustervars[0], str):
            raise ValueError("`clustervars` should be the name of the clustering variable.")
        if len(clustervars) > 1:
            raise ValueError("Cannot handle more than one cluster variable.")
        # Ensure cluster variable does not vary over time within an ID
        cluster_tv = dta.groupby(idname)[clustervars[0]].nunique() == 1
        if not cluster_tv.all():
            raise ValueError("Cannot handle time-varying cluster variables.")

    # Multiplier Bootstrap
    if not clustervars:  # No clustering
        n_clusters = n
        bres = np.sqrt(n) * run_multiplier_bootstrap(inf_func, biters, pl, cores)
    else:  # Clustered bootstrap
        cluster_var = clustervars[0]
        n_clusters = len(data[cluster_var].drop_duplicates())

        # Map cluster IDs to indices
        cluster = dta[[idname, cluster_var]].drop_duplicates().values[:, 1]
        cluster_n = dta.groupby(cluster).size().values

        # Compute cluster means for influence function
        cluster_mean_if = pd.DataFrame(inf_func).groupby(cluster).sum().values / cluster_n
        bres = np.sqrt(n_clusters) * run_multiplier_bootstrap(cluster_mean_if, biters, pl, cores)

    # Ensure `bres` is at least a 2D array
    if bres.ndim == 1:
        bres = np.expand_dims(bres, axis=0)
    elif bres.ndim > 2:
        bres = bres.transpose()

    # Remove degenerate dimensions
    valid_cols = np.logical_and(~np.isnan(np.sum(bres, axis=0)), np.sum(bres**2, axis=0) > np.sqrt(np.finfo(float).eps) * 10)
    bres = bres[:, valid_cols]

    # Bootstrap variance matrix
    V = np.cov(bres, rowvar=False) if bres.shape[1] > 1 else np.var(bres, axis=0, keepdims=True)

    # Bootstrap standard error
    q75, q25 = np.quantile(bres, [0.75, 0.25], axis=0, method="inverted_cdf")
    qnorm_75, qnorm_25 = norm.ppf(0.75), norm.ppf(0.25)
    bSigma = (q75 - q25) / (qnorm_75 - qnorm_25)

    # Critical value for uniform confidence band
    bT = np.nanmax(np.abs(bres / bSigma), axis=1)
    crit_val = np.nanquantile(bT, 1 - alp, method="inverted_cdf")

    # Compute standard errors
    se = np.full(valid_cols.shape, np.nan)
    se[valid_cols] = bSigma / np.sqrt(n_clusters)

    return {
        "bres": bres,
        "V": V,
        "se": se,
        "cval": crit_val
    }



def multiplier_bootstrap(inf_func, biters):
    """
    Performs multiplier bootstrap using Rademacher weights.

    Parameters:
    - inf_func: Influence function matrix (n x k).
    - biters: Number of bootstrap iterations.

    Returns:
    - Bootstrapped influence function values (biters x k).
    """
    n, K = inf_func.shape
    biters = int(biters)
    innerMat = np.zeros((n, K))
    Ub = np.zeros(n)
    outMat = np.zeros((biters,K))

    for b in range(biters):
        # Draw Rademacher weights (±1 with equal probability)
        Ub = np.random.choice([1, -1], size=(n, 1))

        # Apply weights to influence function
        innerMat = inf_func * Ub

        # Compute mean for each bootstrap iteration
        outMat[b] = np.mean(innerMat, axis=0)

    return outMat

def run_multiplier_bootstrap(inf_func, biters, pl=False, cores=1):
    """
    Wrapper function to run multiplier bootstrap with optional parallelization.

    Parameters:
    - inf_func: Influence function matrix (NumPy array).
    - biters: Number of bootstrap iterations.
    - pl: Boolean, whether to use parallel computation (default: False).
    - cores: Number of cores to use for parallelization (default: 1).

    Returns:
    - Bootstrapped influence function values (biters x k).
    """
    n = inf_func.shape[0]

    # If parallel processing is enabled and data is large enough
    if pl and cores > 1 and n > 2500:
        # Split the iterations across cores
        chunk_size = biters // cores
        chunk_sizes = [chunk_size] * cores
        chunk_sizes[0] += biters - sum(chunk_sizes)  # Adjust to match biters exactly

        # Run parallel bootstrap
        results = Parallel(n_jobs=cores)(
            delayed(multiplier_bootstrap)(inf_func, chunk) for chunk in chunk_sizes
        )
        
        # Combine results
        results = np.vstack(results)

    else:
        # Run bootstrap normally
        results = multiplier_bootstrap(inf_func, biters)

    return results
