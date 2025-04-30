import pandas as pd
import numpy as np
import warnings
from scipy.stats import chi2, norm
from .bootstrap import bootstrap_influence_function

def get_wide_data(data: pd.DataFrame, yname, idname, tname):
    """
    Converts long-format panel data into wide format with pre- and post-treatment outcomes.
    """
    if len(data[tname].unique()) != 2:
        raise ValueError("get_wide_data only works for exactly 2 periods of panel data.")

    # Sort by ID and time period
    data = data.sort_values([idname, tname])

    # Compute pre- and post-treatment outcomes
    data[".y1"] = data.groupby(idname)[yname].shift(-1)  # Future outcome
    data[".y0"] = data[yname]  # Baseline outcome
    data[".dy"] = data[".y1"] - data[".y0"]  # Outcome difference

    # Drop rows with missing values
    return data.dropna()


def compute_chi_square_p_value(W, q):
    """ Computes the chi-square p-value for the Wald test. """
    return 1 - chi2.cdf(W, q)

def compute_standard_cval(alpha):
    """ Computes the critical value from a standard normal distribution. """
    return norm.ppf(1 - alpha / 2)

def compute_critical_value(alpha, bres):
    """
    Computes the critical value for uniform confidence bands using bootstrap.
    This is required when `cband` is set to True.
    """
    if bres is None or bres.shape[0] == 0:
        return compute_standard_cval(alpha)  # Default to standard critical value

    # Compute bootstrap standard deviation estimates using interquartile range (IQR)
    bSigma = np.apply_along_axis(
        lambda b: (np.percentile(b, 75) - np.percentile(b, 25)) / (norm.ppf(0.75) - norm.ppf(0.25)),
        axis=0,
        arr=bres
    )

    # Replace very small values with NaN to avoid numerical issues
    bSigma[bSigma <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    # Compute the max-t statistic
    bT = np.apply_along_axis(
        lambda b: np.nanmax(np.abs(b / bSigma)) if not np.all(np.isnan(b / bSigma)) else np.nan,
        axis=1,
        arr=bres
    )

    cval = np.nanpercentile(bT, 100 * (1 - alpha))

    # Warn if critical value is unreasonably large
    if cval >= 7:
        warnings.warn(
            "Simultaneous critical value is too large to be reliable. "
            "This usually happens when the number of observations per group is small "
            "and/or there is little variation in outcomes."
        )

    return cval
  

# ================================================
# Functions for getting standard errors for aggte
# ================================================

def wif(keepers, pg, weights_ind, G, group):
    def TorF(cond):
        """Custom truth-test function handling NaNs."""
        cond = np.asarray(cond)
        mask = np.zeros_like(cond, dtype=bool)
        mask[~np.isnan(cond)] = cond[~np.isnan(cond)].astype(bool)
        return mask

    pg = np.array(pg)
    group = np.array(group)

    # Effect of estimating weights in the numerator
    if1 = np.empty((len(weights_ind), len(keepers)))
    for i, k in enumerate(keepers):
        mask = TorF(G == group[k])
        numerator = (weights_ind * mask) - pg[k]
        denominator = np.sum(pg[keepers])
        result = numerator[:, None] / denominator
        if1[:, i] = result.squeeze()

    # Effect of estimating weights in the denominator
    if2 = np.empty((len(weights_ind), len(keepers)))
    for i, k in enumerate(keepers):
        mask = TorF(G == group[k])
        numerator = (weights_ind * mask) - pg[k]
        if2[:, i] = numerator.squeeze()

    if2 = np.sum(if2, axis=1)
    multiplier = pg[keepers] / (np.sum(pg[keepers]) ** 2)
    if2 = np.outer(if2, multiplier)

    # Final WIF factor
    wif_factor = if1 - if2
    return wif_factor


def get_agg_inf_func(att, inffunc, whichones, weights_agg, wif=None):
    # Ensure weights are in array form
    weights_agg = np.asarray(weights_agg)

    # Multiply influence function by weights and sum over selected columns
    thisinffunc = np.dot(inffunc[:, whichones], weights_agg)

    # Add influence function of the weights, if provided
    if wif is not None:
        att_selected = np.array(att[whichones])
        thisinffunc = thisinffunc + np.dot(wif, att_selected)

    return thisinffunc

    
def get_se(thisinffunc, DIDparams=None):
    #thisinffunc = np.asarray(thisinffunc).squeeze()
    alpha = 0.05
    bstrap = False
    if DIDparams is not None:
        bstrap = DIDparams['bstrap']
        alpha = DIDparams['alp']
        cband = DIDparams['cband']
        n = len(thisinffunc)

    if bstrap:
        bout = bootstrap_influence_function(thisinffunc, DIDparams)
        return bout['se']
    else:
        return np.sqrt(np.mean((thisinffunc)**2) / n)
        