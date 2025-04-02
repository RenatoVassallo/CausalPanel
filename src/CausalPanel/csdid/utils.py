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
'''
def wif(keepers, pg, weights_ind, G, group):
    """
    Compute the influence function for weights.

    Parameters:
    - keepers (list): Indices of ATT(g,t) values to include.
    - pg (ndarray): Probability of each group.
    - weights_ind (ndarray): Individual weights.
    - G (ndarray): Group membership for each observation.
    - group (ndarray): Vector of group values.

    Returns:
    - Influence function matrix (n x k).
    """

    def bool_mask(cond):
        """Custom function to handle boolean conditions with NaNs."""
        cond = np.asarray(cond, dtype=bool)  # Ensure boolean dtype
        cond[np.isnan(cond)] = False  # Convert NaNs to False
        return cond

    # Convert inputs to NumPy arrays
    pg = np.asarray(pg)
    group = np.asarray(group)
    weights_ind = np.asarray(weights_ind)
    G = np.asarray(G).flatten()
    keepers = np.asarray(keepers, dtype=int)  # Ensure `keepers` is an integer array

    # Ensure indexing is valid
    if np.any(keepers >= len(group)):
        raise ValueError(f"`keepers` indices ({keepers}) exceed the length of `group` ({len(group)}).")

    # Compute effect of estimating weights in the numerator
    if1 = np.array([
        (weights_ind * bool_mask(G == group[k])) - pg[k]
        for k in keepers
    ]).T / np.sum(pg[keepers])  # Transpose ensures (n x k)

    # Compute effect of estimating weights in the denominator
    if2_base = np.array([
        (weights_ind * bool_mask(G == group[k])) - pg[k]
        for k in keepers
    ])  # Shape: (k, n)

    if2_sum = np.sum(if2_base, axis=0)  # Sum over `keepers` → Shape: (n,)
    multiplier = pg[keepers] / (np.sum(pg[keepers]) ** 2)  # Shape: (k,)
    if2 = np.outer(if2_sum, multiplier)  # Shape: (n, k)

    # Compute final weight influence function
    wif_output = if1 - if2  # Shape: (n x k)

    # Debug shapes
    print(f"\n✔️ wif() Output Shapes:")
    print(f"  wif_output shape: {wif_output.shape}")  # Expected (n, k)

    return wif_output
'''

def wif(keepers, pg, weights_ind, G, group):
    
    def bool_mask(cond):
        """Custom function to handle boolean conditions with NaNs."""
        cond = np.asarray(cond, dtype=bool)  # Ensure boolean dtype
        cond[np.isnan(cond)] = False  # Convert NaNs to False
        return cond
    # note: weights are all of the form P(G=g|cond)/sum_cond(P(G=g|cond))
    # this is equal to P(G=g)/sum_cond(P(G=g)) which simplifies things here
    pg = np.array(pg)
    group = np.array(group)
    
    # effect of estimating weights in the numerator
    if1 = np.empty((len(weights_ind), len(keepers)))
    for i, k  in enumerate(keepers):
        numerator = (weights_ind * 1 * bool_mask(G == group[k])) - pg[k]
        # denominator = sum(np.array(pg)[keepers]) )[:, None]  
        denominator = np.sum(pg[keepers])

        result = numerator[:, None]  / denominator
        if1[:, i] = result.squeeze()
    
    # effect of estimating weights in the denominator
    if2 = np.empty((len(weights_ind), len(keepers)))
    for i, k  in enumerate(keepers):
        numerator = ( weights_ind * 1 * bool_mask(G == group[k]) ) - pg[k]
        # result = numerator.to_numpy()[:, None]  @ multipler[:, None].T
        if2[:, i] = numerator.squeeze()
    if2 = np.sum(if2, axis=1)    
    multiplier = ( pg[keepers] / sum( pg[keepers] ) ** 2 )   
    if2 = np.outer( if2 , multiplier)

    # if1 = [((weights_ind * 1*TorF(G==group[k])) - pg[k]) / sum(pg[keepers]) for k in keepers]
    # if2 = np.dot(np.array([weights_ind*1*TorF(G==group[k]) - pg[k] for k in keepers]).T, pg[keepers]/(sum(pg[keepers])**2))
    wif_factor = if1 - if2
    # Debug shapes
    print(f"\n✔️ wif() Output Shapes:")
    print(f"  wif_output shape: {wif_factor.shape}")  # Expected (n, k)
    return wif_factor


def get_agg_inf_func(att, inffunc, whichones, weights_agg, wif=None):
    print("\n🔍 get_agg_inf_func() STARTED")
    print("  inffunc shape: ", inffunc.shape)
    print("  whichones: ", whichones)
    print("  weights_agg shape: ", np.shape(weights_agg))

    # Check if selected columns are all zeros
    for i in whichones:
        if np.all(inffunc[:, i] == 0):
            print(f"🚨 Column {i} of `inffunc` is all zeros! This may be the cause of zero influence.")

    # Ensure weights_agg is properly formatted
    weights_agg = np.asarray(weights_agg).reshape(-1, 1)

    # Compute weighted influence function
    thisinffunc = np.dot(inffunc[:, whichones], weights_agg)
    print("✅ After weight multiplication, thisinffunc shape: ", thisinffunc.shape)

    # If thisinffunc is still all zeros, warn
    if np.all(thisinffunc == 0):
        print("⚠ Warning: `thisinffunc` is all zeros after weight multiplication!")

    # Incorporate wif
    if wif is not None:
        print("  wif shape: ", wif.shape)
        print("  att[whichones] shape: ", np.array(att[whichones]).shape)

        att_selected = np.array(att[whichones]).reshape(-1, 1)

        # Check for NaNs in wif
        print("⚠ NaNs in wif:", np.isnan(wif).sum(), "out of", wif.size)

        if wif.shape[1] != att_selected.shape[0]:
            print(f"🚨 Shape Mismatch: wif {wif.shape}, att_selected {att_selected.shape}")
            raise ValueError("`wif` and `att_selected` must have compatible shapes for matrix multiplication.")

        wif_term = np.dot(wif, att_selected)

        # Check NaNs in wif_term
        if np.isnan(wif_term).any():
            print("⚠ Warning: `wif_term` contains NaNs! Handling it...")
            wif_term = np.nan_to_num(wif_term, nan=0.0)

        thisinffunc = thisinffunc + wif_term

    print("✅ Final thisinffunc shape: ", thisinffunc.shape)
    print("\nFinal thisinffunc: ", thisinffunc)

    if np.isnan(thisinffunc).any():
        print("🚨 Warning: NaN values detected in final `thisinffunc` output!")

    return thisinffunc



def get_se(thisinffunc, DIDparams=None):
    """
    Compute standard errors from an influence function.

    Parameters:
    - thisinffunc (array-like): Influence function values.
    - DIDparams (dict, optional): Dictionary containing bootstrap settings.

    Returns:
    - Standard error (scalar).
    """

    # Convert to NumPy array and remove extra dimensions
    thisinffunc = np.asarray(thisinffunc).squeeze()

    # 🛑 If `thisinffunc` is empty, return NaN
    if thisinffunc.size == 0:
        print("⚠ Warning: `thisinffunc` is empty! Returning NaN.")
        return np.nan

    # 🛑 If all values are NaN, return NaN
    if np.isnan(thisinffunc).all():
        print("⚠ Warning: `thisinffunc` contains only NaNs! Returning NaN.")
        return np.nan

    # Default parameters
    alp = 0.05
    bstrap = False
    n = len(thisinffunc)  # Sample size

    if DIDparams is not None:
        bstrap = DIDparams.get("bstrap", False)
        alp = DIDparams.get("alp", 0.05)

    if bstrap:
        bout = bootstrap_influence_function(thisinffunc, DIDparams)
        return bout["se"]
    else:
        return np.sqrt(np.nanmean(thisinffunc ** 2) / n)  # Use `nanmean` to ignore NaNs
    
        