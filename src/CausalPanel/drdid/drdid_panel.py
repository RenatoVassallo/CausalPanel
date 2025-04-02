import numpy as np
import statsmodels.api as sm

def drdid_panel(y1, y0, D, covariates=None, i_weights=None, inffunc=True, verbose=False):
    """
    Locally Efficient Doubly Robust DiD estimator with panel data.

    Parameters:
    - y1, y0: Outcome variables (arrays) at post- and pre-treatment
    - D: Binary treatment indicator (1 = treated, 0 = control)
    - covariates: Covariate matrix (optional)
    - i_weights: Observation weights (optional)
    - inffunc: If True, returns the influence function
    - verbose: If True, prints debug info

    Returns:
    - dr_att: Estimated ATT
    - dr_att_inf_func: Influence function of ATT (or None if inffunc=False)
    """
    # Flatten and convert inputs
    y1 = np.asarray(y1).flatten()
    y0 = np.asarray(y0).flatten()
    D = np.asarray(D).flatten()
    deltaY = y1 - y0
    n = len(D)

    # Handle covariates and add intercept
    if covariates is None:
        int_cov = np.ones((n, 1))
    else:
        covariates = np.asarray(covariates)
        if np.all(covariates[:, 0] == 1):
            int_cov = covariates
        else:
            int_cov = np.column_stack((np.ones(n), covariates))

    # Handle and normalize weights
    if i_weights is None:
        i_weights = np.ones(n)
    else:
        i_weights = np.asarray(i_weights).flatten()
        if np.any(i_weights < 0):
            raise ValueError("Weights must be non-negative.")
    i_weights = i_weights / np.mean(i_weights)

    # Estimate Propensity Score using weighted logistic regression
    pscore_model = sm.GLM(D, int_cov, family=sm.families.Binomial(), freq_weights=i_weights)
    pscore_results = pscore_model.fit()
    if not pscore_results.converged:
        print("⚠️ Warning: Logistic regression did not converge.")
    if np.any(np.isnan(pscore_results.params)):
        raise ValueError("Propensity score model has NaN coefficients (possible multicollinearity).")

    ps_fit = np.clip(pscore_results.predict(), 1e-16, 1 - 1e-16)

    # Outcome regression on control group (WLS)
    control_mask = (D == 0)
    reg_model = sm.WLS(deltaY[control_mask], int_cov[control_mask], weights=i_weights[control_mask])
    reg_results = reg_model.fit()
    if np.any(np.isnan(reg_results.params)):
        raise ValueError("Outcome regression model has NaN coefficients (possible multicollinearity).")

    out_delta = np.dot(int_cov, reg_results.params)

    # Compute ATT
    w_treat = i_weights * D
    w_cont = i_weights * ps_fit * (1 - D) / (1 - ps_fit)

    dr_att_treat = w_treat * (deltaY - out_delta)
    dr_att_cont = w_cont * (deltaY - out_delta)

    eta_treat = np.mean(dr_att_treat) / np.mean(w_treat)
    eta_cont = np.mean(dr_att_cont) / np.mean(w_cont)

    dr_att = eta_treat - eta_cont

    if not inffunc:
        return dr_att, None

    # Compute influence function components
    weights_ols = i_weights * (1 - D)
    wols_x = weights_ols[:, None] * int_cov
    wols_eX = weights_ols[:, None] * (deltaY - out_delta)[:, None] * int_cov

    try:
        XpX_inv = np.linalg.inv(np.dot(wols_x.T, int_cov) / n)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix in outcome regression (try removing collinear covariates).")

    asy_lin_rep_wols = np.dot(wols_eX, XpX_inv)

    score_ps = i_weights[:, None] * (D - ps_fit)[:, None] * int_cov
    Hessian_ps = pscore_results.cov_params() * n  # GLM gives cov matrix of MLE
    asy_lin_rep_ps = np.dot(score_ps, Hessian_ps)

    # Decompose influence function
    inf_treat_1 = dr_att_treat - w_treat * eta_treat
    M1 = np.mean(w_treat[:, None] * int_cov, axis=0)
    inf_treat_2 = np.dot(asy_lin_rep_wols, M1)
    inf_treat = (inf_treat_1 - inf_treat_2) / np.mean(w_treat)

    inf_cont_1 = dr_att_cont - w_cont * eta_cont
    M2 = np.mean(w_cont[:, None] * (deltaY - out_delta - eta_cont)[:, None] * int_cov, axis=0)
    inf_cont_2 = np.dot(asy_lin_rep_ps, M2)
    M3 = np.mean(w_cont[:, None] * int_cov, axis=0)
    inf_cont_3 = np.dot(asy_lin_rep_wols, M3)
    inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / np.mean(w_cont)

    dr_att_inf_func = inf_treat - inf_control
    se = np.std(dr_att_inf_func, ddof=1) / np.sqrt(n)

    if verbose:
        print(f"ATT: {dr_att:.6f} ± {1.96 * se:.6f} (SE: {se:.6f})")

    return dr_att, dr_att_inf_func
    
