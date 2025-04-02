import numpy as np
import statsmodels.api as sm

def drdid_panel(y1, y0, D, covariates=None, i_weights=None, inffunc=False):
    """
    Locally Efficient Doubly Robust DiD estimator with panel data.

    Parameters:
    - y1 (array-like): Outcome variable at post-treatment period.
    - y0 (array-like): Outcome variable at pre-treatment period.
    - D (array-like): Binary treatment indicator (1 = treated, 0 = control).
    - covariates (array-like, optional): Covariate matrix for propensity score and outcome regression.
    - i_weights (array-like, optional): Observation weights.
    - inffunc (bool, optional): If True, returns the influence function.

    Returns:
    - dr_att: Estimated average treatment effect on the treated (ATT).
    - dr_att_inf_func: Influence function of ATT (if inffunc=True, else None).
    """
    # Convert inputs to numpy arrays
    y1, y0, D = map(np.asarray, (y1, y0, D))
    deltaY = y1 - y0
    n = len(D)

    # Handle covariates
    if covariates is None:
        int_cov = np.ones((n, 1))
    else:
        covariates = np.asarray(covariates)
        if np.all(covariates[:, 0] == 1):
            int_cov = covariates  # Already an intercept-only model
        else:
            int_cov = np.column_stack((np.ones(n), covariates))  # Add intercept
            
    # Handle weights
    if i_weights is None:
        i_weights = np.ones(n)
    else:
        i_weights = np.asarray(i_weights)
        if np.any(i_weights < 0):
            raise ValueError("Weights must be non-negative.")
        
    # Normalize weights
    i_weights = i_weights / np.mean(i_weights)

    # Estimate propensity scores using weighted logistic regression
    pscore_model = sm.GLM(D, int_cov, family=sm.families.Binomial(), freq_weights=i_weights)
    pscore_results = pscore_model.fit()

    if not pscore_results.converged:
        print("Warning: Logistic regression did not converge.")

    if np.any(np.isnan(pscore_results.params)):
        raise ValueError("Propensity score model coefficients contain NaNs. Check for multicollinearity.")

    ps_fit = pscore_results.predict()
    ps_fit = np.minimum(ps_fit, 1 - 1e-16)  # Ensure no division by zero

    # Compute outcome regression for control group using weighted OLS
    control_mask = (D == 0)
    reg_model = sm.WLS(deltaY[control_mask], int_cov[control_mask], weights=i_weights[control_mask])
    reg_results = reg_model.fit()

    if np.any(np.isnan(reg_results.params)):
        raise ValueError("Outcome regression model coefficients contain NaNs. Check for multicollinearity.")

    out_delta = np.dot(int_cov, reg_results.params)

    # Compute DR ATT estimator
    w_treat = i_weights * D
    w_cont = i_weights * ps_fit * (1 - D) / (1 - ps_fit)

    dr_att_treat = w_treat * (deltaY - out_delta)
    dr_att_cont = w_cont * (deltaY - out_delta)

    eta_treat = np.mean(dr_att_treat) / np.mean(w_treat)
    eta_cont = np.mean(dr_att_cont) / np.mean(w_cont)

    dr_att = eta_treat - eta_cont

    # Compute influence function
    weights_ols = i_weights * (1 - D)
    wols_x = weights_ols[:, np.newaxis] * int_cov
    wols_eX = weights_ols[:, np.newaxis] * (deltaY - out_delta)[:, np.newaxis] * int_cov

    try:
        XpX_inv = np.linalg.inv(np.dot(wols_x.T, int_cov) / n)
    except np.linalg.LinAlgError:
        raise ValueError("Singular matrix in regression model. Consider removing collinear covariates.")

    asy_lin_rep_wols = np.dot(wols_eX, XpX_inv)

    score_ps = i_weights[:, np.newaxis] * (D - ps_fit)[:, np.newaxis] * int_cov
    Hessian_ps = pscore_results.cov_params() * n  # Hessian from model
    asy_lin_rep_ps = np.dot(score_ps, Hessian_ps)

    inf_treat_1 = dr_att_treat - w_treat * eta_treat
    M1 = np.mean(w_treat[:, np.newaxis] * int_cov, axis=0)
    inf_treat_2 = np.dot(asy_lin_rep_wols, M1)
    inf_treat = (inf_treat_1 - inf_treat_2) / np.mean(w_treat)

    inf_cont_1 = dr_att_cont - w_cont * eta_cont
    M2 = np.mean(w_cont[:, np.newaxis] * (deltaY - out_delta - eta_cont)[:, np.newaxis] * int_cov, axis=0)
    inf_cont_2 = np.dot(asy_lin_rep_ps, M2)
    M3 = np.mean(w_cont[:, np.newaxis] * int_cov, axis=0)
    inf_cont_3 = np.dot(asy_lin_rep_wols, M3)
    inf_control = (inf_cont_1 + inf_cont_2 - inf_cont_3) / np.mean(w_cont)

    dr_att_inf_func = inf_treat - inf_control

    # Ensure standard error is computed exactly as in the original function
    se = np.std(dr_att_inf_func, ddof=1) / np.sqrt(n)

    # Return influence function only if requested
    if not inffunc:
        dr_att_inf_func = None

    return dr_att, dr_att_inf_func
    
