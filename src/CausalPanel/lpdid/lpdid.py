import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy.stats import norm
from tabulate import tabulate
import matplotlib.pyplot as plt

def lpdid(df, y, time, unit, treat, pre, post, reweight=False, composition_correction=False):
    """
    Local Projections Difference-in-Differences (LP-DiD) estimation with proper filtering conditions.

    Parameters:
    df (pd.DataFrame): The dataset used in the analysis.
    y (str): The dependent variable (outcome).
    time (str): The column representing time.
    unit (str): The column representing unit ID.
    treat (str): The column indicating treatment status (0 before treatment, 1 after treatment).
    pre (int): The number of pre-treatment periods.
    post (int): The number of post-treatment periods.
    reweight (bool): If True, applies custom reweighting; otherwise, defaults to weight = 1.
    composition_correction (bool): If True, removes later-treated observations from control.

    Returns:
    dict: A dictionary including coefficient table, observation counts, and regression results.
    """
    pre_window, post_window = -pre, post

    # Ensure df is a copy before modifications
    df = df[[y, time, unit, treat]].copy()

    # Convert unit to integer IDs if not already numeric
    if not pd.api.types.is_integer_dtype(df[unit]):
        df[unit], unit_mapping = pd.factorize(df[unit])
    else:
        unit_mapping = None

    df[time] = df[time].astype(int)

    # Identify first treatment period per unit
    treated_df = df[df[treat] > 0].copy()
    treat_dates = treated_df.groupby(unit)[time].min().reset_index()
    treat_dates.columns = [unit, 'treat_date']

    # Merge treatment dates
    df = df.merge(treat_dates, on=unit, how='left')
    df['rel_time'] = df[time] - df['treat_date']
    
    # Define treatment indicator and first difference variable
    df['treat_indicator'] = (df['rel_time'] >= 0).astype(int)
    df['treat_diff'] = df['treat_indicator'].diff().fillna(0).clip(lower=0)

    # Set default weights to 1 if reweighting is not used
    if not reweight:
        df['reweight_0'] = df['reweight_use'] = 1

    # Initialize storage for results
    betas = []
    SEs = []
    nobs = []
    reg_results = []
    
    # Compute backward differences (pre-treatment effects)
    for j in range(abs(pre_window), 0, -1):  # Includes pre1
        df[f'Dm{j}y'] = df.groupby(unit)[y].shift(j) - df.groupby(unit)[y].shift(1)  # Compute lag differences

        # Define regression formula
        formula = f'Dm{j}y ~ treat_diff'
        
        # Apply filtering conditions
        lim = df[f'Dm{j}y'].notna() & df['treat_diff'].notna() & df[treat].notna()
        
        if composition_correction:
            max_t = df[time].max()
            lim = lim & ((df['treat_diff'] == 1) | (df.groupby(unit)[treat].shift(-post_window).fillna(0) == 0)) & \
                  (df['treat_date'].isna() | (df['treat_date'] < max_t - post_window))
        else:
            lim = lim & ((df['treat_diff'] == 1) | (df[treat] == 0))

        if lim.sum() > 0:
            weights = df.loc[lim, 'reweight_use'].values if reweight else None
            model = smf.ols(formula, data=df[lim]).fit(
                cov_type="cluster", cov_kwds={"groups": df.loc[lim, unit].values}, weights=weights
            )
            betas.append(model.params['treat_diff'])
            SEs.append(model.bse['treat_diff'])
            nobs.append(model.nobs)
            reg_results.append(model)
        else:
            betas.append(np.nan)
            SEs.append(np.nan)
            nobs.append(np.nan)
            reg_results.append(None)

    # Compute forward differences (post-treatment effects)
    for j in range(0, post_window + 1):
        df[f'D{j}y'] = df.groupby(unit)[y].shift(-j) - df.groupby(unit)[y].shift(1)  # Compute lead differences

        # Define regression formula
        formula = f'D{j}y ~ treat_diff'
        
        # Apply filtering conditions
        lead_treat = df.groupby(unit)[treat].shift(-j)
        lim = df[f'D{j}y'].notna() & df['treat_diff'].notna() & lead_treat.notna()
        
        if composition_correction:
            max_t = df[time].max()
            lim = lim & ((df['treat_diff'] == 1) | (df.groupby(unit)[treat].shift(-post_window).fillna(0) == 0)) & \
                  (df['treat_date'].isna() | (df['treat_date'] < max_t - post_window))
        else:
            lim = lim & ((df['treat_diff'] == 1) | (lead_treat.fillna(0) == 0))

        if lim.sum() > 0:
            weights = df.loc[lim, 'reweight_use'].values if reweight else None
            model = smf.ols(formula, data=df[lim]).fit(
                cov_type="cluster", cov_kwds={"groups": df.loc[lim, unit].values}, weights=weights
            )
            betas.append(model.params['treat_diff'])
            SEs.append(model.bse['treat_diff'])
            nobs.append(model.nobs)
            reg_results.append(model)
        else:
            betas.append(np.nan)
            SEs.append(np.nan)
            nobs.append(np.nan)
            reg_results.append(None)

    # Construct output dataframe
    coeftable = pd.DataFrame({
        'window': list(range(pre_window, post_window + 1)),
        'Estimate': betas,
        'Std. Error': SEs,
        'nobs': nobs
    })

    coeftable['t value'] = coeftable['Estimate'] / coeftable['Std. Error']
    coeftable['P>|t|'] = norm.sf(abs(coeftable['t value']))
    coeftable['CI Lower'] = coeftable['Estimate'] - 1.96 * coeftable['Std. Error']
    coeftable['CI Upper'] = coeftable['Estimate'] + 1.96 * coeftable['Std. Error']

    # Formatting for output
    coeftable['E-time'] = coeftable['window'].apply(lambda x: f'pre{abs(x)}' if x < 0 else (f'tau{x}' if x > 0 else 'tau0'))
    coeftable = coeftable.sort_values(by='window').reset_index(drop=True)
    coeftable = coeftable[['E-time', 'Estimate', 'Std. Error', 't value', 'P>|t|', 'CI Lower', 'CI Upper', 'nobs']]

    # Print formatted table
    print("\nLP-DiD Event Study Estimates\n")
    print(tabulate(coeftable, headers='keys', tablefmt='outline', floatfmt=".6f"))

    return {
        'coeftable': coeftable,
        'reg_results': reg_results,
        'unit_mapping': dict(enumerate(unit_mapping)) if unit_mapping is not None else None
    }


def plot_lpdid(reg, conf=0.95, segments=True, add=False,
               xlab="Time to Treatment", ylab="Coefficient Estimate and 95% Confidence Interval",
               ylim=None, main="", x_shift=0,
               point_size=5, color="black", opacity=1):
    """
    Plot LP-DiD Event Study Estimates
    
    Parameters:
    reg (pd.DataFrame): DataFrame containing LP-DiD results from lpdid function.
    conf (float): Confidence level (default: 0.95).
    segments (bool): Whether to show confidence intervals (default: True).
    add (bool): Whether to add to an existing figure (default: False).
    xlab (str): X-axis label.
    ylab (str): Y-axis label.
    ylim (tuple): Limits for the Y-axis (default: auto).
    main (str): Title of the plot.
    x_shift (float): Shift in x-axis values.
    point_size (int): Size of the points.
    color (str): Color of points and confidence intervals.
    opacity (float): Opacity of points and confidence intervals.

    Returns:
    None: Displays the plot.
    """

    # Extract relevant values
    coeftable = reg.copy()
    coeftable["t"] = coeftable["E-time"].str.replace("pre", "-").str.replace("tau", "").astype(int)
    
    conf_z = norm.ppf(1 - (1 - conf) / 2)
    coeftable["uCI"] = coeftable["Estimate"] + conf_z * coeftable["Std. Error"]
    coeftable["lCI"] = coeftable["Estimate"] - conf_z * coeftable["Std. Error"]

    # Determine y-axis limits
    if ylim is None:
        ylim = (min(coeftable["lCI"]), max(coeftable["uCI"]))

    # Create the plot
    if not add:
        plt.figure(figsize=(8, 5))
        plt.scatter(coeftable["t"] + x_shift, coeftable["Estimate"], s=point_size * 10,
                    color=color, alpha=opacity, label="Estimate")
        plt.axhline(y=0, color="black", linestyle="dashed")
        plt.axvline(x=-1, color="black", linestyle="dashed")

        # Labels and title
        plt.xlabel(xlab)
        plt.ylabel(ylab)
        plt.title(main)
        plt.ylim(ylim)
    
    # Add confidence intervals as vertical segments
    if segments:
        for i, row in coeftable.iterrows():
            plt.plot([row["t"] + x_shift, row["t"] + x_shift], [row["uCI"], row["lCI"]],
                     color=color, alpha=opacity, linewidth=1)

    plt.show()  # Display the plot