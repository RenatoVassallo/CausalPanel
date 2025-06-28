import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
from scipy.stats import norm
from tabulate import tabulate
from IPython.display import display
from .plot_lpdid import plot_lpdid

class LPDID:
    def __init__(self, df: pd.DataFrame, y: str, treat: str, time: str, unit: str, pre: int, post: int,
                 lags: int = 0, reweight: bool = False, control_group: str = "notyet", clean_control_L: int = 4,
                 absorbing: bool = True):
        self.df = df.copy()
        self.y = y
        self.treat = treat
        self.time = time
        self.unit = unit
        self.pre = pre
        self.post = post
        self.lags = lags
        self.reweight = reweight
        self.control_group = control_group
        self.clean_control_L = clean_control_L
        self.absorbing = absorbing
        self.results = None

        self._prepare_data()

    def _prepare_data(self):
        df = self.df[[self.y, self.time, self.unit, self.treat]].copy()
        df[self.time] = df[self.time].astype(int)
        df = df.sort_values([self.unit, self.time])

        if not pd.api.types.is_integer_dtype(df[self.unit]):
            df[self.unit], _ = pd.factorize(df[self.unit])

        # Treat timing
        treated_df = df[df[self.treat] > 0]
        treat_dates = treated_df.groupby(self.unit)[self.time].min().reset_index()
        treat_dates.columns = [self.unit, 'treat_date']
        df = df.merge(treat_dates, on=self.unit, how='left')

        # Relative timing
        df['rel_time'] = df[self.time] - df['treat_date']
        df['treat_indicator'] = (df['rel_time'] >= 0).astype(int)
        df['treat_diff'] = df['treat_indicator'].diff().fillna(0).clip(lower=0)
        df['reweight_use'] = 1

        df['D_treat'] = df.groupby(self.unit)[self.treat].diff().fillna(0).abs()
        df['past_events'] = df.groupby(self.unit)['D_treat'].cumsum().fillna(0)

        # Clean control sample logic
        self._generate_clean_controls(df)
        self.df = df

    def _generate_clean_controls(self, df):
        if self.absorbing:
            for h in range(0, self.post + 1):
                df[f"CCS_{h}"] = ((df["D_treat"] == 1) | (df.groupby(self.unit)[self.treat].shift(-h).fillna(0) == 0)).astype(int)
            for h in range(1, self.pre + 1):
                df[f"CCS_m{h}"] = df["CCS_0"]
        else:
            df["CCS_0"] = 0
            cond = (df["D_treat"].isin([0, 1])) & (df.groupby(self.unit)["D_treat"].shift(1).fillna(0).abs() != 1)
            for k in range(2, self.clean_control_L + 1):
                cond &= (df.groupby(self.unit)["D_treat"].shift(k).fillna(0).abs() != 1)
            df.loc[cond, "CCS_0"] = 1

            for h in range(1, self.post + 1):
                df[f"CCS_{h}"] = 0
                df.loc[(df[f"CCS_{h - 1}"] == 1) &
                       (df.groupby(self.unit)["D_treat"].shift(-h).fillna(0).abs() != 1), f"CCS_{h}"] = 1

            df["CCS_m1"] = df["CCS_0"]
            for h in range(2, self.pre + 1):
                df[f"CCS_m{h}"] = 0
                df.loc[(df[f"CCS_m{h - 1}"] == 1) &
                       (df.groupby(self.unit)[f"CCS_m{h - 1}"].shift(1).fillna(0) == 1), f"CCS_m{h}"] = 1

        if self.control_group == "never":
            df['max_treat'] = df.groupby(self.unit)[self.treat].transform('max')
            df['never_treated'] = (df['max_treat'] == 0).astype(int)
            for h in range(0, self.post + 1):
                df.loc[(df['D_treat'] == 0) & (df['never_treated'] == 0), f"CCS_{h}"] = 0
            for h in range(2, self.pre + 1):
                df.loc[(df['D_treat'] == 0) & (df['never_treated'] == 0), f"CCS_m{h}"] = 0
            df.drop(columns=['max_treat'], inplace=True)

        elif self.control_group == "notyet":
            df['first_obs'] = df.groupby(self.unit)[self.time].transform('min')
            df['status_entry_help'] = df[self.treat].where(df[self.time] == df['first_obs'])
            df['status_entry'] = df.groupby(self.unit)['status_entry_help'].transform('max')

            for j in range(0, self.post + 1):
                df[f'nyt_{j}'] = 0
                df.loc[(df.groupby(self.unit)['past_events'].shift(-j).fillna(0) == 0) &
                       (df['status_entry'] == 0), f'nyt_{j}'] = 1

            for j in range(0, self.post + 1):
                df.loc[(df['D_treat'] == 0) & (df[f'nyt_{j}'] == 0), f'CCS_{j}'] = 0
            for j in range(2, self.pre + 1):
                df.loc[(df['D_treat'] == 0) & (df['nyt_0'] == 0), f'CCS_m{j}'] = 0

    def _run_regression(self, dep_var: str, shift: int, is_post: bool):
        """
        Run OLS regression with optional lag controls.
        """
        df = self.df.copy()

        df[dep_var] = (
            df.groupby(self.unit)[self.y]
            .shift(-shift if is_post else shift) -
            df.groupby(self.unit)[self.y].shift(1)
        )

        # Base formula
        formula = f'{dep_var} ~ treat_diff'

        # Add lagged outcomes to formula
        for l in range(1, self.lags + 1):
            lag_col = f'{self.y}_lag{l}'
            df[lag_col] = df.groupby(self.unit)[self.y].shift(l)
            formula += f' + {lag_col}'

        # Treatment condition logic (same as before)
        lead_treat = df.groupby(self.unit)[self.treat].shift(-shift if is_post else shift)
        lim = df[dep_var].notna() & df['treat_diff'].notna() & lead_treat.notna()

        # Apply sample restrictions (same logic as before)
        if self.control_group == "never":
            max_t = df[self.time].max()
            lim &= (
                (df['treat_diff'] == 1) |
                (df.groupby(self.unit)[self.treat].shift(-self.post).fillna(0) == 0)
            ) & (
                df['treat_date'].isna() | (df['treat_date'] < max_t - self.post)
            )
        else:
            if self.absorbing:
                treated_cond = (df['treat_diff'] == 1) & (df.get('CCS_0', 1) == 1)
                control_cond = (lead_treat.fillna(0) == 0)
            else:
                treated_cond = (
                    (df['treat_diff'] == 1) &
                    (df.get(f'CCS_{self.post}', 0) == 1) &
                    (df.get(f'CCS_m{self.pre}', 0) == 1)
                )
                control_cond = (
                    (lead_treat.fillna(0) == 0) &
                    (df.get(f'nyt_{self.post}', 0) == 1)
                )
            lim &= treated_cond | control_cond

        if lim.sum() > 0:
            df_lim = df.loc[lim].copy()

            # Drop rows with missing lags
            if self.lags:
                lagged_vars = [f'{self.y}_lag{l}' for i in range(1, self.lags + 1)]
                df_lim = df_lim.dropna(subset=lagged_vars)

            # Make sure group and weights are aligned with df_lim
            groups = df_lim[self.unit].to_numpy()

            if self.reweight:
                weights = df_lim['reweight_use'].to_numpy()
            else:
                weights = None

            assert len(groups) == len(df_lim), "Mismatch between groups and df_lim"
            if weights is not None:
                assert len(weights) == len(df_lim), "Mismatch between weights and df_lim"

            model = smf.ols(formula, data=df_lim).fit(
                cov_type="cluster",
                cov_kwds={"groups": groups},
                weights=weights
            )
            return model.params['treat_diff'], model.bse['treat_diff'], model.nobs
        else:
            return np.nan, np.nan, np.nan

    def _summary(self, results: pd.DataFrame) -> pd.DataFrame:
        """
        Summarize the LP-DiD regression results with statistical significance information.
        """
        coeftable = results.copy()
        coeftable['t value'] = coeftable['Estimate'] / coeftable['Std. Error']
        coeftable['P>|t|'] = 2 * norm.sf(abs(coeftable['t value']))  # Two-tailed
        coeftable['CI Lower'] = coeftable['Estimate'] - 1.96 * coeftable['Std. Error']
        coeftable['CI Upper'] = coeftable['Estimate'] + 1.96 * coeftable['Std. Error']
        coeftable['E-time'] = coeftable['window'].apply(
            lambda x: f'pre{abs(x)}' if x < 0 else (f'tau{x}' if x > 0 else 'tau0')
        )

        coeftable = coeftable[['E-time', 'Estimate', 'Std. Error', 't value', 'P>|t|',
                            'CI Lower', 'CI Upper', 'nobs']]

        print("\nLP-DiD Event Study Estimates\n")
        print(tabulate(coeftable, headers='keys', tablefmt='outline', floatfmt=".6f"))
        
        return coeftable

    def fit(self):
        """
        Fit the LP-DiD model and store event-study results internally.
        """
        betas, SEs, nobs = [], [], []

        for j in range(self.pre, 0, -1):
            b, se, n = self._run_regression(f'Dm{j}y', j, is_post=False)
            betas.append(b)
            SEs.append(se)
            nobs.append(n)

        for j in range(0, self.post + 1):
            b, se, n = self._run_regression(f'D{j}y', j, is_post=True)
            betas.append(b)
            SEs.append(se)
            nobs.append(n)

        windows = list(range(-self.pre, self.post + 1))
        self.results = pd.DataFrame({
            'window': windows,
            'Estimate': betas,
            'Std. Error': SEs,
            'nobs': nobs
        })

        # Normalize placebo estimate to zero for interpretability
        self.results.loc[self.results['window'] == -1, 'Estimate'] = 0

        # Generate summary table
        self.results = self._summary(self.results)

        return self  

    def event_study_plot(self, conf: float = 0.95, xlabel: str = "Time to treatment", ylabel: str = "Estimates and 95% CI", 
                        title: str = "LP-DiD Results", ylim: tuple = None, ci_type: str = "area", xthicks_interval: int = 5,
                        yticks_interval: int = 5, color: str = "black"):
        """
        Plot the LP-DiD results with confidence intervals.
        """
        if self.results is None:
            raise ValueError("No results available. Run fit() before plotting.")

        p = plot_lpdid(self.results, conf=conf, xlab=xlabel, ylab=ylabel, title=title, ylim=ylim, ci_type=ci_type,
                       xticks_interval=xthicks_interval, yticks_interval=yticks_interval, color=color)
        display(p)
        return p