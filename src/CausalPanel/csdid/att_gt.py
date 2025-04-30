import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from IPython.display import display
import logging
from .preprocess import preprocess_did
from .compute_att_gt import compute_att_gt
from .compute_aggte import compute_aggte
from .bootstrap import bootstrap_influence_function
from .summary import summarize_att_gt, summarize_agg_dynamic
from .plots import individual_plot
from .utils import *

class ATTGT:
    """
    A class for estimating the Callaway and Sant'Anna Difference-in-Differences (CSDID) model.
    
    This implementation follows the approach used in the R package `csdid` and `did`, which estimate 
    group-time average treatment effects with multiple periods.
    """
    
    def __init__(self, data: pd.DataFrame, yname: str, idname: str, tname: str, gname: str, 
                 control_group: str = "never", xformla: str = None, est_method: str = "dr",
                 clustervars: list = None, weightsname: str = None, anticipation: int = 0, 
                 cband: bool = False, biters: int = 1000, alp: float = 0.05,
                 bstrap: bool = False, verbose = False):
        """
        Initialize the CSDID model.
        
        Parameters:
        data (pd.DataFrame): Panel dataset.
        yname (str): Name of the outcome variable.
        idname (str): Name of the unit identifier variable.
        tname (str): Name of the time variable.
        gname (str): Name of the group (first treatment) variable.
        control_group (str): Control group specification ("never" or "notyet").
        xformla (str or list): Covariates formula or list of variable names.
        clustervars (list or None): Variables to cluster standard errors.
        weightsname (str): Name of weights column in the data.
        anticipation (int): Number of anticipation periods.
        cband (bool): Whether to compute confidence bands.
        biters (int): Number of bootstrap iterations.
        alp (float): Significance level.
        bstrap (bool): Whether to use bootstrap.
        est_method (str): Estimation method ("dr" for doubly robust).
        base_period (str): Base period for comparison.
        """
        self.data = data
        self.yname = yname
        self.gname = gname
        self.idname = idname
        self.tname = tname
        self.xformla = xformla
        self.est_method = est_method
        self.control_group = control_group
        self.weightsname = weightsname
        self.anticipation = anticipation
        self.alp = alp
        self.cband = cband
        self.biters = biters
        self.clustervars = clustervars
        self.bstrap = bstrap
        self.verbose = verbose
        
        # Placeholder for estimation results
        self.att_gt_results = None
        self.aggte_results = None

    def estimate_att_gt(self):
        """
        Estimate group-time average treatment effects (ATT(g,t)).
        """
        # Preprocess data
        # ======================================
        dp = preprocess_did(
            data=self.data,
            yname=self.yname,
            idname=self.idname,
            tname=self.tname,
            gname=self.gname,
            control_group=self.control_group,
            xformla=self.xformla,
            est_method=self.est_method,
            clustervars=self.clustervars,
            weightsname=self.weightsname,
            anticipation=self.anticipation,
            alp=self.alp,
            bstrap=self.bstrap,
            biters=self.biters,
            cband=self.cband
        )

        # Compute ATT(g,t) and influence function
        # ======================================
        result, inffunc = compute_att_gt(dp, self.verbose)
        
        # Extract ATT(g,t) info
        group = result["group"]
        tt = result["time"]
        att = result["att"]
        
        # Estimate variance and standard errors
        n = list(map(len, inffunc))
        cval, se, V = (
            compute_standard_cval(self.alp),
            np.std(inffunc, axis=1, ddof = 1) / np.sqrt(n),
            np.zeros(len(att)),
        )
        
        # Handle clustering warning
        if self.clustervars and not self.bstrap:
            logging.warning("Clustering standard errors requires bootstrap; standard errors may not account for clustering.")

        # Bootstrap Variance Computation
        # ======================================
        if self.bstrap:
            bout = bootstrap_influence_function(inffunc.T, dp)
            bres = bout["bres"]
            cval, se, V = bout['cval'], bout['se'], bout['V']
        
        if self.bstrap and self.cband:
            cval = compute_critical_value(self.alp, bres)
        
        # Store Results
        # ======================================
        self.att_gt_results = {
            "group": group, "time": tt, "att": att,
            "se": se, "c": cval, "V_analytical": V, 
            "inffunc": inffunc.T, "n": n,
            "alp": self.alp, "DIDparams": dp
        }
        
        # Summarize ATT(g,t) estimation results.
        # ======================================
        if self.att_gt_results is None:
            print("No results available.")
        else:
            summarize_att_gt(self.att_gt_results)
            
        return self

    
    def plot_att_gt(self, ylim=None, xlab="Periods to treatment", ylab="ATT(g,t)", title="Group", 
                xgap=1, group=None, ref_line=0, color_scheme=1, relative_time=False, max_plots_displayed = 5):
        """
        Plot the estimated group-time ATT results.

        Parameters:
        - ylim (tuple): Limits for the y-axis.
        - xlab (str): Label for the x-axis.
        - ylab (str): Label for the y-axis.
        - title (str): Title for the plot.
        - xgap (int): Interval for x-axis labels.
        - ncol (int): Number of columns in the plot layout.
        - legend (bool): Whether to include a legend (default: True).
        - group (list): Specific groups to include (default: all).
        - ref_line (float or None): Reference line (default: 0, set to None to disable).
        - theming (bool): Apply theming (default: True).
        - grtitle (str): Prefix for each group’s title.
        """
        if self.att_gt_results is None:
            print("No results available to plot. Run estimate_att_gt() first.")
            return
        
        # Pass the results to individual_plot (ggdid equivalent)
        plot_list = individual_plot(self.att_gt_results, ylim=ylim, xlab=xlab, ylab=ylab, title=title,
                        xgap=xgap, group=group, ref_line=ref_line, color_scheme=color_scheme,
                        relative_time=relative_time)
        
        # Limit number of displayed plots
        if len(plot_list) > max_plots_displayed:
            print(f"Too many groups to display at once ({len(plot_list)}). Showing only the first {max_plots_displayed}.")
            plot_list = plot_list[:max_plots_displayed]
            
        for plot in plot_list:
                display(plot)
        
        return plot_list
        
    
    def aggregate_att_gt(self, agg_method="group", balance_e=None, min_e=-np.inf, max_e=np.inf, 
                         na_rm=False, bstrap=None, biters=None, cband=None, alp=None, clustervars=None):
        """
        Aggregate the ATT(g,t) estimates to obtain an overall or dynamic ATT.

        Parameters:
        - agg_method (str): Type of aggregation ("simple", "dynamic", "group", "calendar").
        - balance_e: Balance cutoff for event study.
        - min_e (float): Minimum period for event study.
        - max_e (float): Maximum period for event study.
        - na_rm (bool): Remove NA values before aggregation.
        - bstrap (bool): Whether to bootstrap (default: None, inherits from original settings).
        - biters (int): Number of bootstrap iterations.
        - cband (bool): Whether to compute confidence bands.
        - alp (float): Significance level.
        - clustervars (list): Variables for clustering.

        Returns:
        - Aggregated ATT estimates.
        """
        if self.att_gt_results is None:
            print("No ATT(g,t) results available. Run `estimate_att_gt()` first.")
            return None

        # Compute aggregated ATT estimates
        self.aggte_results = compute_aggte(
            MP=self.att_gt_results, agg_method=agg_method, balance_e=balance_e,
            min_e=min_e, max_e=max_e, na_rm=na_rm, bstrap=bstrap,
            biters=biters, cband=cband, alp=alp, clustervars=clustervars
        )
        print(f"Aggregated ATT using method {agg_method}")

        return self.aggte_results

    
    def summarize_aggte(self):
        """
        Summarize aggregated ATT results.
        """
        if self.aggte_results is None:
            print("No aggregated results available. Run aggregate_att_gt() first.")
        elif self.aggte_results["agg_method"] == "dynamic":
            print("Summary of aggregated ATT for ", self.aggte_results["agg_method"], "method")
            summarize_agg_dynamic(self.aggte_results)
        elif self.aggte_results["agg_method"] == "group":   
            print("Summary of aggregated ATT for ", self.aggte_results["agg_method"], "method")
            summarize_agg_group(self.att_gt_results)
        else:   
            print("No aggregated results available. Run aggregate_att_gt() first.")
    
    def plot_aggte(self, ylim=(-0.3, 0.3)):
        """
        Plot the aggregated ATT results.
        
        Parameters:
        ylim (tuple): Limits for the y-axis.
        """
        if self.aggte_results is None:
            print("No results available to plot. Run aggregate_att_gt() first.")
        else:
            # Placeholder for plotting logic
            plt.figure(figsize=(8, 5))
            plt.title("Aggregated ATT Estimates")
            plt.ylim(ylim)
            plt.show()

# Example Usage (Once implemented):
# csdid_model = CSDID(yname="lemp", gname="first.treat", idname="countyreal", tname="year", xformla=["cov1", "cov2"], data=mpdta)
# csdid_model.estimate_att_gt()
# csdid_model.summarize_att_gt()
# csdid_model.plot_att_gt()
# csdid_model.aggregate_att_gt(method="dynamic")
# csdid_model.summarize_aggte()
# csdid_model.plot_aggte()

