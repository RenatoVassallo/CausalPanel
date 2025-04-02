import numpy as np
import pandas as pd
import warnings
from scipy import stats
from .utils import get_se, get_agg_inf_func, wif  # Helper functions
from .bootstrap import bootstrap_influence_function

def compute_aggte(MP, agg_method="group", balance_e=None, min_e = float('-inf'), max_e = float('inf'), 
                  na_rm=False, bstrap=None, biters=None, cband=None, alp=None, clustervars=None):
    """
    Compute aggregated ATT estimates from ATT(g,t).

    Parameters:
    - MP (dict): ATT(g,t) estimation results (from `estimate_att_gt()`).
    - agg_method (str): Aggregation type ("simple", "dynamic", "group", "calendar").
    - balance_e: Balance cutoff for event study.
    - min_e (float): Minimum event period.
    - max_e (float): Maximum event period.
    - na_rm (bool): Remove NAs.
    - bstrap (bool): Bootstrap variance estimation.
    - biters (int): Number of bootstrap iterations.
    - cband (bool): Confidence bands.
    - alp (float): Significance level.
    - clustervars (list): Clustering variables.

    Returns:
    - Aggregated ATT results.
    """
    # =============================================================================
    # Unpack MP Object
    # =============================================================================
    group = np.array(MP["group"])
    t = np.array(MP["time"])
    att = np.array(MP["att"])
    dp = MP["DIDparams"]
    inffunc = MP["inffunc"]
    n = MP["n"]
    
    print("Shape aggte inffunc: ", inffunc.shape)

    tlist = np.array(dp["tlist"])
    glist = np.array(dp["glist"])
    data = dp["data"]
    gname = dp["gname"]
    tname = dp["tname"]
    idname = dp["idname"]

    # **Override Parameters if Custom Settings Provided**
    clustervars = clustervars or dp.get("clustervars")
    bstrap = bstrap if bstrap is not None else dp.get("bstrap")
    biters = biters if biters is not None else dp.get("biters")
    alp = alp if alp is not None else dp.get("alp")
    cband = cband if cband is not None else dp.get("cband")

    # **Update MP Object for Bootstrap Computation**
    MP["DIDparams"]["clustervars"] = clustervars
    MP["DIDparams"]["bstrap"] = bstrap
    MP["DIDparams"]["biters"] = biters
    MP["DIDparams"]["alp"] = alp
    MP["DIDparams"]["cband"] = cband

    # =============================================================================
    # Treat Data
    # =============================================================================
    if agg_method not in ["simple", "dynamic", "group", "calendar"]:
        raise ValueError("`agg_method` must be one of ['simple', 'dynamic', 'group', 'calendar']")

    # Remove Missing Values if `na_rm=True`
    if na_rm:
        notna = ~np.isnan(att)
        group, t, att = group[notna], t[notna], att[notna]
        inffunc = inffunc[:, notna]
        glist = np.sort(np.unique(group))

        if agg_method == "group":
            # Ensure Non-Missing Post-Treatment ATTs Exist for Each Group
            gnotna = np.array([
                np.any(~np.isnan(att[np.where((group == g) & (g <= t))]))
                for g in glist
            ])
            glist = glist[gnotna]
            not_all_na = np.isin(group, glist)
            group, t, att = group[not_all_na], t[not_all_na], att[not_all_na]
            inffunc = inffunc[:, not_all_na]
            glist = np.sort(np.unique(group))

    if not na_rm and np.any(np.isnan(att)):
        raise ValueError("Missing values found in ATT(g,t). Set `na_rm=True` to remove.")

    dta = data[data[tname] == tlist[0]]

    # =============================================================================
    # Data Organization and Recoding
    # =============================================================================
    original_t = t.copy()
    original_group = group.copy()
    original_glist = glist.copy()
    original_tlist = tlist.copy()

    # Ensure time periods are sequential
    original_gtlist = np.sort(np.unique(np.concatenate((original_tlist, original_glist))))
    unique_t = list(range(1, len(original_gtlist) + 1))

    # Function to switch between original and new time values
    def t2orig(t):
        return original_gtlist[unique_t.index(t)] if t in unique_t else np.nan

    def orig2t(orig):
        idx = np.where(original_gtlist == orig)[0]
        return unique_t[idx[0]] if len(idx) > 0 else None

    # Recode time, group, and glist
    t = np.array([orig2t(orig) for orig in original_t])
    group = np.array([orig2t(orig) for orig in original_group])
    glist = np.array([orig2t(orig) for orig in original_glist])
    tlist = np.asarray(list(set(t)))
    maxT = max(t)

    # =============================================================================
    # Compute Weights
    # =============================================================================
    weights_ind = dta[".w"].to_numpy()  

    # Compute probability of being in each group: P(G = g)
    group_map = {g: np.mean(weights_ind * (dta[gname].to_numpy() == g)) for g in original_glist}
    pg = np.array([group_map[g] for g in original_glist])
    pgg = pg.copy()  # Store original probabilities

    # Align `pg` with `group` safely
    pg = np.array([group_map.get(g, 0) for g in group])

    # Identify Post-Treatment ATT(g,t)
    keepers = [i for i in range(len(group)) if group[i] <= t[i] <= (group[i] + max_e)]

    # Convert group identifiers to numeric values
    G = [orig2t(g) for g in dta[gname].to_numpy()]
    
    # =============================================================================
    # Dynamic Aggregation
    # =============================================================================
    if agg_method == "dynamic":
        # Compute event times: e = t - g
        eseq = np.unique(np.array(original_t) - np.array(original_group))
        eseq = np.sort(eseq)  # Sort the unique values

        # Convert key arrays to NumPy arrays for performance
        original_t = np.array(original_t)
        original_group = np.array(original_group)
        pg = np.array(pg)
        include_balanced_gt = np.repeat(True, len(original_group))

        # Apply sample balancing if required
        if balance_e is not None:
            include_balanced_gt = (t2orig(maxT) - original_group >= balance_e)
            eseq = np.unique(original_t[include_balanced_gt] - original_group[include_balanced_gt])
            eseq = np.sort(eseq)
            eseq = eseq[(eseq <= balance_e) & (eseq >= balance_e - t2orig(maxT) + t2orig(1))]

        # Filter event times within min_e and max_e
        eseq = eseq[(eseq >= min_e) & (eseq <= max_e)]

        # Compute ATT for each event time
        dynamic_att_e = []
        for e in eseq:
            whiche = np.where((original_t - original_group == e) & include_balanced_gt)
            atte = att[whiche]
            pge = pg[whiche] / np.sum(pg[whiche])
            dynamic_att_e.append(np.sum(atte * pge))

        # Compute standard errors
        dynamic_se_inner = []
        for e in eseq:
            whiche = np.where((original_t - original_group == e) & include_balanced_gt)[0]
            pge = pg[whiche] / np.sum(pg[whiche] + np.finfo(float).eps)  # Avoid division by zero
            wif_e = wif(whiche, pg, weights_ind, G, group)
            inf_func_e = get_agg_inf_func(att = att,
                                          inffunc = inffunc,
                                          whichones = whiche,
                                          weights_agg = pge,
                                          wif = wif_e)[:, None]  # Ensure correct shape
            print("inf_func_e for get_Se: ", inf_func_e)
            se_e = get_se(inf_func_e, dp)
            dynamic_se_inner.append({"inf_func": inf_func_e, "se": se_e})

        # Extract standard errors
        dynamic_se_e = np.array([x["se"] for x in dynamic_se_inner])
        dynamic_se_e[dynamic_se_e <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

        # Compute influence function matrix
        dynamic_inf_func_e = np.column_stack([x["inf_func"] for x in dynamic_se_inner if x["inf_func"] is not None])

        # Compute confidence bands
        dynamic_crit_val = stats.norm.ppf(1 - alp / 2)
        if dp["cband"]:
            if not dp["bstrap"]:
                warnings.warn("Bootstrap required for simultaneous confidence band computation.")

            dynamic_crit_val = bootstrap_influence_function(dynamic_inf_func_e, dp)["crit_val"]

            if np.isnan(dynamic_crit_val) or np.isinf(dynamic_crit_val):
                warnings.warn("Simultaneous critical value is NA. Using pointwise confidence intervals.")
                dynamic_crit_val = stats.norm.ppf(1 - alp / 2)
                dp["cband"] = False

            if dynamic_crit_val < stats.norm.ppf(1 - alp / 2):
                warnings.warn("Simultaneous confidence band smaller than normal approximation. Using pointwise confidence intervals.")
                dynamic_crit_val = stats.norm.ppf(1 - alp / 2)
                dp["cband"] = False

            if dynamic_crit_val >= 7:
                warnings.warn("Simultaneous critical value is too large, suggesting low sample size or low variation.")

        # Compute overall ATT by averaging over positive dynamics (e >= 0)
        epos = eseq >= 0
        dynamic_att = np.nanmean(dynamic_att_e[epos])  # Avoid errors due to NaNs

        # Compute overall influence function
        if np.any(epos):
            dynamic_inf_func = get_agg_inf_func(
                att=np.array(dynamic_att_e)[epos],
                inffunc=np.array(dynamic_inf_func_e[:, epos]),
                whichones=np.arange(np.sum(epos)),
                weights_agg=np.full(np.sum(epos), 1 / np.sum(epos)),
                wif=None
            )[:, None]
        else:
            dynamic_inf_func = np.full((len(inf_func_e), 1), np.nan)

        # Compute standard error for overall dynamic ATT
        dynamic_se = get_se(dynamic_inf_func, dp)
        if dynamic_se is not None and dynamic_se <= np.sqrt(np.finfo(float).eps) * 10:
            dynamic_se = None

        return {
            "agg_method": "dynamic",
            "att": dynamic_att,
            "se": dynamic_se,
            "event_times": eseq,
            "att_by_event_time": dynamic_att_e,
            "se_by_event_time": dynamic_se_e,
            "influence_function": dynamic_inf_func,
            "critical_value": dynamic_crit_val
        }

    # =============================================================================
    # Aggregation by Simple ATT
    # =============================================================================
    if agg_method == "simple":
        simple_att = np.sum(att[keepers] * pg[keepers]) / np.sum(pg[keepers])
        simple_att = None if np.isnan(simple_att) else simple_att

        # Influence function computation
        print("keepers: ", len(keepers))
        print("pg: ", pg.shape)
        print("weights_ind: ", weights_ind.shape)
        print("G: ", len(G))
        print("group: ", group.shape)

        simple_wif = wif(keepers, pg, weights_ind, G, group)
        simple_if = get_agg_inf_func(att=att, 
                                     inffunc=inffunc, 
                                     whichones=keepers,
                                     weights_agg=pg[keepers] / np.sum(pg[keepers]), 
                                     wif=simple_wif)[:, None]

        # Compute standard errors
        simple_se = get_se(simple_if, dp)
        simple_se = None if simple_se and simple_se <= np.sqrt(np.finfo(float).eps) * 10 else simple_se

        return {
            "agg_method": "simple",
            "att": simple_att,
            "se": simple_se,
            "influence_function": simple_if,
        }

    # -------------------------------
    # Aggregation by Dynamic ATT
    # -------------------------------
    elif agg_method == "dynamic":
        # Get event time (relative to treatment)
        event_time = t - group
        unique_e = np.unique(event_time)

        # Aggregate ATT by event time
        dynamic_att = {e: np.mean(att[event_time == e]) for e in unique_e}
        dynamic_se = {e: np.std(inffunc[:, event_time == e]) / np.sqrt(len(event_time == e)) for e in unique_e}

        return {
            "agg_method": "dynamic",
            "event_time": unique_e,
            "att": dynamic_att,
            "se": dynamic_se,
        }

    # -------------------------------
    # Aggregation by Group ATT
    # -------------------------------
    elif agg_method == "group":
        group_att = {g: np.mean(att[group == g]) for g in np.unique(group)}
        group_se = {g: np.std(inffunc[:, group == g]) / np.sqrt(len(group == g)) for g in np.unique(group)}

        return {
            "agg_method": "group",
            "group": np.unique(group),
            "att": group_att,
            "se": group_se,
        }

    else:
        raise ValueError("Invalid aggregation method. Choose from 'simple', 'dynamic', 'group'.")