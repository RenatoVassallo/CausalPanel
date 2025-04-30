import numpy as np
import warnings
from scipy import stats
from .utils import get_se, get_agg_inf_func, wif
from .bootstrap import bootstrap_influence_function


def aggte_simple(att, inffunc, pg, keepers, weights, G, group, dp):
    weights_agg = pg[keepers] / np.sum(pg[keepers])
    att_agg = np.sum(att[keepers] * weights_agg)
    wif_part = wif(keepers, pg, weights, G, group)
    inf_func = get_agg_inf_func(att, inffunc, keepers, weights_agg, wif_part)[:, None]
    se = get_se(inf_func, dp)
    if se is not None and se <= np.sqrt(np.finfo(float).eps) * 10:
        se = None

    return {
        "agg_method": "simple",
        "att": att_agg,
        "se": se,
        "egt": None,
        "att_egt": None,
        "se_egt": None,
        "critical_value": stats.norm.ppf(1 - dp["alp"] / 2),
        "influence_function": {"agg_inf_func": inf_func},
        "DIDparams": dp
    }


def aggte_group(att, inffunc, group, time, glist, pg, pgg, keepers, weights, G, dp, original_glist):
    att_g, se_g, inf_func_g = [], [], []
    for g in glist:
        mask = (group == g) & (time >= g) & (time <= (group + max(time)))
        att_val = np.mean(att[mask]) if np.any(mask) else np.nan
        att_g.append(att_val)
        wgts = pg[mask] / np.sum(pg[mask])
        inf = get_agg_inf_func(att, inffunc, np.where(mask)[0], wgts, None)[:, None]
        inf_func_g.append(inf)
        se_g.append(get_se(inf, dp))

    att_g = np.array(att_g)
    inf_func_g = np.hstack(inf_func_g)
    se_g = np.array(se_g)
    se_g[se_g <= np.sqrt(np.finfo(float).eps) * 10] = np.nan

    crit_val = stats.norm.ppf(1 - dp["alp"] / 2)
    if dp["cband"]:
        if not dp["bstrap"]:
            print("Used bootstrap procedure to compute simultaneous confidence band")
        crit_val = bootstrap_influence_function(inf_func_g, dp)["crit_val"]
        if np.isnan(crit_val) or np.isinf(crit_val) or crit_val >= 7:
            dp["cband"] = False
            crit_val = stats.norm.ppf(1 - dp["alp"] / 2)

    att_agg = np.sum(att_g * pgg) / np.sum(pgg)
    wif_val = wif(np.arange(len(glist)), pgg, weights, G, group)
    inf_func = get_agg_inf_func(att_g, inf_func_g, np.arange(len(glist)), pgg / np.sum(pgg), wif_val)[:, None]
    se = get_se(inf_func, dp)
    if se is not None and se <= np.sqrt(np.finfo(float).eps) * 10:
        se = None

    return {
        "agg_method": "group",
        "att": att_agg,
        "se": se,
        "egt": original_glist,
        "att_egt": att_g,
        "se_egt": se_g,
        "critical_value": crit_val,
        "influence_function": {
            "agg_inf_func": inf_func,
            "egt_inf_func": inf_func_g
        },
        "DIDparams": dp
    }


def aggte_dynamic(att, inffunc, original_time, original_group, pg, max_t, balance_e, min_e, max_e,
                  weights, G, group, dp, t2orig):
    eseq = np.unique(original_time - original_group)
    include_mask = np.repeat(True, len(original_group))
    if balance_e is not None:
        include_mask = (t2orig(max_t) - original_group >= balance_e)
        eseq = np.unique(original_time[include_mask] - original_group[include_mask])
        eseq = eseq[(eseq <= balance_e) & (eseq >= balance_e - t2orig(max_t) + t2orig(1))]

    eseq = eseq[(eseq >= min_e) & (eseq <= max_e)]
    att_e, se_e, inf_func_e = [], [], []

    for e in eseq:
        idx = np.where((original_time - original_group == e) & include_mask)[0]
        if len(idx) == 0:
            att_e.append(np.nan)
            inf_func_e.append(np.full((inffunc.shape[0], 1), np.nan))
            se_e.append(np.nan)
            continue
        pge = pg[idx] / np.sum(pg[idx])
        inf = get_agg_inf_func(att, inffunc, idx, pge, wif(idx, pg, weights, G, group))[:, None]
        inf_func_e.append(inf)
        se_e.append(get_se(inf, dp))
        att_e.append(np.sum(att[idx] * pge))

    inf_func_mat = np.hstack(inf_func_e)
    se_e = np.array(se_e)
    se_e[se_e <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
    epos = eseq >= 0
    att_avg = np.mean(np.array(att_e)[epos])
    inf_func = get_agg_inf_func(np.array(att_e)[epos], inf_func_mat[:, epos],
                                np.arange(np.sum(epos)), np.full(np.sum(epos), 1 / np.sum(epos)), None)[:, None]
    se = get_se(inf_func, dp)
    if se is not None and se <= np.sqrt(np.finfo(float).eps) * 10:
        se = None

    crit_val = stats.norm.ppf(1 - dp["alp"] / 2)
    if dp["cband"]:
        if not dp["bstrap"]:
            print("Used bootstrap procedure to compute simultaneous confidence band")
        crit_val = bootstrap_influence_function(inf_func_mat, dp)["crit_val"]
        if np.isnan(crit_val) or np.isinf(crit_val) or crit_val >= 7:
            dp["cband"] = False
            crit_val = stats.norm.ppf(1 - dp["alp"] / 2)

    return {
        "agg_method": "dynamic",
        "att": att_avg,
        "se": se,
        "egt": eseq,
        "att_egt": att_e,
        "se_egt": se_e,
        "critical_value": crit_val,
        "influence_function": {
            "agg_inf_func": inf_func,
            "egt_inf_func": inf_func_mat
        },
        "DIDparams": dp
    }


def aggte_calendar(att, inffunc, group, time, pg, weights, G, tlist, glist, dp, t2orig):
    min_g = min(group)
    calendar_tlist = tlist[tlist >= min_g]
    att_t, inf_func_t, se_t = [], [], []

    for t1 in calendar_tlist:
        idx = np.where((time == t1) & (group <= time))[0]
        if len(idx) == 0:
            att_t.append(np.nan)
            inf_func_t.append(np.full((inffunc.shape[0], 1), np.nan))
            se_t.append(np.nan)
            continue
        pgt = pg[idx] / np.sum(pg[idx])
        inf = get_agg_inf_func(att, inffunc, idx, pgt, wif(idx, pg, weights, G, group))[:, None]
        inf_func_t.append(inf)
        se_t.append(get_se(inf, dp))
        att_t.append(np.sum(pgt * att[idx]))

    inf_func_mat = np.hstack(inf_func_t)
    se_t = np.array(se_t)
    se_t[se_t <= np.sqrt(np.finfo(float).eps) * 10] = np.nan
    att_avg = np.nanmean(att_t)
    inf_func = get_agg_inf_func(att_t, inf_func_mat, np.arange(len(calendar_tlist)),
                                np.full(len(calendar_tlist), 1 / len(calendar_tlist)), None)[:, None]
    se = get_se(inf_func, dp)
    if se is not None and se <= np.sqrt(np.finfo(float).eps) * 10:
        se = None

    crit_val = stats.norm.ppf(1 - dp["alp"] / 2)
    if dp["cband"]:
        if not dp["bstrap"]:
            warnings.warn("Used bootstrap procedure to compute simultaneous confidence band")
        crit_val = bootstrap_influence_function(inf_func_mat, dp)["crit_val"]
        if np.isnan(crit_val) or np.isinf(crit_val) or crit_val >= 7:
            dp["cband"] = False
            crit_val = stats.norm.ppf(1 - dp["alp"] / 2)

    return {
        "agg_method": "calendar",
        "att": att_avg,
        "se": se,
        "egt": list(map(t2orig, calendar_tlist)),
        "att_egt": att_t,
        "se_egt": se_t,
        "critical_value": crit_val,
        "influence_function": {
            "agg_inf_func": inf_func,
            "egt_inf_func": inf_func_mat
        },
        "DIDparams": dp
    }