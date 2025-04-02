import numpy as np
import pandas as pd
import patsy
import logging
from .utils import get_wide_data
from CausalPanel.drdid.drdid_panel import drdid_panel

def compute_att_gt(dp, verbose=False):
    """
    Computes the ATT(g,t) estimates for Difference-in-Differences (DID) 
    using drdid_panel for panel data only.
    """

    # ---------------------------------------------------------------------
    # Unpack DID Parameters
    # ---------------------------------------------------------------------
    data = dp["data"].copy()
    yname, tname, idname, xformla = dp["yname"], dp["tname"], dp["idname"], dp["xformla"]
    weightsname, control_group, anticipation, gname = dp["weightsname"], dp["control_group"], dp["anticipation"], dp["gname"]
    tlist, glist, nT, nG = dp["tlist"], dp["glist"], dp["nT"], dp["nG"]  # Time periods & treated groups

    # Calculate time periods and adjustment factor
    tlist_len = len(tlist) - 1 
    tfac = 1 
    
    # ---------------------------------------------------------------------
    # Initialize Storage Lists
    # ---------------------------------------------------------------------
    att_est, group, time, post_array = [], [], [], []
    inf_func = []  # Stores influence functions

    def add_att_data(g, t, att, pst, inf_f):
        """ Appends ATT results to storage lists. """
        group.append(g)
        time.append(t)
        att_est.append(att)
        post_array.append(pst)
        inf_func.append(inf_f)  

    # Control group check
    nevertreated = control_group == "never"
    if nevertreated:
        data[".C"] = (data[gname] == 0).astype(int)

    data[".y"] = data[yname]

    # ---------------------------------------------------------------------
    # Loop Over Groups
    # ---------------------------------------------------------------------
    for g_index, g in enumerate(glist):
        group_treatment = glist[g_index]
        data[".G"] = (data[gname] == group_treatment).astype(int)

        # Loop over time periods
        for t in range(tlist_len):
            current_period = tlist[t + tfac]
            pret = max(0, t)  # Ensure `pret` is valid

            # Assign control group
            if not nevertreated:
                data[".C"] = ((data[gname] == 0) | 
                              ((data[gname] > (tlist[max(t, pret) + tfac] + anticipation)) &
                               (data[gname] != group_treatment))).astype(int)

            # Handle post-treatment period
            if group_treatment <= current_period:
                pretreatment_candidates = np.where((np.array(tlist) + anticipation) < group_treatment)[0]
                if len(pretreatment_candidates) > 0:
                    pret = pretreatment_candidates[-1]  # Last available pre-treatment period
                else:
                    logging.warning(f"No pre-treatment periods available for group {group_treatment}. Skipping.")
                    continue  # Skip this group

            # Print debugging info (optional)
            if verbose:
                print(f"Current period: {current_period}")
                print(f"Current group: {group_treatment}")
                print(f"Set pretreatment period to {tlist[pret]}")

            # ---------------------------------------------------------------------
            # Prepare Data for `drdid_panel`
            # ---------------------------------------------------------------------
            post_treat = int(group_treatment <= current_period)

            # Subset the data for the current and pretreatment periods
            disdat = data[(data[tname] == current_period) | (data[tname] == tlist[pret])].copy()

            # Convert to wide formats
            disdat = get_wide_data(disdat, yname, idname, tname)
            n = len(disdat)
            dis_mask = (disdat[".G"] == 1) | (disdat[".C"] == 1)
            disdat = disdat.loc[dis_mask, :]
            n1 = len(disdat)  # Number of observations

            if n1 == 0:
                logging.warning(f"No data for ATT estimation for group {group_treatment} at time {current_period}.")
                add_att_data(group_treatment, current_period, att=np.nan, pst=post_treat, inf_f=np.zeros(len(data)))
                continue

            # Extract variables for estimation
            G, C = disdat[".G"].values, disdat[".C"].values
            Ypre = disdat[".y0"].values if current_period > pret else disdat[".y1"]
            Ypost = disdat[".y0"].values if current_period < pret else disdat[".y1"]
            w = disdat[".w"].values
            covariates = patsy.dmatrix(xformla, disdat, return_type='dataframe').values

            # ---------------------------------------------------------------------
            # Compute ATT(g,t) using drdid_panel
            # ---------------------------------------------------------------------
            try:
                att_gt, att_inf_func = drdid_panel(Ypost, Ypre, G, covariates=covariates, i_weights=w, inffunc=True)
  
                # Initialize influence function storage
                inf_zeros = np.zeros(n, dtype=float)
                att_inf = (n / n1) * att_inf_func
                inf_zeros[dis_mask.to_numpy()] = att_inf

                # Store Results
                add_att_data(group_treatment, current_period, att=att_gt, pst=post_treat, inf_f=inf_zeros)

            except Exception as e:
                logging.warning(f"Error computing ATT for group {group_treatment} at {current_period}: {str(e)}")
                add_att_data(group_treatment, current_period, att=np.nan, pst=post_treat, inf_f=np.zeros(n))

    # Convert to NumPy arrays for better performance
    #group, time, att_est, post_array = map(np.array, [group, time, att_est, post_array])

    # Stack influence functions, ensuring non-empty
    inf_func = np.vstack(inf_func) if inf_func else np.zeros((len(group), len(data)))

    return {
        "group": group,
        "time": time,
        "att": att_est,
        "post": post_array
    }, inf_func