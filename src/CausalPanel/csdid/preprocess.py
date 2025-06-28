import numpy as np
import pandas as pd
import patsy
import logging

def preprocess_did(yname, tname, idname, gname, data: pd.DataFrame, control_group: str = "never", 
                   xformla: str = None, est_method: str = "dr", clustervars: list = None, 
                   weightsname: str = None, anticipation: int = 0, alp: float = 0.05, 
                   bstrap: bool = True, biters: int = 1000, cband: bool = False) -> dict:
    """
    Preprocess data and perform error checks for Difference-in-Differences estimation.
    """
    # Validate control group selection
    if control_group not in ["never", "notyet"]:
        raise ValueError("control_group must be either 'never' or 'notyet'")

    # Ensure data is a DataFrame
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    # Convert non-numeric ID/time/group columns to integers
    factor_mappings = {}
    for col in [idname, tname, gname]:
        if not np.issubdtype(data[col].dtype, np.number):
            data[col], mapping = pd.factorize(data[col])
            factor_mappings[col] = dict(enumerate(mapping))
        else:
            factor_mappings[col] = None

    # Handle covariate formula
    if xformla is None:
        xformla = "1"
    else:
        try:
            formula_vars = patsy.dmatrices(xformla + " ~ 1", data, return_type='dataframe')[1].columns.tolist()
            # Remove "Intercept" from the missing variables check
            missing_vars = [var for var in formula_vars if var not in data.columns and var != "Intercept"]
            if missing_vars:
                raise ValueError(f"Missing variables in data: {', '.join(missing_vars)}")
        except Exception as e:
            raise ValueError(f"Error processing formula: {str(e)}")

    # Select relevant columns
    relevant_columns = [idname, tname, yname, gname] + ([weightsname] if weightsname else []) + (clustervars or [])

    # Create design matrix for covariates
    x_data = patsy.dmatrix(xformla, data, return_type='dataframe')
    data = pd.concat([data[relevant_columns], x_data], axis=1)

    # Drop missing values
    n_orig = len(data)
    data.dropna(inplace=True)
    n_dropped = n_orig - len(data)
    if n_dropped:
        logging.warning(f"Dropped {n_dropped} rows due to missing data")

    # Assign weights
    data[".w"] = data[weightsname] if weightsname else 1
    if ".w" in relevant_columns:  # Ensure ".w" doesn't overwrite an existing column
        raise ValueError("Column '.w' conflicts with existing column in data")

    # Handling treatment groups
    tlist = sorted(data[tname].unique())
    max_time = max(tlist, default=0)

    # Convert units treated after max_time to "never treated" (0)
    data.loc[data[gname] > max_time, gname] = 0
    glist = sorted(data[gname].unique())

    # Ensure a never-treated group exists
    if 0 not in glist:
        if control_group == "never":
            raise ValueError("No available never-treated group")
        else:
            max_treated = max(glist)
            data = data[data[tname] < (max_treated - anticipation)]
            tlist, glist = sorted(data[tname].unique()), sorted(data[gname].unique())
            glist = [g for g in glist if g < max(glist)]

    # Filter treated groups
    glist = [g for g in glist if g > 0 and g > min(tlist) + anticipation]

    # Drop units treated in the first period
    treated_first_period = (data[gname] <= min(tlist)) & (data[gname] != 0)
    n_first_dropped = data[treated_first_period][idname].nunique()

    if n_first_dropped > 0:
        logging.warning(f"Dropped {n_first_dropped} units treated in the first period")
        data = data[data[gname].isin([0] + glist)]
        tlist, glist = sorted(data[tname].unique()), sorted(data[gname].unique())

    # Balance panel dataset
    # ----------------------------------------------------------------
    n_before = data[idname].nunique()
    data = data.sort_values([idname, tname]).reset_index(drop=True)
    nt = len(data[tname].unique())
    data = data.groupby(idname).filter(lambda x: len(x) == nt)
    n_after = data[idname].nunique()

    if len(data) == 0:
        raise ValueError("All observations dropped when making panel balanced. Consider setting `panel=False`.")

    if n_before > n_after:
        logging.warning(f"Dropped {n_before - n_after} observations while balancing panel.")
    
    n = len(data[data[tname] == tlist[0]])

    # Check for Small Groups
    # ----------------------------------------------------------------
    if len(glist) == 0:
        raise f"No valid groups. The variable in '{gname}' should be expressed as the time a unit is first treated (0 if never-treated)."
    if len(tlist) == 2:
        cband = False
    # Count observations per group and compute group sizes relative to tlist length
    group_sizes = data.groupby(gname).size().reset_index(name="count")
    group_sizes["ratio"] = group_sizes["count"] / len(tlist)

    # Required minimum group size (number of covariates + buffer)
    reqsize = len(patsy.dmatrix(xformla, data, return_type='dataframe').columns) + 5

    # Identify small groups
    small_groups = group_sizes.query("ratio < @reqsize")

    # Warn about small groups
    if not small_groups.empty:
        small_group_list = ", ".join(map(str, small_groups[gname].tolist()))
        logging.warning(f"Be aware that some small groups exist: {small_group_list}.")

        if 0 in small_groups[gname].values and control_group == "never":
            raise ValueError("Never-treated group is too small. Consider setting control_group='notyet'.")

    # Final Data Sorting
    # ----------------------------------------------------------------
    data = data.sort_values([idname, tname]).reset_index(drop=True)

    # Construct return object (DIDparams equivalent)
    dp = {
        "data": data,
        "tlist": tlist,
        "glist": glist,
        "nT": len(tlist),  # Number of time periods
        "nG": len(glist),  # Number of treated groups
        "n": n,
        "idname": idname,
        "tname": tname,
        "yname": yname,
        "gname": gname,
        "xformla": xformla,
        "control_group": control_group,
        "est_method": est_method,
        "clustervars": clustervars,
        "anticipation": anticipation,
        "weightsname": weightsname,
        "alp": alp,
        "bstrap": bstrap,
        "biters": biters,
        "cband": cband,
        'factor_mappings': factor_mappings
    }
    return dp