import numpy as np
import pandas as pd
from tabulate import tabulate

def summarize_att_gt(results):
    """
    Summarizes and prints group-time average treatment effects (ATT(g,t)).

    Parameters:
    - results: Dictionary containing ATT(g,t) estimates and related statistics.

    Returns:
    - summary_df: A formatted DataFrame with ATT(g,t) results.
    """
    print("\nGroup-Time Average Treatment Effects - Callaway and Sant'Anna (2021)\n")

    # Confidence Band Text
    confidence_level = 100 * (1 - results["alp"])
    cband_type = "Simultaneous " if results["DIDparams"].get("bstrap", False) and results["DIDparams"].get("cband", False) else "Pointwise "
    cband_text = f"[{confidence_level:.0f}% {cband_type}"

    # Compute Confidence Bands
    cband_lower = results["att"] - results["c"] * results["se"]
    cband_upper = results["att"] + results["c"] * results["se"]

    # Significance Indicator (* if confidence band does not cover 0)
    sig = (cband_upper < 0) | (cband_lower > 0)
    sig[np.isnan(sig)] = False
    sig_text = np.where(sig, "*", "")

    # Create Summary DataFrame
    summary_df = pd.DataFrame({
        "Group": results["group"],
        "Time": results["time"],
        "ATT(g,t)": np.round(results["att"], 4),
        "Std. Error": np.round(results["se"], 4),
        cband_text: np.round(cband_lower, 4),
        "Conf. Band]": np.round(cband_upper, 4),
        "Signif.": sig_text
    })

    # Print the summary table using `tabulate`
    print(tabulate(summary_df, headers="keys", tablefmt="outline", showindex=False))
    
    print("\n---")
    print("Signif. codes: `*' confidence band does not cover 0\n")

    # Control Group
    control_group = results["DIDparams"].get("control_group", None)
    control_group_text = "Never Treated" if control_group == "never" else "Not Yet Treated" if control_group == "notyet" else None

    if control_group_text:
        print(f"Control Group:  {control_group_text}")

    # Anticipation Periods
    print(f"Anticipation Periods:  {results['DIDparams'].get('anticipation', 'Not specified')}")

    # Estimation Method
    est_method = results["DIDparams"].get("est_method", "")
    est_method_text = {
        "dr": "Doubly Robust",
        "ipw": "Inverse Probability Weighting",
        "reg": "Outcome Regression"
    }.get(est_method, est_method)

    if est_method_text:
        print(f"Estimation Method:  {est_method_text}\n")

    return summary_df




def summarize_agg_dynamic(results):
    """
    Summarizes and prints dynamic (event-study) average treatment effects.

    Parameters:
    - results: Dictionary containing aggregated dynamic ATT estimates and related statistics.

    Returns:
    - summary_df: A formatted DataFrame with event-time ATT results.
    """
    print("\nOverall Summary of ATT's Based on Event-Study (Dynamic Aggregation)\n")

    # Overall ATT
    overall_att = results["att"].item()
    overall_se = results["se"].item()
    crit_val = results["critical_value"].item()

    ci_low = overall_att - crit_val * overall_se
    ci_high = overall_att + crit_val * overall_se
    signif_star = "*" if ci_low * ci_high > 0 else ""

    print(f"#>      ATT    Std. Error     [ 95%  Conf. Int.]  ")
    print(f"#>  {overall_att:8.4f}    {overall_se:10.4f}    {ci_low:8.4f}   {ci_high:8.4f} {signif_star}\n")

    # Dynamic effects
    print("#> Dynamic Effects:\n")

    # Confidence Band Text
    confidence_level = 100 * (1 - results["DIDparams"].get("alp", 0.05))
    cband_type = "Simultaneous " if results["DIDparams"].get("bstrap", False) and results["DIDparams"].get("cband", False) else "Pointwise "
    cband_text = f"[{confidence_level:.0f}% {cband_type}"

    # Extract and compute event-specific stats
    egt = np.array(results["egt"]).flatten()
    att_egt = np.array(results["att_egt"]).flatten()
    se_egt = np.array(results["se_egt"]).flatten()
    cband_lower = (att_egt - crit_val * se_egt).flatten()
    cband_upper = (att_egt + crit_val * se_egt).flatten()
    sig = ((cband_upper < 0) | (cband_lower > 0)).flatten()
    sig[np.isnan(sig)] = False
    sig_text = np.where(sig, "*", "")

    # Create DataFrame
    summary_df = pd.DataFrame({
        "Event Time": egt,
        "Estimate": np.round(att_egt, 4),
        "Std. Error": np.round(se_egt, 4),
        cband_text: np.round(cband_lower, 4),
        "Conf. Band]": np.round(cband_upper, 4),
        "Signif.": sig_text
    })

    # Print with tabulate
    print(tabulate(summary_df, headers="keys", tablefmt="outline", showindex=False))
    print("\n---")
    print("Signif. codes: `*' confidence band does not cover 0\n")

    # Control Group
    control_group = results["DIDparams"].get("control_group", None)
    control_group_text = "Never Treated" if control_group == "never" else "Not Yet Treated" if control_group == "notyet" else "Not specified"
    print(f"Control Group:  {control_group_text}")

    # Anticipation Periods
    anticipation = results["DIDparams"].get("anticipation", "Not specified")
    print(f"Anticipation Periods:  {anticipation}")

    # Estimation Method
    est_method = results["DIDparams"].get("est_method", "")
    est_method_text = {
        "dr": "Doubly Robust",
        "ipw": "Inverse Probability Weighting",
        "reg": "Outcome Regression"
    }.get(est_method, est_method)

    if est_method_text:
        print(f"Estimation Method:  {est_method_text}\n")

    return summary_df