import pandas as pd
import numpy as np
from scipy.stats import norm
from plotnine import *

def plot_lpdid(reg, conf=0.95, xlab="Time to Treatment",
               ylab="Coefficient Estimates and 95% CI", title="",
               ylim=None, point_size=5, color="black", opacity=1,
               ci_type="area", yticks_interval=None, xticks_interval=None):

    coeftable = reg.copy()
    coeftable["t"] = (
        coeftable["E-time"].str.replace("pre", "-").str.replace("tau", "").astype(int)
    )

    z = norm.ppf(1 - (1 - conf) / 2)
    coeftable["uCI"] = coeftable["Estimate"] + z * coeftable["Std. Error"]
    coeftable["lCI"] = coeftable["Estimate"] - z * coeftable["Std. Error"]

    # Axis tick intervals centered at 0
    if yticks_interval:
        y_min = ylim[0] if ylim else np.floor(coeftable["lCI"].min())
        y_max = ylim[1] if ylim else np.ceil(coeftable["uCI"].max())
        y_ticks_neg = np.arange(0, y_min - yticks_interval, -yticks_interval)[::-1]
        y_ticks_pos = np.arange(0, y_max + yticks_interval, yticks_interval)
        y_breaks = np.unique(np.concatenate((y_ticks_neg, y_ticks_pos)))
    else:
        y_breaks = None

    if xticks_interval:
        x_min = coeftable["t"].min()
        x_max = coeftable["t"].max()
        x_ticks_neg = np.arange(0, x_min - xticks_interval, -xticks_interval)[::-1]
        x_ticks_pos = np.arange(0, x_max + xticks_interval, xticks_interval)
        x_breaks = np.unique(np.concatenate((x_ticks_neg, x_ticks_pos)))
    else:
        x_breaks = None

    p = (
        ggplot(coeftable, aes(x="t", y="Estimate"))
        + geom_hline(yintercept=0, linetype="dashed", color="gray")
        + geom_vline(xintercept=-1, linetype="dashed", color="gray")
        + labs(x=xlab, y=ylab, title=title)
        + theme_minimal()
        + theme(
            axis_title=element_text(size=11),
            axis_text=element_text(size=10),
            plot_title=element_text(size=12, weight="bold"),
            panel_background=element_rect(fill="white", colour=None),
            plot_background=element_rect(fill="white", colour=None),
            panel_grid_major=element_line(color="gray", alpha=0.2),
            panel_grid_minor=element_line(color="gray", alpha=0.1),
            axis_line=element_line(color="black", size=0.5),
            axis_ticks=element_line(color="black", size=0.5)
        )
    )

    if ci_type == "bars":
        p += geom_point(size=point_size, color=color, alpha=opacity)
        p += geom_errorbar(aes(ymin="lCI", ymax="uCI"), width=0.2, color=color, alpha=opacity)
    elif ci_type == "area":
        p += geom_line(size=1.1, color=color, alpha=opacity)
        p += geom_point(size=4, color=color, alpha=opacity)
        # Shaded area and borders
        p += geom_ribbon(aes(ymin="lCI", ymax="uCI"), fill=color, alpha=0.1)
        p += geom_line(aes(y="uCI"), color=color, linetype="dashed", size=0.5, alpha=0.2)
        p += geom_line(aes(y="lCI"), color=color, linetype="dashed", size=0.5, alpha=0.2)
    else:
        raise ValueError("ci_type must be either 'bars' or 'area'")

    if ylim:
        p += coord_cartesian(ylim=ylim)
    if y_breaks is not None:
        p += scale_y_continuous(breaks=y_breaks)
    if x_breaks is not None:
        p += scale_x_continuous(breaks=x_breaks)

    return p



def plot_lpdid_multiple(results_list, labels, conf=0.95,
                        xlab="Time to Treatment", ylab="Coefficient Estimates and 95% CI",
                        title="", ylim=None, colors=["black", "grey"],
                        yticks_interval=None, xticks_interval=None):

    assert len(results_list) == len(labels) == len(colors), "Mismatched input lengths"

    dfs = []
    for df, label in zip(results_list, labels):
        temp = df.copy()
        temp["t"] = temp["E-time"].str.replace("pre", "-").str.replace("tau", "").astype(int)
        z = norm.ppf(1 - (1 - conf) / 2)
        temp["uCI"] = temp["Estimate"] + z * temp["Std. Error"]
        temp["lCI"] = temp["Estimate"] - z * temp["Std. Error"]
        temp["group"] = label
        dfs.append(temp)

    all_data = pd.concat(dfs)
    all_data["group"] = pd.Categorical(all_data["group"], categories=labels, ordered=True)

    # Axis ticks
    if yticks_interval:
        y_min = ylim[0] if ylim else np.floor(all_data["lCI"].min())
        y_max = ylim[1] if ylim else np.ceil(all_data["uCI"].max())
        y_ticks_neg = np.arange(0, y_min - yticks_interval, -yticks_interval)[::-1]
        y_ticks_pos = np.arange(0, y_max + yticks_interval, yticks_interval)
        y_breaks = np.unique(np.concatenate((y_ticks_neg, y_ticks_pos)))
    else:
        y_breaks = None

    if xticks_interval:
        x_min = all_data["t"].min()
        x_max = all_data["t"].max()
        x_ticks_neg = np.arange(0, x_min - xticks_interval, -xticks_interval)[::-1]
        x_ticks_pos = np.arange(0, x_max + xticks_interval, xticks_interval)
        x_breaks = np.unique(np.concatenate((x_ticks_neg, x_ticks_pos)))
    else:
        x_breaks = None

    p = (
        ggplot(all_data, aes(x="t", group="group", color="group", fill="group"))
        + geom_line(aes(y="Estimate"), size=1.1)
        + geom_point(aes(y="Estimate"), size=4)
        # Shaded CI ribbon
        + geom_ribbon(aes(ymin="lCI", ymax="uCI"), alpha=0.3, linetype="dashed")
        # CI upper and lower dashed lines
        + geom_line(aes(y="uCI"), linetype="dashed", size=0.5, alpha=0.5)
        + geom_line(aes(y="lCI"), linetype="dashed", size=0.5, alpha=0.5)
        # Reference lines
        + geom_hline(yintercept=0, linetype="dashed", color="gray")
        + geom_vline(xintercept=-1, linetype="dashed", color="gray")
        # Labels and theme
        + labs(x=xlab, y=ylab, title=title, color="Group", fill="Group")
        + scale_color_manual(values=colors)
        + scale_fill_manual(values=colors)
        + theme_minimal()
        + theme(
            legend_position='bottom',
            legend_title=element_blank(),
            legend_text=element_text(size=9),
            axis_title=element_text(size=11),
            axis_text=element_text(size=10),
            plot_title=element_text(size=12, weight="bold"),
            panel_background=element_rect(fill="white", colour=None),
            plot_background=element_rect(fill="white", colour=None),
            panel_grid_major=element_line(color="gray", alpha=0.2),
            panel_grid_minor=element_line(color="gray", alpha=0.1),
            axis_line=element_line(color="black", size=0.5),
            axis_ticks=element_line(color="black", size=0.5)
        )
    )

    if ylim:
        p += coord_cartesian(ylim=ylim)
    if y_breaks is not None:
        p += scale_y_continuous(breaks=y_breaks)
    if x_breaks is not None:
        p += scale_x_continuous(breaks=x_breaks)

    return p