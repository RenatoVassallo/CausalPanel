import numpy as np
import pandas as pd
from plotnine import *

def individual_plot(results, group=None, ylim=None, xlab="Periods to treatment", 
                    ylab="ATT(g,t)", xgap=1, title="Group", ref_line=0, 
                    color_scheme=1, relative_time=False):
    """
    Generates a **list of plots** with confidence intervals as rectangles.

    Parameters:
    - results: Dictionary containing ATT(g,t) estimates.
    - group: List of specific groups to plot (default: all).
    - ylim: Tuple (ymin, ymax) for y-axis limits.
    - xlab: Label for x-axis.
    - ylab: Label for y-axis.
    - title: Plot title (default: "Group"). Set to False to remove title.
    - xgap: Interval between x-axis labels.
    - ref_line: Value for reference line (default: 0, set to None to disable).
    - color_scheme: Integer (1-5) to select color themes for pre/post-treatment.
    - relative_time: If True, recenter x-axis so that Time = Group is set to 0.

    Returns:
    - List of ggplot objects (one for each group).
    """

    # Define color schemes
    color_palettes = [
        ("#87b8d1", "#d187b8"),
        ("#457b9d", "#e63946"),
        ("#2a9d8f", "#e76f51"),
        ("#4c72b0", "#dd8452"),
        ("#6a4c93", "#ff6f61")
    ]
    pre_color, post_color = color_palettes[color_scheme - 1]

    # Convert results dictionary to a DataFrame
    df = pd.DataFrame({
        "Time": results["time"],
        "Group": results["group"],
        "ATT(g,t)": results["att"],
        "Std. Error": results["se"],
        "c": results["c"]
    })
    
    df["Group"] = df["Group"].astype(int)

    # Compute confidence intervals
    df["Lower CI"] = df["ATT(g,t)"] - df["c"] * df["Std. Error"]
    df["Upper CI"] = df["ATT(g,t)"] + df["c"] * df["Std. Error"]
    df["Center CI"] = (df["Upper CI"] + df["Lower CI"]) / 2
    df["Tile Height"] = df["Upper CI"] - df["Lower CI"]

    # Identify post-treatment periods
    df["Post"] = np.where(df["Time"] >= df["Group"], "Post-treatment", "Pre-treatment")
    df["Post"] = pd.Categorical(df["Post"], categories=["Pre-treatment", "Post-treatment"], ordered=True)

    # Select groups if specified
    unique_groups = df["Group"].unique()
    if group is None:
        group = unique_groups
    else:
        group = [g for g in group if g in unique_groups]

    plots = []
    
    for g in group:
        group_data = df[df["Group"] == g].copy()

        # Adjust x-axis if relative_time is enabled
        if relative_time:
            group_data["Rel_Time"] = group_data["Time"] - g
            x_var = "Rel_Time"
            x_labels = sorted(group_data["Rel_Time"].unique())
        else:
            x_var = "Time"
            x_labels = sorted(group_data["Time"].unique())

        # Ensure 0 is always included in x-axis breaks
        min_t, max_t = min(x_labels), max(x_labels)
        left_ticks = list(range(0, min_t-1, -xgap))
        right_ticks = list(range(0, max_t+1, xgap))
        dabreaks = sorted(set(left_ticks + right_ticks))

        # Adjust point size and tile width
        num_atts = len(group_data)
        if num_atts <= 6:
            point_size = 4
            tile_width = 0.5
        elif num_atts <= 12:
            point_size = 3
            tile_width = 0.7
        else:
            point_size = 2
            tile_width = 1

        # Set title conditionally
        title_text = f"{title} {g}" if title else None

        # Generate plot
        p = (
            ggplot(group_data, aes(x=x_var, y="ATT(g,t)", fill="Post"))
            + geom_tile(aes(y="Center CI", height="Tile Height", color="Post"), width=tile_width, alpha=0.4)
            + geom_point(aes(color="Post"), size=point_size)
            + scale_fill_manual(values=[pre_color, post_color])
            + scale_color_manual(values=[pre_color, post_color])
            + scale_x_continuous(breaks=dabreaks, labels=[str(b) for b in dabreaks])
            + scale_y_continuous(limits=ylim)
            + labs(x=xlab, y=ylab, title=title_text)
            + theme_minimal()
            + theme(
                figure_size=(5, 4),
                panel_background=element_rect(fill="white", color="white"),
                plot_background=element_rect(fill="white", color="white"),
                legend_position="bottom",
                legend_title=element_blank(),
                axis_text=element_text(size=10),
                axis_title=element_text(size=10),
                plot_title=element_text(size=14, face="bold") if title else element_blank(),
                strip_background=element_rect(fill="white", color="white"),
                
                # Add Dashed Soft Grey Grid Lines:
                panel_grid_major=element_line(color="grey", linetype="dashed", size=0.9, alpha=0.1),  # Soft grey dashed grid
                panel_grid_minor=element_blank(),  # Remove minor grid lines
                
                axis_line_x=element_line(color="black", size=1),
                axis_line_y=element_line(color="black", size=1),
                axis_ticks_major_x=element_line(color="black", size=1),
                axis_ticks_minor_x=element_blank(),
                axis_ticks_major_y=element_line(color="black", size=1),
                axis_ticks_minor_y=element_blank()
            )
        )

        # Add reference line if enabled
        if ref_line is not None:
            p += geom_hline(yintercept=ref_line, linetype="dashed", color="black")

        plots.append(p)

    return plots