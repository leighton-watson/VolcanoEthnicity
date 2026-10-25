"""Compare area- and building-weighted ashfall exposure estimates.

The script produces three figures and a CSV table:

* A two-panel comparison of total population exposed. Point colour identifies
  the eruption scenario, while marker shape and size identify ashfall
  thickness. The dashed line shows agreement between the methods.
* A two-panel heatmap of differences in exposed-population composition,
  calculated as building-weighted minus area-weighted percentage points. Every
  finite comparison is printed in its heatmap cell.
* A two-panel scatter plot of the relative difference in exposed ethnic-group
  counts, calculated as 100 * (area - building) / building, against the
  building-weighted number of people exposed. Values below 10 people are
  excluded, and solid and dashed lines show binned summaries.
* A wide CSV table containing the Figure 2 percentage-point differences, with
  one row per region, scenario, and ashfall threshold and one column per
  ethnicity.

Ethnicity is total-response data, so percentages need not sum to 100. Heatmap
rows for which either method estimates fewer than
``MIN_EXPOSED_FOR_COMPOSITION`` exposed people are masked because percentage
composition is unstable for very small populations.

Run this script from Spyder or from the project root after running
``Auckland_run_analysis.py`` and ``Taranaki_run_analysis.py``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


# -----------------------------------------------------------------------------
# Paths and figure settings
# -----------------------------------------------------------------------------
SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "OUTPUT"

AUCKLAND_FILE = OUTPUT_DIR / "Auckland" / "ALL_summary_long.csv"
TARANAKI_FILE = (
    OUTPUT_DIR / "Taranaki" / "analysis" / "ALL_summary_long.csv"
)

TOTAL_OUTPUT_PNG = OUTPUT_DIR / "Fig_area_vs_building_total_population.png"
TOTAL_OUTPUT_PDF = OUTPUT_DIR / "Fig_area_vs_building_total_population.pdf"
COMPOSITION_OUTPUT_PNG = (
    OUTPUT_DIR / "Fig_area_vs_building_ethnicity_percentage_points.png"
)
COMPOSITION_OUTPUT_PDF = (
    OUTPUT_DIR / "Fig_area_vs_building_ethnicity_percentage_points.pdf"
)
TREND_OUTPUT_PNG = (
    OUTPUT_DIR / "Fig_percentage_difference_vs_building_population.png"
)
TREND_OUTPUT_PDF = (
    OUTPUT_DIR / "Fig_percentage_difference_vs_building_population.pdf"
)
COMPOSITION_OUTPUT_CSV = (
    OUTPUT_DIR / "area_vs_building_ethnicity_percentage_points.csv"
)

TOTAL_FIGURE_SIZE = (15, 7.5)
COMPOSITION_FIGURE_SIZE = (15, 17)
TREND_FIGURE_SIZE = (15, 7.5)
OUTPUT_DPI = 300

AXIS_LABEL_FONTSIZE = 16
AXIS_TICK_FONTSIZE = 16
TITLE_FONTSIZE = 18
LEGEND_FONTSIZE = 12
LEGEND_TITLE_FONTSIZE = 14

# Display the completed figures in Spyder's Plots pane (or the active
# Matplotlib graphics backend) in addition to saving them. Set to False for
# save-only batch execution.
SHOW_FIGURES = True

# Mask ethnic-composition comparisons when either allocation method estimates
# fewer than this number of exposed people. The default displays every
# comparison for which both methods estimate a non-zero exposed population.
# Any comparison with zero total exposure is always reported as NA.
MIN_EXPOSED_FOR_COMPOSITION = 0.0

# Heatmap colours saturate beyond this absolute percentage-point difference.
# Saturated cells retain their numerical annotation.
HEATMAP_LIMIT_PP = 6.0

# In Figure 2, report the percentage-point difference as NA when fewer than
# this many people in the ethnic group are exposed under the building-weighted
# method. The exposed count is still printed in the cell. Figure 3 excludes
# these small-denominator comparisons.
MIN_ETHNIC_EXPOSED_FOR_PERCENTAGE = 10.0

# Population-size bins used to summarise Figure 3. The upper edge is excluded,
# except for the final open-ended bin.
POPULATION_BINS = [
    (10.0, 100.0, "10–99"),
    (100.0, 1_000.0, "100–999"),
    (1_000.0, 10_000.0, "1,000–9,999"),
    (10_000.0, 100_000.0, "10,000–99,999"),
    (100_000.0, np.inf, "≥100,000"),
]

METHOD_ORDER = ["area", "buildings"]
ETHNICITY_ORDER = ["Eurother", "Maori", "Pacific", "Asian", "MELAA"]
ETHNICITY_LABELS = {
    "Eurother": "European\nor Other",
    "Maori": "Māori",
    "Pacific": "Pacific",
    "Asian": "Asian",
    "MELAA": "MELAA",
}

AUCKLAND_SCENARIO_ORDER = list("ABCDEFGH")
TARANAKI_SCENARIO_ORDER = [
    "S1", "S2", "S3", "M1", "M2", "M3", "L1", "L2", "L3"
]

AUCKLAND_THRESHOLDS = [
    ("10_mm", "≥10 mm"),
    ("1_mm", "≥1 mm"),
    ("trace", "Trace"),
]
TARANAKI_THRESHOLDS = [
    ("100_mm", "≥100 mm"),
    ("10_mm", "≥10 mm"),
    ("1_mm", "≥1 mm"),
]

# Scenario colours match the corresponding Results bar-graph scripts.
AUCKLAND_SCENARIO_COLOURS = dict(
    zip(
        AUCKLAND_SCENARIO_ORDER,
        [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#999999",
        ],
    )
)
TARANAKI_SCENARIO_COLOURS = dict(
    zip(
        TARANAKI_SCENARIO_ORDER,
        [
            "#E69F00",
            "#56B4E9",
            "#009E73",
            "#F0E442",
            "#0072B2",
            "#D55E00",
            "#CC79A7",
            "#999999",
            "#000000",
        ],
    )
)

# Marker size increases with ashfall thickness. Shape also identifies the
# threshold so the figure remains interpretable when printed in greyscale.
THRESHOLD_STYLES = {
    "trace": {"marker": "o", "size": 48},
    "1_mm": {"marker": "^", "size": 70},
    "10_mm": {"marker": "s", "size": 94},
    "100_mm": {"marker": "D", "size": 122},
}


def require_columns(data, required, source):
    """Raise a clear error when an analysis output has an unexpected schema."""
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise ValueError(
            f"{source} is missing required columns: {', '.join(missing)}"
        )


def load_region_data(path, region, total_column):
    """Load and validate one region's long-format comparison table."""
    if not path.exists():
        raise FileNotFoundError(
            f"Could not find {region} comparison data: {path}\n"
            f"Run the {region} analysis before making this figure."
        )

    data = pd.read_csv(path)
    required = {
        "scenario",
        "method",
        "ash_threshold_key",
        "ethnicity",
        "exposed",
        "percent_of_exposed_population",
        total_column,
    }
    require_columns(data, required, path)

    data = data.loc[data["method"].isin(METHOD_ORDER)].copy()
    data["scenario"] = data["scenario"].astype(str).str.upper()
    data[total_column] = pd.to_numeric(data[total_column], errors="raise")
    data["exposed"] = pd.to_numeric(data["exposed"], errors="raise")
    data["percent_of_exposed_population"] = pd.to_numeric(
        data["percent_of_exposed_population"], errors="raise"
    )

    if data.empty:
        raise ValueError(f"No area/buildings rows were found in {path}")

    return data


def prepare_total_comparison(data, total_column):
    """Return one paired total-population estimate per scenario and threshold."""
    totals = data[
        ["scenario", "ash_threshold_key", "method", total_column]
    ].drop_duplicates()

    duplicate_counts = totals.groupby(
        ["scenario", "ash_threshold_key", "method"]
    ).size()
    if (duplicate_counts > 1).any():
        duplicates = duplicate_counts.loc[duplicate_counts > 1]
        raise ValueError(
            "Conflicting total-population values were found for:\n"
            + duplicates.to_string()
        )

    paired = totals.pivot(
        index=["scenario", "ash_threshold_key"],
        columns="method",
        values=total_column,
    ).reset_index()

    missing_methods = [method for method in METHOD_ORDER if method not in paired]
    if missing_methods:
        raise ValueError(
            "The comparison table is missing methods: "
            + ", ".join(missing_methods)
        )

    missing_pairs = paired.loc[paired[METHOD_ORDER].isna().any(axis=1)]
    if not missing_pairs.empty:
        print(
            "Warning: dropping scenario-threshold combinations without both "
            "allocation methods:\n"
            + missing_pairs[
                ["scenario", "ash_threshold_key"] + METHOD_ORDER
            ].to_string(index=False),
            flush=True,
        )
        paired = paired.dropna(subset=METHOD_ORDER).copy()

    # Express the difference relative to the preferred building-weighted
    # estimate, consistent with the relative-difference plot below.
    denominator = paired["buildings"].replace(0, np.nan)
    paired["signed_percent_difference"] = (
        100.0 * (paired["area"] - paired["buildings"]) / denominator
    )

    # Symmetric difference is used only to rank labels. It treats both methods
    # equally and remains bounded when one estimate is zero.
    symmetric_denominator = paired["area"].abs() + paired["buildings"].abs()
    paired["symmetric_percent_difference"] = np.where(
        symmetric_denominator > 0,
        200.0
        * (paired["buildings"] - paired["area"])
        / symmetric_denominator,
        np.nan,
    )
    return paired


def prepare_composition_difference(data, totals, scenario_order, thresholds):
    """Return composition differences and area/building exposed counts."""
    composition = data.pivot_table(
        index=["scenario", "ash_threshold_key", "ethnicity"],
        columns="method",
        values="percent_of_exposed_population",
        aggfunc="first",
    ).reset_index()

    missing_methods = [
        method for method in METHOD_ORDER if method not in composition.columns
    ]
    if missing_methods:
        raise ValueError(
            "The ethnicity comparison is missing methods: "
            + ", ".join(missing_methods)
        )

    composition["difference_pp"] = (
        composition["buildings"] - composition["area"]
    )
    # The composition table already uses ``area`` and ``buildings`` for the
    # two percentage estimates. Rename the total-population columns before the
    # merge so pandas does not replace both pairs with ``_x``/``_y`` suffixes.
    population_totals = totals[
        ["scenario", "ash_threshold_key", "area", "buildings"]
    ].rename(
        columns={
            "area": "area_total_population",
            "buildings": "building_total_population",
        }
    )
    composition = composition.merge(
        population_totals,
        on=["scenario", "ash_threshold_key"],
        how="left",
        validate="many_to_one",
    )

    unstable = (
        (
            composition["area_total_population"]
            < MIN_EXPOSED_FOR_COMPOSITION
        )
        | (
            composition["building_total_population"]
            < MIN_EXPOSED_FOR_COMPOSITION
        )
        | (composition["area_total_population"] <= 0)
        | (composition["building_total_population"] <= 0)
    )
    composition.loc[unstable, "difference_pp"] = np.nan

    row_keys = [
        (scenario, threshold_key)
        for threshold_key, _ in thresholds
        for scenario in scenario_order
    ]
    row_index = pd.MultiIndex.from_tuples(
        row_keys, names=["scenario", "ash_threshold_key"]
    )

    matrix = composition.pivot_table(
        index=["scenario", "ash_threshold_key"],
        columns="ethnicity",
        values="difference_pp",
        aggfunc="first",
        dropna=False,
    )
    matrix = matrix.reindex(index=row_index, columns=ETHNICITY_ORDER)

    exposed_by_method = {}
    for method in METHOD_ORDER:
        exposed_by_method[method] = data.loc[
            data["method"] == method
        ].pivot_table(
            index=["scenario", "ash_threshold_key"],
            columns="ethnicity",
            values="exposed",
            aggfunc="first",
            dropna=False,
        )
        exposed_by_method[method] = exposed_by_method[method].reindex(
            index=row_index,
            columns=ETHNICITY_ORDER,
        )

    area_exposed = exposed_by_method["area"]
    building_exposed = exposed_by_method["buildings"]
    return matrix, area_exposed, building_exposed


def add_panel_label(ax, label):
    """Add a bold panel label above an axis."""
    ax.text(
        -0.10,
        1.05,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=14,
        fontweight="bold",
    )


def threshold_label_lookup(thresholds):
    """Return a key-to-display-label dictionary."""
    return dict(thresholds)


def plot_total_comparison(
    ax,
    paired,
    panel_title,
    thresholds,
    scenario_colours,
    axis_maximum,
):
    """Plot paired total-population estimates for one region."""
    for scenario, colour in scenario_colours.items():
        scenario_data = paired.loc[paired["scenario"] == scenario]
        for _, row in scenario_data.iterrows():
            threshold_key = row["ash_threshold_key"]
            if threshold_key not in THRESHOLD_STYLES:
                raise ValueError(
                    f"No marker style is defined for '{threshold_key}'"
                )
            style = THRESHOLD_STYLES[threshold_key]
            ax.scatter(
                row["area"],
                row["buildings"],
                s=style["size"],
                marker=style["marker"],
                color=colour,
                edgecolor="black",
                linewidth=0.7,
                alpha=0.92,
                zorder=3,
            )

    ax.plot(
        [0, axis_maximum],
        [0, axis_maximum],
        linestyle="--",
        color="0.35",
        linewidth=1.2,
        zorder=1,
    )

    # A symmetric-log scale preserves zero estimates while still displaying
    # populations spanning several orders of magnitude.
    linear_threshold = max(10.0, min(100.0, axis_maximum / 1000.0))
    ax.set_xscale("symlog", linthresh=linear_threshold)
    ax.set_yscale("symlog", linthresh=linear_threshold)
    ax.set_xlim(left=0, right=axis_maximum)
    ax.set_ylim(bottom=0, top=axis_maximum)

    ax.set_xlabel("Area-weighted exposed population")
    ax.set_ylabel("Building-weighted exposed population")
    ax.set_title(
        panel_title,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=AXIS_TICK_FONTSIZE,
    )
    ax.grid(True, which="major", color="0.88", linewidth=0.7, zorder=0)
    ax.grid(True, which="minor", color="0.94", linewidth=0.5, zorder=0)

    scenario_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor=colour,
            markeredgecolor="black",
            markeredgewidth=0.6,
            markersize=7,
            label=scenario,
        )
        for scenario, colour in scenario_colours.items()
    ]
    scenario_legend = ax.legend(
        handles=scenario_handles,
        title="Scenario",
        loc="upper left",
        ncol=3,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=True,
        framealpha=0.92,
        borderpad=0.5,
        columnspacing=0.9,
        handletextpad=0.4,
    )
    ax.add_artist(scenario_legend)

    threshold_handles = [
        Line2D(
            [0],
            [0],
            marker=THRESHOLD_STYLES[key]["marker"],
            linestyle="none",
            markerfacecolor="0.70",
            markeredgecolor="black",
            markeredgewidth=0.7,
            markersize=np.sqrt(THRESHOLD_STYLES[key]["size"]),
            label=label,
        )
        for key, label in thresholds
    ]
    ax.legend(
        handles=threshold_handles,
        title="Ashfall threshold",
        loc="lower right",
        ncol=1,
        fontsize=LEGEND_FONTSIZE,
        title_fontsize=LEGEND_TITLE_FONTSIZE,
        frameon=True,
        framealpha=0.92,
        borderpad=0.7,
        labelspacing=1.25,
        handletextpad=0.9,
    )


def format_heatmap_row_labels(matrix, thresholds):
    """Format scenario-threshold labels in the matrix's current order."""
    label_lookup = threshold_label_lookup(thresholds)
    return [
        f"{scenario}  {label_lookup.get(threshold, threshold)}"
        for scenario, threshold in matrix.index
    ]


def prepare_composition_csv(region, matrix, building_exposed, thresholds):
    """Return the displayed composition table in an Excel-friendly layout."""
    displayed = matrix.mask(
        building_exposed < MIN_ETHNIC_EXPOSED_FOR_PERCENTAGE
    ).copy()
    displayed = displayed.reset_index()

    threshold_labels = threshold_label_lookup(thresholds)
    displayed["ash_threshold_key"] = displayed[
        "ash_threshold_key"
    ].map(threshold_labels).fillna(displayed["ash_threshold_key"])

    displayed.insert(0, "Region", region)
    displayed = displayed.rename(
        columns={
            "scenario": "Scenario",
            "ash_threshold_key": "Ashfall threshold",
            **{
                ethnicity: ETHNICITY_LABELS.get(ethnicity, ethnicity).replace(
                    "\n", " "
                )
                for ethnicity in ETHNICITY_ORDER
            },
        }
    )
    return displayed


def plot_composition_heatmap(
    ax,
    matrix,
    building_exposed,
    panel_title,
    thresholds,
    norm,
    cmap,
):
    """Plot composition differences with building-weighted exposed counts."""
    values = matrix.to_numpy(dtype=float)
    exposed_counts = building_exposed.to_numpy(dtype=float)
    display_values = values.copy()
    display_values[
        (~np.isfinite(exposed_counts))
        | (exposed_counts < MIN_ETHNIC_EXPOSED_FOR_PERCENTAGE)
    ] = np.nan
    masked_values = np.ma.masked_invalid(display_values)
    image = ax.imshow(
        masked_values,
        aspect="auto",
        interpolation="none",
        cmap=cmap,
        norm=norm,
    )

    ax.set_xticks(np.arange(len(ETHNICITY_ORDER)))
    ax.set_xticklabels(
        [ETHNICITY_LABELS[key] for key in ETHNICITY_ORDER],
        fontsize=AXIS_TICK_FONTSIZE,
    )
    ax.set_yticks(np.arange(len(matrix.index)))
    ax.set_yticklabels(
        format_heatmap_row_labels(matrix, thresholds),
        fontsize=AXIS_TICK_FONTSIZE,
    )
    ax.set_title(
        panel_title,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Ethnic group")
    ax.set_ylabel("Scenario and ashfall threshold")
    ax.tick_params(axis="both", which="both", labelsize=AXIS_TICK_FONTSIZE)

    # Delineate the threshold blocks.
    scenarios_per_threshold = len(matrix.index) // len(thresholds)
    for block in range(1, len(thresholds)):
        ax.axhline(
            block * scenarios_per_threshold - 0.5,
            color="black",
            linewidth=1.0,
        )

    # Print the percentage-point difference on the first line and the
    # building-weighted exposed population for that ethnicity on the second.
    # Values outside the colour limits remain visible numerically even though
    # their colours saturate. N/A indicates that the difference is undefined
    # or that fewer than 10 people are exposed in that ethnic group.
    for row_index in range(values.shape[0]):
        for column_index in range(values.shape[1]):
            value = display_values[row_index, column_index]
            exposed_count = exposed_counts[row_index, column_index]
            if not np.isfinite(exposed_count):
                count_label = "—"
            else:
                count_label = f"{exposed_count:,.0f}"
            if not np.isfinite(value):
                label = f"N/A\n{count_label}"
                text_colour = "0.35"
            else:
                label = f"{value:+.2f}\n{count_label}"
                text_colour = (
                    "white"
                    if abs(value) >= 0.60 * HEATMAP_LIMIT_PP
                    else "black"
                )
            ax.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                fontsize=14,
                color=text_colour,
                linespacing=1.20,
            )

    return image


def flatten_relative_exposure_values(area_exposed, building_exposed):
    """Return building counts and relative area-minus-building differences."""
    aligned_area = area_exposed.reindex(
        index=building_exposed.index,
        columns=building_exposed.columns,
    )
    area_counts = aligned_area.to_numpy(dtype=float).ravel()
    exposed_counts = building_exposed.to_numpy(dtype=float).ravel()
    with np.errstate(divide="ignore", invalid="ignore"):
        differences = 100.0 * (
            area_counts - exposed_counts
        ) / exposed_counts
    finite = (
        np.isfinite(differences)
        & np.isfinite(area_counts)
        & np.isfinite(exposed_counts)
        & (exposed_counts >= MIN_ETHNIC_EXPOSED_FOR_PERCENTAGE)
    )
    return exposed_counts[finite], differences[finite]


def add_population_bin_summaries(
    ax,
    exposed_counts,
    differences,
    x_maximum,
):
    """Add median and 10th/90th-percentile lines within population bins."""
    for lower, upper, _ in POPULATION_BINS:
        if np.isfinite(upper):
            in_bin = (exposed_counts >= lower) & (exposed_counts < upper)
            line_right = min(upper, x_maximum)
        else:
            in_bin = exposed_counts >= lower
            line_right = x_maximum

        line_left = max(lower, MIN_ETHNIC_EXPOSED_FOR_PERCENTAGE)
        if not np.any(in_bin) or line_right <= line_left:
            continue

        median = float(np.median(differences[in_bin]))
        percentile_10 = float(np.percentile(differences[in_bin], 10))
        percentile_90 = float(np.percentile(differences[in_bin], 90))
        ax.hlines(
            median,
            line_left,
            line_right,
            color="black",
            linewidth=3.0,
            linestyle="-",
            zorder=5,
        )
        ax.hlines(
            percentile_10,
            line_left,
            line_right,
            color="black",
            linewidth=2.0,
            linestyle="--",
            zorder=5,
        )
        ax.hlines(
            percentile_90,
            line_left,
            line_right,
            color="black",
            linewidth=2.0,
            linestyle="--",
            zorder=5,
        )

    for boundary in (100.0, 1_000.0, 10_000.0, 100_000.0):
        if boundary < x_maximum:
            ax.axvline(
                boundary,
                color="0.72",
                linewidth=0.8,
                linestyle=":",
                zorder=1,
            )


def plot_difference_vs_population(
    ax,
    area_exposed,
    building_exposed,
    panel_title,
    x_maximum,
    y_maximum,
):
    """Plot relative method differences against exposed ethnic populations."""
    exposed_counts, differences = flatten_relative_exposure_values(
        area_exposed,
        building_exposed,
    )

    ax.scatter(
        exposed_counts,
        differences,
        s=48,
        marker="o",
        facecolor="black",
        edgecolor="black",
        linewidth=0.4,
        alpha=0.72,
        zorder=3,
    )
    add_population_bin_summaries(
        ax,
        exposed_counts,
        differences,
        x_maximum,
    )
    ax.axhline(
        0,
        color="0.35",
        linestyle="--",
        linewidth=1.2,
        zorder=1,
    )

    # Values below 10 people are excluded, allowing a conventional logarithmic
    # population axis. Relative differences use a linear y-axis.
    ax.set_xscale("log")
    ax.set_xlim(
        left=MIN_ETHNIC_EXPOSED_FOR_PERCENTAGE,
        right=x_maximum,
    )
    ax.set_ylim(bottom=-y_maximum, top=y_maximum)

    ax.set_title(
        panel_title,
        fontsize=TITLE_FONTSIZE,
        fontweight="bold",
    )
    ax.set_xlabel("Building-weighted exposed population")
    ax.set_ylabel(
        "Relative exposure difference\n"
        "100 × (area - building) / building (%)"
    )
    ax.tick_params(
        axis="both",
        which="both",
        labelsize=AXIS_TICK_FONTSIZE,
    )
    ax.grid(True, which="major", color="0.88", linewidth=0.7, zorder=0)
    ax.grid(True, which="minor", color="0.94", linewidth=0.5, zorder=0)

    legend_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="black",
            markeredgecolor="black",
            markersize=6,
            alpha=0.72,
            label="Individual comparison",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=3.0,
            linestyle="-",
            label="Bin median",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            linewidth=2.0,
            linestyle="--",
            label="Bin 10th/90th percentiles",
        ),
    ]
    ax.legend(
        handles=legend_handles,
        loc="upper right",
        frameon=True,
        framealpha=0.92,
        fontsize=LEGEND_FONTSIZE,
    )


def print_summary(region, paired, composition_matrix, building_exposed):
    """Print manuscript-oriented comparison statistics."""
    relative = paired["signed_percent_difference"].dropna()
    absolute_relative = relative.abs()
    # Match the values displayed in Figure 2 by excluding cells reported as
    # NA because the building-weighted ethnic-group count is too small.
    displayed_composition = composition_matrix.mask(
        building_exposed < MIN_ETHNIC_EXPOSED_FOR_PERCENTAGE
    )
    composition_values = displayed_composition.to_numpy(dtype=float)
    finite_composition = np.abs(
        composition_values[np.isfinite(composition_values)]
    )

    print(f"\n{region}")
    print("-" * len(region))
    print(f"Paired scenario-threshold comparisons: {len(paired)}")
    if len(absolute_relative):
        print(
            "Median absolute difference in total exposure: "
            f"{absolute_relative.median():.2f}%"
        )
        for tolerance in (1, 5, 10):
            count = int((absolute_relative <= tolerance).sum())
            print(
                f"Comparisons within {tolerance}%: "
                f"{count} of {len(absolute_relative)}"
            )

        large_comparisons = paired.loc[
            paired["buildings"] > 10_000,
            "signed_percent_difference",
        ].dropna().abs()
        print(
            "Comparisons with more than 10,000 people exposed "
            f"(building-weighted): {len(large_comparisons)}"
        )
        if len(large_comparisons):
            print(
                "Median absolute difference for these comparisons: "
                f"{large_comparisons.median():.2f}%"
            )
            print(
                "Maximum absolute difference for these comparisons: "
                f"{large_comparisons.max():.2f}%"
            )
    if len(finite_composition):
        print(
            "Median absolute ethnic-composition difference: "
            f"{np.median(finite_composition):.2f} percentage points"
        )
        print(
            "Maximum absolute ethnic-composition difference: "
            f"{np.max(finite_composition):.2f} percentage points"
        )
        print("Ethnic-composition ranges (building - area):")
        for ethnicity in displayed_composition.columns:
            differences = displayed_composition[ethnicity].dropna()
            label = ETHNICITY_LABELS.get(ethnicity, ethnicity).replace(
                "\n", " "
            )
            if differences.empty:
                print(f"  {label}: no valid comparisons")
            else:
                print(
                    f"  {label}: {differences.min():.2f} to "
                    f"{differences.max():.2f} percentage points"
                )
    print(
        "Composition cells are reported as NA when the building-weighted "
        "ethnic-group estimate is below "
        f"{MIN_ETHNIC_EXPOSED_FOR_PERCENTAGE:,.0f} people, or when either "
        "method estimates zero total exposure."
    )


def main():
    """Load the analysis outputs, construct the figure, and save it."""
    auckland_data = load_region_data(
        AUCKLAND_FILE,
        region="Auckland",
        total_column="exposed_population",
    )
    taranaki_data = load_region_data(
        TARANAKI_FILE,
        region="Taranaki",
        total_column="total_exposed_population",
    )

    auckland_totals = prepare_total_comparison(
        auckland_data, total_column="exposed_population"
    )
    taranaki_totals = prepare_total_comparison(
        taranaki_data, total_column="total_exposed_population"
    )

    (
        auckland_composition,
        auckland_area_exposed,
        auckland_building_exposed,
    ) = prepare_composition_difference(
        auckland_data,
        auckland_totals,
        AUCKLAND_SCENARIO_ORDER,
        AUCKLAND_THRESHOLDS,
    )
    (
        taranaki_composition,
        taranaki_area_exposed,
        taranaki_building_exposed,
    ) = prepare_composition_difference(
        taranaki_data,
        taranaki_totals,
        TARANAKI_SCENARIO_ORDER,
        TARANAKI_THRESHOLDS,
    )

    composition_csv = pd.concat(
        [
            prepare_composition_csv(
                "Auckland",
                auckland_composition,
                auckland_building_exposed,
                AUCKLAND_THRESHOLDS,
            ),
            prepare_composition_csv(
                "Taranaki",
                taranaki_composition,
                taranaki_building_exposed,
                TARANAKI_THRESHOLDS,
            ),
        ],
        ignore_index=True,
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    composition_csv.to_csv(
        COMPOSITION_OUTPUT_CSV,
        index=False,
        na_rep="N/A",
    )

    print_summary(
        "Auckland",
        auckland_totals,
        auckland_composition,
        auckland_building_exposed,
    )
    print_summary(
        "Taranaki",
        taranaki_totals,
        taranaki_composition,
        taranaki_building_exposed,
    )

    plt.rcParams.update(
        {
            "font.size": AXIS_TICK_FONTSIZE,
            "axes.titlesize": TITLE_FONTSIZE,
            "axes.labelsize": AXIS_LABEL_FONTSIZE,
            "xtick.labelsize": AXIS_TICK_FONTSIZE,
            "ytick.labelsize": AXIS_TICK_FONTSIZE,
            "legend.fontsize": LEGEND_FONTSIZE,
            "legend.title_fontsize": LEGEND_TITLE_FONTSIZE,
            "figure.dpi": 120,
        }
    )

    # ------------------------------------------------------------------
    # Figure 1: paired total exposed populations
    # ------------------------------------------------------------------
    total_figure, total_axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=TOTAL_FIGURE_SIZE,
        constrained_layout=True,
    )
    ax_auckland_total, ax_taranaki_total = total_axes

    common_axis_maximum = 1.05 * max(
        float(auckland_totals[["area", "buildings"]].max().max()),
        float(taranaki_totals[["area", "buildings"]].max().max()),
    )
    common_axis_maximum = max(1.0, common_axis_maximum)

    plot_total_comparison(
        ax_auckland_total,
        auckland_totals,
        "(a) Auckland",
        AUCKLAND_THRESHOLDS,
        AUCKLAND_SCENARIO_COLOURS,
        common_axis_maximum,
    )
    plot_total_comparison(
        ax_taranaki_total,
        taranaki_totals,
        "(b) Taranaki",
        TARANAKI_THRESHOLDS,
        TARANAKI_SCENARIO_COLOURS,
        common_axis_maximum,
    )

    total_figure.savefig(
        TOTAL_OUTPUT_PNG,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )
    total_figure.savefig(TOTAL_OUTPUT_PDF, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Figure 2: building-minus-area ethnic composition differences
    # ------------------------------------------------------------------
    composition_figure, composition_axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=COMPOSITION_FIGURE_SIZE,
        constrained_layout=True,
    )
    ax_auckland_composition, ax_taranaki_composition = composition_axes

    colour_map = plt.get_cmap("RdBu_r").copy()
    colour_map.set_bad(color="0.88")
    colour_norm = TwoSlopeNorm(
        vmin=-HEATMAP_LIMIT_PP,
        vcenter=0.0,
        vmax=HEATMAP_LIMIT_PP,
    )

    plot_composition_heatmap(
        ax_auckland_composition,
        auckland_composition,
        auckland_building_exposed,
        "(a) Auckland",
        AUCKLAND_THRESHOLDS,
        colour_norm,
        colour_map,
    )
    heatmap_image = plot_composition_heatmap(
        ax_taranaki_composition,
        taranaki_composition,
        taranaki_building_exposed,
        "(b) Taranaki",
        TARANAKI_THRESHOLDS,
        colour_norm,
        colour_map,
    )

    colour_bar = composition_figure.colorbar(
        heatmap_image,
        ax=[ax_auckland_composition, ax_taranaki_composition],
        orientation="horizontal",
        fraction=0.045,
        pad=0.03,
        extend="both",
    )
    colour_bar.set_label(
        "Difference in exposed-population composition "
        "(building - area, percentage points)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    colour_bar.ax.tick_params(labelsize=AXIS_TICK_FONTSIZE)

    composition_figure.savefig(
        COMPOSITION_OUTPUT_PNG,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )
    composition_figure.savefig(COMPOSITION_OUTPUT_PDF, bbox_inches="tight")

    # ------------------------------------------------------------------
    # Figure 3: relative count difference versus exposed population
    # ------------------------------------------------------------------
    auckland_x, auckland_y = flatten_relative_exposure_values(
        auckland_area_exposed,
        auckland_building_exposed,
    )
    taranaki_x, taranaki_y = flatten_relative_exposure_values(
        taranaki_area_exposed,
        taranaki_building_exposed,
    )
    all_x = np.concatenate([auckland_x, taranaki_x])
    all_y = np.concatenate([auckland_y, taranaki_y])
    if all_x.size == 0:
        raise ValueError("No finite values are available for the trend figure")

    trend_x_maximum = max(1.0, 1.05 * float(np.max(all_x)))
    trend_y_maximum = max(1.0, 1.05 * float(np.max(np.abs(all_y))))

    trend_figure, trend_axes = plt.subplots(
        nrows=1,
        ncols=2,
        figsize=TREND_FIGURE_SIZE,
        constrained_layout=True,
    )
    ax_auckland_trend, ax_taranaki_trend = trend_axes
    plot_difference_vs_population(
        ax_auckland_trend,
        auckland_area_exposed,
        auckland_building_exposed,
        "(a) Auckland",
        trend_x_maximum,
        trend_y_maximum,
    )
    plot_difference_vs_population(
        ax_taranaki_trend,
        taranaki_area_exposed,
        taranaki_building_exposed,
        "(b) Taranaki",
        trend_x_maximum,
        trend_y_maximum,
    )
    trend_figure.savefig(
        TREND_OUTPUT_PNG,
        dpi=OUTPUT_DPI,
        bbox_inches="tight",
    )
    trend_figure.savefig(TREND_OUTPUT_PDF, bbox_inches="tight")

    print(f"\nSaved: {TOTAL_OUTPUT_PNG}")
    print(f"Saved: {TOTAL_OUTPUT_PDF}")
    print(f"Saved: {COMPOSITION_OUTPUT_PNG}")
    print(f"Saved: {COMPOSITION_OUTPUT_PDF}")
    print(f"Saved: {TREND_OUTPUT_PNG}")
    print(f"Saved: {TREND_OUTPUT_PDF}")
    print(f"Saved: {COMPOSITION_OUTPUT_CSV}")

    if SHOW_FIGURES:
        plt.show()
    else:
        plt.close(total_figure)
        plt.close(composition_figure)
        plt.close(trend_figure)


if __name__ == "__main__":
    main()
