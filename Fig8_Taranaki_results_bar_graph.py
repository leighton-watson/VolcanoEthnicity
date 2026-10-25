"""Plot Taranaki ashfall exposure percentages and total population.

Run ``run_Taranaki_analysis.py`` first. Each threshold row compares estimated
ethnicity percentages in the left panel with Taranaki regional and national
population percentages, and total exposed population in the right panel across
the nine scenarios.

Ethnicity is total-response data, so the percentages need not sum to 100.
"""

# %%
from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# %%
# -----------------------------------------------------------------------------
# Paths and plotting settings
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_FOLDER = (
    PROJECT_ROOT / "MakeFigures" / "OUTPUT" / "Taranaki" / "analysis"
)
TARANAKI_SUMMARY_FILE = ANALYSIS_FOLDER / "ALL_summary_pivot.csv"
TARANAKI_POPULATION_FILE = ANALYSIS_FOLDER / "ALL_population_exposed.csv"
NATIONAL_PERCENT_FILE = (
    PROJECT_ROOT / "OUTPUT" / "national_ethnicity_percentages_sa2_2018.csv"
)
OUTPUT_FOLDER = PROJECT_ROOT / "MakeFigures" / "OUTPUT"
OUTPUT_FILE = OUTPUT_FOLDER / "Fig_Taranaki_results.png"

# Use "area" or "buildings". This should match one of the methods produced by
# run_Taranaki_analysis.py.
METHOD_TO_PLOT = "buildings"

# Threshold rows are plotted in this order.
THRESHOLDS_TO_PLOT = [
    ("100_mm", "Ashfall ≥ 100 mm"),
    ("10_mm", "Ashfall ≥ 10 mm"),
    ("1_mm", "Ashfall ≥ 1 mm"),
]

SCENARIO_ORDER = ["S1", "S2", "S3", "M1", "M2", "M3", "L1", "L2", "L3"]
ETHNICITY_ORDER = [
    ("Eurother", "European\nor Other", "European or Other"),
    ("Maori", "Māori", "Maori"),
    ("Pacific", "Pacific", "Pacific"),
    ("Asian", "Asian", "Asian"),
    ("MELAA", "MELAA", "MELAA"),
]
NATIONAL_BAR_LABEL = "National"
REGIONAL_BAR_LABEL = "Taranaki region"

# Taranaki regional ethnicity percentages for comparison with the scenario
# exposure estimates and national population percentages.
REGIONAL_PERCENTAGES = {
    "Eurother": 85.5,
    "Maori": 20.2,
    "Pacific": 2.2,
    "Asian": 4.8,
    "MELAA": 0.6,
}

# The eight-colour palette used by the Auckland figure, extended with black for
# the ninth Taranaki scenario. The national comparison remains white + hatched.
SCENARIO_COLOURS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # grey
    "#000000",  # black
]

FIGURE_SIZE = (18, 19)
OUTPUT_DPI = 300
AXIS_LABEL_FONTSIZE = 14
TICK_FONTSIZE = 14
LEGEND_FONTSIZE = 14
PANEL_LABEL_FONTSIZE = 16

SCRIPT_START = perf_counter()


def report_progress(message):
    """Print an immediately flushed progress message with elapsed time."""
    elapsed = perf_counter() - SCRIPT_START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def load_taranaki_percentages(summary_path, method, threshold_key):
    """Load and order the Taranaki ethnicity exposure percentages."""
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Taranaki summary not found: {summary_path}\n"
            "Run MakeFigures/run_Taranaki_analysis.py first."
        )

    report_progress(f"Reading Taranaki percentages: {summary_path}")
    data = pd.read_csv(summary_path)
    required_column = "ash_threshold_key"
    if required_column not in data.columns:
        raise ValueError(
            f"The summary has no '{required_column}' column. Run the updated "
            "MakeFigures/run_Taranaki_analysis.py first."
        )
    data = data.loc[
        (data["method"] == method)
        & (data["ash_threshold_key"] == threshold_key)
    ].copy()
    if data.empty:
        all_data = pd.read_csv(summary_path)
        available = (
            all_data[["method", "ash_threshold_key"]]
            .drop_duplicates()
            .sort_values(["method", "ash_threshold_key"])
            .to_dict("records")
        )
        raise ValueError(
            f"No rows found for method '{method}' and threshold "
            f"'{threshold_key}'. Available combinations: {available}"
        )

    missing_scenarios = [
        scenario
        for scenario in SCENARIO_ORDER
        if scenario not in set(data["scenario"].astype(str))
    ]
    if missing_scenarios:
        raise ValueError(
            "The Taranaki summary is missing scenarios: "
            + ", ".join(missing_scenarios)
        )

    data["scenario"] = pd.Categorical(
        data["scenario"].astype(str),
        categories=SCENARIO_ORDER,
        ordered=True,
    )
    data = data.sort_values("scenario")

    for field, _, _ in ETHNICITY_ORDER:
        data[field] = pd.to_numeric(data[field], errors="raise")
        if data[field].max(skipna=True) <= 1:
            data[field] = 100 * data[field]

    report_progress(
        f"Prepared {len(data):,} Taranaki scenarios for method '{method}' "
        f"and threshold '{threshold_key}'"
    )
    return data


def load_exposed_population_totals(population_path, method, threshold_key):
    """Load and order the estimated total population exposed per scenario."""
    if not population_path.exists():
        raise FileNotFoundError(
            f"Taranaki population summary not found: {population_path}\n"
            "Run MakeFigures/run_Taranaki_analysis.py first."
        )

    report_progress(f"Reading total exposed population: {population_path}")
    data = pd.read_csv(population_path)
    required_column = "ash_threshold_key"
    if required_column not in data.columns:
        raise ValueError(
            f"The population summary has no '{required_column}' column. Run "
            "the updated MakeFigures/run_Taranaki_analysis.py first."
        )
    data = data.loc[
        (data["method"] == method)
        & (data["ash_threshold_key"] == threshold_key)
    ].copy()
    if data.empty:
        available_methods = sorted(
            pd.read_csv(population_path, usecols=["method"])["method"]
            .dropna()
            .unique()
        )
        raise ValueError(
            f"No population totals found for method '{method}' and threshold "
            f"'{threshold_key}'. Available methods: {available_methods}"
        )

    data["exposed_population"] = pd.to_numeric(
        data["exposed_population"],
        errors="raise",
    )
    data["scenario"] = data["scenario"].astype(str)
    totals = data.set_index("scenario")["exposed_population"]
    totals = totals.reindex(SCENARIO_ORDER)
    if totals.isna().any():
        missing = totals.index[totals.isna()].tolist()
        raise ValueError(f"Missing exposed population totals for: {missing}")

    print(
        f"\nEstimated population exposed ({method} method, {threshold_key}):",
        flush=True,
    )
    for scenario, total in totals.items():
        print(f"  Scenario {scenario}: {total:,.0f} people", flush=True)
    print(flush=True)
    return totals


def load_national_percentages(national_path):
    """Load national 2018 ethnicity percentages for comparison."""
    if not national_path.exists():
        raise FileNotFoundError(
            f"National ethnicity percentages not found: {national_path}"
        )

    report_progress(f"Reading national percentages: {national_path}")
    national = pd.read_csv(national_path)
    lookup = {}
    for _, display_label, national_label in ETHNICITY_ORDER:
        match = national.loc[national["ethnicity"] == national_label]
        if match.empty:
            raise ValueError(f"Could not find national ethnicity '{national_label}'")
        lookup[display_label] = float(
            match["percent_of_total_population"].iloc[0]
        )
    return lookup


def add_panel_label(ax, label):
    """Add a bold panel label in the upper-left corner of an axis."""
    ax.text(
        0.01,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=PANEL_LABEL_FONTSIZE,
        fontweight="bold",
    )


def plot_threshold_row(
    ax_percent,
    ax_total,
    taranaki_data,
    regional_percentages,
    national_percentages,
    exposed_population_totals,
    threshold_title,
    panel_labels,
    percent_upper_limit,
    total_upper_limit,
):
    """Plot one threshold's ethnicity percentages and population totals."""
    scenario_labels = taranaki_data["scenario"].astype(str).tolist()
    n_scenarios = len(scenario_labels)
    n_bars = n_scenarios + 2
    group_positions = np.arange(len(ETHNICITY_ORDER))
    bar_width = 0.80 / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width

    for scenario_index, scenario in enumerate(scenario_labels):
        scenario_row = taranaki_data.loc[
            taranaki_data["scenario"].astype(str) == scenario
        ].iloc[0]
        values = [scenario_row[field] for field, _, _ in ETHNICITY_ORDER]
        ax_percent.bar(
            group_positions + offsets[scenario_index],
            values,
            width=bar_width,
            color=SCENARIO_COLOURS[scenario_index],
            label=f"Scenario {scenario}",
        )

    regional_values = [
        regional_percentages[field]
        for field, _, _ in ETHNICITY_ORDER
    ]
    ax_percent.bar(
        group_positions + offsets[-2],
        regional_values,
        width=bar_width,
        color="0.75",
        edgecolor="black",
        linewidth=1.2,
        hatch="..",
        label=REGIONAL_BAR_LABEL,
    )

    national_values = [
        national_percentages[display_label]
        for _, display_label, _ in ETHNICITY_ORDER
    ]
    ax_percent.bar(
        group_positions + offsets[-1],
        national_values,
        width=bar_width,
        color="white",
        edgecolor="black",
        linewidth=1.2,
        hatch="//",
        label=NATIONAL_BAR_LABEL,
    )

    ax_percent.set_xticks(group_positions)
    ax_percent.set_xticklabels(
        [display_label for _, display_label, _ in ETHNICITY_ORDER],
        fontsize=TICK_FONTSIZE,
    )
    ax_percent.set_ylabel(
        "Percent",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax_percent.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_percent.set_ylim(0, percent_upper_limit)
    ax_percent.grid(axis="y", color="0.85", linewidth=0.8)
    ax_percent.set_axisbelow(True)
    ax_percent.set_title(threshold_title, fontsize=16, fontweight="bold")
    add_panel_label(ax_percent, panel_labels[0])

    totals_thousands = exposed_population_totals.to_numpy() / 1_000
    total_positions = np.arange(n_scenarios)
    bars = ax_total.bar(
        total_positions,
        totals_thousands,
        width=0.72,
        color=SCENARIO_COLOURS[:n_scenarios],
    )
    ax_total.set_xticks(total_positions)
    ax_total.set_xticklabels(scenario_labels, fontsize=TICK_FONTSIZE)
    ax_total.set_xlabel("Scenario", fontsize=AXIS_LABEL_FONTSIZE)
    ax_total.set_ylabel(
        "Estimated population exposed (thousands)",
        fontsize=AXIS_LABEL_FONTSIZE,
    )
    ax_total.tick_params(axis="y", labelsize=TICK_FONTSIZE)
    ax_total.set_ylim(0, total_upper_limit)
    ax_total.grid(axis="y", color="0.85", linewidth=0.8)
    ax_total.set_axisbelow(True)
    ax_total.set_title(threshold_title, fontsize=16, fontweight="bold")
    label_offset = total_upper_limit * 0.012
    for bar, value in zip(bars, totals_thousands):
        ax_total.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )
    add_panel_label(ax_total, panel_labels[1])
    return ax_percent.get_legend_handles_labels()


def plot_all_thresholds(
    threshold_results,
    regional_percentages,
    national_percentages,
):
    """Create three threshold rows with percentage and population panels."""
    report_progress("Building the six-panel Taranaki results figure...")
    fig, axes = plt.subplots(
        3,
        2,
        figsize=FIGURE_SIZE,
        sharey="col",
        gridspec_kw={"width_ratios": [2.2, 1]},
    )

    national_maximum = max(national_percentages.values())
    regional_maximum = max(regional_percentages.values())
    exposure_maximum = max(
        result["data"][field].max(skipna=True)
        for result in threshold_results
        for field, _, _ in ETHNICITY_ORDER
    )
    percent_upper_limit = 1.10 * max(
        national_maximum,
        regional_maximum,
        exposure_maximum,
    )

    maximum_total_thousands = max(
        result["totals"].max() / 1_000
        for result in threshold_results
    )
    total_upper_limit = (
        1.14 * maximum_total_thousands
        if maximum_total_thousands > 0
        else 1
    )

    panel_labels = [("(a)", "(b)"), ("(c)", "(d)"), ("(e)", "(f)")]
    legend_handles = None
    legend_labels = None

    for row, (result, labels) in enumerate(zip(threshold_results, panel_labels)):
        handles, handle_labels = plot_threshold_row(
            ax_percent=axes[row, 0],
            ax_total=axes[row, 1],
            taranaki_data=result["data"],
            regional_percentages=regional_percentages,
            national_percentages=national_percentages,
            exposed_population_totals=result["totals"],
            threshold_title=result["title"],
            panel_labels=labels,
            percent_upper_limit=percent_upper_limit,
            total_upper_limit=total_upper_limit,
        )
        if legend_handles is None:
            legend_handles = handles
            legend_labels = handle_labels

    fig.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        ncol=5,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
    )
    fig.tight_layout(rect=(0, 0.07, 1, 1), h_pad=2.5, w_pad=1.5)
    report_progress("Six-panel Taranaki results figure built")
    return fig


def main():
    """Load the exposure summaries, build the figure, and save it."""
    report_progress("Starting Taranaki results bar graph")
    national_percentages = load_national_percentages(NATIONAL_PERCENT_FILE)

    threshold_results = []
    for threshold_key, threshold_title in THRESHOLDS_TO_PLOT:
        taranaki_data = load_taranaki_percentages(
            TARANAKI_SUMMARY_FILE,
            METHOD_TO_PLOT,
            threshold_key,
        )
        population_totals = load_exposed_population_totals(
            TARANAKI_POPULATION_FILE,
            METHOD_TO_PLOT,
            threshold_key,
        )
        threshold_results.append(
            {
                "key": threshold_key,
                "title": threshold_title,
                "data": taranaki_data,
                "totals": population_totals,
            }
        )

    fig = plot_all_thresholds(
        threshold_results,
        REGIONAL_PERCENTAGES,
        national_percentages,
    )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    report_progress(f"Saving 300 dpi figure to: {OUTPUT_FILE}")
    fig.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches="tight")
    report_progress("Displaying figure; close the plot window to finish")
    plt.show()
    report_progress("Taranaki results plotting complete")


if __name__ == "__main__":
    main()
