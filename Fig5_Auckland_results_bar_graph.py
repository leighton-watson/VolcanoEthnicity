
"""
Plot Auckland scenario exposure ethnicity percentages.

The bar chart groups bars by ethnicity. Within each ethnicity group, bars show
each Auckland scenario plus the Auckland regional and national population
percentages for comparison. The percentages and total exposed population use
exposed ``erp18`` as their denominator, as calculated by
``Auckland_run_analysis.py``. Because ethnicity is total-response data, the
percentages need not sum to 100.

Run this script from Spyder or from the project root.
"""

from pathlib import Path
from time import perf_counter

SCRIPT_START = perf_counter()


def report_progress(message):
    """Print a timestamped progress message immediately."""
    elapsed = perf_counter() - SCRIPT_START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


report_progress("Starting auckland_results_bar_graph.py")
report_progress("Importing Matplotlib...")
import matplotlib.pyplot as plt
report_progress("Matplotlib imported; importing NumPy...")
import numpy as np
report_progress("NumPy imported; importing pandas...")
import pandas as pd
report_progress("Python packages imported")


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent

AUCKLAND_SUMMARY_FILE = SCRIPT_DIR / "OUTPUT" / "Auckland" / "ALL_summary_pivot.csv"
AUCKLAND_POPULATION_FILE = (
    SCRIPT_DIR / "OUTPUT" / "Auckland" / "ALL_population_exposed.csv"
)
NATIONAL_PERCENT_FILE = (
    PROJECT_ROOT / "OUTPUT" / "national_ethnicity_percentages_sa2_2018.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "OUTPUT" 
OUTPUT_FILE = OUTPUT_DIR / "Fig_auckland_results.png"

METHOD_TO_PLOT = "buildings"  # use "area" or "buildings"

# Threshold rows are plotted in this order.
THRESHOLDS_TO_PLOT = [
    ("10_mm", "Ashfall ≥ 10 mm"),
    ("1_mm", "Ashfall ≥ 1 mm"),
    ("trace", "Trace ashfall (> 0 mm)"),
]

FIGURE_SIZE = (18, 19)

ETHNICITY_ORDER = [
    ("Eurother", "European or Other", "European or Other"),
    ("Maori", "Māori", "Maori"),
    ("Pacific", "Pacific", "Pacific"),
    ("Asian", "Asian", "Asian"),
    ("MELAA", "MELAA", "MELAA"),
]

NATIONAL_BAR_LABEL = "National"
REGIONAL_BAR_LABEL = "Auckland region"

# Auckland regional ethnicity percentages for comparison with the scenario
# exposure estimates and national population percentages.
REGIONAL_PERCENTAGES = {
    "Eurother": 53.2,
    "Maori": 11.5,
    "Pacific": 15.7,
    "Asian": 29.1,
    "MELAA": 2.4,
}

SCENARIO_COLOURS = [
    "#E69F00",  # orange
    "#56B4E9",  # sky blue
    "#009E73",  # bluish green
    "#F0E442",  # yellow
    "#0072B2",  # blue
    "#D55E00",  # vermillion
    "#CC79A7",  # reddish purple
    "#999999",  # grey
]


def get_auckland_summary_path():
    """Return the corrected Auckland summary file path."""
    report_progress("Looking for the Auckland summary CSV...")
    if AUCKLAND_SUMMARY_FILE.exists():
        report_progress(f"Found Auckland summary: {AUCKLAND_SUMMARY_FILE}")
        return AUCKLAND_SUMMARY_FILE
    raise FileNotFoundError(
        f"Could not find the corrected summary: {AUCKLAND_SUMMARY_FILE}\n"
        "Run MakeFigures/Auckland_run_analysis.py first."
    )


def load_auckland_exposure_percentages(summary_path, method, threshold_key):
    """Load Auckland scenario exposure shares and convert to percentages."""
    file_size_mb = summary_path.stat().st_size / (1024 * 1024)
    report_progress(
        f"Reading Auckland summary ({file_size_mb:.2f} MB). "
        "OneDrive may download it now..."
    )
    read_start = perf_counter()
    data = pd.read_csv(summary_path)
    report_progress(
        f"Auckland summary loaded in {perf_counter() - read_start:.1f}s "
        f"({len(data):,} rows)"
    )
    required_column = "ash_threshold_key"
    if required_column not in data.columns:
        raise ValueError(
            f"The summary has no '{required_column}' column. Run the updated "
            "MakeFigures/Auckland_run_analysis.py first."
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

    data = data.sort_values("scenario")

    for field, _, _ in ETHNICITY_ORDER:
        data[field] = pd.to_numeric(data[field], errors="coerce")
        if data[field].max(skipna=True) <= 1:
            data[field] = data[field] * 100

    report_progress(
        f"Prepared {len(data):,} Auckland rows for method '{method}' and "
        f"threshold '{threshold_key}'"
    )
    return data


def load_exposed_population_totals(population_path, method, threshold_key):
    """Load exposed ERP totals rather than summing ethnicity responses."""
    if not population_path.exists():
        raise FileNotFoundError(
            f"Corrected population totals are missing: {population_path}\n"
            "Run MakeFigures/Auckland_run_analysis.py first."
        )

    file_size_mb = population_path.stat().st_size / (1024 * 1024)
    report_progress(
        f"Reading exposure totals ({file_size_mb:.2f} MB). "
        "OneDrive may download them now..."
    )
    read_start = perf_counter()
    population_summary = pd.read_csv(population_path)
    report_progress(
        f"Exposure totals loaded in {perf_counter() - read_start:.1f}s "
        f"({len(population_summary):,} rows)"
    )

    required_column = "ash_threshold_key"
    if required_column not in population_summary.columns:
        raise ValueError(
            f"The population summary has no '{required_column}' column. Run "
            "the updated MakeFigures/Auckland_run_analysis.py first."
        )
    method_data = population_summary.loc[
        (population_summary["method"] == method)
        & (population_summary["ash_threshold_key"] == threshold_key)
    ].copy()
    if method_data.empty:
        available_methods = sorted(
            population_summary["method"].dropna().unique()
        )
        raise ValueError(
            f"No exposure totals found for method '{method}' and threshold "
            f"'{threshold_key}'. Available methods: {available_methods}"
        )

    method_data["exposed_population"] = pd.to_numeric(
        method_data["exposed_population"],
        errors="raise",
    )
    method_data["scenario"] = method_data["scenario"].astype(str)
    totals = method_data.set_index("scenario")["exposed_population"].sort_index()

    print(
        f"\nEstimated population exposed to ashfall ({method} method, "
        f"{threshold_key}):",
        flush=True,
    )
    for scenario, total in totals.items():
        print(f"  Scenario {scenario}: {total:,.0f} people", flush=True)
    print(flush=True)
    return totals


def load_national_percentages(national_path):
    """Load national ethnicity percentages from the SA2 national summary."""
    file_size_mb = national_path.stat().st_size / (1024 * 1024)
    report_progress(
        f"Reading national percentages ({file_size_mb:.2f} MB). "
        "OneDrive may download it now..."
    )
    read_start = perf_counter()
    national = pd.read_csv(national_path)
    report_progress(
        f"National percentages loaded in {perf_counter() - read_start:.1f}s "
        f"({len(national):,} rows)"
    )
    national_lookup = {}

    for _, display_label, national_label in ETHNICITY_ORDER:
        match = national.loc[national["ethnicity"] == national_label]
        if match.empty:
            raise ValueError(f"Could not find national ethnicity '{national_label}'")
        national_lookup[display_label] = float(match["percent_of_total_population"].iloc[0])

    report_progress("Prepared national comparison percentages")
    return national_lookup


def plot_threshold_row(
    ax_percent,
    ax_total,
    auckland_data,
    regional_percentages,
    national_percentages,
    exposed_population_totals,
    threshold_title,
    panel_labels,
    percent_upper_limit,
    total_upper_limit,
):
    """Plot one threshold's ethnicity percentages and population totals."""
    scenario_labels = auckland_data["scenario"].astype(str).tolist()
    bar_labels = scenario_labels + [REGIONAL_BAR_LABEL, NATIONAL_BAR_LABEL]
    n_bars = len(bar_labels)
    n_groups = len(ETHNICITY_ORDER)

    if len(scenario_labels) > len(SCENARIO_COLOURS):
        raise ValueError(
            f"The figure has {len(scenario_labels)} scenarios but only "
            f"{len(SCENARIO_COLOURS)} scenario colours."
        )

    group_positions = np.arange(n_groups)
    bar_width = 0.78 / n_bars
    offsets = (np.arange(n_bars) - (n_bars - 1) / 2) * bar_width
    colours = SCENARIO_COLOURS[:len(scenario_labels)]

    for scenario_index, scenario_label in enumerate(scenario_labels):
        values = [
            auckland_data.loc[
                auckland_data["scenario"].astype(str) == scenario_label,
                field,
            ].iloc[0]
            for field, _, _ in ETHNICITY_ORDER
        ]
        ax_percent.bar(
            group_positions + offsets[scenario_index],
            values,
            width=bar_width,
            color=colours[scenario_index],
            label=f"Scenario {scenario_label}",
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
        fontsize=14,
    )
    ax_percent.set_ylabel(
        "Percent",
        fontsize=14,
    )
    ax_percent.tick_params(axis="y", labelsize=14)
    ax_percent.set_ylim(0, percent_upper_limit)
    ax_percent.grid(axis="y", color="0.85", linewidth=0.8)
    ax_percent.set_axisbelow(True)
    ax_percent.set_title(threshold_title, fontsize=16, fontweight="bold")
    ax_percent.text(
        0.01,
        0.98,
        panel_labels[0],
        transform=ax_percent.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    totals = exposed_population_totals.reindex(scenario_labels)
    if totals.isna().any():
        missing = totals.index[totals.isna()].tolist()
        raise ValueError(f"Missing exposed population totals for scenarios: {missing}")

    totals_millions = totals.to_numpy() / 1_000_000
    total_positions = np.arange(len(scenario_labels))
    total_bars = ax_total.bar(
        total_positions,
        totals_millions,
        width=0.72,
        color=colours,
    )
    ax_total.set_xticks(total_positions)
    ax_total.set_xticklabels(scenario_labels, fontsize=14)
    ax_total.set_xlabel("Scenario", fontsize=14)
    ax_total.set_ylabel("Population exposed (millions)", fontsize=14)
    ax_total.tick_params(axis="y", labelsize=14)
    ax_total.set_ylim(0, total_upper_limit)
    ax_total.grid(axis="y", color="0.85", linewidth=0.8)
    ax_total.set_axisbelow(True)
    ax_total.set_title(threshold_title, fontsize=16, fontweight="bold")
    label_offset = total_upper_limit * 0.012
    for bar, value in zip(total_bars, totals_millions):
        ax_total.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + label_offset,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=14,
        )
    ax_total.text(
        0.02,
        0.98,
        panel_labels[1],
        transform=ax_total.transAxes,
        ha="left",
        va="top",
        fontsize=16,
        fontweight="bold",
    )

    return ax_percent.get_legend_handles_labels()


def plot_all_thresholds(
    threshold_results,
    regional_percentages,
    national_percentages,
):
    """Create three threshold rows with percentage and population panels."""
    report_progress("Building the six-panel threshold comparison figure...")
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

    maximum_total_millions = max(
        result["totals"].max() / 1_000_000
        for result in threshold_results
    )
    total_upper_limit = (
        1.14 * maximum_total_millions
        if maximum_total_millions > 0
        else 1
    )

    panel_labels = [("(a)", "(b)"), ("(c)", "(d)"), ("(e)", "(f)")]
    legend_handles = None
    legend_labels = None

    for row, (result, labels) in enumerate(zip(threshold_results, panel_labels)):
        handles, handle_labels = plot_threshold_row(
            ax_percent=axes[row, 0],
            ax_total=axes[row, 1],
            auckland_data=result["data"],
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
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1), h_pad=2.5, w_pad=1.5)
    report_progress("Six-panel threshold comparison figure built")
    return fig


def main():
    report_progress("Beginning data-loading workflow")
    summary_path = get_auckland_summary_path()
    national_percentages = load_national_percentages(NATIONAL_PERCENT_FILE)

    threshold_results = []
    for threshold_key, threshold_title in THRESHOLDS_TO_PLOT:
        auckland_data = load_auckland_exposure_percentages(
            summary_path,
            METHOD_TO_PLOT,
            threshold_key,
        )
        exposed_population_totals = load_exposed_population_totals(
            AUCKLAND_POPULATION_FILE,
            METHOD_TO_PLOT,
            threshold_key,
        )
        threshold_results.append(
            {
                "key": threshold_key,
                "title": threshold_title,
                "data": auckland_data,
                "totals": exposed_population_totals,
            }
        )

    fig = plot_all_thresholds(
        threshold_results,
        REGIONAL_PERCENTAGES,
        national_percentages,
    )

    report_progress(f"Creating output directory: {OUTPUT_DIR}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report_progress(f"Saving 300 dpi figure to: {OUTPUT_FILE}")
    save_start = perf_counter()
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    report_progress(f"Figure saved in {perf_counter() - save_start:.1f}s")
    report_progress("Displaying figure; close the plot window to finish the script")
    plt.show()
    report_progress("Plot window closed; script finished")


if __name__ == "__main__":
    main()
