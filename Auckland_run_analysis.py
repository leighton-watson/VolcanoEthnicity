"""AVF ashfall exposure workflow converted from AVF_multiscenario.ipynb.

Run this file from Spyder, JupyterLab, or a Miniforge Prompt. Paths are
resolved relative to this file, so the script does not depend on your current
working directory.

The workflow calculates exposure for 10 mm, 1 mm, and trace (> 0 mm) ashfall.
Total exposed population is calculated from the 2018 estimated resident
population (``erp18``). Ethnicity uses total-response counts, meaning that one
person may be represented in more than one ethnic group. The ethnicity
percentages can therefore correctly sum to more than 100 percent.
"""

# %%
from pathlib import Path
import gc
import logging

import geopandas as gpd
import numpy as np
import pandas as pd
from shapely import wkt


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("avf_workflow")


# %%
# ---------------------------
# Paths and settings
# ---------------------------
SCRIPT_DIR = Path(__file__).resolve().parent if "__file__" in globals() else Path.cwd()
PROJECT_ROOT = SCRIPT_DIR.parent 

POPULATION_CSV = (
    PROJECT_ROOT
    / "DATA"
    / "PopulationData"
    / "estimated-resident-population-at-30-june-2018-by-statistical.csv"
)
BUILDING_FOOTPRINTS = (
    PROJECT_ROOT
    / "DATA"
    / "AucklandBuildings"
    / "nz-building-outlines.shp"
)

SCENARIO_DIR = (
    PROJECT_ROOT
    / "DATA"
    / "Auckland"
)

                
OUTPUT_FOLDER = SCRIPT_DIR / "OUTPUT" / "Auckland"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SCENARIOS = list("ABCDEFGH")
TARGET_CRS = "EPSG:2193"

SA2_KEY = "SA22020_V1_00"
TOTAL_POPULATION_FIELD = "erp18"
ETHNICITY_COLS = ["Eurother", "Maori", "Pacific", "Asian", "MELAA"]

# Thresholds are processed and displayed in this order. Trace ash uses every
# polygon with a positive thickness, matching ``auckland_scenario.py``.
ASH_EXPOSURE_THRESHOLDS = [
    {"key": "10_mm", "label": "10 mm", "minimum_mm": 10.0},
    {"key": "1_mm", "label": "1 mm", "minimum_mm": 1.0},
    {"key": "trace", "label": "Trace ashfall", "minimum_mm": 0.0},
]
THICKNESS_COLUMN = "thickness"

ETHNICITY_DISPLAY_NAMES = {
    "Eurother": "European or Other",
    "Maori": "Māori",
    "Pacific": "Pacific",
    "Asian": "Asian",
    "MELAA": "MELAA",
}

# Set this to False if you only want area-weighted outputs.
USE_BUILDINGS = True


# %%
def safe_wkt(value):
    """Parse one WKT geometry value, returning None for invalid rows."""
    try:
        if pd.isnull(value):
            return None
        return wkt.loads(value)
    except Exception:
        return None


def load_population_sa2(population_csv):
    """Load the SA2 population CSV and convert its WKT column to a GeoDataFrame."""
    if not population_csv.exists():
        raise FileNotFoundError(f"Population CSV not found: {population_csv}")

    pop_df = pd.read_csv(population_csv)

    wkt_col = None
    for col in pop_df.columns:
        if "WKT" in col.upper():
            wkt_col = col
            break

    if wkt_col is None:
        raise ValueError("No WKT column found in the population CSV.")

    pop_df["geometry"] = pop_df[wkt_col].apply(safe_wkt)
    pop_gdf = gpd.GeoDataFrame(pop_df, geometry="geometry")
    pop_gdf = pop_gdf[pop_gdf.geometry.notnull()].copy()
    pop_gdf = pop_gdf[~pop_gdf.geometry.is_empty].copy()
    count_fields = [TOTAL_POPULATION_FIELD, *ETHNICITY_COLS]
    for field in count_fields:
        pop_gdf[field] = pd.to_numeric(pop_gdf[field], errors="coerce")
        pop_gdf.loc[pop_gdf[field] < 0, field] = np.nan

    pop_gdf = pop_gdf[
        [SA2_KEY, TOTAL_POPULATION_FIELD, *ETHNICITY_COLS, "geometry"]
    ].copy()

    first_geom = pop_gdf.geometry.iloc[0]
    minx, miny, _maxx, _maxy = first_geom.bounds

    # NZTM coordinates are outside longitude/latitude ranges; otherwise assume
    # the WKT is lon/lat and reproject it below.
    inferred_crs = "EPSG:2193" if abs(minx) > 180 or abs(miny) > 90 else "EPSG:4326"
    pop_gdf = pop_gdf.set_crs(inferred_crs, allow_override=True)

    if pop_gdf.crs != TARGET_CRS:
        pop_gdf = pop_gdf.to_crs(TARGET_CRS)

    bounds = pop_gdf.total_bounds
    if not np.isfinite(bounds).all():
        raise ValueError("Population geometry is invalid; check WKT parsing and CRS.")

    print("Population SA2 loaded:", len(pop_gdf))
    print("CRS:", pop_gdf.crs)
    print("Bounds:", bounds)

    return pop_gdf


def load_buildings(building_footprints, target_crs):
    """Load building footprints if available."""
    if not building_footprints.exists():
        print("Building footprints not found; building-weighted outputs will be skipped.")
        print("Expected relative path:", building_footprints)
        return None

    buildings_gdf = gpd.read_file(building_footprints)

    if buildings_gdf.crs != target_crs:
        buildings_gdf = buildings_gdf.to_crs(target_crs)

    print("Buildings loaded:", len(buildings_gdf))
    return buildings_gdf


def load_scenario_vector(SCENARIO_DIR, scenario, target_crs):
    """Load all valid ashfall polygons for one Auckland scenario."""
    file_path = SCENARIO_DIR / f"Scenario_{scenario}_Tephra.gpkg"

    if not file_path.exists():
        print(f"Missing file: {file_path}")
        return None

    gdf = gpd.read_file(file_path)

    if THICKNESS_COLUMN not in gdf.columns:
        raise ValueError(
            f"{file_path.name} does not contain a '{THICKNESS_COLUMN}' column."
        )

    gdf[THICKNESS_COLUMN] = pd.to_numeric(gdf[THICKNESS_COLUMN], errors="coerce")
    gdf = gdf.loc[gdf[THICKNESS_COLUMN].notna()].copy()

    if gdf.crs != target_crs:
        gdf = gdf.to_crs(target_crs)

    return gdf


def filter_scenario_for_threshold(scenario_gdf, minimum_mm):
    """Select polygons meeting a thickness threshold; trace means > 0 mm."""
    if minimum_mm == 0:
        return scenario_gdf.loc[
            scenario_gdf[THICKNESS_COLUMN] > 0
        ].copy()
    return scenario_gdf.loc[
        scenario_gdf[THICKNESS_COLUMN] >= minimum_mm
    ].copy()


# %%
def compute_sa2_hazard(sa2_gdf, hazard_gdf):
    """Compute exposed hazard area and proportional area for each SA2."""
    hazard_union = hazard_gdf.unary_union

    sa2 = sa2_gdf.copy()
    sa2["hazard_area_m2"] = 0.0
    sa2["hazard_pct_area"] = 0.0

    for idx, row in sa2.iterrows():
        if row.geometry is None:
            continue

        inter = row.geometry.intersection(hazard_union)

        if not inter.is_empty:
            sa2.at[idx, "hazard_area_m2"] = inter.area
            sa2.at[idx, "hazard_pct_area"] = inter.area / row.geometry.area

    return sa2


def compute_building_exposure(sa2_gdf, buildings_gdf, hazard_gdf):
    """Compute total and hazard-intersecting building counts within exposed SA2s."""
    sindex = buildings_gdf.sindex
    hazard_union = hazard_gdf.unary_union
    results = []

    sa2_exposed = sa2_gdf[sa2_gdf["hazard_area_m2"] > 0].copy()

    for _, row in sa2_exposed.iterrows():
        sa2_poly = row.geometry

        idx = list(sindex.intersection(sa2_poly.bounds))
        cand = buildings_gdf.iloc[idx]
        cand = cand[cand.intersects(sa2_poly)]

        total = len(cand)

        if total == 0:
            results.append([row[SA2_KEY], 0, 0, 0.0])
            continue

        hazard_in_sa2 = sa2_poly.intersection(hazard_union)
        hazard_buildings = cand.intersects(hazard_in_sa2).sum() if not hazard_in_sa2.is_empty else 0

        results.append([row[SA2_KEY], total, hazard_buildings, hazard_buildings / total])

    return pd.DataFrame(
        results,
        columns=[
            SA2_KEY,
            "total_buildings",
            "hazard_buildings",
            "hazard_buildings_pct",
        ],
    )


def compute_population_exposure(sa2_df, pct_col):
    """Multiply resident and ethnicity counts by an exposure ratio column."""
    df = sa2_df.copy()
    df = df[df[pct_col] > 0]

    count_fields = [TOTAL_POPULATION_FIELD, *ETHNICITY_COLS]
    for col in count_fields:
        df[f"{col}_exposed"] = df[col] * df[pct_col]

    cols = [SA2_KEY, pct_col] + [f"{col}_exposed" for col in count_fields]
    return df[cols]


# %%
def run_scenarios(pop_gdf, buildings_gdf=None):
    """Run every scenario at all configured ashfall thresholds."""
    summary_area = []
    summary_building = []

    for scenario in SCENARIOS:
        print("\nRunning scenario:", scenario)

        scenario_gdf = load_scenario_vector(SCENARIO_DIR, scenario, pop_gdf.crs)
        if scenario_gdf is None:
            print("No data for", scenario)
            continue

        for threshold in ASH_EXPOSURE_THRESHOLDS:
            threshold_key = threshold["key"]
            threshold_label = threshold["label"]
            minimum_mm = threshold["minimum_mm"]
            comparison = ">" if minimum_mm == 0 else ">="
            print(
                f"  Threshold {threshold_label}: thickness {comparison} "
                f"{minimum_mm:g} mm"
            )

            hazard = filter_scenario_for_threshold(scenario_gdf, minimum_mm)
            sa2_out = compute_sa2_hazard(pop_gdf, hazard)

            area_eth = compute_population_exposure(sa2_out, "hazard_pct_area")
            area_eth["scenario"] = scenario
            area_eth["method"] = "area"
            area_eth["ash_threshold_key"] = threshold_key
            area_eth["ash_threshold_label"] = threshold_label
            area_eth["ash_threshold_mm"] = minimum_mm
            summary_area.append(area_eth)
            area_eth.to_csv(
                OUTPUT_FOLDER
                / f"{scenario}_{threshold_key}_ethnicity_area.csv",
                index=False,
            )

            if buildings_gdf is not None:
                building_exp = compute_building_exposure(
                    sa2_out,
                    buildings_gdf,
                    hazard,
                )
                sa2_building_out = sa2_out.merge(
                    building_exp,
                    on=SA2_KEY,
                    how="left",
                ).fillna(0)

                build_eth = compute_population_exposure(
                    sa2_building_out,
                    "hazard_buildings_pct",
                )
                build_eth["scenario"] = scenario
                build_eth["method"] = "buildings"
                build_eth["ash_threshold_key"] = threshold_key
                build_eth["ash_threshold_label"] = threshold_label
                build_eth["ash_threshold_mm"] = minimum_mm
                summary_building.append(build_eth)
                build_eth.to_csv(
                    OUTPUT_FOLDER
                    / f"{scenario}_{threshold_key}_ethnicity_building.csv",
                    index=False,
                )

            del hazard, sa2_out
        del scenario_gdf
        gc.collect()

    print("Done all scenarios.")
    return summary_area, summary_building


def save_threshold_ethnicity_tables(global_summary):
    """Save scenario-row ethnicity tables ordered 10 mm, 1 mm, then trace."""
    threshold_keys = [threshold["key"] for threshold in ASH_EXPOSURE_THRESHOLDS]
    threshold_labels = {
        threshold["key"]: threshold["label"]
        for threshold in ASH_EXPOSURE_THRESHOLDS
    }
    ordered_columns = pd.MultiIndex.from_product(
        [threshold_keys, ETHNICITY_COLS],
        names=["ash_threshold_key", "ethnicity"],
    )

    tables = {}
    for method in ["area", "buildings"]:
        method_summary = global_summary.loc[
            global_summary["method"] == method
        ].copy()
        if method_summary.empty:
            continue

        table = method_summary.pivot_table(
            index="scenario",
            columns=["ash_threshold_key", "ethnicity"],
            values="percent_of_exposed_population",
        )
        table = table.reindex(index=SCENARIOS, columns=ordered_columns)
        table.columns = [
            (
                f"{threshold_labels[threshold_key]} | "
                f"{ETHNICITY_DISPLAY_NAMES[ethnicity]} (%)"
            )
            for threshold_key, ethnicity in table.columns
        ]
        table.index.name = "scenario"

        output_path = (
            OUTPUT_FOLDER
            / f"Auckland_ethnicity_percentages_{method}_by_threshold.csv"
        )
        table.to_csv(output_path, float_format="%.2f")
        print(f"Saved threshold table: {output_path}")
        print(table.round(2))
        tables[method] = table

    return tables


def save_global_summaries(summary_area, summary_building):
    """Save ethnicity percentages and true exposed-population totals."""
    summaries = summary_area + summary_building
    if not summaries:
        raise ValueError("No scenario summaries were created.")

    all_summary = pd.concat(summaries, ignore_index=True)

    ethnicity_exposure_cols = [f"{col}_exposed" for col in ETHNICITY_COLS]
    population_exposure_col = f"{TOTAL_POPULATION_FIELD}_exposed"

    population_summary = (
        all_summary.groupby(
            [
                "scenario",
                "method",
                "ash_threshold_key",
                "ash_threshold_label",
                "ash_threshold_mm",
            ],
            as_index=False,
        )[population_exposure_col]
        .sum()
        .rename(columns={population_exposure_col: "exposed_population"})
    )
    population_summary.to_csv(
        OUTPUT_FOLDER / "ALL_population_exposed.csv",
        index=False,
    )

    all_long = all_summary.melt(
        id_vars=[
            SA2_KEY,
            "scenario",
            "method",
            "ash_threshold_key",
            "ash_threshold_label",
            "ash_threshold_mm",
        ],
        value_vars=ethnicity_exposure_cols,
        var_name="ethnicity",
        value_name="exposed",
    )
    all_long["ethnicity"] = all_long["ethnicity"].str.replace("_exposed", "", regex=False)

    global_summary = (
        all_long.groupby(
            [
                "scenario",
                "method",
                "ash_threshold_key",
                "ash_threshold_label",
                "ash_threshold_mm",
                "ethnicity",
            ]
        )["exposed"]
        .sum()
        .reset_index()
    )

    global_summary = global_summary.merge(
        population_summary,
        on=[
            "scenario",
            "method",
            "ash_threshold_key",
            "ash_threshold_label",
            "ash_threshold_mm",
        ],
        how="left",
    )
    global_summary["pct"] = (
        global_summary["exposed"] / global_summary["exposed_population"]
    )
    global_summary["percent_of_exposed_population"] = 100 * global_summary["pct"]

    global_summary.to_csv(OUTPUT_FOLDER / "ALL_summary_long.csv", index=False)

    pivot = global_summary.pivot_table(
        index=[
            "scenario",
            "method",
            "ash_threshold_key",
            "ash_threshold_label",
            "ash_threshold_mm",
        ],
        columns="ethnicity",
        values="pct",
    ).reset_index()

    pivot.to_csv(OUTPUT_FOLDER / "ALL_summary_pivot.csv", index=False)
    threshold_tables = save_threshold_ethnicity_tables(global_summary)
    print("\nEstimated population exposed to ashfall:")
    for row in population_summary.itertuples(index=False):
        print(
            f"  Scenario {row.scenario} ({row.method}, "
            f"{row.ash_threshold_label}): "
            f"{row.exposed_population:,.0f} people"
        )
    print()
    print(pivot)
    return global_summary, pivot, population_summary, threshold_tables


# %%
RUN_DOUBLE_CHECKS = False

pop_gdf = load_population_sa2(POPULATION_CSV)
buildings = load_buildings(BUILDING_FOOTPRINTS, pop_gdf.crs) if USE_BUILDINGS else None
summary_area, summary_building = run_scenarios(pop_gdf, buildings)
global_summary, pivot, population_summary, threshold_tables = save_global_summaries(
    summary_area,
    summary_building,
)


# %%
if RUN_DOUBLE_CHECKS:
    print((pop_gdf["Pacific"] > pop_gdf["Maori"]).sum())
    print(pop_gdf[pop_gdf["Pacific"] > pop_gdf["Maori"]][[SA2_KEY, "Pacific", "Maori"]])

    scenario_a = load_scenario_vector(SCENARIO_DIR, "A", pop_gdf.crs)
    hazard = filter_scenario_for_threshold(scenario_a, 1.0)
    sa2_out = compute_sa2_hazard(pop_gdf, hazard)

    print(
        sa2_out.sort_values("hazard_pct_area", ascending=False)[
            [SA2_KEY, "hazard_pct_area"] + ETHNICITY_COLS
        ].head(10)
    )

    total_exposed = len(sa2_out[sa2_out["hazard_pct_area"] > 0])
    pacific_gt_maori = (
        sa2_out.loc[sa2_out["hazard_pct_area"] > 0, "Pacific"]
        > sa2_out.loc[sa2_out["hazard_pct_area"] > 0, "Maori"]
    ).sum()

    print("Exposed SA2 total:", total_exposed)
    print("Pacific > Maori count:", pacific_gt_maori)
    print("Percentage:", pacific_gt_maori / total_exposed)
