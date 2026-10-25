"""Estimate Taranaki population and ethnicity exposure to cumulative ashfall.

This is the Taranaki counterpart to ``run_Auckland_analysis.py``. It uses the
nine cumulative rasters created by ``prepare_taranaki_scenario_data.py`` so the
analysis and the Taranaki scenario maps use exactly the same summed ashfall.
Exposure is calculated at 1, 10, and 100 mm thresholds.

Two estimates are produced:

1. ``area`` assumes population is distributed evenly through each SA2.
2. ``buildings`` assumes population is distributed in proportion to building
   footprints in each SA2.

Ethnicity counts use total-response ethnicity data, so one person may be
included in more than one ethnic group. Consequently, ethnicity percentages
can sum to more than 100 percent. Total exposed population is calculated
separately from the 2018 estimated resident population (``erp18``).

Run ``prepare_taranaki_scenario_data.py`` first. This script performs the slow
analysis and writes CSV files that ``taranaki_results_bar_graph.py`` can plot.
"""

# %%
from json import load
from pathlib import Path
from time import perf_counter
import gc

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely.geometry import shape
from shapely.ops import unary_union


# %%
# -----------------------------------------------------------------------------
# Paths and analysis settings
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
POPULATION_CSV = (
    PROJECT_ROOT
    / "DATA"
    / "PopulationData"
    / "estimated-resident-population-at-30-june-2018-by-statistical.csv"
)

# Set USE_BUILDINGS to False if only the faster area-weighted estimate is required.
BUILDING_FOOTPRINTS = (
    PROJECT_ROOT 
    / "DATA" 
    / "TaranakiBuildings" 
    / "nz-building-outlines.shp"
)
USE_BUILDINGS = True

CACHE_FOLDER = PROJECT_ROOT / "MakeFigures" / "OUTPUT" / "Taranaki" / "cache"
MANIFEST_FILE = CACHE_FOLDER / "taranaki_figure_manifest.json"
OUTPUT_FOLDER = (
    PROJECT_ROOT / "MakeFigures" / "OUTPUT" / "Taranaki" / "analysis"
)

SCENARIOS = ["S1", "S2", "S3", "M1", "M2", "M3", "L1", "L2", "L3"]
TARGET_CRS = "EPSG:2193"

# Thresholds are analysed and displayed in this order.
ASH_EXPOSURE_THRESHOLDS = [
    {"key": "1_mm", "label": "1 mm", "minimum_mm": 1.0},
    {"key": "10_mm", "label": "10 mm", "minimum_mm": 10.0},
    {"key": "100_mm", "label": "100 mm", "minimum_mm": 100.0},
]

SA2_KEY = "SA22020_V1_00"
SA2_NAME_FIELD = "SA22020_V1_00_NAME"
TOTAL_POPULATION_FIELD = "erp18"
ETHNICITY_COLS = ["Eurother", "Maori", "Pacific", "Asian", "MELAA"]
ETHNICITY_DISPLAY_NAMES = {
    "Eurother": "European or Other",
    "Maori": "Māori",
    "Pacific": "Pacific",
    "Asian": "Asian",
    "MELAA": "MELAA",
}

SCRIPT_START = perf_counter()


def report_progress(message):
    """Print an immediately flushed progress message with elapsed time."""
    elapsed = perf_counter() - SCRIPT_START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def load_manifest():
    """Load the cache manifest written by the Taranaki preparation script."""
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Prepared Taranaki data not found: {MANIFEST_FILE}\n"
            "Run MakeFigures/prepare_taranaki_scenario_data.py first."
        )

    with MANIFEST_FILE.open(encoding="utf-8") as stream:
        manifest = load(stream)

    missing_scenarios = [
        scenario
        for scenario in SCENARIOS
        if scenario not in manifest.get("scenarios", {})
    ]
    if missing_scenarios:
        raise ValueError(
            "The Taranaki cache manifest is missing scenarios: "
            + ", ".join(missing_scenarios)
        )
    return manifest


def load_population_sa2(manifest):
    """Load cached Taranaki SA2 geometries and attach population counts."""
    sa2_path = CACHE_FOLDER / manifest["sa2_cache"]
    if not sa2_path.exists():
        raise FileNotFoundError(
            f"Prepared Taranaki SA2 data not found: {sa2_path}\n"
            "Run MakeFigures/prepare_taranaki_scenario_data.py first."
        )
    if not POPULATION_CSV.exists():
        raise FileNotFoundError(f"Population CSV not found: {POPULATION_CSV}")

    report_progress("Loading cached Taranaki SA2 geometries...")
    sa2_geometry = gpd.read_file(sa2_path, layer="sa2")[[SA2_KEY, "geometry"]]
    if sa2_geometry.crs is None:
        raise ValueError(f"Prepared SA2 data has no CRS: {sa2_path}")
    if sa2_geometry.crs.to_string() != TARGET_CRS:
        sa2_geometry = sa2_geometry.to_crs(TARGET_CRS)

    use_columns = [
        SA2_KEY,
        SA2_NAME_FIELD,
        TOTAL_POPULATION_FIELD,
        *ETHNICITY_COLS,
    ]
    report_progress("Loading 2018 population and ethnicity counts...")
    attributes = pd.read_csv(POPULATION_CSV, usecols=use_columns)
    for field in [TOTAL_POPULATION_FIELD, *ETHNICITY_COLS]:
        attributes[field] = pd.to_numeric(attributes[field], errors="coerce")
        attributes.loc[attributes[field] < 0, field] = np.nan

    population = sa2_geometry.merge(attributes, on=SA2_KEY, how="left")
    missing_population = population[TOTAL_POPULATION_FIELD].isna().sum()
    if missing_population:
        report_progress(
            f"Warning: {missing_population} cached SA2 regions have no population value."
        )

    report_progress(f"Prepared {len(population):,} Taranaki SA2 regions")
    return population


def load_buildings(building_path, population):
    """Load, reproject, and spatially crop the Taranaki building footprints."""
    if not building_path.exists():
        report_progress(
            "Building footprints were not found; building-weighted outputs "
            "will be skipped."
        )
        print(f"Expected file: {building_path}", flush=True)
        return None

    file_size_mb = building_path.stat().st_size / (1024 * 1024)
    report_progress(
        f"Loading the {file_size_mb:.0f} MB Taranaki building shapefile..."
    )
    # This regional shapefile is stored in geographic coordinates (EPSG:4167),
    # while the population bounds use NZTM (EPSG:2193). Loading it with an NZTM
    # bbox would therefore return zero records, so reproject before cropping.
    buildings = gpd.read_file(building_path)
    if buildings.crs is None:
        raise ValueError(f"Building footprints have no CRS: {building_path}")
    if buildings.crs.to_string() != TARGET_CRS:
        report_progress(
            f"Reprojecting buildings from {buildings.crs} to {TARGET_CRS}..."
        )
        buildings = buildings.to_crs(TARGET_CRS)

    bounds = tuple(population.total_bounds)
    xmin, ymin, xmax, ymax = bounds
    buildings = buildings.cx[xmin:xmax, ymin:ymax].copy()
    buildings = buildings.loc[
        buildings.geometry.notna() & ~buildings.geometry.is_empty
    ].copy()
    if buildings.empty:
        raise ValueError(
            "No building footprints overlap the cached Taranaki SA2 extent. "
            f"Check the building file and its CRS: {building_path}"
        )
    report_progress(f"Loaded {len(buildings):,} Taranaki building footprints")
    return buildings


def get_scenario_raster_path(manifest, scenario):
    """Return one prepared cumulative-ashfall raster path."""
    raster_path = CACHE_FOLDER / manifest["scenarios"][scenario]["raster"]
    if not raster_path.exists():
        raise FileNotFoundError(
            f"Prepared raster not found: {raster_path}\n"
            "Run MakeFigures/prepare_taranaki_scenario_data.py first."
        )
    return raster_path


def raster_to_ash_polygon(raster_path, threshold_mm):
    """Convert raster cells meeting the ashfall threshold to one polygon."""
    with rasterio.open(raster_path) as source:
        if source.crs is None:
            raise ValueError(f"Ashfall raster has no CRS: {raster_path}")
        if source.crs.to_string() != TARGET_CRS:
            raise ValueError(
                f"{raster_path.name} uses {source.crs}; expected {TARGET_CRS}."
            )
        ashfall = source.read(1, masked=False).astype(np.float32)
        transform = source.transform

    valid = np.isfinite(ashfall)
    if threshold_mm == 0:
        exposed_mask = valid & (ashfall > 0)
    else:
        exposed_mask = valid & (ashfall >= threshold_mm)

    if not exposed_mask.any():
        return None

    polygons = [
        shape(geometry)
        for geometry, value in shapes(
            exposed_mask.astype(np.uint8),
            mask=exposed_mask,
            transform=transform,
        )
        if value == 1
    ]
    return unary_union(polygons) if polygons else None


def compute_area_exposure(population, ash_polygon):
    """Calculate the fraction of each SA2 covered by cumulative ashfall."""
    result = population.copy()
    result["ash_area_m2"] = 0.0
    result["ash_pct_area"] = 0.0
    if ash_polygon is None or ash_polygon.is_empty:
        return result

    candidate_index = list(result.sindex.intersection(ash_polygon.bounds))
    for index in result.index[candidate_index]:
        sa2_geometry = result.at[index, "geometry"]
        intersection = sa2_geometry.intersection(ash_polygon)
        if intersection.is_empty or sa2_geometry.area <= 0:
            continue
        result.at[index, "ash_area_m2"] = intersection.area
        result.at[index, "ash_pct_area"] = min(
            intersection.area / sa2_geometry.area,
            1.0,
        )
    return result


def compute_building_exposure(sa2_exposure, buildings, ash_polygon):
    """Calculate each SA2's fraction of buildings inside the ash footprint."""
    columns = [
        SA2_KEY,
        "total_buildings",
        "ash_buildings",
        "ash_buildings_pct",
    ]
    if buildings is None or ash_polygon is None or ash_polygon.is_empty:
        return pd.DataFrame(columns=columns)

    building_index = buildings.sindex
    rows = []
    exposed_sa2 = sa2_exposure.loc[sa2_exposure["ash_pct_area"] > 0]

    for _, sa2_row in exposed_sa2.iterrows():
        sa2_geometry = sa2_row.geometry
        possible_index = list(building_index.intersection(sa2_geometry.bounds))
        candidates = buildings.iloc[possible_index]
        candidates = candidates.loc[candidates.intersects(sa2_geometry)]
        total_buildings = len(candidates)

        if total_buildings == 0:
            ash_buildings = 0
            ash_buildings_pct = 0.0
        else:
            ash_within_sa2 = sa2_geometry.intersection(ash_polygon)
            ash_buildings = int(candidates.intersects(ash_within_sa2).sum())
            ash_buildings_pct = ash_buildings / total_buildings

        rows.append(
            {
                SA2_KEY: sa2_row[SA2_KEY],
                "total_buildings": total_buildings,
                "ash_buildings": ash_buildings,
                "ash_buildings_pct": ash_buildings_pct,
            }
        )

    return pd.DataFrame(rows, columns=columns)


def calculate_exposed_counts(
    sa2_exposure,
    exposure_field,
    scenario,
    method,
    threshold,
):
    """Apply an SA2 exposure fraction to resident and ethnicity counts."""
    exposed = sa2_exposure.loc[sa2_exposure[exposure_field] > 0].copy()
    count_fields = [TOTAL_POPULATION_FIELD, *ETHNICITY_COLS]
    for field in count_fields:
        exposed[f"{field}_exposed"] = exposed[field] * exposed[exposure_field]

    output_columns = [
        SA2_KEY,
        SA2_NAME_FIELD,
        exposure_field,
        "ash_area_m2",
        *[f"{field}_exposed" for field in count_fields],
    ]
    per_sa2 = exposed[output_columns].copy()
    per_sa2.insert(0, "scenario", scenario)
    per_sa2.insert(1, "method", method)
    per_sa2.insert(2, "ash_threshold_key", threshold["key"])
    per_sa2.insert(3, "ash_threshold_label", threshold["label"])
    per_sa2.insert(4, "ash_threshold_mm", threshold["minimum_mm"])

    total_population = per_sa2[f"{TOTAL_POPULATION_FIELD}_exposed"].sum()
    summary_rows = []
    for ethnicity in ETHNICITY_COLS:
        ethnicity_exposed = per_sa2[f"{ethnicity}_exposed"].sum()
        fraction = (
            ethnicity_exposed / total_population if total_population > 0 else 0.0
        )
        summary_rows.append(
            {
                "scenario": scenario,
                "method": method,
                "ash_threshold_key": threshold["key"],
                "ash_threshold_label": threshold["label"],
                "ash_threshold_mm": threshold["minimum_mm"],
                "ethnicity": ethnicity,
                "exposed": ethnicity_exposed,
                "total_exposed_population": total_population,
                "pct": fraction,
                "percent_of_exposed_population": 100 * fraction,
            }
        )

    population_summary = {
        "scenario": scenario,
        "method": method,
        "ash_threshold_key": threshold["key"],
        "ash_threshold_label": threshold["label"],
        "ash_threshold_mm": threshold["minimum_mm"],
        "exposed_population": total_population,
    }
    return per_sa2, pd.DataFrame(summary_rows), population_summary


def print_scenario_summary(
    scenario,
    method,
    threshold,
    summary,
    population_summary,
):
    """Print total and ethnicity exposure estimates for one scenario/method."""
    total_population = population_summary["exposed_population"]
    print(
        f"\nScenario {scenario} ({method}, ash >= "
        f"{threshold['minimum_mm']:g} mm):",
        flush=True,
    )
    print(f"  Total population exposed: {total_population:,.0f}", flush=True)
    for row in summary.itertuples(index=False):
        print(
            f"  {row.ethnicity}: {row.exposed:,.0f} "
            f"({row.percent_of_exposed_population:.1f}%)",
            flush=True,
        )


def run_scenarios(population, buildings, manifest):
    """Run all nine scenarios at each configured ashfall threshold."""
    long_summaries = []
    population_summaries = []

    for number, scenario in enumerate(SCENARIOS, start=1):
        report_progress(
            f"Scenario {scenario} ({number}/{len(SCENARIOS)}): loading ashfall"
        )
        raster_path = get_scenario_raster_path(manifest, scenario)

        for threshold in ASH_EXPOSURE_THRESHOLDS:
            threshold_key = threshold["key"]
            threshold_label = threshold["label"]
            minimum_mm = threshold["minimum_mm"]
            report_progress(
                f"Scenario {scenario}, {threshold_label}: creating ash footprint"
            )
            ash_polygon = raster_to_ash_polygon(raster_path, minimum_mm)
            if ash_polygon is None:
                report_progress(
                    f"Scenario {scenario} has no ashfall >= {minimum_mm:g} mm"
                )

            report_progress(
                f"Scenario {scenario}, {threshold_label}: "
                "calculating area-weighted exposure"
            )
            area_exposure = compute_area_exposure(population, ash_polygon)
            area_per_sa2, area_summary, area_population = calculate_exposed_counts(
                area_exposure,
                "ash_pct_area",
                scenario,
                "area",
                threshold,
            )
            area_per_sa2.to_csv(
                OUTPUT_FOLDER
                / f"{scenario}_{threshold_key}_exposure_area.csv",
                index=False,
            )
            long_summaries.append(area_summary)
            population_summaries.append(area_population)
            print_scenario_summary(
                scenario,
                "area",
                threshold,
                area_summary,
                area_population,
            )

            if buildings is not None:
                report_progress(
                    f"Scenario {scenario}, {threshold_label}: "
                    "calculating building-weighted exposure"
                )
                building_fractions = compute_building_exposure(
                    area_exposure,
                    buildings,
                    ash_polygon,
                )
                building_exposure = area_exposure.merge(
                    building_fractions,
                    on=SA2_KEY,
                    how="left",
                )
                building_fields = [
                    "total_buildings",
                    "ash_buildings",
                    "ash_buildings_pct",
                ]
                building_exposure[building_fields] = (
                    building_exposure[building_fields].fillna(0)
                )
                building_per_sa2, building_summary, building_population = (
                    calculate_exposed_counts(
                        building_exposure,
                        "ash_buildings_pct",
                        scenario,
                        "buildings",
                        threshold,
                    )
                )
                building_per_sa2.to_csv(
                    OUTPUT_FOLDER
                    / f"{scenario}_{threshold_key}_exposure_buildings.csv",
                    index=False,
                )
                long_summaries.append(building_summary)
                population_summaries.append(building_population)
                print_scenario_summary(
                    scenario,
                    "buildings",
                    threshold,
                    building_summary,
                    building_population,
                )

            del ash_polygon, area_exposure
            if buildings is not None:
                del building_exposure

        gc.collect()

    return long_summaries, population_summaries


def save_threshold_ethnicity_tables(long_summary):
    """Save scenario-row ethnicity tables ordered 1, 10, then 100 mm."""
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
        method_summary = long_summary.loc[
            long_summary["method"] == method
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
            / f"Taranaki_ethnicity_percentages_{method}_by_threshold.csv"
        )
        table.to_csv(output_path, float_format="%.2f")
        report_progress(f"Saved threshold table: {output_path}")
        print(table.round(2))
        tables[method] = table

    return tables


def save_combined_summaries(long_summaries, population_summaries):
    """Write combined long, pivot, and total-population summary CSV files."""
    if not long_summaries:
        raise ValueError("No Taranaki scenario summaries were created.")

    long_summary = pd.concat(long_summaries, ignore_index=True)
    scenario_order = pd.CategoricalDtype(SCENARIOS, ordered=True)
    long_summary["scenario"] = long_summary["scenario"].astype(scenario_order)
    threshold_order = {
        threshold["key"]: index
        for index, threshold in enumerate(ASH_EXPOSURE_THRESHOLDS)
    }
    long_summary["_threshold_order"] = long_summary["ash_threshold_key"].map(
        threshold_order
    )
    long_summary = long_summary.sort_values(
        ["method", "_threshold_order", "scenario", "ethnicity"]
    ).drop(columns="_threshold_order")
    long_summary["scenario"] = long_summary["scenario"].astype(str)
    long_summary.to_csv(OUTPUT_FOLDER / "ALL_summary_long.csv", index=False)

    pivot = long_summary.pivot_table(
        index=[
            "scenario",
            "method",
            "ash_threshold_key",
            "ash_threshold_label",
            "ash_threshold_mm",
        ],
        columns="ethnicity",
        values="pct",
        observed=False,
    ).reset_index()
    pivot["scenario"] = pivot["scenario"].astype(scenario_order)
    pivot["_threshold_order"] = pivot["ash_threshold_key"].map(threshold_order)
    pivot = pivot.sort_values(
        ["method", "_threshold_order", "scenario"]
    ).drop(columns="_threshold_order")
    pivot["scenario"] = pivot["scenario"].astype(str)
    pivot.to_csv(OUTPUT_FOLDER / "ALL_summary_pivot.csv", index=False)

    population_summary = pd.DataFrame(population_summaries)
    population_summary["scenario"] = population_summary["scenario"].astype(
        scenario_order
    )
    population_summary["_threshold_order"] = population_summary[
        "ash_threshold_key"
    ].map(threshold_order)
    population_summary = population_summary.sort_values(
        ["method", "_threshold_order", "scenario"]
    ).drop(columns="_threshold_order")
    population_summary["scenario"] = population_summary["scenario"].astype(str)
    population_summary.to_csv(
        OUTPUT_FOLDER / "ALL_population_exposed.csv",
        index=False,
    )

    threshold_tables = save_threshold_ethnicity_tables(long_summary)
    report_progress(f"Saved combined summaries in {OUTPUT_FOLDER}")
    return long_summary, pivot, population_summary, threshold_tables


def main():
    """Run the complete Taranaki exposure analysis."""
    report_progress("Starting Taranaki cumulative-ashfall exposure analysis")
    if any(
        threshold["minimum_mm"] < 0
        for threshold in ASH_EXPOSURE_THRESHOLDS
    ):
        raise ValueError("Ashfall exposure thresholds cannot be negative.")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    manifest = load_manifest()
    population = load_population_sa2(manifest)
    buildings = (
        load_buildings(BUILDING_FOOTPRINTS, population)
        if USE_BUILDINGS
        else None
    )
    long_summaries, population_summaries = run_scenarios(
        population,
        buildings,
        manifest,
    )
    save_combined_summaries(long_summaries, population_summaries)
    report_progress("Taranaki exposure analysis complete")


if __name__ == "__main__":
    main()
