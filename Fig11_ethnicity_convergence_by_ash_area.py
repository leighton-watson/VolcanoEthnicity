"""
Plot how exposed ethnicity percentages approach regional composition.

For each Auckland and Taranaki ashfall scenario, this script calculates:

    x: ash-covered area as a percentage of the plotted regional SA2 area
    y: exposed ethnicity percentage for each ethnicity group

Regional composition is calculated from the same SA2 geometries used for the
plotting region and shown as dashed horizontal reference lines.

Run this script from Spyder or from the project root.
"""

# %%
from json import load
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import shapes
from shapely import wkt
from shapely.geometry import shape


# %%
# ---------------------------
# Paths and settings
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_FOLDER = PROJECT_ROOT / "MakeFigures" / "OUTPUT"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
OUTPUT_FILE = OUTPUT_FOLDER / "Fig_Ethnicity_Convergence_By_Ash_Area.png"
OUTPUT_DATA_FILE = OUTPUT_FOLDER / "Fig_Ethnicity_Convergence_By_Ash_Area_points.csv"

POPULATION_DATA_FILE = (
    PROJECT_ROOT
    / "DATA"
    / "PopulationData"
    / "estimated-resident-population-at-30-june-2018-by-statistical.csv"
)
AUCKLAND_DIR = PROJECT_ROOT / "DATA" / "Auckland"
TARANAKI_CACHE_FOLDER = PROJECT_ROOT / "MakeFigures" / "OUTPUT" / "Taranaki" / "cache"
TARANAKI_MANIFEST_FILE = TARANAKI_CACHE_FOLDER / "taranaki_figure_manifest.json"

MAP_CRS = "EPSG:2193"
TOTAL_POPULATION_FIELD = "erp18"
SA2_CODE_FIELD = "SA22020_V1_00"
SA2_NAME_FIELD = "SA22020_V1_00_NAME"
AREA_FIELD = "area_m2"
THICKNESS_COLUMN = "thickness"

ETHNICITY_FIELDS = {
    "Eurother": "European or Other",
    "Maori": "Māori",
    "Pacific": "Pacific",
    "Asian": "Asian",
    "MELAA": "MELAA",
}
ETHNICITY_PERCENT_FIELDS = {
    field: f"p{field}"
    for field in ETHNICITY_FIELDS
}
ETHNICITY_COLOURS = {
    "Eurother": "#d95f02",
    "Maori": "#1b9e77",
    "Pacific": "#377eb8",
    "Asian": "#984ea3",
    "MELAA": "#e41a1c",
}

AUCKLAND_SCENARIOS = list("ABCDEFGH")
AUCKLAND_SCENARIO_PATHS = {
    scenario: AUCKLAND_DIR / f"Scenario_{scenario}_Tephra.gpkg"
    for scenario in AUCKLAND_SCENARIOS
}
AUCKLAND_EXTENT_BUFFER_FRACTION = 0.00
AUCKLAND_EXTENT_INSET_FRACTION = 0.05

TARANAKI_SCENARIOS = ["S1", "S2", "S3", "M1", "M2", "M3", "L1", "L2", "L3"]

# Trace uses all positive ashfall values; the other thresholds are in mm.
ASH_THRESHOLDS = [0, 1, 10, 25, 100, 500, 1000]
THRESHOLD_MARKERS = {
    0: "o",
    1: "s",
    10: "^",
    25: "D",
    100: "P",
    500: "X",
    1000: "*",
}

FIGURE_SIZE = (14, 14)
PANEL_TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
TICK_LABEL_FONTSIZE = 12
LEGEND_FONTSIZE = 13
OUTPUT_DPI = 300


# %%
def load_sa2_population_data(csv_path):
    """Load SA2 population/ethnicity counts and calculate geometry area."""
    use_columns = [
        "WKT",
        SA2_CODE_FIELD,
        SA2_NAME_FIELD,
        TOTAL_POPULATION_FIELD,
        *ETHNICITY_FIELDS.keys(),
    ]
    data = pd.read_csv(csv_path, usecols=use_columns)

    for field in [TOTAL_POPULATION_FIELD, *ETHNICITY_FIELDS.keys()]:
        data[field] = pd.to_numeric(data[field], errors="coerce")
        data.loc[data[field] == -999, field] = np.nan
        data[field] = data[field].fillna(0)

    geometry = data["WKT"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(data.drop(columns="WKT"), geometry=geometry, crs="EPSG:2193")
    gdf[AREA_FIELD] = gdf.geometry.area
    return gdf.to_crs(MAP_CRS)


def ensure_ethnicity_count_fields(gdf):
    """Ensure ethnicity count fields exist, rebuilding them from percentages."""
    gdf = gdf.copy()
    gdf[TOTAL_POPULATION_FIELD] = pd.to_numeric(
        gdf[TOTAL_POPULATION_FIELD],
        errors="coerce",
    ).fillna(0)

    for count_field, percent_field in ETHNICITY_PERCENT_FIELDS.items():
        if count_field in gdf.columns:
            gdf[count_field] = pd.to_numeric(
                gdf[count_field],
                errors="coerce",
            ).fillna(0)
            continue

        if percent_field not in gdf.columns:
            raise KeyError(
                f"Missing both '{count_field}' and '{percent_field}' in SA2 data."
            )

        percentages = pd.to_numeric(gdf[percent_field], errors="coerce").fillna(0)
        gdf[count_field] = gdf[TOTAL_POPULATION_FIELD] * percentages / 100

    return gdf


def load_auckland_scenarios():
    """Load Auckland scenario polygons in the analysis CRS."""
    scenarios = {}
    for scenario, path in AUCKLAND_SCENARIO_PATHS.items():
        if not path.exists():
            raise FileNotFoundError(f"Auckland scenario file not found: {path}")
        gdf = gpd.read_file(path)
        if THICKNESS_COLUMN not in gdf.columns:
            raise ValueError(f"{path.name} has no '{THICKNESS_COLUMN}' column.")
        scenarios[scenario] = gdf.to_crs(MAP_CRS)
    return scenarios


def get_combined_extent(gdfs):
    """Return the combined bounds of a dictionary of GeoDataFrames."""
    bounds = np.array([gdf.total_bounds for gdf in gdfs.values()])
    return (
        bounds[:, 0].min(),
        bounds[:, 1].min(),
        bounds[:, 2].max(),
        bounds[:, 3].max(),
    )


def buffer_extent(extent, buffer_fraction):
    """Expand or contract an extent by a fraction of its width and height."""
    xmin, ymin, xmax, ymax = extent
    x_buffer = (xmax - xmin) * buffer_fraction
    y_buffer = (ymax - ymin) * buffer_fraction
    return (
        xmin - x_buffer,
        ymin - y_buffer,
        xmax + x_buffer,
        ymax + y_buffer,
    )


def inset_extent(extent, inset_fraction):
    """Zoom in by shrinking each side of the extent by a fixed fraction."""
    xmin, ymin, xmax, ymax = extent
    x_inset = (xmax - xmin) * inset_fraction
    y_inset = (ymax - ymin) * inset_fraction
    return (
        xmin + x_inset,
        ymin + y_inset,
        xmax - x_inset,
        ymax - y_inset,
    )


def crop_to_extent(gdf, extent):
    """Keep geometries intersecting the plotting extent."""
    xmin, ymin, xmax, ymax = extent
    return gdf.cx[xmin:xmax, ymin:ymax].copy()


def regional_ethnicity_percentages(sa2_gdf):
    """Calculate regional ethnicity percentages from the plotted SA2 data."""
    total_population = sa2_gdf[TOTAL_POPULATION_FIELD].sum(skipna=True)
    return {
        field: sa2_gdf[field].sum(skipna=True) / total_population * 100
        for field in ETHNICITY_FIELDS
    }


def threshold_label(threshold):
    """Return a short display label for an ashfall threshold."""
    if threshold == 0:
        return "Trace > 0 mm"
    return f">= {threshold:g} mm"


def make_exposure_record(region, scenario, threshold, hazard_geometry, sa2_gdf):
    """Calculate one exposure-composition point for a scenario threshold."""
    if hazard_geometry is None or hazard_geometry.is_empty:
        return None

    exposed_area = 0.0
    exposed_population = 0.0
    exposed_ethnicity = {field: 0.0 for field in ETHNICITY_FIELDS}

    candidate_sa2 = sa2_gdf.loc[sa2_gdf.geometry.intersects(hazard_geometry)].copy()
    for _, row in candidate_sa2.iterrows():
        if row[AREA_FIELD] <= 0 or row[TOTAL_POPULATION_FIELD] <= 0:
            continue

        intersection = row.geometry.intersection(hazard_geometry)
        if intersection.is_empty:
            continue

        intersection_area = intersection.area
        exposure_fraction = intersection_area / row[AREA_FIELD]
        exposed_area += intersection_area
        exposed_population += row[TOTAL_POPULATION_FIELD] * exposure_fraction
        for field in ETHNICITY_FIELDS:
            exposed_ethnicity[field] += row[field] * exposure_fraction

    if exposed_area <= 0 or exposed_population <= 0:
        return None

    total_region_area = sa2_gdf[AREA_FIELD].sum()
    record = {
        "region": region,
        "scenario": scenario,
        "threshold_mm": threshold,
        "threshold_label": threshold_label(threshold),
        "ash_area_percent": exposed_area / total_region_area * 100,
        "exposed_population": exposed_population,
    }
    for field in ETHNICITY_FIELDS:
        record[field] = exposed_ethnicity[field] / exposed_population * 100
    return record


def calculate_auckland_records(sa2_population_gdf):
    """Calculate Auckland exposure-composition points."""
    scenarios = load_auckland_scenarios()
    extent = get_combined_extent(scenarios)
    extent = buffer_extent(extent, AUCKLAND_EXTENT_BUFFER_FRACTION)
    extent = inset_extent(extent, AUCKLAND_EXTENT_INSET_FRACTION)
    sa2_gdf = crop_to_extent(sa2_population_gdf, extent)

    records = []
    for scenario, scenario_gdf in scenarios.items():
        valid = scenario_gdf.loc[scenario_gdf[THICKNESS_COLUMN].notna()].copy()
        for threshold in ASH_THRESHOLDS:
            if threshold == 0:
                threshold_gdf = valid.loc[valid[THICKNESS_COLUMN] > 0]
            else:
                threshold_gdf = valid.loc[valid[THICKNESS_COLUMN] >= threshold]
            if threshold_gdf.empty:
                continue

            hazard_geometry = threshold_gdf.geometry.union_all()
            record = make_exposure_record(
                region="Auckland",
                scenario=scenario,
                threshold=threshold,
                hazard_geometry=hazard_geometry,
                sa2_gdf=sa2_gdf,
            )
            if record is not None:
                records.append(record)

    return pd.DataFrame(records), regional_ethnicity_percentages(sa2_gdf)


def load_taranaki_inputs():
    """Load cached Taranaki SA2 data and raster paths."""
    if not TARANAKI_MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Taranaki manifest not found: {TARANAKI_MANIFEST_FILE}\n"
            "Run MakeFigures/prepare_taranaki_scenario_data.py first."
        )

    with TARANAKI_MANIFEST_FILE.open(encoding="utf-8") as stream:
        manifest = load(stream)

    sa2_path = TARANAKI_CACHE_FOLDER / manifest["sa2_cache"]
    sa2_gdf = gpd.read_file(sa2_path, layer="sa2")
    if sa2_gdf.crs is None:
        sa2_gdf = sa2_gdf.set_crs(manifest["crs"])
    if sa2_gdf.crs.to_string() != MAP_CRS:
        sa2_gdf = sa2_gdf.to_crs(MAP_CRS)
    sa2_gdf = ensure_ethnicity_count_fields(sa2_gdf)
    sa2_gdf[AREA_FIELD] = sa2_gdf.geometry.area

    raster_paths = {
        scenario: TARANAKI_CACHE_FOLDER / manifest["scenarios"][scenario]["raster"]
        for scenario in TARANAKI_SCENARIOS
    }
    return sa2_gdf, raster_paths


def raster_threshold_geometry(raster_path, threshold):
    """Convert a raster threshold mask to a dissolved polygon geometry."""
    with rasterio.open(raster_path) as source:
        ashfall = source.read(1, masked=False).astype(float)
        transform = source.transform
        raster_crs = source.crs

    ashfall[~np.isfinite(ashfall)] = 0
    if threshold == 0:
        mask = ashfall > 0
    else:
        mask = ashfall >= threshold
    if not np.any(mask):
        return None

    polygons = [
        shape(geometry)
        for geometry, value in shapes(
            mask.astype("uint8"),
            mask=mask,
            transform=transform,
        )
        if value == 1
    ]
    if not polygons:
        return None

    hazard_gdf = gpd.GeoDataFrame(geometry=polygons, crs=raster_crs).to_crs(MAP_CRS)
    return hazard_gdf.geometry.union_all()


def calculate_taranaki_records():
    """Calculate Taranaki exposure-composition points."""
    sa2_gdf, raster_paths = load_taranaki_inputs()
    records = []
    for scenario, raster_path in raster_paths.items():
        if not raster_path.exists():
            raise FileNotFoundError(f"Taranaki raster not found: {raster_path}")

        for threshold in ASH_THRESHOLDS:
            hazard_geometry = raster_threshold_geometry(raster_path, threshold)
            record = make_exposure_record(
                region="Taranaki",
                scenario=scenario,
                threshold=threshold,
                hazard_geometry=hazard_geometry,
                sa2_gdf=sa2_gdf,
            )
            if record is not None:
                records.append(record)

    return pd.DataFrame(records), regional_ethnicity_percentages(sa2_gdf)


def plot_ethnicity_panel(ax, records, regional_percentages, field):
    """Plot one ethnicity's exposure-composition convergence panel."""
    data = records.sort_values("ash_area_percent")
    positive_area = data.loc[data["ash_area_percent"] > 0, "ash_area_percent"]
    colour = ETHNICITY_COLOURS[field]
    for threshold in sorted(data["threshold_mm"].unique()):
        threshold_data = data.loc[data["threshold_mm"] == threshold]
        ax.scatter(
            threshold_data["ash_area_percent"],
            threshold_data[field],
            color=colour,
            marker=THRESHOLD_MARKERS.get(threshold, "o"),
            s=70,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.9,
            zorder=3,
        )
    ax.axhline(
        regional_percentages[field],
        color=colour,
        linestyle="--",
        linewidth=1.2,
        alpha=0.8,
    )

    ax.tick_params(labelsize=TICK_LABEL_FONTSIZE)
    ax.grid(True, alpha=0.25)
    ax.set_xscale("linear")
    if not positive_area.empty:
        ax.set_xlim(left=positive_area.min() * 0.8)


def add_combined_legend(fig):
    """Add threshold and regional-composition legends below the panels."""
    threshold_handles = [
        Line2D(
            [0],
            [0],
            marker=THRESHOLD_MARKERS[threshold],
            color="black",
            markerfacecolor="white",
            linestyle="None",
            label=threshold_label(threshold),
        )
        for threshold in ASH_THRESHOLDS
    ]
    regional_handle = Line2D(
        [0],
        [0],
        color="black",
        linestyle="--",
        label="Regional composition",
    )

    legend_handles = threshold_handles + [regional_handle]
    fig.legend(
        handles=legend_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.025),
        ncol=4,
        frameon=True,
        fontsize=LEGEND_FONTSIZE,
        markerscale=1.4,
    )


def plot_convergence_figure(
    auckland_records,
    auckland_regional_percentages,
    taranaki_records,
    taranaki_regional_percentages,
):
    """Create the ethnicity-row convergence figure."""
    fig, axes = plt.subplots(
        nrows=len(ETHNICITY_FIELDS),
        ncols=2,
        figsize=FIGURE_SIZE,
        constrained_layout=False,
        sharex="col",
        sharey=False,
    )

    axes[0, 0].set_title("Auckland", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
    axes[0, 1].set_title("Taranaki", fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")

    for row_index, (field, display_label) in enumerate(ETHNICITY_FIELDS.items()):
        plot_ethnicity_panel(
            axes[row_index, 0],
            auckland_records,
            auckland_regional_percentages,
            field,
        )
        plot_ethnicity_panel(
            axes[row_index, 1],
            taranaki_records,
            taranaki_regional_percentages,
            field,
        )
        axes[row_index, 0].set_ylabel(
            f"{display_label} (%)",
            fontsize=AXIS_LABEL_FONTSIZE,
        )
        axes[row_index, 1].set_ylabel(
            f"{display_label} (%)",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

    for ax in axes[:-1, :].flat:
        ax.set_xlabel("")

    for ax in axes[-1, :]:
        ax.set_xlabel(
            "Ash-covered area (% of regional area)",
            fontsize=AXIS_LABEL_FONTSIZE,
        )

    add_combined_legend(fig)

    fig.subplots_adjust(
        left=0.10,
        right=0.98,
        top=0.95,
        bottom=0.12,
        hspace=0.18,
        wspace=0.18,
    )
    return fig


def main():
    """Calculate convergence data and save the figure."""
    sa2_population_gdf = load_sa2_population_data(POPULATION_DATA_FILE)
    auckland_records, auckland_regional = calculate_auckland_records(sa2_population_gdf)
    taranaki_records, taranaki_regional = calculate_taranaki_records()
    pd.concat([auckland_records, taranaki_records], ignore_index=True).to_csv(
        OUTPUT_DATA_FILE,
        index=False,
    )

    fig = plot_convergence_figure(
        auckland_records,
        auckland_regional,
        taranaki_records,
        taranaki_regional,
    )
    fig.savefig(OUTPUT_FILE, dpi=OUTPUT_DPI, bbox_inches="tight")

    # Keep the figure visible when running interactively in Spyder.
    plt.show()


if __name__ == "__main__":
    main()
