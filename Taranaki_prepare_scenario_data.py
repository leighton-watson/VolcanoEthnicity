"""Prepare cached Taranaki ashfall and SA2 data for figure generation.

Run this script when the raw ashfall rasters or population CSV change. It sums
all phase rasters for each scenario, reprojects the cumulative ashfall to New
Zealand Transverse Mercator, crops empty raster margins, and writes lightweight
cached inputs for ``taranaki_scenario.py``.

The plotting script does not repeat this analysis, making visual iterations
substantially faster.
"""

# %%
from json import dump
from pathlib import Path
from time import perf_counter

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.merge import merge
from rasterio.transform import array_bounds
from rasterio.warp import calculate_default_transform, reproject, Resampling
from shapely import wkt


# %%
# ---------------------------
# Paths and analysis settings
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ASH_FOLDER = PROJECT_ROOT / "DATA" / "Taranaki" / "Taranaki_data" / "ash"
POPULATION_DATA_FILE = (
    PROJECT_ROOT
    / "DATA"
    / "PopulationData"
    / "estimated-resident-population-at-30-june-2018-by-statistical.csv"
)
CACHE_FOLDER = PROJECT_ROOT / "MakeFigures" / "OUTPUT" / "Taranaki" / "cache"
SA2_CACHE_FILE = CACHE_FOLDER / "taranaki_sa2_population_ethnicity.gpkg"
MANIFEST_FILE = CACHE_FOLDER / "taranaki_figure_manifest.json"

SCENARIOS = ["S1", "S2", "S3", "M1", "M2", "M3", "L1", "L2", "L3"]
EXPECTED_PHASE_RASTERS = 6
TARGET_CRS = "EPSG:2193"
RASTER_RESAMPLING = Resampling.nearest
CROP_PADDING_CELLS = 1

SA2_KEY = "SA22020_V1_00"
SA2_NAME_FIELD = "SA22020_V1_00_NAME"
TOTAL_POPULATION_FIELD = "erp18"
POPULATION_DENSITY_FIELD = "population_density_km2"
ETHNICITY_PERCENT_FIELDS = [
    "pEurother",
    "pMaori",
    "pPacific",
    "pAsian",
    "pMELAA",
]

SCRIPT_START = perf_counter()


def report_progress(message):
    """Print an immediately flushed progress message with elapsed time."""
    elapsed = perf_counter() - SCRIPT_START
    print(f"[{elapsed:7.1f}s] {message}", flush=True)


def scenario_cache_path(scenario):
    """Return the output path for one cumulative scenario raster."""
    return CACHE_FOLDER / f"{scenario}_cumulative_ashfall_mm.tif"


def list_phase_rasters(ash_folder, scenario):
    """Return all phase rasters belonging to one scenario."""
    prefix = f"{scenario.lower()}_"
    return sorted(
        path
        for path in ash_folder.glob("*.tif")
        if path.name.lower().startswith(prefix)
    )


def crop_to_positive_ash(array, transform):
    """Crop empty margins while retaining a zero border around positive ash."""
    positive_rows, positive_columns = np.where(array > 0)
    if positive_rows.size == 0:
        raise ValueError("The cumulative ashfall raster contains no positive values.")

    row_start = int(positive_rows.min())
    row_stop = int(positive_rows.max()) + 1
    column_start = int(positive_columns.min())
    column_stop = int(positive_columns.max()) + 1

    cropped = array[row_start:row_stop, column_start:column_stop]
    cropped = np.pad(
        cropped,
        pad_width=CROP_PADDING_CELLS,
        mode="constant",
        constant_values=0,
    )
    cropped_transform = transform * rasterio.Affine.translation(
        column_start - CROP_PADDING_CELLS,
        row_start - CROP_PADDING_CELLS,
    )
    return cropped, cropped_transform


def sum_scenario_phase_rasters(phase_paths, target_crs):
    """Sum all phase rasters and return a cropped cumulative raster."""
    datasets = [rasterio.open(path) for path in phase_paths]
    try:
        source_crs = datasets[0].crs
        if source_crs is None:
            raise ValueError(f"Raster has no CRS: {phase_paths[0]}")

        for dataset, path in zip(datasets, phase_paths):
            if dataset.crs != source_crs:
                raise ValueError(
                    f"All phase rasters must use one CRS; {path.name} uses "
                    f"{dataset.crs}, expected {source_crs}."
                )

        cumulative, source_transform = merge(
            datasets,
            method="sum",
            nodata=0,
            dtype="float32",
        )
    finally:
        for dataset in datasets:
            dataset.close()

    cumulative = cumulative[0]
    cumulative[~np.isfinite(cumulative)] = 0
    cumulative[cumulative < 0] = 0

    if source_crs.to_string() != target_crs:
        source_bounds = array_bounds(
            cumulative.shape[0],
            cumulative.shape[1],
            source_transform,
        )
        target_transform, target_width, target_height = calculate_default_transform(
            source_crs,
            target_crs,
            cumulative.shape[1],
            cumulative.shape[0],
            *source_bounds,
        )
        reprojected = np.zeros((target_height, target_width), dtype=np.float32)
        reproject(
            source=cumulative,
            destination=reprojected,
            src_transform=source_transform,
            src_crs=source_crs,
            src_nodata=0,
            dst_transform=target_transform,
            dst_crs=target_crs,
            dst_nodata=0,
            resampling=RASTER_RESAMPLING,
        )
        cumulative = reprojected
        source_transform = target_transform

    return crop_to_positive_ash(cumulative, source_transform)


def write_cumulative_raster(output_path, array, transform):
    """Write one prepared cumulative ashfall GeoTIFF."""
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype="float32",
        crs=TARGET_CRS,
        transform=transform,
        nodata=0,
        compress="deflate",
        predictor=3,
    ) as destination:
        destination.write(array.astype(np.float32), 1)


def prepare_ashfall_rasters():
    """Prepare all nine cumulative scenario rasters and their shared extent."""
    scenario_metadata = {}
    combined_bounds = []

    for scenario in SCENARIOS:
        phase_paths = list_phase_rasters(ASH_FOLDER, scenario)
        if not phase_paths:
            raise FileNotFoundError(f"No ashfall rasters found for scenario {scenario}.")
        if len(phase_paths) != EXPECTED_PHASE_RASTERS:
            report_progress(
                f"Warning: {scenario} has {len(phase_paths)} phase rasters; "
                f"expected {EXPECTED_PHASE_RASTERS}."
            )

        report_progress(
            f"Summing {len(phase_paths)} phase rasters for scenario {scenario}..."
        )
        cumulative, transform = sum_scenario_phase_rasters(
            phase_paths,
            TARGET_CRS,
        )
        output_path = scenario_cache_path(scenario)
        write_cumulative_raster(output_path, cumulative, transform)

        west, south, east, north = array_bounds(
            cumulative.shape[0],
            cumulative.shape[1],
            transform,
        )
        bounds = [west, south, east, north]
        combined_bounds.append(bounds)
        scenario_metadata[scenario] = {
            "raster": output_path.name,
            "phase_rasters": [path.name for path in phase_paths],
            "bounds": bounds,
            "minimum_mm": float(cumulative[cumulative > 0].min()),
            "maximum_mm": float(cumulative.max()),
        }
        report_progress(f"Wrote cumulative ashfall: {output_path}")

    bounds_array = np.asarray(combined_bounds)
    combined_extent = [
        float(bounds_array[:, 0].min()),
        float(bounds_array[:, 1].min()),
        float(bounds_array[:, 2].max()),
        float(bounds_array[:, 3].max()),
    ]
    return scenario_metadata, combined_extent


def prepare_sa2_cache(combined_extent):
    """Parse, calculate, crop, and cache the SA2 attributes used by the plots."""
    use_columns = [
        "WKT",
        SA2_KEY,
        SA2_NAME_FIELD,
        TOTAL_POPULATION_FIELD,
        *ETHNICITY_PERCENT_FIELDS,
    ]
    report_progress("Reading and parsing the SA2 population CSV...")
    data = pd.read_csv(POPULATION_DATA_FILE, usecols=use_columns)

    numeric_fields = [TOTAL_POPULATION_FIELD, *ETHNICITY_PERCENT_FIELDS]
    for field in numeric_fields:
        data[field] = pd.to_numeric(data[field], errors="coerce")
        data.loc[data[field] == -999, field] = pd.NA

    geometry = data["WKT"].apply(wkt.loads)
    sa2_gdf = gpd.GeoDataFrame(
        data.drop(columns="WKT"),
        geometry=geometry,
        crs=TARGET_CRS,
    )
    area_km2 = sa2_gdf.geometry.area / 1_000_000
    sa2_gdf[POPULATION_DENSITY_FIELD] = (
        sa2_gdf[TOTAL_POPULATION_FIELD] / area_km2
    )

    xmin, ymin, xmax, ymax = combined_extent
    sa2_gdf = sa2_gdf.cx[xmin:xmax, ymin:ymax].copy()
    if sa2_gdf.empty:
        raise ValueError("No SA2 regions intersect the combined ashfall extent.")

    sa2_gdf.to_file(SA2_CACHE_FILE, layer="sa2", driver="GPKG")
    report_progress(f"Wrote prepared SA2 data: {SA2_CACHE_FILE}")
    return len(sa2_gdf)


def write_manifest(scenario_metadata, combined_extent, sa2_count):
    """Write paths and extent metadata consumed by the plotting script."""
    manifest = {
        "crs": TARGET_CRS,
        "aggregation": "sum of all phase rasters",
        "combined_extent": combined_extent,
        "sa2_cache": SA2_CACHE_FILE.name,
        "sa2_count": sa2_count,
        "scenarios": scenario_metadata,
    }
    with MANIFEST_FILE.open("w", encoding="utf-8") as stream:
        dump(manifest, stream, indent=2)
    report_progress(f"Wrote plotting manifest: {MANIFEST_FILE}")


def main():
    """Prepare all cached inputs used by the Taranaki plotting script."""
    report_progress("Starting Taranaki figure-data preparation")
    CACHE_FOLDER.mkdir(parents=True, exist_ok=True)
    scenario_metadata, combined_extent = prepare_ashfall_rasters()
    sa2_count = prepare_sa2_cache(combined_extent)
    write_manifest(scenario_metadata, combined_extent, sa2_count)
    report_progress("Preparation complete; taranaki_scenario.py is ready to run")


if __name__ == "__main__":
    main()
