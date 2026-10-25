"""
Explain area-weighted vs building-weighted exposure using one SA2 region.

This script plots the Coatesville SA2 polygon, building footprints inside it,
and the Scenario D trace ashfall area. It is intended as a simple conceptual
figure showing how ash can cover only part of an SA2, while buildings are
clustered within that area.

Run this script from Spyder or from the project root.
"""

from math import atan, cos, sinh
from pathlib import Path

import contextily as ctx
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import pandas as pd
from shapely import wkt


PROJECT_ROOT = Path(__file__).resolve().parents[1]
POPULATION_DATA_FILE = (
    PROJECT_ROOT
    / "DATA"
    / "PopulationData"
    / "estimated-resident-population-at-30-june-2018-by-statistical.csv"
)
BUILDINGS_FILE = PROJECT_ROOT / "DATA" / "AucklandBuildings" / "nz-building-outlines.shp"
SCENARIO_D_FILE = PROJECT_ROOT / "DATA" / "Auckland" / "Scenario_D_Tephra.gpkg"
OUTPUT_DIR = PROJECT_ROOT / "OUTPUT"
OUTPUT_FILE = OUTPUT_DIR / "coatesville_buildings_scenario_d_trace_ash.png"

SA2_NAME_FIELD = "SA22020_V1_00_NAME"
SA2_NAME = "Oneroa West"
THICKNESS_COLUMN = "thickness"
TARGET_CRS = "EPSG:2193"
MAP_CRS = "EPSG:3857"

ADD_BASEMAP = True
BASEMAP_SOURCE = ctx.providers.CartoDB.PositronNoLabels
MAP_BUFFER_METRES = 200
TRACE_ASH_THRESHOLD = 0
TRACE_BOUNDARY_CLOSE_BUFFER_METRES = 300
SCALE_BAR_KM = 1


def load_sa2_region(csv_path, sa2_name):
    """Load one SA2 polygon by name from the Stats NZ CSV."""
    use_columns = ["WKT", "SA22020_V1_00", SA2_NAME_FIELD]
    data = pd.read_csv(csv_path, usecols=use_columns)
    match = data.loc[data[SA2_NAME_FIELD].str.casefold() == sa2_name.casefold()].copy()

    if match.empty:
        close_names = data.loc[
            data[SA2_NAME_FIELD].str.contains(sa2_name, case=False, na=False),
            SA2_NAME_FIELD,
        ].tolist()
        raise ValueError(
            f"Could not find SA2 named '{sa2_name}'. Similar matches: {close_names}"
        )

    match["geometry"] = match["WKT"].apply(wkt.loads)
    sa2_gdf = gpd.GeoDataFrame(match.drop(columns="WKT"), geometry="geometry", crs=TARGET_CRS)
    return sa2_gdf


def load_buildings_inside_sa2(buildings_path, sa2_gdf):
    """Read and clip building footprints inside the selected SA2 polygon."""
    sa2_geom = sa2_gdf.geometry.union_all()
    buildings = gpd.read_file(buildings_path, bbox=tuple(sa2_gdf.total_bounds))

    if buildings.crs != sa2_gdf.crs:
        buildings = buildings.to_crs(sa2_gdf.crs)

    buildings = buildings.loc[buildings.intersects(sa2_geom)].copy()
    buildings["geometry"] = buildings.geometry.intersection(sa2_geom)
    buildings = buildings.loc[~buildings.geometry.is_empty].copy()
    return buildings


def load_trace_ash_for_sa2(scenario_path, sa2_gdf):
    """Load Scenario D trace ash polygons across the displayed map area."""
    map_window = sa2_gdf.geometry.buffer(MAP_BUFFER_METRES).total_bounds
    ash = gpd.read_file(scenario_path, bbox=tuple(map_window))

    if ash.crs != sa2_gdf.crs:
        ash = ash.to_crs(sa2_gdf.crs)

    ash[THICKNESS_COLUMN] = pd.to_numeric(ash[THICKNESS_COLUMN], errors="coerce")
    ash = ash.loc[ash[THICKNESS_COLUMN] > TRACE_ASH_THRESHOLD].copy()
    ash = ash.loc[~ash.geometry.is_empty].copy()
    return ash


def get_clean_trace_boundary(trace_ash_gdf):
    """Close small gaps between ashfall grid cells before drawing the boundary."""
    if trace_ash_gdf.empty:
        return None

    closed_geometry = (
        trace_ash_gdf.geometry.buffer(TRACE_BOUNDARY_CLOSE_BUFFER_METRES)
        .union_all()
        .buffer(-TRACE_BOUNDARY_CLOSE_BUFFER_METRES)
    )
    return closed_geometry.boundary


def calculate_exposure_summary(sa2_gdf, buildings_gdf, trace_ash_gdf):
    """Calculate area-weighted and building-weighted exposure summaries."""
    sa2_geom = sa2_gdf.geometry.union_all()

    if trace_ash_gdf.empty:
        ash_geom = None
        ash_area_m2 = 0.0
        area_percent = 0.0
        buildings_in_ash = 0
    else:
        ash_geom = trace_ash_gdf.geometry.union_all()
        ash_area_m2 = sa2_geom.intersection(ash_geom).area
        area_percent = 100 * ash_area_m2 / sa2_geom.area
        buildings_in_ash = buildings_gdf.intersects(ash_geom).sum()

    total_buildings = len(buildings_gdf)
    building_percent = (
        100 * buildings_in_ash / total_buildings
        if total_buildings > 0
        else 0.0
    )

    return {
        "sa2_area_m2": sa2_geom.area,
        "ash_area_m2": ash_area_m2,
        "area_percent": area_percent,
        "total_buildings": total_buildings,
        "buildings_in_ash": int(buildings_in_ash),
        "building_percent": building_percent,
    }


def add_plot_frame(ax):
    """Show a black frame while keeping map axes hidden."""
    ax.set_axis_off()
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.2)


def add_scale_bar(ax, extent, length_km=SCALE_BAR_KM):
    """Add a ground-distance scale bar to a Web Mercator map."""
    xmin, ymin, xmax, ymax = extent
    map_width = xmax - xmin
    map_height = ymax - ymin

    # EPSG:3857 stretches distances by sec(latitude), so convert the desired
    # ground distance to its corresponding length in projected map units.
    earth_radius_metres = 6_378_137
    centre_y = (ymin + ymax) / 2
    latitude_radians = atan(sinh(centre_y / earth_radius_metres))
    projected_length = length_km * 1_000 / cos(latitude_radians)

    right = xmax - 0.04 * map_width
    left = right - projected_length
    bar_y = ymin + 0.055 * map_height
    tick_height = 0.012 * map_height

    halo = [path_effects.Stroke(linewidth=5, foreground="white"), path_effects.Normal()]
    ax.plot(
        [left, right],
        [bar_y, bar_y],
        color="black",
        linewidth=2.5,
        solid_capstyle="butt",
        path_effects=halo,
        zorder=10,
    )
    for x in (left, right):
        ax.plot(
            [x, x],
            [bar_y - tick_height / 2, bar_y + tick_height / 2],
            color="black",
            linewidth=2.5,
            path_effects=halo,
            zorder=10,
        )

    label = ax.text(
        (left + right) / 2,
        bar_y + 0.012 * map_height,
        f"{length_km:g} km",
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        color="black",
        zorder=10,
    )
    label.set_path_effects(
        [path_effects.Stroke(linewidth=3, foreground="white"), path_effects.Normal()]
    )


def plot_coatesville_example(sa2_gdf, buildings_gdf, trace_ash_gdf, exposure_summary):
    """Create the explanatory Coatesville figure."""
    sa2_web = sa2_gdf.to_crs(MAP_CRS)
    buildings_web = buildings_gdf.to_crs(MAP_CRS)
    trace_ash_web = trace_ash_gdf.to_crs(MAP_CRS)

    xmin, ymin, xmax, ymax = sa2_web.total_bounds
    extent = (
        xmin - MAP_BUFFER_METRES,
        ymin - MAP_BUFFER_METRES,
        xmax + MAP_BUFFER_METRES,
        ymax + MAP_BUFFER_METRES,
    )

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])

    if ADD_BASEMAP:
        try:
            ctx.add_basemap(
                ax,
                source=BASEMAP_SOURCE,
                attribution=False,
                reset_extent=False,
            )
        except Exception as exc:
            print(f"Basemap skipped: {exc}")

    ax.set_xlim(extent[0], extent[2])
    ax.set_ylim(extent[1], extent[3])

    if not trace_ash_web.empty:
        trace_ash_web.plot(
            ax=ax,
            facecolor="red",
            edgecolor="none",
            alpha=0.28,
            label="Scenario D trace ash",
            zorder=2,
        )

    sa2_web.plot(
        ax=ax,
        facecolor="none",
        edgecolor="black",
        linewidth=2.0,
        label=f"{SA2_NAME} SA2 boundary",
        zorder=4,
    )

    buildings_web.plot(
        ax=ax,
        facecolor="#003f8c",
        edgecolor="#001f4d",
        linewidth=1,
        alpha=1.0,
        label="Building footprints",
        zorder=5,
    )

    add_scale_bar(ax, extent)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )

    return fig


def main():
    sa2_gdf = load_sa2_region(POPULATION_DATA_FILE, SA2_NAME)
    buildings_gdf = load_buildings_inside_sa2(BUILDINGS_FILE, sa2_gdf)
    trace_ash_gdf = load_trace_ash_for_sa2(SCENARIO_D_FILE, sa2_gdf)
    exposure_summary = calculate_exposure_summary(sa2_gdf, buildings_gdf, trace_ash_gdf)

    print("Coatesville Scenario D trace ash exposure")
    print(f"SA2 area covered by ash: {exposure_summary['area_percent']:.1f}%")
    print(
        "Buildings in ash: "
        f"{exposure_summary['buildings_in_ash']} of "
        f"{exposure_summary['total_buildings']} "
        f"({exposure_summary['building_percent']:.1f}%)"
    )

    fig = plot_coatesville_example(
        sa2_gdf,
        buildings_gdf,
        trace_ash_gdf,
        exposure_summary,
    )

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")
    plt.show()


if __name__ == "__main__":
    main()
