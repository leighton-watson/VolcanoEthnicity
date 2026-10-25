"""Plot Taranaki cumulative ashfall, population density, and ethnicity.

Run ``prepare_taranaki_scenario_data.py`` once before this script. The prepared
rasters and SA2 GeoPackage are reused here, so plotting styles can be adjusted
without repeating raster aggregation or WKT parsing.

Two figures are produced:

1. Nine cumulative ashfall scenarios over SA2 population density.
2. Five SA2 ethnicity percentage maps with a shared legend panel.
"""

# %%
from json import load
from pathlib import Path

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import rasterio
from shapely.geometry import Point


# %%
# -----------------------------------------------------------------------------
# Plotting settings: edit these without rerunning the preparation/analysis step
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CACHE_FOLDER = PROJECT_ROOT / "MakeFigures" / "OUTPUT" / "Taranaki" / "cache"
MANIFEST_FILE = CACHE_FOLDER / "taranaki_figure_manifest.json"
OUTPUT_FOLDER = PROJECT_ROOT / "MakeFigures" / "OUTPUT"
SCENARIO_OUTPUT_FILE = OUTPUT_FOLDER / "Fig_Taranaki_Scenarios.png"
ETHNICITY_OUTPUT_FILE = OUTPUT_FOLDER / "Fig_Taranaki_Ethnicity.png"

SCENARIOS = ["S1", "S2", "S3", "M1", "M2", "M3", "L1", "L2", "L3"]

# Cumulative ashfall contour levels and colours. Edit this ordered mapping to
# change the contour styling without rerunning the preparation/analysis step.
ASH_CONTOUR_COLOURS = {
    1: "#FFFF00",     # yellow
    10: "#FFA500",     # orange
    25: "#FF0000",    # red
    100: "#B22222",   # dark red
    500: "#67000D",   # very dark red
    1000: "#000000",  # black
}
ASH_CONTOUR_LEVELS_MM = list(ASH_CONTOUR_COLOURS)
ASH_CONTOUR_LINEWIDTH_RANGE = (0.8, 2.4)

TOTAL_POPULATION_FIELD = "erp18"
POPULATION_DENSITY_FIELD = "population_density_km2"
POPULATION_DENSITY_COLOR_LIMITS = (0, 100)

POPULATION_DENSITY_CMAP = "magma"

ETHNICITY_PERCENT_FIELDS = {
    "pEurother": "(a) European or Other",
    "pMaori": "(b) Māori",
    "pPacific": "(c) Pacific",
    "pAsian": "(d) Asian",
    "pMELAA": "(e) MELAA",
}
ETHNICITY_LEGEND_LABELS = {
    "pEurother": "European\nor Other",
    "pMaori": "Māori",
    "pPacific": "Pacific",
    "pAsian": "Asian",
    "pMELAA": "MELAA",
}
ETHNICITY_PERCENT_CMAPS = {
    "pEurother": "Oranges",
    "pMaori": "Greens",
    "pPacific": "Blues",
    "pAsian": "Purples",
    "pMELAA": "Reds",

}
ETHNICITY_LIMIT_INTERVAL = 5

EXTENT_BUFFER_FRACTION = -0.02
TARANAKI_VENT_LONLAT = (174.0632467, -39.29790130)
S1_LABELLED_PLACES_LONLAT = {
    "New Plymouth": (174.0839, -39.0579),
    "Stratford": (174.2840, -39.3370),
    "Hāwera": (174.2833, -39.5917),
}
VENT_MARKER_STYLE = {
    "marker": "*",
    "s": 250,
    "facecolor": "cyan",
    "edgecolor": "black",
    "linewidth": 0.8,
    "zorder": 6,
}

SCENARIO_FIGURE_SIZE = (14, 16)
ETHNICITY_FIGURE_SIZE = (14, 9)
OUTPUT_DPI = 300
PANEL_TITLE_FONTSIZE = 14
COLORBAR_LABEL_FONTSIZE = 14
COLORBAR_TICK_FONTSIZE = 12
LEGEND_CATEGORY_FONTSIZE = 14
ASH_LEGEND_FONTSIZE = 12
SA2_EDGE_COLOR = "black"
SA2_EDGE_LINEWIDTH = 0.2


# %%
def load_prepared_inputs():
    """Load cached rasters, SA2 attributes, CRS, and combined ashfall extent."""
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Prepared figure data not found: {MANIFEST_FILE}\n"
            "Run MakeFigures/prepare_taranaki_scenario_data.py first."
        )

    with MANIFEST_FILE.open(encoding="utf-8") as stream:
        manifest = load(stream)

    sa2_path = CACHE_FOLDER / manifest["sa2_cache"]
    if not sa2_path.exists():
        raise FileNotFoundError(
            f"Prepared SA2 data not found: {sa2_path}\n"
            "Run MakeFigures/prepare_taranaki_scenario_data.py first."
        )
    sa2_gdf = gpd.read_file(sa2_path, layer="sa2")

    scenario_paths = {}
    for scenario in SCENARIOS:
        try:
            raster_name = manifest["scenarios"][scenario]["raster"]
        except KeyError as exc:
            raise KeyError(f"Scenario {scenario} is missing from {MANIFEST_FILE}") from exc
        raster_path = CACHE_FOLDER / raster_name
        if not raster_path.exists():
            raise FileNotFoundError(
                f"Prepared raster not found: {raster_path}\n"
                "Run MakeFigures/prepare_taranaki_scenario_data.py first."
            )
        scenario_paths[scenario] = raster_path

    map_crs = manifest["crs"]
    if sa2_gdf.crs is None:
        raise ValueError(f"Prepared SA2 data has no CRS: {sa2_path}")
    if sa2_gdf.crs.to_string() != map_crs:
        sa2_gdf = sa2_gdf.to_crs(map_crs)

    return sa2_gdf, scenario_paths, manifest["combined_extent"], map_crs


def add_extent_buffer(extent, buffer_fraction):
    """Add a proportional margin around the combined positive-ash extent."""
    xmin, ymin, xmax, ymax = extent
    x_buffer = (xmax - xmin) * buffer_fraction
    y_buffer = (ymax - ymin) * buffer_fraction
    return (
        xmin - x_buffer,
        ymin - y_buffer,
        xmax + x_buffer,
        ymax + y_buffer,
    )


def set_map_extent(ax, extent):
    """Set map bounds before plotting map data."""
    xmin, ymin, xmax, ymax = extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


def get_vent_point(map_crs):
    """Return the Taranaki vent marker location in the map CRS."""
    longitude, latitude = TARANAKI_VENT_LONLAT
    return (
        gpd.GeoSeries([Point(longitude, latitude)], crs="EPSG:4326")
        .to_crs(map_crs)
        .iloc[0]
    )


def make_place_label_geodataframe(map_crs):
    """Create labelled place points for the S1 panel."""
    place_data = [
        {"name": name, "longitude": lon, "latitude": lat}
        for name, (lon, lat) in S1_LABELLED_PLACES_LONLAT.items()
    ]
    gdf = gpd.GeoDataFrame(
        place_data,
        geometry=gpd.points_from_xy(
            [place["longitude"] for place in place_data],
            [place["latitude"] for place in place_data],
        ),
        crs="EPSG:4326",
    )
    return gdf.to_crs(map_crs)


def plot_vent_marker(ax, vent_point):
    """Plot the Taranaki vent marker."""
    ax.scatter([vent_point.x], [vent_point.y], **VENT_MARKER_STYLE)


def add_place_labels(ax, place_gdf):
    """Add town labels to a map panel."""
    for _, row in place_gdf.iterrows():
        ax.annotate(
            row["name"],
            xy=(row.geometry.x, row.geometry.y),
            xytext=(4, 4),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
            ha="left",
            va="bottom",
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
            zorder=7,
        )


def style_map_axis(ax):
    """Hide ticks while retaining a thin black frame and equal map aspect."""
    ax.set_frame_on(True)
    ax.tick_params(
        left=False,
        bottom=False,
        labelleft=False,
        labelbottom=False,
    )
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(1.0)
    ax.set_aspect("equal")


def style_legend_axis(ax):
    """Hide ticks and frame for legend-only panels."""
    ax.set_axis_off()


def add_panel_title(ax, title):
    """Add a consistent title above a map panel."""
    ax.set_title(
        title,
        fontsize=PANEL_TITLE_FONTSIZE,
        fontweight="bold",
    )


def contour_linewidths():
    """Return progressively thicker line widths for configured ash levels."""
    minimum, maximum = ASH_CONTOUR_LINEWIDTH_RANGE
    return np.linspace(minimum, maximum, len(ASH_CONTOUR_LEVELS_MM))


def plot_ashfall_contours(ax, raster_path):
    """Draw the configured cumulative ashfall contours from one cached raster."""
    with rasterio.open(raster_path) as source:
        ashfall = source.read(1, masked=False).astype(float)
        transform = source.transform

    ashfall[~np.isfinite(ashfall)] = 0
    if not np.any(ashfall > 0):
        print(f"No positive ashfall values to plot in {raster_path.name}")
        return

    maximum_ashfall = float(ashfall.max())
    configured_widths = contour_linewidths()
    valid_contour_styles = [
        (level, linewidth, ASH_CONTOUR_COLOURS[level])
        for level, linewidth in zip(ASH_CONTOUR_LEVELS_MM, configured_widths)
        if level <= maximum_ashfall
    ]
    if not valid_contour_styles:
        return

    levels, linewidths, colours = zip(*valid_contour_styles)
    x_coordinates = transform.c + (np.arange(ashfall.shape[1]) + 0.5) * transform.a
    y_coordinates = transform.f + (np.arange(ashfall.shape[0]) + 0.5) * transform.e
    ax.contour(
        x_coordinates,
        y_coordinates,
        ashfall,
        levels=levels,
        colors=colours,
        linewidths=linewidths,
        zorder=5,
    )


def add_horizontal_colourbar(
    fig,
    legend_ax,
    colour_limits,
    cmap_name,
    label,
    ticks=None,
):
    """Add a horizontal colourbar to a dedicated legend axis."""
    style_legend_axis(legend_ax)
    field_min, field_max = colour_limits
    norm = mpl.colors.Normalize(vmin=field_min, vmax=field_max)
    scalar_mappable = mpl.cm.ScalarMappable(
        norm=norm,
        cmap=plt.get_cmap(cmap_name),
    )
    scalar_mappable.set_array([])

    colorbar_ax = legend_ax.inset_axes([0.08, 0.48, 0.84, 0.20])
    colorbar = fig.colorbar(
        scalar_mappable,
        cax=colorbar_ax,
        orientation="horizontal",
    )
    if ticks is not None:
        colorbar.set_ticks(ticks)
    colorbar.set_label(label, fontsize=COLORBAR_LABEL_FONTSIZE)
    colorbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)


def add_ashfall_legend(legend_ax):
    """Add line samples for every configured ashfall contour level and the vent."""
    style_legend_axis(legend_ax)
    handles = [
        Line2D(
            [0],
            [0],
            color=ASH_CONTOUR_COLOURS[level],
            linewidth=linewidth,
            label=f"{level:g} mm",
        )
        for level, linewidth in zip(
            ASH_CONTOUR_LEVELS_MM,
            contour_linewidths(),
        )
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker=VENT_MARKER_STYLE["marker"],
            color="black",
            markerfacecolor=VENT_MARKER_STYLE["facecolor"],
            markeredgecolor=VENT_MARKER_STYLE["edgecolor"],
            markersize=12,
            linestyle="None",
            label="Vent",
        )
    )
    legend_ax.legend(
        handles=handles,
        title="Cumulative ashfall thickness",
        loc="center",
        ncol=4,
        frameon=True,
        fontsize=ASH_LEGEND_FONTSIZE,
        title_fontsize=COLORBAR_LABEL_FONTSIZE,
    )


def plot_scenario_figure(
    sa2_gdf,
    scenario_paths,
    extent,
    map_crs,
    output_file,
):
    """Plot nine cumulative ashfall scenarios over SA2 population density."""
    fig = plt.figure(figsize=SCENARIO_FIGURE_SIZE, constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=4,
        ncols=3,
        height_ratios=[1, 1, 1, 0.20],
    )
    map_axes = [
        fig.add_subplot(grid[row, column])
        for row in range(3)
        for column in range(3)
    ]
    density_legend_ax = fig.add_subplot(grid[3, 0])
    ashfall_legend_ax = fig.add_subplot(grid[3, 1:])
    density_min, density_max = POPULATION_DENSITY_COLOR_LIMITS
    vent_point = get_vent_point(map_crs)
    place_gdf = make_place_label_geodataframe(map_crs)

    for ax, scenario in zip(map_axes, SCENARIOS):
        set_map_extent(ax, extent)
        sa2_gdf.plot(
            column=POPULATION_DENSITY_FIELD,
            ax=ax,
            cmap=POPULATION_DENSITY_CMAP,
            vmin=density_min,
            vmax=density_max,
            alpha=0.75,
            linewidth=SA2_EDGE_LINEWIDTH,
            edgecolor=SA2_EDGE_COLOR,
            legend=False,
            missing_kwds={"color": "lightgrey"},
            zorder=2,
        )
        plot_ashfall_contours(ax, scenario_paths[scenario])
        plot_vent_marker(ax, vent_point)
        if scenario == "S1":
            add_place_labels(ax, place_gdf)
        add_panel_title(ax, scenario)
        style_map_axis(ax)

    add_horizontal_colourbar(
        fig=fig,
        legend_ax=density_legend_ax,
        colour_limits=POPULATION_DENSITY_COLOR_LIMITS,
        cmap_name=POPULATION_DENSITY_CMAP,
        label="Population density (people per km²)",
    )
    add_ashfall_legend(ashfall_legend_ax)

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
        wspace=0.03,
        hspace=0.12,
    )
    fig.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches="tight")
    print(f"Saved scenario figure: {output_file}")
    return fig


def plot_sa2_attribute_panel(
    ax,
    sa2_gdf,
    field,
    title,
    cmap_name,
    colour_limits,
    extent,
    map_crs,
):
    """Plot one SA2 attribute panel using the shared map styling."""
    field_min, field_max = colour_limits
    set_map_extent(ax, extent)
    sa2_gdf.plot(
        column=field,
        ax=ax,
        cmap=cmap_name,
        vmin=field_min,
        vmax=field_max,
        alpha=0.8,
        linewidth=SA2_EDGE_LINEWIDTH,
        edgecolor=SA2_EDGE_COLOR,
        legend=False,
        missing_kwds={"color": "lightgrey"},
        zorder=2,
    )
    add_panel_title(ax, title)
    style_map_axis(ax)


def get_ethnicity_percent_color_limits(gdf):
    """Set each upper bound to the nearest 5% below its observed maximum."""
    colour_limits = {}
    for field in ETHNICITY_PERCENT_FIELDS:
        observed_max = gdf[field].max(skipna=True)
        if not np.isfinite(observed_max):
            raise ValueError(f"No valid percentage values found for '{field}'")

        rounded_max = (
            np.floor(observed_max / ETHNICITY_LIMIT_INTERVAL)
            * ETHNICITY_LIMIT_INTERVAL
        )
        rounded_max = max(ETHNICITY_LIMIT_INTERVAL, rounded_max)
        colour_limits[field] = (0, rounded_max)

    return colour_limits


def add_ethnicity_percent_colorbars(fig, legend_ax, colour_limits):
    """Add one percentage colourbar per ethnicity in the empty subplot."""
    style_legend_axis(legend_ax)
    legend_ax.text(
        0.5,
        0.96,
        "Percent of population (%)",
        transform=legend_ax.transAxes,
        ha="center",
        va="top",
        fontsize=COLORBAR_LABEL_FONTSIZE,
        fontweight="bold",
    )

    row_positions = [0.79, 0.62, 0.45, 0.28, 0.11]
    for (field, _title), row_y in zip(
        ETHNICITY_PERCENT_FIELDS.items(),
        row_positions,
    ):
        field_min, field_max = colour_limits[field]
        cmap = plt.get_cmap(ETHNICITY_PERCENT_CMAPS[field])
        norm = mpl.colors.Normalize(vmin=field_min, vmax=field_max)
        scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])

        legend_ax.text(
            0.04,
            row_y + 0.025,
            ETHNICITY_LEGEND_LABELS[field],
            transform=legend_ax.transAxes,
            ha="left",
            va="center",
            fontsize=LEGEND_CATEGORY_FONTSIZE,
        )
        colorbar_ax = legend_ax.inset_axes([0.40, row_y, 0.54, 0.045])
        colorbar = fig.colorbar(
            scalar_mappable,
            cax=colorbar_ax,
            orientation="horizontal",
        )
        colorbar.set_ticks([field_min, (field_min + field_max) / 2, field_max])
        colorbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:g}"))
        colorbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)


def plot_ethnicity_figure(sa2_gdf, extent, map_crs, output_file):
    """Plot five ethnicity percentage maps with a shared legend panel."""
    ethnicity_colour_limits = get_ethnicity_percent_color_limits(sa2_gdf)
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=ETHNICITY_FIGURE_SIZE,
        constrained_layout=False,
    )
    axes = axes.ravel()

    for ax, (field, title) in zip(axes[:5], ETHNICITY_PERCENT_FIELDS.items()):
        plot_sa2_attribute_panel(
            ax=ax,
            sa2_gdf=sa2_gdf,
            field=field,
            title=title,
            cmap_name=ETHNICITY_PERCENT_CMAPS[field],
            colour_limits=ethnicity_colour_limits[field],
            extent=extent,
            map_crs=map_crs,
        )

    add_ethnicity_percent_colorbars(
        fig=fig,
        legend_ax=axes[-1],
        colour_limits=ethnicity_colour_limits,
    )

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=0.04,
        wspace=0.05,
        hspace=0.08,
    )
    fig.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches="tight")
    print(f"Saved ethnicity figure: {output_file}")
    return fig


def main():
    """Load prepared inputs and create both Taranaki figures."""
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)
    sa2_gdf, scenario_paths, combined_extent, map_crs = load_prepared_inputs()
    extent = add_extent_buffer(combined_extent, EXTENT_BUFFER_FRACTION)

    plot_scenario_figure(
        sa2_gdf=sa2_gdf,
        scenario_paths=scenario_paths,
        extent=extent,
        map_crs=map_crs,
        output_file=SCENARIO_OUTPUT_FILE,
    )
    plot_ethnicity_figure(
        sa2_gdf=sa2_gdf,
        extent=extent,
        map_crs=map_crs,
        output_file=ETHNICITY_OUTPUT_FILE,
    )
    plt.show()


if __name__ == "__main__":
    main()
