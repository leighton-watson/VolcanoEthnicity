"""
Plot Auckland tephra scenarios over SA2 population density.

This is a project-rooted copy of DATA/Auckland/scenario_tephra_visualization_and_summary.py
with updated paths and a new 3 by 3 map layout:

    panels: scenarios A-H, followed by legends

Tephra thickness is drawn as threshold boundaries over SA2 population density maps.
Run this script from Spyder or from the project root.
"""

# %%
from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import numpy as np
import pandas as pd
from shapely import wkt


# %%
# ---------------------------
# Paths and settings
# ---------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUCKLAND_DIR = PROJECT_ROOT / "DATA" / "Auckland"
POPULATION_DATA_FILE = (
    PROJECT_ROOT
    / "DATA"
    / "PopulationData"
    / "estimated-resident-population-at-30-june-2018-by-statistical.csv"
)
OUTPUT_FOLDER = PROJECT_ROOT / "MakeFigures" / "OUTPUT"
OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

SCENARIO_LETTERS = list("ABCDEFGH")
SCENARIO_PATHS = {
    letter: AUCKLAND_DIR / f"Scenario_{letter}_Tephra.gpkg"
    for letter in SCENARIO_LETTERS
}
SCENARIO_NAMES = {
    "A": "Auckland Airport",
    "B": "Ōtāhuhu",
    "C": "Mangere Bridge",
    "D": "Mt. Eden Suburb",
    "E": "Waitematā Port",
    "F": "Birkenhead",
    "G": "Rangitoto Channel",
    "H": "Rangitoto Island",
}

THICKNESS_COLUMN = "thickness"
MAP_CRS = "EPSG:3857"

TOTAL_POPULATION_FIELD = "erp18"
POPULATION_DENSITY_FIELD = "population_density_km2"

# Use None to use the observed min/max inside the plotted area.
POPULATION_DENSITY_COLOR_LIMITS = (0, 5000)

TEPHRA_BOUNDARY_LEVELS = [1, 10, 25]
TRACE_ASH_THRESHOLD = 0
EXTENT_BUFFER_FRACTION = 0.00
EXTENT_INSET_FRACTION = 0.05
TRACE_LINEWIDTH = 3.0
BOUNDARY_LINEWIDTH = 3.0
BOUNDARY_CLOSE_BUFFER_METRES = 300
POPULATION_DENSITY_CMAP = "magma"
SA2_EDGE_COLOR = "black"
SA2_EDGE_LINEWIDTH = 0.2
VENT_MARKER_STYLE = {
    "marker": "*",
    "s": 250,
    "facecolor": "cyan",
    "edgecolor": "black",
    "linewidth": 0.8,
    "zorder": 6,
}
OUTPUT_DPI = 300
PANEL_TITLE_FONTSIZE = 16
COLORBAR_LABEL_FONTSIZE = 14
COLORBAR_TICK_FONTSIZE = 12
LEGEND_CATEGORY_FONTSIZE = 14
ASH_LEGEND_FONTSIZE = 12
TEPHRA_BOUNDARY_STYLES = {
    1: {"color": "yellow", "label": "1 mm"},
    10: {"color": "orange", "label": "10 mm"},
    25: {"color": "red", "label": "25 mm"},
}


# %%
def load_scenarios(scenario_paths):
    """Load scenario GPKGs, reproject to MAP_CRS, and calculate shared extent."""
    scenarios = {}
    all_bounds = []
    thickness_values = []

    for scenario_label, path in scenario_paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Scenario file not found: {path}")

        print("Reading:", path)
        gdf = gpd.read_file(path)

        if THICKNESS_COLUMN not in gdf.columns:
            raise ValueError(f"{path.name} does not contain a '{THICKNESS_COLUMN}' column.")

        gdf = gdf.to_crs(MAP_CRS)
        scenarios[scenario_label] = gdf
        all_bounds.append(gdf.total_bounds)
        thickness_values.append(gdf[THICKNESS_COLUMN])

    all_bounds = np.array(all_bounds)
    xmin = all_bounds[:, 0].min()
    ymin = all_bounds[:, 1].min()
    xmax = all_bounds[:, 2].max()
    ymax = all_bounds[:, 3].max()

    x_buffer = (xmax - xmin) * EXTENT_BUFFER_FRACTION
    y_buffer = (ymax - ymin) * EXTENT_BUFFER_FRACTION
    extent = (
        xmin - x_buffer,
        ymin - y_buffer,
        xmax + x_buffer,
        ymax + y_buffer,
    )

    thickness = pd.concat(thickness_values, ignore_index=True).dropna()
    thickness_min = thickness.min()
    thickness_max = thickness.max()

    return scenarios, extent, thickness_min, thickness_max


def load_sa2_population_data(csv_path):
    """Load the Stats NZ SA2 CSV and convert WKT geometries to a GeoDataFrame."""
    use_columns = [
        "WKT",
        "SA22020_V1_00",
        "SA22020_V1_00_NAME",
        TOTAL_POPULATION_FIELD,
    ]

    data = pd.read_csv(csv_path, usecols=use_columns)

    data[TOTAL_POPULATION_FIELD] = pd.to_numeric(
        data[TOTAL_POPULATION_FIELD],
        errors="coerce",
    )
    data.loc[data[TOTAL_POPULATION_FIELD] == -999, TOTAL_POPULATION_FIELD] = pd.NA

    geometry = data["WKT"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(data.drop(columns="WKT"), geometry=geometry, crs="EPSG:2193")
    area_km2 = gdf.geometry.area / 1_000_000
    gdf[POPULATION_DENSITY_FIELD] = gdf[TOTAL_POPULATION_FIELD] / area_km2
    return gdf.to_crs(MAP_CRS)


def crop_to_extent(gdf, extent):
    """Keep geometries that intersect the plotting extent."""
    xmin, ymin, xmax, ymax = extent
    return gdf.cx[xmin:xmax, ymin:ymax].copy()


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


def get_population_density_colour_limits(gdf):
    """Return colour limits for population density."""
    observed_min = gdf[POPULATION_DENSITY_FIELD].min(skipna=True)
    observed_max = gdf[POPULATION_DENSITY_FIELD].max(skipna=True)
    manual_min, manual_max = POPULATION_DENSITY_COLOR_LIMITS
    field_min = observed_min if manual_min is None else manual_min
    field_max = observed_max if manual_max is None else manual_max
    return field_min, field_max


def report_population_density_summary(gdf):
    """Print the mean and maximum population density in the plotted SA2 data."""
    valid = gdf.loc[gdf[POPULATION_DENSITY_FIELD].notna()]
    if valid.empty:
        print("Population density summary unavailable: no valid values")
        return

    mean_density = valid[POPULATION_DENSITY_FIELD].mean()
    max_density = valid[POPULATION_DENSITY_FIELD].max()
    max_sa2_names = valid.loc[
        valid[POPULATION_DENSITY_FIELD] == max_density,
        "SA22020_V1_00_NAME",
    ].astype(str)
    print(
        "Maximum population density in plotted area: "
        f"{max_density:,.1f} people per km² "
        f"({', '.join(max_sa2_names)})"
    )
    print(
        "Average population density in plotted area: "
        f"{mean_density:,.1f} people per km²"
    )


def get_tephra_boundary_levels(thickness_min, thickness_max):
    """Return configured tephra threshold levels that fall within the data range."""
    return [
        level
        for level in TEPHRA_BOUNDARY_LEVELS
        if thickness_min <= level <= thickness_max
    ]


def plot_threshold_boundary(ax, gdf, color, linestyle, linewidth):
    """Plot a cleaned outer boundary around threshold polygons."""
    if gdf.empty:
        return

    # The tephra files are many small cells with slight gaps. Buffering before
    # and after unioning closes those gaps so we draw an envelope, not every cell.
    closed_geometry = (
        gdf.geometry.buffer(BOUNDARY_CLOSE_BUFFER_METRES)
        .union_all()
        .buffer(-BOUNDARY_CLOSE_BUFFER_METRES)
    )
    boundary = closed_geometry.boundary
    gpd.GeoSeries([boundary], crs=gdf.crs).plot(
        ax=ax,
        color=color,
        linewidth=linewidth,
        linestyle=linestyle,
        alpha=0.9,
        zorder=5,
    )


def plot_tephra_boundaries(ax, scenario_gdf, boundary_levels):
    """Plot tephra thickness as polygon threshold boundaries."""
    valid = scenario_gdf.loc[scenario_gdf[THICKNESS_COLUMN].notna()].copy()
    if valid.empty:
        return

    trace = valid.loc[valid[THICKNESS_COLUMN] > TRACE_ASH_THRESHOLD]
    plot_threshold_boundary(
        ax=ax,
        gdf=trace,
        color="black",
        linestyle="-",
        linewidth=TRACE_LINEWIDTH,
    )

    for level in boundary_levels:
        threshold = valid.loc[valid[THICKNESS_COLUMN] >= level]
        style = TEPHRA_BOUNDARY_STYLES[level]
        plot_threshold_boundary(
            ax=ax,
            gdf=threshold,
            color=style["color"],
            linestyle="-",
            linewidth=BOUNDARY_LINEWIDTH,
        )


def plot_vent_marker(ax, scenario_gdf):
    """Mark the highest-thickness cell as the scenario vent location."""
    valid = scenario_gdf.loc[scenario_gdf[THICKNESS_COLUMN].notna()].copy()
    if valid.empty:
        return

    max_thickness_index = valid[THICKNESS_COLUMN].idxmax()
    vent_point = valid.loc[max_thickness_index].geometry.centroid
    ax.scatter(
        [vent_point.x],
        [vent_point.y],
        **VENT_MARKER_STYLE,
    )


def set_map_extent(ax, map_extent):
    """Set map bounds before plotting SA2 data."""
    xmin, ymin, xmax, ymax = map_extent
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)


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


def add_colourbar_legend(fig, legend_ax, colour_limits, cmap_name, label):
    """Add a horizontal colourbar to a dedicated legend axis."""
    style_legend_axis(legend_ax)
    field_min, field_max = colour_limits
    cmap = plt.get_cmap(cmap_name)
    norm = mpl.colors.Normalize(vmin=field_min, vmax=field_max)
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    cbar_ax = legend_ax.inset_axes([0.08, 0.48, 0.84, 0.20])
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(label, fontsize=COLORBAR_LABEL_FONTSIZE)
    cbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)


def add_scenario_legend_panel(fig, legend_ax, density_colour_limits):
    """Add population density and ashfall legends to the empty map panel."""
    style_legend_axis(legend_ax)

    field_min, field_max = density_colour_limits
    cmap = plt.get_cmap(POPULATION_DENSITY_CMAP)
    norm = mpl.colors.Normalize(vmin=field_min, vmax=field_max)
    scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar_mappable.set_array([])

    cbar_ax = legend_ax.inset_axes([0.10, 0.76, 0.80, 0.08])
    cbar = fig.colorbar(scalar_mappable, cax=cbar_ax, orientation="horizontal")
    cbar.set_label(
        "Population density\n(people per km²)",
        fontsize=COLORBAR_LABEL_FONTSIZE,
    )
    cbar.ax.xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)

    contour_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            linestyle="-",
            linewidth=TRACE_LINEWIDTH,
            label="Trace ash > 0 mm",
        ),
    ]
    for level in TEPHRA_BOUNDARY_LEVELS:
        style = TEPHRA_BOUNDARY_STYLES[level]
        contour_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle="-",
                linewidth=BOUNDARY_LINEWIDTH,
                label=style["label"],
            )
        )
    contour_handles.append(
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
        handles=contour_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.12),
        ncol=2,
        frameon=True,
        fontsize=ASH_LEGEND_FONTSIZE,
    )


def plot_scenario_overview(
    scenarios,
    sa2_gdf,
    extent,
    density_colour_limits,
    boundary_levels,
    output_file,
):
    """Plot eight scenarios and legends in a 3 by 3 map grid."""
    fig = plt.figure(figsize=(14, 14), constrained_layout=False)
    grid = fig.add_gridspec(
        nrows=3,
        ncols=3,
    )
    map_axes = [
        fig.add_subplot(grid[row, column])
        for row in range(3)
        for column in range(3)
    ]
    scenario_axes = map_axes[:8]
    legend_ax = map_axes[8]
    density_min, density_max = density_colour_limits

    for ax, scenario_label in zip(scenario_axes, SCENARIO_LETTERS):
        scenario_gdf = scenarios[scenario_label]
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

        plot_tephra_boundaries(ax, scenario_gdf, boundary_levels)
        plot_vent_marker(ax, scenario_gdf)

        ax.set_title(
            f"Scenario {scenario_label}: {SCENARIO_NAMES[scenario_label]}",
            fontsize=PANEL_TITLE_FONTSIZE,
            fontweight="bold",
        )
        style_map_axis(ax)

    add_scenario_legend_panel(fig, legend_ax, density_colour_limits)

    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.98,
        bottom=0.02,
        wspace=0.03,
        hspace=0.10,
    )
    fig.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.show()


# %%
scenarios, extent, thickness_min, thickness_max = load_scenarios(SCENARIO_PATHS)
extent = inset_extent(extent, EXTENT_INSET_FRACTION)
sa2_population_gdf = load_sa2_population_data(POPULATION_DATA_FILE)
sa2_population_gdf = crop_to_extent(sa2_population_gdf, extent)
report_population_density_summary(sa2_population_gdf)
density_colour_limits = get_population_density_colour_limits(sa2_population_gdf)
boundary_levels = get_tephra_boundary_levels(thickness_min, thickness_max)

plot_scenario_overview(
    scenarios=scenarios,
    sa2_gdf=sa2_population_gdf,
    extent=extent,
    density_colour_limits=density_colour_limits,
    boundary_levels=boundary_levels,
    output_file=OUTPUT_FOLDER / "Fig_Auckland_Scenarios.png",
)


# %%
# ---------------------------
# Ethnicity percentage maps using the same scenario extent
# ---------------------------
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

# Use a separate sequential colormap for each ethnicity.
ETHNICITY_PERCENT_CMAPS = {
    "pEurother": "Oranges",
    "pMaori": "Greens",
    "pPacific": "Blues",
    "pAsian": "Purples",
    "pMELAA": "Reds",
}

# Set the lower and upper bounds for each ethnicity colormap here.
ETHNICITY_PERCENT_COLOR_LIMITS = {
    "pEurother": (0, 100),
    "pMaori": (0, 50),
    "pPacific": (0, 80),
    "pAsian": (0, 80),
    "pMELAA": (0, 20),
}
ETHNICITY_PERCENT_OUTPUT_FILE = OUTPUT_FOLDER / "Fig_Auckland_Ethnicity.png"


def load_sa2_ethnicity_percent_data(csv_path):
    """Load SA2 ethnicity percentage columns and convert WKT geometries."""
    use_columns = [
        "WKT",
        "SA22020_V1_00",
        "SA22020_V1_00_NAME",
        *ETHNICITY_PERCENT_FIELDS.keys(),
    ]

    data = pd.read_csv(csv_path, usecols=use_columns)

    for field in ETHNICITY_PERCENT_FIELDS:
        data[field] = pd.to_numeric(data[field], errors="coerce")
        data.loc[data[field] == -999, field] = pd.NA

    geometry = data["WKT"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(data.drop(columns="WKT"), geometry=geometry, crs="EPSG:2193")
    return gdf.to_crs(MAP_CRS)


def add_ethnicity_percent_colorbars(fig, colorbar_ax, colour_limits):
    """Add one percentage colourbar per ethnicity in the empty subplot."""
    style_legend_axis(colorbar_ax)

    colorbar_ax.text(
        0.5,
        0.96,
        "Percent of population (%)",
        transform=colorbar_ax.transAxes,
        ha="center",
        va="top",
        fontsize=COLORBAR_LABEL_FONTSIZE,
        fontweight="bold",
    )

    row_positions = [0.79, 0.62, 0.45, 0.28, 0.11]
    for (field, title), row_y in zip(
        ETHNICITY_PERCENT_FIELDS.items(),
        row_positions,
    ):
        field_min, field_max = colour_limits[field]
        cmap = plt.get_cmap(ETHNICITY_PERCENT_CMAPS[field])
        norm = mpl.colors.Normalize(vmin=field_min, vmax=field_max)
        scalar_mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar_mappable.set_array([])

        field_label = ETHNICITY_LEGEND_LABELS[field]
        colorbar_ax.text(
            0.04,
            row_y + 0.025,
            field_label,
            transform=colorbar_ax.transAxes,
            ha="left",
            va="center",
            fontsize=LEGEND_CATEGORY_FONTSIZE,
        )

        inset_ax = colorbar_ax.inset_axes([0.40, row_y, 0.54, 0.045])
        cbar = fig.colorbar(
            scalar_mappable,
            cax=inset_ax,
            orientation="horizontal",
        )
        cbar.set_ticks([field_min, (field_min + field_max) / 2, field_max])
        cbar.ax.tick_params(labelsize=COLORBAR_TICK_FONTSIZE)


def plot_ethnicity_percent_maps(gdf, map_extent, output_file):
    """Plot SA2 ethnicity percentages using the same extent as the scenario figure."""
    colour_limits = ETHNICITY_PERCENT_COLOR_LIMITS
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(14, 9),
        constrained_layout=False,
    )
    axes = axes.ravel()

    for ax, (field, title) in zip(axes, ETHNICITY_PERCENT_FIELDS.items()):
        field_min, field_max = colour_limits[field]
        set_map_extent(ax, map_extent)

        gdf.plot(
            column=field,
            ax=ax,
            cmap=ETHNICITY_PERCENT_CMAPS[field],
            vmin=field_min,
            vmax=field_max,
            alpha=0.75,
            linewidth=SA2_EDGE_LINEWIDTH,
            edgecolor=SA2_EDGE_COLOR,
            legend=False,
            missing_kwds={"color": "lightgrey"},
            zorder=2,
        )

        ax.set_title(title, fontsize=PANEL_TITLE_FONTSIZE, fontweight="bold")
        style_map_axis(ax)

    add_ethnicity_percent_colorbars(fig, axes[-1], colour_limits)


    fig.subplots_adjust(
        left=0.02,
        right=0.98,
        top=0.92,
        bottom=0.04,
        wspace=0.05,
        hspace=0.08,
    )
    fig.savefig(output_file, dpi=OUTPUT_DPI, bbox_inches="tight")
    plt.show()


sa2_ethnicity_percent_gdf = load_sa2_ethnicity_percent_data(POPULATION_DATA_FILE)
sa2_ethnicity_percent_gdf = crop_to_extent(sa2_ethnicity_percent_gdf, extent)
plot_ethnicity_percent_maps(
    gdf=sa2_ethnicity_percent_gdf,
    map_extent=extent,
    output_file=ETHNICITY_PERCENT_OUTPUT_FILE,
)
