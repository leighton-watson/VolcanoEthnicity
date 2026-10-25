"""
Map 2018 estimated resident population ethnicity percentages by SA2.

The source file is a Stats NZ CSV with WKT geometries in NZTM2000
(EPSG:2193). Run this script from Spyder or from the project root.
"""

from pathlib import Path

import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from shapely.geometry import Polygon
from shapely import wkt


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_FILE = (
    PROJECT_DIR
    / "DATA"
    / "PopulationData"
    / "estimated-resident-population-at-30-june-2018-by-statistical.csv"
)
TERRITORIAL_AUTHORITY_FILE = (
    PROJECT_DIR
    / "DATA"
    / "TerritorialAuthority"
    / "territorial-authority-2025.shp"
)
OUTPUT_DIR = PROJECT_DIR / "MakeFigures" / "OUTPUT"
OUTPUT_FILE = OUTPUT_DIR / "Fig_national_ethnicity.png"

SA2_CODE_FIELD = "SA22020_V1_00"
SA2_NAME_FIELD = "SA22020_V1_00_NAME"
TOTAL_POPULATION_FIELD = "erp18"
SA2_AREA_FIELD = "sa2_area_km2"
TA_CODE_FIELD = "territorial_authority_code"
TA_NAME_FIELD = "territorial_authority_name"
TA_POPULATION_FIELD = "ta_total_population"
TA_AREA_FIELD = "ta_sa2_area_km2"
TA_POPULATION_DENSITY_FIELD = "ta_population_density_km2"

ETHNICITY_FIELDS = {
    "pEurother": "(a) European or Other",
    "pMaori": "(b) Māori",
    "pPacific": "(c) Pacific",
    "pAsian": "(d) Asian",
    "pMELAA": "(e) MELAA",
}

ETHNICITY_CMAPS = {
    "pEurother": "Oranges",
    "pMaori": "Greens",
    "pPacific": "Blues",
    "pAsian": "Purples",
    "pMELAA": "Reds",
}

# Set manual colour limits here. Use None for either value to use the observed
# minimum or maximum for that ethnicity.
ETHNICITY_COLOR_LIMITS = {
    "pEurother": (0, 100),
    "pMaori": (0, 100),
    "pPacific": (0, 10),
    "pAsian": (0, 10),
    "pMELAA": (0, 5),
}

TA_POPULATION_DENSITY_COLOR_LIMITS = (0, 250)

EXCLUDED_SA2_NAMES = {"Area Outside Region"}

TA_CODE_FIELD_CANDIDATES = [
    "TA2025_V1_00",
    "TA2025_V1_",
    "TA2025_V_1",
    "TA2025_00",
    "TA2025_V1",
    "TA2025",
]
TA_NAME_FIELD_CANDIDATES = [
    "TA2025_V1_00_NAME",
    "TA2025_V1__NAME",
    "TA2025_V_2",
    "TA2025_V_3",
    "TA2025_V1_",
    "TA2025_NAM",
    "TA2025NAME",
    "TA_NAME",
    "NAME",
]

# North Island mask vertices in longitude/latitude (EPSG:4326). These are
# transformed to NZTM2000 (EPSG:2193) before filtering the SA2 geometries.
NORTH_ISLAND_MASK_LONLAT = [
    (170.9710121, -33.84105666),
    (172.4137559, -39.67208486),
    (174.2112728, -40.45044899),
    (174.4477882, -41.37981803),
    (179.2490504, -42.08581020),
    (180.7627489, -35.47514935),
    (175.9851382, -32.42256562),
]

VOLCANO_LOCATIONS_LONLAT = {
    "Taranaki": (174.0632467, -39.29790130),
    "Ruapehu": (175.5643953, -39.28055990),
    "Whakaari": (177.1835350, -37.52125927),
    "Auckland Volcanic Field": (174.7642771, -36.87733413),
    "Tongariro/Ngauruhoe": (175.6467227, -39.14193241),
}

VOLCANO_LABEL_OFFSETS = {
    "Taranaki": (-110000, 0),
    "Ruapehu": (45000, -35000),
    "Whakaari": (45000, 35000),
    "Auckland Volcanic Field": (45000, 50000),
    "Tongariro/Ngauruhoe": (55000, 40000),
}

def load_sa2_ethnicity_data(csv_path):
    """Load the Stats NZ CSV and convert WKT geometries to a GeoDataFrame."""
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
        data.loc[data[field] == -999, field] = pd.NA

    geometry = data["WKT"].apply(wkt.loads)
    gdf = gpd.GeoDataFrame(data.drop(columns="WKT"), geometry=geometry, crs="EPSG:2193")
    gdf = gdf.loc[~gdf[SA2_NAME_FIELD].isin(EXCLUDED_SA2_NAMES)].copy()
    gdf[SA2_AREA_FIELD] = gdf.geometry.area / 1_000_000
    return gdf


def find_field(columns, candidates, field_description):
    """Return the first matching field from a list of candidate names."""
    for candidate in candidates:
        if candidate in columns:
            return candidate

    lower_to_field = {column.lower(): column for column in columns}
    for candidate in candidates:
        matching_field = lower_to_field.get(candidate.lower())
        if matching_field is not None:
            return matching_field

    raise ValueError(
        f"Could not find {field_description}. Available fields: {list(columns)}"
    )


def infer_ta_code_field(gdf):
    """Infer the territorial authority code field if shapefile names are truncated."""
    for column in gdf.columns:
        if column == "geometry":
            continue

        values = gdf[column].dropna().astype(str).str.strip()
        if values.empty:
            continue

        if values.str.fullmatch(r"\d{3}").all():
            return column

    raise ValueError(
        "Could not infer territorial authority code field. "
        f"Available fields: {list(gdf.columns)}"
    )


def infer_ta_name_field(gdf, code_field):
    """Infer the territorial authority name field if shapefile names are truncated."""
    text_columns = [
        column
        for column in gdf.columns
        if column not in {code_field, "geometry"}
        and pd.api.types.is_object_dtype(gdf[column])
    ]
    if not text_columns:
        raise ValueError(
            "Could not infer territorial authority name field. "
            f"Available fields: {list(gdf.columns)}"
        )

    return max(
        text_columns,
        key=lambda column: gdf[column].dropna().astype(str).str.len().mean(),
    )


def load_territorial_authorities(shapefile_path, target_crs):
    """Load territorial authority boundaries with standardised code/name fields."""
    gdf = gpd.read_file(shapefile_path)
    if gdf.crs is None:
        gdf = gdf.set_crs(target_crs)
    else:
        gdf = gdf.to_crs(target_crs)

    try:
        source_code_field = find_field(
            gdf.columns,
            TA_CODE_FIELD_CANDIDATES,
            "territorial authority code field",
        )
    except ValueError:
        source_code_field = infer_ta_code_field(gdf)

    try:
        name_field_candidates = [
            candidate
            for candidate in TA_NAME_FIELD_CANDIDATES
            if candidate != source_code_field
        ]
        source_name_field = find_field(
            gdf.columns,
            name_field_candidates,
            "territorial authority name field",
        )
        if source_name_field == source_code_field:
            raise ValueError
    except ValueError:
        source_name_field = infer_ta_name_field(gdf, source_code_field)

    gdf = gdf.rename(
        columns={
            source_code_field: TA_CODE_FIELD,
            source_name_field: TA_NAME_FIELD,
        }
    )
    return gdf[[TA_CODE_FIELD, TA_NAME_FIELD, "geometry"]].copy()


def add_territorial_authority_density(sa2_gdf, ta_gdf):
    """Assign SA2s to TAs, aggregate density by TA, and keep SA2 geometries."""
    gdf = sa2_gdf.copy()
    gdf[TA_CODE_FIELD] = pd.NA
    gdf[TA_NAME_FIELD] = pd.NA
    representative_points = gdf.geometry.representative_point()

    for _, ta_row in ta_gdf.iterrows():
        within_ta = representative_points.within(ta_row.geometry)
        gdf.loc[within_ta, TA_CODE_FIELD] = ta_row[TA_CODE_FIELD]
        gdf.loc[within_ta, TA_NAME_FIELD] = ta_row[TA_NAME_FIELD]

    missing = gdf.loc[gdf[TA_CODE_FIELD].isna(), SA2_NAME_FIELD]
    if not missing.empty:
        missing_names = ", ".join(missing.astype(str).head(10))
        raise ValueError(
            "Some SA2 areas were not assigned to a territorial authority: "
            f"{missing_names}"
        )

    ta_summary = (
        gdf.groupby([TA_CODE_FIELD, TA_NAME_FIELD], dropna=False)
        .agg(
            {
                TOTAL_POPULATION_FIELD: "sum",
                SA2_AREA_FIELD: "sum",
            }
        )
        .rename(
            columns={
                TOTAL_POPULATION_FIELD: TA_POPULATION_FIELD,
                SA2_AREA_FIELD: TA_AREA_FIELD,
            }
        )
        .reset_index()
    )
    ta_summary[TA_POPULATION_DENSITY_FIELD] = (
        ta_summary[TA_POPULATION_FIELD] / ta_summary[TA_AREA_FIELD]
    )

    return gdf.merge(
        ta_summary[
            [
                TA_CODE_FIELD,
                TA_NAME_FIELD,
                TA_POPULATION_FIELD,
                TA_AREA_FIELD,
                TA_POPULATION_DENSITY_FIELD,
            ]
        ],
        on=[TA_CODE_FIELD, TA_NAME_FIELD],
        how="left",
    )


def restrict_to_north_island(gdf):
    """Keep SA2 areas whose representative point falls within the North Island mask."""
    north_island_mask = (
        gpd.GeoSeries([Polygon(NORTH_ISLAND_MASK_LONLAT)], crs="EPSG:4326")
        .to_crs(gdf.crs)
        .iloc[0]
    )
    representative_points = gdf.geometry.representative_point()
    north_island_gdf = gdf.loc[representative_points.within(north_island_mask)].copy()
    return north_island_gdf


def make_volcano_geodataframe(target_crs):
    """Create volcano point locations and transform them to the map CRS."""
    volcano_data = pd.DataFrame(
        [
            {"name": name, "longitude": lon, "latitude": lat}
            for name, (lon, lat) in VOLCANO_LOCATIONS_LONLAT.items()
        ]
    )
    volcano_gdf = gpd.GeoDataFrame(
        volcano_data,
        geometry=gpd.points_from_xy(volcano_data["longitude"], volcano_data["latitude"]),
        crs="EPSG:4326",
    )
    return volcano_gdf.to_crs(target_crs)


def dissolve_sa2_geometries_to_tas(gdf):
    """Dissolve SA2 geometries by TA while retaining TA-level density values."""
    return gdf[
        [
            TA_CODE_FIELD,
            TA_NAME_FIELD,
            TA_POPULATION_FIELD,
            TA_AREA_FIELD,
            TA_POPULATION_DENSITY_FIELD,
            "geometry",
        ]
    ].dissolve(
        by=[TA_CODE_FIELD, TA_NAME_FIELD],
        as_index=False,
        aggfunc="first",
    )


def plot_ethnicity_maps(gdf):
    """Create a 2 by 3 subplot figure for ethnicity and population density."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=False)
    axes = axes.ravel()
    xmin, ymin, xmax, ymax = gdf.total_bounds

    for ax, (field, title) in zip(axes, ETHNICITY_FIELDS.items()):
        observed_min = gdf[field].min(skipna=True)
        observed_max = gdf[field].max(skipna=True)
        manual_min, manual_max = ETHNICITY_COLOR_LIMITS[field]
        field_min = observed_min if manual_min is None else manual_min
        field_max = observed_max if manual_max is None else manual_max

        gdf.plot(
            column=field,
            ax=ax,
            cmap=ETHNICITY_CMAPS[field],
            vmin=field_min,
            vmax=field_max,
            linewidth=0,
            edgecolor="none",
            missing_kwds={"color": "lightgrey", "label": "No data"},
            legend=True,
            legend_kwds={
                "label": f"Percent of population (%)",
                "shrink": 0.75,
                "pad": 0.01,
            },
        )
        ax.set_title(title, fontsize=12, loc='left', fontweight="bold")
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_axis_off()
        ax.set_aspect("equal")

    ta_density_gdf = dissolve_sa2_geometries_to_tas(gdf)
    observed_min = ta_density_gdf[TA_POPULATION_DENSITY_FIELD].min(skipna=True)
    observed_max = ta_density_gdf[TA_POPULATION_DENSITY_FIELD].max(skipna=True)
    manual_min, manual_max = TA_POPULATION_DENSITY_COLOR_LIMITS
    field_min = observed_min if manual_min is None else manual_min
    field_max = observed_max if manual_max is None else manual_max

    ta_density_gdf.plot(
        column=TA_POPULATION_DENSITY_FIELD,
        ax=axes[-1],
        cmap="Greys",
        vmin=field_min,
        vmax=field_max,
        linewidth=0.2,
        edgecolor="black",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        legend=True,
        legend_kwds={
            "label": "Population density (people per km²)",
            "shrink": 0.75,
            "pad": 0.01,
        },
    )
    volcano_gdf = make_volcano_geodataframe(gdf.crs)
    volcano_gdf.plot(
        ax=axes[-1],
        marker="^",
        markersize=70,
        color="yellow",
        edgecolor="black",
        linewidth=0.8,
        zorder=5,
    )
    for _, row in volcano_gdf.iterrows():
        label_dx, label_dy = VOLCANO_LABEL_OFFSETS[row["name"]]
        point_x = row.geometry.x
        point_y = row.geometry.y
        axes[-1].annotate(
            row["name"],
            xy=(point_x, point_y),
            xytext=(point_x + label_dx, point_y + label_dy),
            fontsize=10,
            ha="center",
            va="center",
            # arrowprops={
            #     "arrowstyle": "-",
            #     "color": "black",
            #     "linewidth": 0.7,
            # },
            bbox={
                "boxstyle": "round,pad=0.15",
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.75,
            },
            zorder=6,
        )
    axes[-1].set_title("(f) Population Density", fontsize=12, loc='left', fontweight="bold")
    axes[-1].set_xlim(xmin, xmax)
    axes[-1].set_ylim(ymin, ymax)
    axes[-1].set_axis_off()
    axes[-1].set_aspect("equal")

    fig.subplots_adjust(
        left=0.01,
        right=0.99,
        top=0.88,
        bottom=0.02,
        wspace=-0.6,
        hspace=0.15,
    )

    return fig


def main():
    gdf = load_sa2_ethnicity_data(DATA_FILE)
    gdf = restrict_to_north_island(gdf)
    ta_gdf = load_territorial_authorities(TERRITORIAL_AUTHORITY_FILE, gdf.crs)
    gdf = add_territorial_authority_density(gdf, ta_gdf)
    fig = plot_ethnicity_maps(gdf)

    OUTPUT_DIR.mkdir(exist_ok=True)
    fig.savefig(OUTPUT_FILE, dpi=300, bbox_inches="tight")

    # Keep the figure visible when running interactively in Spyder.
    plt.show()


if __name__ == "__main__":
    main()
