from __future__ import annotations

import json
import math
import re
import textwrap
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
WORKBOOK = ROOT / "reference_cache" / "data" / "vietnam" / "vietnam_stroke_centers_130_en_source.xlsx"
NATURAL_EARTH = (
    ROOT
    / "runs"
    / "network_only_20260622_1645"
    / "east-timor_data"
    / "cache"
    / "boundaries"
    / "ne_10m_admin_0_countries"
    / "ne_10m_admin_0_countries.shp"
)
OUT_DIR = ROOT / "outputs" / "article_components"
FIG_DIR = OUT_DIR / "figures"

STROKE_CENTER_SOURCE = {
    "description": "Vietnam Stroke Association list, 2024 update, collected by Trang Luu as a 01/2025 workbook",
    "url_2024_list": "https://hoidotquyvietnam.com/danh-sach-cac-benh-vien-co-don-vi-hoac-trung-tam-san-sang-cap-cuu-dot-quy-ban-cap-nhat-nam-2024/",
    "later_2025_update_url": "https://hoidotquyvietnam.com/danh-sach-cac-benh-vien-co-don-vi-hoac-trung-tam-san-sang-cap-cuu-dot-quy-ban-cap-nhat-thang-09-2025/",
    "later_2025_count": 170,
}

COMPONENT_FACTS = {
    "diagnostic_source": "Mail exchange with Trang Luu, 2026-06-06 to 2026-06-07",
    "components_found": 6064,
    "component_0_nodes": 25176488,
    "component_1_nodes": 52259,
    "tiny_private_component_nodes": 15,
    "stroke_centers_component_0": 129,
    "stroke_centers_component_1": 1,
    "population_points_component_0": 407939,
    "population_points_component_1": 899,
    "matrix_rows_under_150km": 3483052,
    "source_table_80": {
        "nearest_node": 7780318361,
        "component_id": 0,
        "snap_distance_m": 67.89,
        "retained_rows": 29719,
    },
    "source_table_119": {
        "nearest_node": 6984784178,
        "component_id": 1,
        "snap_distance_m": 27.38,
        "retained_rows": 899,
    },
}


def column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref)
    if letters is None:
        raise ValueError(f"Invalid cell reference: {cell_ref}")
    result = 0
    for char in letters.group(0):
        result = result * 26 + (ord(char) - ord("A") + 1)
    return result - 1


def read_shared_strings(zf: zipfile.ZipFile) -> list[str]:
    try:
        raw = zf.read("xl/sharedStrings.xml")
    except KeyError:
        return []
    root = ET.fromstring(raw)
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    strings: list[str] = []
    for item in root.findall("x:si", namespace):
        text_parts = [node.text or "" for node in item.findall(".//x:t", namespace)]
        strings.append("".join(text_parts))
    return strings


def parse_cell(cell: ET.Element, shared_strings: list[str]) -> object:
    cell_type = cell.attrib.get("t")
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    if cell_type == "inlineStr":
        return "".join(node.text or "" for node in cell.findall(".//x:t", namespace))
    value = cell.find("x:v", namespace)
    if value is None or value.text is None:
        return ""
    raw = value.text
    if cell_type == "s":
        return shared_strings[int(raw)]
    try:
        numeric = float(raw)
    except ValueError:
        return raw
    if numeric.is_integer():
        return int(numeric)
    return numeric


def read_first_sheet(path: Path) -> list[dict[str, object]]:
    with zipfile.ZipFile(path) as zf:
        shared_strings = read_shared_strings(zf)
        root = ET.fromstring(zf.read("xl/worksheets/sheet1.xml"))
    namespace = {"x": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
    rows: list[list[object]] = []
    for row in root.findall(".//x:sheetData/x:row", namespace):
        values: list[object] = []
        for cell in row.findall("x:c", namespace):
            idx = column_index(cell.attrib["r"])
            while len(values) <= idx:
                values.append("")
            values[idx] = parse_cell(cell, shared_strings)
        rows.append(values)

    if not rows:
        return []
    headers = [str(value).strip() for value in rows[0]]
    records: list[dict[str, object]] = []
    for values in rows[1:]:
        record = {
            header: values[idx] if idx < len(values) else ""
            for idx, header in enumerate(headers)
            if header
        }
        if any(value != "" for value in record.values()):
            records.append(record)
    return records


def load_vietnam_geometry() -> gpd.GeoDataFrame:
    world = gpd.read_file(NATURAL_EARTH)
    text_cols = [col for col in ("ADMIN", "NAME", "NAME_EN", "SOVEREIGNT", "ISO_A3") if col in world.columns]
    mask = False
    for col in text_cols:
        mask = mask | world[col].astype(str).str.contains("Vietnam|Viet Nam|VNM", case=False, na=False)
    vietnam = world.loc[mask].copy()
    if vietnam.empty:
        raise RuntimeError("Could not find Vietnam in Natural Earth boundary file")
    return vietnam.to_crs(4326)


def point(record: dict[str, object]) -> tuple[float, float]:
    return float(record["longitude"]), float(record["latitude"])


def make_figure(records: list[dict[str, object]], vietnam: gpd.GeoDataFrame) -> dict[str, str]:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    lons = [float(row["longitude"]) for row in records]
    lats = [float(row["latitude"]) for row in records]
    by_tt = {int(row["TT"]): row for row in records}
    tt80 = by_tt[80]
    tt119 = by_tt[119]

    fig = plt.figure(figsize=(13.2, 7.6), dpi=220)
    grid = GridSpec(1, 2, width_ratios=[1.08, 1.0], wspace=0.34)
    ax_map = fig.add_subplot(grid[0, 0])
    ax_notes = fig.add_subplot(grid[0, 1])
    fig.suptitle("Component-aware snapping in the Vietnam stroke-center network", fontsize=14, y=0.985)

    vietnam.plot(ax=ax_map, facecolor="#f3f0e8", edgecolor="#465661", linewidth=0.8)
    ax_map.scatter(lons, lats, s=14, c="#2f78b7", alpha=0.58, linewidths=0, label="stroke centers")

    lon80, lat80 = point(tt80)
    lon119, lat119 = point(tt119)
    ax_map.scatter([lon80], [lat80], s=72, facecolors="white", edgecolors="#d95f02", linewidths=2.0, zorder=5)
    ax_map.scatter([lon119], [lat119], s=92, marker="*", c="#1b9e77", edgecolors="#0b4f3a", linewidths=0.8, zorder=6)

    ax_map.annotate(
        "TT 80\nartifact avoided",
        xy=(lon80, lat80),
        xytext=(lon80 + 0.35, lat80 + 0.55),
        arrowprops={"arrowstyle": "-", "color": "#d95f02", "lw": 1.2},
        fontsize=8,
        color="#7a3300",
        ha="left",
        va="bottom",
    )
    ax_map.annotate(
        "TT 119\nPhu Quoc",
        xy=(lon119, lat119),
        xytext=(104.9, 10.15),
        arrowprops={"arrowstyle": "-", "color": "#1b9e77", "lw": 1.2},
        fontsize=8,
        color="#0b4f3a",
        ha="left",
        va="center",
    )

    phu_quoc_box = (103.75, 9.78, 0.64, 0.88)
    ax_map.add_patch(
        Rectangle(
            (phu_quoc_box[0], phu_quoc_box[1]),
            phu_quoc_box[2],
            phu_quoc_box[3],
            fill=False,
            edgecolor="#1b9e77",
            linewidth=1.4,
            linestyle="--",
        )
    )
    ax_map.set_xlim(102.0, 110.2)
    ax_map.set_ylim(8.2, 23.7)
    ax_map.set_aspect("equal")
    ax_map.set_title("Raw stroke-center coordinates", loc="left", fontsize=10.5, pad=9)
    ax_map.set_xlabel("Longitude")
    ax_map.set_ylabel("Latitude")
    ax_map.grid(color="#cbd3d8", linewidth=0.4, alpha=0.55)
    ax_map.text(
        102.08,
        8.35,
        "Raw source: 130-row workbook, ID column TT",
        fontsize=7.5,
        color="#465661",
        va="bottom",
    )

    ax_notes.axis("off")
    ax_notes.set_title("Component restriction is a modeling decision", loc="left", fontsize=10.5, pad=9)

    labels = [
        ("Component 0", "mainland road system", COMPONENT_FACTS["component_0_nodes"], "kept", "#2f78b7"),
        ("Component 1", "Phu Quoc island road system", COMPONENT_FACTS["component_1_nodes"], "kept", "#1b9e77"),
        ("Tiny component", "private-terrain artifact", COMPONENT_FACTS["tiny_private_component_nodes"], "excluded", "#d95f02"),
    ]
    x0, y0 = 0.05, 0.82
    max_log = math.log10(COMPONENT_FACTS["component_0_nodes"])
    for idx, (name, desc, nodes, status, color) in enumerate(labels):
        y = y0 - idx * 0.165
        width = 0.62 * math.log10(nodes) / max_log
        ax_notes.add_patch(Rectangle((x0, y - 0.032), width, 0.052, transform=ax_notes.transAxes, color=color, alpha=0.82))
        ax_notes.text(x0, y + 0.045, f"{name}: {nodes:,} nodes", transform=ax_notes.transAxes, fontsize=10, weight="bold", color="#263238")
        ax_notes.text(x0 + 0.01, y - 0.07, f"{desc} - {status}", transform=ax_notes.transAxes, fontsize=8.5, color="#465661")

    bullets = [
        f"{COMPONENT_FACTS['components_found']:,} weak components were detected.",
        f"{COMPONENT_FACTS['stroke_centers_component_0']} centers snapped to component 0; "
        f"{COMPONENT_FACTS['stroke_centers_component_1']} to component 1.",
        f"TT 119 is Vinmec Phu Quoc; component 1 also serves "
        f"{COMPONENT_FACTS['population_points_component_1']:,} island population points.",
        "TT 80 illustrates the risk: unrestricted snapping can pick a tiny disconnected private-road fragment.",
        "With --snap-components 0,1, TT 80 is on component 0 at 67.89 m; TT 119 is on component 1 at 27.38 m.",
        "The choice is substantive: mainland-only uses component 0; Vietnam including Phu Quoc uses 0 and 1.",
    ]
    y = 0.285
    for bullet in bullets:
        wrapped = textwrap.fill(bullet, width=68)
        ax_notes.text(
            0.06,
            y,
            f"- {wrapped}",
            transform=ax_notes.transAxes,
            fontsize=8.2,
            color="#263238",
            va="top",
            linespacing=1.18,
        )
        y -= 0.039 * (wrapped.count("\n") + 1) + 0.025

    png = FIG_DIR / "vietnam_component_snapping_example.png"
    pdf = FIG_DIR / "vietnam_component_snapping_example.pdf"
    fig.savefig(png, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {"png": str(png), "pdf": str(pdf)}


def write_section(figure_paths: dict[str, str]) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = Path(figure_paths["png"]).as_posix()
    pdf_path = Path(figure_paths["pdf"]).as_posix()
    markdown = f"""# Road Components Are Part Of The Accessibility Model

Road-network distances are only meaningful within connected road systems. A
population point and a facility may be geographically close, but if their
nearest road nodes lie in different weak connected components of the extracted
OpenStreetMap graph, the routing engine has no valid road path between them.
This is not a numerical nuisance: it changes which residents can be considered
served by a facility, and therefore changes the maximum-covering optimization
problem.

The Vietnam stroke-center data exposed this issue clearly. Under unrestricted
nearest-node snapping, one source location was assigned to a tiny disconnected
road fragment inside hospital terrain. The snap distance was small, but the
road component had only {COMPONENT_FACTS['tiny_private_component_nodes']} nodes,
so the resulting source had missing or infinite distances to the main road
network. The revised pipeline therefore allows component-aware snapping through
`--snap-components`. Components are ordered by size; `0` is the largest road
system, `1` the second largest, and so on.

The source table used for these experiments is the 130-row workbook collected
by Trang Luu from the Vietnam Stroke Association list available as the 2024
update and used as a 01/2025 working version. Trang also noted a later
09/2025 website update with 170 centers. We therefore treat the 130-row
workbook as the fixed replication input for the present experiments, and the
170-center list as a future data refresh rather than mixing the two versions.

For Vietnam, the diagnostic run found {COMPONENT_FACTS['components_found']:,}
weak connected components. The largest component had
{COMPONENT_FACTS['component_0_nodes']:,} nodes. The second-largest component had
{COMPONENT_FACTS['component_1_nodes']:,} nodes and corresponds to Phu Quoc
island. This second component should not be discarded mechanically, because the
stroke-center table contains TT 119, International Hospital Vinmec Phu Quoc,
and {COMPONENT_FACTS['population_points_component_1']:,} aggregate population
points also snap to that island component. The paper therefore treats component
selection as a modeling choice: `--snap-components 0` gives a mainland-only
analysis, while `--snap-components 0,1` includes Phu Quoc but excludes tiny
private or spurious fragments.

In the diagnostic example, `--snap-components 0,1` assigned
{COMPONENT_FACTS['stroke_centers_component_0']} stroke centers to component 0
and {COMPONENT_FACTS['stroke_centers_component_1']} to component 1. The
previously problematic TT 80 snapped to component 0 at 67.89 m and retained
29,719 source-target rows under the 150 km cap. TT 119 snapped to component 1
at 27.38 m and retained 899 island source-target rows. This illustrates the
general principle used in the fresh experiments: realistic accessibility needs
road or walking network distances, and it also needs explicit treatment of the
connected components on which those distances are computed.

![Vietnam component snapping example]({png_path})

Figure: Component-aware snapping in the Vietnam stroke-center case. The figure
uses the raw 130-row stroke-center workbook for facility coordinates and
annotates the component diagnostics from the June 2026 pipeline exchange.
Pre-snapped parquet files are not used as optimization inputs for the fresh
paper runs.
"""
    (OUT_DIR / "vietnam_component_snapping_section.md").write_text(markdown, encoding="utf-8")

    latex = rf"""\subsection{{Road Components Are Part of the Accessibility Model}}

Road-network distances are only meaningful within connected road systems. A
population point and a facility may be geographically close, but if their
nearest road nodes lie in different weak connected components of the extracted
OpenStreetMap graph, the routing engine has no valid road path between them.
This is not a numerical nuisance: it changes which residents can be considered
served by a facility, and therefore changes the maximum-covering optimization
problem.

The Vietnam stroke-center data exposed this issue clearly. Under unrestricted
nearest-node snapping, one source location was assigned to a tiny disconnected
road fragment inside hospital terrain. The snap distance was small, but the
road component had only {COMPONENT_FACTS['tiny_private_component_nodes']} nodes,
so the resulting source had missing or infinite distances to the main road
network. The revised pipeline therefore allows component-aware snapping through
\texttt{{--snap-components}}. Components are ordered by size: 0 is the largest
road system, 1 the second largest, and so on.

The source table used for these experiments is the 130-row workbook collected
by Trang Luu from the Vietnam Stroke Association list available as the 2024
update and used as a 01/2025 working version. Trang also noted a later
09/2025 website update with 170 centers. We therefore treat the 130-row
workbook as the fixed replication input for the present experiments, and the
170-center list as a future data refresh rather than mixing the two versions.

For Vietnam, the diagnostic run found {COMPONENT_FACTS['components_found']:,}
weak connected components. The largest component had
{COMPONENT_FACTS['component_0_nodes']:,} nodes. The second-largest component had
{COMPONENT_FACTS['component_1_nodes']:,} nodes and corresponds to Phu Quoc
island. This second component should not be discarded mechanically, because the
stroke-center table contains TT 119, International Hospital Vinmec Phu Quoc,
and {COMPONENT_FACTS['population_points_component_1']:,} aggregate population
points also snap to that island component. The paper therefore treats component
selection as a modeling choice: \texttt{{--snap-components 0}} gives a
mainland-only analysis, while \texttt{{--snap-components 0,1}} includes Phu
Quoc but excludes tiny private or spurious fragments.

\begin{{figure}}[tbp]
    \centering
    \includegraphics[width=\linewidth]{{{pdf_path}}}
    \caption{{Component-aware snapping in the Vietnam stroke-center case. The
    figure uses the raw 130-row stroke-center workbook for facility coordinates
    and annotates the component diagnostics from the June 2026 pipeline
    exchange. Pre-snapped parquet files are not used as optimization inputs for
    the fresh paper runs.}}
    \label{{fig:vietnam-component-snapping}}
\end{{figure}}
"""
    (OUT_DIR / "vietnam_component_snapping_section.tex").write_text(latex, encoding="utf-8")


def main() -> None:
    records = read_first_sheet(WORKBOOK)
    vietnam = load_vietnam_geometry()
    figure_paths = make_figure(records, vietnam)
    write_section(figure_paths)
    payload = {
        "workbook": str(WORKBOOK),
        "records": len(records),
        "figure_paths": figure_paths,
        "component_facts": COMPONENT_FACTS,
        "stroke_center_source": STROKE_CENTER_SOURCE,
    }
    (OUT_DIR / "vietnam_component_snapping_manifest.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
