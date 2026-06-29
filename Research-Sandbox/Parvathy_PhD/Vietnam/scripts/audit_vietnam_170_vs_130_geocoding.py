from __future__ import annotations

import csv
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(r"C:\work\codex\sandboxes\Conclude_Parvathy_thesis")
BASE = ROOT / "reference_cache" / "data" / "vietnam"

OLD_PATH = BASE / "vietnam_stroke_centers_130_en_source.xlsx"
INCLUSION_V1 = BASE / "vietnam_stroke_centers_130_vs_170_inclusion_reviewed.csv"
NEW_PATH = BASE / "vietnam_stroke_centers_170_vi_vnsa_2025_09_google_geocoded_compact.csv"

OUT_INCLUSION_CSV = BASE / "vietnam_stroke_centers_130_vs_170_inclusion_reviewed_v2.csv"
OUT_INCLUSION_MD = BASE / "vietnam_stroke_centers_130_vs_170_inclusion_check_reviewed_v2.md"
OUT_AUDIT_CSV = BASE / "vietnam_stroke_centers_130_vs_170_geocode_coordinate_audit_v2.csv"
OUT_AUDIT_MD = BASE / "vietnam_stroke_centers_130_vs_170_geocode_coordinate_audit_v2.md"


CONFIRMED_STATUSES = {
    "confirmed_by_name",
    "confirmed_admin_rename_same_address",
    "confirmed_renamed_same_address",
    "confirmed_renamed_same_coordinate",
    "confirmed_renamed_same_locality",
    "confirmed_admin_rename_same_coordinate",
    "confirmed_admin_rename_needs_coordinate_review",
    "confirmed_same_address",
}


OVERRIDES = {
    7: {
        "review_status": "not_confirmed_in_new_source",
        "new_tt": None,
        "note": (
            "Old row is Hanoi Heart Hospital facility 2 on Vo Chi Cong. "
            "The new 170 list contains Hanoi Heart Hospital facility 1 only, "
            "so the previous fuzzy match was not accepted."
        ),
    },
    33: {
        "review_status": "confirmed_renamed_same_address",
        "new_tt": 9,
        "note": (
            "Old Bac Ninh provincial hospital at Bo Son/Vo Cuong matches new "
            "Bac Ninh no. 2 at Nguyen Quyen/Vo Cuong; coordinate difference is small."
        ),
    },
    51: {
        "review_status": "confirmed_admin_rename_same_coordinate",
        "new_tt": 107,
        "note": (
            "Old Yen Bai provincial hospital matches new Lao Cai no. 1 around "
            "Au Lau/Tien Phong after administrative/source update; coordinates "
            "are within about 0.5 km."
        ),
    },
    52: {
        "review_status": "confirmed_admin_rename_needs_coordinate_review",
        "new_tt": 105,
        "note": (
            "Old Lao Cai provincial hospital is more plausibly new Lao Cai no. 2 "
            "than new Lao Cai no. 1. Google geocode for TT105 is low confidence "
            "and should be checked manually."
        ),
    },
    66: {
        "review_status": "confirmed_renamed_same_coordinate",
        "new_tt": 144,
        "note": (
            "Old Thao Nguyen hospital and new Van Ho regional hospital share "
            "essentially the same coordinate and Thao Nguyen appears in the new address."
        ),
    },
    69: {
        "review_status": "not_confirmed_in_new_source",
        "new_tt": None,
        "note": (
            "Old Bac Quang Binh hospital should not be matched to new Quang Binh "
            "in Tuyen Quang; this was a false fuzzy match."
        ),
    },
    110: {
        "review_status": "confirmed_same_address",
        "new_tt": 2,
        "note": (
            "Old An Giang regional hospital at 917 Ton Duc Thang, Chau Doc "
            "matches new Chau Doc general hospital. The previous fuzzy match to "
            "An Giang provincial hospital was incorrect."
        ),
    },
}


OLD_COORD_NOTES = {
    34: (
        "Old coordinate appears inconsistent with the old/new Hai Duong address; "
        "new Google point is rooftop at the current official address."
    ),
    45: (
        "Old coordinate is in southern Vietnam and is inconsistent with Hoa Binh; "
        "new coordinate is more plausible but only medium confidence."
    ),
    49: (
        "Old coordinate is in southern Vietnam and is inconsistent with Tam Nong, "
        "Phu Tho; new coordinate is more plausible but partial-match medium confidence."
    ),
}


def fnum(value):
    try:
        if value is None or str(value).strip() == "":
            return None
        result = float(value)
        if math.isnan(result):
            return None
        return result
    except Exception:
        return None


def haversine_m(lat1, lon1, lat2, lon2):
    if None in (lat1, lon1, lat2, lon2):
        return None
    radius = 6_371_000.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    return 2 * radius * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def distance_flag(distance_m):
    if distance_m is None:
        return "no_coordinate_comparison"
    if distance_m <= 100:
        return "same_point_<=100m"
    if distance_m <= 500:
        return "near_100_500m"
    if distance_m <= 2_000:
        return "review_0.5_2km"
    if distance_m <= 10_000:
        return "review_2_10km"
    return "major_disagreement_>10km"


def load_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main():
    old = pd.read_excel(OLD_PATH)
    old_cols = list(old.columns)
    old_by_tt = {int(row[old_cols[0]]): row for _, row in old.iterrows()}

    rows = load_csv(INCLUSION_V1)
    new_rows = load_csv(NEW_PATH)
    new_by_tt = {int(row["tt"]): row for row in new_rows}

    for row in rows:
        old_tt = int(row["old_tt"])
        if old_tt in OVERRIDES:
            override = OVERRIDES[old_tt]
            row["review_status"] = override["review_status"]
            row["review_note"] = override["note"]
            if override["new_tt"] is None:
                for key in ["best_new_tt", "best_new_name", "best_new_address", "best_new_province"]:
                    row[key] = ""
            else:
                new_row = new_by_tt[override["new_tt"]]
                row["best_new_tt"] = str(override["new_tt"])
                row["best_new_name"] = new_row["ten_benh_vien"]
                row["best_new_address"] = new_row["dia_chi"]
                row["best_new_province"] = new_row["tinh_thanh_pho"]
        elif old_tt in OLD_COORD_NOTES:
            row["review_note"] = (row.get("review_note", "") + " " + OLD_COORD_NOTES[old_tt]).strip()

    write_csv(OUT_INCLUSION_CSV, rows)

    counts = Counter(row["review_status"] for row in rows)
    confirmed_count = sum(counts[status] for status in CONFIRMED_STATUSES)
    not_count = counts["not_confirmed_in_new_source"]
    needs_identity = [row for row in rows if row["review_status"] not in CONFIRMED_STATUSES]

    inclusion_md = [
        "# Reviewed Inclusion Check v2: old 130 vs new 170 Vietnam stroke-center source",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Reviewed audit CSV: `{OUT_INCLUSION_CSV}`",
        "",
        "## Bottom line",
        "",
        f"- Confirmed in the new 170 list: {confirmed_count}/130",
        f"- Not confirmed in the new 170 list: {not_count}/130",
        "",
        (
            "The old 130 is not a clean subset of the 170: three old rows are not "
            "confirmed in the new list. Several administrative renames are confirmed "
            "by address or coordinate evidence."
        ),
        "",
        "## Not confirmed",
        "",
        "| old TT | old name | note |",
        "|---:|---|---|",
    ]
    for row in needs_identity:
        inclusion_md.append(
            f"| {row['old_tt']} | {row['old_name'].replace(chr(10), ' ')} | {row['review_note']} |"
        )
    inclusion_md.extend(
        [
            "",
            "## Manual corrections applied in v2",
            "",
            "| old TT | old name | new TT | new name | status | note |",
            "|---:|---|---:|---|---|---|",
        ]
    )
    for old_tt in sorted(OVERRIDES):
        row = next(candidate for candidate in rows if int(candidate["old_tt"]) == old_tt)
        inclusion_md.append(
            f"| {row['old_tt']} | {row['old_name'].replace(chr(10), ' ')} | "
            f"{row['best_new_tt']} | {row['best_new_name']} | {row['review_status']} | "
            f"{row['review_note']} |"
        )
    OUT_INCLUSION_MD.write_text("\n".join(inclusion_md) + "\n", encoding="utf-8")

    audit_rows = []
    for row in rows:
        old_tt = int(row["old_tt"])
        old_row = old_by_tt[old_tt]
        new_tt = int(row["best_new_tt"]) if row.get("best_new_tt") else None
        new_row = new_by_tt.get(new_tt) if new_tt else None

        old_lat = fnum(old_row.get("latitude"))
        old_lon = fnum(old_row.get("longitude"))
        new_lat = fnum(new_row.get("google_lat")) if new_row else None
        new_lon = fnum(new_row.get("google_lon")) if new_row else None

        distance_m = haversine_m(old_lat, old_lon, new_lat, new_lon)
        flag = distance_flag(distance_m)
        confidence = new_row.get("geocode_confidence") if new_row else ""
        partial = new_row.get("google_partial_match") if new_row else ""
        location_type = new_row.get("google_location_type") if new_row else ""

        reasons = []
        if row["review_status"] not in CONFIRMED_STATUSES:
            reasons.append(row["review_status"])
        if flag in {"review_0.5_2km", "review_2_10km", "major_disagreement_>10km", "no_coordinate_comparison"}:
            reasons.append(flag)
        if confidence == "low":
            reasons.append("low_google_confidence")
        if partial == "True":
            reasons.append("google_partial_match")
        if old_tt in OLD_COORD_NOTES:
            reasons.append("old_coordinate_likely_wrong")

        audit_rows.append(
            {
                "old_tt": old_tt,
                "old_name": str(old_row[old_cols[1]]).replace("\n", " "),
                "old_address": str(old_row[old_cols[4]]).replace("\n", " "),
                "old_latitude": old_lat if old_lat is not None else "",
                "old_longitude": old_lon if old_lon is not None else "",
                "review_status_130_to_170": row["review_status"],
                "new_tt": new_tt if new_tt else "",
                "new_name": new_row.get("ten_benh_vien") if new_row else "",
                "new_address": new_row.get("dia_chi") if new_row else "",
                "new_province": new_row.get("tinh_thanh_pho") if new_row else "",
                "google_latitude": new_lat if new_lat is not None else "",
                "google_longitude": new_lon if new_lon is not None else "",
                "old_vs_google_distance_m": round(distance_m, 1) if distance_m is not None else "",
                "distance_flag": flag,
                "geocode_confidence": confidence,
                "google_location_type": location_type,
                "google_partial_match": partial,
                "google_formatted_address": new_row.get("google_formatted_address") if new_row else "",
                "audit_review_needed": "yes" if reasons else "no",
                "audit_reasons": ";".join(reasons),
                "review_note": row.get("review_note", ""),
            }
        )
    write_csv(OUT_AUDIT_CSV, audit_rows)

    distance_counts = Counter(row["distance_flag"] for row in audit_rows)
    confidence_counts = Counter(row["geocode_confidence"] for row in audit_rows if row["geocode_confidence"])
    review_rows = [row for row in audit_rows if row["audit_review_needed"] == "yes"]
    major_rows = [row for row in audit_rows if row["distance_flag"] == "major_disagreement_>10km"]
    not_confirmed = [row for row in audit_rows if row["review_status_130_to_170"] not in CONFIRMED_STATUSES]
    other_review = [row for row in review_rows if row not in major_rows and row not in not_confirmed]

    audit_md = [
        "# Coordinate Audit v2: Google-geocoded 170-row file against old 130-row workbook",
        "",
        f"Created UTC: {datetime.now(timezone.utc).isoformat()}",
        "",
        f"Row-level audit CSV: `{OUT_AUDIT_CSV}`",
        "",
        "## Summary",
        "",
        f"- Old rows audited: {len(audit_rows)}",
        f"- Confirmed old rows with a new-row match: {confirmed_count}/130",
        f"- Not confirmed old rows: {not_count}/130",
        f"- Rows flagged for coordinate or identity review: {len(review_rows)}/130",
        "",
        "Distance categories after v2 mapping corrections:",
    ]
    for key in [
        "same_point_<=100m",
        "near_100_500m",
        "review_0.5_2km",
        "review_2_10km",
        "major_disagreement_>10km",
        "no_coordinate_comparison",
    ]:
        audit_md.append(f"- {key}: {distance_counts[key]}")
    audit_md.extend(["", "Google confidence among matched old rows:"])
    for key, value in sorted(confidence_counts.items()):
        audit_md.append(f"- {key}: {value}")

    audit_md.extend(["", "## Not confirmed in new 170", "", "| old TT | old name | note |", "|---:|---|---|"])
    for row in not_confirmed:
        audit_md.append(f"| {row['old_tt']} | {row['old_name']} | {row['review_note']} |")

    audit_md.extend(["", "## Major coordinate disagreements above 10 km", ""])
    if major_rows:
        audit_md.extend(
            [
                "| old TT | old name | new TT | new name | distance km | confidence | location type | reasons |",
                "|---:|---|---:|---|---:|---|---|---|",
            ]
        )
        for row in sorted(major_rows, key=lambda item: float(item["old_vs_google_distance_m"] or 0), reverse=True):
            audit_md.append(
                f"| {row['old_tt']} | {row['old_name']} | {row['new_tt']} | {row['new_name']} | "
                f"{float(row['old_vs_google_distance_m']) / 1000:.1f} | {row['geocode_confidence']} | "
                f"{row['google_location_type']} | {row['audit_reasons']} |"
            )
    else:
        audit_md.append("None.")

    audit_md.extend(["", "## Other rows needing coordinate review", ""])
    if other_review:
        audit_md.extend(
            [
                "| old TT | old name | new TT | new name | distance m | confidence | location type | reasons |",
                "|---:|---|---:|---|---:|---|---|---|",
            ]
        )
        for row in sorted(other_review, key=lambda item: float(item["old_vs_google_distance_m"] or 0), reverse=True):
            audit_md.append(
                f"| {row['old_tt']} | {row['old_name']} | {row['new_tt']} | {row['new_name']} | "
                f"{row['old_vs_google_distance_m']} | {row['geocode_confidence']} | "
                f"{row['google_location_type']} | {row['audit_reasons']} |"
            )
    else:
        audit_md.append("None.")

    audit_md.extend(
        [
            "",
            "## Recommendation",
            "",
            (
                "Use the 170-row Google-geocoded file only after resolving the three "
                "not-confirmed old facilities and manually checking the rows flagged "
                "for coordinate review. The audit suggests some old 130 workbook "
                "coordinates were probably wrong, so large old-vs-new distance is not "
                "always evidence against the new Google point."
            ),
        ]
    )
    OUT_AUDIT_MD.write_text("\n".join(audit_md) + "\n", encoding="utf-8")

    print(OUT_INCLUSION_MD)
    print(OUT_AUDIT_MD)
    print("confirmed", confirmed_count, "not_confirmed", not_count, "review_rows", len(review_rows))
    print("distance_counts", dict(distance_counts))


if __name__ == "__main__":
    main()
