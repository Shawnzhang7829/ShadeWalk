"""
Build the 21-archetype x 2-day_type x 24-hour occupancy density table from the
20 SGP archetype IDF files.

For each IDF:
  1. parse `People` object -> `People per Floor Area {person/m2}`
  2. parse `Schedule:Year` named "Building Occupancy" -> Schedule:Week:Daily
  3. parse Schedule:Week:Daily -> day refs for Monday and Sunday
  4. parse Schedule:Day:Interval for those days -> 24-hour occupancy fractions

Output:
  - density(arch, day_type, hour) in person / m^2  (= ppl_per_m2 * occupancy)
  - DataFrame: arch x (day_type, hour) saved to CSV/parquet for inspection

Usage:
  python -m Patronage_Flow.build_occupancy_table
"""
from __future__ import annotations
import re
from pathlib import Path

import numpy as np
import pandas as pd

from Patronage_Flow import Constants as C


# ---------------------------------------------------------------------------
# IDF mini-parser (no eppy / no IDD required)
# ---------------------------------------------------------------------------
def parse_idf_objects(path: Path) -> list[tuple[str, list[str]]]:
    """Yield (object_type, [field1, field2, ...]) for every object in the IDF.

    Strips inline comments (`! ...`) and trailing/leading whitespace; objects
    are terminated by `;`.
    """
    text = path.read_text(encoding="utf-8")
    text = re.sub(r"!.*", "", text)            # strip comments
    chunks = text.split(";")
    objects = []
    for chunk in chunks:
        # KEEP empty fields so positional indices match the IDF schema
        fields = [f.strip() for f in chunk.split(",")]
        # but drop pure-whitespace prefix/suffix that surrounds object boundaries
        while fields and fields[0] == "":
            fields.pop(0)
        while fields and fields[-1] == "":
            fields.pop()
        if not fields:
            continue
        objects.append((fields[0], fields[1:]))
    return objects


def find_objects(objects, type_name: str, name: str | None = None):
    """Filter objects by case-insensitive type match and optional name match."""
    type_lc = type_name.lower()
    out = []
    for obj_type, fields in objects:
        if obj_type.lower() != type_lc:
            continue
        if name is not None and (not fields or fields[0] != name):
            continue
        out.append((obj_type, fields))
    return out


def hour_from_time_token(token: str) -> int:
    """Parse 'HH:MM' or 'HH' to integer end-hour for EnergyPlus 'Until X:00'.
    24:00 maps to 24."""
    token = token.strip()
    if ":" in token:
        h, m = token.split(":")
        h = int(h)
        m = int(m)
        # EnergyPlus 'Until HH:MM' applies up to (but not incl.) HH:MM
        # If MM != 00, round up to next hour bucket for our hourly model
        if m == 0:
            return h
        return h + 1
    return int(token)


def parse_day_interval_to_24h(objects, day_name: str) -> list[float]:
    """Return a 24-element list of fractional occupancy for the named day."""
    arr = [0.0] * 24
    matches = find_objects(objects, "Schedule:Day:Interval", day_name)
    if not matches:
        # try Schedule:Day:Hourly fallback
        for obj_type, fields in find_objects(objects, "Schedule:Day:Hourly", day_name):
            # fields: name, type_limits, hour1_value, hour2_value, ..., hour24_value
            try:
                vals = [float(v) for v in fields[2:26]]
                if len(vals) == 24:
                    return vals
            except ValueError:
                continue
        return arr

    _, fields = matches[0]
    # Schedule:Day:Interval fields:
    #   name, type_limits, interpolate, time1, value1, time2, value2, ...
    t_prev = 0
    i = 3
    while i + 1 < len(fields):
        time_str = fields[i]
        # accept "Until: 09:00" form too
        time_str = re.sub(r"^(?:until\s*:?)\s*", "", time_str, flags=re.IGNORECASE).strip()
        try:
            value = float(fields[i + 1])
        except ValueError:
            break
        t_curr = hour_from_time_token(time_str)
        for h in range(t_prev, min(t_curr, 24)):
            arr[h] = value
        t_prev = t_curr
        i += 2
    return arr


def parse_compact_to_24h(objects, sched_name: str, day_filter: str = "weekday"):
    """Parse a Schedule:Compact for one of the day groups.

    Schedule:Compact has the format:
        Schedule:Compact, name, type_limits,
        Through: <date>, For: <day_groups>, Until: HH:MM, value, Until: ..., ...,
        For: <other_day_groups>, Until: ...;

    day_filter ∈ {'weekday', 'weekend'} picks which `For:` block to read.
    """
    arr = [0.0] * 24
    matches = find_objects(objects, "Schedule:Compact", sched_name)
    if not matches:
        return arr
    _, fields = matches[0]
    # field 0 is type_limits, then series of "Through:..." / "For:..." /
    # "Until:..." / value tokens
    i = 1
    in_target = False
    t_prev = 0
    while i < len(fields):
        f = fields[i]
        flow = f.lower()
        if flow.startswith("through"):
            i += 1
            continue
        if flow.startswith("for"):
            day_spec = flow[3:].lstrip(":").strip()
            wants = day_filter == "weekday"
            # AllDays / Weekdays / Mon / Tue / ... / Weekends / Sat / Sun / Holiday
            tokens = re.split(r"\s+", day_spec)
            tokens = [t for t in tokens if t]
            is_weekday = any(t in {"weekdays", "monday", "tuesday", "wednesday",
                                    "thursday", "friday", "alldays"} for t in tokens)
            is_weekend = any(t in {"weekends", "saturday", "sunday", "alldays"} for t in tokens)
            in_target = (wants and is_weekday) or (not wants and is_weekend)
            t_prev = 0
            i += 1
            continue
        if flow.startswith("until"):
            time_str = re.sub(r"^until\s*:?", "", f, flags=re.IGNORECASE).strip()
            try:
                value = float(fields[i + 1])
            except (IndexError, ValueError):
                break
            if in_target:
                t_curr = hour_from_time_token(time_str)
                for h in range(t_prev, min(t_curr, 24)):
                    arr[h] = value
                t_prev = t_curr
            i += 2
            continue
        i += 1
    return arr


# ---------------------------------------------------------------------------
# extract one IDF
# ---------------------------------------------------------------------------
def extract_idf_occupancy(idf_path: Path) -> tuple[float, list[float], list[float]]:
    """Return (people_per_m2, weekday_24h, weekend_24h) for one IDF."""
    objects = parse_idf_objects(idf_path)

    # 1. People per Floor Area
    ppl_per_m2 = None
    occ_sched_name = "Building Occupancy"
    for obj_type, fields in find_objects(objects, "People"):
        # People fields (E+ schema):
        #   name, zone, schedule, calc_method, num_people, ppl_per_area,
        #   floor_area_per_person, fraction_radiant, ...
        # the "schedule" is field index 2
        if len(fields) >= 6:
            try:
                ppl_per_m2 = float(fields[5])
            except ValueError:
                continue
            occ_sched_name = fields[2] or "Building Occupancy"
            break
    if ppl_per_m2 is None:
        raise RuntimeError(f"People object not found in {idf_path.name}")

    # 2. trace Schedule:Year -> Schedule:Week:Daily -> Schedule:Day:Interval
    weekday = [0.0] * 24
    weekend = [0.0] * 24

    year = find_objects(objects, "Schedule:Year", occ_sched_name)
    if year:
        _, yfields = year[0]
        # fields: name, type_limits, week_name1, sm1, sd1, em1, ed1, [week_name2, ...]
        if len(yfields) >= 3:
            week_name = yfields[2]
            wk = find_objects(objects, "Schedule:Week:Daily", week_name)
            if wk:
                _, wfields = wk[0]
                # fields: name, sun, mon, tue, wed, thu, fri, sat, holiday, ...
                if len(wfields) >= 8:
                    sun = wfields[1]
                    mon = wfields[2]
                    weekday = parse_day_interval_to_24h(objects, mon)
                    weekend = parse_day_interval_to_24h(objects, sun)

    # 3. fallback: Schedule:Compact directly named occ_sched_name
    if max(weekday) == 0 and max(weekend) == 0:
        wd = parse_compact_to_24h(objects, occ_sched_name, "weekday")
        we = parse_compact_to_24h(objects, occ_sched_name, "weekend")
        if max(wd) > 0 or max(we) > 0:
            weekday, weekend = wd, we

    return ppl_per_m2, weekday, weekend


# ---------------------------------------------------------------------------
# archetype <-> IDF mapping
# ---------------------------------------------------------------------------
ARCHETYPE_TO_IDF: dict[str, str | tuple[str, str]] = {
    "hdb":                "01_HDB_SGP_2025_V5.idf",
    "landed_property":    "02_SFH_SGP_2025_V5.idf",
    "private_apartment":  "03_MFH_SGP_2025_V5.idf",
    "hotel":              "04_Hotel_SGP_2025_V5.idf",
    "retail":             "05_Retail_SGP_2025_V5.idf",
    "office":             "06_Office_SGP_2025_V5.idf",
    "business_park":      "07_BusinessPark_SGP_2025_V5.idf",
    "hospital":           "08_Hospital_SGP_2025_V5.idf",
    "clinic":             "09_Polyclinic_SGP_2025_V5.idf",
    "nursing_home":       "10_NursingHome_SGP_2025_V5.idf",
    "ihl":                "11_IHL_SGP_2025_V5.idf",
    "non_ihl":            "12_NonIHL_SGP_2025_V5.idf",
    "industrial":         "13_IndustrialB1_SGP_2025_V5.idf",
    "data_centre":        "15_DataCentre_SGP_2025_V5.idf",
    "community_cultural": "16_CivicCommunity_Cultural_SGP_2025_V5.idf",
    "sports":             "17_SportsRec_SGP_2025_V5.idf",
    "restaurant":         "18_Restaurant_SGP_2025_V5.idf",
    "hawker_centre":      "19_HawkerCentre_SGP_2025_V5.idf",
    "supermarket":        "20_Supermarket_SGP_2025_V5.idf",
    # blends (mean of two IDF profiles)
    "mixed_development":  ("06_Office_SGP_2025_V5.idf", "05_Retail_SGP_2025_V5.idf"),
    "shophouse":          ("05_Retail_SGP_2025_V5.idf", "03_MFH_SGP_2025_V5.idf"),
}


# ---------------------------------------------------------------------------
# post-processing: schedule shift + minimum floor
# ---------------------------------------------------------------------------
# IDF "Schedule:Building Occupancy" describes WHEN PEOPLE ARE INSIDE
# (HVAC perspective). For pedestrian flow, we care WHEN PEOPLE ARE WALKING
# TO/FROM the building -- this happens both BEFORE peak occupancy (arrival
# walks) and AFTER peak occupancy (departure walks).
#
# A simple shift -1h captured AM arrival but BROKE PM departure (e.g., office
# at hour 18 ended up = 0). The fix is a SYMMETRIC rolling-max window:
#   weight(h) = max( occupancy(h-1), occupancy(h), occupancy(h+1) )
# This preserves both shoulders -- if the building has occupancy at h-1, h,
# or h+1, then at hour h there is walking activity (arriving or departing).
# Applied to all archetypes by default; opt out per archetype if needed.
ROLLING_MAX_WINDOW = 1   # +/- 1 hour each side -> 3-hour window centered

# IDF schedules sometimes drop to exactly 0 (e.g., HDB 10-18, office 0-8).
# In reality buildings always have *some* people (residents, on-call staff,
# cleaners, security, etc.). Floor = MIN_FLOOR_FRACTION * archetype weekly
# peak prevents "no destinations exist" artifacts in the Huff allocation.
MIN_FLOOR_FRACTION = 0.10


def _rolling_max(arr: list[float], window: int = 1) -> list[float]:
    """Rolling max with +/- window neighbors. arr has 24 elements; clamp at edges.
    For each hour h, output[h] = max(arr[h-window:h+window+1])."""
    n = len(arr)
    out = [0.0] * n
    for h in range(n):
        lo = max(0, h - window)
        hi = min(n, h + window + 1)
        out[h] = max(arr[lo:hi])
    return out


def _apply_floor(arr: list[float], peak: float, frac: float) -> list[float]:
    """Apply a minimum floor = peak * frac to all hours."""
    floor = peak * frac
    return [max(v, floor) for v in arr]


def build_density_table(idf_dir: Path) -> pd.DataFrame:
    """Build a DataFrame indexed by archetype, columns = MultiIndex(day_type, hour),
    values = people-per-m^2 at that hour."""
    rows = {}
    cache: dict[str, tuple[float, list[float], list[float]]] = {}

    def get(idf_name: str):
        if idf_name not in cache:
            cache[idf_name] = extract_idf_occupancy(idf_dir / idf_name)
        return cache[idf_name]

    for arch, src in ARCHETYPE_TO_IDF.items():
        if isinstance(src, tuple):
            a, b = src
            ppl_a, wd_a, we_a = get(a)
            ppl_b, wd_b, we_b = get(b)
            ppl = (ppl_a + ppl_b) / 2
            wd = [(x + y) / 2 for x, y in zip(wd_a, wd_b)]
            we = [(x + y) / 2 for x, y in zip(we_a, we_b)]
        else:
            ppl, wd, we = get(src)

        # raw density before adjustments
        wd_raw = [ppl * v for v in wd]
        we_raw = [ppl * v for v in we]

        # symmetric rolling max so both AM arrival (h-1 shoulder) and PM
        # departure (h+1 shoulder) walking activities are captured at hour h
        wd_max = _rolling_max(wd_raw, ROLLING_MAX_WINDOW)
        we_max = _rolling_max(we_raw, ROLLING_MAX_WINDOW)

        # weekly peak (used for floor)
        weekly_peak = max(max(wd_max), max(we_max)) if wd_max or we_max else 0.0
        wd_final = _apply_floor(wd_max, weekly_peak, MIN_FLOOR_FRACTION)
        we_final = _apply_floor(we_max, weekly_peak, MIN_FLOOR_FRACTION)

        rows[arch] = {
            **{("WEEKDAY", h): wd_final[h] for h in range(24)},
            **{("WEEKENDS/HOLIDAY", h): we_final[h] for h in range(24)},
        }

    df = pd.DataFrame.from_dict(rows, orient="index")
    df.index.name = "archetype"
    df.columns = pd.MultiIndex.from_tuples(df.columns, names=["DAY_TYPE", "HOUR"])
    return df


def main():
    idf_dir = C.ROOT / "AllArhcetypes_SGP_2025_V5"
    out_path = C.ROOT / "Patronage_Flow" / "lookup" / "occupancy_density.parquet"
    out_path_csv = out_path.with_suffix(".csv")

    df = build_density_table(idf_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path)
    df.to_csv(out_path_csv)
    print(f"[occupancy] wrote {out_path}")
    print(f"[occupancy] wrote {out_path_csv}")
    print()
    print("--- per-archetype summary (people / m^2) ---")
    pk_w = df["WEEKDAY"].max(axis=1)
    pk_we = df["WEEKENDS/HOLIDAY"].max(axis=1)
    avg_w = df["WEEKDAY"].mean(axis=1)
    summary = pd.DataFrame({
        "weekday_peak": pk_w,
        "weekday_mean": avg_w,
        "weekend_peak": pk_we,
        "peak_hour_weekday": df["WEEKDAY"].idxmax(axis=1),
    })
    print(summary.to_string())


if __name__ == "__main__":
    main()
