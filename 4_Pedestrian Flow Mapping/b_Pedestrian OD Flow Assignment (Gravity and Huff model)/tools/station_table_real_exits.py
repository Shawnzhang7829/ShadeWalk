# -*- coding: utf-8 -*-
"""Station table v4: one record per REAL exit point / bus stop, values rebuilt from the raw LTA DataMall month files.

Why
---
`export_station_hourly.py` writes one row per (MRT / LRT exit x line code) and one row per bus-stop shapefile row.  Three things
make those rows unusable as origin weights without care: (1) `tot_*` is the whole-station value repeated on every row; (2) an
interchange station gets one full set of exit rows per line code (Orchard NS22 / TE14: 13 exits x 2 codes = 26 rows on 13 points),
so `inj_* = tot / n_exits` still counts the station once per code; (3) before 2026-09-17 the raw-CSV loader parsed bus-stop codes as
integers, dropped the leading zero of the central-area codes (01xxx-09xxx) and exported those 225 stops with 0 ridership.
This tool rebuilds the table from the raw month files on the geometry of the export:
  * the station value of a (code, day type, hour) cell = raw monthly tap-in + tap-out / days (22 weekdays, 9 weekend days by default);
    an interchange is ONE raw code ('TE14/NS22') = one station;
  * MRT / LRT: the rows of all line codes of a station are collapsed to its distinct exit points (coincident points once); every
    exit point gets one record with inj_* = station value / number of real exit points; a point shared by two raw codes
    (Stevens DT10 / TE11) carries both shares; PT_CODE = alphabetically first line code, LTA_CODE = raw code(s), CODES = codes on the point;
  * bus stops: one record per shapefile row; a code with two rows shares its value over them;
  * codes of the export that are absent from the raw month keep 0; raw codes without a row in the export are reported (no geometry).
Usage
-----
python tools/station_table_real_exits.py --export output/station_hourly_ridership.gpkg
       --raw-dir "Station flow/2026-01/node" --tag 202601 --out output/station_hourly_ridership_v4.gpkg [--weekdays 22 --weekend-days 9]
Check printed at the end: every covered station's records add up to the raw value in all 48 (day type, hour) slots.
"""
import argparse, os, sys
import numpy as np, pandas as pd, geopandas as gpd

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument('--export', required=True, help='station_hourly_ridership.gpkg written by export_station_hourly.py (rows = exits x line codes, bus stops)')
ap.add_argument('--raw-dir', required=True, help='folder with transport_node_bus_<tag>.csv and transport_node_train_<tag>.csv')
ap.add_argument('--tag', required=True, help='YYYYMM of the month files, e.g. 202601')
ap.add_argument('--out', required=True, help='output GeoPackage')
ap.add_argument('--weekdays', type=float, default=22.0, help='divisor for WEEKDAY volumes (calendar weekdays of the month)')
ap.add_argument('--weekend-days', type=float, default=9.0, help='divisor for WEEKENDS/HOLIDAY volumes')
ap.add_argument('--snap', type=float, default=0.05, help='exit points closer than this (m) are the same point')
args = ap.parse_args()
SLOTS = [(day, h) for day in ('weekday', 'weekends') for h in range(24)]; DAY = {'weekday': 'WEEKDAY', 'weekends': 'WEEKENDS/HOLIDAY'}; ND = {'weekday': args.weekdays, 'weekends': args.weekend_days}
TAG = [f'{day}_{h:02d}' for day, h in SLOTS]

# ---- raw month files (codes as text!) -> per raw code: 48-slot arrays of tap-in / tap-out per day
VIN, VOUT = {}, {}                                                            # raw code -> (48,) arrays
for kind in ('train', 'bus'):
    c = pd.read_csv(os.path.join(args.raw_dir, f'transport_node_{kind}_{args.tag}.csv'), dtype=str); cols = {x.upper(): x for x in c.columns}
    c = c.dropna(subset=[cols['PT_CODE'], cols['TIME_PER_HOUR'], cols['DAY_TYPE']])
    c = c.assign(code=c[cols['PT_CODE']].str.strip(), day=c[cols['DAY_TYPE']].str.strip().str.upper(), h=c[cols['TIME_PER_HOUR']].astype(float).astype(int),
                 vin=c[cols['TOTAL_TAP_IN_VOLUME']].fillna('0').astype(float), vout=c[cols['TOTAL_TAP_OUT_VOLUME']].fillna('0').astype(float))
    g = c.groupby(['code', 'day', 'h'])[['vin', 'vout']].sum()
    idx = pd.MultiIndex.from_tuples([(DAY[d], h) for d, h in SLOTS]); div = np.array([ND[d] for d, h in SLOTS])
    for code, sub in g.groupby(level=0):
        r = sub.droplevel(0).reindex(idx).fillna(0.0); VIN[code] = r.vin.values / div; VOUT[code] = r.vout.values / div
    print(f'raw {kind}: {c.code.nunique():,} codes')
part2raw = {p.strip(): code for code in VIN for p in code.split('/')}

# ---- export rows
ex = gpd.read_file(args.export); code = ex['PT_CODE'].fillna('').astype(str).str.strip(); src = ex['source'].astype(str).str.upper()
is_mrt = src.str.contains('MRT').values; rawcode = np.array([part2raw.get(c, '') for c in code], dtype=object)
print(f'export rows {len(ex):,} (MRT {int(is_mrt.sum())}, bus {int((~is_mrt).sum())}) | rows matched to a raw code {int((rawcode != "").sum()):,}')
Z = np.zeros(48); recs = []
def rec_base(pt_code, lta, codes, source, n_exits, n_codes, n_rows, geom): return {'PT_CODE': pt_code, 'LTA_CODE': lta, 'CODES': codes, 'source': source, 'n_exits': float(n_exits), 'n_codes': int(n_codes), 'n_rows': int(n_rows), 'geometry': geom}
def fill(rec, vin, vout, inj):
    for j, t in enumerate(TAG): rec[f'in_{t}'] = float(vin[j]); rec[f'out_{t}'] = float(vout[j]); rec[f'tot_{t}'] = float(vin[j] + vout[j]); rec[f'inj_{t}'] = float(inj[j])
    return rec
# MRT / LRT: one record per distinct exit point, all line codes of the station together
m = ex[is_mrt].copy(); m['code'] = code[is_mrt].values; m['rawcode'] = rawcode[is_mrt]; m = m[m.rawcode != '']
m['pt'] = [(round(p.x / args.snap) * args.snap, round(p.y / args.snap) * args.snap) for p in m.geometry]; n_real = m.groupby('rawcode').pt.nunique()
for pt, grp in m.groupby('pt', sort=False):
    rcs = sorted(grp.rawcode.unique()); codes_all = sorted(grp.code.unique())
    vin = sum((VIN[rc] for rc in rcs), Z.copy()); vout = sum((VOUT[rc] for rc in rcs), Z.copy()); inj = sum(((VIN[rc] + VOUT[rc]) / n_real[rc] for rc in rcs), Z.copy())
    recs.append(fill(rec_base(codes_all[0], '+'.join(rcs), ','.join(codes_all), 'MRT', n_real[rcs[0]], len(codes_all), 1, grp.geometry.iloc[0]), vin, vout, inj))
# bus stops: one record per export row; duplicated codes share
b = ex[~is_mrt].copy(); b['code'] = code[~is_mrt].values; b['rawcode'] = rawcode[~is_mrt]; n_dup = b.groupby('code').code.transform('size').astype(float).values
for j, (_, r) in enumerate(b.iterrows()):
    if r.rawcode: vin, vout = VIN[r.rawcode], VOUT[r.rawcode]; inj = (vin + vout) / n_dup[j]
    else: vin = vout = inj = Z
    recs.append(fill(rec_base(r.code, r.rawcode or r.code, r.code, 'BUS', 1, 1, n_dup[j], r.geometry), vin, vout, inj))
out = gpd.GeoDataFrame(recs, geometry='geometry', crs=ex.crs)
for day in ('weekday', 'weekends'):
    for pre in ('in', 'out', 'tot'): out[f'{pre}_{day}_total'] = out[[f'{pre}_{day}_{h:02d}' for h in range(24)]].sum(axis=1)
# ---- check: for every raw code with records, its records' inj add up to the station value in all 48 slots
# (a point shared by several raw codes carries the sum of their shares; attribute its record to each code by the code's own share)
share = {}                                                                     # raw code -> accumulated (48,)
for rec_ in recs:
    rcs = rec_['LTA_CODE'].split('+'); v = np.array([rec_[f'inj_{t}'] for t in TAG])
    if rec_['source'] == 'MRT' and rcs[0] in n_real:
        parts = [(rc, (VIN[rc] + VOUT[rc]) / n_real[rc]) for rc in rcs if rc in VIN]; tot = sum((p for _, p in parts), Z.copy())
        for rc, p in parts: share[rc] = share.get(rc, Z.copy()) + np.where(tot > 0, v * p / np.where(tot > 0, tot, 1), 0.0)
    elif rec_['source'] == 'BUS' and rcs[0] in VIN: share[rcs[0]] = share.get(rcs[0], Z.copy()) + v
bad = sum(int((np.abs(share[rc] - (VIN[rc] + VOUT[rc])) > 0.05).sum()) for rc in share)
print(f'check: (station, slot) cells where the records do not add up to the raw value: {bad} of {len(share) * 48} (must be 0)')
miss = sorted(set(VIN) - set(share)); print(f'raw codes without a record (no geometry in the export): {len(miss)} {miss[:6]}')
print(f'records {len(out):,} (MRT exit points {int((out.source == "MRT").sum())}, bus stops {int((out.source == "BUS").sum())}) | 14:00 weekday total {out.inj_weekday_14.sum():,.0f}')
out.to_file(args.out, driver='GPKG'); print('SAVED', args.out)
