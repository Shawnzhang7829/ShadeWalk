# Module 5 - Reconstruction of the pedestrian network with shade-facility centrelines

Builds the routable pedestrian network used by Module 4a from the OSM pedestrian network plus the centrelines of
the two shade facilities (arcades from Module 3b, covered linkways from Module 2). Facility centrelines are
inserted as first-class network edges, OSM lines that duplicate them are removed, and every facility is connected
to the surrounding network so that routes can enter and leave the shaded walkways. Environment A.

## 1. Stages

| Stage | Script | Rule set | Output (`src` tags) |
|---|---|---|---|
| 1 | `scripts/step4_1_city.py` | Arcade centrelines = the arcade facade runs (`step2b_runs_sg.gpkg`) offset 1 m into the arcade strip along their long axis (`ARC_OFFSET = 1.0`); covered-linkway centrelines = Voronoi medial axis of the polygons (`STEP 1.0 m`, `PRUNE 4.0 m`, `SIMP 3.0 m`). Every facility polygon gets a `fac_id`. | `step4_1_shade_centerline_SG.gpkg` (`arcade`, `linkway`) |
| 2a | `scripts/step4_2a_fac.py` | Facility-internal cleaning: join dangling ends across parts of the same facility and end-to-end between facilities when closer than 5 m (`D5`), preferring the straightest connection (deflection >= 120 deg, `ANG`); then delete linkway spurs < 3 m (`DEL`) that hang on one side only (Voronoi artefacts; arcade lines are not pruned); each connected block becomes one line unit (`line_id`). | `step4_fac_lines.gpkg` (`facconn` = new joins) |
| 2b | `scripts/step4_2b_osm.py` | OSM-internal cleaning: read `Highway_OSM.gpkg`, drop motorway / trunk classes (`EXCL_HW`), connect every degree-1 end to the nearest segment within 15 m (`D15`, foot-of-perpendicular split, shortest and non-redundant). | `step4_osm_lines.gpkg` (`osm`, `osmlink`) |
| 3a | `scripts/step4_3a_cut_osm.py` | Remove OSM lines that duplicate a facility centreline: >= 80 % of the line inside the 8 m band (4 m each side, `BUF`) **and** parallel (< 30 deg); crossing lines are kept. | `step4_osm_kept.gpkg` |
| 3b | `scripts/step4_3b_link.py` | Connect facilities to the OSM network: each facility end (degree 1) to the nearest OSM segment within 15 m (`D150`); each arcade corner (after simplification, deflection > 30 deg) to the nearest OSM segment within 30 m (`D_ARC`). Rejections: link crosses a building unless an OSM line already runs through it; link overlaps the facility line by > 50 % (back-tracking); landing point lies under a facility polygon; conflicts with an existing link (`DCLASH 4 m`). | `step4_connected.gpkg` (`fac2osm`) |
| 4 | `scripts/step4_4_global.py` | Global closure: every remaining degree-1 end joins the nearest segment within 15 m, cross-component joins first (to attach islands), no through-building links. | `step4_network_final.gpkg` (`gapfill`) |
| - | `scripts/step4_analyze.py` | Connected-component composition of the final network (main component, isolated facility blocks, OSM fragments). | report |

Order of the whole procedure (as fixed in the design notes): facilities first (5 m joins, then 3 m spur removal),
OSM second (15 m), facility-to-OSM third (15 m ends / 30 m arcade corners, never landing under a facility),
global 15 m closure last.

## 2. Result (Singapore)

| Layer | Segments |
|---|---|
| facility centrelines (stage 1) | 15,873 |
| facility lines after cleaning (2a) | 16,467 |
| OSM lines after cleaning (2b) | 404,613 |
| OSM lines kept after duplicate removal (3a) | 400,151 |
| facilities connected (3b) | 459,646 |
| **final network** | **469,434** |

The final layer has the fields `src` (`osm`, `osmlink`, `arcade`, `linkway`, `facconn`, `fac2osm`, `gapfill`) and
`length`; Module 4a maps them to `footpath` / `shade` / `bridge` (`make_prep.py`) and keeps the main connected
component for routing. Final composition: osm 332,725, osmlink 88,225, fac2osm 21,057, linkway 12,803,
arcade 7,455, gapfill 6,395, facconn 774 segments; 19,824 km in total, main component 16,545 km.

Verification of the released scripts (2026-09-10): the six stages were re-run from the published inputs and
reproduced every production layer exactly (same feature counts at each stage, same source composition, identical
sorted segment lengths); the whole chain took about 15 min.

## 3. Path configuration

Every script defines its inputs and `OUT` folder in the constants at the top
(`ARC`, `RUNS`, `LKW`, `OSM`, `BLDG`, `OUT`). Edit them before running; run the stages in the order 1, 2a, 2b, 3a, 3b, 4.

```bat
python scripts\step4_1_city.py
python scripts\step4_2a_fac.py
python scripts\step4_2b_osm.py
python scripts\step4_3a_cut_osm.py
python scripts\step4_3b_link.py
python scripts\step4_4_global.py
python scripts\step4_analyze.py
```
