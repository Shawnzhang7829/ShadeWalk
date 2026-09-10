# Module 4a - Coolest-route algorithm and route-cost calculation

Assigns station-anchored pedestrian demand to the reconstructed network under two routing rules - shortest path
and coolest path - and quantifies the shade gained, the detour accepted and the flow carried by each shade
facility. The same routing core drives the ShadeWalk web tool (`webapp/`). Environment A.

## 1. Route cost

For every network edge *i* with length `l_i` and 14:00 shade fraction `sigma_i` (share of the edge in shadow):

```
rho_i   = (1 - sigma_i) + lambda          street resistance
omega_i = l_i * rho_i                       edge cost
coolest route = argmin over paths of  sum(omega_i)         (Dijkstra)
shortest route = argmin of  sum(l_i)                        (lambda -> infinity)
```

`lambda` is the *shade reward* parameter: the extra distance a walker accepts in exchange for shade. The paper uses
`lambda = 0.2`, i.e. a reward ratio `eta = lambda / (1 + lambda) = 1/6`: one metre in the sun costs as much as six
metres in the shade, so a detour pays off only if at least one sixth of the extra length is converted into avoided
sun. `step4_4f_flow_lam.py` sweeps 18 values of lambda (3.0 ... 0.001); `step4_4e_flow_detour.py` is the alternative
formulation with a hard detour cap `tau` (coolest route among paths with length <= tau x shortest).

## 2. Demand model

Every trip has one end at a transit station (MRT / LRT exit or bus stop) and the other at a building.
Station demand = 14:00 weekday ridership (`tot_weekday_14` from Module 4b). It is distributed to the buildings
within the walking catchment (network distance <= 800 m for MRT / LRT, <= 400 m for bus, cut-off Dijkstra) in
proportion to `building weight x exp(-d / 350 m)`, where the building weight is the 14:00 occupancy weight of
Module 4b (`weight_weekday_14`). Stations and buildings snap to the nearest network node within 120 m; pairs closer
than 20 m are dropped. Flow on an edge = sum of the demand of all OD pairs whose route uses the edge.

## 3. Scripts

| Step | Script | Purpose |
|---|---|---|
| 1 | `scripts/01_prep/make_prep.py` | read `step4_network_final.gpkg` (Module 5), cluster endpoints (0.5 m), label connected components (`comp`, main = 0) and map `src` to `footpath` (osm, osmlink) / `shade` (arcade, linkway, facconn) / `bridge` (fac2osm, gapfill) |
| 2 | `scripts/02_edge_shade/step4_4a_city_building_shadow.py` | LOD1 baseline: 14:00 shadow of buildings only (numpy port of the SOLWEIG shadow function, tiled, 13:30 sun position, 120 px margin) |
| 2b | `step4_4a_bldtree_shadow.py` | buildings + trees baseline (no facilities), vegetation as a separate opaque canopy |
| 3 | `step4_4b_city_edge_shade.py` | per-edge shade fraction from the full-system 14:00 shadow raster (Module 1) and from the building-only raster, sampled every 2 m; edges through building footprints count as fully shaded; builds the node / edge graph (`u`, `v`) |
| 3b | `step4_4b_esn.py`, `step4_edge_facility.py`, `step4_edge_class_1m.py`, `step4_network_split_1m.py`, `edge_px_permetre.py` | no-facility shade per edge, dominant facility class per edge (arcade > linkway > tree > building), per-metre attribution of shade sources, split of the network by shade class |
| 4 | `scripts/03_routing_flow/step4_4c_city_routing.py` | shortest and coolest (lambda = 0.15) routing for all station-building pairs, per-edge flows `flow_short`, `flow_cool`, metrics per building type |
| 4b | `step4_4c_orig_flow.py`, `step4_4c_orig_coolflow.py` | the same on the original footpath-only network (`flow_orig`, `flow_cool_orig`) to separate the network effect from the behaviour effect |
| 5 | `step4_4f_flow_lam.py` | flows for the lambda sweep (paper setting lambda = 0.2 included), summary of shade, detour and facility share per lambda |
| 5b | `step4_4e_flow_detour.py` | tau-capped variant (tau = 1.0, 1.2, 1.5, 2.0) |
| 6 | `step4_4d_city_viz.py` | city-wide maps of edge shade and flow |

Runtime: edge shade about 10 min (two 44,000 x 27,000 rasters in memory as uint8); one lambda value of the
city-wide routing about 1-2 h (one cut-off Dijkstra per station, 5,921 stations).

Verification of the released scripts (2026-09-10): `make_prep.py` and `step4_4b_city_edge_shade.py` were re-run
on the published network (469,434 edges, 21 s + 91 s). The graph is identical to the production edge layer
(component labels, `u` / `v`); the shade fractions are identical on 99.1 % of the edges (`shade_full`) and 97.6 %
(`shade_bld`); the differing edges lie where the building-footprint mask and the arcade shadows were updated after
the June production run (the rasters on disk post-date that run; the old rasters are no longer available).

## 4. Results (Singapore, 14:00, all trips)

| Routing | mean length (m) | mean shade | detour | facility share of shaded metres |
|---|---|---|---|---|
| shortest | 428.3 | 0.348 | 1.000 | 0.166 |
| coolest, lambda = 1.0 | 438.5 | 0.439 | 1.020 | 0.229 |
| coolest, lambda = 0.2 (paper) | 474.3 | 0.518 | 1.093 | 0.286 |
| coolest, lambda = 0.15 | 480.3 | 0.526 | 1.106 | 0.291 |
| coolest, lambda = 0.001 | 528.5 | 0.574 | 1.217 | 0.321 |

The per-edge products (`step4_4_edges_flow_SG.gpkg` with `shade_full`, `shade_bld`, `flow_short`, `flow_cool`,
`flow_orig`; `flow_lam_<lambda>.npy`; `step4_4_od_metrics_SG.csv`) are distributed with the data repository.

## 5. Web tool

`webapp/` contains the generators of the ShadeWalk web tool (MapLibre GL single-file HTML with in-browser
Dijkstra routing, hourly shade layers, facility layers, 3D view and pedestrian flow) and the English build of the
paper version, `webapp/dist/nav_app_paper_en.html.gz` (gunzip to open in a browser). See `webapp/README.md`.

## 6. Path configuration

Inputs are defined by the constants at the top of each script (`OUT`, `BW`, `ST`, `FULL`, `BLD`, `BREM`, `CAT`,
`NA`); they point to the Module 1 rasters, the Module 4b weight layers and the Module 5 network on the original
workstation. Edit them before running.
