# Module 4a - Coolest-route algorithm and route-cost calculation

Turns the reconstructed pedestrian network of Module 5 into a routing graph with a 14:00 shade fraction on every
edge (from the shadow rasters of Module 1), defines the street resistance and the edge cost of the coolest-route
rule, attributes the shade of every edge to its source (building, tree, arcade, covered linkway) and generates the
ShadeWalk web tool (`webapp/`), whose in-browser Dijkstra uses the same cost. The city-wide OD assignment and the
flow products on these costs are Module 4b (`routing_flow/`). Environment A.

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
sun. The lambda sweep (18 values, 3.0 ... 0.001) and the alternative formulation with a hard detour cap tau are run
by the flow scripts of Module 4b (`step4_4f_flow_lam.py`, `step4_4e_flow_detour.py`).

## 2. Edge shade

`sigma_i` is sampled every 2 m along the edge from the full-system 14:00 shadow raster of Module 1 (buildings, trees,
covered linkways, arcades; 1 = sunlit, 0 = shaded) and, for the building-only baseline, from a building-only shadow
raster computed here; edges through building footprints count as fully shaded (indoor). The hourly variant samples the
24-band raster for every hour 08-18 and feeds the Daily-time slider of the web tool. The dominant shade source of every
edge (arcade > linkway > tree > building) and the per-metre attribution of shade to its sources come from the 14:00
shadow-category raster.

## 3. Scripts

| Step | Script | Purpose |
|---|---|---|
| 1 | `scripts/01_prep/make_prep.py` | read `step4_network_final.gpkg` (Module 5), cluster endpoints (0.5 m), label connected components (`comp`, main = 0) and map `src` to `footpath` (osm, osmlink) / `shade` (arcade, linkway, facconn) / `bridge` (fac2osm, gapfill) |
| 2 | `scripts/02_edge_shade/step4_4a_city_building_shadow.py` | LOD1 baseline: 14:00 shadow of buildings only (numpy port of the SOLWEIG shadow function, tiled, 13:30 sun position, 120 px margin) |
| 2b | `step4_4a_bldtree_shadow.py` | buildings + trees baseline (no facilities), vegetation as a separate opaque canopy |
| 3 | `step4_4b_city_edge_shade.py` | per-edge shade fraction from the full-system 14:00 shadow raster (Module 1) and from the building-only raster, sampled every 2 m; edges through building footprints count as fully shaded; builds the node / edge graph (`u`, `v`) |
| 3c | `step4_4b_city_edge_shade_hourly.py` | per-edge shade fraction for every hour 08-18 from the 24-band shadow raster (band = hour + 1; the same 2 m sampling and building-interior rule as 3, the 14:00 layer is checked against `shade_full`), `edge_shade_hourly_SG.npz`; feeds the Daily-time slider of the web app (network shade colouring, coolest routing and route cards follow the hour) |
| 3b | `step4_4b_esn.py`, `step4_edge_facility.py`, `step4_edge_class_1m.py`, `step4_network_split_1m.py`, `edge_px_permetre.py` | no-facility shade per edge, dominant facility class per edge (arcade > linkway > tree > building), per-metre attribution of shade sources, split of the network by shade class |

The graph (`step4_4_edges_SG.gpkg`, `step4_4_nodes_SG.gpkg`) and the per-edge facility class (`edge_facility_SG.npy`)
are the inputs of the routing / flow scripts of Module 4b.

Runtime: edge shade about 10 min (two 44,000 x 27,000 rasters in memory as uint8).

Verification of the released scripts (2026-09-10, updated 2026-09-17): `make_prep.py` and `step4_4b_city_edge_shade.py`
were re-run on the published network (469,434 edges, 21 s + 91 s). The graph is identical to the production edge layer
(component labels, `u` / `v`). On 2026-09-10 the shade fractions agreed on 99.1 % of the edges (`shade_full`) and 97.6 %
(`shade_bld`): the production layer of June had been sampled with the building-footprint mask of the previous arcade data
set (the mask on disk had been updated together with the arcade shadows). On 2026-09-17 the production layer was regenerated
with the released script and the rasters on disk; it is now identical, on every edge, to the corrected per-edge shade used by the
paper chain, and the routing / flow products (Module 4b), the hourly edge shade and the web app were recomputed from it.

## 4. Web tool

`webapp/` contains the generators of the ShadeWalk web tool (MapLibre GL single-file HTML with in-browser
Dijkstra routing on the edge cost above, hourly shade layers, facility layers, 3D view and the pedestrian flows of
Module 4b) and the English build of the paper version, `webapp/dist/nav_app_paper_en.html.gz` (gunzip to open in a
browser). See `webapp/README.md`.

## 5. Path configuration

Inputs are defined by the constants at the top of each script (`OUT`, `FULL`, `BLD`, `BREM`, `CAT`, `NA`); they point
to the Module 1 rasters and the Module 5 network on the original workstation. The web-tool generator also reads the
flow products and the station / building tables of Module 4b. Edit the constants before running.
