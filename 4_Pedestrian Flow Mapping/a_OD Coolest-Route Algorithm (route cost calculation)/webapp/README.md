# ShadeWalk web tool

Single-file web application (MapLibre GL + three.js, all data inlined as base64 / typed arrays) that lets a user
pick an origin and a destination anywhere in Singapore and compares the shortest route with the coolest route
computed in the browser (Dijkstra on the main component of the reconstructed network, edge weight
`omega_e = l_e x ((1 - sigma_e) + lambda)`, shade reward slider for lambda). It also shows the hourly shade layers (Daily time slider 08-18: the network shade colouring, the coolest routing and the route cards follow the hour, the flow fields stay at 14:00), the
shade facilities (arcades, covered linkways, trees, buildings), the edge weight `omega_e`, a route-level and a
street-level topology graph, per-edge pedestrian flow (shortest / coolest / original network), the 14:00 station-to-building
OD pairs of the demand model (OD flow) and a 3D view with a pedestrian-perspective walk-through.

Further controls: Demo scenarios (origin type x destination type, HDB / Office / Transit / Dining / Store, five
pre-selected pairs per combination); Basemap > Subzone (Master Plan 2019 subzone boundaries and names over any basemap);
Network view > Inspect edge (while on, a click on the network selects the nearest edge and shows its length, its shade
`sigma_e` at the Daily time hour, `rho_e = (1 - sigma_e) + lambda`, `omega_e = l_e * rho_e` and its 14:00 shortest / coolest
pedestrian flow, instead of setting an origin or destination). Heavy layers (OD lines, 3D extrusions) are removed from
the map when the user leaves them and rebuilt on return, and the 3D arcade columns are generated for the current view only,
which keeps the memory of the page bounded when switching between views.

## Distributed build

`dist/nav_app_paper_en.html.gz` - the English "paper routing" build (about 143 MB uncompressed; gzip-compressed for the
repository). Decompress and open it in a modern browser:

```bash
gunzip -k "dist/nav_app_paper_en.html.gz"
```

The page needs internet access for the MapLibre / three.js libraries and the basemap tiles (Esri World Light Gray Canvas, OpenFreeMap Positron, OneMap
or Google satellite); all ShadeWalk data are embedded. `dist/demo_scenarios.json` holds the
demonstration origin-destination pairs of the "Demo scenarios" panel (25 origin-type x destination-type combinations, up to
five pairs each, all within one mile and with shortest and coolest routes on different streets).

## Generators

| Script | Purpose |
|---|---|
| `make_nav_app_maplibre.py` | builds `nav_app.html` from the Module 4a / 5 products: network edges and nodes, edge shade / facility / flow arrays (`flow_cooltau_*.npy`, `flow_lam_*.npy`), arcade / linkway / building rings, tree points, station ridership, hourly shade layers (`make_hourly_cache.py`), hourly edge shade (`edge_shade_hourly_SG.npz`, embedded as base64 uint8 layers), demo OD pairs, OD flow pairs (`od_pairs_app.npz`), subzone overlay (`subzone_data.py`) |
| `make_paper_version.py` | post-processes `nav_app.html` into the paper build: replaces the expert-weighted topology metrics by the paper definition (`rho = (1 - shade) + lambda`, `omega = sum(l * rho)`, lambda = 0.2 with slider) |
| `make_en_navapp.py` | produces the English UI (`nav_app_en.html`, `nav_app_paper_en.html`) from the Chinese build by ordered string replacement and reports any untranslated remainder |
| `make_nav_app_mobile.py` | slim mobile build (routing core + three hourly shade frames) |
| `make_hourly_cache.py` | 10 m hourly shade / category layers as base64 PNG (parallel, resumable) |
| `make_tiles.py`, `tile_rasters.py` | XYZ tile pyramids (z11-z18) of the tree canopy, 14:00 shade and 14:00 category rasters for the local tile server variant |
| `od_export_app.py` | OD table of the 14:00 demand model for the OD flow layer (`od_pairs_app.npz`): station -> building pairs with Huff-allocated persons, line ends on the web tool's building centroids |
| `subzone_data.py` | Subzone overlay data (Master Plan 2019 subzone rings in page metres, label points), embedded by `make_nav_app_maplibre.py` |
| `step4_demo_scenarios.py`, `step4_merge_saved_demo.py` | choose the demonstration OD pairs (`_demo_scenarios_od25.json`: 25 origin-type x destination-type combinations, pairs within one mile whose shortest and coolest routes differ, checked against both route definitions of the page) and merge user-exported favourites |

The generator `make_nav_app_maplibre.py`, `make_paper_version.py` and `make_en_navapp.py` contain the original
Chinese UI strings of the bilingual application by design (the English page is derived from them); they are kept
verbatim so that the shipped HTML can be regenerated exactly. All other scripts of this repository are English.

Inputs are defined by the constants at the top of `make_nav_app_maplibre.py` (`OUT`, `WEB`, `ARC`, `LKW`, `BLDG`,
`CDSM`, `TREE3`, `STA`) and are listed in `../INPUT_DATA.md`.
