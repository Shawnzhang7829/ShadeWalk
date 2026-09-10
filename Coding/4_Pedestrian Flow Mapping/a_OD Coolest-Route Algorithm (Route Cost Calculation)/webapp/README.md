# ShadeWalk web tool

Single-file web application (MapLibre GL + three.js, all data inlined as base64 / typed arrays) that lets a user
pick an origin and a destination anywhere in Singapore and compares the shortest route with the coolest route
computed in the browser (Dijkstra on the main component of the reconstructed network, edge cost
`length x ((1 - shade) + lambda)`, shade reward slider for lambda). It also shows the hourly shade layers, the
shade facilities (arcades, covered linkways, trees, buildings), the segment cost `omega_i`, a route-level and a
street-level topology graph, per-edge pedestrian flow (shortest / coolest / original network) and a 3D view with a
pedestrian-perspective walk-through.

## Distributed build

`dist/nav_app_paper_en.html.gz` - the English "paper routing" build (about 129 MB uncompressed; gzip-compressed for the
repository). Decompress and open it in a modern browser:

```bash
gunzip -k "dist/nav_app_paper_en.html.gz"
```

The page needs internet access for the MapLibre / three.js libraries and the basemap tiles (CARTO, OneMap,
OpenStreetMap or Google satellite); all ShadeWalk data are embedded. `dist/demo_scenarios.json` holds the
demonstration origin-destination pairs shown in the "Demo" panel.

## Generators

| Script | Purpose |
|---|---|
| `make_nav_app_maplibre.py` | builds `nav_app.html` from the Module 4a / 5 products: network edges and nodes, edge shade / facility / flow arrays (`flow_cooltau_*.npy`, `flow_lam_*.npy`), arcade / linkway / building rings, tree points, station ridership, hourly shade layers (`make_hourly_cache.py`), demo OD pairs |
| `make_paper_version.py` | post-processes `nav_app.html` into the paper build: replaces the expert-weighted topology metrics by the paper definition (`rho = (1 - shade) + lambda`, `omega = sum(l * rho)`, lambda = 0.2 with slider) |
| `make_en_navapp.py` | produces the English UI (`nav_app_en.html`, `nav_app_paper_en.html`) from the Chinese build by ordered string replacement and reports any untranslated remainder |
| `make_nav_app_mobile.py` | slim mobile build (routing core + three hourly shade frames) |
| `make_hourly_cache.py` | 10 m hourly shade / category layers as base64 PNG (parallel, resumable) |
| `make_tiles.py`, `tile_rasters.py` | XYZ tile pyramids (z11-z18) of the tree canopy, 14:00 shade and 14:00 category rasters for the local tile server variant |
| `step4_demo_scenarios.py`, `step4_merge_saved_demo.py` | choose demonstration OD pairs (HDB -> bus, office -> MRT, ...) and merge user-exported favourites |

The generator `make_nav_app_maplibre.py`, `make_paper_version.py` and `make_en_navapp.py` contain the original
Chinese UI strings of the bilingual application by design (the English page is derived from them); they are kept
verbatim so that the shipped HTML can be regenerated exactly. All other scripts of this repository are English.

Inputs are defined by the constants at the top of `make_nav_app_maplibre.py` (`OUT`, `WEB`, `ARC`, `LKW`, `BLDG`,
`CDSM`, `TREE3`, `STA`) and are listed in `../INPUT_DATA.md`.
