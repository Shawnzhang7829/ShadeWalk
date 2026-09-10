# Module 1 - input data inventory

All rasters must share one grid (same CRS, resolution, extent and pixel alignment); the package checks this in
`preprocessor.ppr()`. For Singapore the grid is 1 m, EPSG:3414 (SVY21), 44,000 x 27,000 px,
bounds 2938.6 / 23953.5 / 46938.6 / 50953.5 (E / S / W / N in metres). File names follow the convention
`SUB_<city>_Polygon_<layer>_<n>m.tif`.

## A. Base rasters (external inputs)

| Layer | File (Singapore) | Content | Source / how produced |
|---|---|---|---|
| DEM | `SUB_SG_Polygon_DEM_1m.tif` | bare-earth elevation, float32, nodata 0 | ALOS PALSAR radiometrically terrain-corrected DEM (12.5 m, ASF DAAC), resampled to the 1 m city grid (land mask = DEM > 0 is used for statistics); released as `SG_DEM_1m.tif` in the data record |
| CDSM | `SUB_SG_Polygon_CDSM_1m.tif` | tree-canopy height above ground | Meta 1 m global canopy height map (Tolan et al., 2024), clipped and aligned to the 1 m grid; pixels without trees are nodata (-3.4e38) and are cleaned to 0 by `prepare_newarcade.py` -> `SUB_SG_Polygon_CDSMclean_1m.tif` (released as `SG_CDSM_tree_height_1m.tif`) |
| Meteorological forcing | `Forcing_data/S50_Clementi Road*.txt` | hourly UMEP-format forcing (air temperature, RH, pressure, wind, Kdown, ...) | Meteorological Service Singapore station S50 (Clementi Road); five files included in `sample_data/forcing`: 2026-03-01 (paper run), 2026-03-20, 06-21, 09-23, 12-22 (equinox / solstice runs) |

## B. Vectors from the other modules (rasterised by `scripts/01_prepare_rasters`)

| Vector | Fields used | Produced by | Rasterised to |
|---|---|---|---|
| `step2_building_remain_sg.gpkg` | `height` (m above ground) | Module 3b (`step2_buffer_split.py`) - building footprints minus the arcade strip | `SUB_SG_Polygon_DSMremain_1m.tif` (= DEM + height) and `SUB_SG_Polygon_BREMAIN_1m.tif` (0/1 mask) |
| `step2_arcade_sg.gpkg` | `bld_h` (building height), `arc_h` (arcade ceiling height, <= 3.6 m and <= `bld_h`) | Module 3b | `SUB_SG_Polygon_ADSM_1m.tif` (top = `bld_h`) and `SUB_SG_Polygon_ADSMB_1m.tif` (base = min(`arc_h`, max(top - 0.5, 0.5))) |
| `covered_linkway_SG_island_tv_pednet_bridged.gpkg` | geometry only | Module 2 (final covered-linkway polygons) | `SUB_SG_Polygon_LDSMpednet_1m.tif` (constant 3.0 m) |
| `4000pixel_polygon_SVY21.shp` | geometry | test-area polygon (4000 x 3974 px window, bounds 29041.6 / 30822.5 / 33041.6 / 34796.5) | window for the test-area workflow |

Rasterisation uses the pixel-centre rule (`gdal.RasterizeLayer`, `ATTRIBUTE=` or a constant burn value) so that the
masks match the vectors pixel for pixel; the arcade strip is not part of `building_remain`, therefore the ground
under the floating box is at DEM level without any additional carving.

## C. Derived rasters produced inside this module

| Raster | Produced by | Used as |
|---|---|---|
| `SUB_SG_Polygon_WALLS0_1m.tif`, `SUB_SG_Polygon_ASPECT0_1m.tif` | `prepare_newarcade.py` (all zeros) | `wallheight_filename`, `wallaspect_filename` in shadow-only mode |
| real wall height / aspect (only for Tmrt / UTCI) | `step3b_walls_parallel.py <DSM> <walls_out> <aspect_out> 12` | optional |
| `processed_inputs/<layer>/<layer>_<x>_<y>.tif` | `preprocessor.ppr()` (tiling, 4000 px + 100 px halo) | reused by the tile-wise driver when `reuse_tiles=True` |
| `processed_inputs/metfiles/metfile_<x>_<y>.txt` | `preprocessor.ppr()` | per-tile forcing (identical copies) |

## D. Test area

`sample_data/test_area_4000px/` is not shipped in the repository (binary rasters). The test-area rasters are the
same layers as above clipped to the `4000pixel_polygon_SVY21` window (file names `SG_<layer>_1m_test.tif`); they are
regenerated with `step3t_rebuild_aligned.py` from the city rasters and run with
`scripts/02_run_shadow/step3c_b10_final.py`.

## E. Data not included and where to obtain it

| Item | Reason | Where |
|---|---|---|
| whole-city rasters (DEM, CDSM, DSMremain, ADSM, ADSMB, LDSMpednet, BREMAIN; 0.5-1 GB each) | size | data repository accompanying the paper (see the top-level README) |
| tile outputs `output_folder/` (about 132 GB per date) and merged 24-band rasters (0.5-2 GB each) | size | 14:00 extracts are available from the data repository |
| SOLWEIG-GPU sample data (New Delhi example) | upstream | https://doi.org/10.5281/zenodo.18561860 |
