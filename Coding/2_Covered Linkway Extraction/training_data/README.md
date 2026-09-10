# Covered-linkway training dataset

1,111 annotated tiles of 1024 x 1024 px at 0.3 m (Singapore, EPSG:3414): 808 training tiles and 303 validation
tiles (7:3 split, fixed). Every tile is identified by its pixel offset in the island image
`SG_google_map_03m_SVY21.tif` (179,096 x 115,025 px): `tile_x<col>_y<row>` means the window starting at
column `col` and row `row`.

| Folder / file | Content |
|---|---|
| `grid/tiles_meta.csv` | one row per tile: `filename, split, col_off, row_off, width, height, transform_a..f, xmin, ymin, xmax, ymax, crs` (2,082 rows: annotated tiles plus blank grid cells) |
| `grid/tiles_fishnet.gpkg` | the same grid as polygons with a `split` field (`train` / `val` / `blank`); `tiles_fishnet_train.gpkg` and `tiles_fishnet_val.gpkg` are the two subsets |
| `labels_json/train/*.json`, `labels_json/val/*.json` | LabelMe (version 6.0) polygon annotations, one file per tile, label `linkway`; tiles without any linkway have an empty `shapes` list |
| `masks/train/*.png`, `masks/val/*.png` | binary masks rasterised from the polygons (uint8, 0 = background, 255 = covered linkway) |

The RGB image tiles are **not** included because the underlying satellite basemap cannot be redistributed. To rebuild
them, cut 1024 x 1024 windows at the offsets given in `grid/tiles_meta.csv` from imagery of the same grid
(`rasterio.windows.Window(col_off, row_off, 1024, 1024)`), or regenerate the grid for your own imagery with
`../data_prep/make_fishnet_v3.py`. Masks can be regenerated from the JSON files with the helper
`generate_masks_from_json()` in `../baseline_sam2unet/cl_pipeline.py`.

Annotation protocol: every roof of a covered walkway (HDB linkways, bus-stop links, covered bridges between blocks)
visible in the imagery was digitised as a polygon; tree-canopy occlusions were left as gaps (they are bridged in
post-processing, see the module README). Buildings, arcades under buildings and uncovered footpaths are background.
