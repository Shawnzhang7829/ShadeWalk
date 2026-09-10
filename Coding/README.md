# ShadeWalk - pipeline code

One folder per module, numbered in pipeline order. The repository-level documentation (overview, workflow figure,
environments, verification, citation) is in the [top-level README](../README.md).

| Folder | Module |
|---|---|
| [`1_SOLWEIG_GPU (ADSM LDSM)`](<1_SOLWEIG_GPU (ADSM LDSM)/README.md>) | SOLWEIG-GPU shadow model with covered-linkway (LDSM) and arcade (ADSM / ADSMB) layers |
| [`2_Covered Linkway Extraction`](<2_Covered Linkway Extraction/README.md>) | GeoSAM + TopoLoRA segmentation of covered linkways, training dataset and checkpoint |
| [`3_Arcade Extraction and Projection`](<3_Arcade Extraction and Projection/README.md>) | a: arcade detection in street-view images; b: projection onto building footprints |
| [`4_Pedestrian Flow Mapping`](<4_Pedestrian Flow Mapping/README.md>) | a: coolest-route cost model, flows and web tool; b: madina patronage-betweenness flow assignment |
| [`5_OSM Network Reconstruction`](<5_OSM Network Reconstruction/README.md>) | pedestrian network reconstruction with facility centrelines |
| `docs/` | workflow figure, software environments |
| `PROVENANCE.md` | origin of every file and the list of intentional edits |
