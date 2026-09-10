# Software environments

The pipeline was developed and run on one Windows 11 workstation (NVIDIA RTX 6000 Ada, 48 GB VRAM, driver 572.42; 12 CPU cores used for the parallel steps). Three separate Python environments were used because the deep-learning stacks and the GIS stacks have incompatible dependency sets. Each module README states which environment it needs.

## Environment A - `geo` (Python 3.11.9, pyenv-win)

Used by: raster preparation in Module 1, Module 3b (projection), Module 4a (edge shade, routing, flows), Module 4b (madina flows), Module 5 (network reconstruction).

| Package | Version |
|---|---|
| numpy | 2.4.4 |
| pandas | 3.0.2 |
| pyarrow | 24.0.0 |
| geopandas | 1.1.3 |
| shapely | 2.1.2 |
| pyproj | 3.7.2 |
| fiona | 1.10.1 |
| pyogrio | 0.12.1 |
| rasterio | 1.4.4 |
| networkx | 3.6.1 |
| scipy | 1.17.1 |
| scikit-learn | 1.8.0 |
| matplotlib | 3.10.9 |
| joblib | 1.5.3 |
| pillow | 12.2.0 |
| tqdm | 4.67.3 |
| madina | 0.0.15 (editable install of the upstream repository, see Module 4b) |

Install (example):

```bash
pip install numpy pandas pyarrow geopandas shapely pyproj fiona pyogrio rasterio networkx scipy scikit-learn matplotlib joblib pillow tqdm
```

## Environment B - `qgis-torch` (QGIS 3.40.15 LTR, Python 3.12)

Used by: Module 1 (SOLWEIG-GPU runs, needs GDAL and the UMEP processing provider available from the QGIS Python) and Module 3a (CLIP ViT-L/14 embedding through Hugging Face transformers). Scripts for this environment are launched with `run_qgis.bat` (Module 1) or `python-qgis-ltr.bat`, which put the QGIS `bin` and `apps/Python312` folders on `PATH` and set `GDAL_DATA`, `GDAL_DRIVER_PATH` and `PROJ_LIB`.

| Package | Version |
|---|---|
| GDAL | 3.12.1 |
| numpy | 1.26.4 |
| pandas | 2.2.3 |
| torch | 2.7.1+cu118 |
| torchvision | 0.22.1+cu118 |
| transformers | 5.10.2 |
| rasterio | 1.4.3 |
| geopandas | 1.0.1 |
| shapely | 2.0.6 |
| scipy | 1.15.1 |
| scikit-learn | 1.6.1 |
| joblib | 1.5.3 |
| pillow | 11.1.0 |
| netCDF4 / xarray | 1.7.4 / 2026.2.0 |
| supy | 2026.1.28rc1 |

## Environment C - `sam2` (conda, Python 3.10.20)

Used by: Module 2 (GeoSAM + TopoLoRA training, inference, post-processing and evaluation; SAM2-UNet baseline).

| Package | Version |
|---|---|
| torch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| segment-anything | 1.0 (Meta, Apache-2.0) |
| SAM-2 | 1.0 (Meta `sam2` repository, editable install; baseline only) |
| clip | 1.0 (OpenAI CLIP, MIT) |
| numpy | 1.26.4 |
| rasterio | 1.4.4 |
| geopandas | 1.1.3 |
| shapely | 2.1.2 |
| pyproj | 3.7.1 |
| fiona / pyogrio | 1.10.1 / 0.12.1 |
| scikit-image | 0.25.2 |
| scipy | 1.15.3 |
| opencv-python | (any 4.x) |
| pillow | 12.1.1 |

Install (example):

```bash
conda create -n sam2 python=3.10
conda activate sam2
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install git+https://github.com/facebookresearch/segment-anything.git
pip install git+https://github.com/openai/CLIP.git
pip install numpy==1.26.4 rasterio geopandas shapely pyproj fiona pyogrio scikit-image scipy opencv-python pillow
```

## External model weights (not included, download separately)

| Weight | Used by | Source |
|---|---|---|
| `sam_vit_h_4b8939.pth` (2.56 GB) | Module 2 | https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth |
| `sam2.1_hiera_large.pt` | Module 2 baseline only | https://github.com/facebookresearch/sam2 |
| CLIP ViT-B/32 (OpenAI `clip` package) | Module 2 text embedding (cached copy included as `checkpoints/clip_linkway_emb.pth`) | downloaded automatically by the `clip` package |
| `openai/clip-vit-large-patch14` | Module 3a | downloaded automatically by Hugging Face `transformers` |
