@echo off
REM ============================================================
REM  Covered Linkway - full-island reproduction (final model)
REM  1. streaming inference  2. vectorise  3. pedestrian-network filter
REM  4. footpath-corridor bridging (edit constants in the script)
REM  Edit the four <...> paths below before running (environment C, conda "sam2").
REM ============================================================
setlocal
set PY=python
set HERE=%~dp0
set IMG=<path to SG_google_map_03m_SVY21.tif>
set SAM=<path to sam_vit_h_4b8939.pth>
set BOUND=<path to Island_boarder.shp>
set BLDG=<path to SG_Building_SVY21.gpkg>
set PEDNET=<path to pedestrian_network_filtered.gpkg>
set OUT=%HERE%out
if not exist "%OUT%" mkdir "%OUT%"

echo.
echo [STEP 1/4] Full-island inference (about 3.5 h on an RTX 6000 Ada)
%PY% "%HERE%inference\inference_autonomous_island.py" ^
  --input_tif "%IMG%" ^
  --sam_ckpt "%SAM%" ^
  --trained_ckpt "%HERE%checkpoints\geosam_topolora_autonomous_lean_dice0.7138.pth" ^
  --text_emb "%HERE%checkpoints\clip_linkway_emb.pth" ^
  --output_tif "%OUT%\covered_linkway_SG_island_tv.tif" ^
  --boundary_shp "%BOUND%" ^
  --building_gpkg "%BLDG%" ^
  --threshold 0.5
if errorlevel 1 goto :err

echo.
echo [STEP 2/4] Vectorise + min area 25 m2 (about 4 min)
%PY% "%HERE%postprocess\vectorise_generic.py" --src_tif "%OUT%\covered_linkway_SG_island_tv.tif" --out_gpkg "%OUT%\covered_linkway_SG_island_tv.gpkg" --min_area_m2 25
if errorlevel 1 goto :err

echo.
echo [STEP 3/4] Pedestrian-network 5 m soft constraint (about 12 min)
%PY% "%HERE%postprocess\pednet_filter_generic.py" --src_gpkg "%OUT%\covered_linkway_SG_island_tv.gpkg" --src_tif "%OUT%\covered_linkway_SG_island_tv.tif" ^
  --out_gpkg "%OUT%\covered_linkway_SG_island_tv_pednet.gpkg" --out_tif "%OUT%\covered_linkway_SG_island_tv_pednet.tif" --buffer 5 --pednet "%PEDNET%"
if errorlevel 1 goto :err

echo.
echo [STEP 4/4] Footpath-corridor bridging: set SRC_TIF / PEDN / OUT_TIF / OUT_GPKG in postprocess\footpath_bridge_island.py, then run
%PY% "%HERE%postprocess\footpath_bridge_island.py"
if errorlevel 1 goto :err

echo.
echo DONE. Final product: %OUT%\covered_linkway_SG_island_tv_pednet_bridged.tif / .gpkg
goto :eof

:err
echo.
echo *** A step failed (errorlevel %errorlevel%). Check the log above. ***
exit /b 1
