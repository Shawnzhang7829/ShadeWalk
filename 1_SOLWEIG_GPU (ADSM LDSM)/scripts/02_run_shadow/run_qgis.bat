@echo off
REM Run python in the QGIS/OSGeo4W environment: gdal (bin on PATH) + torch from the user site-packages
set "QR=C:\Program Files\QGIS 3.40.15"
set "PATH=%QR%\bin;%QR%\apps\Python312;%QR%\apps\Python312\Scripts;%PATH%"
set "GDAL_DATA=%QR%\apps\gdal\share\gdal"
set "GDAL_DRIVER_PATH=%QR%\apps\gdal\lib\gdalplugins"
set "PROJ_LIB=%QR%\share\proj"
set PYTHONIOENCODING=utf-8
set HF_HUB_DISABLE_PROGRESS_BARS=1
"%QR%\apps\Python312\python.exe" %*
