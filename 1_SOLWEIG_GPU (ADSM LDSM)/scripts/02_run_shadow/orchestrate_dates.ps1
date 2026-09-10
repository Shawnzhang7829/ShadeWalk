# Detached orchestrator for the 4 solstice/equinox SOLWEIG shadow runs.
# Per date: driver (tile-wise, restart on death/stall) -> 77 tiles -> merge_dates.py
# (24-band merged + 11:00-15:00 extracts under merge_images\<tag>) -> delete the
# 132 GB tile folder only if _MERGE_OK exists -> next date.  Log: _orchestrate.out
$BASE = "D:\Claude\SVI_FFW\TIF_shadow_newarcade"
$BAT  = "C:\Program Files\QGIS 3.40.15\bin\python-qgis-ltr.bat"
$PY   = "C:\Users\City Syntax Lab\.pyenv\pyenv-win\shims\python.bat"
$LOGD = "D:\Claude\SVI_FFW\output\step3_adsm"
$DRV  = "$LOGD\driver_tilewise_dates.py"
$MRG  = "$LOGD\merge_dates.py"
$OUT  = "$LOGD\_orchestrate.out"
$NT   = 77
function Log($m){ "$(Get-Date -f 'MM-dd HH:mm:ss') $m" | Out-File $OUT -Append -Encoding utf8 }
$DATES = @(
  @{ tag = "2026-03-20_spring_equinox";  date = "2026-03-20"; met = "S50_Clementi Road_2026-03-20_spring_equinox_doy79.txt" },
  @{ tag = "2026-06-21_summer_solstice"; date = "2026-06-21"; met = "S50_Clementi Road_2026-06-21_summer_solstice_doy172.txt" },
  @{ tag = "2026-09-23_autumn_equinox";  date = "2026-09-23"; met = "S50_Clementi Road_2026-09-23_autumn_equinox_doy266.txt" },
  @{ tag = "2026-12-22_winter_solstice"; date = "2026-12-22"; met = "S50_Clementi Road_2026-12-22_winter_solstice_doy356.txt" }
)
Log "orchestrator start (pid $PID)"
foreach ($d in $DATES) {
  $tag = $d.tag
  $tileDir = "$BASE\output_folder_$tag"
  $mergeDir = "$BASE\merge_images\$tag"
  if (Test-Path "$mergeDir\_MERGE_OK") { Log "[$tag] already merged, skip"; continue }
  New-Item -ItemType Directory -Force $tileDir | Out-Null
  $env:SW_DATE = $d.date
  $env:SW_OUT  = $tileDir
  $env:SW_MET  = "$BASE\Forcing_data\$($d.met)"
  Log "[$tag] driver phase: DATE=$($env:SW_DATE) MET=$($d.met)"
  $prev = 0; $stall = 0; $restarts = 0
  for ($i = 0; $i -lt 1440; $i++) {          # up to 12 h per date (30 s ticks; measured ~165 s per tile -> ~3.6 h per date)
    $n = (Get-ChildItem $tileDir -Recurse -Filter "Shadow_*.tif" -ErrorAction SilentlyContinue).Count
    if ($n -ge $NT) { Log "[$tag] ALL $NT tiles done"; break }
    $alive = [bool](Get-CimInstance Win32_Process -Filter "Name='python.exe'" -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*driver_tilewise_dates*" })
    if ($n -gt $prev) { $prev = $n; $stall = 0 } else { $stall++ }
    if ((-not $alive) -or ($stall -ge 30)) {   # dead, or no new tile for 15 min (driver itself kills a tile after 10 min)
      if ($alive) {
        $drv = @(Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $_.CommandLine -like "*driver_tilewise_dates*" })
        $ids = $drv | ForEach-Object { $_.ProcessId }
        Get-CimInstance Win32_Process -Filter "Name='python.exe'" | Where-Object { $ids -contains $_.ParentProcessId } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }   # spawned tile workers
        $drv | ForEach-Object { Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue }
      }
      $restarts++
      Log "[$tag] (re)start driver #$restarts (tiles=$n alive=$alive stall=$stall)"
      Start-Process -FilePath $BAT -ArgumentList "`"$DRV`"" -WindowStyle Hidden -RedirectStandardOutput "$LOGD\_drv_$tag.log" -RedirectStandardError "$LOGD\_drv_$tag.err"
      $stall = 0; Start-Sleep -Seconds 60
    }
    Start-Sleep -Seconds 30
  }
  $n = (Get-ChildItem $tileDir -Recurse -Filter "Shadow_*.tif" -ErrorAction SilentlyContinue).Count
  if ($n -lt $NT) { Log "[$tag] GAVE UP with $n/$NT tiles -> skipping merge"; continue }
  Log "[$tag] merge phase"
  $p = Start-Process -FilePath $PY -ArgumentList "`"$MRG`" `"$tileDir`" `"$mergeDir`"" -WindowStyle Hidden -Wait -PassThru -RedirectStandardOutput "$LOGD\_merge_$tag.log" -RedirectStandardError "$LOGD\_merge_$tag.err"
  if ((Test-Path "$mergeDir\_MERGE_OK") -and ($p.ExitCode -eq 0)) {
    Log "[$tag] merge OK -> deleting tile folder $tileDir"
    Remove-Item -Recurse -Force $tileDir -ErrorAction SilentlyContinue
    Log "[$tag] tile folder removed; free space $([math]::Round((Get-PSDrive D).Free/1GB)) GB"
  } else {
    Log "[$tag] merge FAILED (exit $($p.ExitCode)); tile folder kept"
  }
}
Log "orchestrator exit"
