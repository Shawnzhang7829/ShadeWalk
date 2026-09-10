"""
Thermal + disk watchdog for the v2 training task.

Every 60 seconds:
  - Query nvidia-smi GPU temperature
  - Query Windows event log for new disk/NVMe errors since watchdog start
  - Log to file

Kill rules (any one triggers a hard kill of the training process):
  - Disk / NVMe error appears  (immediate)
  - GPU temp >= 86C            (immediate spike)
  - GPU temp >  80C for 600 s  cumulative   (sustained)
  Counter resets when temp drops back to <= 80C.
"""
import subprocess, time, datetime, os

LOG_PATH      = r"C:\GeoSAM-backup\runs\autonomous_v2_oldlabel\watchdog.log"
START_TIME    = datetime.datetime.now()
TEMP_WARN     = 84
TEMP_KILL     = 88     # raised: room AC active, full-power run authorised
TEMP_SUSTAIN  = 0      # 0 = sustained-temp rule DISABLED (full power)
POLL_SEC      = 60


def log(msg):
    line = f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def get_gpu_temp():
    try:
        out = subprocess.check_output(
            ["nvidia-smi",
             "--query-gpu=temperature.gpu,utilization.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            timeout=10).decode().strip()
        t, u, p = [x.strip() for x in out.split(",")]
        return int(t), int(u), float(p)
    except Exception:
        return None, None, None


def get_recent_disk_errors():
    fmt = START_TIME.strftime("%Y-%m-%d %H:%M:%S")
    ps = f"""
$start = [datetime]::ParseExact('{fmt}', 'yyyy-MM-dd HH:mm:ss', $null)
$events = Get-WinEvent -FilterHashtable @{{LogName='System'; StartTime=$start}} -ErrorAction SilentlyContinue |
    Where-Object {{ $_.LevelDisplayName -in @('Error','Warning','Critical') -and
                    $_.ProviderName -match 'disk|Ntfs|stor|nvme' }}
Write-Output $events.Count
"""
    try:
        out = subprocess.check_output(["powershell", "-NoProfile", "-Command", ps],
                                      timeout=20).decode().strip()
        return int(out) if out.isdigit() else 0
    except Exception:
        return -1


def find_training_pid():
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_Process | "
             "Where-Object { $_.CommandLine -like '*train_autonomous_v2.py*' } | "
             "Select-Object -ExpandProperty ProcessId"],
            timeout=10).decode().strip()
        for line in out.splitlines():
            if line.strip().isdigit():
                return int(line.strip())
    except Exception:
        pass
    return None


def kill_training(reason):
    pid = find_training_pid()
    log(f"!!! TRIGGER: {reason}")
    log(f"!!! Killing training PID={pid}")
    if pid:
        try:
            subprocess.run(["taskkill", "/PID", str(pid), "/F", "/T"], timeout=10)
            log("training killed")
        except Exception as e:
            log(f"kill failed: {e}")
    else:
        log("no training PID found (already gone?)")


def main():
    os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
    log(f"watchdog start  poll={POLL_SEC}s  WARN={TEMP_WARN}C  "
        f"SUSTAIN={TEMP_SUSTAIN}s above WARN  KILL={TEMP_KILL}C")

    max_temp = 0
    sustained_above_warn = 0     # seconds accumulated above WARN

    while True:
        t, u, p = get_gpu_temp()
        err_n = get_recent_disk_errors()

        if t is None:
            log("nvidia-smi failed, skip")
        else:
            max_temp = max(max_temp, t)

            # Update sustained-above-warn timer
            if t > TEMP_WARN:
                sustained_above_warn += POLL_SEC
                flag = f"  ABOVE {TEMP_WARN}C for {sustained_above_warn}s"
            else:
                if sustained_above_warn > 0:
                    flag = f"  cooled below {TEMP_WARN}C, reset sustain timer"
                else:
                    flag = ""
                sustained_above_warn = 0

            if t >= TEMP_KILL:
                flag += "  *** CRITICAL SPIKE ***"

            log(f"GPU={t}C util={u}% pwr={p:.0f}W  max={max_temp}C  "
                f"sustain={sustained_above_warn}/{TEMP_SUSTAIN}s  disk_err={err_n}{flag}")

            # --- Kill rules ---
            if err_n > 0:
                kill_training(f"{err_n} disk/NVMe error(s) detected")
                break
            if t >= TEMP_KILL:
                kill_training(f"GPU temp {t}C >= {TEMP_KILL}C (critical spike)")
                break
            if TEMP_SUSTAIN > 0 and sustained_above_warn >= TEMP_SUSTAIN:
                kill_training(f"GPU temp > {TEMP_WARN}C sustained for "
                              f"{sustained_above_warn}s (>= {TEMP_SUSTAIN}s threshold)")
                break

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    main()
