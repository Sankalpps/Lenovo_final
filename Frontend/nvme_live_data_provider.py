import json
import queue
import subprocess
import threading
import time
from typing import Dict, Optional


class LiveTelemetryError(RuntimeError):
    pass


class LiveDataProvider:
    """
    Background SMART telemetry reader for NVMe drives.

    Simulator schema produced:
    - Power_On_Hours
    - Total_TBW_TB
    - Total_TBR_TB
    - Temperature_C
    - Percent_Life_Used
    - Media_Errors
    - Unsafe_Shutdowns
    - CRC_Errors
    - Read_Error_Rate
    - Write_Error_Rate
    """

    SCHEMA_FIELDS = [
        "Power_On_Hours",
        "Total_TBW_TB",
        "Total_TBR_TB",
        "Temperature_C",
        "Percent_Life_Used",
        "Media_Errors",
        "Unsafe_Shutdowns",
        "CRC_Errors",
        "Read_Error_Rate",
        "Write_Error_Rate",
    ]

    def __init__(self, device: Optional[str] = None, interval_sec: float = 1.0):
        self.device = device
        self.interval_sec = float(interval_sec)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, float]] = None
        self._latest_source = "none"
        self._error: Optional[str] = None
        self._samples: "queue.Queue[Dict[str, float]]" = queue.Queue(maxsize=1)
        self._est_read_tb = 0.0
        self._est_write_tb = 0.0
        self._est_last_ts = time.time()

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="LiveDataProvider")
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_latest(self, timeout: float = 2.5) -> Dict[str, float]:
        with self._lock:
            if self._latest is not None:
                return dict(self._latest)
            last_error = self._error

        try:
            sample = self._samples.get(timeout=timeout)
            return dict(sample)
        except queue.Empty as exc:
            raise LiveTelemetryError(last_error or "Timed out waiting for live telemetry") from exc

    def get_status(self) -> Dict[str, str]:
        with self._lock:
            return {
                "source": self._latest_source,
                "error": self._error or "",
            }

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload, source = self._read_smart_json()
                mapped = self._map_to_simulator_schema(payload)

                with self._lock:
                    self._latest = mapped
                    self._latest_source = source
                    self._error = None

                if self._samples.full():
                    try:
                        self._samples.get_nowait()
                    except queue.Empty:
                        pass
                self._samples.put_nowait(mapped)
            except Exception as exc:
                with self._lock:
                    self._error = str(exc)

            self._stop_event.wait(self.interval_sec)

    def _read_smart_json(self):
        device_candidates = [
            self.device,
            "/dev/nvme0n1",
            "/dev/nvme0",
            "/dev/sda",
            "/dev/PhysicalDrive0",
        ]
        device_candidates = [d for d in device_candidates if d]

        for device in device_candidates:
            try:
                proc = subprocess.run(
                    ["smartctl", "-j", "-a", device],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
            except FileNotFoundError:
                break
            if proc.stdout.strip():
                try:
                    return json.loads(proc.stdout), f"smartctl:{device}"
                except json.JSONDecodeError:
                    continue

        if self.device:
            nvme_devices = [self.device]
        else:
            nvme_devices = ["/dev/nvme0n1", "/dev/nvme0"]

        for device in nvme_devices:
            try:
                proc = subprocess.run(
                    ["nvme", "smart-log", "-o", "json", device],
                    capture_output=True,
                    text=True,
                    timeout=8,
                    check=False,
                )
            except FileNotFoundError:
                break
            if proc.stdout.strip():
                try:
                    parsed = json.loads(proc.stdout)
                    return {"nvme_smart_health_information_log": parsed}, f"nvme-cli:{device}"
                except json.JSONDecodeError:
                    continue

        windows_payload = self._read_windows_fallback_json()
        if windows_payload:
            return windows_payload, "powershell-fallback"

        raise LiveTelemetryError("smartctl/nvme-cli output unavailable. Install smartmontools or nvme-cli and ensure drive path is accessible.")

    def _read_windows_fallback_json(self):
        script = r"""
        $disk = Get-PhysicalDisk | Select-Object -First 1 FriendlyName,HealthStatus,Size
        if (-not $disk) {
            throw "No physical disk found"
        }

        $os = Get-CimInstance Win32_OperatingSystem | Select-Object -First 1 LastBootUpTime
        $uptime = if ($os) { ((Get-Date) - $os.LastBootUpTime).TotalHours } else { 0 }

        $read = (Get-Counter '\PhysicalDisk(_Total)\Disk Read Bytes/sec' -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue
        $write = (Get-Counter '\PhysicalDisk(_Total)\Disk Write Bytes/sec' -ErrorAction SilentlyContinue).CounterSamples[0].CookedValue

        $rel = $null
        try {
            $rel = Get-PhysicalDisk | Get-StorageReliabilityCounter -ErrorAction SilentlyContinue | Select-Object -First 1 Temperature,Wear,ReadErrorsTotal,WriteErrorsTotal,PowerOnHours,UnsafeShutdownCount
        } catch {
            $rel = $null
        }

        [PSCustomObject]@{
            health_status = [string]$disk.HealthStatus
            size_bytes = [double]$disk.Size
            uptime_hours = [double]$uptime
            read_bps = [double]$read
            write_bps = [double]$write
            power_on_hours = if ($rel -and $rel.PowerOnHours) { [double]$rel.PowerOnHours } else { [double]$uptime }
            temperature_c = if ($rel -and $rel.Temperature) { [double]$rel.Temperature } else { $null }
            wear_used = if ($rel -and $rel.Wear) { [double]$rel.Wear } else { $null }
            read_errors = if ($rel -and $rel.ReadErrorsTotal) { [double]$rel.ReadErrorsTotal } else { 0 }
            write_errors = if ($rel -and $rel.WriteErrorsTotal) { [double]$rel.WriteErrorsTotal } else { 0 }
            unsafe_shutdowns = if ($rel -and $rel.UnsafeShutdownCount) { [double]$rel.UnsafeShutdownCount } else { 0 }
        } | ConvertTo-Json -Compress
        """

        try:
            proc = subprocess.run(
                ["powershell.exe", "-NoProfile", "-Command", script],
                capture_output=True,
                text=False,
                timeout=10,
                check=False,
            )
        except FileNotFoundError:
            return None

        stdout_text = ""
        for enc in ("utf-8", "utf-16", "utf-16-le", "cp1252"):
            try:
                stdout_text = proc.stdout.decode(enc).strip()
                if stdout_text:
                    break
            except UnicodeDecodeError:
                continue

        if not stdout_text:
            return None

        start = stdout_text.find("{")
        end = stdout_text.rfind("}")
        if start != -1 and end != -1 and end > start:
            stdout_text = stdout_text[start:end + 1]

        try:
            raw = json.loads(stdout_text)
        except json.JSONDecodeError:
            return None

        now = time.time()
        dt = max(0.2, now - self._est_last_ts)
        self._est_last_ts = now

        self._est_read_tb += max(0.0, self._to_float(raw.get("read_bps"), 0.0)) * dt / 1_000_000_000_000.0
        self._est_write_tb += max(0.0, self._to_float(raw.get("write_bps"), 0.0)) * dt / 1_000_000_000_000.0

        raw["estimated_total_tbr_tb"] = self._est_read_tb
        raw["estimated_total_tbw_tb"] = self._est_write_tb

        return {"windows_fallback": raw}

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _temperature_to_celsius(raw_temp) -> float:
        if isinstance(raw_temp, dict):
            if "current" in raw_temp:
                return LiveDataProvider._to_float(raw_temp.get("current"), 0.0)
            if "value" in raw_temp:
                return LiveDataProvider._to_float(raw_temp.get("value"), 0.0)

        temp = LiveDataProvider._to_float(raw_temp, 0.0)
        if temp > 200:
            return max(0.0, temp - 273.15)
        return temp

    @staticmethod
    def _data_units_to_tb(data_units: float) -> float:
        bytes_total = data_units * 512000.0
        return bytes_total / 1_000_000_000_000.0

    def _map_to_simulator_schema(self, smart_json: Dict) -> Dict[str, float]:
        if "windows_fallback" in smart_json:
            raw = smart_json.get("windows_fallback", {})
            power_on_hours = self._to_float(raw.get("power_on_hours"), self._to_float(raw.get("uptime_hours"), 1.0))
            media_errors = self._to_float(raw.get("read_errors"), 0.0)
            crc_errors = self._to_float(raw.get("write_errors"), 0.0)
            unsafe_shutdowns = self._to_float(raw.get("unsafe_shutdowns"), 0.0)

            temp = self._to_float(raw.get("temperature_c"), 40.0)
            if temp <= 0:
                temp = 40.0

            wear_used = self._to_float(raw.get("wear_used"), 0.0)
            if wear_used <= 0:
                health = str(raw.get("health_status", "")).lower()
                wear_used = 10.0 if "healthy" in health else 60.0

            denom = max(power_on_hours, 1.0)
            return {
                "Power_On_Hours": power_on_hours,
                "Total_TBW_TB": round(self._to_float(raw.get("estimated_total_tbw_tb"), 0.0), 4),
                "Total_TBR_TB": round(self._to_float(raw.get("estimated_total_tbr_tb"), 0.0), 4),
                "Temperature_C": round(temp, 2),
                "Percent_Life_Used": max(0.0, min(100.0, wear_used)),
                "Media_Errors": media_errors,
                "Unsafe_Shutdowns": unsafe_shutdowns,
                "CRC_Errors": crc_errors,
                "Read_Error_Rate": round(min(50.0, (media_errors / denom) * 100.0), 3),
                "Write_Error_Rate": round(min(50.0, (crc_errors / denom) * 100.0), 3),
            }

        nvme = smart_json.get("nvme_smart_health_information_log", {})

        power_on_hours = self._to_float(nvme.get("power_on_hours"), 0.0)
        media_errors = self._to_float(nvme.get("media_errors"), 0.0)
        unsafe_shutdowns = self._to_float(nvme.get("unsafe_shutdowns"), 0.0)
        err_log_entries = self._to_float(nvme.get("num_err_log_entries"), 0.0)

        data_units_read = self._to_float(nvme.get("data_units_read"), 0.0)
        data_units_written = self._to_float(nvme.get("data_units_written"), 0.0)

        denom = max(power_on_hours, 1.0)
        read_error_rate = min(50.0, (media_errors / denom) * 100.0)
        write_error_rate = min(50.0, (err_log_entries / denom) * 100.0)

        return {
            "Power_On_Hours": power_on_hours,
            "Total_TBW_TB": round(self._data_units_to_tb(data_units_written), 4),
            "Total_TBR_TB": round(self._data_units_to_tb(data_units_read), 4),
            "Temperature_C": round(self._temperature_to_celsius(nvme.get("temperature")), 2),
            "Percent_Life_Used": self._to_float(nvme.get("percentage_used"), 0.0),
            "Media_Errors": media_errors,
            "Unsafe_Shutdowns": unsafe_shutdowns,
            "CRC_Errors": err_log_entries,
            "Read_Error_Rate": round(read_error_rate, 3),
            "Write_Error_Rate": round(write_error_rate, 3),
        }
