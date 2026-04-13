import json
import queue
import subprocess
import threading
import time
from typing import Dict, Generator, Optional


class LiveDataProvider:
    """
    Polls live NVMe SMART telemetry and maps it to the simulator schema.

    Expected simulator schema (exact keys):
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

    SMART fields used from smartctl/nvme:
    - temperature
    - percentage_used
    - media_errors
    - unsafe_shutdowns
    - data_units_read
    - data_units_written
    - num_err_log_entries (used as CRC_Errors proxy)
    - power_on_hours
    - available_spare (read but not required by simulator)
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

    def __init__(self, device: str = "/dev/nvme0n1", interval_sec: float = 1.0):
        self.device = device
        self.interval_sec = float(interval_sec)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._lock = threading.Lock()
        self._latest: Optional[Dict[str, float]] = None
        self._samples: "queue.Queue[Dict[str, float]]" = queue.Queue(maxsize=1)

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, name="LiveDataProvider", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=2.0)

    def get_latest(self, timeout: float = 2.0) -> Dict[str, float]:
        with self._lock:
            if self._latest is not None:
                return dict(self._latest)

        sample = self._samples.get(timeout=timeout)
        return dict(sample)

    def stream(self, timeout: float = 2.0) -> Generator[Dict[str, float], None, None]:
        """
        Non-blocking integration point for simulator loops.
        Yields newest mapped telemetry snapshots while polling thread runs.
        """
        while not self._stop_event.is_set():
            yield self.get_latest(timeout=timeout)

    def _poll_loop(self) -> None:
        while not self._stop_event.is_set():
            try:
                payload = self._read_smart_json()
                mapped = self._map_to_simulator_schema(payload)

                with self._lock:
                    self._latest = mapped

                if self._samples.full():
                    try:
                        self._samples.get_nowait()
                    except queue.Empty:
                        pass
                self._samples.put_nowait(mapped)
            except Exception:
                # Keep polling even if one sample fails.
                pass

            self._stop_event.wait(self.interval_sec)

    def _read_smart_json(self) -> Dict:
        """
        Primary source: smartctl JSON output.
        Fallback: nvme-cli JSON output.
        """
        smartctl_cmd = ["smartctl", "-j", "-a", self.device]
        smartctl_proc = subprocess.run(
            smartctl_cmd,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )

        if smartctl_proc.stdout.strip():
            try:
                return json.loads(smartctl_proc.stdout)
            except json.JSONDecodeError:
                pass

        nvme_cmd = ["nvme", "smart-log", "-o", "json", self.device]
        nvme_proc = subprocess.run(
            nvme_cmd,
            capture_output=True,
            text=True,
            timeout=8,
            check=False,
        )

        if nvme_proc.stdout.strip():
            return {"nvme_smart_health_information_log": json.loads(nvme_proc.stdout)}

        raise RuntimeError("Unable to read SMART telemetry from smartctl or nvme-cli")

    @staticmethod
    def _to_float(value, default: float = 0.0) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return float(default)

    @staticmethod
    def _temperature_to_celsius(raw_temp) -> float:
        # smartctl may return Kelvin or a dict with current Celsius.
        if isinstance(raw_temp, dict):
            if "current" in raw_temp:
                return LiveDataProvider._to_float(raw_temp.get("current"), 0.0)
            if "value" in raw_temp:
                return LiveDataProvider._to_float(raw_temp.get("value"), 0.0)

        temp = LiveDataProvider._to_float(raw_temp, 0.0)
        if temp > 200:  # likely Kelvin
            return max(0.0, temp - 273.15)
        return temp

    @staticmethod
    def _data_units_to_tb(data_units: float) -> float:
        # NVMe data unit = 1000 * 512 bytes = 512000 bytes.
        bytes_total = data_units * 512000.0
        return bytes_total / 1_000_000_000_000.0

    def _map_to_simulator_schema(self, smart_json: Dict) -> Dict[str, float]:
        nvme = smart_json.get("nvme_smart_health_information_log", {})

        power_on_hours = self._to_float(nvme.get("power_on_hours"), 0.0)
        media_errors = self._to_float(nvme.get("media_errors"), 0.0)
        unsafe_shutdowns = self._to_float(nvme.get("unsafe_shutdowns"), 0.0)
        err_log_entries = self._to_float(nvme.get("num_err_log_entries"), 0.0)

        data_units_read = self._to_float(nvme.get("data_units_read"), 0.0)
        data_units_written = self._to_float(nvme.get("data_units_written"), 0.0)

        # Approximate rates from cumulative counters over lifetime.
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


if __name__ == "__main__":
    # Example only: replace with your real simulator object import.
    # from your_simulator_module import simulator

    class _DemoSimulator:
        def predict(self, telemetry: Dict[str, float]) -> Dict:
            return {"received": telemetry, "status": "ok"}

    simulator = _DemoSimulator()
    provider = LiveDataProvider(device="/dev/nvme0n1", interval_sec=1.0)
    provider.start()

    try:
        for sample in provider.stream():
            prediction = simulator.predict(sample)
            print(prediction)
            time.sleep(0.05)
    except KeyboardInterrupt:
        pass
    finally:
        provider.stop()