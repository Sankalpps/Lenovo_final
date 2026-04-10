"""
Failure Detection Algorithms for NVMe Drives
============================================

Implements detection algorithms for:
- Mode 2: Thermal Failure (high temperature anomalies)
- Mode 3: Power-Related Failure (unsafe shutdowns and corruption)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


# ============================================================================
# MODE 2: THERMAL FAILURE DETECTION
# ============================================================================

class ThermalFailureDetector:
    """
    Detects Thermal Failures based on temperature patterns and error correlation.
    
    Principle:
    - High sustained temperature (typically >70°C)
    - Results in error spikes (CRC errors, read errors)
    - Temperature-to-error correlation indicates thermal stress
    """
    
    THERMAL_THRESHOLD = 70.0  # Celsius
    ERROR_SPIKE_THRESHOLD = 5  # errors per measurement window
    CORRELATION_THRESHOLD = 0.7  # Strong correlation between temp and errors
    
    def __init__(self):
        self.detection_history = []
    
    def detect_thermal_failure(self, 
                              temperature_data: np.ndarray,
                              crc_errors: np.ndarray,
                              read_errors: np.ndarray,
                              measurement_window: int = 10) -> Dict:
        """
        Detects thermal failure patterns in drive telemetry.
        
        Args:
            temperature_data: Array of temperature readings (Celsius)
            crc_errors: Array of CRC error counts
            read_errors: Array of read error counts
            measurement_window: Window size for analysis (number of measurements)
        
        Returns:
            Dict with:
                - is_thermal_failure: Boolean indicator
                - severity: 0-1 score
                - contributing_factors: List of detected issues
                - threshold_violations: Count of high-temp violations
                - thermal_stability: Average temperature stability
        """
        
        results = {
            'is_thermal_failure': False,
            'severity': 0.0,
            'contributing_factors': [],
            'threshold_violations': 0,
            'thermal_stability': 0.0,
            'max_temperature': float(np.max(temperature_data)) if len(temperature_data) > 0 else 0,
            'mean_temperature': float(np.mean(temperature_data)) if len(temperature_data) > 0 else 0,
        }
        
        if len(temperature_data) < measurement_window:
            return results
        
        # 1. CHECK SUSTAINED HIGH TEMPERATURE
        high_temp_mask = temperature_data > self.THERMAL_THRESHOLD
        threshold_violations = np.sum(high_temp_mask)
        results['threshold_violations'] = int(threshold_violations)
        
        if threshold_violations > measurement_window * 0.5:  # >50% readings above threshold
            results['contributing_factors'].append('Sustained high temperature')
            results['severity'] += 0.4
        
        # 2. DETECT ERROR SPIKES DURING HIGH TEMPERATURE
        total_errors = crc_errors + read_errors
        error_spike_mask = total_errors > self.ERROR_SPIKE_THRESHOLD
        
        if np.sum(error_spike_mask) > 0:
            # Check correlation between high temp and errors
            correlation = self._calculate_correlation(temperature_data, total_errors)
            results['temp_error_correlation'] = correlation
            
            if correlation > self.CORRELATION_THRESHOLD:
                results['contributing_factors'].append('Strong temperature-error correlation')
                results['severity'] += 0.35
            elif correlation > 0.5:
                results['contributing_factors'].append('Moderate temperature-error correlation')
                results['severity'] += 0.15
            
            if np.sum(error_spike_mask) > measurement_window * 0.3:
                results['contributing_factors'].append('Frequent error spikes')
                results['severity'] += 0.2
        
        # 3. CALCULATE THERMAL STABILITY (opposite of volatility)
        # Low stability = high variance in temperature (unstable cooling)
        temp_variance = float(np.var(temperature_data))
        temp_std = float(np.std(temperature_data))
        
        # Normalize variance to 0-1 scale (higher = less stable)
        thermal_instability = min(1.0, temp_std / 20.0)  # Normalize by 20°C std as reference
        results['thermal_stability'] = 1.0 - thermal_instability
        
        if thermal_instability > 0.6:
            results['contributing_factors'].append('Thermal instability (high temperature variance)')
            results['severity'] += 0.1
        
        # Final decision
        results['severity'] = min(1.0, results['severity'])
        results['is_thermal_failure'] = results['severity'] > 0.5
        
        return results
    
    def _calculate_correlation(self, x: np.ndarray, y: np.ndarray) -> float:
        """Calculate Pearson correlation between two arrays."""
        if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
            return 0.0
        return float(np.corrcoef(x, y)[0, 1])
    
    def detect_temperature_anomalies(self, temperature_data: np.ndarray, 
                                     window_size: int = 5) -> np.ndarray:
        """
        Detects sudden temperature spikes using rolling statistics.
        
        Returns:
            Boolean array indicating anomaly points
        """
        if len(temperature_data) < window_size:
            return np.zeros(len(temperature_data), dtype=bool)
        
        anomalies = np.zeros(len(temperature_data), dtype=bool)
        
        for i in range(window_size, len(temperature_data)):
            window = temperature_data[i-window_size:i]
            current = temperature_data[i]
            window_mean = np.mean(window)
            window_std = np.std(window)
            
            # Anomaly if current reading is >2 std devs above window mean
            if current > window_mean + 2 * window_std:
                anomalies[i] = True
        
        return anomalies


# ============================================================================
# MODE 3: POWER-RELATED FAILURE DETECTION
# ============================================================================

class PowerRelatedFailureDetector:
    """
    Detects Power-Related Failures based on:
    - Multiple unsafe shutdowns
    - Power state instability
    - Corruption errors caused by abrupt power loss
    - CRC errors from incomplete transactions
    
    Principle:
    - Each unsafe shutdown can cause data corruption
    - Multiple shutdowns increase likelihood of media errors and CRC issues
    - Power-related failures are deterministic with shutdown count
    """
    
    UNSAFE_SHUTDOWN_THRESHOLD = 5  # count
    CORRUPTION_ERROR_THRESHOLD = 10  # CRC/media errors per measurement window
    POWER_STABILITY_THRESHOLD = 0.3  # low = unstable power
    
    def __init__(self):
        self.detected_patterns = []
    
    def detect_power_related_failure(self,
                                    unsafe_shutdown_count: int,
                                    crc_errors: np.ndarray,
                                    media_errors: np.ndarray,
                                    power_cycle_count: int,
                                    data_corruption_events: int) -> Dict:
        """
        Detects Power-Related Failures through unsafe shutdown impact analysis.
        
        Args:
            unsafe_shutdown_count: Number of unsafe/sudden shutdowns
            crc_errors: Array of CRC error counts over time
            media_errors: Array of media error counts over time
            power_cycle_count: Total number of power cycles
            data_corruption_events: Count of detected data corruption instances
        
        Returns:
            Dict with:
                - is_power_failure: Boolean indicator
                - severity: 0-1 score
                - contributing_factors: List of issues
                - corruption_risk: Risk of data corruption
                - power_stability: Power delivery stability score (0-1)
                - corruption_error_rate: Detected corruption error pattern
        """
        
        results = {
            'is_power_failure': False,
            'severity': 0.0,
            'contributing_factors': [],
            'corruption_risk': 0.0,
            'power_stability': 1.0,
            'corruption_events': data_corruption_events,
            'unsafe_shutdown_count': unsafe_shutdown_count,
        }
        
        # 1. ASSESS UNSAFE SHUTDOWN IMPACT
        if unsafe_shutdown_count > self.UNSAFE_SHUTDOWN_THRESHOLD:
            shutdown_risk = min(1.0, unsafe_shutdown_count / 20.0)  # Normalize to 0-1
            results['contributing_factors'].append(
                f'Excessive unsafe shutdowns: {unsafe_shutdown_count}'
            )
            results['severity'] += 0.35
            results['power_stability'] -= 0.4
            results['corruption_risk'] += shutdown_risk * 0.3
        elif unsafe_shutdown_count > 0:
            results['contributing_factors'].append(
                f'Unsafe shutdowns detected: {unsafe_shutdown_count}'
            )
            results['severity'] += 0.15
            results['power_stability'] -= 0.1
        
        # 2. ANALYZE CRC ERRORS (typically caused by incomplete writes from power loss)
        if len(crc_errors) > 0:
            crc_total = np.sum(crc_errors)
            crc_mean = np.mean(crc_errors)
            
            if crc_total > self.CORRUPTION_ERROR_THRESHOLD:
                results['contributing_factors'].append(
                    f'CRC errors detected: {int(crc_total)} total'
                )
                results['severity'] += 0.25
                results['corruption_risk'] += 0.3
            
            # Check for CRC spike after unsafe shutdown
            if len(crc_errors) > 1 and crc_errors[-1] > crc_mean * 2:
                results['contributing_factors'].append('CRC error spike (post-shutdown)')
                results['severity'] += 0.15
                results['corruption_risk'] += 0.2
        
        # 3. ANALYZE MEDIA ERRORS (data corruption indicator)
        if len(media_errors) > 0:
            media_total = np.sum(media_errors)
            
            if media_total > 0:
                results['contributing_factors'].append(
                    f'Media errors (corruption): {int(media_total)} events'
                )
                results['severity'] += 0.2
                results['corruption_risk'] += 0.3
        
        # 4. DATA CORRUPTION EVENTS
        if data_corruption_events > 0:
            results['contributing_factors'].append(
                f'Data corruption events detected: {data_corruption_events}'
            )
            results['severity'] += 0.35
            results['corruption_risk'] = 1.0
        
        # 5. CALCULATE POWER STABILITY SCORE
        # Based on ratio of unsafe shutdowns to total power cycles
        if power_cycle_count > 0:
            unsafe_ratio = unsafe_shutdown_count / power_cycle_count
            if unsafe_ratio > 0.2:  # >20% shutdowns are unsafe
                results['power_stability'] *= 0.2
                results['contributing_factors'].append('Poor power supply stability')
                results['severity'] += 0.1
            elif unsafe_ratio > 0.1:
                results['power_stability'] *= 0.5
        
        # Finalize
        results['severity'] = min(1.0, results['severity'])
        results['power_stability'] = max(0.0, results['power_stability'])
        results['corruption_risk'] = min(1.0, results['corruption_risk'])
        results['is_power_failure'] = results['severity'] > 0.5
        
        return results
    
    def _assess_shutdown_severity(self, unsafe_count: int, 
                                   power_cycles: int) -> float:
        """
        Assess severity based on unsafe shutdown frequency.
        Returns 0-1 severity score.
        """
        if power_cycles == 0:
            return min(1.0, unsafe_count / 10.0)
        
        unsafe_ratio = unsafe_count / power_cycles
        
        # Scale: 0% unsafe = 0.0, 50% unsafe = 1.0
        return min(1.0, unsafe_ratio * 2.0)
    
    def detect_corruption_pattern(self, crc_history: np.ndarray,
                                  media_error_history: np.ndarray,
                                  shutdown_timestamps: List[int]) -> Dict:
        """
        Detects corruption patterns correlated with unsafe shutdowns.
        
        Args:
            crc_history: Array of CRC errors over time
            media_error_history: Array of media errors over time
            shutdown_timestamps: List of indices where shutdowns occurred
        
        Returns:
            Dict with correlation analysis
        """
        analysis = {
            'corruption_after_shutdown': 0,
            'pattern_detected': False,
            'avg_time_to_corruption': 0,
        }
        
        if len(shutdown_timestamps) == 0:
            return analysis
        
        # Check if errors spikes occur after shutdowns
        for shutdown_idx in shutdown_timestamps:
            if shutdown_idx + 1 < len(crc_history):
                # Check next 3 measurements for error spike
                if np.sum(crc_history[shutdown_idx+1:min(shutdown_idx+4, len(crc_history))]) > 5:
                    analysis['corruption_after_shutdown'] += 1
        
        if analysis['corruption_after_shutdown'] > 0:
            analysis['pattern_detected'] = True
            analysis['avg_time_to_corruption'] = 1 if analysis['corruption_after_shutdown'] else 0
        
        return analysis


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def predict_failure_mode(drive_telemetry: Dict) -> Dict:
    """
    Main prediction function combining both failure detectors.
    
    Args:
        drive_telemetry: Dict with keys:
            - temperature: np.ndarray
            - crc_errors: np.ndarray
            - read_errors: np.ndarray
            - unsafe_shutdown_count: int
            - media_errors: np.ndarray
            - power_cycle_count: int
            - data_corruption_events: int
    
    Returns:
        Dict with thermal and power-related failure predictions
    """
    
    thermal_detector = ThermalFailureDetector()
    power_detector = PowerRelatedFailureDetector()
    
    # Thermal failure detection
    thermal_result = thermal_detector.detect_thermal_failure(
        drive_telemetry.get('temperature', np.array([])),
        drive_telemetry.get('crc_errors', np.array([])),
        drive_telemetry.get('read_errors', np.array([]))
    )
    
    # Power-related failure detection
    power_result = power_detector.detect_power_related_failure(
        drive_telemetry.get('unsafe_shutdown_count', 0),
        drive_telemetry.get('crc_errors', np.array([])),
        drive_telemetry.get('media_errors', np.array([])),
        drive_telemetry.get('power_cycle_count', 0),
        drive_telemetry.get('data_corruption_events', 0)
    )
    
    return {
        'mode_2_thermal': thermal_result,
        'mode_3_power': power_result,
        'primary_failure_mode': 2 if thermal_result['is_thermal_failure'] else (
            3 if power_result['is_power_failure'] else None
        ),
        'combined_severity': max(thermal_result['severity'], power_result['severity'])
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Example 1: Thermal Failure Detection
    print("=" * 70)
    print("MODE 2: THERMAL FAILURE DETECTION")
    print("=" * 70)
    
    # Simulate temperature data with sustained high temperatures
    temperature = np.array([45, 48, 52, 65, 72, 75, 76, 74, 73, 75, 76, 72, 71, 68, 45])
    crc_errors = np.array([0, 0, 1, 2, 5, 8, 10, 9, 8, 7, 9, 6, 3, 1, 0])
    read_errors = np.array([0, 0, 0, 1, 2, 6, 8, 7, 6, 5, 7, 4, 2, 0, 0])
    
    thermal_detector = ThermalFailureDetector()
    thermal_result = thermal_detector.detect_thermal_failure(temperature, crc_errors, read_errors)
    
    print(f"Thermal Failure Detected: {thermal_result['is_thermal_failure']}")
    print(f"Severity Score: {thermal_result['severity']:.2f}")
    print(f"Max Temperature: {thermal_result['max_temperature']:.1f}°C")
    print(f"Mean Temperature: {thermal_result['mean_temperature']:.1f}°C")
    print(f"Threshold Violations: {thermal_result['threshold_violations']}")
    print(f"Contributing Factors: {thermal_result['contributing_factors']}")
    
    # Example 2: Power-Related Failure Detection
    print("\n" + "=" * 70)
    print("MODE 3: POWER-RELATED FAILURE DETECTION")
    print("=" * 70)
    
    power_detector = PowerRelatedFailureDetector()
    power_result = power_detector.detect_power_related_failure(
        unsafe_shutdown_count=8,
        crc_errors=np.array([2, 3, 5, 8, 12, 15, 18, 20]),
        media_errors=np.array([0, 0, 1, 2, 3, 5, 7, 9]),
        power_cycle_count=50,
        data_corruption_events=3
    )
    
    print(f"Power-Related Failure Detected: {power_result['is_power_failure']}")
    print(f"Severity Score: {power_result['severity']:.2f}")
    print(f"Corruption Risk: {power_result['corruption_risk']:.2f}")
    print(f"Power Stability: {power_result['power_stability']:.2f}")
    print(f"Unsafe Shutdowns: {power_result['unsafe_shutdown_count']}")
    print(f"Corruption Events: {power_result['corruption_events']}")
    print(f"Contributing Factors: {power_result['contributing_factors']}")
