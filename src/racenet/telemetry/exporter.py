"""
Telemetry exporter - Export telemetry to various formats.

Provides:
- CSV export
- JSON export
- Binary format for large datasets
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Optional
from pathlib import Path
import json
import csv
import numpy as np

from racenet.telemetry.recorder import TelemetryRecorder


class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


@dataclass
class ExporterConfig:
    """Exporter configuration."""
    output_dir: str = "./telemetry_data"
    include_metadata: bool = True
    compress: bool = False


class TelemetryExporter:
    """Export telemetry data to files.
    
    Supports multiple formats for analysis in external tools.
    """
    
    def __init__(self, config: ExporterConfig | None = None):
        """Initialize exporter.
        
        Args:
            config: Exporter configuration
        """
        self.config = config or ExporterConfig()
        
        # Ensure output directory exists
        self._output_path = Path(self.config.output_dir)
        self._output_path.mkdir(parents=True, exist_ok=True)
    
    def export_csv(
        self,
        recorder: TelemetryRecorder,
        filename: str = "telemetry.csv",
        channels: List[str] | None = None,
    ) -> Path:
        """Export telemetry to CSV file.
        
        Args:
            recorder: Telemetry recorder with data
            filename: Output filename
            channels: Channels to export (None = all)
            
        Returns:
            Path to exported file
        """
        output_file = self._output_path / filename
        
        # Get channels to export
        if channels is None:
            channels = list(recorder.channels.keys())
        
        # Get all timestamps
        all_times = set()
        for name in channels:
            ch = recorder.get_channel(name)
            if ch:
                all_times.update(ch.get_times().tolist())
        
        times = sorted(all_times)
        
        # Build rows
        with open(output_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ["time"] + channels
            writer.writerow(header)
            
            # Data rows (interpolated to common time base)
            for t in times:
                row = [f"{t:.4f}"]
                for name in channels:
                    ch = recorder.get_channel(name)
                    if ch:
                        # Find nearest value (simple approach)
                        ch_times = ch.get_times()
                        ch_values = ch.get_values()
                        if len(ch_times) > 0:
                            idx = np.argmin(np.abs(ch_times - t))
                            if abs(ch_times[idx] - t) < 0.02:  # Within 20ms
                                row.append(f"{ch_values[idx]:.3f}")
                            else:
                                row.append("")
                        else:
                            row.append("")
                    else:
                        row.append("")
                
                writer.writerow(row)
        
        return output_file
    
    def export_json(
        self,
        recorder: TelemetryRecorder,
        filename: str = "telemetry.json",
    ) -> Path:
        """Export telemetry to JSON file.
        
        Args:
            recorder: Telemetry recorder with data
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        output_file = self._output_path / filename
        
        data = {
            "metadata": recorder.get_state() if self.config.include_metadata else {},
            "channels": {},
        }
        
        for name, channel in recorder.channels.items():
            data["channels"][name] = {
                "times": channel.get_times().tolist(),
                "values": channel.get_values().tolist(),
                "statistics": channel.get_state(),
            }
        
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, cls=NumpyEncoder)
        
        return output_file
    
    def export_numpy(
        self,
        recorder: TelemetryRecorder,
        filename: str = "telemetry.npz",
    ) -> Path:
        """Export telemetry to NumPy compressed file.
        
        Args:
            recorder: Telemetry recorder with data
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        output_file = self._output_path / filename
        
        arrays = {}
        for name, channel in recorder.channels.items():
            arrays[f"{name}_times"] = channel.get_times()
            arrays[f"{name}_values"] = channel.get_values()
        
        np.savez_compressed(output_file, **arrays)
        
        return output_file
    
    def export_lap_summary(
        self,
        recorder: TelemetryRecorder,
        filename: str = "lap_summary.json",
    ) -> Path:
        """Export lap-by-lap summary.
        
        Args:
            recorder: Telemetry recorder with data
            filename: Output filename
            
        Returns:
            Path to exported file
        """
        output_file = self._output_path / filename
        
        laps = []
        for lap in range(recorder.current_lap + 1):
            lap_data = {
                "lap": lap,
                "channels": {},
            }
            
            for name in recorder.channels.keys():
                times, values = recorder.get_lap_data(lap, name)
                if len(values) > 0:
                    lap_data["channels"][name] = {
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "mean": float(np.mean(values)),
                    }
            
            laps.append(lap_data)
        
        with open(output_file, 'w') as f:
            json.dump({"laps": laps}, f, indent=2)
        
        return output_file
