# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2018- C.A.M. Gerlach
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# -----------------------------------------------------------------------------

"""
I/O module to interface with mPING and NEXRAD data sources to obtain data.
"""

# Third party imports
import numpy as np

# Local imports
from hpyd.utils import get_gate_idx_for_lonlat


def extract_radar_data_for_reports(radar_file, report_df, range_bins=5,
                                   azimuth_bins_std=3, azimuth_bins_sr=5):
    """Extract various data points from the radar data for given reports."""
    radar_data_sub_list = []
    try:
        # Get report rows for this radar file
        current_reports = (
            report_df.loc[report_df["key"] == radar_file.key])
        if current_reports.empty:
            return []
        radar_data = radar_file.open_pyart()

        for report in current_reports.itertuples():
            # Fill metadata in output dict
            row_data = {"id": report.id}
            row_data["VCP"] = radar_data.metadata["vcp_pattern"]

            # Do per-sweep retrievals
            sweeps_gathered = 0
            for sweep_index in range(radar_data.nsweeps):
                if sweeps_gathered >= 5:
                    break
                if sweep_index != 0 and (
                        radar_data.fixed_angle["data"][sweep_index] <=
                        radar_data.fixed_angle["data"][sweep_index - 1]):
                    continue

                # Calculate point position in sweep
                start_ray, end_ray = radar_data.get_start_end(sweep_index)
                sweep_lons = (
                    radar_data.gate_longitude["data"][start_ray:(end_ray + 1)])
                sweep_lats = (
                    radar_data.gate_latitude["data"][start_ray:(end_ray + 1)])
                ray_idx, range_idx = get_gate_idx_for_lonlat(
                    sweep_lons, sweep_lats, report.lon, report.lat)
                ray_idx_abs = ray_idx + start_ray

                # Increase the bin count if Super-Res
                if radar_data.rays_per_sweep["data"][sweep_index] >= 540:
                    azimuth_bins = azimuth_bins_sr
                else:
                    azimuth_bins = azimuth_bins_std

                # Get basic data values
                row_data["elevation_" + str(sweeps_gathered)] = (
                    radar_data.fixed_angle["data"][sweep_index])
                row_data["range_" + str(sweeps_gathered)] = (
                    radar_data.range["data"][range_idx])
                row_data["azimuth_" + str(sweeps_gathered)] = (
                    radar_data.azimuth["data"][ray_idx_abs])

                # Get radar moment mean values
                for field_name, field_data in radar_data.fields.items():
                    ray_min = ray_idx_abs - azimuth_bins
                    ray_max = ray_idx_abs + azimuth_bins + 1
                    range_min = range_idx - range_bins
                    range_max = range_idx + range_bins + 1

                    average_bins = field_data["data"][ray_min:ray_max,
                                                      range_min:range_max]
                    row_data[field_name + "_" + str(sweeps_gathered)] = (
                        np.nanmean(average_bins.filled(np.NaN)))

                sweeps_gathered += 1

            radar_data_sub_list.append(row_data)

    except Exception as e:
        print(str(type(e)) + ": " + str(e) + " on file " + str(radar_file.key))
        return []

    return radar_data_sub_list
