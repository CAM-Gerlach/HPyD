#!/usr/bin/env python3
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
Main script for HPyD.
"""

# Standard library imports
import os
import requests
import time

# Third party imports
import joblib
import numpy as np
import pandas as pd
try:
    import pyart
except (ImportError, KeyError):
    # Hack to fix known problem with legacy basemap componenets in PyART
    os.environ["PROJ_LIB"] = "C:/Anaconda3/envs/pyart/Library/share"
    import pyart
import nexradaws

# Local imports
from hpyd.utils import (
    convert_point_to_tuple,
    get_gate_idx_for_lonlat,
    calc_distance_haversine,
    )
from hypd.io import (
    extract_radar_data_for_reports
    )


# mPING setup
MPING_API_KEY_PATH = "mping_api_key.txt"
MPING_URL = "https://mping.ou.edu/mping/api/v2/{endpoint}"
MPING_ENDPOINT = "reports"
MPING_REQUEST_HEADERS = {
    "content-type": "application/json",
    "Authorization": "Token {api_key}",
    }

# Domain/scope knobs
RADAR_SITE = "KLWX"
RADAR_LON = -77.4876
RADAR_LAT = 38.9763
MIN_DISTANCE = 10000
MAX_DISTANCE = 100000
MPING_REQUEST_PARAMS = {
    "category": "Rain/Snow",
    "point": "{radar_lon},{radar_lat}".format(radar_lon=RADAR_LON,
                                              radar_lat=RADAR_LAT),
    "dist": MAX_DISTANCE,
    }
BEGIN_TRAINING_DATA = pd.Timestamp("2017-12-01")
END_TRAINING_DATA = pd.Timestamp("2018-04-01")
BEGIN_TEST_DATA = pd.Timestamp("2016-12-01")
END_TEST_DATA = pd.Timestamp("2017-04-01")
CATS_TODROP = ["Freezing", "Mixed"]
RAIN_CATS = ["Drizzle"]

# Binning and tuning settings
RANGE_BINS = 5
AZIMUTH_BINS_STD = 3
AZIMUTH_BINS_SR = 5

# Runtime settings
USE_CACHED_DATA = True
DATA_DIR = "data/"


# Create param dicts for the training and test data requests
mping_request_params_training = MPING_REQUEST_PARAMS.copy()
mping_request_params_training["obtime_gte"] = BEGIN_TRAINING_DATA
mping_request_params_training["obtime_lte"] = END_TRAINING_DATA
mping_request_params_test = MPING_REQUEST_PARAMS.copy()
mping_request_params_test["obtime_gte"] = BEGIN_TEST_DATA
mping_request_params_test["obtime_lte"] = END_TEST_DATA

# Read mPING API key
with open(MPING_API_KEY_PATH, encoding="utf_8") as mping_keyfile:
    mping_api_key = mping_keyfile.readline().strip()
MPING_REQUEST_HEADERS["Authorization"] = (
    MPING_REQUEST_HEADERS["Authorization"].format(api_key=mping_api_key))


# %% Download and process mPING data

if not USE_CACHED_DATA:
    mping_reports_list = []
    next_url = True
    while next_url:
        if next_url is True:
            responce = requests.get(MPING_URL.format(endpoint=MPING_ENDPOINT),
                                    params=mping_request_params_test,
                                    headers=MPING_REQUEST_HEADERS)
        else:
            responce = requests.get(next_url, headers=MPING_REQUEST_HEADERS)
        responce.raise_for_status()
        responce_content = responce.json()
        mping_reports_list += responce_content["results"]
        next_url = responce_content["next"]

df_reports = pd.DataFrame(mping_reports_list)
# df_reports.set_index("id", inplace=True)

# Filter and re-categorize report types
df_reports = df_reports[
    ~df_reports.description.str.contains("|".join(CATS_TODROP))]
df_reports["description"] = (
    df_reports["description"].str.replace("|".join(RAIN_CATS), "Rain"))
df_reports = df_reports.astype({"category": "category",
                                "description": "category"})

# Process datatypes
df_reports["obtime"] = pd.to_datetime(df_reports["obtime"])
df_reports["lon"], df_reports["lat"] = zip(
    *df_reports["geom"].map(convert_point_to_tuple))
df_reports.drop("geom", axis="columns", inplace=True)

# Drop observations closer than specified distance to the radar
df_reports["ground_range"] = calc_distance_haversine(
    df_reports["lon"], df_reports["lat"], RADAR_LON, RADAR_LAT)
df_reports = df_reports[df_reports["ground_range"] > 10000]


# %% Download and process radar data

if not USE_CACHED_DATA:
    # Get availible radar volumes
    aws_nexrad_interface = nexradaws.NexradAwsInterface()
    scans_availible = aws_nexrad_interface.get_avail_scans_in_range(
        BEGIN_TEST_DATA, END_TEST_DATA, RADAR_SITE)
    df_scans = pd.DataFrame([scan.__dict__ for scan in scans_availible])
    df_scans["scan_time"] = df_scans["scan_time"].dt.tz_localize(None)
    df_scans["filelist_index"] = df_scans.index

    # Merge with reports df to find needed scans
    df_reports.sort_values("obtime", inplace=True)
    df_scans.sort_values("scan_time", inplace=True)
    test_data = pd.merge_asof(
        df_reports,
        df_scans.drop(["radar_id", "last_modified", "_scan_time_re"], axis=1),
        left_on="obtime", right_on="scan_time")
    scans_todownload = np.sort(test_data["filelist_index"]
                               .unique()).tolist()

    # Download final scans
    radar_dir = DATA_DIR + "radar_test/"
    os.makedirs(radar_dir, exist_ok=True)
    download_results = aws_nexrad_interface.download(
        [scans_availible[idx] for idx in scans_todownload],
        radar_dir, threads=8)


# %% Extract relevant radar gates and variables for each report

if USE_CACHED_DATA:
    raise Exception()
radar_data_list_test = []
time_start = time.time()
for idx, radar_file in enumerate(download_results._successfiles[0:8]):
    try:
        print("Processing file {file_number} of {total_files} "
              "| Elapsed: {time_elapsed} s, Est. Remaining: {time_remaining} s"
              .format(file_number=idx,
                      total_files=len(download_results._successfiles),
                      time_elapsed=time.time() - time_start,
                      time_remaining=(
                          (time.time() - time_start)
                          * (len(download_results._successfiles) - idx + 1)
                          / (idx + 1))))

        # Get report rows for this radar file
        current_reports = (
            test_data.loc[test_data["key"] == radar_file.key])
        if len(current_reports) < 1:
            continue
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
                    azimuth_bins = AZIMUTH_BINS_SR
                else:
                    azimuth_bins = AZIMUTH_BINS_STD

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
                    range_min = range_idx - RANGE_BINS
                    range_max = range_idx + RANGE_BINS + 1

                    average_bins = field_data["data"][ray_min:ray_max,
                                                      range_min:range_max]
                    row_data[field_name + "_" + str(sweeps_gathered)] = (
                        np.nanmean(average_bins))

                sweeps_gathered += 1

            radar_data_list_test.append(row_data)

    except Exception as e:
        print(str(type(e)) + ": " + str(e) + " on file " + str(radar_file.key))

joblib.dump(radar_data_list_test,
            "data/radar_data_list_test.pickle",
            protocol=4)





