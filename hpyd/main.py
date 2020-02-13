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

# Third party imports
import joblib
import numpy as np
import pandas as pd
import sklearn.ensemble
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
    calc_distance_haversine,
    )
from hpyd.io import (
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
                                    params=mping_request_params_training,
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
        BEGIN_TRAINING_DATA, END_TRAINING_DATA, RADAR_SITE)
    df_scans = pd.DataFrame([scan.__dict__ for scan in scans_availible])
    df_scans["scan_time"] = df_scans["scan_time"].dt.tz_localize(None)
    df_scans["filelist_index"] = df_scans.index

    # Merge with reports df to find needed scans
    df_reports.sort_values("obtime", inplace=True)
    df_scans.sort_values("scan_time", inplace=True)
    df_reports_scans = pd.merge_asof(
        df_reports,
        df_scans.drop(["radar_id", "last_modified", "_scan_time_re"], axis=1),
        left_on="obtime", right_on="scan_time")
    scans_todownload = np.sort(df_reports_scans["filelist_index"]
                               .unique()).tolist()

    # Download final scans
    radar_dir = DATA_DIR + "radar_training/"
    os.makedirs(radar_dir, exist_ok=True)
    download_results = aws_nexrad_interface.download(
        [scans_availible[idx] for idx in scans_todownload],
        radar_dir, threads=8)


# %% Extract relevant radar gates and variables for each report

if not USE_CACHED_DATA:
    if __name__ == "__main__":
        radar_data_training_list = (
            joblib.Parallel(n_jobs=6, verbose=50, backend="multiprocessing",
                            pre_dispatch=24)
             (joblib.delayed(extract_radar_data_for_reports)(
                 radar_file=radar_file,
                 report_df=df_reports_scans)
              for radar_file in download_results._successfiles))

    if isinstance(radar_data_training_list[0], list):
        radar_data_training_list = [data_dict
                                    for sublist in radar_data_training_list
                                    for data_dict in sublist]
    radar_data_training = pd.DataFrame(radar_data_training_list,
                                       dtype=np.float32)
    training_data = df_reports_scans.merge(radar_data_training,
                                           on="id", how="inner")

    joblib.dump(training_data, DATA_DIR + "training_data.pickle", protocol=4)


# %% Do the same for the test dataset

if not USE_CACHED_DATA:
    if __name__ == "__main__":
        radar_data_test_list = (
            joblib.Parallel(n_jobs=6, verbose=50, backend="multiprocessing",
                            pre_dispatch=24)
             (joblib.delayed(extract_radar_data_for_reports)(
                 radar_file=radar_file,
                 report_df=test_data)
              for radar_file in download_results._successfiles))

    if isinstance(radar_data_test_list[0], list):
        radar_data_test_list = [data_dict
                                for sublist in radar_data_test_list
                                for data_dict in sublist]
    radar_data_test = pd.DataFrame(radar_data_test_list, dtype=np.float32)
    test_data = df_reports_scans.merge(radar_data_test, on="id", how="inner")

    joblib.dump(test_data, DATA_DIR + "test_data.pickle", protocol=4)


# %% Cleanup data prior to running ML

training_data = joblib.load(DATA_DIR + "training_data.pickle")
test_data = joblib.load(DATA_DIR + "test_data.pickle")

training_data.drop(["spectrum_width_0", "velocity_0",
                    "spectrum_width_1", "velocity_1"],
                   axis="columns", inplace=True)
test_data.drop(["spectrum_width_0", "velocity_0",
                "spectrum_width_1", "velocity_1"],
               axis="columns", inplace=True)

training_data["description"] = (
    training_data["description"].astype(str).str
    .replace("Ice Pellets/Sleet", "Snow and/or Graupel")).astype("category")
test_data["description"] = (
    test_data["description"].astype(str).str
    .replace("Ice Pellets/Sleet", "Snow and/or Graupel")).astype("category")
training_data["hour"] = training_data["obtime"].dt.hour
test_data["hour"] = test_data["obtime"].dt.hour

DROP_COLS = [
     "category", "description", "description_id", "id", "obtime", "awspath",
     "filename", "key", "scan_time", "filelist_index", "lat", "lon",
     "range_1", "range_2", "range_3", "range_4", "range_0",
     "azimuth_1", "azimuth_2", "azimuth_3", "azimuth_4", "azimuth_0",
     ]

training_data_predictors = training_data.copy().drop(DROP_COLS, axis="columns")
training_data_predictand = training_data.copy()["description"]
test_data_predictors = test_data.copy().drop(DROP_COLS, axis="columns")
test_data_predictand = test_data.copy()["description"]

training_data_predictors = training_data_predictors.apply(
    lambda df_col: df_col.fillna(-32.0) if "reflectivity" in df_col.name
    else df_col.fillna(-1))
test_data_predictors = test_data_predictors.apply(
    lambda df_col: df_col.fillna(-32.0) if "reflectivity" in df_col.name
    else df_col.fillna(-1))


# %% Train RandomForest

TOTAL_RUNS = 100
if not USE_CACHED_DATA:
    model_list = []
    for i in range(TOTAL_RUNS):
        print("Run " + str(i + 1) + " of " + str(TOTAL_RUNS))
        rf_model = sklearn.ensemble.RandomForestClassifier(n_estimators=100,
                                                           max_depth=17,
                                                           random_state=i,
                                                           oob_score=True)
        trained_rf_model = rf_model.fit(training_data_predictors,
                                        training_data_predictand)
        model_list.append(trained_rf_model)

accuracy_list = [
    trained_model.score(test_data_predictors, test_data_predictand)
    for trained_model in model_list]
accuracy_mean = np.mean(accuracy_list)
accuracy_sigma = np.std(accuracy_list)

oob_list = [trained_model.oob_score_ for trained_model in model_list]
oob_mean = np.mean(oob_list)
oob_sigma = np.std(oob_list)

importance_arr = np.array([trained_model.feature_importances_
                           for trained_model in model_list])
importance_mean = np.mean(importance_arr, axis=0)
importance_sigma = np.std(importance_arr, axis=0)
importance_mean_df = pd.DataFrame({"Predictor": test_data_predictors.columns,
                                   "Importance": importance_mean,
                                   "ConfidenceInterval": importance_sigma * 2})
importance_table = importance_mean_df.to_latex(
    multicolumn=True, bold_rows=False, na_rep="NA", float_format="%.4f")

predictions_list = [
    trained_model.predict(test_data_predictors)
    for trained_model in model_list]


# %%

import time

t1 = time.time()
for radar_file_test in download_results._successfiles:
    radar_data_test = pyart.io.read_nexrad_archive(
        radar_file_test.filepath, scans=[0, 2], delay_field_loading=True)
print(time.time() - t1)
