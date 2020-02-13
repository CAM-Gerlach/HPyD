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
General utility functions for working with radar data in PyART.
"""


# Standard library imports
import math

# Third party imports
import numpy as np
import pyproj


def convert_point_to_tuple(point_obj):
    """Convert a geom dict with a list of coords to a lon, lat tuple."""
    lon, lat = tuple(point_obj["coordinates"])
    return lon, lat


def calc_distance_haversine(lon1, lat1, lon2, lat2):
    """Calculate the true distance between pairs of geographic coordinates."""
    lon1, lat1, lon2, lat2 = map(np.radians, [lon1, lat1, lon2, lat2])

    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1

    a = (np.sin(delta_lon / 2.0) ** 2 + np.cos(lat1)
         * np.cos(lat2) * np.sin(delta_lat / 2.0) ** 2)

    distance_m = pyproj.pj_ellps["WGS84"]["a"] * 2 * np.arcsin(np.sqrt(a))
    return distance_m


def get_gate_idx_for_lonlat(sweep_lons, sweep_lats, point_lon, point_lat):
    """Get the index location of the radar volume closest to a given point."""
    delta_lon = (sweep_lons - point_lon) * math.cos(math.radians(point_lat))
    delta_lat = (sweep_lats - point_lat)
    distances = np.sqrt((delta_lon ** 2) + (delta_lat ** 2))
    ray_idx, range_idx = np.unravel_index(distances.argmin(), distances.shape)
    return ray_idx, range_idx
