import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import folium

import json
import math
import time

"""
Porto Taxi Dataset - Polyline Bounds Validation

This script validates GPS polylines from the Porto taxi dataset.

AI Assistance Attribution:
- Bounding box calculation: Google Gemini
- Optimization and clarifications: Anthropic Claude
"""

MIN_POLYLINE_POINTS = 8


def load_data(filepath="dataset/porto/porto.csv"):
    return pd.read_csv(filepath)


# found lat and lon: https://latitude.to/map/pt/portugal/cities/porto
lat, lon = (41.14961, -8.61099)  # center point


def calculate_bounding_box_coord(radius):
    """
    Calculate a bounding box (rectangular bounds) from a center point and radius.

    This creates an approximate circle by converting km to degrees:
    - Latitude: ~111.32 km per degree (constant worldwide)
    - Longitude: varies with latitude due to Earth's curvature

    AI help:
    -  The conversion logic (especially the use of math.cos(math.radians(lat))
        for longitude) and the specific suggestion of a 30 km radius for the
        Porto Metropolitan Area was provided by Google's Gemini (conversational
        AI model), based on user-provided center coordinates and the goal of
        encompassing where the area of the dataset is from

    Args:
        radius: Distance in kilometers from center point

    Returns:
        Dictionary with min/max longitude and latitude bounds
    """
    lat_distance_const = 111.32  # in km / degree
    lat_range = radius / lat_distance_const  # convert km to degrees

    # Longitude distance depends on latitude (converges at poles)
    # At equator: 111.32 km/degree, at poles: 0 km/degree
    lon_range = radius / (lat_distance_const * math.cos(math.radians(lat)))

    min_lat = lat - lat_range
    max_lat = lat + lat_range
    min_lon = lon - lon_range
    max_lon = lon + lon_range

    return {
        "min_lon": min_lon,
        "max_lon": max_lon,
        "min_lat": min_lat,
        "max_lat": max_lat,
    }


# Create bounds for Porto metropolitan area (30km radius from center)
PORTO_BOUNDS = calculate_bounding_box_coord(radius=30)

print(f"Calculated bounds: {PORTO_BOUNDS}")


# POLYLINE VALIDATION - OPTIMIZED VERSION (AI-ASSISTED)
# takes in [[-8.618643, 41.141412], [-8.618499, 41.141376], [-8.618346, 41.141353]]
# verifies that each lat and lon for each point is valid
def validate_single_polyline(polyline_str, min_polyline_points):
    """
    Validate whether a polyline is usable for analysis.

    AI was used to optimize this function by:
    - Suggesting numpy array conversion for vectorized operations
    - Implementing efficient boolean masking instead of loops
    - Proper error handling for malformed data

    A polyline is considered INVALID if:
    1. It's empty, null, or malformed
    2. It has fewer than 8 GPS points (too short to be meaningful)
    3. Any GPS coordinates are outside Porto bounds

    Args:
        polyline_str: JSON string of GPS coordinates like:
                      "[[-8.618643, 41.141412], [-8.618499, 41.141376], ...]"

    Returns:
        True if polyline is INVALID (out of bounds or malformed)
        False if polyline is VALID (all points within bounds)
    """
    try:
        # Handle empty/null cases
        if pd.isna(polyline_str) or polyline_str == "[]":
            return True  # Invalid

        # Parse JSON
        polyline = json.loads(polyline_str)
        if not polyline:
            return True

        # Convert to numpy array for VECTORIZED operations (much faster than loops)
        # Shape will be (n_points, 2) where each row is [longitude, latitude]
        coords = np.array(polyline, dtype=float)

        # coords.shape[0] is number of points
        if coords.shape[0] < min_polyline_points:
            return True  # Invalid - too few points (< 8)

        # Extract all longitudes and latitudes as separate arrays
        lons = coords[:, 0]  # First column: all longitude values
        lats = coords[:, 1]  # Second column: all latitude values

        # This checks ALL points at once (numpy optimized in C, ~100x faster than Python loops)
        in_bounds = (
            (lons >= PORTO_BOUNDS["min_lon"])
            & (lons <= PORTO_BOUNDS["max_lon"])
            & (lats >= PORTO_BOUNDS["min_lat"])
            & (lats <= PORTO_BOUNDS["max_lat"])
        )

        if not in_bounds.all():
            return True  # Invalid - contains out-of-bound coordinates

        # Passed all checks - this is a valid polyline
        return False

    except Exception:
        return True  # Treat errors as invalid


def count_invalid_trips_single_core(df):
    """
    Count how many trips have GPS coordinates outside Porto metropolitan area.

    AI-assisted optimization:
    - Added progress indicator for long-running operations
    - Simplified iteration logic
    - Better variable naming

    Args:
        df: DataFrame with POLYLINE column

    Returns:
        Number of invalid trips
    """
    polyline_col = df["POLYLINE"]
    missing_data_col = df["MISSING_DATA"]
    missing_data_true_count = 0
    missing_data_false_count = 0
    total = len(polyline_col)

    for i, (polyline_str, missing) in enumerate(zip(polyline_col, missing_data_col)):
        # Progress indicator every 100k rows
        if i % 100000 == 0 and i > 0:
            print(
                f"  Progress: {i:,}/{total:,} ({i/total*100:.1f}%) - Invalid so far: {missing_data_true_count + missing_data_false_count :,}"
            )

        is_invalid = validate_single_polyline(polyline_str, MIN_POLYLINE_POINTS)

        if is_invalid:
            if missing:
                missing_data_true_count += 1
            else:
                missing_data_false_count += 1

    return {
        0: missing_data_true_count,
        1: missing_data_false_count,
    }


# OLD VERSION (SLOWER) - kept for reference - Very slow!
# def validate_coordinates(polyline_str):
#     for lon, lat in polyline_str:
#         if not (
#             PORTO_BOUNDS["min_lon"] <= lon <= PORTO_BOUNDS["max_lon"]
#             and PORTO_BOUNDS["min_lat"] <= lat <= PORTO_BOUNDS["max_lat"]
#         ):
#             return False
#     return True
# def count_invalid_trips():
#     polyline_col = df["POLYLINE"]
#     count = 0
#     for polyline_string in polyline_col:
#         try:
#             if pd.isna(polyline_string) or polyline_string == "[]":
#                 continue
#             polyline = json.loads(polyline_string)
#             if not validate_coordinates(polyline):
#                 count += 1
#         except:
#             count += 1
#     return count


# Expected output:
# - ~6,000-8,000 invalid trips (0.4-0.5%) - up from ~3,500 due to <8 point filter
# - Most trips stay within 30km of Porto center
# - Invalid trips include: very short trips (<8 points), airport trips, long-distance rides, or GPS errors


def plot_trip_lengths(df):

    valid_mask = (
        pd.notna(df["POLYLINE"])
        & (df["POLYLINE"] != "[]")
        & (df["MISSING_DATA"] == False)
    )

    df["POLYLINE_LENGTH"] = 0

    df.loc[valid_mask, "POLYLINE_LENGTH"] = df.loc[valid_mask, "POLYLINE"].apply(
        lambda x: len(json.loads(x))
    )
    # plt.figure(figsize=(10, 6))
    # plt.hist(df["POLYLINE_LENGTH"], bins=200, color="black", log=True)
    # plt.xlabel("Number of GPS points")
    # plt.ylabel("Frequency")
    # plt.title("Distribution of trip lengths")
    # plt.xlim(0, 500)
    # plt.show()
    average_length = df["POLYLINE_LENGTH"].mean()
    median_length = df["POLYLINE_LENGTH"].median()
    min_length = df["POLYLINE_LENGTH"].min()
    max_length = df["POLYLINE_LENGTH"].max()
    std_dev = df["POLYLINE_LENGTH"].std()

    # print(f"Average points per trip: {average_length:.2f}")
    # print(f"Median points per trip: {median_length}")
    # print(f"Minimum points per trip: {min_length}")
    # print(f"Maximum points per trip: {max_length}")
    # print(f"Standard deviation for trip lengths {std_dev}")

    # print("Looking at longer trips:::")
    # long_trips = df[df["POLYLINE_LENGTH"] > 100]

    # print(long_trips[["POLYLINE_LENGTH", "MISSING_DATA"]].describe())
    # print(long_trips.sort_values("POLYLINE_LENGTH", ascending=False).head(10))

    long_trips = df.nlargest(10, "POLYLINE_LENGTH")

    print("Top 10 longest trips:")
    for index, row in long_trips.iterrows():
        print(
            f"Trip {index}: Has {row["POLYLINE_LENGTH"]} points, where the total time is {(row["POLYLINE_LENGTH"] * 15) / 3600} hours"
        )
        print("\n")
    # for index, row in long_trips.iterrows():
    #     polyline = json.loads(row["POLYLINE"])
    #     map_obj = plot_trips_map(polyline)
    #     map_obj.save(f"trip_{index}.html")


def plot_trips_map(polyline):
    m = folium.Map(location=[41.14961, -8.61099], zoom_start=12)

    folium_coords = [[lat, lon] for lon, lat in polyline]

    folium.PolyLine(folium_coords, color="blue", weight=2.5, opacity=1).add_to(m)

    if folium_coords:
        folium.Marker(
            folium_coords[0], popup="Start", icon=folium.Icon(color="green")
        ).add_to(m)
        folium.Marker(
            folium_coords[-1], popup="End", icon=folium.Icon(color="red")
        ).add_to(m)

    return m


if __name__ == "__main__":
    df = load_data()
    # print("\n=== Processing ===")
    # start = time.time()
    # invalid_count_dict = count_invalid_trips_single_core(df)
    # elapsed = time.time() - start
    # print(
    #     f"\nFound {invalid_count_dict[0]:,} invalid trips out of {len(df):,} total, where MISSING_DATA = TRUE"
    # )
    # print(
    #     f"\nFound {invalid_count_dict[1]:,} invalid trips out of {len(df):,} total, where MISSING_DATA = FALSE"
    # )
    # total_invalid = invalid_count_dict[0] + invalid_count_dict[1]
    # print(f"Percentage: {total_invalid/len(df)*100:.2f}%")
    # print(f"Time elapsed: {elapsed:.2f} seconds")

    # # Breakdown (optional - for better insight)
    # print("\nBreakdown of invalid reasons:")
    # print("- Empty/null polylines")
    # print("- Polylines with < 8 points")
    # print("- Polylines with out-of-bound coordinates")
    # print("(Note: Some trips may have multiple issues)")

    plot_trip_lengths(df)
