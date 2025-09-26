import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json


valid_chars = ["A", "B", "C"]

df = pd.read_csv("dataset/porto/porto.csv")
# print(f"Dataset shape: {df.shape}")
# print(f"Memory usage: {df.memory_usage().sum() / 1024**2:.2f} MB")
# print("\nNull values:")
# print(df.isnull().sum())

# missing_data_count = df["MISSING_DATA"].sum()  # how many rows have missing data
# total_rows = len(df)
# print(f"\nRows with MISSING_DATA = True: {missing_data_count}")


# # Analyze polyline points for each row
# def count_polyline_points(polyline_str):
#     """Count the number of GPS points in a polyline"""
#     try:
#         if pd.isna(polyline_str) or polyline_str == "[]":
#             return 0
#         polyline = json.loads(polyline_str)
#         return len(polyline)
#     except:
#         return 0


# df["polyline_points"] = df["POLYLINE"].apply(count_polyline_points)

# print(f"\nPolyline analysis:")
# print(f"Rows with empty polylines: {(df['polyline_points'] == 0).sum()}")
# print(f"Mean points per polyline: {df['polyline_points'].mean():.2f}")
# print(f"Median points per polyline: {df['polyline_points'].median():.2f}")
# print(f"Max points in a polyline: {df['polyline_points'].max()}")
# print(f"Min points in a polyline: {df['polyline_points'].min()}")

# print(f"\nPolyline points distribution:")
# print(df["polyline_points"].value_counts().head(10))

# cross_analysis = pd.crosstab(
#     df["MISSING_DATA"], df["polyline_points"] == 0, margins=True, margins_name="Total"
# )
# print(f"\nCross-analysis (MISSING_DATA vs Empty Polylines):")
# print(cross_analysis)

PORTO_BOUNDS = {"min_lon": -8.7, "max_lon": -8.5, "min_lat": 41.0, "max_lat": 41.3}

# takes in [[-8.618643, 41.141412], [-8.618499, 41.141376], [-8.618346, 41.141353]]


# verifies that each lat and lon for each point is valid
def validate_coordinates(polyline_str):
    for sublist in trajectory:
        for lon, lat in sublist:
            if not (
                PORTO_BOUNDS["min_lon"] <= lon <= PORTO_BOUNDS["max_lon"]
                and PORTO_BOUNDS["min_lat"] <= lat <= PORTO_BOUNDS["max_lat"]
            ):
                return False
    return True


def count_invalid_daytype_entries():
    day_type_col = df["DAY_TYPE"]
    count_wrong = 0
    for day_type in day_type_col:
        if day_type not in valid_chars:
            count_wrong += 1
    return count_wrong


def count_invalid_call_type():
    call_type_col = df["CALL_TYPE"]

    return (~call_type_col.isin(valid_chars)).sum()


print(
    f"How many rows have wrong day type: {count_invalid_daytype_entries()} out of {len(df)}"
)

print(
    f"How many rows have wrong call_type: {count_invalid_call_type()} out of {len(df)}"
)
