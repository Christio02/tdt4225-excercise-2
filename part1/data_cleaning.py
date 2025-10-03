import pandas as pd
import ast

from part1.eda.verify_polyline_bounds import validate_single_polyline


raw_file = "dataset/porto/porto.csv"
clean_file = "dataset/porto/porto_cleaned.csv"

df = pd.read_csv(raw_file, nrows=100_000)


def handle_duplicates(df):
    """
    Handle duplicate TRIP_IDs by:
    1. Remove the identical duplicate rows with identical TRIP_IDs
    2. Keep different rows with the same TRIP_ID but give them unique IDs
    """
    # Remove identical duplicate rows
    columns_to_check = [col for col in df.columns if col != "POLYLINE"]
    df = df.drop_duplicates(subset=columns_to_check, keep="first")

    # Handle different rows with the same TRIP_ID
    df["TRIP_ID_count"] = df.groupby("TRIP_ID").size()
    duplicate_trip_ids = df[df["TRIP_ID_count"] > 1].index

    for trip_id in duplicate_trip_ids:
        mask = df["TRIP_ID"] == trip_id
        duplicate_rows = df[mask]

        for i, (index, row) in enumerate(duplicate_rows.iterrows()):
            if i > 0:  # Skip the first occurrence
                new_trip_id = f"{row['TRIP_ID'] + 10000000000000}"
                df.loc[index, "TRIP_ID"] = new_trip_id

    df = df.drop("TRIP_ID_count", axis=1)

    return df


# Polyline cleaning

polyline_col = df["POLYLINE"]
val_results = polyline_col.apply(
    lambda polyline: validate_single_polyline(
        polyline_str=polyline, min_polyline_points=8, max_polyline_points=480
    ),
)
df_clean = df[val_results.apply(lambda x: x["valid"])].copy()

# Handle duplicates
df_clean = handle_duplicates(df_clean)

# Calculate the taxi trip end time, and taxi trip duration
df_clean["TRIP_DURATION"] = df_clean["num_points"] * 15
df_clean["END_TIME"] = df_clean["TIMESTAMP"] + df_clean["TRIP_DURATION"]

df_clean.to_csv(clean_file, index=False)


# remove MISSING_DATA column, as rows affected by this has already been cleaned
df_clean = df_clean.drop(columns="MISSING_DATA", errors="ignore")

df_clean.to_csv(clean_file, index=False)

print(f"Cleaned: {len(df_clean):,} rows ({len(df_clean)/len(df)*100:.1f}%)")
print(f"Small cleaned copy saved to {clean_file} ({len(df)} rows)")
