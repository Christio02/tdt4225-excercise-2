import pandas as pd
import ast

def handle_duplicates(df):
    """
    Handle duplicate TRIP_IDs by:
    1. Remove the identical duplicate rows with identical TRIP_IDs
    2. Keep different rows with the same TRIP_ID but give them unique IDs
    """
    # Remove identical duplicate rows
    columns_to_check = [col for col in df.columns if col != 'POLYLINE']
    df = df.drop_duplicates(subset=columns_to_check, keep="first")

    # Handle different rows with the same TRIP_ID
    df["TRIP_ID_count"] = df.groupby("TRIP_ID").size()
    duplicate_trip_ids = df[df["TRIP_ID_count"] > 1].index

    for trip_id in duplicate_trip_ids:
        mask = df["TRIP_ID"] == trip_id
        duplicate_rows = df[mask]

        for i, (index, row) in enumerate(duplicate_rows.iterrows()):
            if i > 0: # Skip the first occurrence
                new_trip_id = f"{row['TRIP_ID'] + 10000000000000}"
                df.loc[index, "TRIP_ID"] = new_trip_id
    
    df = df.drop("TRIP_ID_count", axis=1)

    return df


raw_file = "dataset/porto/porto.csv"
clean_file = "dataset/porto/porto_cleaned.csv"

# Temporary for testing
df = pd.read_csv(raw_file, nrows=100_000)

# Cleaning
df["POLYLINE"] = df["POLYLINE"].apply(ast.literal_eval)
df["num_points"] = df["POLYLINE"].apply(len)
df = df[df["num_points"] >= 8]

# Handle duplicates
df = handle_duplicates(df)

df.to_csv(clean_file, index=False)

print(f"Small cleaned copy saved to {clean_file} ({len(df)} rows)")



