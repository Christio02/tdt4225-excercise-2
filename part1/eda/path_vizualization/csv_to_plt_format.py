import pandas as pd
import os
from datetime import datetime
import ast


def load_data(filepath="dataset/porto/porto.csv"):
    return pd.read_csv(filepath)


def parse_polyline(polyline_str):
    """Parse polyline string to list of coordinates"""
    if pd.isna(polyline_str) or polyline_str == "[]":
        return []
    try:
        return ast.literal_eval(polyline_str)
    except:
        return []


def create_trajectory_files(df, output_dir="trajectory_data"):
    """
    Create .plt trajectory files for each taxi ID from the Porto dataset
    """
    # Create output directory
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Group the data by their TAXI_ID
    taxi_groups = df.groupby("TAXI_ID")

    for taxi_id, taxi_data in taxi_groups:
        # Create taxi folder
        taxi_dir = os.path.join(output_dir, f"taxi_{taxi_id:03d}")
        trajectory_dir = os.path.join(taxi_dir, "Trajectory")

        if not os.path.exists(trajectory_dir):
            os.makedirs(trajectory_dir)

        # Process each trip for this TAXI_ID
        for idx, row in taxi_data.iterrows():
            trip_id = row["TRIP_ID"]
            timestamp = row["TIMESTAMP"]
            polyline = parse_polyline(row["POLYLINE"])

            # Skip polylines that are empty
            if not polyline:
                continue

            # Create filename for a trip based on trips start time
            start_time = datetime.fromtimestamp(timestamp)
            filename = f"{start_time.strftime('%Y%m%d%H%M%S')}.plt"
            filepath = os.path.join(trajectory_dir, filename)

            # Write the .plt file
            with open(filepath, "w") as f:
                # Write header
                f.write("Geolife trajectory\n")
                f.write("WGS 84\n")
                f.write("Altitude is in Feet\n")
                f.write("Reserved 3\n")
                f.write("0,2,255,My Track,0,0,2,8421376\n")
                f.write("0\n")

                # Write GPS/polyline points
                current_timestamp = timestamp
                for i, (lon, lat) in enumerate(polyline):
                    # Calculate the time for each point (15 seconds for each point)
                    point_time = datetime.fromtimestamp(current_timestamp + i * 15)

                    # Format: lat,lon,unused,altitude,days_since_1899,date,time
                    days_since_1899 = (point_time - datetime(1899, 12, 30)).days

                    f.write(
                        f"{lat},{lon},0,0,{days_since_1899},"
                        f"{point_time.strftime('%Y-%m-%d')},{point_time.strftime('%H:%M:%S')}\n"
                    )

        print(f"Created trajectory files for Taxi {taxi_id}")


def create_labels_file(df, output_dir="trajectory_data"):
    """
    Create labels.txt files for each taxi based on trip data
    """
    taxi_groups = df.groupby("TAXI_ID")

    for taxi_id, taxi_data in taxi_groups:
        taxi_dir = os.path.join(output_dir, f"taxi_{taxi_id:03d}")

        if not os.path.exists(taxi_dir):
            continue

        labels_file = os.path.join(taxi_dir, "labels.txt")

        with open(labels_file, "w") as f:
            for idx, row in taxi_data.iterrows():
                timestamp = row["TIMESTAMP"]
                polyline = parse_polyline(row["POLYLINE"])

                if not polyline:
                    continue

                start_time = datetime.fromtimestamp(timestamp)

                # Estimate end time based on polyline length
                end_time = datetime.fromtimestamp(timestamp + len(polyline) * 15)

                # Determine the transportation mode based on call type
                call_type = row.get("CALL_TYPE", "C")
                if call_type == "A":
                    mode = "taxi_central"
                elif call_type == "B":
                    mode = "taxi_stand"
                else:
                    mode = "taxi_street"

                # Format: start_date start_time end_date end_time mode
                f.write(
                    f"{start_time.strftime('%Y/%m/%d')} {start_time.strftime('%H:%M:%S')} "
                    f"{end_time.strftime('%Y/%m/%d')} {end_time.strftime('%H:%M:%S')} {mode}\n"
                )


def main():
    df = load_data()

    # Filter out missing data
    df = df.dropna(subset=["POLYLINE"])
    df = df[df["POLYLINE"] != "[]"]

    print(f"Processing {len(df)} trips from {df['TAXI_ID'].nunique()} taxis")
    create_trajectory_files(df)
    create_labels_file(df)

    print("Trajectory files created successfully!")


if __name__ == "__main__":
    main()
