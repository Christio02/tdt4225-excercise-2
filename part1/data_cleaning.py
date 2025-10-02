import pandas as pd
##Midlertidig sjekk
file_path = "../dataset/porto/porto.csv"

df = pd.read_csv(file_path)

# --- 1. Duplikate TRIP_ID med ulike turer ---
trip_id_groups = df.groupby("TRIP_ID").nunique()

problematic_trip_ids = trip_id_groups[trip_id_groups.max(axis=1) > 1]
print(f"Trip_IDs som peker på flere forskjellige turer: {len(problematic_trip_ids)}")
if not problematic_trip_ids.empty:
    print(problematic_trip_ids.head())

# --- 2. Identiske turer med ulike TRIP_ID ---
df_no_id = df.drop(columns=["TRIP_ID"])

duplicates_diff_id = df_no_id[df_no_id.duplicated(keep=False)]

print(f"Antall turer som er duplisert men har forskjellig TRIP_ID: {len(duplicates_diff_id)}")
if not duplicates_diff_id.empty:
    print(duplicates_diff_id.head())
