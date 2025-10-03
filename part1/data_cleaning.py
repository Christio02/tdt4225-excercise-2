import pandas as pd
import ast

from part1.eda.verify_polyline_bounds import validate_single_polyline


raw_file = "dataset/porto/porto.csv"
clean_file = "dataset/porto/porto_cleaned.csv"

# Temporary for testing
df = pd.read_csv(raw_file, nrows=50000)


# Polyline cleaning

polyline_col = df["POLYLINE"]
val_results = polyline_col.apply(
    lambda polyline: validate_single_polyline(
        polyline_str=polyline, min_polyline_points=8, max_polyline_points=480
    ),
)
df_clean = df[val_results.apply(lambda x: x["valid"])].copy()


# Cleaning
# df["POLYLINE"] = df["POLYLINE"].apply(ast.literal_eval)
# df["num_points"] = df["POLYLINE"].apply(len)
# df = df[df["num_points"] >= 8]


# remove MISSING_DATA column, as rows affected by this has already been cleaned
df_clean = df_clean.drop(columns="MISSING_DATA", errors="ignore")

df_clean.to_csv(clean_file, index=False)

print(f"Cleaned: {len(df_clean):,} rows ({len(df_clean)/len(df)*100:.1f}%)")
print(f"Small cleaned copy saved to {clean_file} ({len(df)} rows)")
