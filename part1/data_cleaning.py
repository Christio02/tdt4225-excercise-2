import pandas as pd
import ast

raw_file = "../dataset/porto/porto.csv"
clean_file = "../dataset/porto/porto_cleaned.csv"

# Temporary for testing
df = pd.read_csv(raw_file, nrows=10_000)

# Cleaning
df["POLYLINE"] = df["POLYLINE"].apply(ast.literal_eval)
df["num_points"] = df["POLYLINE"].apply(len)
df = df[df["num_points"] >= 8]

df.to_csv(clean_file, index=False)

print(f"Small cleaned copy saved to {clean_file} ({len(df)} rows)")
