import sqlite3
import pandas as pd
import ast

# Sandbox
clean_file = "dataset/porto/porto_cleaned.csv"
db_file = "porto_sandbox.db"

print("Loading CSV into SQLite...")
df = pd.read_csv(clean_file)

df["POLYLINE"] = df["POLYLINE"].apply(ast.literal_eval)
df["num_points"] = df["POLYLINE"].apply(len)
df["trip_duration_s"] = df["num_points"] * 15
df["trip_duration_min"] = df["trip_duration_s"] / 60
df["trip_duration_hour"] = df["trip_duration_min"] / 60
df["POLYLINE"] = df["POLYLINE"].apply(str)

conn = sqlite3.connect(db_file)

df.to_sql("porto", conn, index=False, if_exists="replace")
print("Data loaded into SQLite table 'porto'")

queries = [
    # Question 1 - How many taxis, trips, and total GPS points are there? TODO Missing GPS points (!)
    """
    SELECT COUNT(DISTINCT TAXI_ID) AS Taxis, COUNT(DISTINCT TRIP_ID) AS Trips
    FROM porto; 
    """,
    # Question 2 - What is the average number of trips per taxi?
    """
    SELECT COUNT(*) / COUNT(DISTINCT TAXI_ID) AS Average_Trips_Per_Taxi
    FROM porto; 
    """,
    # Question 3 - List the top 20 taxis with the most trips.
    """
    SELECT TAXI_ID, COUNT(*) AS trip_count
    FROM porto
    WHERE num_points >= 1
    GROUP BY TAXI_ID
    ORDER BY trip_count DESC
    LIMIT 20;
    """,
    # Question 4 - a. What is the most used call type per taxi?
    """
    SELECT CALL_TYPE, COUNT(CALL_TYPE)  AS AMOUNT
    FROM porto 
    GROUP BY CALL_TYPE;
    """,
    # Question 4 - b. For each call type, compute the average trip duration and distance, and also report the share of trips starting in four time bands: 00–06, 06–12, 12–18, and 18–24
    """
    SELECT 
    COUNT(*) AS Interval_00_06
    FROM porto
    WHERE CAST(strftime('%H', TIMESTAMP, 'unixepoch') AS INTEGER) BETWEEN 0 AND 6;

    """,
    """
    SELECT 
    COUNT(*) AS Interval_06_12
    FROM porto
    WHERE CAST(strftime('%H', TIMESTAMP, 'unixepoch') AS INTEGER) BETWEEN 6 AND 12;
    """,
    """
    SELECT 
    COUNT(*) AS Interval_12_18
    FROM porto
    WHERE CAST(strftime('%H', TIMESTAMP, 'unixepoch') AS INTEGER) BETWEEN 12 AND 18;
    """,
    """
    SELECT 
    COUNT(*) AS Interval_18_24
    FROM porto
    WHERE CAST(strftime('%H', TIMESTAMP, 'unixepoch') AS INTEGER) BETWEEN 18 AND 24;
    """,
    # Question 5 - Find the taxis with the most total hours driven as well as total distance driven. List them in order of total hours.
    """
    SELECT TAXI_ID, SUM(trip_duration_hour) AS Total_hours
    FROM porto
    GROUP BY TAXI_ID
    ORDER BY Total_hours DESC
    LIMIT 20;
    """,  # Limit is only temporary
    # Question 6 - Find the trips that passed within 100 m of Porto City Hall. (longitude, latitude) = (-8.62911, 41.15794)
]

for q in queries:
    print(f"\nRunning query:\n{q.strip()}")
    result = pd.read_sql(q, conn)
    print(result)
