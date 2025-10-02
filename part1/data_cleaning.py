import sqlite3
import pandas as pd

# Sandbox
clean_file = "../dataset/porto/porto_cleaned.csv"
db_file = "porto_sandbox.db"

print("Loading CSV into SQLite...")
df = pd.read_csv(clean_file)

conn = sqlite3.connect(db_file)

df.to_sql("porto", conn, index=False, if_exists="replace")
print("Data loaded into SQLite table 'porto'")

# Taxi ID of top 20 trip takers.
queries = [
    #Question 1 - How many taxis, trips, and total GPS points are there? TODO Missing GPS points (!)
    """
    SELECT COUNT(DISTINCT TAXI_ID) AS Taxis, COUNT(DISTINCT TRIP_ID) AS Trips
    FROM porto; 
    """,
    #Question 2 - What is the average number of trips per taxi?
    """
    SELECT COUNT(*) / COUNT(DISTINCT TAXI_ID) AS Average_Trips_Per_Taxi
    FROM porto; 
    """,
    #Question 3 - List the top 20 taxis with the most trips.
    """
    SELECT TAXI_ID, COUNT(*) AS trip_count
    FROM porto
    WHERE num_points >= 1
    GROUP BY TAXI_ID
    ORDER BY trip_count DESC
    LIMIT 20;
    """,
    #Question 4 - a. What is the most used call type per taxi?
    """
    SELECT CALL_TYPE, COUNT(CALL_TYPE)  AS AMOUNT
    FROM porto 
    GROUP BY CALL_TYPE;
    """
    #Question 4 - b.
]

for q in queries:
    print(f"\nRunning query:\n{q.strip()}")
    result = pd.read_sql(q, conn)
    print(result)

