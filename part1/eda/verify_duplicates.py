import time
import pandas as pd

def load_data(filepath="dataset/porto/porto.csv"):
    return pd.read_csv(filepath)


def check_duplicate_trip_ids(df):
    """
    Check for duplicate TRIP_IDs in the dataframe and check if the rows are identical.

    Args:
        df: DataFrame containing the data with a 'TRIP_ID' column.
    """
    trip_id_counts = df["TRIP_ID"].value_counts()
    duplicate_trip_ids = trip_id_counts[trip_id_counts > 1]
    print("Antall dupliserte TRIP_ID:", duplicate_trip_ids.shape[0])
    print(duplicate_trip_ids.head())

    identical_count = 0
    different_count = 0

    for trip_id in duplicate_trip_ids.index:
        duplicates = df[df["TRIP_ID"] == trip_id]
        rows_without_trip_id = duplicates.drop(columns=["TRIP_ID"])

        if rows_without_trip_id.nunique().max() == 1:
            identical_count += 1
            print(f"\nTRIP_ID {trip_id} has duplicate rows that are identical:")
            print(duplicates)
        else:
            has_duplicates = rows_without_trip_id.duplicated().any()
            if has_duplicates:
                different_count += 1
                print(f"\nTRIP_ID {trip_id} has duplicate rows that are different:")
                print(duplicates)
                print("These are the duplicate rows:")
                print(rows_without_trip_id[rows_without_trip_id.duplicated(keep=False)])
    
    print(f"\nSummary:")
    print(f"TRIP_IDs with identical duplicates: {identical_count}")
    print(f"TRIP_IDs with different duplicates: {different_count}")

def check_duplicate_rows(df):
    """
    Check for duplicate rows in the dataframe ignoring the TRIP_ID column.

    Args:
        df: DataFrame containing the data with a 'TRIP_ID' column.
    """
    df_without_trip_id = df.drop(columns=["TRIP_ID"])
    duplicates = df_without_trip_id.duplicated()
    print("\nChecking for duplicate rows (ignoring TRIP_ID):")
    print(f"Number of duplicate rows (ignoring TRIP_ID): {duplicates.sum()}")
    if duplicates.sum() > 0:
        print("\nThese are the duplicate rows (ignoring TRIP_ID):")
        print(df[duplicates])


if __name__ == "__main__":
    df = load_data()
    start = time.time()

    print("\n \n=== Checking for Duplicate TRIP_IDs ===")
    check_duplicate_trip_ids(df)

    print("\n \n=== Checking for Duplicate Rows ===")
    check_duplicate_rows(df)

    elapsed = time.time() - start
    print(f"\nTotal elapsed time: {elapsed:.2f} seconds")