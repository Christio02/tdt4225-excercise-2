import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import folium
import json
import math
import time

"""
Porto Taxi Dataset - Polyline Bounds Validation

This script validates GPS polylines from the Porto taxi dataset.

AI Assistance Attribution:
- Bounding box calculation: Google Gemini
- Optimization and clarifications: Anthropic Claude
"""

MIN_POLYLINE_POINTS = 8
MAX_POLYLINE_POINTS = 480  # 2 hours


def load_data(filepath="dataset/porto/porto.csv"):
    return pd.read_csv(filepath)


# found lat and lon: https://latitude.to/map/pt/portugal/cities/porto
lat, lon = (41.14961, -8.61099)  # center point

# OLD VERSION (SLOWER) - kept for reference - Very slow!
# def validate_coordinates(polyline_str):
#     for lon, lat in polyline_str:
#         if not (
#             PORTO_BOUNDS["min_lon"] <= lon <= PORTO_BOUNDS["max_lon"]
#             and PORTO_BOUNDS["min_lat"] <= lat <= PORTO_BOUNDS["max_lat"]
#         ):
#             return False
#     return True
# def count_invalid_trips():
#     polyline_col = df["POLYLINE"]
#     count = 0
#     for polyline_string in polyline_col:
#         try:
#             if pd.isna(polyline_string) or polyline_string == "[]":
#                 continue
#             polyline = json.loads(polyline_string)
#             if not validate_coordinates(polyline):
#                 count += 1
#         except:
#             count += 1
#     return count


# Expected output:
# - ~6,000-8,000 invalid trips (0.4-0.5%) - up from ~3,500 due to <8 point filter
# - Most trips stay within 30km of Porto center
# - Invalid trips include: very short trips (<8 points), airport trips, long-distance rides, or GPS errors


def calculate_bounding_box_coord(radius):
    """
    Calculate a bounding box (rectangular bounds) from a center point and radius.

    This creates an approximate circle by converting km to degrees:
    - Latitude: ~111.32 km per degree (constant worldwide)
    - Longitude: varies with latitude due to Earth's curvature

    AI help:
    -  The conversion logic (especially the use of math.cos(math.radians(lat))
        for longitude) and the specific suggestion of a 30 km radius for the
        Porto Metropolitan Area was provided by Google's Gemini (conversational
        AI model), based on user-provided center coordinates and the goal of
        encompassing where the area of the dataset is from

    Args:
        radius: Distance in kilometers from center point

    Returns:
        Dictionary with min/max longitude and latitude bounds
    """
    lat_distance_const = 111.32  # in km / degree
    lat_range = radius / lat_distance_const  # convert km to degrees

    # Longitude distance depends on latitude (converges at poles)
    # At equator: 111.32 km/degree, at poles: 0 km/degree
    lon_range = radius / (lat_distance_const * math.cos(math.radians(lat)))

    min_lat = lat - lat_range
    max_lat = lat + lat_range
    min_lon = lon - lon_range
    max_lon = lon + lon_range

    return {
        "min_lon": min_lon,
        "max_lon": max_lon,
        "min_lat": min_lat,
        "max_lat": max_lat,
    }


# Create bounds for Porto metropolitan area (30km radius from center)
PORTO_BOUNDS = calculate_bounding_box_coord(radius=30)

print(f"Calculated bounds: {PORTO_BOUNDS}")


def validate_single_polyline(polyline_str, min_polyline_points, max_polyline_points):
    """
    Validate whether a polyline is usable for analysis.

    AI was used to optimize this function by:
    - Suggesting numpy array conversion for vectorized operations
    - Implementing efficient boolean masking instead of loops
    - Proper error handling for malformed data

    A polyline is considered INVALID if:
    1. It's empty, null, or malformed
    2. It has fewer than min_polyline_points GPS points (too short to be meaningful)
    3. It has more than max_polyline_points GPS points (too long to be realistic)
    4. Any GPS coordinates are outside Porto bounds

    Args:
        polyline_str: JSON string of GPS coordinates like:
                      "[[-8.618643, 41.141412], [-8.618499, 41.141376], ...]"
        min_polyline_points: Minimum number of points required
        max_polyline_points: Maximum number of points allowed

    Returns:
        Dictionary with validation results:
        {
            'valid': bool,
            'reason': str,  # 'valid', 'empty', 'too_short', 'too_long', 'out_of_bounds', 'malformed'
            'point_count': int or None
        }
    """
    try:
        # Handle empty/null cases
        if pd.isna(polyline_str) or polyline_str == "[]":
            return {"valid": False, "reason": "empty", "point_count": 0}

        # Parse JSON
        polyline = json.loads(polyline_str)
        if not polyline:
            return {"valid": False, "reason": "empty", "point_count": 0}

        # Convert to numpy array for VECTORIZED operations (much faster than loops)
        # Shape will be (n_points, 2) where each row is [longitude, latitude]
        coords = np.array(polyline, dtype=float)
        point_count = coords.shape[0]

        # Check minimum points
        if point_count < min_polyline_points:
            return {"valid": False, "reason": "too_short", "point_count": point_count}

        # Check maximum points
        if point_count > max_polyline_points:
            return {"valid": False, "reason": "too_long", "point_count": point_count}

        # Extract all longitudes and latitudes as separate arrays
        lons = coords[:, 0]  # First column: all longitude values
        lats = coords[:, 1]  # Second column: all latitude values

        # Check if ALL points are within bounds (numpy optimized in C)
        in_bounds = (
            (lons >= PORTO_BOUNDS["min_lon"])
            & (lons <= PORTO_BOUNDS["max_lon"])
            & (lats >= PORTO_BOUNDS["min_lat"])
            & (lats <= PORTO_BOUNDS["max_lat"])
        )

        if not in_bounds.all():
            return {
                "valid": False,
                "reason": "out_of_bounds",
                "point_count": point_count,
            }

        # Passed all checks - this is a valid polyline
        return {"valid": True, "reason": "valid", "point_count": point_count}

    except Exception:
        return {"valid": False, "reason": "malformed", "point_count": None}


def count_invalid_trips(df, min_points, max_points):
    """
    Count invalid trips by category with progress tracking.

    Separates length-based invalidity from geographic invalidity.

    Returns:
        Dictionary with counts by reason and missing_data status
    """
    polylines = df["POLYLINE"]
    missing = df["MISSING_DATA"]

    # Counters for trips with MISSING_DATA=True
    counts_missing = {
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "out_of_bounds": 0,
        "malformed": 0,
        "valid": 0,
    }

    # Counters for trips with MISSING_DATA=False
    counts_valid_flag = {
        "empty": 0,
        "too_short": 0,
        "too_long": 0,
        "out_of_bounds": 0,
        "malformed": 0,
        "valid": 0,
    }

    for i, (poly, miss) in enumerate(zip(polylines, missing)):
        if i % 100000 == 0 and i > 0:
            total_invalid = (
                sum(counts_missing.values())
                + sum(counts_valid_flag.values())
                - counts_missing["valid"]
                - counts_valid_flag["valid"]
            )
            print(
                f"  Progress: {i:,}/{len(polylines):,} ({i / len(polylines) * 100:.1f}%) - Invalid so far: {total_invalid:,}"
            )

        result = validate_single_polyline(poly, min_points, max_points)

        if miss:
            counts_missing[result["reason"]] += 1
        else:
            counts_valid_flag[result["reason"]] += 1

    return {
        "missing_data_true": counts_missing,
        "missing_data_false": counts_valid_flag,
    }


def analyze_trip_statistics(df):
    """Calculate comprehensive trip length statistics."""
    valid_mask = (
        pd.notna(df["POLYLINE"])
        & (df["POLYLINE"] != "[]")
        & (df["MISSING_DATA"] == False)
    )

    df["POLYLINE_LENGTH"] = 0
    df.loc[valid_mask, "POLYLINE_LENGTH"] = df.loc[valid_mask, "POLYLINE"].apply(
        lambda x: len(json.loads(x))
    )

    lengths = df[df["POLYLINE_LENGTH"] > 0]["POLYLINE_LENGTH"]

    stats = {
        "mean": lengths.mean(),
        "median": lengths.median(),
        "std": lengths.std(),
        "q1": lengths.quantile(0.25),
        "q3": lengths.quantile(0.75),
        "p95": lengths.quantile(0.95),
        "p99": lengths.quantile(0.99),
        "min": lengths.min(),
        "max": lengths.max(),
    }

    print("=== Trip Length Statistics ===")
    print(f"Mean: {stats['mean']:.1f} points ({stats['mean'] * 15 / 60:.1f} min)")
    print(f"Median: {stats['median']:.1f} points ({stats['median'] * 15 / 60:.1f} min)")
    print(f"Std Dev: {stats['std']:.1f} points")
    print(f"Q1: {stats['q1']:.1f} points ({stats['q1'] * 15 / 60:.1f} min)")
    print(f"Q3: {stats['q3']:.1f} points ({stats['q3'] * 15 / 60:.1f} min)")
    print(
        f"95th percentile: {stats['p95']:.1f} points ({stats['p95'] * 15 / 60:.1f} min)"
    )
    print(
        f"99th percentile: {stats['p99']:.1f} points ({stats['p99'] * 15 / 60:.1f} min)"
    )

    return stats, lengths, df


def analyze_bounds_impact(lengths, lower=8, upper=480):
    """Analyze impact of chosen bounds."""
    print(f"\n=== Bounds Impact Analysis ===")
    print(f"Lower bound: {lower} points ({lower * 15 / 60:.1f} min)")
    print(f"Upper bound: {upper} points ({upper * 15 / 60:.0f} min)")

    too_short = (lengths < lower).sum()
    too_long = (lengths > upper).sum()
    valid = ((lengths >= lower) & (lengths <= upper)).sum()

    print(
        f"\nRemoved (too short): {too_short:,} ({too_short / len(lengths) * 100:.2f}%)"
    )
    print(f"Removed (too long): {too_long:,} ({too_long / len(lengths) * 100:.2f}%)")
    print(f"Remaining valid: {valid:,} ({valid / len(lengths) * 100:.2f}%)")

    # Compare thresholds
    print(f"\n=== Alternative Thresholds ===")
    for low, high in [(5, 360), (8, 480), (10, 400), (8, 200)]:
        kept = ((lengths >= low) & (lengths <= high)).sum()
        print(
            f"[{low:3d}, {high:3d}]: {kept:,} ({kept / len(lengths) * 100:.1f}%) | {low * 15 / 60:.1f}-{high * 15 / 60:.0f} min"
        )


def justify_bounds(stats, lengths, lower=8, upper=480):
    """Provide comprehensive justification for bounds."""
    print(f"\n{'=' * 70}")
    print("BOUND JUSTIFICATION")
    print("=" * 70)

    print(f"\n1. LOWER BOUND: {lower} points ({lower * 15 / 60:.1f} minutes)")
    removed_lower = (lengths < lower).sum()
    print(
        f"   Removes: {removed_lower:,} trips ({removed_lower / len(lengths) * 100:.2f}%)"
    )
    print(f"   Rationale: Too short for meaningful taxi trip")
    print(f"   - {lower} points × 15 sec = {lower * 15} seconds")
    print(f"   - Likely GPS errors, cancelled trips, or data artifacts")

    print(f"\n2. UPPER BOUND: {upper} points ({upper * 15 / 60:.0f} minutes)")
    removed_upper = (lengths > upper).sum()
    print(
        f"   Removes: {removed_upper:,} trips ({removed_upper / len(lengths) * 100:.2f}%)"
    )
    print(f"   Rationale: Geographic constraints of Porto")
    print(f"   - Porto metro area: ~30km diameter")
    print(f"   - Average urban speed: 20-30 km/h with traffic")
    print(f"   - Maximum reasonable trip: ~2 hours")
    print(
        f"   - Statistical context: 99th percentile = {stats['p99']:.0f} points ({stats['p99'] * 15 / 60:.1f} min)"
    )


def visualize_distribution(lengths, stats, lower=8, upper=480):
    """Create comprehensive visualization of trip length distribution.
    This was mostly made using AI, as the knowledge of plotting within the team was minimal
    Used: Claud sonnet 4.5

    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # 1. Full distribution (log scale)
    axes[0, 0].hist(lengths, bins=100, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 0].set_yscale("log")
    axes[0, 0].axvline(
        stats["median"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {stats['median']:.0f}",
    )
    axes[0, 0].axvline(
        lower, color="orange", linestyle="--", linewidth=2, label=f"Lower: {lower}"
    )
    axes[0, 0].axvline(
        upper, color="darkred", linestyle="--", linewidth=2, label=f"Upper: {upper}"
    )
    axes[0, 0].set_xlabel("GPS Points")
    axes[0, 0].set_ylabel("Frequency (log)")
    axes[0, 0].set_title("Full Distribution (Log Scale)")
    axes[0, 0].legend()

    # 2. Zoomed (0-200 points)
    zoom_data = lengths[lengths <= 200]
    axes[0, 1].hist(zoom_data, bins=50, color="steelblue", edgecolor="black", alpha=0.7)
    axes[0, 1].axvline(
        stats["median"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"Median: {stats['median']:.0f}",
    )
    axes[0, 1].axvline(
        lower, color="orange", linestyle="--", linewidth=2, label=f"Lower: {lower}"
    )
    axes[0, 1].set_xlabel("GPS Points")
    axes[0, 1].set_ylabel("Frequency")
    axes[0, 1].set_title("Distribution (0-200 points)")
    axes[0, 1].legend()

    # 3. Histogram with KDE
    axes[1, 0].hist(
        lengths, bins=50, density=True, color="lightblue", alpha=0.7, edgecolor="black"
    )

    # Add KDE curve
    kde = gaussian_kde(lengths)
    x_range = np.linspace(lengths.min(), lengths.max(), 100)
    axes[1, 0].plot(x_range, kde(x_range), color="red", linewidth=2, label="KDE")
    axes[1, 0].axvline(
        stats["median"],
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Median: {stats['median']:.0f}",
    )
    axes[1, 0].axvline(
        lower, color="orange", linestyle="--", linewidth=2, label=f"Lower: {lower}"
    )
    axes[1, 0].axvline(
        upper, color="darkred", linestyle="--", linewidth=2, label=f"Upper: {upper}"
    )
    axes[1, 0].set_xlabel("GPS Points")
    axes[1, 0].set_ylabel("Density")
    axes[1, 0].set_title("Histogram with KDE")
    axes[1, 0].legend()

    # 4. Upper tail
    upper_tail = lengths[lengths > stats["p95"]]
    axes[1, 1].hist(upper_tail, bins=30, color="coral", edgecolor="black", alpha=0.7)
    axes[1, 1].axvline(
        stats["p99"],
        color="red",
        linestyle="--",
        linewidth=2,
        label=f"99th: {stats['p99']:.0f}",
    )
    axes[1, 1].axvline(
        upper, color="darkred", linestyle="--", linewidth=2, label=f"Upper: {upper}"
    )
    axes[1, 1].set_xlabel("GPS Points")
    axes[1, 1].set_ylabel("Frequency")
    axes[1, 1].set_title(f"Upper 5% (>{stats['p95']:.0f} points)")
    axes[1, 1].legend()

    plt.tight_layout()
    plt.savefig("trip_length_analysis.png", dpi=300, bbox_inches="tight")
    plt.show()


def print_validation_summary(results):
    """Print detailed summary of validation results."""
    print("\n" + "=" * 70)
    print("VALIDATION SUMMARY")
    print("=" * 70)

    missing_true = results["missing_data_true"]
    missing_false = results["missing_data_false"]

    print("\nTrips with MISSING_DATA=True:")
    print(f"  Empty/Null: {missing_true['empty']:,}")
    print(f"  Too Short (<{MIN_POLYLINE_POINTS} points): {missing_true['too_short']:,}")
    print(f"  Too Long (>{MAX_POLYLINE_POINTS} points): {missing_true['too_long']:,}")
    print(f"  Out of Bounds: {missing_true['out_of_bounds']:,}")
    print(f"  Malformed: {missing_true['malformed']:,}")
    print(f"  Valid: {missing_true['valid']:,}")
    total_missing = sum(missing_true.values())
    print(f"  TOTAL: {total_missing:,}")

    print("\nTrips with MISSING_DATA=False:")
    print(f"  Empty/Null: {missing_false['empty']:,}")
    print(
        f"  Too Short (<{MIN_POLYLINE_POINTS} points): {missing_false['too_short']:,}"
    )
    print(f"  Too Long (>{MAX_POLYLINE_POINTS} points): {missing_false['too_long']:,}")
    print(f"  Out of Bounds: {missing_false['out_of_bounds']:,}")
    print(f"  Malformed: {missing_false['malformed']:,}")
    print(f"  Valid: {missing_false['valid']:,}")
    total_valid_flag = sum(missing_false.values())
    print(f"  TOTAL: {total_valid_flag:,}")

    # Overall statistics
    total_trips = total_missing + total_valid_flag
    total_invalid = total_trips - missing_true["valid"] - missing_false["valid"]

    print(f"\nOVERALL:")
    print(f"  Total trips: {total_trips:,}")
    print(
        f"  Total valid: {missing_true['valid'] + missing_false['valid']:,} ({(missing_true['valid'] + missing_false['valid'])/total_trips*100:.2f}%)"
    )
    print(f"  Total invalid: {total_invalid:,} ({total_invalid/total_trips*100:.2f}%)")

    # Breakdown by invalidity type (excluding valid trips)
    print(f"\nINVALID TRIP BREAKDOWN:")
    total_empty = missing_true["empty"] + missing_false["empty"]
    total_short = missing_true["too_short"] + missing_false["too_short"]
    total_long = missing_true["too_long"] + missing_false["too_long"]
    total_bounds = missing_true["out_of_bounds"] + missing_false["out_of_bounds"]
    total_malformed = missing_true["malformed"] + missing_false["malformed"]

    print(f"  Length-based issues:")
    print(f"    - Empty/Null: {total_empty:,} ({total_empty/total_trips*100:.2f}%)")
    print(f"    - Too Short: {total_short:,} ({total_short/total_trips*100:.2f}%)")
    print(f"    - Too Long: {total_long:,} ({total_long/total_trips*100:.2f}%)")
    print(f"  Geographic issues:")
    print(
        f"    - Out of Bounds: {total_bounds:,} ({total_bounds/total_trips*100:.2f}%)"
    )
    print(f"  Data quality issues:")
    print(
        f"    - Malformed: {total_malformed:,} ({total_malformed/total_trips*100:.2f}%)"
    )


if __name__ == "__main__":
    df = load_data()

    print("=" * 70)
    print("STEP 1: Analyze Trip Lengths")
    print("=" * 70)
    stats, lengths, df = analyze_trip_statistics(df)

    print("\n" + "=" * 70)
    print("STEP 2: Analyze Bounds Impact")
    print("=" * 70)
    analyze_bounds_impact(lengths, MIN_POLYLINE_POINTS, MAX_POLYLINE_POINTS)

    print("\n" + "=" * 70)
    print("STEP 3: Justify Chosen Bounds")
    print("=" * 70)
    justify_bounds(stats, lengths, MIN_POLYLINE_POINTS, MAX_POLYLINE_POINTS)

    print("\n" + "=" * 70)
    print("STEP 4: Visualize Distribution")
    print("=" * 70)
    visualize_distribution(lengths, stats, MIN_POLYLINE_POINTS, MAX_POLYLINE_POINTS)

    print("\n" + "=" * 70)
    print("STEP 5: Validate Polylines with Chosen Bounds")
    print("=" * 70)
    start = time.time()
    results = count_invalid_trips(df, MIN_POLYLINE_POINTS, MAX_POLYLINE_POINTS)

    print_validation_summary(results)
    print(f"\nValidation completed in {time.time() - start:.2f}s")
