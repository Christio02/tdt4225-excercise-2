import numpy as np
import pandas as pd
import glob
import os.path
import datetime
import os
import matplotlib.pyplot as plt

def read_plt(plt_file):
    """
    Read a single .plt trajectory file and return the GPS points as a DataFrame
    """
    try:
        # Check if the file exists and isn´t empty
        if not os.path.exists(plt_file) or os.path.getsize(plt_file) == 0:
            print(f"Warning: Empty or missing file {plt_file}")
            return None
            
        points = pd.read_csv(plt_file, skiprows=6, header=None,
                             parse_dates=[[5, 6]], infer_datetime_format=True)

        # Check if the file has any data points
        if points.empty:
            print(f"Warning: No data found in {plt_file}")
            return None

        # Renamed columns
        points.rename(inplace=True, columns={'5_6': 'time', 0: 'lat', 1: 'lon', 3: 'alt'})

        # Remove unused columns
        points.drop(inplace=True, columns=[2, 4])

        return points
        
    except pd.errors.EmptyDataError:
        print(f"Warning: Empty data in file {plt_file}")
        return None
    except Exception as e:
        print(f"Warning: Error reading {plt_file}: {e}")
        return None

# Converts the call types into enums
mode_names = ['taxi_stand', 'taxi_central', 'taxi_street']
mode_ids = {s : i + 1 for i, s in enumerate(mode_names)}
mode_ids['unknown'] = 0  # For weird points that doesn´t fit into any label 

def read_labels(labels_file):
    """
    Read the labels.txt file for a TAXI_ID and returns a DataFrame with the labels
    """
    if not os.path.exists(labels_file):
        return None
        
    try:
        labels_data = []
        
        with open(labels_file, 'r') as file:
            for line in file:
                parts = line.strip().split(' ')
                if len(parts) >= 5:
                    start_time_str = f"{parts[0]} {parts[1]}"
                    end_time_str = f"{parts[2]} {parts[3]}"
                    location_type = parts[4]
                    
                    try:
                        start_time = datetime.datetime.strptime(start_time_str, "%Y/%m/%d %H:%M:%S")
                        end_time = datetime.datetime.strptime(end_time_str, "%Y/%m/%d %H:%M:%S")
                        
                        labels_data.append({
                            'start_time': start_time,
                            'end_time': end_time,
                            'label': location_type
                        })
                    except ValueError:
                        continue
        
        if not labels_data:
            return None
            
        labels = pd.DataFrame(labels_data)
        
        # replace 'label' column with integer encoding
        labels['label'] = [mode_ids.get(i, 0) for i in labels['label']]

        return labels
        
    except Exception as e:
        print(f"Warning: Error reading labels file {labels_file}: {e}")
        return None


def apply_labels(points, labels):
    """
    Apply transportation mode labels to points based on time intervals
    """
    # If the label file doesn´t exist or is empty, mark all points as unknown and exit
    if labels is None or labels.empty:
        points['label'] = 0 
        points['label_name'] = 'unknown'
        return
    
    # Initialize all label points as unknown in the start
    points['label'] = 0
    points['label_name'] = 'unknown'
    
    # For each time interval in the label file, set the corresponding call type for the points
    for _, label_row in labels.iterrows():
        mask = (points['time'] >= label_row['start_time']) & \
               (points['time'] <= label_row['end_time'])
        points.loc[mask, 'label'] = label_row['label']
        
        # Also add the label names, not just the id
        label_name = [k for k, v in mode_ids.items() if v == label_row['label']]
        if label_name:
            points.loc[mask, 'label_name'] = label_name[0]

def read_taxi(taxi_folder):
    """
    Read all trajectory files for a single TAXI_ID
    """
    plt_files = glob.glob(os.path.join(taxi_folder, 'Trajectory', '*.plt'))
    if not plt_files:
        plt_files = glob.glob(os.path.join(taxi_folder, '*.plt'))
    
    # Return an empty DataFrame if there are no files to read
    if not plt_files:
        return pd.DataFrame()
    
    # Read all files, filtering out None and empty results
    valid_dfs = []
    for plt_file in plt_files:
        df = read_plt(plt_file)
        if df is not None and not df.empty:
            valid_dfs.append(df)
    
    if not valid_dfs:
        print(f"Warning: No valid trajectory files in {taxi_folder}")
        return pd.DataFrame()
    
    df = pd.concat(valid_dfs, ignore_index=True)
    
    # Sort all the paths in the dataframe by time to ensure proper ordering
    df = df.sort_values('time').reset_index(drop=True)

    labels_file = os.path.join(taxi_folder, 'labels.txt')
    if os.path.exists(labels_file):
        labels = read_labels(labels_file)
        apply_labels(df, labels)
    else:
        df['label'] = 0  # Unknown
        df['label_name'] = 'unknown'

    return df

def read_all_taxis(folder):
    """
    Read all trajectory data from all taxis
    """
    subfolders = [f for f in os.listdir(folder) if f.startswith('taxi_')]
    dfs = []
    successful_taxis = 0
    
    for i, sf in enumerate(subfolders):
        print('[%d/%d] processing taxi %s' % (i + 1, len(subfolders), sf))
        df = read_taxi(os.path.join(folder, sf))
        
        if not df.empty:
            # Extract taxi ID from folder name (e.g., 'taxi_20000001' -> 20000001)
            taxi_id = int(sf.split('_')[1])
            df['taxi'] = taxi_id
            dfs.append(df)
            successful_taxis += 1
    
    print(f"Successfully processed {successful_taxis} out of {len(subfolders)} taxis")
    
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

def load_porto_trajectories(trajectory_dir="trajectory_data"):
    """
    Main function to load all Porto taxi trajectories
    """
    print(f"Loading trajectories from {trajectory_dir}...")
    df = read_all_taxis(trajectory_dir)
    
    if df.empty:
        print("No trajectory data found!")
        return df
    
    print(f"Loaded {len(df)} GPS points from {df['taxi'].nunique()} taxis")
    print("\nDataset overview:")
    print(df.head())
    print(f"\nColumns: {list(df.columns)}")
    print(f"Date range: {df['time'].min()} to {df['time'].max()}")
 
    if 'label_name' in df.columns:
        print("\nLabel distribution:")
        label_counts = df['label_name'].value_counts()
        for label, count in label_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {label}: {count} points ({percentage:.1f}%)")
    
    return df

def visualize_trajectories(df, sample_size=100000, color_by_label=False):
    """
    Visualize trajectories using matplotlib (with optional label coloring)
    """
    # If the dataset is too large, sample a subset for visualization
    if len(df) > sample_size:
        df_sample = df.sample(sample_size)
        print(f"Sampling {sample_size} points from {len(df)} total points")
    else:
        df_sample = df
    
    plt.figure(figsize=(15, 10))
    
    if color_by_label and 'label_name' in df_sample.columns:
        label_colors = {
            'taxi_stand': 'blue',
            'taxi_central': 'red',
            'taxi_street': 'green',
            'unknown': 'gray'
        }
        
        for label in df_sample['label_name'].unique():
            label_data = df_sample[df_sample['label_name'] == label]
            plt.scatter(label_data['lon'], label_data['lat'], 
                       alpha=0.6, s=0.5, 
                       color=label_colors.get(label, 'gray'),
                       label=f"{label} ({len(label_data)} points)")
        
        plt.legend()
        plt.title('Porto Taxi Trajectories (Colored by Location Type)')
    else:
        plt.scatter(df_sample['lon'], df_sample['lat'], alpha=0.1, s=0.1)
        plt.title('Porto Taxi Trajectories')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.show()

def visualize_individual_taxi(df, taxi_id, max_trips=5, color_by_label=False):
    """
    Visualize trajectories for a specific TAXI_ID
    """
    
    taxi_data = df[df['taxi'] == taxi_id]
    if taxi_data.empty:
        print(f"No data found for taxi {taxi_id}")
        return
    
    plt.figure(figsize=(15, 10))
    
    if color_by_label and 'label_name' in taxi_data.columns:
        label_colors = {
            'taxi_stand': 'blue',
            'taxi_central': 'red', 
            'taxi_street': 'green',
            'unknown': 'gray'
        }
        
        for label in taxi_data['label_name'].unique():
            label_data = taxi_data[taxi_data['label_name'] == label]
            if not label_data.empty:
                plt.scatter(label_data['lon'], label_data['lat'], 
                           alpha=0.7, s=20, 
                           color=label_colors.get(label, 'gray'),
                           label=f"{label} ({len(label_data)} points)")
        
        plt.legend()
        plt.title(f'Taxi {taxi_id} Trajectories (Colored by Location Type)')
    else:
        taxi_data['date'] = taxi_data['time'].dt.date
        dates = taxi_data['date'].unique()[:max_trips]
        
        colors = plt.cm.tab10(np.linspace(0, 1, len(dates)))
        
        for i, date in enumerate(dates):
            day_data = taxi_data[taxi_data['date'] == date]
            plt.plot(day_data['lon'], day_data['lat'], 'o-', alpha=0.7, 
                    color=colors[i], label=f"{date}", markersize=2)
        
        plt.legend()
        plt.title(f'Taxi {taxi_id} Trajectories (First {len(dates)} days)')
    
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.grid(True, alpha=0.3)
    plt.show()

def analyze_label_patterns(df):
    """
    Analyze patterns in taxi location labels
    """
    if 'label_name' not in df.columns:
        print("No label data available for analysis")
        return
    
    print("\n" + "="*50)
    print("LABEL PATTERN ANALYSIS")
    print("="*50)
    
    # Overall distribution of labels
    print("\n1. Overall Label Distribution:")
    label_counts = df['label_name'].value_counts()
    for label, count in label_counts.items():
        percentage = (count / len(df)) * 100
        print(f"   {label}: {count:,} points ({percentage:.1f}%)")
    
    # Hourly label patterns
    if not df.empty:
        df['hour'] = df['time'].dt.hour
        
        print("\n2. Hourly Activity Patterns:")
        hourly_labels = df.groupby(['hour', 'label_name']).size().unstack(fill_value=0)
        
        plt.figure(figsize=(15, 8))
        
        plt.subplot(2, 1, 1)
        for label in hourly_labels.columns:
            plt.plot(hourly_labels.index, hourly_labels[label], 
                    marker='o', label=label, linewidth=2)
        plt.xlabel('Hour of Day')
        plt.ylabel('Number of Points')
        plt.title('Location Types by Hour of Day')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Percentage view of hourly label patterns
        plt.subplot(2, 1, 2)
        hourly_pct = hourly_labels.div(hourly_labels.sum(axis=1), axis=0) * 100
        for label in hourly_pct.columns:
            plt.plot(hourly_pct.index, hourly_pct[label], 
                    marker='o', label=label, linewidth=2)
        plt.xlabel('Hour of Day')
        plt.ylabel('Percentage of Points (%)')
        plt.title('Location Types Distribution by Hour (Percentage)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

def get_trajectory_stats(df):
    """
    Get basic statistics about the trajectory data including labels
    """
    stats = {
        'total_points': len(df),
        'num_taxis': df['taxi'].nunique(),
        'date_range': (df['time'].min(), df['time'].max()),
        'lat_range': (df['lat'].min(), df['lat'].max()),
        'lon_range': (df['lon'].min(), df['lon'].max()),
        'points_per_taxi': df.groupby('taxi').size().describe()
    }
    
    # Add label statistics if the file has labels available
    if 'label_name' in df.columns:
        stats['label_distribution'] = df['label_name'].value_counts()
        stats['taxis_with_labels'] = df[df['label_name'] != 'unknown']['taxi'].nunique()
    
    return stats

def debug_plt_files(trajectory_dir="trajectory_data", max_check=5):
    """
    Debug function to check the content of the .plt files
    """
    subfolders = [f for f in os.listdir(trajectory_dir) if f.startswith('taxi_')]
    
    checked = 0
    for sf in subfolders[:max_check]:
        taxi_folder = os.path.join(trajectory_dir, sf)
        plt_files = glob.glob(os.path.join(taxi_folder, 'Trajectory', '*.plt'))
        
        if not plt_files:
            plt_files = glob.glob(os.path.join(taxi_folder, '*.plt'))
        
        print(f"\nChecking taxi {sf}:")
        print(f"  Found {len(plt_files)} .plt files")
        
        # Check for the labels file
        labels_file = os.path.join(taxi_folder, 'labels.txt')
        if os.path.exists(labels_file):
            print(f"  Labels file found: {os.path.getsize(labels_file)} bytes")
        else:
            print("  No labels file found")

        # Checks the first three files because we have too much data
        for plt_file in plt_files[:3]:
            file_size = os.path.getsize(plt_file) if os.path.exists(plt_file) else 0
            print(f"  File: {os.path.basename(plt_file)}, Size: {file_size} bytes")
            
            if file_size > 0:
                try:
                    with open(plt_file, 'r') as f:
                        lines = f.readlines()[:10] 
                    print(f"First few lines:")
                    for j, line in enumerate(lines):
                        print(f"    {j}: {line.strip()}")
                except Exception as e:
                    print(f"Error reading file: {e}")
            else:
                print("File is empty!")

        checked += 1
        if checked >= max_check:
            break

def interactive_analysis(df):
    """
    Interactive analysis menu
    """
    while True:
        print("\n" + "="*50)
        print("Porto Taxi Trajectory Analysis")
        print("="*50)
        print("1. Show dataset statistics")
        print("2. Visualize all trajectories")
        print("3. Visualize trajectories (colored by labels)")
        print("4. Visualize specific taxi")
        print("5. Visualize specific taxi (colored by labels)")
        print("6. Analyze label patterns")
        print("7. Show top 10 most active taxis")
        print("8. Save to pickle file")
        print("9. Exit")
        
        choice = input("\nSelect option (1-9): ")
        
        if choice == '1':
            stats = get_trajectory_stats(df)
            print("\nTrajectory Statistics:")
            for key, value in stats.items():
                print(f"{key}: {value}")
                
        elif choice == '2':
            visualize_trajectories(df, color_by_label=False)
            
        elif choice == '3':
            visualize_trajectories(df, color_by_label=True)
            
        elif choice == '4':
            taxi_ids = df['taxi'].unique()
            print(f"Available taxi IDs: {sorted(taxi_ids)[:10]}...")
            try:
                taxi_id = int(input("Enter taxi ID: "))
                visualize_individual_taxi(df, taxi_id, color_by_label=False)
            except ValueError:
                print("Invalid taxi ID")
                
        elif choice == '5':
            taxi_ids = df['taxi'].unique()
            print(f"Available taxi IDs: {sorted(taxi_ids)[:10]}...")
            try:
                taxi_id = int(input("Enter taxi ID: "))
                visualize_individual_taxi(df, taxi_id, color_by_label=True)
            except ValueError:
                print("Invalid taxi ID")
                
        elif choice == '6':
            analyze_label_patterns(df)
            
        elif choice == '7':
            top_taxis = df.groupby('taxi').size().sort_values(ascending=False).head(10)
            print("\nTop 10 most active taxis:")
            for taxi_id, count in top_taxis.items():
                print(f"Taxi {taxi_id}: {count} GPS points")
                
        elif choice == '8':
            filename = input("Enter filename (default: porto_trajectories.pkl): ").strip()
            if not filename:
                filename = 'porto_trajectories.pkl'
            df.to_pickle(filename)
            print(f"Saved to '{filename}'")
            
        elif choice == '9':
            print("Goodbye!")
            break
            
        else:
            print("Invalid option. Please select 1-9.")


if __name__ == "__main__":
    debug_option = input("Debug .plt files first? (y/n): ")
    if debug_option.lower() == 'y':
        print("Debugging .plt files...")
        debug_plt_files("trajectory_data")
    
    # Load all taxi trajectories
    df = load_porto_trajectories("trajectory_data")
    
    if not df.empty:
        # Start interactive analysis
        interactive_analysis(df)
    else:
        print("No data loaded. Please check your trajectory_data folder.")