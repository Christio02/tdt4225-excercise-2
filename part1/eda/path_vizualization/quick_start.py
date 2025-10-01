import pandas as pd
import matplotlib.pyplot as plt
from visualize_paths import *

def quick_start():
    """Quick start for path visualization with the saved pickle file"""
    pickle_files = ['porto_trajectories.pkl']
    
    # Try to find the pickle file that was saved
    df = None
    for pickle_file in pickle_files:
        if os.path.exists(pickle_file):
            print(f"Found {pickle_file}, loading...")
            df = pd.read_pickle(pickle_file)
            break
    
    if df is None:
        print("No pickle file found. Run the full processing first.")
        return
    
    print(f"Loaded {len(df)} GPS points from {df['taxi'].nunique()} taxis")
    
    # Menu
    while True:
        print("\n" + "="*40)
        print("QUICK VISUALIZATION MENU")
        print("="*40)
        print("1. Visualize all trajectories")
        print("2. Visualize all trajectories (colored by labels)") 
        print("3. Label analysis")
        print("4. Visualize a specific taxi")
        print("5. Full interactive menu") # Connects to the full menu in visualize_paths.py
        print("6. Exit")
        
        choice = input("\nSelect (1-6): ")
        
        if choice == '1':
            visualize_trajectories(df, color_by_label=False)
        elif choice == '2':
            visualize_trajectories(df, color_by_label=True)
        elif choice == '3':
            analyze_label_patterns(df)
        elif choice == '4':
            taxi_id = int(input("Enter taxi ID: "))
            visualize_individual_taxi(df, taxi_id, color_by_label=True)
        elif choice == '5':
            interactive_analysis(df)
        elif choice == '6':
            break

if __name__ == "__main__":
    quick_start()