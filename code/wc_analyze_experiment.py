# -*- coding: utf-8 -*-
"""
Script: wc_analyze_experiment (Modified for FPS-in-Filename)

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
This script batch-analyzes a WELLCOUNTER experiment from image sequences where
the FPS is embedded in the filenames.

Requirements:
1. Raw image sequence data stored in dedicated folders.
2. A CSV file named '...treatments.csv' containing treatment information.

Functionality:
Iterates through each sample defined in the treatment file, finds the
corresponding data folder, performs analysis, and combines the results.

Author: Claus-Peter Stelzer
Date: 2025-02-07
Modification Date: 2025-09-22
"""

import os
import pandas as pd
import wellcounter_imaging_module as wim
import wellcounter_motion_module as wmm
from datetime import datetime

# --- Configuration ---
main_dir = "D:/popgrowth_20250627/"
data_base_dir = os.path.join(main_dir, "image_sequences")
treat_file = "popgrowth_20250627_treatments.csv"
outfile = "popgrowth_20250627_results.csv"

# --- Main Analysis ---

treat_path = os.path.join(main_dir, treat_file)
outpath = os.path.join(main_dir, outfile)

try:
    treat_df = pd.read_csv(treat_path)
except FileNotFoundError:
    print(f"Error: Treatment file not found at {treat_path}")
    exit()

if os.path.exists(outpath):
    os.remove(outpath)

# Find all unique dates present in the data directory to search through
# This makes the script more flexible than a fixed date range
all_folders = [d for d in os.listdir(data_base_dir) if os.path.isdir(os.path.join(data_base_dir, d))]
dates_in_data = sorted(list(set([folder.split('_')[0] for folder in all_folders])))

print(f"Found data for dates: {dates_in_data}")

# Iterate through each sample from the treatments file
for index, row in treat_df.iterrows():
    batch_no = row['batch']
    plate_no = row['plate']
    well_no = row['well']
    
    # Try to find a matching folder for any of the available dates
    found_folder = False
    for date_str in dates_in_data:
        folder_name = f"{date_str}_plate{plate_no}_well{well_no}"
        run_folder_path = os.path.join(data_base_dir, folder_name)

        if os.path.isdir(run_folder_path):
            print("\n" + "="*50)
            print(f"Analyzing: {folder_name}")
            print("="*50)
            
            # Perform analysis (FPS is now determined automatically inside the functions)
            count_df = wim.count_particles(run_folder_path)
            motion_df = wmm.perform_motion_analysis(run_folder_path)
            
            # Create a DataFrame for the current row's data
            current_row_df = row.to_frame().T
            # Add the date to the row for completeness
            current_row_df['date'] = date_str
            
            # Join results
            concatenated_df = pd.concat([current_row_df.reset_index(drop=True), count_df, motion_df], axis=1)
            
            # Save results iteratively
            header = not os.path.exists(outpath)
            concatenated_df.to_csv(outpath, mode='a', index=False, header=header)
            
            found_folder = True
            break # Move to the next well in the treatment file
    
    if not found_folder:
        print(f"Warning: No data folder found for batch {batch_no}, plate {plate_no}, well {well_no} on any available date.")

print("\nAnalysis complete. Results saved to:", outpath)