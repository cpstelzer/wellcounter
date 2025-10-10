# -*- coding: utf-8 -*-
"""
Script: wc_analyze_experiment (Modified for FPS-in-Filename, multi-date support)

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
Iterates through each sample defined in the treatment file, finds all
corresponding data folders across multiple dates, performs analysis for each,
and combines the results into a single output file.

Author: Claus-Peter Stelzer
Date: 2025-02-07
Modification Date: 2025-10-09 (Revised multi-date iteration)
"""

import os
import pandas as pd
import wellcounter_imaging_module as wim
import wellcounter_motion_module as wmm
from datetime import datetime

# --- Configuration ---
main_dir = "D:/wellcounter/popgrowth_20251001"
data_base_dir = os.path.join(main_dir, "image_sequences")
treat_file = "popgrowth_20251001_treatments.csv"
outfile = "popgrowth_20251001_results.csv"

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

# Identify all date prefixes from existing data folders
all_folders = [d for d in os.listdir(data_base_dir) if os.path.isdir(os.path.join(data_base_dir, d))]
dates_in_data = sorted(list(set([folder.split('_')[0] for folder in all_folders])))

print(f"Found data for dates: {dates_in_data}")

# --- Iterate through each sample in the treatment file ---
for index, row in treat_df.iterrows():
    batch_no = row['batch']
    plate_no = row['plate']
    well_no = row['well']
    found_folder = False

    print("\n" + "-"*60)
    print(f"Processing sample: batch={batch_no}, plate={plate_no}, well={well_no}")
    print("-"*60)

    # Check each date folder for matching samples
    for date_str in dates_in_data:
        folder_name = f"{date_str}_batch{batch_no}_plate{plate_no}_well{well_no}"
        run_folder_path = os.path.join(data_base_dir, folder_name)

        if os.path.isdir(run_folder_path):
            found_folder = True
            print("\n" + "="*50)
            print(f"Analyzing: {folder_name}")
            print("="*50)

            try:
                # Perform image analysis
                count_df = wim.count_particles(run_folder_path)
                # Optional: motion analysis can be re-enabled if desired
                # motion_df = wmm.perform_motion_analysis(run_folder_path)

                # Combine treatment metadata with analysis results
                current_row_df = row.to_frame().T
                current_row_df['date'] = date_str

                # Combine dataframes (without motion analysis)
                concatenated_df = pd.concat(
                    [current_row_df.reset_index(drop=True), count_df],
                    axis=1
                )

                # Save results iteratively
                header = not os.path.exists(outpath)
                concatenated_df.to_csv(outpath, mode='a', index=False, header=header)

            except Exception as e:
                print(f"Error analyzing {folder_name}: {e}")

    if not found_folder:
        print(f"Warning: No data folders found for batch {batch_no}, plate {plate_no}, well {well_no} on any date.")

print("\nAnalysis complete. Results saved to:", outpath)
