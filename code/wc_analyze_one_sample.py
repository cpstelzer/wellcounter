# -*- coding: utf-8 -*-
"""
Script: wc_analyze_one_sample (Modified for FPS-in-Filename)

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
This script counts microorganisms and analyzes their swimming behavior
from a single sample's image sequence. The FPS is automatically detected
from the image filenames.

To use this script:
1) Enter the path to the folder containing the image sequence below.
2) Execute the script.

Author: Claus-Peter Stelzer
Date: 2025-02-07
Modification Date: 2025-09-22
"""

import os
import pandas as pd
import wellcounter_imaging_module as wim
import wellcounter_motion_module as wmm
import wellcounter_male_module as wma

# --- Configuration ---
# Enter the full path to the folder containing the image sequence
run_folder_path = "C:/wellcounter/test/20251017_batch0_plate42_well6/"


# --- Analysis ---
if not os.path.isdir(run_folder_path):
    print(f"Error: The specified folder does not exist: {run_folder_path}")
else:
    # FPS is determined automatically by the modules
    
    # Calculate avg. number of organisms
    count_df,_ = wim.count_particles(run_folder_path)

    positions_df, long_exposure_image = wma.generate_long_exposure_image_custom(run_folder_path, 0.5, 12, 105)
    #maledetect_df = wim.analyze_long_exposure_particles_advanced(long_exposure_image, run_folder_path)
    maledetect_df = wma.analyze_long_exposure_particles_advanced(long_exposure_image, run_folder_path)
    # Perform motion analysis
    #motion_df = wmm.perform_motion_analysis(run_folder_path)

    # Print the results
    print("\n" + "="*40)
    print("Analysis of", os.path.basename(run_folder_path), "complete:")
    print("="*40)
    print("\nParticle Count Results:")
    print(count_df)
    #print(positions_df)
    print(maledetect_df)
    #print("\nMotion Analysis Results:")
    #print(motion_df)
    