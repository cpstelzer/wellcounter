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
    count_df, particles_df = wim.count_particles(run_folder_path)


    aggregated_df, frame_stats_df = wim.count_complete(run_folder_path)

    #males_df,_ = wma.count_males(run_folder_path)
    # Generate long-exposure image and analyze particles
    #ref_frame_no = 0
    #rec_direction = 'forward'
    
    #merged_df = wma.run_male_analysis(
    #    run_folder_path,
    #    ref_frame_no,
    #    rec_direction,
    #)
    
    # Perform motion analysis
    #motion_df = wmm.perform_motion_analysis(run_folder_path)


    # Print the results
    print("\n" + "="*40)
    print("Analysis of", os.path.basename(run_folder_path), "complete:")
    print("="*40)
    print("\nParticle Count Results:")
    print(count_df)
    print("\nParticle Details:")
    print(particles_df)
    print("\nAggregated Count Results:")
    print(aggregated_df)
    print("\nFrame-wise Count Statistics:")
    print(frame_stats_df)
    #print("\nMale Count Results:")  
    #print(males_df)

    #print("\nMerged LEI/ref-frame metrics:")
    #print(merged_df)
    
    #print("\nMotion Analysis Results:")
    #print(motion_df)
    
