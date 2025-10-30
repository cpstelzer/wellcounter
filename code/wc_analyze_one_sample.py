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
from the image filenames. It also demonstrates how to build a long-exposure
image (LEI), analyze male-type traces, and merge LEI trace metrics with
their corresponding particles in the chosen reference frame.

To use this script:
1) Enter the path to the folder containing the image sequence below.
2) Adjust the LEI parameters if needed (analysis duration, detection thresholds,
   reference frame index, and direction).
3) Execute the script (``python code/wc_analyze_one_sample.py``) to generate
   the LEI, evaluate long-exposure traces, and inspect the merged metrics
   printed to the console.

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

# Key LEI parameters used both for generation and matching
ANALYSIS_DURATION = 0.5
MICRO_THRESHOLD = 12
MIN_MICRO_AREA = 105
REF_FRAME_NO = 0
REC_DIRECTION = "forward"


# --- Analysis ---
if not os.path.isdir(run_folder_path):
    print(f"Error: The specified folder does not exist: {run_folder_path}")
else:
    # FPS is determined automatically by the modules
    
    # Calculate avg. number of organisms
    count_df,_ = wim.count_particles(run_folder_path)

    positions_df, long_exposure_image = wma.generate_long_exposure_image_custom(
        run_folder_path,
        analysis_duration=ANALYSIS_DURATION,
        microorganism_threshold=MICRO_THRESHOLD,
        min_microorganism_area=MIN_MICRO_AREA,
        ref_frame_no=REF_FRAME_NO,
        rec_direction=REC_DIRECTION,
    )

    maledetect_df = wma.analyze_long_exposure_particles_advanced(long_exposure_image, run_folder_path)

    try:
        merged_metrics, match_diagnostics = wma.match_lei_traces_to_reference_particles(
            maledetect_df,
            positions_df,
            run_folder_path,
            ref_frame_no=REF_FRAME_NO,
            rec_direction=REC_DIRECTION,
        )
    except ValueError as exc:
        print(f"\n[match_lei_traces_to_reference_particles] Could not complete matching: {exc}")
        merged_metrics = pd.DataFrame()
        match_diagnostics = pd.DataFrame()

    # Perform motion analysis
    #motion_df = wmm.perform_motion_analysis(run_folder_path)

    # Print the results
    print("\n" + "="*40)
    print("Analysis of", os.path.basename(run_folder_path), "complete:")
    print("="*40)
    print("\nParticle Count Results:")
    print(count_df)
    print(positions_df)
    print("\nLEI particle metrics:")
    print(maledetect_df)
    if not merged_metrics.empty:
        print("\nMerged LEI/reference-frame metrics:")
        print(merged_metrics)
        print("\nMatch diagnostics (including unmatched entries):")
        print(match_diagnostics)
    else:
        print("\nNo merged LEI/reference-frame matches were produced.")
    #print("\nMotion Analysis Results:")
    #print(motion_df)
    