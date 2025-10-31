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

    # Generate long-exposure image and analyze particles
    analysis_duration = 0.5
    microorganism_threshold = 12
    min_microorganism_area = 105
    ref_frame_no = 0
    rec_direction = 'forward'


    positions_df, long_exposure_image = wma.generate_long_exposure_image_custom(run_folder_path,
                                                                                analysis_duration,
                                                                                microorganism_threshold,
                                                                                min_microorganism_area,
                                                                                ref_frame_no,
                                                                                rec_direction)
    
    maledetect_df = wma.analyze_long_exposure_particles_advanced(long_exposure_image, run_folder_path)

    # --- Optional: match LEI traces back to reference-frame particles ---
    # The matching routine links each long-exposure trace (``maledetect_df``)
    # to the particle that seeded it in the reference frame contained in
    # ``positions_df``.  This demonstrates how to exercise the new utilities
    # during end-to-end analysis of a single sample.  The returned
    # ``merged_df`` contains LEI metrics enriched with reference-frame
    # measurements (prefixed with ``ref_``), whereas ``assignments_df``
    # provides diagnostic information about each match (distance to the
    # centerline, position along the trace, orientation cues, etc.).
    merged_df, assignments_df = wma.match_long_exposure_traces_to_reference_particles(
        positions_df,
        maledetect_df,
        run_folder_path=run_folder_path,
        ref_frame_no=ref_frame_no,
        rec_direction=rec_direction,
    )
    
    # Perform motion analysis
    #motion_df = wmm.perform_motion_analysis(run_folder_path)


     # --- Define output folder path (but don't create it yet) ---
    parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
    folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
    output_dir = os.path.join(parent_dir, f"{folder_name}_particle_analysis")

    merged_df.to_csv(os.path.join(output_dir, "merged_df.csv"), index=False)
    assignments_df.to_csv(os.path.join(output_dir, "assignments_df.csv"), index=False)

    # Print the results
    print("\n" + "="*40)
    print("Analysis of", os.path.basename(run_folder_path), "complete:")
    print("="*40)
    print("\nParticle Count Results:")
    print(count_df)
    print(positions_df)
    print(maledetect_df)
    print("\nMerged LEI/ref-frame metrics:")
    print(merged_df)
    print("\nTrace-to-reference assignments (diagnostics):")
    print(assignments_df)
    #print("\nMotion Analysis Results:")
    #print(motion_df)
    
