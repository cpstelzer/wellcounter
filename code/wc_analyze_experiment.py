# -*- coding: utf-8 -*-
"""
Script: wc_analyze_experiment

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
This script demonstrates the principle of batch-analyzing an entire 
WELLCOUNTER experiment. It processes raw data consisting of movies recorded from 
each batch/plate/well over consecutive days of the experiment (e.g., using 
the script 'wc_record_experiment.py'). 

Requirements:
1. Raw video data recorded for each batch/plate/well during the experiment.
2. A CSV file named '...treatments.csv' containing information on which 
   batch/plate/well corresponds to which treatment. The format of this file 
   should be as follows:
    
    batch,plate,well,ac
    1,1,1,23
    1,1,2,15
    1,1,3,23
    ...
    
    (in this example the column 'ac' is a clone identification number)

Functionality:
This script iterates through each sample for each day of the experiment, performs 
analyses on the populations (including counting and motion analysis), and combines 
the results with the information on the treatments.

Key Steps:
1. Load the treatment information from 'treatments.csv'.
2. Iterate through each date within the experiment period.
3. For each batch, plate, and well, derive the path to the corresponding video file.
4. Analyze the video to count the number of organisms and perform motion analysis.
5. Combine the analysis results with the treatment information.
6. Append the results to an output CSV file.

Author: Claus-Peter Stelzer
Date: 2025-02-07

"""

import os
import pandas as pd
import wellcounter_imaging_module as wim
import wellcounter_motion_module as wmm

main_dir = "C:/Users/Anaconda"
data_dir = "D:/popgrowth_20230822/movies"  # Location of videos
treat_file = "popgrowth_20240822_treatments.csv" # File containing info on treatments assigned to each well and plate
outfile = "popgrowth_20240822_results.csv" # Output of this analysis
start_date = 20230822
end_date = 20230828

# Load experiment csv-file
treat_df = pd.read_csv(os.path.join(main_dir, treat_file))

# Initialize an empty DataFrame to collect the results
result_df = pd.DataFrame()

outpath = os.path.join(main_dir, outfile)
  
date = start_date       
    
while date < end_date + 1: # Iterate through each date of the experiment
    
    treat = treat_df
 
    for index, row in treat.iterrows(): # Iterate through each batch, plate and well
        batch_no = row['batch']
        plate_no = row['plate']
        well_no = row['well']
        ac_no = row['ac'] # Contains information of treatment (here: clone number)

        # Derive video path
        video_file = f'{date}_batch{batch_no}_plate{plate_no}_well{well_no}.mp4'
        print("Video file currently analyzed:")
        print(video_file)
        video_path = os.path.join(data_dir, video_file)
        
        # Calculate avg. number of organisms based on three frames of the video
        count_df = wim.count_particles(video_path)      
        
        # Perform analysis of movement behavior
        motion_df = wmm.perform_motion_analysis(video_path)
                 
        # Summarize treatments and date
        tdata = {
            'date': [date],
            'batch': [batch_no],
            'plate': [plate_no],
            'well': [well_no],
            'ac': [ac_no]
        }
        treat_df = pd.DataFrame(tdata)
        
        # Join, collect and store results
        concatenated_df = pd.concat([treat_df, count_df, motion_df], axis=1) # horizontal concatenation
        result_df = result_df.append(concatenated_df, ignore_index=True)
    
        # Save the updated results to the output file after each iteration of the inner loop
        concatenated_df.to_csv(outpath, mode='a', index=False, header=not os.path.exists(os.path.join(main_dir, outfile)))
        
    
    # Update date-variable
    date += 1

        
        
