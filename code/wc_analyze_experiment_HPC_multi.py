# -*- coding: utf-8 -*-
"""
Analyze a finished wellcounter experiment

Author: Stelzer Lab
"""

import os
import sys
import pandas as pd
import wellcounter_imaging_module as wim
import wellcounter_motion_module as wmm

def analyze_experiment(date):
    working_dir = os.environ["SCRATCH"]
    data_dir = os.path.join(working_dir, f"wellcounter/popgrowth_20250627/movies")
    treat_file = os.path.join(working_dir, f"wellcounter/popgrowth_20250627/popgrowth_20250627_treatments.csv")
    outfile = f'popgrowth_{date}_results.csv'
    outpath = os.path.join(working_dir, f"wellcounter/popgrowth_20250627", outfile)

    # Load experiment csv-file
    treat = pd.read_csv(treat_file)

    # Check if the output file already exists and load the processed data
    if os.path.exists(outpath):
        processed_df = pd.read_csv(outpath)
        processed_entries = set(processed_df.apply(lambda row: (row['batch'], row['plate'], row['well']), axis=1))
    else:
        processed_entries = set()

    # Initialize an empty DataFrame to collect the results
    result_df = pd.DataFrame()

    for index, row in treat.iterrows():
        batch_no = row['batch']
        plate_no = row['plate']
        well_no = row['well']
        ac_no = row['ac']

        # Check if this entry has already been processed
        if (batch_no, plate_no, well_no) in processed_entries:
            print(f"Skipping already processed file: batch{batch_no}_plate{plate_no}_well{well_no}")
            continue

        # Derive video path
        video_file = f'{date}_batch{batch_no}_plate{plate_no}_well{well_no}.avi'
        print(f"Video file currently analyzed: {video_file}")
        video_path = os.path.join(data_dir, video_file)

        try:
            # Check if the video file exists before processing
            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"Video file {video_file} not found.")

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

            # Join count_df, motion_df, and treat_df horizontally into a single DataFrame
            concatenated_df = pd.concat([treat_df, count_df, motion_df], axis=1)

            # Concatenate the concatenated DataFrame with the existing result DataFrame
            result_df = pd.concat([result_df, concatenated_df], ignore_index=True)

            # Save the updated results to the output file after each iteration of the inner loop
            concatenated_df.to_csv(outpath, mode='a', index=False, header=not os.path.exists(outpath))

        except FileNotFoundError as fnf_error:
            print(fnf_error)
            continue
        except Exception as e:
            print(f"Error processing file {video_file}: {e}")
            continue

    print(f"Analysis for date {date} completed.")

if __name__ == "__main__":
    # Accept date as a command-line argument
    date = int(sys.argv[1])
    analyze_experiment(date)
