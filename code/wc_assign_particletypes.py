# -*- coding: utf-8 -*-
"""
Script: wc_assign_particletypes

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
This script demonstrates a use-case of the wellcounter_review_module: Establishing
a ground truth dataset of all true particles detectable in the first frame of a movie.
This script allows you to assign one the following categories to each particle: 
    - true detection
    - false positive
    - false negative
Such ground truth data is important if you want to optimize imaging parameters for a 
new study organism or if you made technical changes to your imaging system.

Requirements:
1. An input video file.
2. A 'table_of_particles' CSV file containing information about identified particles 
in the first frame of the video. This table can be generated using the imaging module 
(refer to the example script: wc_analyze_one_sample.py).

Instructions:
- Place the video file in the specified directory (data_dir).
- Run image analysis on the video (e.g., wc_analyze_one_sample.py). This will generate a folder with the same name as the
  video and the extension "_particle_detection", which contains the table_of_particles.csv file
  Run this script by typing: python wc_assign_particletypes.py

Output:
- A new table_of_particles_cat.csv file containing your particle category assignments.
- An image of the first frame with all particles labeled according to their categories.

Example data: You can use the test movie 20240206_fems30_plate42_well2.mp4 (provided as 
supplementary data), which displays a well with 30 rotifers.

Author: Claus-Peter Stelzer
Date: 2025-02-07

"""

import wellcounter_review_module as wrm
import wellcounter_imaging_module as wim
import os
import cv2
import pandas as pd


# Define the directory and video file to be analyzed
data_dir = "C:/wellcounter/sandbox"  # Base directory containing the video file
video_file = "20240206_fems30_plate42_well2.mp4"
video_path = os.path.join(data_dir, video_file)

# Define a folder named after the video file (without extension)
output_dir = os.path.join(data_dir, os.path.splitext(video_file)[0] + "_particle_detection")
os.makedirs(output_dir, exist_ok=True)  # Create the directory if it doesn't already exist

# Extract the first frame of the video file and save it as an image in the output directory
image_path = os.path.join(output_dir, "first_frame.jpg")
first_frame = wim.extract_frame(video_path, delay=0)
cv2.imwrite(image_path, first_frame)
print(f"First frame extracted and saved to {image_path}")

# Path to the table of particles CSV file within the new directory
path_to_top = os.path.join(output_dir, "table_of_particles.csv")

# Read the table of particles from the CSV file
table_of_particles = pd.read_csv(path_to_top)
print("Initial table of particles:")
print(table_of_particles)

# Identify false positives and update the DataFrame
table_of_particles_fp = wrm.user_categorize_particles(video_path, table_of_particles)
print("Table of particles with false positives:")
print(table_of_particles_fp)

# Save the updated table of particles with category assignments to the new directory
output_table_path = os.path.join(output_dir, "table_of_particles_fp.csv")
table_of_particles_fp.to_csv(output_table_path, index=False)
print(f"Updated table of particles saved to {output_table_path}")

# Identify false negatives and update the DataFrame
table_of_particles_cat = wrm.detect_false_negatives(video_path, image_path, table_of_particles_fp)
output_table_path = os.path.join(output_dir, "table_of_particles_cat.csv")
table_of_particles_cat.to_csv(output_table_path, index=False)
print("\nFalse negative detection complete. Results saved to table_of_particles_cat.csv")

# Create an annotated image of the first frame with all particles labeled by their categories
annotated_image_path = os.path.join(output_dir, "first_frame_typelabels.jpg")
wrm.label_particletypes(image_path, table_of_particles_cat)
print(f"Annotated image saved as {annotated_image_path}")

