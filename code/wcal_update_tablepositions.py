# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:48:59 2023

@author: Stelzer Lab
"""

import csv
import os

# Path to the input files
origins_file_path = "C:/CodeLab/wellcounter/code/wellpositions_origins.csv"
results_file_path = "C:/wellcounter/Kurs_2025/subtracted_images/circle_detection_results.csv"

# Path to the output file
output_file_path = "C:/CodeLab/wellcounter/code/wellpositions_origins_driftcompensated.csv"

# Read origins file
origins_data = []
with open(origins_file_path, 'r') as origins_file:
    origins_reader = csv.reader(origins_file)
    next(origins_reader)  # Skip the header row
    for row in origins_reader:
        origins_data.append(row)

# Read results file
results_data = []
with open(results_file_path, 'r') as results_file:
    results_reader = csv.reader(results_file)
    next(results_reader)  # Skip the header row
    for row in results_reader:
        results_data.append(row)

# Perform calculations and generate new data
new_data = [["plate", "well", "originX", "originY"]]  # Header row
for origins_row in origins_data:
    plate, well, originX, originY = origins_row

    for results_row in results_data:
        _, results_plate, results_well, centerX, centerY, _, correctX, correctY, _ = results_row
        if plate == results_plate and well == results_well:
            new_originX = "{:.3f}".format(float(originX) + float(correctX))
            new_originY = "{:.3f}".format(float(originY) + float(correctY))
            new_data.append([plate, well, new_originX, new_originY])
            break

# Write new data to the output file
with open(output_file_path, 'w', newline='') as output_file:
    output_writer = csv.writer(output_file)
    output_writer.writerows(new_data)

print("Output file generated successfully!")
