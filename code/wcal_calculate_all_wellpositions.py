# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:52:33 2023

@author: Stelzer Lab
"""

import csv

def calculate_positions(input_file, output_file):
    with open(input_file, 'r') as input_csv:
        reader = csv.DictReader(input_csv)
        rows = list(reader)

    output_rows = []

    for row in rows:
        plate = row['plate']
        originX = float(row['originX'])
        originY = float(row['originY'])

        # Calculate positions for each well
        well1 = (round(originX + 7.8, 2), round(originY + 3.9, 2))
        well2 = (round(originX + 3.9, 2), round(originY + 3.9, 2))
        well3 = (round(originX, 2), round(originY + 3.9, 2))
        well4 = (round(originX + 7.8, 2), round(originY, 2))
        well5 = (round(originX + 3.9, 2), round(originY, 2))
        well6 = (round(originX, 2), round(originY, 2))

        # Append the calculated positions to the output rows
        output_rows.append({'plate': plate, 'well': 1, 'X': well1[0], 'Y': well1[1]})
        output_rows.append({'plate': plate, 'well': 2, 'X': well2[0], 'Y': well2[1]})
        output_rows.append({'plate': plate, 'well': 3, 'X': well3[0], 'Y': well3[1]})
        output_rows.append({'plate': plate, 'well': 4, 'X': well4[0], 'Y': well4[1]})
        output_rows.append({'plate': plate, 'well': 5, 'X': well5[0], 'Y': well5[1]})
        output_rows.append({'plate': plate, 'well': 6, 'X': well6[0], 'Y': well6[1]})

    # Write the output rows to the output file
    with open(output_file, 'w', newline='') as output_csv:
        fieldnames = ['plate', 'well', 'X', 'Y']
        writer = csv.DictWriter(output_csv, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output_rows)

# Call the function with the input and output filenames
calculate_positions('C:/CodeLab/wellcounter/code/wellpositions_origins_driftcompensated.csv', 'C:/CodeLab/wellcounter/code/wellpositions_all_driftcompensated.csv')
