# -*- coding: utf-8 -*-
"""
Created on Tue Jul 18 15:15:33 2023

@author: Stelzer Lab
"""

import os
import pandas as pd
import matplotlib.pyplot as plt

# Path to the CSV file
csv_file = 'C:/wellcounter/Kurs_2025/subtracted_images/circle_detection_results.csv'

# Output directory for storing scatter plots
output_dir = 'C:/wellcounter/Kurs_2025/subtracted_images/output/'

# Read the CSV file into a DataFrame
data = pd.read_csv(csv_file)

# Create scatter plot of plate vs. correctX
plt.figure()
plt.scatter(data['plate'], data['correctX'])
plt.xlabel('Plate')
plt.ylabel('Correct X')
plt.title('Plate vs. Correct X')

# Save the scatter plot
plt.savefig(os.path.join(output_dir, 'scatter_plot_correctX.png'))
plt.close()

# Create scatter plot of plate vs. correctY
plt.figure()
plt.scatter(data['plate'], data['correctY'])
plt.xlabel('Plate')
plt.ylabel('Correct Y')
plt.title('Plate vs. Correct Y')

# Save the scatter plot
plt.savefig(os.path.join(output_dir, 'scatter_plot_correctY.png'))
plt.close()

print("Scatter plots saved successfully.")
