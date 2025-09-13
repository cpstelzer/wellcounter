# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 10:28:37 2023

@author: Stelzer Lab
"""

import cv2
import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import re


def process_images(folder_path):
    # Create a CSV file for storing the results in the same folder as the original images
    csv_file = open(os.path.join(folder_path, "circle_detection_results.csv"), "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["date", "plate", "well", "centerX", "centerY", "diameter", "correctX", "correctY", "detectionStatus"])

    # Get a list of all image files in the folder
    image_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]

    # X- and Y-coordinate of a perfectly centered well
    center_ideal = np.array([2252, 2252])
    
    # Scaling factor pixels --> cm
    px_to_cm = 1/1375.12

    # Create dictionaries to store correction values for each plate
    correction_values_x = {}
    correction_values_y = {}

    for image_file in image_files:
        # Load the image
        image_path = os.path.join(folder_path, image_file)
        image = cv2.imread(image_path)

        # Check if the image loading was successful
        if image is None:
            print(f"Failed to load image: {image_file}")
            continue

        # Rotate the image 90° clockwise (necessary due to current camera mounting of the WELLCOUNTER!!!)
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)        

        # Extract the date, plate number, and well number from the image filename
        filename = os.path.splitext(image_file)[0]
        date, batch_str, plate_str, well_str = filename.split("_")
        plate = int(plate_str.lstrip("plate"))
        well = int(well_str.lstrip("well"))

        # Scale down the image
        scale_percent = 20  # Adjust the scale as needed
        width = int(image.shape[1] * scale_percent / 100)
        height = int(image.shape[0] * scale_percent / 100)
        dim = (width, height)
        resized_image = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

        # Convert the image to grayscale
        gray = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

        # Perform thresholding
        _, threshold = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # Find contours in the thresholded image
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Find the largest contour (assumed to be the circle)
        largest_contour = max(contours, key=cv2.contourArea)

        # Get the center and radius of the circle in the original image size
        (x, y), radius = cv2.minEnclosingCircle(largest_contour)
        center = np.array([int(x * (100 / scale_percent)), int(y * (100 / scale_percent))])
        radius = int(radius * (100 / scale_percent))
        deviation = center_ideal - center
        correct = -1*(deviation * px_to_cm)

        # Draw the perimeter of the circle on the resized image
        output_image = resized_image.copy()
        cv2.circle(output_image, tuple(center), radius, (0, 255, 0), 2)

        # Save the output image with the detected perimeter
        output_filename = os.path.splitext(image_file)[0] + "_output.jpg"
        output_path = os.path.join(folder_path, "output", output_filename)
        cv2.imwrite(output_path, output_image)

        # Write the results to the CSV file
        csv_writer.writerow([date, plate, well, center[0], center[1], 
                             radius * 2, round(correct[0], 3), round(correct[1],3), "Detected"])

        # Store the correction values for each plate
        if plate not in correction_values_x:
            correction_values_x[plate] = []
        if plate not in correction_values_y:
            correction_values_y[plate] = []
        correction_values_x[plate].append(correct[0]) 
        correction_values_y[plate].append(correct[1]) 

    # Close the CSV file
    csv_file.close()

    # Generate heatmaps for X and Y correction values
    generate_heatmap(correction_values_x, "CorrectX", folder_path)
    generate_heatmap(correction_values_y, "CorrectY", folder_path)


def generate_heatmap(correction_values, title, folder_path):
    # Create a list of plate numbers
    plate_numbers = sorted(correction_values.keys())

    # Arrange plate numbers into groups of 6
    arranged_plates = [plate_numbers[i:i + 6] for i in range(0, len(plate_numbers), 6)]

    # Rearrange the plate numbers to display vertically
    arranged_plates = [list(map(str, x)) for x in zip(*arranged_plates)]

    # Create a color map for the heatmap
    cmap = plt.cm.RdBu_r

    # Create a grid of subplots for heatmaps
    num_rows = len(arranged_plates)
    num_cols = 7
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(15, 10))

    for row_idx, row in enumerate(axes):
        for col_idx, ax in enumerate(row):
            # Set the title for each heatmap
            if row_idx == 0:
                ax.set_title(f"Plates {col_idx * num_rows + 1}-{(col_idx + 1) * num_rows}")

            plate_number_str = arranged_plates[row_idx][col_idx] if col_idx < len(arranged_plates[row_idx]) else None
            if plate_number_str:
                plate_number = int(plate_number_str)
                # Plot the heatmap for the corresponding plate
                values = correction_values[plate_number]
                im = ax.imshow([values], cmap=cmap, aspect='auto', vmin=-0.8, vmax=0.8)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                # Hide the empty subplot
                ax.axis('off')

    # Create a colorbar
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.5)
    cbar.ax.set_ylabel('Correction Value')

    # Set the overall title for the heatmap figure
    fig.suptitle(title, fontsize=14)

    # Save the heatmap figure in the output folder
    output_folder = os.path.join(folder_path, "output")
    os.makedirs(output_folder, exist_ok=True)
    heatmap_filename = f"{title.lower()}_heatmap.png"
    heatmap_path = os.path.join(output_folder, heatmap_filename)
    plt.savefig(heatmap_path)
    plt.close()


# Provide the path to the folder containing the images
folder_path = "C:/wellcounter/Kurs_2025/subtracted_images/"

# Process the images in the folder
process_images(folder_path)
