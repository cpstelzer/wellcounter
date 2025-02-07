# -*- coding: utf-8 -*-
"""
This script automates the process of recording experimental data with the WELLCOUNTER 
using a combination of serial communication, image acquisition, and video recording. 
The main functionalities include controlling an XY-scanning table, acquiring images with 
a camera, subtracting images, and recording videos. The script performs the following steps:

1. Initializes serial connections to control an XY-scanning table and a USB relay for lighting control.
2. Moves the XY-scanning table to specified well positions to capture data.
3. Acquires images using a Basler camera and performs image subtraction.
4. Records a video at each well using the camera.
5. Logs the current position and video recording times to CSV files.

Key components and their functionalities:
- Serial communication setup for controlling the XY-scanning table and Numato USB relay.
- Functions for sending Gcode commands, moving the table, and updating/getting the current position.
- Image acquisition and subtraction using the pypylon library and OpenCV.
- Video recording using the pypylon library and OpenCV.
- Main execution flow that reads positions from a CSV file, moves the table, controls lighting, acquires data, and records videos.

The script assumes specific hardware configurations and paths for data storage, which may need to be adjusted 
based on the actual experimental setup.

Dependencies: csv, serial, time, cv2, os, pypylon, datetime, math, pandas

Usage:
Run the script and provide the batch number when prompted. Ensure that the necessary hardware and setup conditions 
are met before execution (for more details see "WELLCOUNTER: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates 
                          Using Image Acquisition and Analysis with Multiwell Plates".

"""

import csv
import serial
import time
import cv2
import os
from pypylon import pylon
from datetime import date
import math
import pandas as pd


# Serial port configuration
ser = serial.Serial()
ser.baudrate = 9600
ser.port = 'COM5'

# Set up serial connection to Numato USB relay (controls light)
portName = "COM4"  # Adjust if necessary
relayNum = "1"     # Relay number to control
numato = serial.Serial(portName, 19200, timeout=1)

# Path to store the subtracted images
output_folder = "D:/popgrowth_20240313/"

# Delay between sending Gcode commands (in seconds)
command_delay = 3

# Control the delay after capturing each image
image_capture_delay = 2

# XY-scanning table movement parameters
speed = 1.6  # cm/s
acceleration = 1  # mm/s^2

# Global variable to store the previous position
prev_position = (0, 0)

# Add a global variable to store the path to the movies folder
movies_folder = "D:/popgrowth_20240403/"

# Global variable to specify the duration of the video recording (in seconds)
video_duration = 15

# Global variable to specify the frame rate for video recording
frame_rate = 25

def send_gcode_command(command):
    """Send a Gcode command to the XY-scanning table"""
    ser.write(command.encode('utf-8'))
    ser.readline()
    time.sleep(command_delay)

def move_to_position(x, y):
    """Move the XY-scanning table to the specified position"""
    
    gcode_command = f"G1 X{x} Y{y}\n"
    send_gcode_command(gcode_command)
    ser.readline()
    time.sleep(command_delay)

    global prev_position

    # Calculate the traveling distance
    distance = math.sqrt((x - prev_position[0]) ** 2 + (y - prev_position[1]) ** 2)

    print("Previous position:", prev_position[0], ", ", prev_position[1])
    print("New position:", x, ", ", y)
    print("Distance to travel:", distance)
    

    # Calculate the traveling delay based on speed and acceleration
    traveling_delay = distance / speed + (speed / acceleration)

    print("Traveling delay:", traveling_delay)
    print("") # Empty line

    # Add the traveling delay
    time.sleep(traveling_delay)

    # Update the previous position
    prev_position = (x, y)

    # Add a delay after moving to the new position
    time.sleep(image_capture_delay)
    
def update_current_position(position):
    """Update the current position in the CSV file"""
    with open(csv_file, "a") as file:
        writer = csv.writer(file)
        writer.writerow(["", position[0], position[1]])

def get_current_position():
    """Get the previous position from the CSV file or return the initial position"""
    if os.path.isfile(csv_file):
        with open(csv_file, "r") as file:
            reader = csv.reader(file)
            last_row = None
            for row in reader:
                last_row = row
            if last_row:
                _, plate, x, y = last_row
                return float(x), float(y)
    return 0, 0  # Return the initial position if the CSV file doesn't exist or is empty


def acquire_images():
    """Acquire two images with a time interval of 2 seconds"""

    # Connect to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Start the camera
    camera.Open()

    # Set camera parameters if needed (e.g., exposure time, gain, etc.)
    camera.ExposureTime.SetValue(15000)  # Set exposure time to 15 ms

    # Capture picture A
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Convert the grabbed image to a numpy array or perform other processing
        image_a = grabResult.Array

    # Release the grab result and stop the grabbing
    grabResult.Release()
    camera.StopGrabbing()

    # Add a delay after capturing the first image
    time.sleep(image_capture_delay)

    # Capture picture B
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
    grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

    if grabResult.GrabSucceeded():
        # Convert the grabbed image to a numpy array or perform other processing
        image_b = grabResult.Array

    # Release the grab result and stop the grabbing
    grabResult.Release()
    camera.StopGrabbing()

    # Close the camera
    camera.Close()

    return image_a, image_b

  
def save_subtracted_images(image_a, image_b, output_folder, current_date, plate, well, batch):
    """Subtract image B from image A and save the two subtracted pictures"""
    subtracted_image_1 = cv2.subtract(image_a, image_b)
    subtracted_image_2 = cv2.subtract(image_b, image_a)

    filename_1 = f"{current_date}_batch{batch}_plate{plate}_well{well}_a.jpg"
    filename_2 = f"{current_date}_batch{batch}_plate{plate}_well{well}_b.jpg"

    output_path_1 = os.path.join(output_folder, filename_1)
    output_path_2 = os.path.join(output_folder, filename_2)

    cv2.imwrite(output_path_1, subtracted_image_1)
    cv2.imwrite(output_path_2, subtracted_image_2)
    
    # Save original image (for diagnostics only)
    #cv2.imwrite(output_path_1, image_a)
 
    

def record_video(current_date, plate, well, batch):
    

    video_filename = f"{current_date}_batch{batch}_plate{plate}_well{well}.mp4"
    output_path = os.path.join(movies_folder, video_filename)

    # Connect to the first available camera
    camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())

    # Start the camera
    camera.Open()

    # Set camera parameters if needed (e.g., exposure time, gain, etc.)
    camera.ExposureTime.SetValue(15000)  # Set exposure time to 15 ms

    # Create a VideoWriter object to record the video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec

   # Disable OpenVX backend to avoid GStreamer conflicts
    cv2.setUseOpenVX(False)

    video_writer = cv2.VideoWriter(output_path, fourcc, frame_rate, (camera.Width.GetValue(), camera.Height.GetValue()))

    # Start grabbing once
    camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)

    # Start recording
    start_time = time.time()

    while (time.time() - start_time) < video_duration:
        grab_result = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)

        if grab_result.GrabSucceeded():
            # Convert the grabbed image to a numpy array or perform other processing
            image_frame = grab_result.Array
            video_writer.write(image_frame)

        # Release the grab result
        grab_result.Release()

    # Stop recording and release the VideoWriter object
    camera.StopGrabbing()
    video_writer.release()
    
    # Stop the stopwatch
    end_time = time.time()
    video_rec_time = end_time - start_time
    print("Video recording time: ", video_rec_time, "secs")

    # Close the camera
    camera.Close()
    
    return video_rec_time
    

def main(csv_file): 
    
    # Start the stopwatch
    start_time = time.time()

    # Open serial port
    ser.open()
    ser.readline()

    # Unlock table
    send_gcode_command("$x\n")
    
    video_log = pd.read_csv(os.path.join(movies_folder, 'video_log.csv'))

    # Load data from CSV file
    with open(csv_file, "r") as file:
        reader = csv.reader(file)
        next(reader)  # Skip header row if present

        # Get the current date
        current_date = date.today().strftime("%Y%m%d")

        for row in reader:
            plate, well, originX, originY = row
    
            x = float(originX) 
            y = float(originY) 
           
            print("Plate:", plate, " Well:", well)
            
            # Move the XY-scanning table to the initial position
            move_to_position(x, y)
            
            time.sleep(5)  # Wait for 5 secs in the dark
            
            # Turn the LED light on
            numato.write("relay on {}\n\r".format(relayNum).encode())
            print("Relay {} is ON".format(relayNum))
            time.sleep(1)  # Wait for 1 second
            
            # Acquire images (uncomment, if images - in addition to movies - shall be recorded)
            #image_a, image_b = acquire_images()
    
            # Create output filenames and save the subtracted images
            #save_subtracted_images(image_a, image_b, output_folder, current_date, plate, well, batch)
                        
            # Record video at the current position
            video_rec_time = record_video(current_date, plate, well, batch)
            
            # Logging actual video recording time
            curr_log = pd.DataFrame({
                'current_date': [current_date],
                'fems': [batch],
                'plate':[plate],
                'well': [well],
                'video_rec_time': [video_rec_time],
            })
            
            # Collecting the video recording times of different wells
            video_log = pd.concat([video_log, curr_log], ignore_index=True)
            
            # Turn the relay off
            numato.write("relay off {}\n\r".format(relayNum).encode())
            print("Relay {} is OFF".format(relayNum))
            


    # Move the XY-scanning table to the origin (drift compensated)
    move_to_position(0, 0)

    # Close the serial port
    ser.close()
    
    # Saving the video logfile
    video_log.to_csv(os.path.join(movies_folder, 'video_log.csv'), index=False)
    
    # Stop the stopwatch
    end_time = time.time()
    execution_time = end_time - start_time
    
    # Convert the execution time to hours, minutes, and seconds
    hours = int(execution_time // 3600)
    minutes = int((execution_time % 3600) // 60)
    seconds = int(execution_time % 60)

    # Print the total running time
    print(f"Total running time: {hours} hours {minutes} minutes {seconds} seconds")

if __name__ == "__main__":
    csv_file = "wellpositions_all.csv"  # File containing the positions of all plates and wells
    batch = int(input("Enter the batch number: "))  # Prompt user for batch number
    user_input = input("Please ensure that:\n"
                  "1) Plates are in their correct positions, and lids have been removed\n"
                  "2) Plates in columns 3-5 have been pushed to the left\n"
                  "3) White cardboard for alignment has been removed\n"
                  "4) All the lights in the room are turned off\n"
                  "(Press return to continue)\n")
    main(csv_file)
