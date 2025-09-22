# -*- coding: utf-8 -*-
"""
Wellcounter acquisition module (Modified Version 2)

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
This script automates the process of recording experimental data with the WELLCOUNTER.
This modified version incorporates modern data handling features inspired by the
Wellscanner system:

1.  **Configuration via YAML**: Key parameters are loaded from `wc_config.yaml`.
2.  **Individual Frame Storage**: Saves a sequence of individual image frames.
3.  **Organized Folder Structure**: Each well is saved in a dedicated folder.
4.  **Self-Contained Metadata**: The FPS setting is embedded directly into each
    image's filename (e.g., ..._f00001_fps25.png).
5.  **Asynchronous Saving**: Uses a ThreadPoolExecutor for high-performance,
    non-blocking image saving.
6.  **Comprehensive Metadata Logging**: Creates detailed per-well and summary logs.

Dependencies: csv, serial, time, cv2, os, pypylon, datetime, math, pandas, yaml, concurrent.futures

Author: Claus-Peter Stelzer
Date: 2025-02-07
Modification Date: 2025-09-22

"""

import csv
import serial
import time
import cv2
import os
import yaml
import traceback
from pypylon import pylon
from datetime import datetime, date
import math
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Global Hardware and Movement Parameters ---
ser = serial.Serial()
ser.baudrate = 9600
ser.port = 'COM9'
portName = "COM4"
relayNum = "1"
numato = serial.Serial(portName, 19200, timeout=1)
command_delay = 3
speed = 1.6
acceleration = 1
prev_position = (0, 0)

# --- Configuration Loading ---
def load_config(config_path="wc_record_config.yaml"):
    print(f"Loading configuration from: {config_path}")
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print("Configuration loaded successfully.")
        return config
    except FileNotFoundError:
        print(f"FATAL ERROR: Configuration file not found at {config_path}")
        exit(1)
    except yaml.YAMLError as e:
        print(f"FATAL ERROR: Failed to parse configuration file: {e}")
        exit(1)

# --- XY Table Control Functions (Unchanged) ---
def send_gcode_command(command):
    ser.write(command.encode('utf-8'))
    ser.readline()
    time.sleep(command_delay)

def move_to_position(x, y):
    gcode_command = f"G1 X{x} Y{y}\n"
    send_gcode_command(gcode_command)
    ser.readline()
    time.sleep(command_delay)
    global prev_position
    distance = math.sqrt((x - prev_position[0]) ** 2 + (y - prev_position[1]) ** 2)
    print(f"Previous position: {prev_position[0]}, {prev_position[1]}")
    print(f"New position: {x}, {y}")
    print(f"Distance to travel: {distance:.2f}")
    traveling_delay = distance / speed + (speed / acceleration)
    print(f"Traveling delay: {traveling_delay:.2f}s\n")
    time.sleep(traveling_delay)
    prev_position = (x, y)
    time.sleep(2)

# --- Core Acquisition and Saving Function ---
def acquire_and_save_frames(executor, config, run_folder_path, current_date_str, plate, well):
    print(f"--- Starting Frame Acquisition for Plate {plate}, Well {well} ---")
    print(f"Outputting to folder: {run_folder_path}")

    duration = config['acquisition']['duration_sec']
    fps = config['acquisition']['fps']
    exposure = config['acquisition']['exposure_us']
    total_frames_to_record = int(duration * fps)
    
    output_format = config['output']['image_format'].lower()
    if output_format not in ['bmp', 'jpg']:
        output_format = 'png'
    print(f"Saving frames as '.{output_format}'")

    log_fieldnames = [
        "timestamp", "frame_count", "pylon_frame_id",
        "frame_grab_time_ms", "image_save_submit_time_ms",
        "total_loop_iteration_time_ms", "output_filename"
    ]
    log_data = []
    
    camera = None
    try:
        camera = pylon.InstantCamera(pylon.TlFactory.GetInstance().CreateFirstDevice())
        camera.Open()
        camera.ExposureTime.SetValue(exposure)
        
        print(f"Attempting to acquire {total_frames_to_record} frames over {duration}s at {fps} FPS...")
        camera.StartGrabbing(pylon.GrabStrategy_LatestImageOnly)
        
        start_acquisition_time = time.perf_counter()
        futures = {}

        for frame_count in range(total_frames_to_record):
            loop_iter_start_time = time.perf_counter()
            log_entry = {}
            
            grabResult = None
            try:
                grab_start_time = time.perf_counter()
                grabResult = camera.RetrieveResult(5000, pylon.TimeoutHandling_ThrowException)
                grab_end_time = time.perf_counter()
                log_entry["frame_grab_time_ms"] = (grab_end_time - grab_start_time) * 1000

                if grabResult.GrabSucceeded():
                    frame_image = grabResult.Array
                    pylon_frame_id = grabResult.GetBlockID() if hasattr(grabResult, "GetBlockID") else "N/A"
                    
                    # --- FILENAME CHANGE ---
                    # Embed the FPS setting directly into the filename.
                    img_filename = (f"{current_date_str}_plate{plate}_well{well}_"
                                    f"f{frame_count:05d}_fps{int(fps)}.{output_format}")
                    # --- END OF CHANGE ---

                    output_path = os.path.join(run_folder_path, img_filename)

                    save_submit_start_time = time.perf_counter()
                    future = executor.submit(cv2.imwrite, output_path, frame_image)
                    futures[future] = output_path
                    save_submit_end_time = time.perf_counter()

                    log_entry["timestamp"] = datetime.now().isoformat()
                    log_entry["frame_count"] = frame_count
                    log_entry["pylon_frame_id"] = pylon_frame_id
                    log_entry["image_save_submit_time_ms"] = (save_submit_end_time - save_submit_start_time) * 1000
                    log_entry["output_filename"] = output_path
                else:
                    print(f"Frame {frame_count} grab failed: {grabResult.GetErrorDescription()}")
            
            finally:
                if grabResult:
                    grabResult.Release()
            
            loop_iter_end_time = time.perf_counter()
            log_entry["total_loop_iteration_time_ms"] = (loop_iter_end_time - loop_iter_start_time) * 1000
            log_data.append(log_entry)

        print("Finished acquisition loop. Waiting for file saving to complete...")
        
        saved_count, error_count = 0, 0
        for future in as_completed(futures):
            try:
                future.result()
                saved_count += 1
            except Exception as e:
                print(f"ERROR saving frame {futures[future]}: {e}")
                error_count += 1
        print(f"Successfully saved {saved_count} frames with {error_count} errors.")

        log_file_path = os.path.join(run_folder_path, config['paths']['log_filename'])
        pd.DataFrame(log_data).to_csv(log_file_path, index=False)
        print(f"Detailed metadata log saved to: {log_file_path}")
        
        return {'status': 'success' if error_count == 0 else 'error', 'frames_saved': saved_count, 'fps_setting': fps}

    except Exception as e:
        print(f"FATAL ERROR during acquisition for well {well}: {e}")
        traceback.print_exc()
        return {'status': 'failed', 'frames_saved': 0, 'fps_setting': fps}
    
    finally:
        if camera and camera.IsGrabbing(): camera.StopGrabbing()
        if camera and camera.IsOpen(): camera.Close()
        print("--- Camera released ---")

# --- Logging Functions ---
def initialize_summary_log(log_path):
    if not os.path.exists(log_path):
        with open(log_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                "datetime", "batch", "plate", "well",
                "status", "frames_saved", "fps_setting", "run_folder"
            ])
        print(f"Initialized experiment summary log: {log_path}")

def append_to_summary_log(log_path, batch, plate, well, results, run_folder):
    try:
        with open(log_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                batch, plate, well,
                results.get('status', 'failed'),
                results.get('frames_saved', 0),
                results.get('fps_setting', 0),
                run_folder
            ])
    except Exception as e:
        print(f"Error writing to summary log {log_path}: {e}")

# --- Main Execution Block ---
def main(csv_file, batch, config):
    start_time = time.time()
    
    base_output_dir = config['paths']['output_folder_base']
    summary_log_path = os.path.join(base_output_dir, config['paths']['experiment_summary_log_filename'])
    initialize_summary_log(summary_log_path)

    with ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        try:
            ser.open()
            ser.readline()
            send_gcode_command("$x\n")

            with open(csv_file, "r") as file:
                reader = csv.reader(file)
                next(reader)

                current_date_str = date.today().strftime("%Y%m%d")

                for row in reader:
                    plate, well, originX, originY = row
                    x, y = float(originX), float(originY)
                    
                    print(f"\n================ PROCESSING PLATE: {plate}, WELL: {well} ================")
                    
                    run_folder_name = f"{current_date_str}_plate{plate}_well{well}"
                    run_folder_path = os.path.join(base_output_dir, run_folder_name)
                    os.makedirs(run_folder_path, exist_ok=True)
                    
                    move_to_position(x, y)
                    time.sleep(5)
                    
                    numato.write(f"relay on {relayNum}\n\r".encode())
                    print(f"Relay {relayNum} is ON")
                    time.sleep(1)
                    
                    results = acquire_and_save_frames(executor, config, run_folder_path, current_date_str, plate, well)
                    
                    numato.write(f"relay off {relayNum}\n\r".encode())
                    print(f"Relay {relayNum} is OFF")
                    
                    append_to_summary_log(summary_log_path, batch, plate, well, results, run_folder_path)

            move_to_position(0, 0)

        except Exception as e:
            print(f"An error occurred in the main loop: {e}")
            traceback.print_exc()
        
        finally:
            if ser.is_open:
                ser.close()
            print("Serial port closed.")

    end_time = time.time()
    execution_time = end_time - start_time
    hours, rem = divmod(execution_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print(f"\nTotal running time: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")

if __name__ == "__main__":
    try:
        config = load_config()
        csv_file = "C:/CodeLab/wellcounter/code/wellpositions_one.csv"
        batch = int(input("Enter the batch number: "))
        user_input = input("Please ensure that:\n"
                      "1) Plates are in their correct positions, and lids have been removed\n"
                      "2) Plates in columns 3-5 have been pushed to the left\n"
                      "3) White cardboard for alignment has been removed\n"
                      "4) All the lights in the room are turned off\n"
                      "(Press return to continue)\n")
        main(csv_file, batch, config)
    except Exception as e:
        print(f"An unexpected error occurred at the top level: {e}")