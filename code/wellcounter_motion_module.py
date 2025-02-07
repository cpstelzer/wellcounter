# -*- coding: utf-8 -*-
"""
Wellcounter motion module

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter


Description:
This module contains several low-level and high-level functions for analyzing the
swimming behavior of microorganisms recorded in the WELLCOUNTER. Particle 
identification is done with the wellcounter_imaging_module.

Note: Portions of the code in this file were generated using ChatGPT v4.0.
      All AI-generated content has been rigorously validated and tested by the 
      authors. The corresponding author accepts full responsibility for the 
      AI-assisted portions of the code.

Author: Claus-Peter Stelzer
Date: 2025-02-07

"""

import cv2
import wellcounter_imaging_module as wim
import pandas as pd
import numpy as np
import random
import os
import yaml


# Load global config
config_path = "wellcounter_config.yml"
with open(config_path, 'r') as file:
    config = yaml.safe_load(file)
    
    
def record_particle_positions_in_video(video_path):
    
    """
    Records the positions of particles in a video and generates a long exposure image.

    This function processes a video file to detect and record the positions of particles in each frame, 
    within a specified analysis duration. It also generates a long exposure image showing the cumulative paths 
    of these particles.

    The function operates as follows:
    - Opens the video file and determines the frame rate.
    - Calculates the number of frames to process based on the analysis duration and frame rate.
    - Iterates over the specified number of frames, performing the following steps for each frame:
        - Extracts a frame from the video with a specific delay.
        - Performs image subtraction and particle analysis on the extracted frame using specified particle_detection parameters.
        - Records the positions and other attributes of particles detected in the frame.
        - Updates the long exposure image with the trajectory of particles in the current frame.
    - The process results in a DataFrame containing particle information for each processed frame and a long exposure image.

    Args:
        video_path (str): Path to the video file for analysis.
        analysis_duration (float): Duration (in seconds) of the video to be analyzed.
        particle_detection_params (dict): Parameters for particle detection and particle_detection processing.

    Returns:
        tuple: A tuple containing two elements:
               - A DataFrame with recorded particle positions and other attributes for each processed frame.
               - A long exposure image showing the cumulative paths of particles across the processed frames.
    """
    
    motion_params = config['motion']  
    
    # Read frame rate
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        exit()
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    cap.release()

    first_frame = wim.extract_frame(video_path, delay=0)
    height, width = first_frame.shape
    step = 1 / frame_rate
    number_of_iterations = int(motion_params['analysis_duration'] * frame_rate)
    long_exposure_image = np.zeros((height, width), dtype=np.uint8)
    result_df = pd.DataFrame()

    for i in range(number_of_iterations):
        print(f"Motion analysis\nProcessing frame no. {i} of {number_of_iterations}")
        secsA = i * step
        secsB = secsA + 5
        subtr_image, masked_image = wim.image_subtraction_from_video(video_path, delay1=secsA, delay2=secsB)
        table_of_particles, binary_image = wim.analyze_microorganisms(subtr_image)
        table_of_particles.insert(0, 'frame', i + 1)
        result_df = pd.concat([result_df, table_of_particles], ignore_index=True)
        long_exposure_image = cv2.add(long_exposure_image, binary_image)
        long_exposure_image = np.clip(long_exposure_image, 0, 255).astype(np.uint8)
    
    return result_df, long_exposure_image


def track_particles(particles_by_frame):
    
    """
    Tracks particles across frames and assigns them to trajectories based on proximity.

    This function iterates through particles sorted by frame number and assigns each particle 
    to the closest existing trajectory. A new trajectory is created if no existing trajectory 
    is close enough. The function calculates the Euclidean distance between particles and the 
    last point in each trajectory to determine closeness. Trajectories that are shorter than 
    a predefined minimum size are removed at the end.

    Args:
        particles_by_frame (DataFrame): A DataFrame containing particle information for each frame. 
                                        Each row should have at least 'frame', 'X', 'Y', and 'area' columns.
        motion_params (dict): A dictionary containing parameters for motion analysis. 
                              Should include 'max_search_distance' (the maximum distance to search for an 
                              existing trajectory) and 'min_trajectory_size' (the minimum acceptable size 
                              for a trajectory).

    Returns:
        dict: A dictionary of trajectories where each key is a trajectory ID and the value is a list of tuples.
              Each tuple represents a particle's frame number, X and Y coordinates, and other attributes (e.g., area).
              The trajectories are filtered to remove those shorter than the minimum size threshold.
    """
    
    motion_params = config['motion'] 
    
    # Initialize an empty dictionary to store the trajectories.
    trajectories = {}

    # Sort the input DataFrame by frame number for chronological processing.
    input_data = particles_by_frame.sort_values(by='frame')

    # Iterate through each row (particle) in the sorted DataFrame.
    for _, row in input_data.iterrows():
        # Extract frame number, x and y coordinates, and area from the current row.
        frame, x, y, area = row['frame'], row['X'], row['Y'], row['area']

        # Initialize variables to track the closest trajectory and its distance.
        closest_id = None
        min_distance = float('inf')

        # Iterate through each existing trajectory to find the closest one.
        for obj_id, trajectory in trajectories.items():
            # Get the last point (frame, x, y, area) of the current trajectory.
            last_frame, last_x, last_y, last_area = trajectory[-1]

            # Calculate the Euclidean distance between the current particle and the last point of this trajectory.
            distance = ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5

            # Check if this trajectory is the closest one so far and within the max search distance.
            if distance <= motion_params['max_search_distance'] and distance < min_distance:
                # Update the closest trajectory and the minimum distance.
                min_distance = distance
                closest_id = obj_id

        # If a closest trajectory is found, append the current particle to it.
        if closest_id is not None:
            trajectories[closest_id].append((frame, x, y, area))
        else:
            # If no close trajectory is found, start a new trajectory for this particle.
            trajectories[len(trajectories) + 1] = [(frame, x, y, area)]

    # Identify and remove trajectories that are shorter than the minimum size threshold.
    to_remove = [obj_id for obj_id, trajectory in trajectories.items() if len(trajectory) < motion_params['min_trajectory_size']]
    for obj_id in to_remove:
        del trajectories[obj_id]

    # Return the final set of trajectories.
    return trajectories


def visualize_trajectories(original_image, particle_trajectories):
    
    """
    Visualizes particle trajectories on an image and creates an image showing only the trajectories.

    This function takes an original image and a dictionary of particle trajectories. Each trajectory is drawn on the 
    original image. Additionally, a separate image is created that shows only these trajectories on a black background. 
    The function offers an option to use random colors for each trajectory or a single default color.

    The process involves:
    - Checking if the original image is grayscale and converting it to BGR if necessary.
    - Creating a blank black image of the same size as the original.
    - Drawing lines representing particle trajectories on both the original and the blank image.
    - Optionally, randomizing colors for each trajectory.

    Args:
        original_image (numpy.ndarray): The original image on which trajectories will be drawn.
        particle_trajectories (dict): A dictionary with object IDs as keys and lists of trajectory points as values.
        use_random_colors (bool, optional): Flag to determine if each trajectory should be drawn in a random color. 
                                            Defaults to False, using red as the default color.

    Returns:
        tuple: A tuple containing two numpy.ndarrays. The first is the original image with trajectories drawn on it, 
               and the second is an image showing only the trajectories on a black background.
    """
    
    motion_params = config['motion'] 
    
    # Check if the original image is grayscale (2 dimensions)
    if len(original_image.shape) == 2:
        # Convert grayscale to BGR
        image_with_trajectories = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR)
    else:
        # If already a BGR image, make a copy
        image_with_trajectories = original_image.copy()


    # Create a blank image (black) of the same size as original image
    height, width = original_image.shape[:2]
    trajectories_only_image = np.zeros((height, width, 3), dtype=np.uint8)

    # Function to generate a random color
    def random_color():
        return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))

    for object_id, trajectory in particle_trajectories.items():
        if motion_params['trajectory_random_colors']:
            color = random_color()
        else:
            color = (0, 0, 255)  # Red color

        for i in range(1, len(trajectory)):
            prev_point = (int(trajectory[i - 1][1]), int(trajectory[i - 1][2]))
            current_point = (int(trajectory[i][1]), int(trajectory[i][2]))
            # Draw line on the image with trajectories
            cv2.line(image_with_trajectories, prev_point, current_point, color, thickness=2)
            # Draw the same line on the blank image
            cv2.line(trajectories_only_image, prev_point, current_point, color, thickness=2)

    return image_with_trajectories, trajectories_only_image


def extract_movement_variables(trajectories):
    """
    Extracts and calculates various movement variables for each trajectory in the input.

    This function iterates through each trajectory in the provided dictionary (indexed by object IDs). 
    For trajectories with sufficient data (more than one point), it calculates several movement parameters:
    average speed, maximum speed, average directionality, meandering index, displacement, and trajectory size.
    
    The calculations involve:
    - Computing distances between consecutive points.
    - Calculating the total distance traveled and the shortest path (straight line) distance from start to end.
    - Determining the total angle change over the trajectory.
    - Computing the meandering index as the ratio of the total path length to the shortest path distance.

    Args:
        trajectories (dict): A dictionary of trajectories, where each key is an object ID and each value is a 
        list of tuples (frame number, x position, y position, additional data).

    Returns:
        mov_vars_df: A DataFrame with each row representing a trajectory. Columns include object ID, average speed, 
        maximum speed, average directionality, meandering index, displacement, and trajectory size (in number of frames).
    """
    
    def calculate_distance(point1, point2):
        return np.hypot(point2[0] - point1[0], point2[1] - point1[1])

    parameters = []

    for obj_id, trajectory in trajectories.items():
        if len(trajectory) < 2:
            continue  # Skip objects with insufficient data

        distances = []
        total_distance = 0
        total_time = 0
        total_angle_change = 0
        start_point = np.array(trajectory[0][1:3])
        end_point = np.array(trajectory[-1][1:3])
        shortest_path_distance = np.linalg.norm(end_point - start_point)

        for i in range(1, len(trajectory)):
            frame1, x1, y1, _ = trajectory[i - 1]
            frame2, x2, y2, _ = trajectory[i]

            dx = x2 - x1
            dy = y2 - y1
            dt = frame2 - frame1

            distance = calculate_distance((x1, y1), (x2, y2))
            total_distance += distance
            total_time += dt
            distances.append(distance)

            # Calculate the angle change
            angle_change = np.arctan2(dy, dx)
            total_angle_change += abs(angle_change)

        average_speed = total_distance / total_time if total_time > 0 else 0
        maximum_speed = max(distances)
        avg_directionality = total_angle_change / len(trajectory) if len(trajectory) > 0 else np.nan
        meander_index = total_distance / shortest_path_distance if shortest_path_distance > 0 else np.nan
        displacement = total_distance
        trajectory_size = len(trajectory) # Number of frames

        parameters.append([obj_id, average_speed, maximum_speed, avg_directionality, meander_index, displacement, trajectory_size])

    columns = ['obj_id', 'avg_speed', 'max_speed', 'directionality', 'meandering_index', 'displacement', 'trajectory_size']
    mov_vars_df = pd.DataFrame(parameters, columns=columns)

    return mov_vars_df


def summarize_movement_variables(mov_vars_df):
    
    """
    Summarizes movement variables from a DataFrame of particle trajectories.

    This function calculates weighted averages for several movement parameters including average speed, 
    maximum speed, directionality, meandering index, and displacement, based on the size of each trajectory. 
    These averages are weighted by the 'trajectory_size', which represents the significance or prevalence of 
    each trajectory in the dataset.

    Args:
        mov_vars_df (DataFrame): A DataFrame containing movement variables for different trajectories. 
                                 Expected columns include 'avg_speed', 'max_speed', 'directionality', 
                                 'meandering_index', 'displacement', and 'trajectory_size'.

    Returns:
        DataFrame: A summary DataFrame with one row containing the weighted averages of the movement parameters.
    """
    if not mov_vars_df.empty:    
        # Calculate weighted averages
        weighted_avg_speed = (mov_vars_df['avg_speed'] * mov_vars_df['trajectory_size']).sum() / mov_vars_df['trajectory_size'].sum()
        weighted_max_speed = (mov_vars_df['max_speed'] * mov_vars_df['trajectory_size']).sum() / mov_vars_df['trajectory_size'].sum()
        weighted_directionality = (mov_vars_df['directionality'] * mov_vars_df['trajectory_size']).sum() / mov_vars_df['trajectory_size'].sum()
        weighted_meandering_index = (mov_vars_df['meandering_index'] * mov_vars_df['trajectory_size']).sum() / mov_vars_df['trajectory_size'].sum()
        weighted_displacement = (mov_vars_df['displacement'] * mov_vars_df['trajectory_size']).sum() / mov_vars_df['trajectory_size'].sum()
        
    else:
        weighted_avg_speed = np.nan
        weighted_max_speed = np.nan
        weighted_directionality = np.nan
        weighted_meandering_index = np.nan
        weighted_displacement = np.nan
        print("Error: No particle trajectories detected!")
        
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'avg_speed': [round(weighted_avg_speed,3)], # pixels/frame
        'max_speed': [round(weighted_max_speed, 3)], # pixels/frame
        'directionality': [round(weighted_directionality,3)], # radians/frame
        'meandering_index': [round(weighted_meandering_index,3)], 
        'displacement': [round(weighted_displacement, 3)]
    })
        
    return summary_df


def perform_motion_analysis(video_path):
    
    """
    A high-level function that analyzes particle motion in a video by detecting particles, 
    tracking their movement across frames, and summarizing their motion characteristics.
    
    This function loads a configuration file to set parameters for particle detection, wellplate analysis, 
    motion analysis, and outputs. It then processes a given video to record particle positions, 
    tracks these particles to create trajectories, and extracts movement variables for each trajectory. 
    Finally, it summarizes these variables and optionally saves various outputs related to the motion analysis 
    (like CSV files with particle positions and movement variables, images of long exposure with tracks, etc.).
    
    Args:
        video_path (str): Path to the video file that needs to be analyzed.
        config_path (str): Path to the configuration file which contains parameters for particle detection, 
                           wellplate setup, motion analysis, and outputs.
    
    Returns:
        DataFrame: A summary DataFrame that contains aggregated motion analysis results for all tracked trajectories.
    
    Note:
        The function assumes that the necessary libraries and functions like `load_config`, 
        `record_particle_positions_in_video`, `track_particles`, `extract_movement_variables`, 
        `summarize_movement_variables`, and `visualize_trajectories` are available and properly implemented.
        The output files are saved in a directory named after the video file within the same directory as the video.
    """
    
    output_params = config['outputs'] 
        
    positions_df, long_exposure_image = record_particle_positions_in_video(video_path)
    trajectories = track_particles(positions_df)
    
    # Extract movement parameters for each object
    movement_variables = extract_movement_variables(trajectories)
    
    # Combine values of all trajectories
    summary_df = summarize_movement_variables(movement_variables)
  
    if output_params['motion'] == True:
        
        # Extract the folder where the video is stored
        main_dir = os.path.dirname(video_path)
        
        # Extract the video filename without the extension
        filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create an output folder within the folder where the video is stored
        output_path = os.path.join(main_dir, f'{filename}_motion_analysis')
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # Extract an image of the first frame
        first_frame = wim.extract_frame(video_path, delay=0)
        
        # Visualize particle trajectories
        image_with_tracks, tracks_only = visualize_trajectories(long_exposure_image, trajectories)
        
        # Save the results
        positions_df.to_csv(os.path.join(output_path, 'particle_positions.csv' ), index=False)
        cv2.imwrite(os.path.join(output_path, 'long_exposure_image.jpg' ), long_exposure_image)
        cv2.imwrite(os.path.join(output_path, 'first_frame.jpg' ), first_frame)
        cv2.imwrite(os.path.join(output_path, 'long_exposure_image_with_tracks.jpg' ), image_with_tracks)
        cv2.imwrite(os.path.join(output_path, 'tracks.jpg' ), tracks_only)
        movement_variables.to_csv(os.path.join(output_path, 'movement_by_trajectory.csv' ), index=False)
        summary_df.to_csv(os.path.join(output_path, 'summary_motion_analysis.csv' ), index=False)
  
    return summary_df