# -*- coding: utf-8 -*-

"""
Wellcounter imaging module

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
This module contains several low-level and high-level functions for identifying
microorganisms in the WELLCOUNTER based on recorded mp4-movies.

Note: Portions of the code in this file were generated using ChatGPT v4.0.
      All AI-generated content has been rigorously validated and tested by the 
      authors. The corresponding author accepts full responsibility for the 
      AI-assisted portions of the code.

Author: Claus-Peter Stelzer
Date: 2025-02-07

"""


import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import os
import yaml
import math
from scipy.spatial import distance_matrix


def read_config(config_path="wellcounter_config.yml"):
    """
    Helper-function to read the config file.
    """
    try:
        with open(config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        return config
    except Exception as e:
        print(f"Error reading config file: {e}")
        raise


def calculate_measurements(contour):
    """
        
    Calculates various particle measurements for a given contour.

    This function calculates properties such as the area, perimeter, orientation, aspect ratio, 
    solidity, eccentricity, and feret diameter for the provided contour.

    Args:
        contour (ndarray): A contour of a detected particle.

    Returns:
        list: A list of measurements including X, Y coordinates, area, perimeter, 
        orientation, aspect ratio, solidity, eccentricity, and feret diameter.
   

    """
    # Calculate measurements for a single contour
    M = cv2.moments(contour)
    
    if M["m00"] == 0:
        return None  # Avoid division by zero
    
    # Centroid coordinates, converted to integers
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])
    
    # Area and perimeter
    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)
    
    try:
        # Ellipse fitting
        ellipse = cv2.fitEllipse(contour)
        orientation = ellipse[2]
        # Calculate eccentricity
        minor_axis = min(ellipse[1])
        major_axis = max(ellipse[1])
        eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2))
    except cv2.error as e:
        # Handling the specific error for insufficient points
        if "Incorrect size of input array" in str(e):
            orientation = np.nan
            eccentricity = np.nan
        else:
            raise  # Re-raise the exception if it's not the expected one
    
    # Aspect ratio of bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else np.nan
    
    # Solidity
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else np.nan
    
    # Feret Diameter
    (center_EC, radius_EC) = cv2.minEnclosingCircle(np.array(contour))
    feret_diameter = 2 * radius_EC

    return {
        'X': cx,
        'Y': cy,
        'area': area,
        'perimeter': perimeter,
        'orientation': orientation,
        'aspect_ratio': aspect_ratio,
        'solidity': solidity,
        'eccentricity': eccentricity,
        'feret_diameter': feret_diameter,
        'bounding_x': x,
        'bounding_y': y,
        'bounding_w': w,
        'bounding_h': h
    }


def mask_well_area(image):
    
    """
    This function defines an ROI (region of interest) of the size of the well. 
    All pixels outside the ROI are set to 0 (i.e., dark). 
    
    Note, this function     is called only if the wellcounter_config.yml file is set to:
        wellplate > create_mask > True
        
    ... and it should only be used if wellplate-holders had been placed on top of 
    the six-well plates during the recordings.
        
    """
    
    config = read_config()
    wellplate_params = config['wellplate']
    system_params = config['system']
    
    # Convert image to grayscale if it is not already
    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    if wellplate_params['auto_center_mask'] == True:
    
        # Make a threshold of entire well area
        _, threshold = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
        
        # Detect the largest contour, which corresponds to the well area
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Find the minimum enclosing circle for the contour
        (cx, cy), radius = cv2.minEnclosingCircle(largest_contour)
        
        # Convert the floating-point center and radius to integers
        center = (int(cx), int(cy))
        
    elif wellplate_params['auto_center_mask'] == False:
            
        center = (int(system_params['image_width_px']/2), int(system_params['image_height_px']/2))
    
    # Override radius with radius slightly smaller than the radius of a well  
    radius = int(wellplate_params['radius_px'] * 0.9) 
   
    # Generate the circular contour
    circular_contour = np.zeros((system_params['image_width_px'], system_params['image_height_px']), dtype=np.uint8)
    cv2.circle(circular_contour, center, radius, 255, thickness=-1)  # Draw filled circle
    
    # Find contours in the circular image
    contours, _ = cv2.findContours(circular_contour, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Return the largest contour (circular contour)
    contour_well = max(contours, key=cv2.contourArea)
    
    # Create mask and draw the largest contour on it
    mask = np.zeros_like(image)
    cv2.drawContours(mask, [contour_well], -1, 255, thickness=cv2.FILLED)
    
    # Apply mask to the original image
    result_image = cv2.bitwise_and(image, image, mask=mask)
    
    return result_image, mask


def analyze_microorganisms(image):
    
    """
    This function identifies microorganisms in a subtracted image (input) based 
    on parameters set in the wellcounter_config.yml file (> particle_detection), 
    specifically the parameters 'microorganism_threshold' and 'min_microorganism_area'
    
    If the parameter 'filter_by_shape' in the config file is set 'True', an optional, 
    additional filtering of the particles can done. Currently, we do not use this feature 
    in the WELLCOUNTER.
    
    Identifies microorganisms in a subtracted image based on specified parameters.

    This function processes an image to detect microorganisms using parameters 
    defined in the `wellcounter_config.yml` file, specifically the parameters 
    'microorganism_threshold' and 'min_microorganism_area'. The detection involves thresholding 
    and contour finding, followed by optional shape-based filtering of detected particles.  
    If the parameter 'filter_by_shape' in the config file is set 'True', an optional, 
    additional filtering of the particles can done. Currently, this feature is not used
    in the WELLCOUNTER. 
     
    The detected microorganisms are characterized by various morphological measurements.

    Args:
        image (ndarray): The input image in which microorganisms are to be detected.

    Returns:
        tuple: A tuple containing:
            - DataFrame: A data frame with measurements of detected microorganisms, 
            including coordinates (X, Y), area, perimeter, orientation, aspect ratio, 
            solidity, eccentricity, and feret diameter.
            - ndarray: A binary image where detected microorganisms are marked.
    
    """
        
    config=read_config()
    particle_detection_params = config['particle_detection']
    
    #print("Now within analyze_microorganisms...")
    #print("Using pixel threshold: ", particle_detection_params['microorganism_threshold'])
    #print("Using particle area: ", particle_detection_params['min_microorganism_area'])
    
    # Apply preprocessing operations for microorganism detection
    _, threshold = cv2.threshold(image, particle_detection_params['microorganism_threshold'], 255, cv2.THRESH_BINARY)
    threshold = cv2.medianBlur(threshold, particle_detection_params['microorganism_blur'])
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > particle_detection_params['min_microorganism_area']]
    
    # Create an empty grayscale image with the same dimensions as the original image
    binary_image = np.zeros(image.shape[:2], dtype=np.uint8)
    # Draw the filled valid contours in white
    cv2.drawContours(binary_image, valid_contours, -1, (255), thickness=cv2.FILLED)
    
    # Initialize an empty DataFrame
    columns = ["X", "Y", "area", "perimeter", "orientation", "aspect_ratio", "solidity", "eccentricity",
               "feret_diameter", "bounding_x", "bounding_y", "bounding_w", "bounding_h"]
    df = pd.DataFrame(columns=columns)
    
    if valid_contours:
        for contour in valid_contours:
            measurements = calculate_measurements(contour)
            if measurements:
                df.loc[len(df)] = measurements  # Add measurements as a new row
    
    if particle_detection_params['filter_by_shape']:
        # Filter measurements based 0.5% to 99.5% quantiles of true positive particles in the training data
        df = df[(df['solidity'] >= 0.655) & (df['solidity'] <= 0.987) &
                (df['eccentricity'] >= 0.309) & (df['eccentricity'] <= 0.948) &
                (df['aspect_ratio'] >= 0.44) & (df['aspect_ratio'] <= 2.19)]

    # Sort the DataFrame based on ascending Y and then X values
    df = df.sort_values(by=['Y', 'X'], ascending=[True, True])

    return df, binary_image  # Return the DataFrame and binary image of detected microorganisms


def analyze_unsubtracted(image):
    
    """
    This (currently unused) function is similar to the function 'analyze_microorganisms', 
    but it uses an unsubtracted image as input. Thus, it employs a different pixel threshold 
    (default: 70) than 'analyze_microorganisms'. The purpose of this function is to obtain a 
    additional particle size estimate, in case animals show only little movement 
    (and self-occlusion in subtracted images).
    
    """
    
    config=read_config()
    particle_detection_params = config['particle_detection']
    
    #print("Now within analyze_unsubtracted...")
    #print("Using pixel threshold: ", particle_detection_params['unsubtracted_threshold'])
    #print("Using particle area: ", particle_detection_params['min_microorganism_area'])
    
    # Apply preprocessing operations for microorganism detection
    _, threshold = cv2.threshold(image, particle_detection_params['unsubtracted_threshold'], 255, cv2.THRESH_BINARY)
    threshold = cv2.medianBlur(threshold, particle_detection_params['microorganism_blur'])
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > particle_detection_params['min_microorganism_area']]
    
    # Create an empty grayscale image with the same dimensions as the original image
    binary_image = np.zeros(image.shape[:2], dtype=np.uint8)
    # Draw the filled valid contours in white
    cv2.drawContours(binary_image, valid_contours, -1, (255), thickness=cv2.FILLED)
    
    # Initialize an empty DataFrame
    columns = ["X", "Y", "area", "perimeter", "orientation", "aspect_ratio", "solidity", "eccentricity",
               "feret_diameter"]
    df = pd.DataFrame(columns=columns)
    
    if valid_contours:
        for contour in valid_contours:
            measurements = calculate_measurements(contour)
            if measurements:
                df.loc[len(df)] = measurements  # Add measurements as a new row
    
    if particle_detection_params['filter_by_shape']:
        # Filter measurements based 0.5% to 99.5% quantiles of true positive particles in the training data
        df = df[(df['solidity'] >= 0.655) & (df['solidity'] <= 0.987) &
                (df['eccentricity'] >= 0.309) & (df['eccentricity'] <= 0.948) &
                (df['aspect_ratio'] >= 0.44) & (df['aspect_ratio'] <= 2.19)]

    # Sort the DataFrame based on ascending Y and then X values
    df = df.sort_values(by=['Y', 'X'], ascending=[True, True])

    return df, binary_image  # Return the DataFrame and binary image of detected microorganisms



def label_particles(image, table_of_particles):
    
    """   
    Labels detected particles on an image.

    This function takes an image and a data frame containing particle information,
    in particular the X and Y coordinates of each particle, and it labels each 
    particle on the image with a colored circle.

    Args:
        image (ndarray): The input image on which particles will be labeled.
        table_of_particles (DataFrame): A data frame containing the particle information, including X and Y coordinates.

    Returns:
        ndarray: The image with labeled particles.
    
    """
    
    config = read_config()
    particle_detection_params = config['particle_detection']
    
    different_colors=False
    
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    
    sr_factor = particle_detection_params['search_radius_factor']
    dp_area = particle_detection_params['default_particle_area']
    search_radius = round(sr_factor * np.sqrt(dp_area / np.pi))
    
    if not table_of_particles.empty:
        
        for index, row in table_of_particles.iterrows():
            x, y, in_ref, in_query = int(row['X']), int(row['Y']), int(row['in_ref']), int(row['in_query'])
            
            if different_colors:
                if in_ref == 1 and in_query == 0:
                    label_color = (230, 216, 173)  # Light Blue
                elif in_ref == 1 and in_query == 1:
                    label_color = (0, 252, 124)    # Lawn Green
                elif in_ref == 0 and in_query == 1:
                    label_color = (0, 0, 255)      # Red
            else:
                label_color = (0, 252, 124)    # Lawn Green
            
            cv2.circle(image, (x, y), search_radius, label_color, thickness=3)    
            labeled_image = image
        
    else:
        labeled_image = image
     
    
    return labeled_image


    
def extract_frame(video_path, delay=0):
    
    """
    A low-level function to extract a specific frame from a video file.

    This function opens the video file using OpenCV, sets the video to the 
    specified frame number (delay in seconds), and reads the frame.

    Args:
        video_path (str): The path to the video file.
        delay (int): seconds since the start of the movie.

    Returns:
        ndarray: The extracted frame as an image array.
    
    """
        
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    
    # Check if the video file was opened successfully
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return None
   
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
   
    # Calculate the frame position based on the delay
    frame_position = int(delay * cap.get(cv2.CAP_PROP_FPS))
      
    # Check if the delay is greater than the total duration of the video
    if frame_position >= total_frames:
        print("Error: Delay exceeds the total duration of the video.")
        return None
   
    # Set the frame position in the video
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_position)
   
    # Read the frame at the specified position
    ret, frame = cap.read()
   
    # Release the video capture object
    cap.release()
    
    # Check if the frame was read successfully
    if not ret:
        print("Error: Could not read the frame at the specified delay.")
        return None
   
    # Convert the extracted frame to grayscale
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame


def compare_detected_particles(df_ref, df_query):
    
    """
    Compare and merge detected particles from two data frames.

    This function takes two data frames, `df_ref` and `df_query`, each containing
    detected particles with their respective coordinates. The particles are matched
    based on their X and Y coordinates, with a specified search radius determining
    whether two particles are considered identical. The result is a combined data
    frame indicating whether each particle is present in `df_ref`, `df_query`, or both.
    
    Parameters:
    df_ref (pd.DataFrame): The reference data frame containing detected particles.
    df_query (pd.DataFrame): The query data frame containing detected particles.
    
    Returns:
    pd.DataFrame: A data frame with merged and compared particle information, 
                  including columns 'in_ref' and 'in_query' to flag the presence
                  of particles in `df_ref` and `df_query` respectively.
    
    Raises:
    ValueError: If any of the input data frames are None or if they lack the 'X' and 'Y' columns.
    
    Configuration:
    This function relies on settings from the 'wellcounter_config' file, specifically:
    - 'particle_detection > search_radius_factor'
    - 'particle_detection > default_particle_area'
    
    These settings are used to calculate the 'search_radius', which determines the
    matching criteria for particles between the two data frames.
    
    """

    config=read_config()
    particle_detection_params = config['particle_detection']    

    if df_ref is None or df_query is None:
        raise ValueError("Input error in compare_detected_particles: One or both DataFrames are None. ")

    if 'X' not in df_ref.columns or 'Y' not in df_ref.columns or 'X' not in df_query.columns or 'Y' not in df_query.columns:
        raise ValueError("Both dataframes must contain 'X' and 'Y' columns.")
    
    # Read parameters for calculating search radius
    sr_factor = particle_detection_params['search_radius_factor']
    dp_area = particle_detection_params['default_particle_area']
    search_radius = sr_factor * np.sqrt(dp_area / np.pi)
    
    # Step 1: Identify columns only present in df_query
    query_only_columns = set(df_query.columns) - set(df_ref.columns)
    
    # Step 2: Modify df_ref by adding these columns with NaN entries
    for col in query_only_columns:
        df_ref[col] = np.nan
    
    # Step 3: Determine a new set of 'common_columns'
    # Now that df_ref includes all df_query columns, all df_query columns are common
    common_columns = list(set(df_ref.columns) & set(df_query.columns))
    
    # Step 4: Create df_combined with the new common_columns
    df_combined = pd.concat([
        df_ref[common_columns].assign(source='ref'), 
        df_query[common_columns].assign(source='query')
    ], ignore_index=True)
    
    df_combined['X'] = df_combined['X'].astype(int)
    df_combined['Y'] = df_combined['Y'].astype(int)

    tree = BallTree(df_combined[['X', 'Y']].values)
    
    results_list = []
    matched_query_indices = set()

    # Process df_ref particles for matching, with adjusted matching dictionary creation
    for index, row in df_ref.iterrows():
        x, y = row['X'], row['Y']
        query_indices = tree.query_radius([[x, y]], r=search_radius)[0]
        distances, indices = tree.query([[x, y]], k=len(query_indices))
        
        # Filter out indices that are beyond the search radius or already matched
        valid_indices = [(i, dist) for dist, i in zip(distances[0], indices[0]) if i not in matched_query_indices and dist <= search_radius and i >= len(df_ref)]
        
        if valid_indices:
            # Choose the match with the minimum distance
            closest_match_index, _ = min(valid_indices, key=lambda x: x[1])
            matched_query_indices.add(closest_match_index)
            in_query = 1

            # Fetch the attributes of the matching df_query particle
            matched_particle_info = df_combined.iloc[closest_match_index][common_columns].to_dict()
            matched_particle_info.update({'source': 'query', 'in_ref': 1, 'in_query': in_query})
        else:
            closest_match_index = None
            in_query = 0

            # If no match is found, use the original row's attributes but indicate no match in df_query
            matched_particle_info = row[common_columns].to_dict()
            matched_particle_info.update({'source': 'ref', 'in_ref': 1, 'in_query': in_query})
        
        results_list.append(matched_particle_info)

    # Process unmatched df_query particles
    unmatched_query_indices = set(range(len(df_ref), len(df_combined))) - matched_query_indices
    for idx in unmatched_query_indices:
        particle_info = df_combined.iloc[idx].to_dict()
        particle_info.update({'source': 'query', 'in_ref': 0, 'in_query': 1})
        results_list.append(particle_info)

    result_df = pd.DataFrame(results_list)
    return result_df


def spatial_analysis(table_of_particles):
    
    """
       
    Performs spatial analysis on detected particles.

    This function calculates the Nearest Neighbor Index (NNI) to assess the spatial distribution of particles within the well.

    Args:
        table_of_particles (DataFrame): A data frame containing particle information, including X and Y coordinates.

    Returns:
        float: The Nearest Neighbor Index indicating the spatial distribution of particles.
    
    """
    
    # Minimum number of particles required for spatial analysis
    if table_of_particles.shape[0] > 5:
    
        config=read_config()
        wellplate_params = config['wellplate'] 
        
        # Select only the 'X' and 'Y' columns for distance calculations
        coordinates = table_of_particles[['X', 'Y']].values
        
        # Calculate the pairwise distance matrix
        dist_matrix = distance_matrix(coordinates, coordinates)
            
        # Set diagonal to np.nan to ignore self-distance
        np.fill_diagonal(dist_matrix, np.nan)
            
        # Calculate observed mean distance (r_min)
        #r_min = np.nanmean(np.min(dist_matrix, axis=1))
        nearest_dists = np.nanmin(dist_matrix, axis=1)
        r_min = np.nanmean(nearest_dists[~np.isnan(nearest_dists)])  # Only calculate mean for non-nan values
        
        
        # Calculate expected mean distance (r_e) for a random distribution
        A = np.pi * wellplate_params['radius_px'] ** 2  # Area of well calculated from wellplate-radius in pixels
        n = len(table_of_particles)
        r_e = np.sqrt(A / (n * np.pi))
            
        # Calculate NNI
        NNI = r_min / r_e
        
    else:
        NNI = np.nan
       
    return NNI
   

def image_subtraction_from_video(video_path, delay1, delay2):
    
    """
    Perform image subtraction between two frames of a video.
    
    This function extracts two frames from the specified video file, corresponding
    to the times `delay1` and `delay2` seconds from the start of the video. It then
    performs image subtraction to highlight differences between the two frames.
    Additionally, if specified in the configuration, a mask is applied to focus 
    on specific areas of interest, such as wells in a well plate.
    
    Parameters:
    video_path (str): The file path to the input video.
    delay1 (float): The time (in seconds) from the start of the video for the first frame.
    delay2 (float): The time (in seconds) from the start of the video for the second frame.
    
    Returns:
    tuple: A tuple containing:
        - result_image (np.ndarray): The resulting image after subtraction and masking (if applicable).
        - masked_image (np.ndarray): The initial image with the applied mask (if masking is enabled), or the first frame if masking is not enabled.
    
    Configuration:
    The function utilizes settings from a configuration file, specifically:
    - `wellplate > create_mask`: A boolean indicating whether to apply a mask to the subtracted image.
    
    Raises:
    FileNotFoundError: If the specified video file does not exist.
    ValueError: If `delay1` or `delay2` are outside the valid range for the video duration.
    
    """
    
    config=read_config()
    wellplate_params = config['wellplate']
       
    image_a = extract_frame(video_path, delay1)
    image_b = extract_frame(video_path, delay2)
        
    # Subtract images
    subtr_image = cv2.subtract(image_a, image_b)
    
    # Clean up the subtracted image (clamping to the valid range)
    subtr_image = np.clip(subtr_image, 0, 255).astype(np.uint8)
   
    if wellplate_params['create_mask'] == True:
   
        # Create a mask where everything outside a well is black
        masked_image , mask = mask_well_area(image_a)
        
        # Apply mask to subtracted image
        result_image = cv2.bitwise_and(subtr_image, subtr_image, mask=mask)
        
    else:
        result_image = subtr_image
        masked_image = image_a
           
    return result_image, masked_image


def image_analysis_of_sample(video_path, frame1 = 0, frame2 = 2, frame3 = 5):
    
    """
    This high-level function performs image analysis on a sample video to detect and analyze microorganisms.
    It uses specific frames from the video for image subtraction, detecting particles in these subtracted images,
    and then comparing these particles to identify common features. The function also handles the generation
    of outputs, including saving labeled images and a table of detected particles if specified in the config-file.

    Args:
    - video_path (str): The file path of the video to be analyzed.
    - frame1 (float): Time delay in seconds used as the first reference frame for image subtraction. Default is 0.
    - frame2 (float): Time delay in seconds used for the first image subtraction with frame1. Default is 2.
    - frame3 (float): Time delay in seconds used for the second image subtraction with frame1. Default is 5.

    Returns:
    - DataFrame: A pandas DataFrame containing the details of detected particles in the analyzed video. 
                 Each row represents a particle, and the columns contain various properties of these particles.

    The function performs the following steps:
    1. Loads configuration settings from a YAML file.
    2. Subtracts images based on specified frame numbers.
    3. Analyzes microorganisms in the subtracted images.
    4. Adds a dummy column to the tables of detected particles.
    5. Compares detected particles from both image subtractions.
    6. Generates outputs including labeled images and a CSV table of detected particles, based on the output parameters in the configuration.
    
    """
    
    config=read_config()      
    output_params = config['outputs']
        
    # First image subtraction
    subtr_image1, masked_image = image_subtraction_from_video(video_path, delay1 = frame1, delay2 = frame2)
      
    # Second image subtraction
    subtr_image2, _ = image_subtraction_from_video(video_path, delay1 = frame1, delay2 = frame3)
    
    # Analyze microorganisms (first image subtraction)
    table_of_particles1, binary_image1 = analyze_microorganisms(subtr_image1)
               
    # Analyze microorganisms (second image subtraction)
    table_of_particles2, binary_image2 = analyze_microorganisms(subtr_image2)
    
    # Add a dummy column, which is required in 'compare_detected_particles'
    table_of_particles1['particle_type'] = int(0)
    table_of_particles2['particle_type'] = int(0)
 
    
    ########## NEW CODE 08.07.2024) ###########################################
    # Obtain size estimate of all particles based on unsubtracted first frame  #
    # to prevent small size due to self-oclusion in weakly moving individuals   #
                                                                                 #
    # Test if any particles are detected using the subtraction-based method                                         # 
    if not table_of_particles1.empty:                                              #
        # Join particles of both lists                                              #
        top1 = compare_detected_particles(table_of_particles1, table_of_particles2)  #
    else:                                                                             #   
        top1 = table_of_particles1                                                     #
                                                                                        #
                                                                                         #
    # Select specific columns to print                                                    # 
    #columns_to_print = ['X', 'Y', 'area', 'in_ref', 'in_query']                            #
                                                                                            #
    #top1_sel = top1[columns_to_print]                                                        #
    #print("top1_sel:")                                                                        #
    #print(top1_sel)                                                                            #
                                                                                                #
    fframe = extract_frame(video_path, delay=0)                                                  #
    masked_image,_ = mask_well_area(fframe)                                                      #
    top2,binary_image = analyze_unsubtracted(masked_image) # Detect particles                    #
    #cv2.imwrite('masked_image.jpg', masked_image)                                                #
    #cv2.imwrite('binary_image.jpg', binary_image)                                                #
                                                                                                 #
    #top2_sel = top2                                                                             #
    #print("top2_sel:")                                                                         # 
    #print(top2_sel)                                                                           #
                                                                                             #
    # If a match is found, area will be taken from df_query                                 # 
    top3 = compare_detected_particles(top1, top2)                                          #
                                                                                          #
    #top3_sel = top3[columns_to_print]                                                    #
    #print("top3_sel:")                                                                  #
    #print(top3_sel)                                                                    #
                                                                                      #
    # Delete all particles that were missing in the subtraction-based df (df_ref)    #
    table_of_particles = top3[top3['in_ref'] != 0] # Drop rows where in_ref is 0    # 
                                                                                   #
                                                                                  #
    #table_of_particles_sel = table_of_particles[columns_to_print]                # 
    #print("table_of_particles_sel:")                                            #
    #print(table_of_particles_sel)                                              #
    ###########################################################################
    
    # Generate outputs
    if output_params['particle_detection'] == True:
        
        # Extract the folder where the video is stored
        main_dir = os.path.dirname(video_path)
        
        # Extract the video filename without the extension
        filename = os.path.splitext(os.path.basename(video_path))[0]
        
        # Create an output folder within the folder where the video is stored
        output_path = os.path.join(main_dir, f'{filename}_particle_detection')
        
        # Create the directory if it doesn't exist
        if not os.path.exists(output_path):
            os.makedirs(output_path)
            
        # Save an image where all detected particles are labelled
        first_frame = extract_frame(video_path, delay = frame1)
        labeled_image = label_particles(first_frame, table_of_particles)
        cv2.imwrite(os.path.join(output_path, 'frame1_particles.jpg' ), labeled_image)
        
        # Save images of the two subtractions where all detected particles are labelled
        subtr_image1_labels = label_particles(binary_image1, table_of_particles)
        cv2.imwrite(os.path.join(output_path, 'image_subtraction1.jpg' ), subtr_image1_labels)
        subtr_image2_labels = label_particles(binary_image2, table_of_particles)
        cv2.imwrite(os.path.join(output_path, 'image_subtraction2.jpg' ), subtr_image2_labels)
        
        # Save images of the two additional frames used for subtraction
        image_a = extract_frame(video_path, frame2)
        image_b = extract_frame(video_path, frame3)
        cv2.imwrite(os.path.join(output_path, f'frame2_{frame2}sec.jpg' ), image_a)
        cv2.imwrite(os.path.join(output_path, f'frame3_{frame3}sec.jpg' ), image_b)
        
        # Save masked image of the first frame
        cv2.imwrite(os.path.join(output_path, 'frame1_masked_well.jpg' ), masked_image)
        
        # Save binary image of the first frame
        cv2.imwrite(os.path.join(output_path, 'frame1_binary_image.jpg' ), binary_image)      
        
        # Save a table of all detected particles
        table_of_particles.to_csv(os.path.join(output_path, 'table_of_particles.csv' ), index=False)
           
    return table_of_particles


def count_particles(video_path):
    
    """
    Count and analyze microorganisms in a video.

    This function performs a high-level analysis to count microorganisms visible
    in a video. It analyzes frames from the beginning, middle, and end of the video
    to calculate the average number of microorganisms, their average size, and their
    spatial distribution in the well.

    Parameters:
    video_path (str): The file path to the input video.

    Returns:
    pd.DataFrame: A summary data frame containing the following metrics:
        - 'avg_particles': The average number of particles detected across the analyzed frames.
        - 'median_particle_size': The median size of detected particles.
        - 'spatial_nni': The average nearest neighbor index (NNI) indicating the spatial distribution of the particles.

    Raises:
    FileNotFoundError: If the specified video file does not exist.
    ValueError: If the video cannot be opened or processed.

    Workflow:
    1. The function opens the video file and calculates its duration.
    2. It determines three key frames for analysis: the beginning, middle, and near the end of the video.
    3. Each of these frames is analyzed to detect and count particles.
    4. The function computes the average number of particles, the median size of particles, and the spatial NNI.
    5. Results are printed and returned in a summary data frame.
    
    """
          
    # Calculate the duration of the video in seconds
    video = cv2.VideoCapture(video_path)
    
    # Check if the video file was opened successfully
    if not video.isOpened():
        print("Error: Could not open video file.")
        exit()
    
    # Calculate duration of the video
    total_frames = video.get(cv2.CAP_PROP_FRAME_COUNT)
    fps = video.get(cv2.CAP_PROP_FPS)
    duration_in_seconds = total_frames / fps
    video.release()
    
    # Calculate the three frames to be analyzed
    frame1 = 0 # beginning of video
    frame2 = math.floor(duration_in_seconds/2) # near middle of video
    frame3 = math.floor(duration_in_seconds-1) # near end of video
    
    # Perform analysis on three different frames (beginning, middle and end of video)
    table_of_particles3 = image_analysis_of_sample(video_path, frame3, frame2, frame1)
    table_of_particles2 = image_analysis_of_sample(video_path, frame2, frame1, frame3)
    table_of_particles1 = image_analysis_of_sample(video_path, frame1, frame2, frame3)
        
    # Count detected particles
    p1 = table_of_particles1.shape[0]
    p2 = table_of_particles2.shape[0]
    p3 = table_of_particles3.shape[0]
        
    # Assess spatial distribution (Nearest neighbour index)
    nni1 = spatial_analysis(table_of_particles1)
    nni2 = spatial_analysis(table_of_particles2)
    nni3 = spatial_analysis(table_of_particles3)   
    
    # Calculate various metrics
    avg_particles = round ( (p1+p2+p3)/3 , 1) # number of particles
    nni = round( (nni1+nni2+nni3)/3 , 4)           # nearest neighbour index
    median_particle_area = table_of_particles1['area'].median()
        
    print("Individual counts: ",  p1, " ," ,p2, " ,", p3)
    print("Average number of particles: ", avg_particles)
    print("Median size of particles: ", median_particle_area)
    print("Nearest neighbour index: ", nni)
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame({
        'avg_particles': [avg_particles], # pixels/frame
        'median_particle_size': [median_particle_area], # pixels/frame
        'spatial_nni': [nni]
    })
    
    
    return summary_df
    

    
    
    
