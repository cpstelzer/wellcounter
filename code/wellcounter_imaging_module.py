# -*- coding: utf-8 -*-

"""
Wellcounter imaging module (Modified for FPS-in-Filename)

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
This module contains functions for identifying microorganisms. This version is
adapted to work with image sequences where the FPS is embedded in the filename,
making the analysis pipeline independent of external metadata files for this parameter.

Author: Claus-Peter Stelzer
Date: 2025-02-07
Modification Date: 2025-09-22
"""

import cv2
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree
import os
import yaml
import math
import re
from scipy.spatial import distance_matrix
import glob

def read_config(config_path="wellcounter_config.yml"):
    try:
        with open(config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        return config
    except Exception as e:
        print(f"Error reading config file: {e}")
        raise

# --- HELPER FUNCTIONS FOR IMAGE SEQUENCE HANDLING ---

def get_image_file_list(run_folder_path):
    supported_formats = ['*.png', '*.jpg', '*.bmp']
    image_files = []
    for fmt in supported_formats:
        image_files.extend(glob.glob(os.path.join(run_folder_path, fmt)))
    image_files.sort()
    return image_files

def get_fps_from_sequence(run_folder_path):
    """
    Parses the FPS value from the first image file in a sequence.
    Returns:
        float: The FPS value, or a default of 25.0 if not found.
    """
    image_files = get_image_file_list(run_folder_path)
    if not image_files:
        print(f"Warning: No image files found in {run_folder_path}. Cannot determine FPS.")
        return 25.0  # Return a safe default

    first_filename = os.path.basename(image_files[0])
    match = re.search(r'_fps(\d+)\.', first_filename)
    if match:
        return float(match.group(1))
    else:
        print(f"Warning: FPS not found in filename '{first_filename}'. Using default 25.0 FPS.")
        return 25.0

def get_frame_from_sequence(image_file_list, frame_number):
    if frame_number < 0 or frame_number >= len(image_file_list):
        print(f"Error: Frame number {frame_number} is out of bounds.")
        return None
    frame_path = image_file_list[frame_number]
    if not os.path.exists(frame_path):
        print(f"Error: Image file not found at {frame_path}")
        return None
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    return frame

# --- UNCHANGED FUNCTIONS (Included for completeness) ---

def calculate_measurements(contour):
    M = cv2.moments(contour)
    if M["m00"] == 0: return None
    cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
    area, perimeter = cv2.contourArea(contour), cv2.arcLength(contour, True)
    try:
        ellipse = cv2.fitEllipse(contour)
        orientation = ellipse[2]
        minor_axis, major_axis = min(ellipse[1]), max(ellipse[1])
        eccentricity = np.sqrt(1 - (minor_axis ** 2) / (major_axis ** 2)) if major_axis > 0 else 0
    except cv2.error: orientation, eccentricity = np.nan, np.nan
    x, y, w, h = cv2.boundingRect(contour)
    aspect_ratio = float(w) / h if h != 0 else np.nan
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)
    solidity = area / hull_area if hull_area != 0 else np.nan
    (_, radius_EC) = cv2.minEnclosingCircle(contour)
    feret_diameter = 2 * radius_EC
    return {'X': cx, 'Y': cy, 'area': area, 'perimeter': perimeter, 'orientation': orientation, 
            'aspect_ratio': aspect_ratio, 'solidity': solidity, 'eccentricity': eccentricity, 
            'feret_diameter': feret_diameter, 'bounding_x': x, 'bounding_y': y, 'bounding_w': w, 'bounding_h': h}

def mask_well_area(image):
    config = read_config()
    wellplate_params = config['wellplate']
    system_params = config['system']
    if len(image.shape) == 3: image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if wellplate_params['auto_center_mask']:
        _, threshold = cv2.threshold(image, 5, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        (cx, cy), _ = cv2.minEnclosingCircle(largest_contour)
        center = (int(cx), int(cy))
    else: center = (int(system_params['image_width_px']/2), int(system_params['image_height_px']/2))
    radius = int(wellplate_params['radius_px'] * 0.9)
    mask = np.zeros_like(image)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    result_image = cv2.bitwise_and(image, image, mask=mask)
    return result_image, mask

def analyze_microorganisms(image):
    config = read_config()
    params = config['particle_detection']
    _, threshold = cv2.threshold(image, params['microorganism_threshold'], 255, cv2.THRESH_BINARY)
    threshold = cv2.medianBlur(threshold, params['microorganism_blur'])
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > params['min_microorganism_area']]
    binary_image = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(binary_image, valid_contours, -1, 255, thickness=cv2.FILLED)
    df = pd.DataFrame([m for cnt in valid_contours if (m := calculate_measurements(cnt)) is not None])
    if not df.empty and params['filter_by_shape']:
        df = df[(df['solidity'] >= 0.655) & (df['solidity'] <= 0.987) & (df['eccentricity'] >= 0.309) & (df['eccentricity'] <= 0.948) & (df['aspect_ratio'] >= 0.44) & (df['aspect_ratio'] <= 2.19)]
    return df.sort_values(by=['Y', 'X']) if not df.empty else df, binary_image

def analyze_unsubtracted(image):
    config = read_config()
    params = config['particle_detection']
    _, threshold = cv2.threshold(image, params['unsubtracted_threshold'], 255, cv2.THRESH_BINARY)
    threshold = cv2.medianBlur(threshold, params['microorganism_blur'])
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid_contours = [c for c in contours if cv2.contourArea(c) > params['min_microorganism_area']]
    binary_image = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(binary_image, valid_contours, -1, 255, thickness=cv2.FILLED)
    df = pd.DataFrame([m for cnt in valid_contours if (m := calculate_measurements(cnt)) is not None])
    return df.sort_values(by=['Y', 'X']) if not df.empty else df, binary_image

def label_particles(image, table_of_particles):
    config = read_config()
    params = config['particle_detection']
    if len(image.shape) == 2: image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    search_radius = round(params['search_radius_factor'] * np.sqrt(params['default_particle_area'] / np.pi))
    if not table_of_particles.empty:
        for _, row in table_of_particles.iterrows():
            x, y = int(row['X']), int(row['Y'])
            cv2.circle(image, (x, y), search_radius, (0, 252, 124), thickness=3)
    return image

def compare_detected_particles(df_ref, df_query):
    config = read_config()
    params = config['particle_detection']
    if df_ref is None or df_query is None: raise ValueError("Input DataFrames cannot be None.")
    if 'X' not in df_ref.columns or 'Y' not in df_ref.columns or 'X' not in df_query.columns or 'Y' not in df_query.columns: raise ValueError("DataFrames must contain 'X' and 'Y' columns.")
    if df_ref.empty: df_query['in_ref'], df_query['in_query'] = 0, 1; return df_query
    if df_query.empty: df_ref['in_ref'], df_ref['in_query'] = 1, 0; return df_ref
    search_radius = params['search_radius_factor'] * np.sqrt(params['default_particle_area'] / np.pi)
    tree_query = BallTree(df_query[['X', 'Y']].values)
    distances, indices = tree_query.query(df_ref[['X', 'Y']].values, k=1)
    matched_query_indices, matches = set(), []
    for i, (idx, dist) in enumerate(zip(indices.flatten(), distances.flatten())):
        row_dict = df_ref.iloc[i].to_dict()
        if dist <= search_radius and idx not in matched_query_indices:
            matched_query_indices.add(idx)
            row_dict.update(df_query.iloc[idx].to_dict())
            row_dict.update({'in_ref': 1, 'in_query': 1})
        else: row_dict.update({'in_ref': 1, 'in_query': 0})
        matches.append(row_dict)
    unmatched_query = df_query.drop(index=list(matched_query_indices)).copy()
    unmatched_query['in_ref'], unmatched_query['in_query'] = 0, 1
    return pd.concat([pd.DataFrame(matches), unmatched_query], ignore_index=True)

def spatial_analysis(table_of_particles):
    if table_of_particles.shape[0] <= 5: return np.nan
    config = read_config()
    wellplate_params = config['wellplate']
    coordinates = table_of_particles[['X', 'Y']].values
    dist_matrix = distance_matrix(coordinates, coordinates)
    np.fill_diagonal(dist_matrix, np.inf)
    r_min = np.nanmean(np.nanmin(dist_matrix, axis=1))
    A = np.pi * wellplate_params['radius_px'] ** 2
    n = len(table_of_particles)
    r_e = 0.5 / np.sqrt(n / A) if n > 0 else 0
    return r_min / r_e if r_e > 0 else np.nan

# --- MODIFIED HIGH-LEVEL FUNCTIONS ---

def image_subtraction_from_sequence(image_file_list, fps, delay1, delay2):
    config = read_config()
    wellplate_params = config['wellplate']
    frame_num1, frame_num2 = int(delay1 * fps), int(delay2 * fps)
    image_a = get_frame_from_sequence(image_file_list, frame_num1)
    image_b = get_frame_from_sequence(image_file_list, frame_num2)
    if image_a is None or image_b is None: return None, None
    subtr_image = np.clip(cv2.subtract(image_a, image_b), 0, 255).astype(np.uint8)
    if wellplate_params['create_mask']:
        masked_image, mask = mask_well_area(image_a)
        result_image = cv2.bitwise_and(subtr_image, subtr_image, mask=mask)
    else: result_image, masked_image = subtr_image, image_a
    return result_image, masked_image

def image_analysis_of_sample(run_folder_path, image_file_list, frame1_delay=0, frame2_delay=2, frame3_delay=5):
    config=read_config()
    output_params = config['outputs']
    fps = get_fps_from_sequence(run_folder_path)

    subtr_image1, masked_image = image_subtraction_from_sequence(image_file_list, fps, delay1=frame1_delay, delay2=frame2_delay)
    subtr_image2, _ = image_subtraction_from_sequence(image_file_list, fps, delay1=frame1_delay, delay2=frame3_delay)
    if subtr_image1 is None or subtr_image2 is None: return pd.DataFrame()

    table_of_particles1, binary_image1 = analyze_microorganisms(subtr_image1)
    table_of_particles2, binary_image2 = analyze_microorganisms(subtr_image2)
    table_of_particles1['particle_type'], table_of_particles2['particle_type'] = 0, 0
    
    top1 = compare_detected_particles(table_of_particles1, table_of_particles2)
    frame1_num = int(frame1_delay * fps)
    fframe = get_frame_from_sequence(image_file_list, frame1_num)
    if fframe is not None:
        masked_fframe, _ = mask_well_area(fframe)
        top2, _ = analyze_unsubtracted(masked_fframe)
        top3 = compare_detected_particles(top1, top2)
        table_of_particles = top3[top3['in_ref'] != 0].copy().reset_index(drop=True)
    else: table_of_particles = top1

    if output_params['particle_detection']:
        filename = os.path.basename(run_folder_path)
        output_path = os.path.join(os.path.dirname(run_folder_path), f'{filename}_particle_detection')
        os.makedirs(output_path, exist_ok=True)
        first_frame = get_frame_from_sequence(image_file_list, frame1_num)
        if first_frame is not None:
            cv2.imwrite(os.path.join(output_path, 'frame1_particles.jpg'), label_particles(first_frame.copy(), table_of_particles))
            cv2.imwrite(os.path.join(output_path, 'frame1_masked_well.jpg'), masked_image)
        cv2.imwrite(os.path.join(output_path, 'image_subtraction1.jpg' ), label_particles(binary_image1, table_of_particles1))
        cv2.imwrite(os.path.join(output_path, 'image_subtraction2.jpg' ), label_particles(binary_image2, table_of_particles2))
        table_of_particles.to_csv(os.path.join(output_path, 'table_of_particles.csv'), index=False)

    return table_of_particles

def count_particles(run_folder_path):
    image_file_list = get_image_file_list(run_folder_path)
    fps = get_fps_from_sequence(run_folder_path)
    total_frames = len(image_file_list)
    
    if total_frames < (5 * fps + 2):
        print(f"Warning: Not enough frames in {run_folder_path} for full analysis.")
        return pd.DataFrame({'avg_particles': [0], 'median_particle_size': [np.nan], 'spatial_nni': [np.nan]})

    frame_num1, frame_num2, frame_num3 = 1, total_frames // 2, total_frames - int(2*fps) - 2
    delay1, delay2, delay3 = frame_num1 / fps, frame_num2 / fps, frame_num3 / fps
    
    table_of_particles1 = image_analysis_of_sample(run_folder_path, image_file_list, frame1_delay=delay1, frame2_delay=delay1+2)
    table_of_particles2 = image_analysis_of_sample(run_folder_path, image_file_list, frame1_delay=delay2, frame2_delay=delay2-2)
    table_of_particles3 = image_analysis_of_sample(run_folder_path, image_file_list, frame1_delay=delay3, frame2_delay=delay3+2)
        
    p1, p2, p3 = len(table_of_particles1), len(table_of_particles2), len(table_of_particles3)
    nni1, nni2, nni3 = spatial_analysis(table_of_particles1), spatial_analysis(table_of_particles2), spatial_analysis(table_of_particles3)
    avg_particles, nni = round((p1 + p2 + p3) / 3, 1), np.nanmean([nni1, nni2, nni3])
    all_particles = pd.concat([table_of_particles1, table_of_particles2, table_of_particles3])
    median_area = all_particles['area'].median() if not all_particles.empty else np.nan
        
    print(f"Individual counts: {p1}, {p2}, {p3}\nAvg particles: {avg_particles}\nMedian size: {median_area}\nNNI: {nni}")
    return pd.DataFrame({'avg_particles': [avg_particles], 'median_particle_size': [median_area], 'spatial_nni': [nni]})