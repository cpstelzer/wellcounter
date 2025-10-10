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

# --- FIXED FUNCTION (robust to empty/missing columns) ---

def compare_detected_particles(df_ref, df_query):
    config = read_config()
    params = config['particle_detection']

    expected_cols = [
        'X', 'Y', 'area', 'perimeter', 'orientation',
        'aspect_ratio', 'solidity', 'eccentricity', 'feret_diameter',
        'bounding_x', 'bounding_y', 'bounding_w', 'bounding_h',
        'in_ref', 'in_query'
    ]

    def empty_df():
        return pd.DataFrame(columns=expected_cols)

    if df_ref is None and df_query is None:
        return empty_df()
    if df_ref is None:
        df_ref = pd.DataFrame()
    if df_query is None:
        df_query = pd.DataFrame()

    for col in expected_cols:
        if col not in df_ref.columns:
            df_ref[col] = np.nan
        if col not in df_query.columns:
            df_query[col] = np.nan

    if df_ref.empty and df_query.empty:
        return empty_df()
    if df_ref.empty:
        df_query = df_query.copy()
        df_query['in_ref'], df_query['in_query'] = 0, 1
        return df_query[expected_cols]
    if df_query.empty:
        df_ref = df_ref.copy()
        df_ref['in_ref'], df_ref['in_query'] = 1, 0
        return df_ref[expected_cols]

    if 'X' not in df_ref.columns or 'Y' not in df_ref.columns or \
       'X' not in df_query.columns or 'Y' not in df_query.columns:
        return empty_df()

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
        else:
            row_dict.update({'in_ref': 1, 'in_query': 0})
        matches.append(row_dict)

    unmatched_query = df_query.drop(index=list(matched_query_indices)).copy()
    unmatched_query['in_ref'], unmatched_query['in_query'] = 0, 1

    merged = pd.concat([pd.DataFrame(matches), unmatched_query], ignore_index=True)

    for col in expected_cols:
        if col not in merged.columns:
            merged[col] = np.nan

    return merged[expected_cols]


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

# --- MODIFIED HIGH-LEVEL FUNCTIONS (frame-index based; pairwise comparisons) ---

def image_subtraction_from_sequence(image_file_list, frame_idx1, frame_idx2):
    """
    Subtract two frames given by frame indices (frame_idx1 - frame_idx2).
    Returns:
      - result_image: subtracted + masked (or raw subtracted if mask disabled)
      - masked_image: masked version of image_a (used for saving masked well)
    """
    config = read_config()
    wellplate_params = config['wellplate']
    image_a = get_frame_from_sequence(image_file_list, frame_idx1)
    image_b = get_frame_from_sequence(image_file_list, frame_idx2)
    if image_a is None or image_b is None:
        return None, None
    subtr_image = np.clip(cv2.subtract(image_a, image_b), 0, 255).astype(np.uint8)
    if wellplate_params['create_mask']:
        masked_image, mask = mask_well_area(image_a)
        result_image = cv2.bitwise_and(subtr_image, subtr_image, mask=mask)
    else:
        result_image, masked_image = subtr_image, image_a
    return result_image, masked_image

def image_analysis_of_sample(run_folder_path, image_file_list, frame_idx1, frame_idx2):
    """
    Analyze a single pair of frames: subtract frame_idx2 from frame_idx1,
    detect particles on the subtraction and also provide the binary image and
    masked reference image for downstream aggregation and saving.
    Returns:
      (table_of_particles, binary_image, masked_image)
    (no file saving is done here to avoid per-call CSV/image duplication;
     final saving occurs in count_particles() to match original output layout).
    """
    subtr_image, masked_image = image_subtraction_from_sequence(image_file_list, frame_idx1, frame_idx2)
    if subtr_image is None:
        return pd.DataFrame(), None, None
    table_of_particles, binary_image = analyze_microorganisms(subtr_image)
    # Ensure column exists even if empty
    if table_of_particles is None or table_of_particles.empty:
        table_of_particles = pd.DataFrame()
    table_of_particles['particle_type'] = 0
    return table_of_particles, binary_image, masked_image

def count_particles(run_folder_path):
    """
    Top-level controller. Detects whether dataset is a full image series or a
    sampled three-image set. For a series, selects anchors near beginning, middle, end.
    For sampled datasets, parses frame numbers from filenames and sorts them.
    Performs three pairwise analyses: (f1,f2), (f2,f3), (f3,f1), aggregates results
    into a single table_of_particles (as in the original module) and saves outputs
    one level above the input folder in <inputfolder>_particle_analysis/.
    """
    image_file_list = get_image_file_list(run_folder_path)
    total_frames = len(image_file_list)
    if total_frames < 3:
        print(f"Warning: Not enough frames in {run_folder_path} for full analysis.")
        return pd.DataFrame({'avg_particles': [0], 'median_particle_size': [np.nan], 'spatial_nni': [np.nan]})

    # Detect dataset type
    if total_frames <= 5:
        dataset_type = "sampled"
    else:
        dataset_type = "full_series"
    print(f"[count_particles] Detected dataset type: {dataset_type} ({total_frames} images)")

    fps = get_fps_from_sequence(run_folder_path)

    if dataset_type == "full_series":
        frame1 = 1
        frame2 = total_frames // 2
        frame3 = total_frames - int(2 * fps) - 2
        # clamp
        frame1 = min(max(0, int(frame1)), total_frames - 1)
        frame2 = min(max(0, int(frame2)), total_frames - 1)
        frame3 = min(max(0, int(frame3)), total_frames - 1)
        if len({frame1, frame2, frame3}) < 3:
            frame1, frame2, frame3 = 0, total_frames // 2, total_frames - 1
            print(f"[count_particles] Anchor collision detected, falling back to frames {frame1}, {frame2}, {frame3}")
        print(f"[count_particles] Using frames (indices): "
              f"{frame1} ({os.path.basename(image_file_list[frame1])}), "
              f"{frame2} ({os.path.basename(image_file_list[frame2])}), "
              f"{frame3} ({os.path.basename(image_file_list[frame3])})")
    else:
        # sample dataset: parse numeric frame numbers from filenames and sort by that number
        frame_numbers = []
        for f in image_file_list:
            match = re.search(r'_f(\d+)_', os.path.basename(f))
            if match:
                frame_numbers.append(int(match.group(1)))
            else:
                frame_numbers.append(np.nan)
        if any(np.isnan(frame_numbers)):
            print("[count_particles] Warning: could not parse frame numbers from all filenames; using file order.")
            sorted_indices = list(range(total_frames))
        else:
            sorted_indices = np.argsort(frame_numbers)
        frame1, frame2, frame3 = sorted_indices[0], sorted_indices[1], sorted_indices[2]
        print(f"[count_particles] Using sampled frames: "
              f"{os.path.basename(image_file_list[frame1])}, "
              f"{os.path.basename(image_file_list[frame2])}, "
              f"{os.path.basename(image_file_list[frame3])}")

    # Perform the three pairwise analyses (no per-call saving)
    df1, binary1, masked1 = image_analysis_of_sample(run_folder_path, image_file_list, frame1, frame2)
    df2, binary2, masked2 = image_analysis_of_sample(run_folder_path, image_file_list, frame2, frame3)
    df3, binary3, masked3 = image_analysis_of_sample(run_folder_path, image_file_list, frame3, frame1)

    # Aggregate detected particles analogous to previous merging logic:
    # merge df1 and df2, then merge with df3, then cross-check with unsubtracted masked frame1
    try:
        merged12 = compare_detected_particles(df1, df2)
    except Exception:
        merged12 = pd.DataFrame() if df1.empty and df2.empty else (df1 if not df1.empty else df2)
    try:
        merged123 = compare_detected_particles(merged12, df3)
    except Exception:
        merged123 = merged12 if not merged12.empty else df3

    # analyze unsubtracted masked version of frame1 and compare
    fframe = get_frame_from_sequence(image_file_list, frame1)
    if fframe is not None:
        masked_fframe, _ = mask_well_area(fframe)
        top2_unsub, _ = analyze_unsubtracted(masked_fframe)
        merged_with_unsub = compare_detected_particles(merged123, top2_unsub)
        final_table = merged_with_unsub[merged_with_unsub['in_ref'] != 0].copy().reset_index(drop=True)
    else:
        final_table = merged123

    # Diagnostics: print per-pair counts (subtracted / unsubtracted / final kept)
    # For consistency with earlier diagnostics, compute and print numbers:
    # subtracted = len(dfX), unsubtracted = len(top2_unsub if available), final_kept = length after filter
    # We'll print those for each subtraction where available.
    # For pair1: frame1-frame2
    sub1 = len(df1) if df1 is not None else 0
    sub2 = len(df2) if df2 is not None else 0
    sub3 = len(df3) if df3 is not None else 0
    unsub_count = len(top2_unsub) if fframe is not None else np.nan
    # final kept per pair cannot be trivially computed after global merge; we print global diagnostics below
    print(f"[image_analysis_of_sample] Frames {frame1} ({os.path.basename(image_file_list[frame1])}) - "
          f"{frame2} ({os.path.basename(image_file_list[frame2])}): subtracted={sub1}")
    print(f"[image_analysis_of_sample] Frames {frame2} ({os.path.basename(image_file_list[frame2])}) - "
          f"{frame3} ({os.path.basename(image_file_list[frame3])}): subtracted={sub2}")
    print(f"[image_analysis_of_sample] Frames {frame3} ({os.path.basename(image_file_list[frame3])}) - "
          f"{frame1} ({os.path.basename(image_file_list[frame1])}): subtracted={sub3}")

    # Compute reported metrics
    p1, p2, p3 = len(final_table), len(final_table), len(final_table)  # to preserve original single-table behaviour for reporting
    # NOTE: original code reported counts per pair; because we now produce a single merged table (final_table),
    # we use lengths of pair-specific detections to provide per-pair counts for transparency:
    p1_pair, p2_pair, p3_pair = sub1, sub2, sub3
    nni1, nni2, nni3 = spatial_analysis(df1), spatial_analysis(df2), spatial_analysis(df3)
    # For averaging report use the per-pair detection numbers (similar spirit to earlier runs)
    avg_particles = round((p1_pair + p2_pair + p3_pair) / 3, 1)
    nni = np.nanmean([nni1, nni2, nni3])
    all_particles = pd.concat([df1, df2, df3]) if not (df1.empty and df2.empty and df3.empty) else pd.DataFrame()
    median_area = all_particles['area'].median() if not all_particles.empty else np.nan

    print(f"Individual counts: {p1_pair}, {p2_pair}, {p3_pair}\nAvg particles: {avg_particles}\nMedian size: {median_area}\nNNI: {nni}")

    # --- Save outputs in a single output dir one level above input folder ---
    parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
    folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
    output_dir = os.path.join(parent_dir, f"{folder_name}_particle_analysis")
    os.makedirs(output_dir, exist_ok=True)

    config = read_config()
    output_params = config['outputs']
    if output_params.get('particle_detection', False):
        # Save the labelled first frame with final_table
        first_frame = get_frame_from_sequence(image_file_list, frame1)
        if first_frame is not None:
            cv2.imwrite(os.path.join(output_dir, 'frame1_particles.jpg'),
                        label_particles(first_frame.copy(), final_table))
            # For masked well picture: prefer masked1 (mask applied to image_a of first subtraction)
            if masked1 is not None:
                cv2.imwrite(os.path.join(output_dir, 'frame1_masked_well.jpg'), masked1)
        # Save labelled subtraction images (use binary images from pair1 and pair2 to keep two files as original)
        if binary1 is not None:
            cv2.imwrite(os.path.join(output_dir, 'image_subtraction1.jpg'),
                        label_particles(binary1, df1 if not df1.empty else pd.DataFrame()))
        if binary2 is not None:
            cv2.imwrite(os.path.join(output_dir, 'image_subtraction2.jpg'),
                        label_particles(binary2, df2 if not df2.empty else pd.DataFrame()))
        # Save single final table_of_particles.csv (match original filename)
        final_table.to_csv(os.path.join(output_dir, 'table_of_particles.csv'), index=False)

    # Save summary CSV as well (original count_particles returned a DataFrame but did not save a summary CSV).
    # We keep a single summary file as convenience, named <folder>_particle_results.csv
    pd.DataFrame({'avg_particles': [avg_particles], 'median_particle_size': [median_area], 'spatial_nni': [nni]}).to_csv(
        os.path.join(output_dir, f"{folder_name}_particle_results.csv"), index=False)

    print(f"[count_particles] Results saved to: {output_dir}")

    return pd.DataFrame({'avg_particles': [avg_particles], 'median_particle_size': [median_area], 'spatial_nni': [nni]})
