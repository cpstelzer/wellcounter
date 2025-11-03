# -*- coding: utf-8 -*-

"""
Wellcounter imaging module

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
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from math import pi
from typing import Optional

# print(f"[DEBUG] Executing imaging module from: {__file__}", flush=True)

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

def resolve_image_directory(run_folder_path):
    """Return the directory that actually stores the image sequence."""
    jpg_dir = os.path.join(run_folder_path, "jpg")
    return jpg_dir if os.path.isdir(jpg_dir) else run_folder_path

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
    #print(f"Frame_path: {frame_path}") # for debugging
    if not os.path.exists(frame_path):
        print(f"Error: Image file not found at {frame_path}")
        return None
    frame = cv2.imread(frame_path, cv2.IMREAD_GRAYSCALE)
    return frame

def determine_reference_frames(run_folder_path):
    """Determine dataset type and reference frames for a run folder."""
    image_dir = resolve_image_directory(run_folder_path)
    image_file_list = get_image_file_list(image_dir)
    total_frames = len(image_file_list)
    dataset_type = "sampled" if total_frames <= 5 else "full_series"

    frame_numbers = [
        int(m.group(1)) if (m := re.search(r'_f(\d+)_', os.path.basename(f))) else np.nan
        for f in image_file_list
    ]

    reference_indices = []
    if total_frames == 0:
        reference_frame_numbers = []
    elif total_frames < 3:
        reference_indices = list(range(total_frames))
        reference_frame_numbers = [
            frame_numbers[idx] if idx < len(frame_numbers) and not pd.isna(frame_numbers[idx]) else idx
            for idx in reference_indices
        ]
    else:
        if dataset_type == "full_series":
            fps = get_fps_from_sequence(image_dir)
            frame1_idx = 1
            frame2_idx = total_frames // 2
            frame3_idx = total_frames - int(2 * fps) - 2
            frame1_idx = min(max(0, int(frame1_idx)), total_frames - 1)
            frame2_idx = min(max(0, int(frame2_idx)), total_frames - 1)
            frame3_idx = min(max(0, int(frame3_idx)), total_frames - 1)
            reference_indices = [frame1_idx, frame2_idx, frame3_idx]
            if len(set(reference_indices)) < 3:
                reference_indices = [0, total_frames // 2, total_frames - 1]
        else:
            if any(pd.isna(num) for num in frame_numbers):
                sorted_indices = list(range(total_frames))
            else:
                sorted_indices = list(np.argsort(frame_numbers))
            reference_indices = sorted_indices[:3]

        reference_indices = [int(idx) for idx in reference_indices[:3]]
        reference_frame_numbers = [
            frame_numbers[idx] if idx < len(frame_numbers) and not pd.isna(frame_numbers[idx]) else idx
            for idx in reference_indices
        ]

    return dataset_type, reference_indices, reference_frame_numbers, image_file_list, frame_numbers

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

# --- MODIFIED FUNCTION: mask_well_area (added caching support) ---
def mask_well_area(image, cached_center=None, cached_mask=None):
    """
    Mask the well area of an image.
    If cached_center or cached_mask is provided, reuse them to skip recomputation.
    """
    config = read_config()
    wellplate_params = config['wellplate']
    system_params = config['system']

    if len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # --- Reuse cached mask if provided ---
    if cached_mask is not None:
        print("[mask_well_area] Reusing cached well mask", flush=True)
        result_image = cv2.bitwise_and(image, image, mask=cached_mask)
        return result_image, cached_mask

    print("[mask_well_area] Determining new well mask ...", flush=True)

    center = cached_center
    threshold = None
    
    try:
        if center is None and wellplate_params['auto_center_mask']:
            _, threshold = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if len(contours) == 0:
                print("[mask_well_area] No contours found!", flush=True)
                center = (int(system_params['image_width_px'] / 2),
                          int(system_params['image_height_px'] / 2))
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                (cx, cy), _ = cv2.minEnclosingCircle(largest_contour)
                if not (0 <= cx <= system_params['image_width_px'] and 0 <= cy <= system_params['image_height_px']):
                    print(f"[mask_well_area] Invalid center: ({cx:.1f},{cy:.1f}) – using image center", flush=True)
                    cx, cy = system_params['image_width_px'] / 2, system_params['image_height_px'] / 2
                center = (int(cx), int(cy))
        elif center is None:
            center = (int(system_params['image_width_px'] / 2),
                      int(system_params['image_height_px'] / 2))
    except Exception as e:
        print(f"[mask_well_area] ERROR during contour detection: {e}", flush=True)
        center = (int(system_params['image_width_px'] / 2),
                  int(system_params['image_height_px'] / 2))

    radius = int(wellplate_params['radius_px'] * 0.9)
    mask = np.zeros_like(image)
    cv2.circle(mask, center, radius, 255, thickness=-1)
    result_image = cv2.bitwise_and(image, image, mask=mask)
    print(f"[mask_well_area] Well mask determined: center = {center}, radius = {radius}", flush=True)
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
    return df, binary_image

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

def visualize_shape_filtering(image, df_before, df_after, output_path=None):
    """
    Visualize which particles were excluded by the shape filter.

    Parameters
    ----------
    image : np.ndarray
        Grayscale or color image.
    df_before : pandas.DataFrame
        Particle table before filtering.
    df_after : pandas.DataFrame
        Particle table after filtering.
    output_path : str, optional
        If given, saves the resulting image to this path.

    Returns
    -------
    np.ndarray
        Image with kept (green) and excluded (red) particles drawn.
    """
    if df_before is None or df_before.empty:
        print("[visualize_shape_filtering] No particles to visualize.")
        return image
    if len(image.shape) == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

    # Prepare coordinate sets
    kept_coords = set(zip(df_after['X'], df_after['Y']))
    all_coords = set(zip(df_before['X'], df_before['Y']))
    excluded_coords = all_coords - kept_coords

    # Determine radius from config
    config = read_config()
    params = config['particle_detection']
    radius = round(params['search_radius_factor'] *
                   np.sqrt(params['default_particle_area'] / np.pi))

    # Draw green for kept particles
    for (x, y) in kept_coords:
        cv2.circle(image, (int(x), int(y)), radius, (0, 255, 0), 2)

    # Draw red for excluded particles
    for (x, y) in excluded_coords:
        cv2.circle(image, (int(x), int(y)), radius, (0, 0, 255), 2)

    print(f"[visualize_shape_filtering] Kept {len(kept_coords)}, excluded {len(excluded_coords)} particles.")

    if output_path:
        cv2.imwrite(output_path, image)
    return image

def filter_particles_by_shape(df):
    """
    Apply post-detection shape filtering to particle tables.
    Removes particles that are too large (area > 3000) OR too irregular (solidity < 0.2).
    """
    if df is None or df.empty:
        return df
    mask = (df['area'] <= 3000) & (df['solidity'] >= 0.2)
    filtered = df[mask].copy().reset_index(drop=True)
    removed = len(df) - len(filtered)
    if removed > 0:
        print(f"[filter_particles_by_shape] Removed {removed} particles (area>3000 or solidity<0.2)")
    return filtered

def compare_detected_particles(df_ref, df_query, measurement_cols=None):
    """
    Compare detected particles in two data frames (reference and query)
    based on spatial proximity in X, Y coordinates. Returns a merged data frame
    containing matched and unmatched particles, with origin flags.

    Parameters
    ----------
    df_ref : pandas.DataFrame or None
        Reference particle data. Must contain 'X' and 'Y' columns.
    df_query : pandas.DataFrame or None
        Query particle data. Must contain 'X' and 'Y' columns.
    measurement_cols : list of str, optional
        List of measurement columns (e.g., ['area', 'perimeter', ...])
        that should be carried over into the merged result.
        If None, the function will automatically infer all non-essential
        columns shared by df_ref and df_query (excluding in_ref/in_query).

    Returns
    -------
    pandas.DataFrame
        Combined table with columns:
        [X, Y, <measurement_cols>, in_ref, in_query]
    """
    
    config = read_config()
    params = config['particle_detection']

    # --- define essential columns
    essential_cols = ['X', 'Y', 'in_ref', 'in_query']

    # --- determine which measurement columns to include
    if measurement_cols is None:
        # all columns except essentials found in either df
        cols_ref = set(df_ref.columns if df_ref is not None else [])
        cols_query = set(df_query.columns if df_query is not None else [])
        measurement_cols = sorted(list((cols_ref | cols_query) - set(essential_cols)))
    else:
        # make sure it's a clean list of strings
        measurement_cols = [str(c) for c in measurement_cols]

    expected_cols = essential_cols[:2] + measurement_cols + essential_cols[2:]

    # --- helper to create empty output
    def empty_df():
        return pd.DataFrame(columns=expected_cols)

    # --- handle None inputs
    if df_ref is None and df_query is None:
        return empty_df()
    if df_ref is None:
        df_ref = pd.DataFrame()
    if df_query is None:
        df_query = pd.DataFrame()

    # --- ensure all required columns exist
    for col in expected_cols:
        if col not in df_ref.columns:
            df_ref[col] = np.nan
        if col not in df_query.columns:
            df_query[col] = np.nan

    # --- handle empty inputs
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

    # --- require coordinate columns
    if 'X' not in df_ref.columns or 'Y' not in df_ref.columns or \
       'X' not in df_query.columns or 'Y' not in df_query.columns:
        return empty_df()

    # --- perform nearest-neighbor matching
    search_radius = params['search_radius_factor'] * np.sqrt(params['default_particle_area'] / np.pi)
    tree_query = BallTree(df_query[['X', 'Y']].values)
    distances, indices = tree_query.query(df_ref[['X', 'Y']].values, k=1)
    matched_query_indices, matches = set(), []

    for i, (idx, dist) in enumerate(zip(indices.flatten(), distances.flatten())):
        row_dict = df_ref.iloc[i].to_dict() # start with reference row
        if dist <= search_radius and idx not in matched_query_indices:
            matched_query_indices.add(idx)
            row_dict.update(df_query.iloc[idx].to_dict()) # overwrite with query values
            # For matched particles, all measurement columns (X, Y, area, perimeter, etc.) come from df_query
            row_dict.update({'in_ref': 1, 'in_query': 1})
        else:
            row_dict.update({'in_ref': 1, 'in_query': 0})
            # For unmatched reference particles, measurement columns remain from df_ref
        matches.append(row_dict)

    # After processing all reference rows, the unmatched ones from the query are appended
    unmatched_query = df_query.drop(index=list(matched_query_indices)).copy()
    unmatched_query['in_ref'], unmatched_query['in_query'] = 0, 1

    merged = pd.concat([pd.DataFrame(matches), unmatched_query], ignore_index=True)

    # --- ensure expected columns exist and correct order
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


# HIGH-LEVEL FUNCTIONS (frame-index based; pairwise comparisons) 

def image_subtraction_from_sequence(image_file_list, frame_idx1, frame_idx2, cached_mask=None):
    """
    Subtract two frames given by frame indices (frame_idx1 - frame_idx2).
    Returns:
      - result_image: subtracted + masked (or raw subtracted if mask disabled)
      - masked_image: masked version of image_a (used for saving masked well)

    Reuses a cached mask if provided.
    """
    config = read_config()
    wellplate_params = config['wellplate']
    image_a = get_frame_from_sequence(image_file_list, frame_idx1)
    image_b = get_frame_from_sequence(image_file_list, frame_idx2)
    if image_a is None or image_b is None:
        return None, None
    subtr_image = np.clip(cv2.subtract(image_a, image_b), 0, 255).astype(np.uint8)
    if wellplate_params['create_mask']:
        masked_image, mask = mask_well_area(image_a, cached_mask=cached_mask)
        result_image = cv2.bitwise_and(subtr_image, subtr_image, mask=mask)
    else:
        result_image, masked_image = subtr_image, image_a
    return result_image, masked_image

def image_analysis_of_sample(run_folder_path, image_file_list, ref_idx, sub_idx, cached_mask=None):
    """
    Analyze a single pair of frames: subtract frame_idx2 from frame_idx1,
    detect particles on the subtraction and also provide the binary image and
    masked reference image for downstream aggregation and saving.
    Returns:
      (table_of_particles, binary_image, masked_image)
    (no file saving is done here to avoid per-call CSV/image duplication;
     final saving occurs in count_particles() to match original output layout).
    """
    subtr_image, masked_image = image_subtraction_from_sequence(image_file_list, ref_idx, sub_idx, cached_mask=cached_mask)
    if subtr_image is None:
        return pd.DataFrame(), None, None
    table_of_particles, binary_image = analyze_microorganisms(subtr_image)
    # Ensure column exists even if empty
    if table_of_particles is None or table_of_particles.empty:
        table_of_particles = pd.DataFrame()
    table_of_particles['particle_type'] = 0
    return table_of_particles, binary_image, masked_image

def count_particles(
    run_folder_path,
    *,
    reference_indices=None,
    reference_frame_numbers=None,
    dataset_type=None,
):
    """
    Top-level controller. This version implements the "temporal sampling" method
    from the master script. It performs three independent analyses using frames
    from the beginning, middle, and end of the sequence as separate reference
    points. The final metrics are an average of these three independent samples.
    The robust engineering of the 'experimental' script (image sequence input,
    error handling) is retained.
    """
    image_dir = resolve_image_directory(run_folder_path)
    image_file_list = get_image_file_list(image_dir)
    total_frames = len(image_file_list)
    frame_numbers = [
        int(m.group(1)) if (m := re.search(r'_f(\d+)_', os.path.basename(f))) else np.nan
        for f in image_file_list
    ]

    auto_dataset_type = "sampled" if total_frames <= 5 else "full_series"
    dataset_type_resolved = dataset_type or auto_dataset_type
    if dataset_type_resolved not in {"sampled", "full_series"}:
        dataset_type_resolved = auto_dataset_type

    auto_indices = []
    auto_frame_numbers = []
    helper_data = None

    if dataset_type is not None and dataset_type_resolved != auto_dataset_type:
        print(
            f"[count_particles] Detected dataset type: {auto_dataset_type}"
            f" ({total_frames} images)"
        )
        print(
            f"[count_particles] Using provided dataset type override: {dataset_type_resolved}"
        )
    else:
        print(
            f"[count_particles] Detected dataset type: {dataset_type_resolved}"
            f" ({total_frames} images)"
        )

    if total_frames < 3:
        print(f"Warning: Not enough frames in {run_folder_path} for full analysis.")
        summary_df = pd.DataFrame(
            {
                'avg_particles': [0],
                'median_particle_size': [np.nan],
                'spatial_nni': [np.nan],
            }
        )
        empty_particles = pd.DataFrame(columns=['ref_frame', 'ref_frame_index'])
        return summary_df, empty_particles

    cleaned_indices = []
    if reference_indices is not None:
        for idx in reference_indices:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx_int < total_frames:
                cleaned_indices.append(idx_int)

    if len(cleaned_indices) >= 3:
        frame_indices = cleaned_indices[:3]
    else:
        if reference_indices is not None and len(cleaned_indices) < 3:
            print(
                "[count_particles] Warning: Provided reference_indices are invalid or incomplete;"
                " falling back to automatic selection."
            )
        helper_data = determine_reference_frames(run_folder_path)
        auto_dataset_type_helper, auto_indices, auto_frame_numbers, _, helper_frame_numbers = helper_data
        if dataset_type is None:
            dataset_type_resolved = auto_dataset_type_helper
        if not any(not pd.isna(val) for val in frame_numbers):
            frame_numbers = helper_frame_numbers
        frame_indices = auto_indices[:3]

    if len(set(frame_indices)) < 3:
        print("[count_particles] Anchor collision detected, using automatic reference frame set.")
        if helper_data is None:
            helper_data = determine_reference_frames(run_folder_path)
            auto_dataset_type_helper, auto_indices, auto_frame_numbers, _, helper_frame_numbers = helper_data
            if dataset_type is None:
                dataset_type_resolved = auto_dataset_type_helper
            if not any(not pd.isna(val) for val in frame_numbers):
                frame_numbers = helper_frame_numbers
        frame_indices = auto_indices[:3]

    def normalize_frame_value(value, fallback):
        if value is None:
            return fallback
        try:
            if pd.isna(value):
                return fallback
        except TypeError:
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    fallback_numbers = []
    for idx in frame_indices:
        if 0 <= idx < len(frame_numbers) and not pd.isna(frame_numbers[idx]):
            fallback_numbers.append(int(frame_numbers[idx]))
        else:
            fallback_numbers.append(int(idx))

    provided_numbers = reference_frame_numbers or []
    resolved_numbers = []
    for pos, fallback in enumerate(fallback_numbers):
        candidate = provided_numbers[pos] if pos < len(provided_numbers) else None
        if candidate is None and pos < len(auto_frame_numbers):
            candidate = auto_frame_numbers[pos]
        resolved_numbers.append(normalize_frame_value(candidate, fallback))

    if len(frame_indices) < 3:
        raise ValueError(
            "count_particles requires at least three reference frame indices, "
            f"but received {len(frame_indices)}."
        )

    frame1_idx, frame2_idx, frame3_idx = frame_indices[:3]

    print(
        f"[count_particles] Using frames (indices): {frame1_idx}, {frame2_idx}, {frame3_idx}"
    )

    # --- determine the well mask once for the sequence ---
    global_mask = None
    first_frame = get_frame_from_sequence(image_file_list, 0)
    if first_frame is not None:
        _, global_mask = mask_well_area(first_frame)

    # --- Define output folder path (but don't create it yet) ---
    parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
    folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
    output_dir = os.path.join(parent_dir, f"{folder_name}_particle_analysis")

    config = read_config()
    save_outputs = bool(config['outputs'].get('particle_detection', False))

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    def run_single_analysis(ref_idx, sub1_idx, sub2_idx, output_dir):
        """
        Mirrors the logic of the master script's image_analysis_of_sample.
        It takes one reference frame and performs two subtractions against it,
        merges the results, and validates against the unsubtracted frame.
        """
        # Perform the two subtractions from the reference frame
        df_sub1, bin1, masked_ref = image_analysis_of_sample(run_folder_path, image_file_list, ref_idx, sub1_idx, cached_mask=global_mask)
        df_sub2, bin2, _ = image_analysis_of_sample(run_folder_path, image_file_list, ref_idx, sub2_idx, cached_mask=global_mask)

        # Merge the two subtraction results
        merged_subs = compare_detected_particles(df_sub1, df_sub2)

        # Read the config once and decide whether to save
        config = read_config()
        save_outputs = bool(config['outputs'].get('particle_detection', False))

        # Analyze unsubtracted reference frame and compare
        ref_frame = get_frame_from_sequence(image_file_list, ref_idx)
        final_table = merged_subs  # Default if frame is unreadable
        if ref_frame is not None:
            masked_fframe, _ = mask_well_area(ref_frame, cached_mask=global_mask)
            df_unsub, binary_unsub = analyze_unsubtracted(masked_fframe)

            # For debugging only: save intermediate images
            #if save_outputs:
            #    os.makedirs(output_dir, exist_ok=True)
            
            #    if binary_unsub is not None:
            #        cv2.imwrite(os.path.join(output_dir, f'debug_ref{ref_idx}_unsubtracted_binary.png'), binary_unsub)
            #    if masked_fframe is not None:
            #        cv2.imwrite(os.path.join(output_dir, f'debug_ref{ref_idx}_masked.png'), masked_fframe)
            #    if ref_frame is not None:
            #        cv2.imwrite(os.path.join(output_dir, f'debug_ref{ref_idx}_raw.png'), ref_frame)

            merged_with_unsub = compare_detected_particles(merged_subs, df_unsub)
            # Keep only particles detected in the reference frame (i.e., moving particles)
            final_table = merged_with_unsub[merged_with_unsub['in_ref'] != 0].copy().reset_index(drop=True)

        #print(f"[run_single_analysis] RefFrame {ref_idx}: Found {len(df_sub1)} (vs {sub1_idx}) and {len(df_sub2)} (vs {sub2_idx}) particles. Final count: {len(final_table)}")
        return final_table, bin1, bin2, masked_ref, binary_unsub if 'binary_unsub' in locals() else None
    # Perform three INDEPENDENT analyses
    final_table1, binary1, binary2, masked1, binary_unsub1 = run_single_analysis(frame1_idx, frame2_idx, frame3_idx, output_dir)
    final_table2, _, _, _, _ = run_single_analysis(frame2_idx, frame1_idx, frame3_idx, output_dir)
    final_table3, _, _, _, _ = run_single_analysis(frame3_idx, frame1_idx, frame2_idx, output_dir)

    # Safely assign the true frame numbers and indices to each result table
    frame_value_map = {
        frame1_idx: resolved_numbers[0],
        frame2_idx: resolved_numbers[1],
        frame3_idx: resolved_numbers[2],
    }

    final_table1['ref_frame'] = frame_value_map.get(frame1_idx, frame1_idx)
    final_table2['ref_frame'] = frame_value_map.get(frame2_idx, frame2_idx)
    final_table3['ref_frame'] = frame_value_map.get(frame3_idx, frame3_idx)

    final_table1['ref_frame_index'] = frame1_idx
    final_table2['ref_frame_index'] = frame2_idx
    final_table3['ref_frame_index'] = frame3_idx

    # Optional post-detection shape filtering
    config = read_config()
    if config['particle_detection'].get('filter_by_shape', False):
        # Keep copies before filtering (for visualization)
        unfiltered1 = final_table1.copy()
        unfiltered2 = final_table2.copy()
        unfiltered3 = final_table3.copy()

        # Apply filtering
        final_table1 = filter_particles_by_shape(final_table1)
        final_table2 = filter_particles_by_shape(final_table2)
        final_table3 = filter_particles_by_shape(final_table3)

        # If saving outputs, create filtered visualization for the first frame
        if save_outputs:
            first_frame = get_frame_from_sequence(image_file_list, frame1_idx)
            if first_frame is not None and not unfiltered1.empty:
                output_path = os.path.join(output_dir, 'frame1_particles_filtered.jpg')
                visualize_shape_filtering(first_frame.copy(),
                                        df_before=unfiltered1,
                                        df_after=final_table1,
                                        output_path=output_path)

    # Calculate metrics by averaging the independent results
    p1 = len(final_table1)
    p2 = len(final_table2)
    p3 = len(final_table3)

    nni1 = spatial_analysis(final_table1)
    nni2 = spatial_analysis(final_table2)
    nni3 = spatial_analysis(final_table3)

    avg_particles = round((p1 + p2 + p3) / 3, 1)
    nni = np.nanmean([nni1, nni2, nni3])

    # Concatenate and compute metrics
    all_particles = pd.concat([final_table1, final_table2, final_table3], ignore_index=True)
    # Ensure 'ref_frame' is the first column
    cols = ['ref_frame', 'ref_frame_index'] + [c for c in all_particles.columns if c not in ('ref_frame', 'ref_frame_index')]
    all_particles = all_particles[cols]

    median_area = all_particles['area'].median() if not all_particles.empty else np.nan

    print(f"\nIndividual counts: {p1}, {p2}, {p3}\nAvg particles: {avg_particles}\nMedian size: {median_area}\nNNI: {nni}")

    summary_df = pd.DataFrame({'avg_particles': [avg_particles], 'median_particle_size': [median_area], 'spatial_nni': [nni]})

    # --- Output control ---
    config = read_config()
    save_outputs = bool(config['outputs'].get('particle_detection', False))

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

        # Save particle-level outputs (as before)
        first_frame = get_frame_from_sequence(image_file_list, frame1_idx)
        if first_frame is not None:
            cv2.imwrite(os.path.join(output_dir, 'frame1_particles.jpg'),
                        label_particles(first_frame.copy(), final_table1))
            if masked1 is not None:
                cv2.imwrite(os.path.join(output_dir, 'frame1_masked_well.jpg'), masked1)

        if binary1 is not None:
            cv2.imwrite(os.path.join(output_dir, 'image_subtraction1.jpg'),
                        label_particles(binary1, final_table1))
        if binary2 is not None:
            cv2.imwrite(os.path.join(output_dir, 'image_subtraction2.jpg'),
                        label_particles(binary2, final_table1))
            
        # --- Save unsubtracted binary image of first frame (with filtered particles) ---
        if save_outputs and binary_unsub1 is not None:
            unsub_path = os.path.join(output_dir, 'frame1_binary_unsub.jpg')

            # Overlay final (filtered) particle positions on the binary image
            if final_table1 is not None and not final_table1.empty:
                labeled_unsub = label_particles(binary_unsub1.copy(), final_table1)
                cv2.imwrite(unsub_path, labeled_unsub)
                print(f"[count_particles] Saved unsubtracted binary image with {len(final_table1)} filtered particles: {unsub_path}")
            else:
                cv2.imwrite(unsub_path, binary_unsub1)
                print(f"[count_particles] Saved unsubtracted binary image (no particles detected): {unsub_path}")



        all_particles.to_csv(os.path.join(output_dir, 'table_of_particles.csv'), index=False)
        
        # Save summary CSV only when enabled
        summary_df.to_csv(os.path.join(output_dir, f"{folder_name}_particle_results.csv"), index=False)

        print(f"[count_particles] Results saved to: {output_dir}")
    else:
        print("[count_particles] particle_detection disabled — no files or folders created.")

    # Always return results programmatically
    return summary_df, all_particles


def count_complete(
    run_folder_path,
    *,
    collage_metric: str = "centerline_mean_width",
    save_male_outputs: Optional[bool] = None,
):
    """Run an integrated particle and male analysis with combined reporting."""

    (
        dataset_type,
        reference_indices,
        reference_frame_numbers,
        _image_file_list,
        _frame_numbers,
    ) = determine_reference_frames(run_folder_path)

    reference_indices = [int(idx) for idx in reference_indices]

    summary_df, all_particles = count_particles(
        run_folder_path,
        reference_indices=reference_indices,
        reference_frame_numbers=reference_frame_numbers,
        dataset_type=dataset_type,
    )

    from wellcounter_male_module import count_males

    simplified_combined, _ = count_males(
        run_folder_path,
        collage_metric=collage_metric,
        save_outputs=save_male_outputs,
        reference_indices=reference_indices,
        reference_frame_numbers=reference_frame_numbers,
    )

    def normalize_frame_value(value, fallback):
        if value is None:
            return fallback
        try:
            if pd.isna(value):
                return fallback
        except TypeError:
            pass
        try:
            return int(value)
        except (TypeError, ValueError):
            return fallback

    frame_stats = []
    joined_tables = []

    for idx, ref_idx in enumerate(reference_indices):
        ref_value = reference_frame_numbers[idx] if idx < len(reference_frame_numbers) else None
        ref_value = normalize_frame_value(ref_value, ref_idx)

        if 'ref_frame_index' in simplified_combined.columns:
            ref_mask = simplified_combined['ref_frame_index'] == ref_idx
        else:
            ref_mask = pd.Series([False] * len(simplified_combined), index=simplified_combined.index)
        df_ref = simplified_combined[ref_mask].copy()

        if 'ref_frame_index' in all_particles.columns:
            query_mask = all_particles['ref_frame_index'] == ref_idx
        else:
            query_mask = pd.Series([False] * len(all_particles), index=all_particles.index)
        df_query = all_particles[query_mask].copy()

        joined_df = compare_detected_particles(df_ref, df_query).copy()
        if joined_df.empty:
            joined_df = joined_df.copy()

        joined_df['ref_frame'] = ref_value
        joined_df['ref_frame_index'] = ref_idx

        male_series = joined_df['male'] if 'male' in joined_df.columns else pd.Series(False, index=joined_df.index)
        male_series = male_series.fillna(False).astype(bool)
        in_ref = joined_df['in_ref'].fillna(0).astype(int)
        in_query = joined_df['in_query'].fillna(0).astype(int)

        conditions = [
            (in_ref == 1) & (in_query == 1) & (~male_series),
            (in_ref == 1) & (in_query == 1) & (male_series),
            (in_ref == 1) & (in_query == 0) & (male_series),
            (in_ref == 1) & (in_query == 0) & (~male_series),
            (in_ref == 0) & (in_query == 1),
        ]
        choices = ["female", "male", "male", "fpos", "impos"]
        joined_df['particle_type'] = np.select(conditions, choices, default="unknown")

        female_rows = joined_df[joined_df['particle_type'] == "female"]
        nfems = int(len(female_rows))
        nmales = int(len(joined_df[joined_df['particle_type'] == "male"]))
        female_area = pd.to_numeric(female_rows.get('area'), errors='coerce') if 'area' in female_rows else pd.Series(dtype=float)
        mean_area = float(female_area.mean()) if not female_area.empty else np.nan

        frame_stats.append(
            {
                'ref_frame': ref_value,
                'ref_frame_index': ref_idx,
                'nfems': nfems,
                'nmales': nmales,
                'area': mean_area,
                'joined_particles': joined_df,
            }
        )
        joined_tables.append(joined_df)

    if frame_stats:
        frame_stats_df = pd.DataFrame(frame_stats)
    else:
        frame_stats_df = pd.DataFrame(
            columns=['ref_frame', 'ref_frame_index', 'nfems', 'nmales', 'area', 'joined_particles']
        )

    nfems_values = [entry['nfems'] for entry in frame_stats]
    nmales_values = [entry['nmales'] for entry in frame_stats]
    area_values = [entry['area'] for entry in frame_stats]

    mean_fems = float(np.nanmean(nfems_values)) if nfems_values else np.nan
    mean_males = float(np.nanmean(nmales_values)) if nmales_values else np.nan
    mean_area = float(np.nanmean(area_values)) if area_values else np.nan
    sex_ratio = np.nan
    if not np.isnan(mean_fems) and mean_fems != 0:
        sex_ratio = mean_males / mean_fems

    aggregated_data = {
        'dataset_type': dataset_type,
        'frame_count': len(reference_indices),
        'mean_fems': mean_fems,
        'mean_males': mean_males,
        'sex_ratio': sex_ratio,
        'mean_area': mean_area,
    }

    aggregated_df = pd.DataFrame([aggregated_data])
    aggregated_df['reference_indices'] = [reference_indices]
    aggregated_df['reference_frames'] = [reference_frame_numbers]

    if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
        aggregated_df = pd.concat([aggregated_df.reset_index(drop=True), summary_df.reset_index(drop=True)], axis=1)

    combined_joined_df = pd.concat(joined_tables, ignore_index=True) if joined_tables else pd.DataFrame()
    frame_stats_df.attrs['combined_joined_particles'] = combined_joined_df

    return aggregated_df, frame_stats_df

