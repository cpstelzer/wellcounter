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

import numpy as np
import cv2
import networkx as nx
from skimage.morphology import thin

def extract_major_ridge(mask):
    """
    Extract a single, smooth centerline near the geometric middle of an irregular particle.
    Prefers high-distance (central) pixels instead of purely longest endpoints.
    Returns a boolean array of the same shape.
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return np.zeros_like(mask, bool)

    # Distance transform (L2)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Initial ridge region: high-distance zone, thinned
    ridge = dist > 0.5 * dist[mask > 0].max()
    ridge = thin(ridge)

    # Build weighted graph where edges in thick areas are cheaper
    ys, xs = np.nonzero(ridge)
    if len(ys) < 2:
        return ridge

    G = nx.Graph()
    for y, x in zip(ys, xs):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < ridge.shape[0] and 0 <= xx < ridge.shape[1] and ridge[yy, xx]:
                    # Edge cost inversely proportional to centrality (distance value)
                    w = 1.0 / (1e-3 + 0.5 * (dist[y, x] + dist[yy, xx]))
                    G.add_edge((y, x), (yy, xx), weight=w)

    # Find two endpoints that maximize weighted path length
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    max_d, start, end = 0, None, None
    for u, dists in lengths.items():
        for v, d in dists.items():
            if d > max_d:
                max_d, start, end = d, u, v
    if start is None or end is None:
        return ridge

    path = nx.shortest_path(G, start, end, weight='weight')

    # Create final mask
    clean = np.zeros_like(ridge, bool)
    for (y, x) in path:
        clean[y, x] = True

    return clean




def analyze_long_exposure_particles_advanced(long_exposure_image, run_folder_path,
                                             collage_metric="mean_width"):
    """
    Advanced morphological analysis of binary long-exposure images (LEI),
    quantifying 'eyelash-like' traces and creating diagnostic plots.

    Adds a subfunction that builds a collage of fixed-size (250x250 px)
    cropped regions from the first frame, centered on each LEI particle
    and overlaid with its skeleton.

    Parameters
    ----------
    long_exposure_image : np.ndarray
        Binary (0/255) LEI image of summed particle traces.
    run_folder_path : str
        Path to the run folder (used to locate output and input images).
    collage_metric : str, optional
        Column name in metrics DataFrame to sort the collage by.

    Returns
    -------
    pandas.DataFrame
        Metrics per particle.
    """
    
    # --- Load config ---
    try:
        with open("wellcounter_config.yml", "r") as f:
            config = yaml.safe_load(f)
        save_outputs = bool(config.get("outputs", {}).get("particle_detection", False))
    except Exception:
        save_outputs = False

    # --- Output directory (same as imaging module) ---
    parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
    folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
    output_dir = os.path.join(parent_dir, f"{folder_name}_particle_analysis")
    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)

    # --- Binary image preparation ---
    if len(long_exposure_image.shape) == 3:
        gray = cv2.cvtColor(long_exposure_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = long_exposure_image.copy()
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

    labeled = label(binary > 0)
    props = regionprops(labeled)

    if save_outputs:
        overlay = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    results, skeletons = [], []  # store metrics and skeletons

    for idx, region in enumerate(props, start=1):
        if region.area < 10:
            continue

        mask = (labeled == region.label).astype(np.uint8)
        area = int(region.area)

        # Skeleton
        skeleton = skeletonize(mask > 0)
        skel_coords = np.column_stack(np.nonzero(skeleton))
        skeletons.append(skel_coords)
        n_pixels = len(skel_coords)
        if n_pixels < 2:
            continue
        
        # --- Geodesic centerline extraction ---
        ridge = extract_major_ridge(mask)
        region_ridge_pixels = np.argwhere(ridge)
        if region_ridge_pixels.size > 0:
            ridge_length = len(region_ridge_pixels)
        else:
            ridge_length = 0

        # Skeleton metrics
        diffs = np.diff(skel_coords, axis=0)
        step_lengths = np.sqrt((diffs ** 2).sum(axis=1))
        skeleton_length = float(step_lengths.sum())

        dists = np.sqrt(((skel_coords[:, None, :] - skel_coords[None, :, :]) ** 2).sum(axis=2))
        i, j = np.unravel_index(np.argmax(dists), dists.shape)
        chord_length = dists[i, j]
        straightness_ratio = chord_length / skeleton_length if skeleton_length > 0 else np.nan

        if n_pixels >= 3:
            vecs = np.diff(skel_coords.astype(float), axis=0)
            angles = np.arctan2(vecs[:, 0], vecs[:, 1])
            dtheta = np.abs(np.diff(angles))
            dtheta[dtheta > np.pi] -= np.pi
            mean_curvature = float(np.mean(dtheta))
        else:
            mean_curvature = np.nan

        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        width_values = dist_transform[skeleton] * 2
        mean_width = float(np.mean(width_values))
        width_std = float(np.std(width_values))

        # Classical metrics
        contours, _ = cv2.findContours(region.convex_image.astype(np.uint8),
                                       cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            perimeter = cv2.arcLength(contours[0], True)
        else:
            perimeter = np.nan
        solidity = region.solidity
        circularity = (4 * pi * area) / (perimeter ** 2) if perimeter and perimeter > 0 else np.nan

        cx, cy = region.centroid[::-1]

        results.append({
            "particle_id": idx,
            "X": cx,
            "Y": cy,
            "area": area,
            "skeleton_length": skeleton_length,
            "chord_length": chord_length,
            "straightness_ratio": straightness_ratio,
            "mean_curvature": mean_curvature,
            "mean_width": mean_width,
            "width_std": width_std,
            "solidity": solidity,
            "circularity": circularity,
            "ridge_length": ridge_length
        })

        # Visualization overlay
        if save_outputs:
            contour = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
            cv2.drawContours(overlay, contour, -1, (0, 255, 0), 1)
            hue = int(120 * straightness_ratio) if not np.isnan(straightness_ratio) else 0
            hue = max(0, min(120, hue))
            col = tuple(int(c) for c in cv2.cvtColor(
                np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR
            )[0, 0])
            for (y_, x_) in skel_coords:
                cv2.circle(overlay, (int(x_), int(y_)), 0, col, 1)
            label_text = f"{idx}:{straightness_ratio:.2f}"
            cv2.putText(overlay, label_text, (int(cx), int(cy)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    df = pd.DataFrame(results)
    print(f"[analyze_long_exposure_particles_advanced] Analyzed {len(df)} traces.")

    # Save results
    if save_outputs:
        analyzed_path = os.path.join(output_dir, "LEI_males_analyzed.jpg")
        df_path = os.path.join(output_dir, "LEI_males_metrics.csv")
        cv2.imwrite(analyzed_path, overlay)
        df.to_csv(df_path, index=False)
        print(f"[analyze_long_exposure_particles_advanced] Saved: {analyzed_path}")
        print(f"[analyze_long_exposure_particles_advanced] Saved: {df_path}")


        # --- Diagnostic visualization: Geodesic centerline overlay ---
    if save_outputs:
        lei_centerline_overlay = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

        for region in props:
            mask = (labeled == region.label).astype(np.uint8)
            ridge = extract_major_ridge(mask)
            ys, xs = np.nonzero(ridge)
            for y, x in zip(ys, xs):
                if 0 <= y < lei_centerline_overlay.shape[0] and 0 <= x < lei_centerline_overlay.shape[1]:
                    lei_centerline_overlay[y, x] = (0, 0, 255)  # red ridge pixels

        out_path_centerline = os.path.join(output_dir, "LEI_males_centerline.jpg")
        cv2.imwrite(out_path_centerline, lei_centerline_overlay)
        print(f"[analyze_long_exposure_particles_advanced] Geodesic centerline overlay saved: {out_path_centerline}")


    # ----------------------------------------------------------------------
    # --- Subfunction: Diagnostic Collage with skeletons--------------------
    # ----------------------------------------------------------------------
    def create_diagnostic_collage_fixed(df, skeletons, metric="mean_width",
                                        crop_size=250, n_cols=6):
        """
        Create collage of 250x250 px crops centered on LEI particle centroids,
        extracted from the first frame and overlaid with skeleton (yellow).
        """
        import glob

        print("[collage] Starting collage creation...")
        try:
            # Try to find the first frame image automatically
            jpg_dir = os.path.join(run_folder_path, "jpg")
            image_files = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))
            if not image_files:
                print(f"[collage] No images found in {jpg_dir}. Cannot create collage.")
                return
            first_frame_path = image_files[0]
            print(f"[collage] Using first frame: {first_frame_path}")

            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                print(f"[collage] Failed to read {first_frame_path}.")
                return

            df_sorted = df.sort_values(by=metric, ascending=True).reset_index(drop=True)
            n_particles = len(df_sorted)
            n_rows = int(np.ceil(n_particles / n_cols))
            half = crop_size // 2
            h, w = first_frame.shape[:2]
            crops = []

            for i, row in df_sorted.iterrows():
                cx, cy = int(row["X"]), int(row["Y"])
                x1, x2 = max(0, cx - half), min(w, cx + half)
                y1, y2 = max(0, cy - half), min(h, cy + half)
                crop = first_frame[y1:y2, x1:x2].copy()

                # Overlay skeleton in yellow
                skel = skeletons[int(row["particle_id"]) - 1]
                for (yy, xx) in skel:
                    if x1 <= xx < x2 and y1 <= yy < y2:
                        crop[int(yy - y1), int(xx - x1)] = (0, 255, 255)

                crop = cv2.resize(crop, (crop_size, crop_size))
                cv2.putText(crop, f"{metric}={row[metric]:.2f}",
                            (5, crop_size - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                crops.append(crop)

            if not crops:
                print("[collage] No valid crops created.")
                return

            rows = []
            for i in range(n_rows):
                row_imgs = crops[i * n_cols:(i + 1) * n_cols]
                if len(row_imgs) < n_cols:
                    pad_img = np.zeros_like(row_imgs[0])
                    row_imgs += [pad_img] * (n_cols - len(row_imgs))
                rows.append(np.hstack(row_imgs))
            collage = np.vstack(rows)

            collage_path = os.path.join(output_dir, f"particle_collage_by_{metric}.jpg")
            cv2.imwrite(collage_path, collage)
            print(f"[collage] Saved diagnostic collage: {collage_path}")

        except Exception as e:
            import traceback
            print(f"[collage] Error while creating collage:\n{traceback.format_exc()}")


    # ----------------------------------------------------------------------
    # --- Subfunction: Diagnostic Collage with geodesic centerlines --------
    # ----------------------------------------------------------------------
    def create_diagnostic_collage_centerline(df, metric="mean_width",
                                            crop_size=250, n_cols=6):
        """
        Create collage of 250x250 px crops centered on LEI particle centroids,
        extracted from the first frame and overlaid with geodesic centerlines (green).
        Otherwise identical to create_diagnostic_collage_fixed().
        """
        import glob

        print("[collage_centerline] Starting centerline collage creation...")
        try:
            # Locate first frame
            jpg_dir = os.path.join(run_folder_path, "jpg")
            image_files = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))
            if not image_files:
                print(f"[collage_centerline] No images found in {jpg_dir}. Cannot create collage.")
                return
            first_frame_path = image_files[0]
            print(f"[collage_centerline] Using first frame: {first_frame_path}")

            first_frame = cv2.imread(first_frame_path)
            if first_frame is None:
                print(f"[collage_centerline] Failed to read {first_frame_path}.")
                return

            # Sort and prepare layout
            df_sorted = df.sort_values(by=metric, ascending=True).reset_index(drop=True)
            n_particles = len(df_sorted)
            n_rows = int(np.ceil(n_particles / n_cols))
            half = crop_size // 2
            h, w = first_frame.shape[:2]
            crops = []

            for i, row in df_sorted.iterrows():
                cx, cy = int(row["X"]), int(row["Y"])
                x1, x2 = max(0, cx - half), min(w, cx + half)
                y1, y2 = max(0, cy - half), min(h, cy + half)
                crop = first_frame[y1:y2, x1:x2].copy()

                # --- Overlay geodesic centerline (in bright green) ---
                local_mask = np.zeros((h, w), dtype=np.uint8)
                local_mask[labeled == row["particle_id"]] = 1
                mask_crop = local_mask[y1:y2, x1:x2]

                ridge = extract_major_ridge(mask_crop)
                ys, xs = np.nonzero(ridge)
                for y, x in zip(ys, xs):
                    if 0 <= y < crop.shape[0] and 0 <= x < crop.shape[1]:
                        crop[y, x] = (0, 255, 0)

                crop = cv2.resize(crop, (crop_size, crop_size))
                cv2.putText(crop, f"{metric}={row[metric]:.2f}",
                            (5, crop_size - 10), cv2.FONT_HERSHEY_SIMPLEX,
                            0.5, (255, 255, 255), 1, cv2.LINE_AA)
                crops.append(crop)

            if not crops:
                print("[collage_centerline] No valid crops created.")
                return

            # Combine all crops into grid
            rows = []
            for i in range(n_rows):
                row_imgs = crops[i * n_cols:(i + 1) * n_cols]
                if len(row_imgs) < n_cols:
                    pad_img = np.zeros_like(row_imgs[0])
                    row_imgs += [pad_img] * (n_cols - len(row_imgs))
                rows.append(np.hstack(row_imgs))
            collage = np.vstack(rows)

            # Save output
            collage_path = os.path.join(output_dir, f"particle_collage_centerline_by_{metric}.jpg")
            cv2.imwrite(collage_path, collage)
            print(f"[collage_centerline] Saved: {collage_path}")

        except Exception as e:
            import traceback
            print(f"[collage_centerline] Error while creating collage:\n{traceback.format_exc()}")



    # Run collage creation if configured
    if save_outputs:
        create_diagnostic_collage_fixed(df, skeletons, metric=collage_metric)
        create_diagnostic_collage_centerline(df, metric="mean_width",
                                            crop_size=250, n_cols=6)

    return df



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

def count_particles(run_folder_path):
    """
    Top-level controller. This version implements the "temporal sampling" method
    from the master script. It performs three independent analyses using frames
    from the beginning, middle, and end of the sequence as separate reference
    points. The final metrics are an average of these three independent samples.
    The robust engineering of the 'experimental' script (image sequence input,
    error handling) is retained.
    """
    image_file_list = get_image_file_list(run_folder_path)
    total_frames = len(image_file_list)
    if total_frames < 3:
        print(f"Warning: Not enough frames in {run_folder_path} for full analysis.")
        return pd.DataFrame({'avg_particles': [0], 'median_particle_size': [np.nan], 'spatial_nni': [np.nan]})
    
     # --- NEW: determine the well mask once for the sequence ---
    first_frame = get_frame_from_sequence(image_file_list, 0)
    _, global_mask = mask_well_area(first_frame)
    
    # Frame selection logic
    if total_frames <= 5:
        dataset_type = "sampled"
    else:
        dataset_type = "full_series"
    print(f"[count_particles] Detected dataset type: {dataset_type} ({total_frames} images)")

    fps = get_fps_from_sequence(run_folder_path)

    if dataset_type == "full_series":
        frame1_idx = 1
        frame2_idx = total_frames // 2
        frame3_idx = total_frames - int(2 * fps) - 2
        frame1_idx = min(max(0, int(frame1_idx)), total_frames - 1)
        frame2_idx = min(max(0, int(frame2_idx)), total_frames - 1)
        frame3_idx = min(max(0, int(frame3_idx)), total_frames - 1)
        if len({frame1_idx, frame2_idx, frame3_idx}) < 3:
            frame1_idx, frame2_idx, frame3_idx = 0, total_frames // 2, total_frames - 1
            print(f"[count_particles] Anchor collision detected, falling back to frames {frame1_idx}, {frame2_idx}, {frame3_idx}")
    else:
        frame_numbers = [int(m.group(1)) if (m := re.search(r'_f(\d+)_', os.path.basename(f))) else np.nan for f in image_file_list]
        if any(np.isnan(frame_numbers)):
            print("[count_particles] Warning: could not parse frame numbers; using file order.")
            sorted_indices = list(range(total_frames))
        else:
            sorted_indices = np.argsort(frame_numbers)
        frame1_idx, frame2_idx, frame3_idx = sorted_indices[0], sorted_indices[1], sorted_indices[2]

    print(f"[count_particles] Using frames (indices): {frame1_idx}, {frame2_idx}, {frame3_idx}")

    # --- Define output folder path (but don't create it yet) ---
    parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
    folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
    output_dir = os.path.join(parent_dir, f"{folder_name}_particle_analysis")

    config = read_config()
    save_outputs = bool(config['outputs'].get('particle_detection', False))

    if save_outputs:
        os.makedirs(output_dir, exist_ok=True)


    # Helper function for a single, self-contained analysis 
        # Helper function for a single, self-contained analysis 
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

    # --- Add actual reference frame numbers instead of indices ---
    # Extract frame numbers from filenames (e.g. "_f00087_")
    frame_numbers = [
        int(m.group(1)) if (m := re.search(r'_f(\d+)_', os.path.basename(f))) else np.nan
        for f in image_file_list
    ]

    # Safely assign the true frame numbers to each result table
    ref_frame1 = frame_numbers[frame1_idx] if frame1_idx < len(frame_numbers) else np.nan
    ref_frame2 = frame_numbers[frame2_idx] if frame2_idx < len(frame_numbers) else np.nan
    ref_frame3 = frame_numbers[frame3_idx] if frame3_idx < len(frame_numbers) else np.nan

    final_table1['ref_frame'] = ref_frame1
    final_table2['ref_frame'] = ref_frame2
    final_table3['ref_frame'] = ref_frame3


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
    cols = ['ref_frame'] + [c for c in all_particles.columns if c != 'ref_frame']
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
    
