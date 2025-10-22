# -*- coding: utf-8 -*-
"""
wellcounter_male_module.py

Dedicated module for detection and analysis of male-type traces
(thin, fast-moving particles forming 'eyelash-like' footprints)
in long-exposure images (LEIs).

Author: Claus-Peter Stelzer
Date: 2025-10-20
"""

import os
import cv2
import numpy as np
import pandas as pd
import skimage.morphology as morph
import skimage.measure as measure
from scipy.spatial.distance import pdist
from scipy.ndimage import distance_transform_edt
import matplotlib.pyplot as plt

import wellcounter_motion_module as wmm
import wellcounter_imaging_module as wim


# ======================================================================
# --- Basic LEI generation wrapper (calls motion module) ---------------
# ======================================================================

def generate_LEI_for_males(run_folder_path,
                           analysis_duration=0.2,
                           microorganism_threshold=12,
                           min_microorganism_area=105):
    """
    Wrapper for generating a long-exposure image (LEI) optimized for detecting males.
    It temporarily modifies sensitivity parameters and analysis duration.

    Returns:
        df_positions (DataFrame), long_exposure_image (ndarray)
    """
    print(f"[generate_LEI_for_males] Generating LEI (duration={analysis_duration}s)")
    df_positions, lei = wmm.generate_long_exposure_image_custom(
        run_folder_path,
        analysis_duration=analysis_duration,
        microorganism_threshold=microorganism_threshold,
        min_microorganism_area=min_microorganism_area
    )
    return df_positions, lei


# ======================================================================
# --- Shape analysis for male detection --------------------------------
# ======================================================================

def analyze_LEI_for_males(long_exposure_image, run_folder_path,
                          save_outputs=True, collage_metric="mean_width"):
    """
    Analyze a long-exposure image to extract morphological metrics of
    male-type particles.

    Generates per-particle metrics, topology overlay, and collages.

    Returns:
        DataFrame with metrics for all detected particles.
    """
    print("[analyze_LEI_for_males] Starting LEI analysis for males...")

    # Ensure binary image
    if long_exposure_image.ndim == 3:
        gray = cv2.cvtColor(long_exposure_image, cv2.COLOR_BGR2GRAY)
    else:
        gray = long_exposure_image.copy()

    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    labeled = measure.label(binary > 0)
    props = measure.regionprops(labeled)

    results = []
    overlay_topology = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for region in props:
        mask = (labeled == region.label)
        if np.sum(mask) < 10:
            continue

        skeleton = morph.skeletonize(mask)
        coords = np.argwhere(skeleton)
        n_pixels = len(coords)

        # Skip degenerate skeletons
        if n_pixels < 5:
            continue

        # --- Compute shape metrics ---
        skeleton_length = np.sum(morph.thin(mask))
        distmap = distance_transform_edt(mask)
        ys, xs = np.nonzero(skeleton)
        local_widths = distmap[ys, xs] * 2
        mean_width = np.mean(local_widths)
        width_std = np.std(local_widths)

        # --- Straightness and curvature ---
        endpoints = find_endpoints(skeleton)
        if len(endpoints) >= 2:
            chord = np.linalg.norm(np.array(endpoints[0]) - np.array(endpoints[-1]))
            straightness_ratio = chord / (len(coords) + 1e-5)
        else:
            straightness_ratio = np.nan

        curvature = compute_curvature_along_skeleton(coords)
        mean_curvature = np.mean(curvature)

        # --- Geodesic major ridge (centerline) ---
        ridge = extract_major_ridge(mask)
        ridge_length = np.count_nonzero(ridge)
        ys_r, xs_r = np.nonzero(ridge)
        for y, x in zip(ys_r, xs_r):
            overlay_topology[y, x] = (0, 255, 0)  # bright green

        results.append({
            "particle_id": region.label,
            "X": region.centroid[1],
            "Y": region.centroid[0],
            "skeleton_length": skeleton_length,
            "straightness_ratio": straightness_ratio,
            "mean_curvature": mean_curvature,
            "mean_width": mean_width,
            "width_std": width_std,
            "ridge_length": ridge_length,
            "area": region.area,
            "solidity": region.solidity,
            "circularity": (4 * np.pi * region.area) / (region.perimeter ** 2 + 1e-6)
        })

    df = pd.DataFrame(results)

    # --- Save outputs ---
    parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
    folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
    output_dir = os.path.join(parent_dir, f"{folder_name}_male_analysis")
    os.makedirs(output_dir, exist_ok=True)

    if save_outputs:
        cv2.imwrite(os.path.join(output_dir, "LEI_males_topology.jpg"), overlay_topology)
        df.to_csv(os.path.join(output_dir, "LEI_males_metrics.csv"), index=False)
        print(f"[analyze_LEI_for_males] Saved outputs to {output_dir}")

        # Optional: create collages
        if wim.read_config().get("outputs", {}).get("particle_detection", False):
            create_diagnostic_collage_centerline(df, run_folder_path,
                                                 metric=collage_metric,
                                                 output_dir=output_dir)

    return df


# ======================================================================
# --- Supporting geometry and ridge extraction functions ---------------
# ======================================================================

def find_endpoints(skeleton):
    """Return endpoints of a skeleton."""
    kernel = np.ones((3, 3), np.uint8)
    neighbor_count = cv2.filter2D(skeleton.astype(np.uint8), -1, kernel)
    endpoints = np.argwhere((skeleton == 1) & (neighbor_count == 2))
    return endpoints


def compute_curvature_along_skeleton(coords, k=5):
    """Estimate local curvature of skeleton coordinates."""
    if len(coords) < k * 2:
        return np.zeros(len(coords))
    curvature = []
    for i in range(k, len(coords) - k):
        p1, p2, p3 = coords[i - k], coords[i], coords[i + k]
        v1 = p2 - p1
        v2 = p3 - p2
        cosang = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-9)
        curvature.append(np.arccos(np.clip(cosang, -1, 1)))
    return np.array(curvature)


def extract_major_ridge(mask):
    """Extract the single most dominant ridge line (centerline) within a particle mask."""
    dist = distance_transform_edt(mask)
    ridge = morph.skeletonize(dist > 0)
    ridge_filtered = morph.remove_small_objects(ridge, 5)
    return ridge_filtered


# ======================================================================
# --- Diagnostic collage with centerlines ------------------------------
# ======================================================================

def create_diagnostic_collage_centerline(df, run_folder_path, metric="mean_width",
                                         crop_size=250, n_cols=6, output_dir=None):
    """
    Create a collage of 250x250 px crops centered on particle centroids
    overlaid with geodesic centerlines (green).
    """
    import glob
    jpg_dir = os.path.join(run_folder_path, "jpg")
    image_files = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))
    if not image_files:
        print(f"[collage_centerline] No frames found in {jpg_dir}.")
        return

    first_frame = cv2.imread(image_files[0])
    if first_frame is None:
        print(f"[collage_centerline] Failed to read first frame.")
        return

    h, w = first_frame.shape[:2]
    half = crop_size // 2
    df_sorted = df.sort_values(by=metric, ascending=True).reset_index(drop=True)
    n_particles = len(df_sorted)
    n_rows = int(np.ceil(n_particles / n_cols))
    crops = []

    for _, row in df_sorted.iterrows():
        cx, cy = int(row["X"]), int(row["Y"])
        x1, x2 = max(0, cx - half), min(w, cx + half)
        y1, y2 = max(0, cy - half), min(h, cy + half)
        crop = first_frame[y1:y2, x1:x2].copy()

        mask_crop = np.zeros((y2 - y1, x2 - x1), dtype=np.uint8)
        cv2.circle(mask_crop, (crop.shape[1] // 2, crop.shape[0] // 2), 80, 1, -1)
        ridge = extract_major_ridge(mask_crop)
        ys, xs = np.nonzero(ridge)
        for y, x in zip(ys, xs):
            crop[y, x] = (0, 255, 0)

        crop = cv2.resize(crop, (crop_size, crop_size))
        cv2.putText(crop, f"{metric}={row[metric]:.2f}",
                    (5, crop_size - 10), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
        crops.append(crop)

    rows = []
    for i in range(n_rows):
        row_imgs = crops[i * n_cols:(i + 1) * n_cols]
        if len(row_imgs) < n_cols:
            pad_img = np.zeros_like(row_imgs[0])
            row_imgs += [pad_img] * (n_cols - len(row_imgs))
        rows.append(np.hstack(row_imgs))
    collage = np.vstack(rows)

    if output_dir:
        collage_path = os.path.join(output_dir, f"particle_collage_centerline_by_{metric}.jpg")
    else:
        collage_path = os.path.join(run_folder_path, f"particle_collage_centerline_by_{metric}.jpg")

    cv2.imwrite(collage_path, collage)
    print(f"[collage_centerline] Saved: {collage_path}")
