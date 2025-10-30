# -*- coding: utf-8 -*-
"""
wellcounter_male_module.py

Dedicated module for detection and analysis of male-type traces
(thin, fast-moving particles forming 'eyelash-like' footprints)
in long-exposure images (LEIs).

Author: Claus-Peter Stelzer
Date: 2025-10-20
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
import networkx as nx
from skimage.morphology import thin
from typing import Optional
import wellcounter_motion_module as wmm
import wellcounter_imaging_module as wim
import os


_LAST_LEI_METADATA = {}


class IndexlessDataFrame(pd.DataFrame):
    """A DataFrame subclass that hides its index when printed or rendered."""

    _metadata = []

    @property
    def _constructor(self):
        return IndexlessDataFrame

    def __repr__(self):  # pragma: no cover - visual aid only
        return super().to_string(index=False)

    def _repr_html_(self):  # pragma: no cover - visual aid only
        return self.to_html(index=False)


def generate_long_exposure_image_custom(
    run_folder_path,
    analysis_duration: float = 0.5,
    microorganism_threshold: int = 12,
    min_microorganism_area: int = 105,
    ref_frame_no: int = 0,
    rec_direction: str = 'forward'
):
    """
    Generate a Long Exposure Image (LEI) using configurable parameters.

    This function temporarily modifies the configuration file on disk
    to use the specified analysis_duration (in seconds) and microorganism
    detection parameters, runs the standard
    `record_particle_positions_from_sequence()`, and restores the original
    configuration afterward.

    Parameters
    ----------
    run_folder_path : str
        Path to the run folder containing the image sequence (expects subfolder 'jpg').
    analysis_duration : float, optional
        Duration of frame accumulation in seconds. Default is 0.5 s.
    microorganism_threshold : int, optional
        Binary threshold for detecting particles. Default is 12.
    min_microorganism_area : int, optional
        Minimum area (in px²) for detected particles. Default is 105.
    ref_frame_no : int, optional
        Index of the reference frame used for motion analysis. Default is 0.
    rec_direction : {"forward", "reverse"}, optional
        Direction of accumulation relative to the reference frame. Default is
        "forward" (reference frame plus later frames).

    Returns
    -------
    result_df : pandas.DataFrame
        DataFrame containing particle positions from the analyzed frames.
    long_exposure_image : numpy.ndarray
        The generated long-exposure image.
    """

    

    # --- Locate and read config file ---
    config_path = "wellcounter_config.yml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    with open(config_path, "r") as f:
        original_config = yaml.safe_load(f)

    # --- Apply temporary parameters ---
    config['motion']['analysis_duration'] = float(analysis_duration)
    config['particle_detection']['microorganism_threshold'] = int(microorganism_threshold)
    config['particle_detection']['min_microorganism_area'] = int(min_microorganism_area)

    # --- Write modified config to disk ---
    with open(config_path, "w") as f:
        yaml.safe_dump(config, f)
    print("[generate_long_exposure_image_custom] Temporary config written to disk:")
    print(f"  motion.analysis_duration = {config['motion']['analysis_duration']}")
    print(f"  particle_detection.microorganism_threshold = {config['particle_detection']['microorganism_threshold']}")
    print(f"  particle_detection.min_microorganism_area = {config['particle_detection']['min_microorganism_area']}")

    # --- Run the standard particle recording function ---
    try:
        result_df, long_exposure_image = wmm.record_particle_positions_from_sequence(
            run_folder_path,
            ref_frame_no=ref_frame_no,
            rec_direction=rec_direction
        )
    finally:
        # --- Always restore the original config ---
        with open(config_path, "w") as f:
            yaml.safe_dump(original_config, f)
        print("[generate_long_exposure_image_custom] Original config restored.")

    # --- Save the long-exposure image if output enabled ---
    output_params = original_config.get('outputs', {})
    if long_exposure_image is None:
        print("[generate_long_exposure_image_custom] Failed to generate LEI.")
        return result_df, None

    if output_params.get('particle_detection', False):
        parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
        folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
        output_dir = os.path.join(parent_dir, f"{folder_name}_particle_analysis")
        os.makedirs(output_dir, exist_ok=True)

        lei_path = os.path.join(output_dir, "LEI_males.jpg")
        #cv2.imwrite(lei_path, long_exposure_image)

        # --- Log parameter values for reproducibility ---
        log_path = os.path.join(output_dir, "LEI_males_log.txt")
        with open(log_path, "w") as log_file:
            log_file.write("LEI generation parameters:\n")
            log_file.write(f"analysis_duration: {analysis_duration}\n")
            log_file.write(f"microorganism_threshold: {microorganism_threshold}\n")
            log_file.write(f"min_microorganism_area: {min_microorganism_area}\n")
            log_file.write(f"ref_frame_no: {ref_frame_no}\n")
            log_file.write(f"rec_direction: {rec_direction}\n")
            log_file.write(f"output_path: {lei_path}\n")
        print(f"[generate_long_exposure_image_custom] LEI saved: {lei_path}")
        print(f"[generate_long_exposure_image_custom] Parameters logged to: {log_path}")
    else:
        print("[generate_long_exposure_image_custom] particle_detection output disabled — LEI not saved.")

    global _LAST_LEI_METADATA
    _LAST_LEI_METADATA = {
        "run_folder_path": run_folder_path,
        "ref_frame_no": int(ref_frame_no),
        "rec_direction": rec_direction.lower(),
    }

    if isinstance(result_df, pd.DataFrame):
        result_df = IndexlessDataFrame(result_df.reset_index(drop=True))

    return result_df, long_exposure_image

def extract_major_ridge(mask, return_path: bool = False):
    """
    Extract a single, smooth centerline near the geometric middle of an irregular particle.
    Prefers high-distance (central) pixels instead of purely longest endpoints.

    Parameters
    ----------
    mask : np.ndarray of dtype uint8 or bool
        Binary particle mask (nonzero = foreground).
    return_path : bool, optional
        If True, also return the ordered list of (y, x) pixels along the geodesic centerline.

    Returns
    -------
    ridge_mask : np.ndarray (bool)
        Boolean array with True on centerline pixels.
    path_coords : list[tuple[int,int]]  (only if return_path=True)
        Ordered list of (y, x) coordinates from one endpoint to the other.
    """
    mask = mask.astype(np.uint8)
    if mask.sum() == 0:
        return (np.zeros_like(mask, bool), []) if return_path else np.zeros_like(mask, bool)

    # Distance transform (L2)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Initial ridge region: high-distance zone, thinned
    ridge = dist > 0.5 * dist[mask > 0].max()
    ridge = thin(ridge)

    ys, xs = np.nonzero(ridge)
    if len(ys) < 2:
        return (ridge, [(int(ys[0]), int(xs[0]))] if len(ys) == 1 else []) if return_path else ridge

    # Weighted graph: cheaper in central (high-distance) areas
    G = nx.Graph()
    for y, x in zip(ys, xs):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < ridge.shape[0] and 0 <= xx < ridge.shape[1] and ridge[yy, xx]:
                    w = 1.0 / (1e-3 + 0.5 * (dist[y, x] + dist[yy, xx]))
                    G.add_edge((y, x), (yy, xx), weight=w)

    # Endpoints that maximize weighted path length
    lengths = dict(nx.all_pairs_dijkstra_path_length(G, weight='weight'))
    max_d, start, end = 0, None, None
    for u, dists in lengths.items():
        for v, d in dists.items():
            if d > max_d:
                max_d, start, end = d, u, v
    if start is None or end is None:
        return (ridge, []) if return_path else ridge

    # Ordered path
    path = nx.shortest_path(G, start, end, weight='weight')

    # Rasterize
    clean = np.zeros_like(ridge, bool)
    for (y, x) in path:
        clean[y, x] = True

    return (clean, path) if return_path else clean

def _polyline_metrics_from_path(path_xy, dist_transform=None):
    """
    Compute length, chord, straightness, mean curvature, and width stats
    for an ordered list of (y, x) pixels.

    Parameters
    ----------
    path_xy : list[(int,int)]
        Ordered (y, x) polyline.
    dist_transform : np.ndarray or None
        Distance transform over the particle mask, to sample local half-widths.

    Returns
    -------
    dict
        {
          "centerline_length": float,
          "centerline_chord_length": float,
          "centerline_straightness": float or np.nan,
          "centerline_mean_curvature": float or np.nan,
          "centerline_mean_width": float or np.nan,
          "centerline_width_std": float or np.nan,
          "centerline_n_pixels": int
        }
    """
    import numpy as np

    n = len(path_xy)
    out = {
        "centerline_length": 0.0,
        "centerline_chord_length": 0.0,
        "centerline_straightness": np.nan,
        "centerline_mean_curvature": np.nan,
        "centerline_mean_width": np.nan,
        "centerline_width_std": np.nan,
        "centerline_n_pixels": n
    }
    if n < 2:
        return out

    coords = np.array(path_xy, dtype=float)  # (y, x)
    diffs = np.diff(coords, axis=0)
    step_lengths = np.sqrt((diffs ** 2).sum(axis=1))
    length = float(step_lengths.sum())
    out["centerline_length"] = length

    chord = float(np.linalg.norm(coords[-1] - coords[0]))
    out["centerline_chord_length"] = chord
    out["centerline_straightness"] = (chord / length) if length > 0 else np.nan

    # Curvature as mean absolute angle increment between successive segments (your convention)
    if n >= 3:
        # Note: np.arctan2(dy, dx) with dy = delta_y, dx = delta_x
        angles = np.arctan2(diffs[:, 0], diffs[:, 1])
        dtheta = np.abs(np.diff(angles))
        # wrap to [0, pi]
        dtheta[dtheta > np.pi] -= np.pi
        out["centerline_mean_curvature"] = float(np.mean(dtheta))

    # Width sampling: use DT at path pixels (twice radius)
    if dist_transform is not None:
        yy = coords[:, 0].astype(int)
        xx = coords[:, 1].astype(int)
        valid = (yy >= 0) & (yy < dist_transform.shape[0]) & (xx >= 0) & (xx < dist_transform.shape[1])
        if valid.any():
            wvals = dist_transform[yy[valid], xx[valid]] * 2.0
            out["centerline_mean_width"] = float(np.mean(wvals))
            out["centerline_width_std"] = float(np.std(wvals))

    return out



def analyze_long_exposure_particles_advanced(
    long_exposure_image,
    run_folder_path,
    collage_metric="centerline_mean_width",
    ref_frame_no: Optional[int] = None,
):
    """
    Advanced morphological analysis of binary long-exposure images (LEI),
    quantifying 'eyelash-like' traces and creating diagnostic plots.

    Adds a subfunction that builds a collage of fixed-size (250x250 px)
    cropped regions from the reference frame, centered on each LEI particle
    and overlaid with its skeleton.

    Parameters
    ----------
    long_exposure_image : np.ndarray
        Binary (0/255) LEI image of summed particle traces.
    run_folder_path : str
        Path to the run folder (used to locate output and input images).
    collage_metric : str, optional
        Column name in metrics DataFrame to sort the collage by.
    ref_frame_no : int, optional
        Index of the reference frame used when the LEI was generated. If omitted,
        the routine attempts to reuse the most recent value supplied to
        ``generate_long_exposure_image_custom`` for the same ``run_folder_path``.

    Returns
    -------
    pandas.DataFrame
        Metrics per particle.
    """

    if ref_frame_no is None:
        metadata = {}
        if _LAST_LEI_METADATA.get("run_folder_path") == run_folder_path:
            metadata = _LAST_LEI_METADATA
        resolved_ref_frame_no = int(metadata.get("ref_frame_no", 0))
    else:
        resolved_ref_frame_no = int(ref_frame_no)

    if resolved_ref_frame_no < 0:
        print(
            f"[analyze_long_exposure_particles_advanced] Warning: ref_frame_no {resolved_ref_frame_no} is negative. "
            "Clamping to 0 for diagnostics."
        )
        resolved_ref_frame_no = 0

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

    results = []  # store metrics 

    for idx, region in enumerate(props, start=1):
        if region.area < 10:
            continue

        mask = (labeled == region.label).astype(np.uint8)
        area = int(region.area)
       
        # --- Geodesic centerline extraction ---
        ridge_mask, ridge_path = extract_major_ridge(mask, return_path=True)
        region_ridge_pixels = np.argwhere(ridge_mask)
        ridge_length = int(len(region_ridge_pixels)) if region_ridge_pixels.size > 0 else 0


        dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)
        
        # --- Centerline metrics (parallel to skeleton metrics) ---
        centerline_metrics = _polyline_metrics_from_path(ridge_path, dist_transform=dist_transform)

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

        # Assemble record
        results.append({
            "particle_id": idx,
            "X": cx,
            "Y": cy,
            "area": area,

            # Classical
            "solidity": solidity,
            "circularity": circularity,

            # Ridge extent
            "ridge_length": ridge_length,

            # New: Centerline-based
            "centerline_length": centerline_metrics["centerline_length"],
            "centerline_chord_length": centerline_metrics["centerline_chord_length"],
            "centerline_straightness": centerline_metrics["centerline_straightness"],
            "centerline_mean_curvature": centerline_metrics["centerline_mean_curvature"],
            "centerline_mean_width": centerline_metrics["centerline_mean_width"],
            "centerline_width_std": centerline_metrics["centerline_width_std"],
            "centerline_n_pixels": centerline_metrics["centerline_n_pixels"],
        })
        

    df = pd.DataFrame(results)
    print(f"[analyze_long_exposure_particles_advanced] Analyzed {len(df)} traces.")

    # Save results
    if save_outputs:
        analyzed_path = os.path.join(output_dir, "LEI_males_analyzed.jpg")
        df_path = os.path.join(output_dir, "LEI_males_metrics.csv")
        #cv2.imwrite(analyzed_path, overlay)
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
    # --- Subfunction: Diagnostic Collage with geodesic centerlines --------
    # ----------------------------------------------------------------------
    def create_diagnostic_collage_centerline(df, metric="centerline_mean_width",
                                             crop_size=250, n_cols=6):
        """
        Create collage of 250x250 px crops centered on LEI particle centroids,
        extracted from the reference frame and overlaid with geodesic centerlines (green).
        Otherwise identical to create_diagnostic_collage_fixed().
        """
        import glob

        print("[collage_centerline] Starting centerline collage creation...")
        try:
            # Locate reference frame
            jpg_dir = os.path.join(run_folder_path, "jpg")
            image_files = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))
            if not image_files:
                print(f"[collage_centerline] No images found in {jpg_dir}. Cannot create collage.")
                return
            clamped_idx = max(0, min(resolved_ref_frame_no, len(image_files) - 1))
            if clamped_idx != resolved_ref_frame_no:
                print(
                    "[collage_centerline] Warning: requested ref_frame_no "
                    f"{resolved_ref_frame_no} outside available range. Using {clamped_idx} instead."
                )
            reference_frame_path = image_files[clamped_idx]
            print(
                f"[collage_centerline] Using reference frame index {clamped_idx}: {reference_frame_path}"
            )

            reference_frame = cv2.imread(reference_frame_path)
            if reference_frame is None:
                print(f"[collage_centerline] Failed to read {reference_frame_path}.")
                return

            # Sort and prepare layout
            df_sorted = df.sort_values(by=metric, ascending=True).reset_index(drop=True)
            n_particles = len(df_sorted)
            n_rows = int(np.ceil(n_particles / n_cols))
            half = crop_size // 2
            h, w = reference_frame.shape[:2]
            crops = []

            for i, row in df_sorted.iterrows():
                cx, cy = int(row["X"]), int(row["Y"])
                x1, x2 = max(0, cx - half), min(w, cx + half)
                y1, y2 = max(0, cy - half), min(h, cy + half)
                crop = reference_frame[y1:y2, x1:x2].copy()

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
        create_diagnostic_collage_centerline(df, metric="centerline_mean_width",
                                            crop_size=250, n_cols=6)

    return df
