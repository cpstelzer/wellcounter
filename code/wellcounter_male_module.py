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
import ast
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


_LAST_LEI_METADATA = {}


def _infer_reference_frame_value(run_folder_path, ref_frame_index):
    """Return the absolute frame number for a reference frame index."""

    try:
        image_folder = os.path.join(run_folder_path, "jpg")
        image_files = wim.get_image_file_list(image_folder)
        if 0 <= int(ref_frame_index) < len(image_files):
            filename = os.path.basename(image_files[int(ref_frame_index)])
            match = re.search(r"_f(\d+)_", filename)
            if match:
                return int(match.group(1))
    except Exception:
        pass

    try:
        return int(ref_frame_index)
    except Exception:
        return 0


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

    ref_frame_value = None
    frame_sequence = []

    if isinstance(result_df, pd.DataFrame) and not result_df.empty:
        if "frame" in result_df.columns:
            try:
                frame_sequence = sorted({int(v) for v in pd.unique(result_df["frame"]) if pd.notna(v)})
            except Exception:
                frame_sequence = []

    if ref_frame_value is None:
        ref_frame_value = _infer_reference_frame_value(run_folder_path, ref_frame_no)

    _LAST_LEI_METADATA = {
        "run_folder_path": run_folder_path,
        "ref_frame_no": int(ref_frame_no),
        "rec_direction": rec_direction.lower(),
        "ref_frame_value": int(ref_frame_value),
        "ref_frame_index": int(ref_frame_no),
        "frame_sequence": frame_sequence,
    }

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

    # Endpoints that maximize weighted path length (graph diameter)
    best_path = []
    best_length = -1.0
    for component in nx.connected_components(G):
        if not component:
            continue
        seed = next(iter(component))
        # First pass: farthest node from arbitrary seed
        first_lengths = nx.single_source_dijkstra_path_length(G, seed, weight="weight")
        if not first_lengths:
            continue
        farthest_seed = max(first_lengths, key=first_lengths.get)
        # Second pass: farthest node from the previously found endpoint
        second_lengths = nx.single_source_dijkstra_path_length(G, farthest_seed, weight="weight")
        if not second_lengths:
            continue
        farthest_node = max(second_lengths, key=second_lengths.get)
        length = second_lengths[farthest_node]
        if length > best_length:
            best_length = length
            best_path = nx.dijkstra_path(G, farthest_seed, farthest_node, weight="weight")

    if not best_path:
        return (ridge, []) if return_path else ridge

    path = best_path

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
            metadata = _LAST_LEI_METADATA.copy()
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
    particle_diagnostics = []

    for idx, region in enumerate(props, start=1):
        if region.area < 10:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        mask = (labeled[min_row:max_row, min_col:max_col] == region.label)
        mask_uint8 = mask.astype(np.uint8)
        area = int(region.area)

        # --- Geodesic centerline extraction ---
        ridge_mask, ridge_path = extract_major_ridge(mask_uint8, return_path=True)
        ridge_mask = ridge_mask.astype(bool, copy=False)
        ridge_length = int(np.count_nonzero(ridge_mask))

        # Convert centerline coordinates into convenient representations
        ridge_path_local = []
        ridge_path_global = []
        if ridge_path:
            for (ry, rx) in ridge_path:
                ry_int = int(ry)
                rx_int = int(rx)
                ridge_path_local.append((ry_int, rx_int))
                ridge_path_global.append((ry_int + int(min_row), rx_int + int(min_col)))

        if ridge_path_global:
            start_global = ridge_path_global[0]
            end_global = ridge_path_global[-1]
        else:
            centroid_row, centroid_col = region.centroid
            start_global = (int(centroid_row), int(centroid_col))
            end_global = start_global


        dist_transform = cv2.distanceTransform(mask_uint8, cv2.DIST_L2, 5)
        
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
            "bbox_min_row": int(min_row),
            "bbox_min_col": int(min_col),
            "bbox_max_row": int(max_row),
            "bbox_max_col": int(max_col),

            # Classical
            "solidity": solidity,
            "circularity": circularity,

            # Ridge extent
            "ridge_length": ridge_length,

            # Path representations
            "centerline_path_local": ridge_path_local,
            "centerline_path_global": ridge_path_global,
            "centerline_start_y": int(start_global[0]),
            "centerline_start_x": int(start_global[1]),
            "centerline_end_y": int(end_global[0]),
            "centerline_end_x": int(end_global[1]),

            # New: Centerline-based
            "centerline_length": centerline_metrics["centerline_length"],
            "centerline_chord_length": centerline_metrics["centerline_chord_length"],
            "centerline_straightness": centerline_metrics["centerline_straightness"],
            "centerline_mean_curvature": centerline_metrics["centerline_mean_curvature"],
            "centerline_mean_width": centerline_metrics["centerline_mean_width"],
            "centerline_width_std": centerline_metrics["centerline_width_std"],
            "centerline_n_pixels": centerline_metrics["centerline_n_pixels"],
        })

        if save_outputs:
            overlay_slice = overlay[min_row:max_row, min_col:max_col]
            overlay_slice[ridge_mask] = (0, 0, 255)
            particle_diagnostics.append({
                "particle_id": idx,
                "bbox": (min_row, min_col, max_row, max_col),
                "ridge_mask": ridge_mask,
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

        for diag in particle_diagnostics:
            min_row, min_col, max_row, max_col = diag["bbox"]
            ridge = diag["ridge_mask"]
            region_slice = lei_centerline_overlay[min_row:max_row, min_col:max_col]
            region_slice[ridge] = (0, 0, 255)

        out_path_centerline = os.path.join(output_dir, "LEI_males_centerline.jpg")
        cv2.imwrite(out_path_centerline, lei_centerline_overlay)
        print(f"[analyze_long_exposure_particles_advanced] Geodesic centerline overlay saved: {out_path_centerline}")


    # ----------------------------------------------------------------------
    # --- Subfunction: Diagnostic Collage with geodesic centerlines --------
    # ----------------------------------------------------------------------
    def create_diagnostic_collage_centerline(df, metric="centerline_mean_width",
                                             crop_size=250, n_cols=6,
                                             diagnostics=None):
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

            diag_lookup = {d["particle_id"]: d for d in diagnostics or []}

            for i, row in df_sorted.iterrows():
                cx, cy = int(row["X"]), int(row["Y"])
                x1, x2 = max(0, cx - half), min(w, cx + half)
                y1, y2 = max(0, cy - half), min(h, cy + half)
                crop = reference_frame[y1:y2, x1:x2].copy()

                # --- Overlay geodesic centerline (in bright green) ---
                diag = diag_lookup.get(int(row["particle_id"]))
                if diag:
                    min_row, min_col, max_row, max_col = diag["bbox"]
                    ridge = diag["ridge_mask"]
                    overlap_y1 = max(y1, min_row)
                    overlap_y2 = min(y2, max_row)
                    overlap_x1 = max(x1, min_col)
                    overlap_x2 = min(x2, max_col)
                    if overlap_y1 < overlap_y2 and overlap_x1 < overlap_x2:
                        ridge_sub = ridge[overlap_y1 - min_row:overlap_y2 - min_row,
                                          overlap_x1 - min_col:overlap_x2 - min_col]
                        crop_sub = crop[overlap_y1 - y1:overlap_y2 - y1,
                                        overlap_x1 - x1:overlap_x2 - x1]
                        crop_sub[ridge_sub] = (0, 255, 0)

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
                                            crop_size=250, n_cols=6,
                                            diagnostics=particle_diagnostics)

    return df


def _coerce_centerline_path(path_value):
    """Normalize serialized or array-like path representations into a list of (y, x)."""

    if isinstance(path_value, list):
        cleaned = []
        for item in path_value:
            if isinstance(item, (list, tuple)) and len(item) == 2:
                try:
                    cleaned.append((int(item[0]), int(item[1])))
                except Exception:
                    continue
        return cleaned
    if isinstance(path_value, tuple):
        if len(path_value) == 2:
            try:
                return [(int(path_value[0]), int(path_value[1]))]
            except Exception:
                return []
        return [
            (int(item[0]), int(item[1]))
            for item in path_value
            if isinstance(item, (list, tuple)) and len(item) == 2
        ]
    if isinstance(path_value, str):
        try:
            parsed = ast.literal_eval(path_value)
        except (ValueError, SyntaxError):
            return []
        return _coerce_centerline_path(parsed)
    return []


def _build_reference_window_trajectories(result_df, ref_frame_value, max_search_distance, max_order_gap=1):
    """Link detections across frames into short trajectories surrounding the reference frame."""

    if result_df is None or result_df.empty:
        return []
    required_cols = {"frame", "X", "Y"}
    if not required_cols.issubset(result_df.columns):
        missing = ", ".join(sorted(required_cols - set(result_df.columns)))
        raise ValueError(f"result_df is missing required columns: {missing}")

    frame_values = pd.unique(result_df["frame"])
    frame_order = sorted(frame_values)
    frame_to_order = {frame: idx for idx, frame in enumerate(frame_order)}
    grouped = {frame: result_df[result_df["frame"] == frame] for frame in frame_order}

    active_tracks = []
    completed_tracks = []
    next_id = 1

    for frame in frame_order:
        order_idx = frame_to_order[frame]
        frame_rows = grouped[frame]

        for row_index, row in frame_rows.iterrows():
            row_dict = row.to_dict()
            row_dict["_row_index"] = row_index
            row_dict["_order_index"] = order_idx

            best_track = None
            best_distance = float(max_search_distance)

            for track in active_tracks:
                order_gap = order_idx - track["last_order_index"]
                if order_gap <= 0 or order_gap > max_order_gap:
                    continue

                last_point = track["points"][-1]
                try:
                    dist = math.hypot(
                        float(row_dict["X"]) - float(last_point["X"]),
                        float(row_dict["Y"]) - float(last_point["Y"]),
                    )
                except Exception:
                    dist = float("inf")

                if dist < best_distance:
                    best_distance = dist
                    best_track = track

            if best_track is None:
                new_track = {
                    "id": next_id,
                    "points": [row_dict],
                    "last_order_index": order_idx,
                }
                active_tracks.append(new_track)
                next_id += 1
            else:
                best_track["points"].append(row_dict)
                best_track["last_order_index"] = order_idx

        still_active = []
        for track in active_tracks:
            if order_idx - track["last_order_index"] <= max_order_gap:
                still_active.append(track)
            else:
                completed_tracks.append(track)
        active_tracks = still_active

    completed_tracks.extend(active_tracks)

    for track in completed_tracks:
        track["reference_index"] = None
        for idx, point in enumerate(track["points"]):
            try:
                if int(point.get("frame")) == int(ref_frame_value):
                    track["reference_index"] = idx
                    break
            except Exception:
                continue

    return completed_tracks


def _score_trajectory_against_centerline(track, centerline_global_path, direction, reference_index):
    """Compare a tracked particle trajectory against a LEI centerline path."""

    if not centerline_global_path or track is None:
        return None
    if reference_index is None or reference_index < 0 or reference_index >= len(track["points"]):
        return None

    path_coords = np.array(centerline_global_path, dtype=float)
    if path_coords.size == 0:
        return None

    track_coords = np.array([[float(p["Y"]), float(p["X"])] for p in track["points"]], dtype=float)
    if track_coords.size == 0:
        return None

    track_start = track_coords[0]
    track_end = track_coords[-1]
    path_start = path_coords[0]
    path_end = path_coords[-1]

    endpoint_forward = float(np.linalg.norm(track_start - path_start) + np.linalg.norm(track_end - path_end))
    endpoint_reverse = float(np.linalg.norm(track_start - path_end) + np.linalg.norm(track_end - path_start))

    if endpoint_reverse < endpoint_forward:
        path_coords = path_coords[::-1].copy()
        endpoint_penalty = endpoint_reverse
        flipped = True
    else:
        endpoint_penalty = endpoint_forward
        flipped = False

    dist_matrix = distance_matrix(track_coords, path_coords)
    nearest_idx = np.argmin(dist_matrix, axis=1)
    nearest_distances = dist_matrix[np.arange(len(track_coords)), nearest_idx]
    mean_distance = float(np.mean(nearest_distances))
    max_distance = float(np.max(nearest_distances))

    idx_diff = np.diff(nearest_idx)
    backward_steps = idx_diff[idx_diff < 0]
    monotonic_violation_magnitude = float(np.abs(backward_steps).sum())
    monotonic_violation_count = int((idx_diff < 0).sum())

    path_segments = np.diff(path_coords, axis=0)
    path_total_length = float(np.linalg.norm(path_segments, axis=1).sum()) if len(path_coords) >= 2 else 0.0
    track_segments = np.diff(track_coords, axis=0)
    track_total_length = float(np.linalg.norm(track_segments, axis=1).sum()) if len(track_coords) >= 2 else 0.0

    ref_path_idx = int(nearest_idx[reference_index])
    path_steps = max(len(path_coords) - 1, 1)
    track_steps = max(len(track_coords) - 1, 1)
    path_progress = ref_path_idx / path_steps
    track_progress = reference_index / track_steps

    ref_alignment = abs(path_progress - track_progress)
    ref_distance = float(nearest_distances[reference_index])

    frames_before = reference_index
    frames_after = len(track_coords) - reference_index - 1
    path_before = ref_path_idx
    path_after = (len(path_coords) - 1) - ref_path_idx

    denominator_track = max(frames_before + frames_after, 1)
    denominator_path = max(path_before + path_after, 1)
    before_fraction = frames_before / denominator_track
    path_before_fraction = path_before / denominator_path
    before_after_fraction_diff = abs(before_fraction - path_before_fraction)

    length_ratio_difference = abs(track_total_length - path_total_length) / max(path_total_length, 1.0)

    score = (
        endpoint_penalty * 0.1
        + mean_distance
        + max_distance * 0.05
        + monotonic_violation_magnitude * 1.5
        + ref_alignment * 5.0
        + ref_distance * 0.5
        + length_ratio_difference * 2.0
        + before_after_fraction_diff * 3.0
    )

    return {
        "score": float(score),
        "endpoint_penalty": float(endpoint_penalty),
        "mean_distance": mean_distance,
        "max_distance": max_distance,
        "ref_alignment": float(ref_alignment),
        "ref_distance": ref_distance,
        "length_ratio_difference": float(length_ratio_difference),
        "before_after_fraction_diff": float(before_after_fraction_diff),
        "monotonic_violation_count": monotonic_violation_count,
        "monotonic_violation_magnitude": float(monotonic_violation_magnitude),
        "path_orientation_flipped": bool(flipped),
        "path_progress_at_reference": float(path_progress),
        "track_progress_at_reference": float(track_progress),
        "track_total_length": float(track_total_length),
        "path_total_length": float(path_total_length),
        "nearest_index_sequence": nearest_idx.tolist(),
    }


def match_lei_traces_to_reference_particles(
    lei_metrics_df,
    result_df,
    run_folder_path,
    ref_frame_no: Optional[int] = None,
    rec_direction: Optional[str] = None,
    max_endpoint_distance: Optional[float] = 40.0,
    max_mean_distance: Optional[float] = 20.0,
    max_frame_gap: int = 1,
):
    """Match LEI traces to reference-frame particles and merge their metrics."""

    if lei_metrics_df is None or len(lei_metrics_df) == 0:
        return pd.DataFrame(), pd.DataFrame()
    if result_df is None or len(result_df) == 0:
        return pd.DataFrame(), pd.DataFrame()

    if "centerline_path_global" not in lei_metrics_df.columns:
        raise ValueError(
            "LEI metrics must include 'centerline_path_global'. Run "
            "analyze_long_exposure_particles_advanced before matching."
        )

    metadata = {}
    if _LAST_LEI_METADATA.get("run_folder_path") == run_folder_path:
        metadata = _LAST_LEI_METADATA.copy()

    resolved_direction = rec_direction or metadata.get("rec_direction") or "forward"
    resolved_direction = resolved_direction.lower()
    if resolved_direction not in {"forward", "reverse"}:
        raise ValueError("rec_direction must be 'forward' or 'reverse'")

    if ref_frame_no is None:
        ref_frame_no = int(metadata.get("ref_frame_no", 0))
    else:
        ref_frame_no = int(ref_frame_no)

    frame_sequence = metadata.get("frame_sequence")
    if not frame_sequence:
        try:
            frame_sequence = sorted({
                int(v)
                for v in pd.unique(result_df.get("frame", pd.Series(dtype=int)))
                if pd.notna(v)
            })
        except Exception:
            frame_sequence = []

    ref_frame_value = metadata.get("ref_frame_value")
    if ref_frame_value is None and frame_sequence:
        try:
            ref_frame_value = int(frame_sequence[0])
        except Exception:
            ref_frame_value = None

    if ref_frame_value is None:
        ref_frame_value = _infer_reference_frame_value(run_folder_path, ref_frame_no)

    if ref_frame_value is None:
        raise ValueError("Unable to determine the reference frame value from the provided data.")

    config = wim.read_config()
    motion_params = config.get("motion", {})
    max_search_distance = float(motion_params.get("max_search_distance", 50.0))

    trajectories = _build_reference_window_trajectories(
        result_df,
        ref_frame_value,
        max_search_distance=max_search_distance,
        max_order_gap=max(1, int(max_frame_gap)),
    )

    candidate_tracks = [t for t in trajectories if t.get("reference_index") is not None]
    if not candidate_tracks:
        raise ValueError("No trajectories overlap the reference frame; cannot perform matching.")

    track_lookup = {track["id"]: track for track in candidate_tracks}

    lei_metrics_by_id = None
    if "particle_id" in lei_metrics_df.columns:
        lei_metrics_by_id = lei_metrics_df.set_index("particle_id", drop=False)
    else:
        raise ValueError("lei_metrics_df must contain a 'particle_id' column.")

    candidate_matches = []

    for _, lei_row in lei_metrics_df.iterrows():
        particle_id = int(lei_row.get("particle_id", -1))
        if particle_id < 0:
            continue

        path_value = _coerce_centerline_path(lei_row.get("centerline_path_global", []))
        if not path_value:
            continue

        for track in candidate_tracks:
            score_info = _score_trajectory_against_centerline(
                track,
                path_value,
                resolved_direction,
                track.get("reference_index"),
            )
            if score_info is None:
                continue

            if max_endpoint_distance is not None and score_info["endpoint_penalty"] > max_endpoint_distance:
                continue
            if max_mean_distance is not None and score_info["mean_distance"] > max_mean_distance:
                continue

            ref_point = track["points"][track["reference_index"]]
            candidate_matches.append({
                "lei_particle_id": particle_id,
                "track_id": track["id"],
                "ref_row_index": ref_point.get("_row_index"),
                "trajectory_length": len(track["points"]),
                "centerline_length": float(lei_row.get("centerline_length", np.nan)),
                **score_info,
            })

    if not candidate_matches:
        raise ValueError("No candidate matches satisfy the geometric constraints.")

    candidate_matches.sort(key=lambda x: (x["score"], x["mean_distance"]))

    assigned_lei = set()
    assigned_tracks = set()
    final_matches = []

    for candidate in candidate_matches:
        lei_id = candidate["lei_particle_id"]
        track_id = candidate["track_id"]
        if lei_id in assigned_lei or track_id in assigned_tracks:
            continue
        assigned_lei.add(lei_id)
        assigned_tracks.add(track_id)
        final_matches.append(candidate)

    ref_mask = result_df["frame"] == ref_frame_value
    ref_frame_particles = result_df.loc[ref_mask]

    merged_records = []

    for match in final_matches:
        lei_id = match["lei_particle_id"]
        track_id = match["track_id"]
        track = track_lookup[track_id]
        ref_index = match.get("ref_row_index")
        if ref_index is None or ref_index not in result_df.index:
            continue

        lei_row = lei_metrics_by_id.loc[lei_id]
        ref_row = result_df.loc[ref_index]

        combined = {
            "match_score": match["score"],
            "match_mean_distance": match["mean_distance"],
            "match_endpoint_penalty": match["endpoint_penalty"],
            "match_ref_alignment": match["ref_alignment"],
            "match_length_ratio_difference": match["length_ratio_difference"],
            "match_before_after_fraction_diff": match["before_after_fraction_diff"],
            "match_monotonic_violation_count": match["monotonic_violation_count"],
            "match_monotonic_violation_magnitude": match["monotonic_violation_magnitude"],
            "match_path_orientation_flipped": match["path_orientation_flipped"],
            "match_track_total_length": match["track_total_length"],
            "match_centerline_total_length": match["path_total_length"],
            "lei_particle_id": lei_id,
            "ref_row_index": ref_index,
            "matched_track_id": track_id,
            "reference_frame_value": ref_frame_value,
        }

        for col in lei_metrics_df.columns:
            combined[f"LEI_{col}"] = lei_row[col]
        for col in result_df.columns:
            combined[f"REF_{col}"] = ref_row[col]

        merged_records.append(combined)

    merged_df = pd.DataFrame(merged_records)

    matched_ref_indices = {match["ref_row_index"] for match in final_matches if match.get("ref_row_index") is not None}
    matched_lei_ids = {match["lei_particle_id"] for match in final_matches}

    association_records = []
    for match in final_matches:
        association_records.append({
            "status": "matched",
            "lei_particle_id": match["lei_particle_id"],
            "ref_row_index": match["ref_row_index"],
            "track_id": match["track_id"],
            "match_score": match["score"],
            "match_mean_distance": match["mean_distance"],
            "match_endpoint_penalty": match["endpoint_penalty"],
            "match_ref_alignment": match["ref_alignment"],
            "match_length_ratio_difference": match["length_ratio_difference"],
            "match_before_after_fraction_diff": match["before_after_fraction_diff"],
            "match_monotonic_violation_count": match["monotonic_violation_count"],
            "match_monotonic_violation_magnitude": match["monotonic_violation_magnitude"],
            "match_path_orientation_flipped": match["path_orientation_flipped"],
            "match_track_total_length": match["track_total_length"],
            "match_centerline_total_length": match["path_total_length"],
            "reference_frame_value": ref_frame_value,
        })

    for ref_index in ref_frame_particles.index:
        if ref_index in matched_ref_indices:
            continue
        association_records.append({
            "status": "unmatched_reference",
            "lei_particle_id": np.nan,
            "ref_row_index": ref_index,
            "track_id": np.nan,
            "match_score": np.nan,
            "match_mean_distance": np.nan,
            "match_endpoint_penalty": np.nan,
            "match_ref_alignment": np.nan,
            "match_length_ratio_difference": np.nan,
            "match_before_after_fraction_diff": np.nan,
            "match_monotonic_violation_count": np.nan,
            "match_monotonic_violation_magnitude": np.nan,
            "match_path_orientation_flipped": np.nan,
            "match_track_total_length": np.nan,
            "match_centerline_total_length": np.nan,
            "reference_frame_value": ref_frame_value,
        })

    for lei_id in lei_metrics_by_id.index:
        if lei_id in matched_lei_ids:
            continue
        association_records.append({
            "status": "unmatched_lei",
            "lei_particle_id": lei_id,
            "ref_row_index": np.nan,
            "track_id": np.nan,
            "match_score": np.nan,
            "match_mean_distance": np.nan,
            "match_endpoint_penalty": np.nan,
            "match_ref_alignment": np.nan,
            "match_length_ratio_difference": np.nan,
            "match_before_after_fraction_diff": np.nan,
            "match_monotonic_violation_count": np.nan,
            "match_monotonic_violation_magnitude": np.nan,
            "match_path_orientation_flipped": np.nan,
            "match_track_total_length": np.nan,
            "match_centerline_total_length": np.nan,
            "reference_frame_value": ref_frame_value,
        })

    association_df = pd.DataFrame(association_records)

    return merged_df, association_df
