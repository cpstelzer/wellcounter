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
from scipy.optimize import linear_sum_assignment
import glob
from skimage.morphology import skeletonize
from skimage.measure import label, regionprops
from math import pi
from skimage.morphology import thin
from typing import Optional
import heapq
try:
    from joblib import Parallel, delayed
except ImportError:  # pragma: no cover - optional dependency in legacy environments
    Parallel = None  # type: ignore[assignment]
    delayed = None  # type: ignore[assignment]
import wellcounter_motion_module as wmm
import wellcounter_imaging_module as wim
import copy


_LAST_LEI_METADATA = {}


def generate_long_exposure_image_custom(
    run_folder_path,
    ref_frame_no: int = 0,
    rec_direction: str = 'forward'
):
    """
    Generate a Long Exposure Image (LEI) using configuration-managed parameters.

    The function temporarily modifies the configuration file on disk to use
    the ``male_analysis`` values (duration in seconds plus particle detection
    settings), runs the standard ``record_particle_positions_from_sequence``
    routine, and restores the original configuration afterward.

    Parameters
    ----------
    run_folder_path : str
        Path to the run folder containing the image sequence (expects subfolder 'jpg').
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
    original_config = copy.deepcopy(config)

    male_params = config.get('male_analysis', {})
    analysis_duration = float(male_params.get('duration', 0.5))
    microorganism_threshold = int(male_params.get('threshold', 12))
    min_microorganism_area = int(male_params.get('min_area', 105))

    # --- Apply temporary parameters ---
    config['motion']['analysis_duration'] = analysis_duration
    config['particle_detection']['microorganism_threshold'] = microorganism_threshold
    config['particle_detection']['min_microorganism_area'] = min_microorganism_area

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

    frame_suffix = f"_frame{int(ref_frame_no)}"

    if output_params.get('particle_detection', False):
        parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
        folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
        output_dir = os.path.join(parent_dir, f"{folder_name}_particle_analysis")
        os.makedirs(output_dir, exist_ok=True)

        lei_path = os.path.join(output_dir, f"LEI_males{frame_suffix}.jpg")
        #cv2.imwrite(lei_path, long_exposure_image)

        # --- Log parameter values for reproducibility ---
        log_path = os.path.join(output_dir, f"LEI_males_log{frame_suffix}.txt")
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

    return result_df, long_exposure_image

def extract_major_ridge(mask, return_path: bool = False, return_dist: bool = False):
    """
    Extract a single, smooth centerline near the geometric middle of an irregular particle.
    Prefers high-distance (central) pixels instead of purely longest endpoints.

    Parameters
    ----------
    mask : np.ndarray of dtype uint8 or bool
        Binary particle mask (nonzero = foreground).
    return_path : bool, optional
        If True, also return the ordered list of (y, x) pixels along the geodesic centerline.
    return_dist : bool, optional
        If True, return the computed distance transform for reuse by the caller.

    Returns
    -------
    ridge_mask : np.ndarray (bool)
        Boolean array with True on centerline pixels.
    path_coords : list[tuple[int,int]]  (only if return_path=True)
        Ordered list of (y, x) coordinates from one endpoint to the other.
    dist_transform : np.ndarray (float32)  (only if return_dist=True)
        Euclidean distance transform of ``mask`` for reuse downstream.
    """
    mask = mask.astype(np.uint8)
    empty_bool = np.zeros_like(mask, bool)
    empty_dist = np.zeros_like(mask, dtype=np.float32)
    if mask.sum() == 0:
        outputs = [empty_bool]
        if return_path:
            outputs.append([])
        if return_dist:
            outputs.append(empty_dist)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    # Distance transform (L2)
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Initial ridge region: high-distance zone, thinned
    ridge = dist > 0.5 * dist[mask > 0].max()
    ridge = thin(ridge)

    ys, xs = np.nonzero(ridge)
    node_count = len(ys)
    if node_count == 0:
        outputs = [ridge]
        if return_path:
            outputs.append([])
        if return_dist:
            outputs.append(dist)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    coords = list(zip(ys.tolist(), xs.tolist()))
    coord_to_index = {coord: idx for idx, coord in enumerate(coords)}

    neighbors = [[] for _ in range(node_count)]
    weights = [[] for _ in range(node_count)]

    for idx, (y, x) in enumerate(coords):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                yy, xx = y + dy, x + dx
                if 0 <= yy < ridge.shape[0] and 0 <= xx < ridge.shape[1] and ridge[yy, xx]:
                    neighbor_idx = coord_to_index.get((yy, xx))
                    if neighbor_idx is None or neighbor_idx <= idx:
                        continue
                    w = 1.0 / (1e-3 + 0.5 * (dist[y, x] + dist[yy, xx]))
                    neighbors[idx].append(neighbor_idx)
                    weights[idx].append(w)
                    neighbors[neighbor_idx].append(idx)
                    weights[neighbor_idx].append(w)

    # Discover connected components using adjacency lists
    visited = np.zeros(node_count, dtype=bool)
    components = []
    for node in range(node_count):
        if visited[node]:
            continue
        stack = [node]
        visited[node] = True
        component = []
        while stack:
            current = stack.pop()
            component.append(current)
            for neighbor_idx in neighbors[current]:
                if not visited[neighbor_idx]:
                    visited[neighbor_idx] = True
                    stack.append(neighbor_idx)
        components.append(component)

    def _dijkstra(start_idx: int):
        dist_arr = np.full(node_count, np.inf, dtype=np.float64)
        parent = np.full(node_count, -1, dtype=np.int32)
        dist_arr[start_idx] = 0.0
        heap = [(0.0, start_idx)]
        while heap:
            current_dist, current_idx = heapq.heappop(heap)
            if current_dist > dist_arr[current_idx]:
                continue
            for nb, w in zip(neighbors[current_idx], weights[current_idx]):
                nd = current_dist + w
                if nd < dist_arr[nb]:
                    dist_arr[nb] = nd
                    parent[nb] = current_idx
                    heapq.heappush(heap, (nd, nb))
        return dist_arr, parent

    best_path_indices = []
    best_length = -1.0
    for component in components:
        if not component:
            continue
        seed = component[0]
        first_dists, _ = _dijkstra(seed)
        component_dists = first_dists[component]
        if not np.isfinite(component_dists).any():
            continue
        farthest_seed = int(component[np.argmax(component_dists)])
        second_dists, parents = _dijkstra(farthest_seed)
        component_second = second_dists[component]
        if not np.isfinite(component_second).any():
            continue
        farthest_node = int(component[np.argmax(component_second)])
        length = float(second_dists[farthest_node])
        if length < 0:
            continue
        if length > best_length:
            best_length = length
            path_indices = []
            current = farthest_node
            while current != -1:
                path_indices.append(current)
                if current == farthest_seed:
                    break
                current = parents[current]
            if not path_indices or path_indices[-1] != farthest_seed:
                # component disconnected due to missing edges; skip
                continue
            path_indices.reverse()
            best_path_indices = path_indices

    if not best_path_indices:
        outputs = [ridge]
        if return_path:
            outputs.append([])
        if return_dist:
            outputs.append(dist)
        return outputs[0] if len(outputs) == 1 else tuple(outputs)

    path = [(int(coords[idx][0]), int(coords[idx][1])) for idx in best_path_indices]

    clean = np.zeros_like(ridge, bool)
    for (y, x) in path:
        clean[y, x] = True

    outputs = [clean]
    if return_path:
        outputs.append(path)
    if return_dist:
        outputs.append(dist)
    return outputs[0] if len(outputs) == 1 else tuple(outputs)

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


def _compute_centerline_for_region(idx, mask_uint8, min_row, min_col, max_row, max_col):
    """Worker helper to extract ridge and metrics for a single particle mask."""
    ridge_mask, ridge_path, dist_transform = extract_major_ridge(
        mask_uint8, return_path=True, return_dist=True
    )
    ridge_mask = ridge_mask.astype(bool, copy=False)
    ridge_length = int(np.count_nonzero(ridge_mask))

    if ridge_path:
        start_local = ridge_path[0]
        end_local = ridge_path[-1]
        start_global = (
            int(start_local[1] + min_col),
            int(start_local[0] + min_row),
        )
        end_global = (
            int(end_local[1] + min_col),
            int(end_local[0] + min_row),
        )
    else:
        start_global = None
        end_global = None

    centerline_metrics = _polyline_metrics_from_path(ridge_path, dist_transform=dist_transform)

    return {
        "idx": idx,
        "ridge_mask": ridge_mask,
        "ridge_path_local": [(int(y), int(x)) for (y, x) in ridge_path],
        "ridge_length": ridge_length,
        "centerline_metrics": centerline_metrics,
        "start_global": start_global,
        "end_global": end_global,
        "bbox": (int(min_row), int(min_col), int(max_row), int(max_col)),
    }


def _path_local_to_global_xy(path_local, min_row, min_col):
    """Convert a local (y, x) path to global (x, y) coordinates."""
    if not path_local:
        return np.empty((0, 2), dtype=float)
    arr = np.asarray(path_local, dtype=float)
    arr[:, 0] += float(min_row)
    arr[:, 1] += float(min_col)
    return arr[:, ::-1]  # (x, y)


def _polyline_cumulative_lengths_xy(path_xy):
    """Return cumulative arc-lengths for a polyline expressed in (x, y)."""
    if path_xy.size == 0:
        return np.zeros(1, dtype=float)
    if path_xy.shape[0] == 1:
        return np.array([0.0], dtype=float)
    diffs = np.diff(path_xy, axis=0)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    return np.concatenate(([0.0], np.cumsum(seg_lengths)))


def _project_point_to_polyline(point_xy, path_xy, cum_lengths):
    """Project a point onto a polyline, returning distance, arc-length, and projection."""
    if path_xy.size == 0:
        return float("inf"), float("nan"), (float("nan"), float("nan"))
    if path_xy.shape[0] == 1:
        dist = float(np.linalg.norm(point_xy - path_xy[0]))
        return dist, 0.0, (float(path_xy[0][0]), float(path_xy[0][1]))

    best_dist = float("inf")
    best_param = 0.0
    best_point = path_xy[0]

    for i in range(path_xy.shape[0] - 1):
        a = path_xy[i]
        b = path_xy[i + 1]
        ab = b - a
        ab_len_sq = float(np.dot(ab, ab))
        if ab_len_sq == 0.0:
            proj = a
            t = 0.0
        else:
            t = float(np.clip(np.dot(point_xy - a, ab) / ab_len_sq, 0.0, 1.0))
            proj = a + t * ab
        dist = float(np.linalg.norm(point_xy - proj))
        if dist < best_dist:
            best_dist = dist
            best_param = float(cum_lengths[i] + t * math.sqrt(ab_len_sq))
            best_point = proj

    return best_dist, best_param, (float(best_point[0]), float(best_point[1]))


def _collect_frame_points_near_path(frame_groups, frame_order_map, path_xy, cum_lengths, max_distance):
    """Gather per-frame detections that lie within ``max_distance`` of a path."""
    if path_xy.size == 0:
        return []

    collected = []
    for frame_value, sub_df in frame_groups.items():
        order = frame_order_map.get(int(frame_value))
        if order is None:
            continue
        for idx, row in sub_df.iterrows():
            x = row.get("X")
            y = row.get("Y")
            if pd.isna(x) or pd.isna(y):
                continue
            point = np.array([float(x), float(y)], dtype=float)
            dist, param, _ = _project_point_to_polyline(point, path_xy, cum_lengths)
            if dist <= max_distance:
                collected.append((order, dist, param, idx))
    return collected


def _estimate_orientation_from_points(frame_param_pairs):
    """Estimate whether path parameters increase with frame order."""
    if not frame_param_pairs:
        return None

    sorted_pairs = sorted(frame_param_pairs, key=lambda item: item[0])
    n = len(sorted_pairs)
    window = max(1, n // 3)
    early_params = [p[2] for p in sorted_pairs[:window]]
    late_params = [p[2] for p in sorted_pairs[-window:]]
    if not early_params or not late_params:
        return None
    early_mean = float(np.mean(early_params))
    late_mean = float(np.mean(late_params))
    if np.isclose(early_mean, late_mean, atol=1e-3):
        return None
    return late_mean > early_mean


def match_long_exposure_traces_to_reference_particles(
    result_df,
    lei_metrics_df,
    *,
    run_folder_path: Optional[str] = None,
    ref_frame_no: Optional[int] = None,
    rec_direction: Optional[str] = None,
    max_projection_distance: float = 25.0,
    orientation_weight: float = 5.0,
    track_distance: float = 20.0,
):
    """Match LEI traces to particles detected in the reference frame.

    Parameters
    ----------
    result_df : pandas.DataFrame
        Output from :func:`wellcounter_motion_module.record_particle_positions_from_sequence`.
    lei_metrics_df : pandas.DataFrame
        Output from :func:`analyze_long_exposure_particles_advanced`.
    run_folder_path : str, optional
        Used to reuse cached metadata from :func:`generate_long_exposure_image_custom`.
    ref_frame_no : int, optional
        Explicit reference frame index. If omitted the cached metadata is used when available.
    rec_direction : {"forward", "reverse"}, optional
        Recording direction. Reuses cached metadata when omitted.
    max_projection_distance : float, optional
        Maximum distance (in pixels) a reference particle may lie from a trace centerline
        to be considered a candidate match.
    orientation_weight : float, optional
        Weight applied to the deviation along the path when scoring matches.
    track_distance : float, optional
        Distance threshold used to collect per-frame detections around each path while
        estimating its temporal orientation.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        ``(merged_df, assignments_df)`` where ``merged_df`` contains LEI metrics augmented
        with reference-frame measurements (prefixed with ``ref_``), and ``assignments_df``
        provides detailed scoring diagnostics per match.

    Examples
    --------
    The helper is wired into ``wc_analyze_one_sample.py`` so you can exercise it end-to-end::

        positions_df, lei_image = generate_long_exposure_image_custom(...)
        lei_metrics = analyze_long_exposure_particles_advanced(lei_image, run_folder_path)
        merged, diagnostics = match_long_exposure_traces_to_reference_particles(
            positions_df,
            lei_metrics,
            run_folder_path=run_folder_path,
            ref_frame_no=ref_frame_no,
            rec_direction=rec_direction,
        )

    ``merged`` contains one row per LEI trace with the reference-frame columns prefixed by
    ``ref_``; ``diagnostics`` exposes assignment scores for debugging or manual review.
    """

    if lei_metrics_df is None or lei_metrics_df.empty:
        return lei_metrics_df.copy(), pd.DataFrame()

    if result_df is None or result_df.empty or 'frame' not in result_df.columns:
        raise ValueError("result_df must contain detections with a 'frame' column.")

    metadata = {}
    if run_folder_path and _LAST_LEI_METADATA.get("run_folder_path") == run_folder_path:
        metadata = _LAST_LEI_METADATA

    direction = rec_direction or metadata.get("rec_direction") or "forward"
    direction = str(direction).lower()
    if direction not in {"forward", "reverse"}:
        direction = "forward"

    resolved_ref_frame_no = ref_frame_no
    if resolved_ref_frame_no is None:
        resolved_ref_frame_no = metadata.get("ref_frame_no")

    frames_numeric = pd.to_numeric(result_df['frame'], errors='coerce')
    valid_mask = frames_numeric.notna()
    if not valid_mask.any():
        raise ValueError("result_df does not contain usable frame identifiers.")

    work_df = result_df.loc[valid_mask].copy()
    work_df['frame_value'] = frames_numeric.loc[valid_mask].astype(int)

    unique_frames = np.sort(work_df['frame_value'].unique())
    if direction == "reverse":
        unique_frames = unique_frames[::-1]
    frame_order_map = {int(frame): idx for idx, frame in enumerate(unique_frames.tolist())}

    if not frame_order_map:
        raise ValueError("No frames available to perform matching.")

    reference_frame_value = unique_frames[0]
    if resolved_ref_frame_no is not None:
        resolved_ref_frame_no = int(resolved_ref_frame_no)

    ref_df = work_df[work_df['frame_value'] == int(reference_frame_value)].copy()
    if ref_df.empty:
        merged = lei_metrics_df.copy()
        merged['match_ref_source_index'] = pd.NA
        merged['match_ref_frame'] = pd.NA
        merged['match_score'] = np.nan
        merged['match_distance_px'] = np.nan
        merged['match_param_along_trace'] = np.nan
        merged['match_expected_param'] = np.nan
        merged['match_param_deviation'] = np.nan
        merged['match_projected_x'] = np.nan
        merged['match_projected_y'] = np.nan
        merged['trace_orientation_increasing'] = pd.NA
        merged['trace_reference_param_source'] = pd.NA
        merged['trace_reference_expected_param'] = np.nan
        merged['trace_reference_points_used'] = 0
        merged['trace_total_length'] = np.nan
        return merged, pd.DataFrame()

    frame_groups = {frame: group for frame, group in work_df.groupby('frame_value')}

    lei_df = lei_metrics_df.reset_index(drop=True).copy()
    n_traces = len(lei_df)
    n_refs = len(ref_df)

    lei_df['trace_total_length'] = np.nan
    lei_df['trace_orientation_increasing'] = pd.NA
    lei_df['trace_reference_expected_param'] = np.nan
    lei_df['trace_reference_param_source'] = pd.NA
    lei_df['trace_reference_points_used'] = 0

    if n_traces == 0 or n_refs == 0:
        merged = lei_df.copy()
        merged['match_ref_source_index'] = pd.NA
        merged['match_ref_frame'] = pd.NA
        merged['match_score'] = np.nan
        merged['match_distance_px'] = np.nan
        merged['match_param_along_trace'] = np.nan
        merged['match_expected_param'] = np.nan
        merged['match_param_deviation'] = np.nan
        merged['match_projected_x'] = np.nan
        merged['match_projected_y'] = np.nan
        return merged, pd.DataFrame()

    penalty = 1e6
    cost_matrix = np.full((n_traces, n_refs), penalty, dtype=float)
    pair_info = {}

    ref_reset = ref_df.reset_index(drop=False).rename(columns={'index': 'ref_source_index'})

    for i, lei_row in lei_df.iterrows():
        path_local = lei_row.get('centerline_path_local', [])
        min_row = lei_row.get('bbox_min_row')
        min_col = lei_row.get('bbox_min_col')
        if not path_local or pd.isna(min_row) or pd.isna(min_col):
            continue

        path_xy = _path_local_to_global_xy(path_local, min_row, min_col)
        cum_lengths = _polyline_cumulative_lengths_xy(path_xy)
        total_length = float(cum_lengths[-1]) if cum_lengths.size else 0.0
        lei_df.at[i, 'trace_total_length'] = total_length if total_length > 0 else np.nan

        frame_points = _collect_frame_points_near_path(
            frame_groups,
            frame_order_map,
            path_xy,
            cum_lengths,
            max(track_distance, max_projection_distance),
        )
        orientation_flag = _estimate_orientation_from_points(frame_points)
        lei_df.at[i, 'trace_orientation_increasing'] = (
            orientation_flag if orientation_flag is not None else pd.NA
        )
        lei_df.at[i, 'trace_reference_points_used'] = len(frame_points)

        reference_params = [p[2] for p in frame_points if p[0] == 0]
        if reference_params:
            expected_param = float(np.mean(reference_params))
            source = 'reference_points'
        elif orientation_flag is None:
            expected_param = None
            source = 'endpoint_min'
        else:
            expected_param = 0.0 if orientation_flag else total_length
            source = 'orientation_trend'

        if expected_param is not None:
            expected_param = float(max(0.0, min(total_length, expected_param)))
            lei_df.at[i, 'trace_reference_expected_param'] = expected_param
        else:
            lei_df.at[i, 'trace_reference_expected_param'] = np.nan
        lei_df.at[i, 'trace_reference_param_source'] = source

        if path_xy.size == 0:
            continue

        for j, ref_row in ref_reset.iterrows():
            x = ref_row.get('X')
            y = ref_row.get('Y')
            if pd.isna(x) or pd.isna(y):
                continue
            point = np.array([float(x), float(y)], dtype=float)
            dist, param, proj = _project_point_to_polyline(point, path_xy, cum_lengths)
            if not np.isfinite(dist) or dist > max_projection_distance:
                continue

            if total_length <= 0:
                param_norm = 0.0
                expected = 0.0 if expected_param is None else expected_param
            else:
                if expected_param is None:
                    endpoint_param = min(param, total_length - param)
                    param_norm = endpoint_param / max(total_length, 1.0)
                    expected = float(total_length / 2.0)
                else:
                    param_norm = abs(param - expected_param) / max(total_length, 1.0)
                    expected = expected_param

            score = float(dist + orientation_weight * param_norm)
            if score >= penalty:
                continue

            cost_matrix[i, j] = score
            pair_info[(i, j)] = {
                'distance': dist,
                'param': param,
                'param_norm': param_norm,
                'expected_param': expected,
                'projected_point': proj,
                'total_length': total_length,
            }

    if not pair_info:
        merged = lei_df.copy()
        merged['match_ref_source_index'] = pd.NA
        merged['match_ref_frame'] = pd.NA
        merged['match_score'] = np.nan
        merged['match_distance_px'] = np.nan
        merged['match_param_along_trace'] = np.nan
        merged['match_expected_param'] = np.nan
        merged['match_param_deviation'] = np.nan
        merged['match_projected_x'] = np.nan
        merged['match_projected_y'] = np.nan
        assignments_df = pd.DataFrame()
        return merged, assignments_df

    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    assignments = []
    lei_df['match_ref_source_index'] = pd.NA
    lei_df['match_ref_frame'] = pd.NA
    lei_df['match_score'] = np.nan
    lei_df['match_distance_px'] = np.nan
    lei_df['match_param_along_trace'] = np.nan
    lei_df['match_expected_param'] = np.nan
    lei_df['match_param_deviation'] = np.nan
    lei_df['match_projected_x'] = np.nan
    lei_df['match_projected_y'] = np.nan

    for r, c in zip(row_ind, col_ind):
        info = pair_info.get((r, c))
        cost = cost_matrix[r, c]
        if info is None or cost >= penalty:
            continue
        ref_row = ref_reset.iloc[c]
        lei_row = lei_df.iloc[r]

        deviation = info['param'] - info['expected_param']

        lei_df.at[r, 'match_ref_source_index'] = int(ref_row['ref_source_index'])
        lei_df.at[r, 'match_ref_frame'] = int(ref_row['frame_value'])
        lei_df.at[r, 'match_score'] = float(cost)
        lei_df.at[r, 'match_distance_px'] = float(info['distance'])
        lei_df.at[r, 'match_param_along_trace'] = float(info['param'])
        lei_df.at[r, 'match_expected_param'] = float(info['expected_param'])
        lei_df.at[r, 'match_param_deviation'] = float(deviation)
        lei_df.at[r, 'match_projected_x'] = float(info['projected_point'][0])
        lei_df.at[r, 'match_projected_y'] = float(info['projected_point'][1])

        assignments.append({
            'lei_index': int(r),
            'lei_particle_id': lei_row.get('particle_id'),
            'ref_source_index': int(ref_row['ref_source_index']),
            'ref_frame_value': int(ref_row['frame_value']),
            'match_score': float(cost),
            'distance_px': float(info['distance']),
            'param_along_trace': float(info['param']),
            'expected_param': float(info['expected_param']),
            'param_deviation': float(deviation),
            'projected_x': float(info['projected_point'][0]),
            'projected_y': float(info['projected_point'][1]),
            'trace_total_length': float(info['total_length']),
            'orientation_increasing': lei_df.at[r, 'trace_orientation_increasing'],
        })

    assignments_df = pd.DataFrame(assignments)

    ref_prefixed = ref_reset.add_prefix('ref_')
    merged = lei_df.merge(
        ref_prefixed,
        how='left',
        left_on='match_ref_source_index',
        right_on='ref_ref_source_index'
    )
    if 'ref_ref_source_index' in merged.columns:
        merged = merged.drop(columns=['ref_ref_source_index'])

    if resolved_ref_frame_no is not None:
        merged['requested_ref_frame_no'] = int(resolved_ref_frame_no)
    merged['reference_frame_value'] = int(reference_frame_value)
    merged['recording_direction'] = direction

    return merged, assignments_df

def render_male_centerline_collage(
    df,
    *,
    run_folder_path: str,
    resolved_ref_frame_no: int,
    output_dir: str,
    metric: str = "centerline_mean_width",
    crop_size: int = 250,
    n_cols: int = 6,
    particle_diagnostics=None,
):
    """Render the male diagnostic collage with centerline overlays and annotations."""

    if df is None or len(df) == 0:
        print("[collage_centerline] No particle data available for collage rendering.")
        return

    if run_folder_path is None:
        print("[collage_centerline] Missing run folder path; cannot locate reference frames.")
        return

    os.makedirs(output_dir, exist_ok=True)

    print("[collage_centerline] Starting centerline collage creation...")
    try:
        jpg_dir = os.path.join(run_folder_path, "jpg")
        image_files = sorted(glob.glob(os.path.join(jpg_dir, "*.jpg")))
        if not image_files:
            print(f"[collage_centerline] No images found in {jpg_dir}. Cannot create collage.")
            return

        resolved_idx = int(resolved_ref_frame_no or 0)
        clamped_idx = max(0, min(resolved_idx, len(image_files) - 1))
        if clamped_idx != resolved_idx:
            print(
                "[collage_centerline] Warning: requested ref_frame_no "
                f"{resolved_idx} outside available range. Using {clamped_idx} instead."
            )

        reference_frame_path = image_files[clamped_idx]
        print(
            f"[collage_centerline] Using reference frame index {clamped_idx}: {reference_frame_path}"
        )

        reference_frame = cv2.imread(reference_frame_path)
        if reference_frame is None:
            print(f"[collage_centerline] Failed to read {reference_frame_path}.")
            return

        df_sorted = df.sort_values(by=metric, ascending=True).reset_index(drop=True)
        n_particles = len(df_sorted)
        if n_particles == 0:
            print("[collage_centerline] No particles remain after sorting; skipping collage.")
            return

        max_particles = 24
        if n_particles > max_particles:
            print(
                f"[collage_centerline] Limiting collage to the first {max_particles} particles "
                f"out of {n_particles} available."
            )
        df_limited = df_sorted.head(max_particles)
        n_display = len(df_limited)

        n_rows = int(np.ceil(n_display / n_cols)) if n_display else 0
        half = crop_size // 2
        h, w = reference_frame.shape[:2]
        crops = []

        diag_lookup = {}
        if particle_diagnostics:
            diag_lookup = {int(d.get("particle_id")): d for d in particle_diagnostics if "particle_id" in d}

        for _, row in df_limited.iterrows():
            if {
                "X",
                "Y",
                "particle_id",
            } - set(row.index):
                continue

            cx = row.get("X")
            cy = row.get("Y")
            if pd.isna(cx) or pd.isna(cy):
                continue

            cx_i, cy_i = int(cx), int(cy)
            x1, x2 = max(0, cx_i - half), min(w, cx_i + half)
            y1, y2 = max(0, cy_i - half), min(h, cy_i + half)
            crop = reference_frame[y1:y2, x1:x2].copy()

            diag = diag_lookup.get(int(row.get("particle_id")))
            if diag:
                min_row, min_col, max_row, max_col = diag.get("bbox", (0, 0, 0, 0))
                ridge = diag.get("ridge_mask")
                if ridge is not None:
                    overlap_y1 = max(y1, min_row)
                    overlap_y2 = min(y2, max_row)
                    overlap_x1 = max(x1, min_col)
                    overlap_x2 = min(x2, max_col)
                    if overlap_y1 < overlap_y2 and overlap_x1 < overlap_x2:
                        ridge_sub = ridge[
                            overlap_y1 - min_row : overlap_y2 - min_row,
                            overlap_x1 - min_col : overlap_x2 - min_col,
                        ]
                        crop_sub = crop[
                            overlap_y1 - y1 : overlap_y2 - y1,
                            overlap_x1 - x1 : overlap_x2 - x1,
                        ]
                        crop_sub[ridge_sub] = (0, 255, 0)

            crop = cv2.resize(crop, (crop_size, crop_size))

            metric_value = row.get(metric, np.nan)
            metric_text = f"{metric}={metric_value:.2f}" if pd.notna(metric_value) else f"{metric}=NA"

            bl_value = row.get("body_lengths_traveled", np.nan)
            bl_text = (
                f"body_lengths_traveled={bl_value:.2f}" if pd.notna(bl_value) else "body_lengths_traveled=NA"
            )

            cv2.putText(
                crop,
                metric_text,
                (5, crop_size - 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                crop,
                bl_text,
                (5, crop_size - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            crops.append(crop)

        if not crops:
            print("[collage_centerline] No valid crops created.")
            return

        rows = []
        for i in range(n_rows):
            row_imgs = crops[i * n_cols : (i + 1) * n_cols]
            if not row_imgs:
                continue
            if len(row_imgs) < n_cols:
                pad_img = np.zeros_like(row_imgs[0])
                row_imgs += [pad_img] * (n_cols - len(row_imgs))
            rows.append(np.hstack(row_imgs))

        if not rows:
            print("[collage_centerline] No rows assembled for collage.")
            return

        collage = np.vstack(rows)
        frame_suffix = f"_frame{int(resolved_ref_frame_no or 0)}"
        collage_path = os.path.join(
            output_dir, f"particle_collage_centerline_by_{metric}{frame_suffix}.jpg"
        )
        cv2.imwrite(collage_path, collage)
        print(f"[collage_centerline] Saved: {collage_path}")

    except Exception:
        import traceback

        print(f"[collage_centerline] Error while creating collage:\n{traceback.format_exc()}")


def analyze_long_exposure_particles_advanced(
    long_exposure_image,
    run_folder_path,
    collage_metric="centerline_mean_width",
    ref_frame_no: Optional[int] = None,
    *,
    defer_collage: bool = False,
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

    Other Parameters
    ----------------
    defer_collage : bool, optional
        When ``True`` the diagnostic collage is not written immediately. Instead,
        the function stores the rendering context in ``DataFrame.attrs`` so the
        caller can generate the collage after additional annotations (for
        example ``body_lengths_traveled``) have been computed.

    Returns
    -------
    pandas.DataFrame
        Metrics per particle. When ``defer_collage`` is ``True`` and outputs are
        enabled, ``df.attrs['collage_context']`` contains the parameters required
        to render the collage later.
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
    particle_diagnostics = []
    region_infos = []
    centerline_inputs = []

    for idx, region in enumerate(props, start=1):
        if region.area < 10:
            continue

        min_row, min_col, max_row, max_col = region.bbox
        mask = (labeled[min_row:max_row, min_col:max_col] == region.label)
        mask_uint8 = mask.astype(np.uint8, copy=False)
        area = int(region.area)

        region_infos.append({
            "idx": idx,
            "region": region,
            "area": area,
            "min_row": int(min_row),
            "min_col": int(min_col),
            "max_row": int(max_row),
            "max_col": int(max_col),
        })
        centerline_inputs.append((idx, mask_uint8, int(min_row), int(min_col), int(max_row), int(max_col)))

    centerline_results = []
    if centerline_inputs:
        if Parallel is not None and len(centerline_inputs) > 1:
            n_jobs = min(os.cpu_count() or 1, len(centerline_inputs))
            centerline_results = Parallel(n_jobs=n_jobs, prefer="processes")(  # type: ignore[misc]
                delayed(_compute_centerline_for_region)(idx, mask, min_row, min_col, max_row, max_col)
                for idx, mask, min_row, min_col, max_row, max_col in centerline_inputs
            )
        else:
            centerline_results = [
                _compute_centerline_for_region(idx, mask, min_row, min_col, max_row, max_col)
                for idx, mask, min_row, min_col, max_row, max_col in centerline_inputs
            ]

    centerline_map = {res["idx"]: res for res in centerline_results}

    for info in region_infos:
        idx = info["idx"]
        region = info["region"]
        centerline_data = centerline_map.get(idx)
        if centerline_data is None:
            continue

        min_row = info["min_row"]
        min_col = info["min_col"]
        max_row = info["max_row"]
        max_col = info["max_col"]
        area = info["area"]

        ridge_mask = centerline_data["ridge_mask"]
        ridge_path_local = centerline_data["ridge_path_local"]
        ridge_length = centerline_data["ridge_length"]
        start_global = centerline_data["start_global"]
        end_global = centerline_data["end_global"]
        centerline_metrics = centerline_data["centerline_metrics"]

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
            "bbox_min_row": int(min_row),
            "bbox_min_col": int(min_col),
            "bbox_max_row": int(max_row),
            "bbox_max_col": int(max_col),
            "centerline_path_local": ridge_path_local,
            "centerline_endpoint_start": start_global,
            "centerline_endpoint_end": end_global,
        })

        if save_outputs:
            overlay_slice = overlay[min_row:max_row, min_col:max_col]
            overlay_slice[ridge_mask] = (0, 0, 255)
            particle_diagnostics.append({
                "particle_id": idx,
                "bbox": centerline_data["bbox"],
                "ridge_mask": ridge_mask,
            })
        

    df = pd.DataFrame(results)
    print(f"[analyze_long_exposure_particles_advanced] Analyzed {len(df)} traces.")

    # --- For diagnostic purposes (do not delete) ---
    #if save_outputs:
    #    frame_suffix = f"_frame{resolved_ref_frame_no}"
    #    analyzed_path = os.path.join(output_dir, f"LEI_males_analyzed{frame_suffix}.jpg")
    #    df_path = os.path.join(output_dir, f"LEI_males_metrics{frame_suffix}.csv")
    #    cv2.imwrite(analyzed_path, overlay)
    #    df.to_csv(df_path, index=False)
    #    print(f"[analyze_long_exposure_particles_advanced] Saved: {analyzed_path}")
    #    print(f"[analyze_long_exposure_particles_advanced] Saved: {df_path}")


    # --- Diagnostic visualization: Geodesic centerline overlay ---
    if save_outputs:
        lei_centerline_overlay = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2BGR)

        for diag in particle_diagnostics:
            min_row, min_col, max_row, max_col = diag["bbox"]
            ridge = diag["ridge_mask"]
            region_slice = lei_centerline_overlay[min_row:max_row, min_col:max_col]
            region_slice[ridge] = (0, 0, 255)

        frame_suffix = f"_frame{resolved_ref_frame_no}"
        out_path_centerline = os.path.join(output_dir, f"LEI_males_centerline{frame_suffix}.jpg")
        cv2.imwrite(out_path_centerline, lei_centerline_overlay)
        print(f"[analyze_long_exposure_particles_advanced] Geodesic centerline overlay saved: {out_path_centerline}")


    collage_context = {
        "run_folder_path": run_folder_path,
        "resolved_ref_frame_no": resolved_ref_frame_no,
        "output_dir": output_dir,
        "particle_diagnostics": particle_diagnostics,
        "metric": collage_metric,
        "crop_size": 250,
        "n_cols": 6,
    }

    if save_outputs:
        if defer_collage:
            df.attrs["collage_context"] = collage_context
        else:
            render_male_centerline_collage(
                df,
                run_folder_path=run_folder_path,
                resolved_ref_frame_no=resolved_ref_frame_no,
                output_dir=output_dir,
                metric=collage_metric,
                crop_size=250,
                n_cols=6,
                particle_diagnostics=particle_diagnostics,
            )

    return df


def run_male_analysis(
    run_folder_path: str,
    ref_frame_no: int,
    rec_direction: str,
    *,
    collage_metric: str = "centerline_mean_width",
    save_outputs: Optional[bool] = None,
):
    """Execute the full male-trace analysis workflow for a single sample.

    The routine generates a long-exposure image, extracts trace metrics, links
    the traces back to reference-frame detections, augments the merged
    dataframe, and optionally persists the results to disk. All generated
    artefacts embed the ``ref_frame_no`` in their filename to simplify manual
    inspection of multiple reference frames. Long-exposure parameters are read
    from the ``male_analysis`` section of ``wellcounter_config.yml`` at the
    moment the LEI is generated, so no manual propagation of these values is
    required.

    Parameters
    ----------
    run_folder_path : str
        Path to the sample run folder.
    ref_frame_no : int
        Reference frame number used for accumulation and diagnostics.
    rec_direction : {"forward", "reverse"}
        Direction of accumulation relative to the reference frame.
    collage_metric : str, optional
        Sorting metric for the optional diagnostic collage.
    save_outputs : bool, optional
        When ``None`` (default) the flag follows the ``particle_detection``
        output toggle from ``wellcounter_config.yml``. A boolean overrides the
        configuration-driven behaviour.

    Returns
    -------
    tuple
        ``(positions_df, long_exposure_image, maledetect_df, merged_df,
        assignments_df)`` corresponding to the outputs of the constituent
        helpers, with ``merged_df`` already cleaned and augmented.
    """

    positions_df, long_exposure_image = generate_long_exposure_image_custom(
        run_folder_path,
        ref_frame_no,
        rec_direction,
    )

    if long_exposure_image is None or positions_df is None:
        empty = pd.DataFrame()
        return positions_df, long_exposure_image, empty, empty, empty

    maledetect_df = analyze_long_exposure_particles_advanced(
        long_exposure_image,
        run_folder_path,
        collage_metric=collage_metric,
        ref_frame_no=ref_frame_no,
        defer_collage=True,
    )

    merged_df, assignments_df = match_long_exposure_traces_to_reference_particles(
        positions_df,
        maledetect_df,
        run_folder_path=run_folder_path,
        ref_frame_no=ref_frame_no,
        rec_direction=rec_direction,
    )

    merged_df = merged_df.copy()
    if "centerline_path_local" in merged_df.columns:
        merged_df = merged_df.drop(columns=["centerline_path_local"])

    feret_raw = merged_df.get("ref_feret_diameter")
    centerline_raw = merged_df.get("centerline_length")

    feret = (
        pd.to_numeric(feret_raw, errors="coerce")
        if feret_raw is not None
        else pd.Series(np.nan, index=merged_df.index)
    )
    centerline_length = (
        pd.to_numeric(centerline_raw, errors="coerce")
        if centerline_raw is not None
        else pd.Series(np.nan, index=merged_df.index)
    )

    with np.errstate(divide="ignore", invalid="ignore"):
        body_lengths = centerline_length / feret.replace(0, np.nan)
    merged_df["body_lengths_traveled"] = body_lengths

    body_lengths_map = None
    if "particle_id" in merged_df.columns:
        body_lengths_map = merged_df.set_index("particle_id")["body_lengths_traveled"]
    if body_lengths_map is not None and "particle_id" in maledetect_df.columns:
        collage_context = maledetect_df.attrs.get("collage_context")
        maledetect_df = maledetect_df.copy()
        if collage_context is not None:
            maledetect_df.attrs["collage_context"] = collage_context
        maledetect_df["body_lengths_traveled"] = maledetect_df["particle_id"].map(body_lengths_map)

    if save_outputs is None:
        try:
            with open("wellcounter_config.yml", "r") as f:
                config = yaml.safe_load(f)
            save_outputs_flag = bool(config.get("outputs", {}).get("particle_detection", False))
        except Exception:
            save_outputs_flag = False
    else:
        save_outputs_flag = bool(save_outputs)

    if save_outputs_flag:
        parent_dir = os.path.dirname(run_folder_path.rstrip("/\\"))
        folder_name = os.path.basename(run_folder_path.rstrip("/\\"))
        output_dir = os.path.join(parent_dir, f"{folder_name}_particle_analysis")
        os.makedirs(output_dir, exist_ok=True)
        frame_suffix = f"_frame{int(ref_frame_no)}"
        merged_path = os.path.join(output_dir, f"merged_df{frame_suffix}.csv")
        # assignments_path = os.path.join(output_dir, f"assignments_df{frame_suffix}.csv")
        merged_df.to_csv(merged_path, index=False)
        # assignments_df.to_csv(assignments_path, index=False)
        print(f"[run_male_analysis_pipeline] Saved: {merged_path}")
        # print(f"[run_male_analysis_pipeline] Saved: {assignments_path}")

        collage_context = maledetect_df.attrs.get("collage_context")
        if collage_context:
            render_male_centerline_collage(
                merged_df,
                run_folder_path=collage_context.get("run_folder_path", run_folder_path),
                resolved_ref_frame_no=collage_context.get("resolved_ref_frame_no", ref_frame_no or 0),
                output_dir=collage_context.get("output_dir", output_dir),
                metric=collage_context.get("metric", collage_metric),
                crop_size=collage_context.get("crop_size", 250),
                n_cols=collage_context.get("n_cols", 6),
                particle_diagnostics=collage_context.get("particle_diagnostics"),
            )

    return merged_df


def count_males(
    run_folder_path: str,
    *,
    collage_metric: str = "centerline_mean_width",
    save_outputs: Optional[bool] = None,
    reference_indices: Optional[list[int]] = None,
    reference_frame_numbers: Optional[list[int]] = None,
):
    """High-level controller that samples three reference frames for male detection.

    Parameters
    ----------
    run_folder_path : str
        Path to the sample run folder containing the image sequence (expects a
        ``jpg`` subdirectory, matching :func:`run_male_analysis`).
    collage_metric : str, optional
        Sorting metric forwarded to :func:`run_male_analysis` for optional
        diagnostic collages.
    save_outputs : bool, optional
        Overrides the configuration-driven output toggle when provided. When
        ``None`` (default) the behaviour follows the underlying configuration.

    Returns
    -------
    tuple[pandas.DataFrame, pandas.DataFrame]
        The first element (``simplified_df``) contains the condensed male
        summary with columns ``ref_frame``, ``X``, ``Y``, ``male``, and
        ``body_lengths_traveled``. The second element (``detailed_df``) mirrors
        the simplified columns while also including the extended morphology
        metrics copied from the underlying merged analysis output.
    """

    detail_columns = [
        "area",
        "solidity",
        "circularity",
        "ridge_length",
        "centerline_length",
        "centerline_chord_length",
        "centerline_straightness",
        "centerline_mean_curvature",
        "centerline_mean_width",
        "centerline_width_std",
        "centerline_n_pixels",
        "bbox_min_row",
        "bbox_min_col",
        "bbox_max_row",
        "bbox_max_col",
        "ref_area",
        "ref_perimeter",
        "ref_orientation",
        "ref_aspect_ratio",
        "ref_solidity",
        "ref_eccentricity",
        "ref_feret_diameter",
        "ref_bounding_x",
        "ref_bounding_y",
        "ref_bounding_w",
        "ref_bounding_h",
    ]

    def filter_male_particles(
        merged_df: pd.DataFrame,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Project the merged analysis output to simplified and detailed tables."""

        empty_simplified = pd.DataFrame(
            {
                "X": pd.Series(dtype=float),
                "Y": pd.Series(dtype=float),
                "male": pd.Series(dtype=bool),
                "body_lengths_traveled": pd.Series(dtype=float),
            }
        )

        if merged_df is None or merged_df.empty:
            empty_detailed = empty_simplified.copy()
            for column in detail_columns:
                empty_detailed[column] = pd.Series(dtype=float)
            return empty_simplified, empty_detailed

        work_df = merged_df.reset_index(drop=True).copy()

        if "ref_X" in work_df.columns:
            x_series = pd.to_numeric(work_df["ref_X"], errors="coerce")
        else:
            x_series = pd.Series(np.nan, index=work_df.index, dtype=float)

        if "ref_Y" in work_df.columns:
            y_series = pd.to_numeric(work_df["ref_Y"], errors="coerce")
        else:
            y_series = pd.Series(np.nan, index=work_df.index, dtype=float)

        if "body_lengths_traveled" in work_df.columns:
            body_lengths = pd.to_numeric(
                work_df["body_lengths_traveled"], errors="coerce"
            )
        else:
            body_lengths = pd.Series(np.nan, index=work_df.index, dtype=float)

        male_flag = (body_lengths >= 3).fillna(False).astype(bool)

        simplified_data = {
            "X": x_series,
            "Y": y_series,
            "male": male_flag,
            "body_lengths_traveled": body_lengths,
        }

        detailed_data = {
            key: value.copy(deep=True) if hasattr(value, "copy") else value
            for key, value in simplified_data.items()
        }

        for column in detail_columns:
            if column in work_df.columns:
                detailed_series = pd.to_numeric(work_df[column], errors="coerce")
            else:
                detailed_series = pd.Series(np.nan, index=work_df.index, dtype=float)
            detailed_data[column] = detailed_series

        simplified_df = pd.DataFrame(simplified_data)
        detailed_df = pd.DataFrame(detailed_data)

        return simplified_df, detailed_df

    (
        _auto_dataset_type,
        auto_indices,
        auto_frame_numbers,
        image_file_list,
        frame_numbers,
    ) = wim.determine_reference_frames(run_folder_path)

    if not image_file_list:
        print(f"[count_males] No image files found in {run_folder_path}.")
        simplified_columns = ["ref_frame", "ref_frame_index", "X", "Y", "male", "body_lengths_traveled"]
        detailed_columns_full = [
            "ref_frame",
            "ref_frame_index",
            "X",
            "Y",
            "male",
            "body_lengths_traveled",
            *detail_columns,
        ]
        simplified_empty = pd.DataFrame(columns=simplified_columns)
        detailed_empty = pd.DataFrame(columns=detailed_columns_full)
        return simplified_empty, detailed_empty

    total_frames = len(image_file_list)

    cleaned_indices = []
    if reference_indices is not None:
        for idx in reference_indices:
            try:
                idx_int = int(idx)
            except (TypeError, ValueError):
                continue
            if 0 <= idx_int < total_frames:
                cleaned_indices.append(idx_int)

    expected_len = len(auto_indices) if auto_indices else len(cleaned_indices)
    if expected_len == 0 and cleaned_indices:
        expected_len = len(cleaned_indices)

    if cleaned_indices:
        selected_indices = cleaned_indices[:expected_len or len(cleaned_indices)]
        if expected_len and len(selected_indices) < expected_len:
            selected_indices.extend(auto_indices[len(selected_indices):expected_len])
    else:
        selected_indices = auto_indices[:]

    if not selected_indices:
        selected_indices = [0] if total_frames else []

    selected_indices = [int(idx) for idx in selected_indices if 0 <= int(idx) < total_frames]

    auto_map = {idx: val for idx, val in zip(auto_indices, auto_frame_numbers)}

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
    for idx in selected_indices:
        fallback = auto_map.get(idx)
        if fallback is None and 0 <= idx < len(frame_numbers) and not pd.isna(frame_numbers[idx]):
            fallback = int(frame_numbers[idx])
        if fallback is None:
            fallback = int(idx)
        fallback_numbers.append(int(fallback))

    provided_numbers = reference_frame_numbers or []
    resolved_numbers = []
    for pos, fallback in enumerate(fallback_numbers):
        candidate = provided_numbers[pos] if pos < len(provided_numbers) else None
        resolved_numbers.append(normalize_frame_value(candidate, fallback))

    simplified_results = []
    detailed_results = []
    for idx, ref_idx in enumerate(selected_indices):
        direction = "reverse" if idx == len(selected_indices) - 1 else "forward"

        analysis_result = run_male_analysis(
            run_folder_path,
            ref_idx,
            direction,
            collage_metric=collage_metric,
            save_outputs=save_outputs,
        )

        merged_df = (
            analysis_result[-1]
            if isinstance(analysis_result, tuple) and analysis_result
            else analysis_result
        )

        if not isinstance(merged_df, pd.DataFrame):
            merged_df = pd.DataFrame()

        simplified_df, detailed_df = filter_male_particles(merged_df)

        base_value = resolved_numbers[idx] if idx < len(resolved_numbers) else ref_idx
        ref_value = normalize_frame_value(base_value, ref_idx)

        if not simplified_df.empty:
            if "reference_frame_value" in merged_df.columns:
                candidate = merged_df["reference_frame_value"].iloc[0]
                if pd.notna(candidate):
                    ref_value = normalize_frame_value(candidate, ref_value)
            elif "requested_ref_frame_no" in merged_df.columns:
                candidate = merged_df["requested_ref_frame_no"].iloc[0]
                if pd.notna(candidate):
                    ref_value = normalize_frame_value(candidate, ref_value)

        simplified_df.insert(0, "ref_frame", ref_value)
        simplified_df.insert(1, "ref_frame_index", int(ref_idx))
        detailed_df.insert(0, "ref_frame", ref_value)
        detailed_df.insert(1, "ref_frame_index", int(ref_idx))
        simplified_results.append(simplified_df)
        detailed_results.append(detailed_df)

    simplified_columns = ["ref_frame", "ref_frame_index", "X", "Y", "male", "body_lengths_traveled"]
    detailed_columns_full = [
        "ref_frame",
        "ref_frame_index",
        "X",
        "Y",
        "male",
        "body_lengths_traveled",
        *detail_columns,
    ]

    if not simplified_results:
        simplified_empty = pd.DataFrame(columns=simplified_columns)
        detailed_empty = pd.DataFrame(columns=detailed_columns_full)
        return simplified_empty, detailed_empty

    simplified_combined = pd.concat(simplified_results, ignore_index=True)
    detailed_combined = pd.concat(detailed_results, ignore_index=True)

    return (
        simplified_combined[simplified_columns],
        detailed_combined[detailed_columns_full],
    )
