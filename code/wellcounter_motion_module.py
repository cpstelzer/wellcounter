# -*- coding: utf-8 -*-
"""
Wellcounter motion module (Modified for FPS-in-Filename)

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
This module analyzes swimming behavior from image sequences where the FPS is
embedded in the filename.

Author: Claus-Peter Stelzer
Date: 2025-02-07
Modification Date: 2025-10-15
"""

import cv2
import wellcounter_imaging_module as wim
import pandas as pd
import numpy as np
import random
import os
import re
import yaml


# --- UNCHANGED FUNCTIONS (Included for completeness) ---

def track_particles(particles_by_frame):
    config = wim.read_config()
    motion_params = config['motion']
    trajectories = {}
    input_data = particles_by_frame.sort_values(by='frame')
    for _, row in input_data.iterrows():
        frame, x, y, area = row['frame'], row['X'], row['Y'], row['area']
        closest_id, min_distance = None, float('inf')
        for obj_id, trajectory in trajectories.items():
            _, last_x, last_y, _ = trajectory[-1]
            distance = np.hypot(x - last_x, y - last_y)
            if distance <= motion_params['max_search_distance'] and distance < min_distance:
                min_distance, closest_id = distance, obj_id
        if closest_id is not None:
            trajectories[closest_id].append((frame, x, y, area))
        else:
            trajectories[len(trajectories) + 1] = [(frame, x, y, area)]
    return {k: v for k, v in trajectories.items() if len(v) >= motion_params['min_trajectory_size']}


def visualize_trajectories(original_image, particle_trajectories):
    config = wim.read_config()
    motion_params = config['motion']
    image_with_trajectories = cv2.cvtColor(original_image, cv2.COLOR_GRAY2BGR) if len(original_image.shape) == 2 else original_image.copy()
    h, w = original_image.shape[:2]
    trajectories_only_image = np.zeros((h, w, 3), dtype=np.uint8)
    for _, trajectory in particle_trajectories.items():
        color = tuple(np.random.randint(0, 256, 3).tolist()) if motion_params['trajectory_random_colors'] else (0, 0, 255)
        for i in range(1, len(trajectory)):
            p1 = (int(trajectory[i-1][1]), int(trajectory[i-1][2]))
            p2 = (int(trajectory[i][1]), int(trajectory[i][2]))
            cv2.line(image_with_trajectories, p1, p2, color, thickness=2)
            cv2.line(trajectories_only_image, p1, p2, color, thickness=2)
    return image_with_trajectories, trajectories_only_image


def extract_movement_variables(trajectories):
    parameters = []
    for obj_id, trajectory in trajectories.items():
        if len(trajectory) < 2: continue
        distances, total_distance, total_time, total_angle_change = [], 0, 0, 0
        for i in range(1, len(trajectory)):
            frame1, x1, y1, _ = trajectory[i-1]
            frame2, x2, y2, _ = trajectory[i]
            dx, dy, dt = x2 - x1, y2 - y1, frame2 - frame1
            dist = np.hypot(dx, dy)
            distances.append(dist)
            total_distance += dist
            total_time += dt
            total_angle_change += abs(np.arctan2(dy, dx))
        start_point, end_point = np.array(trajectory[0][1:3]), np.array(trajectory[-1][1:3])
        shortest_path = np.linalg.norm(end_point - start_point)
        avg_speed = total_distance / total_time if total_time > 0 else 0
        avg_directionality = total_angle_change / len(trajectory) if len(trajectory) > 0 else np.nan
        meander = total_distance / shortest_path if shortest_path > 0 else np.nan
        parameters.append([obj_id, avg_speed, max(distances), avg_directionality, meander, total_distance, len(trajectory)])
    columns = ['obj_id', 'avg_speed', 'max_speed', 'directionality', 'meandering_index', 'displacement', 'trajectory_size']
    return pd.DataFrame(parameters, columns=columns)


def summarize_movement_variables(mov_vars_df):
    if mov_vars_df.empty:
        print("Error: No particle trajectories detected!")
        return pd.DataFrame(columns=['avg_speed', 'max_speed', 'directionality', 'meandering_index', 'displacement'])
    summary = {}
    for col in ['avg_speed', 'max_speed', 'directionality', 'meandering_index', 'displacement']:
        if col in mov_vars_df.columns:
            weighted_avg = (mov_vars_df[col] * mov_vars_df['trajectory_size']).sum() / mov_vars_df['trajectory_size'].sum()
            summary[col] = [round(weighted_avg, 3)]
    return pd.DataFrame(summary)


# --- MODIFIED HIGH-LEVEL FUNCTIONS ---

def record_particle_positions_from_sequence(
    run_folder_path,
    ref_frame_no: int = 0,
    rec_direction: str = 'forward'
):
    """Record particle positions and accumulate a long-exposure image.

    Parameters
    ----------
    run_folder_path : str
        Path to the run folder containing the image sequence (expects subfolder 'jpg').
    ref_frame_no : int, optional
        Index of the reference frame that anchors the analysis window. Defaults to 0
        (the first frame).
    rec_direction : {"forward", "reverse"}, optional
        Direction in which frames are sampled relative to ``ref_frame_no``. Use
        ``"forward"`` to include the reference frame and later frames, or
        ``"reverse"`` to include preceding frames up to the reference frame.

    Returns
    -------
    tuple[pandas.DataFrame, numpy.ndarray | None]
        DataFrame of detected particles and the accumulated long-exposure image.
    """

    config = wim.read_config()
    motion_params = config['motion']

    # --- account for subdirectory containing images ---
    image_folder = os.path.join(run_folder_path, "jpg")

    image_file_list = wim.get_image_file_list(image_folder)
    fps = wim.get_fps_from_sequence(image_folder)
    total_frames = len(image_file_list)

    if total_frames == 0:
        return pd.DataFrame(), None

    # Extract the real frame numbers embedded in filenames (e.g. "_f00087_")
    frame_number_lookup = []
    frame_number_pattern = re.compile(r"_f(\d+)_")
    for image_path in image_file_list:
        match = frame_number_pattern.search(os.path.basename(image_path))
        frame_number_lookup.append(int(match.group(1)) if match else None)

    if ref_frame_no < 0 or ref_frame_no >= total_frames:
        print(
            f"[record_particle_positions_from_sequence] Warning: ref_frame_no {ref_frame_no} "
            f"out of bounds for {total_frames} frames. Falling back to 0."
        )
        ref_frame_no = 0

    direction = rec_direction.lower()
    if direction not in {"forward", "reverse"}:
        raise ValueError(
            f"Invalid rec_direction '{rec_direction}'. Expected 'forward' or 'reverse'."
        )

    reference_frame = wim.get_frame_from_sequence(image_file_list, ref_frame_no)
    if reference_frame is None:
        return pd.DataFrame(), None

    _, global_mask = wim.mask_well_area(reference_frame)

    height, width = reference_frame.shape
    long_exposure_image = np.zeros((height, width), dtype=np.uint8)
    result_df = pd.DataFrame()

    subtraction_offset = int(5 * fps)
    step = 1 if direction == "forward" else -1

    available_frames = (
        total_frames - ref_frame_no
        if direction == "forward"
        else ref_frame_no + 1
    )
    number_of_iterations = min(int(motion_params['analysis_duration'] * fps), available_frames)

    current_idx = ref_frame_no
    for iteration in range(number_of_iterations):
        if current_idx < 0 or current_idx >= total_frames:
            break

        print(
            "Motion analysis\nProcessing frame no. "
            f"{iteration + 1} of {number_of_iterations} (frame index {current_idx})"
        )

        frame_a_idx = current_idx
        if direction == "forward":
            frame_b_idx = min(frame_a_idx + subtraction_offset, total_frames - 1)
        else:
            frame_b_idx = max(frame_a_idx - subtraction_offset, 0)

        subtr_image, _ = wim.image_subtraction_from_sequence(
            image_file_list,
            frame_a_idx,
            frame_b_idx,
            cached_mask=global_mask
        )
        if subtr_image is None:
            current_idx += step
            continue

        table_of_particles, binary_image = wim.analyze_microorganisms(subtr_image)

        actual_frame_number = None
        if 0 <= frame_a_idx < len(frame_number_lookup):
            actual_frame_number = frame_number_lookup[frame_a_idx]
        table_of_particles.insert(
            0,
            'frame',
            int(actual_frame_number) if actual_frame_number is not None else int(frame_a_idx)
        )

        result_df = pd.concat([result_df, table_of_particles], ignore_index=True)
        long_exposure_image = cv2.add(long_exposure_image, binary_image)

        current_idx += step

    return result_df.reset_index(drop=True), long_exposure_image

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

    import os
    import cv2
    import yaml
    import wellcounter_imaging_module as wim

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
        result_df, long_exposure_image = record_particle_positions_from_sequence(
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
        cv2.imwrite(lei_path, long_exposure_image)

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

    return result_df, long_exposure_image





def perform_motion_analysis(run_folder_path):
    """
    Performs high-level motion analysis on image sequences located in 'run_folder_path/jpg'.
    Outputs motion analysis results and associated graphical visualizations if enabled in the config file.
    """

    config = wim.read_config()
    output_params = config['outputs']

    # --- Image data resides in a 'jpg' subfolder ---
    image_folder = os.path.join(run_folder_path, "jpg")

    # --- Record particle positions from sequence ---
    positions_df, long_exposure_image = record_particle_positions_from_sequence(run_folder_path)
    if long_exposure_image is None or positions_df.empty:
        print("[perform_motion_analysis] No valid frames or particles detected.")
        return pd.DataFrame()

    # --- Track particles and extract motion parameters ---
    trajectories = track_particles(positions_df)
    movement_variables = extract_movement_variables(trajectories)
    summary_df = summarize_movement_variables(movement_variables)

    # --- Output control ---
    if output_params.get('motion', False):
        filename = os.path.basename(run_folder_path.rstrip("/\\"))
        output_path = os.path.join(os.path.dirname(run_folder_path.rstrip("/\\")), f"{filename}_motion_analysis")
        os.makedirs(output_path, exist_ok=True)

        # --- Retrieve first frame from jpg subfolder ---
        first_frame = wim.get_frame_from_sequence(wim.get_image_file_list(image_folder), 0)
        if first_frame is None:
            print("[perform_motion_analysis] Warning: could not read first frame for visualization.")

        # --- Visualize trajectories ---
        image_with_tracks, tracks_only = visualize_trajectories(long_exposure_image, trajectories)

        # --- Save results ---
        positions_df.to_csv(os.path.join(output_path, 'particle_positions.csv'), index=False)
        cv2.imwrite(os.path.join(output_path, 'long_exposure_image.jpg'), long_exposure_image)
        if first_frame is not None:
            cv2.imwrite(os.path.join(output_path, 'first_frame.jpg'), first_frame)
        cv2.imwrite(os.path.join(output_path, 'long_exposure_image_with_tracks.jpg'), image_with_tracks)
        cv2.imwrite(os.path.join(output_path, 'tracks.jpg'), tracks_only)
        movement_variables.to_csv(os.path.join(output_path, 'movement_by_trajectory.csv'), index=False)
        summary_df.to_csv(os.path.join(output_path, 'summary_motion_analysis.csv'), index=False)

        print(f"[perform_motion_analysis] Motion analysis outputs saved to: {output_path}")
    else:
        print("[perform_motion_analysis] Motion outputs disabled in configuration.")

    return summary_df

