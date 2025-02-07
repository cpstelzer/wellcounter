# -*- coding: utf-8 -*-
"""
Wellcounter motion module

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter

Description:
A Python module containing functions for reviewing detected particles one by one,
assigning particle type categories, labeling them in an image, and detecting false negatives.

As an example of using this module, see the script wc_assign_particletypes.py.

Note: Portions of the code in this file were generated using ChatGPT v4.0.
      All AI-generated content has been rigorously validated and tested by the 
      authors. The corresponding author accepts full responsibility for the 
      AI-assisted portions of the code.

Author: Claus-Peter Stelzer
Date: 2025-02-07
"""

import cv2
import numpy as np
import tkinter as tk
import pandas as pd

# Define particle types with updated categories
ptype1 = "true_moving"     # Dark green
ptype2 = "true_stationary" # Light green
ptype3 = "false_positive"  # Red
ptype4 = "false_negative"  # Cyan

# Updated color definitions for all categories
type_colors = {
    1: (35, 140, 35),    # Dark green
    2: (100, 240, 100),  # Light green
    3: (0, 0, 255),      # Red
    4: (255, 255, 0)     # Cyan
}

def label_particletypes(image_path, table_of_particles_cat):
    """
    Annotates an image with labeled particle types, including false negatives.
    """
    image = cv2.imread(image_path)

    for index, row in table_of_particles_cat.iterrows():
        x, y, area, particle_type = int(row['X']), int(row['Y']), int(row['area']), int(row['particle_type'])
        if area > 500:
            radius = int(2 * np.sqrt(area / np.pi))
        else:
            radius = int(2 * np.sqrt(500 / np.pi))
                
        color = type_colors.get(particle_type, (0, 0, 0))
        cv2.circle(image, (x, y), radius, color, thickness=3)

    # Updated legend text
    legend_texts = ['Particle type:', f'1: {ptype1}', f'2: {ptype2}', f'3: {ptype3}', f'4: {ptype4}']
    font_scale = 2
    font_thickness = 3
    line_height = int(cv2.getTextSize('1', cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][1] * 1.5)
    legend_height = len(legend_texts) * line_height

    legend_position = (image.shape[1] - 600, image.shape[0] - 80 - legend_height)

    for i, text in enumerate(legend_texts):
        y_offset = legend_position[1] + (i * line_height)
        color = (255, 255, 255) if i == 0 else type_colors.get(i, (0, 0, 0))
        cv2.putText(image, text, (legend_position[0], y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, font_thickness)

    output_path = image_path.replace('.jpg', '_typelabels.jpg')
    cv2.imwrite(output_path, image)
    print(f"Marked image saved as {output_path}")

def crop_and_play_video(input_video_path, x, y, particle_area):
    """
    Crops a video around a particle with color-matched ring.
    """
    cap = cv2.VideoCapture(input_video_path)
    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    magnification_factor = 2
    crop_size = 128
    
    x_start = max(0, x - crop_size // 2)
    y_start = max(0, y - crop_size // 2)
    x_end = min(frame_width, x_start + crop_size)
    y_end = min(frame_height, y_start + crop_size)

    ring_radius = int(magnification_factor * 4 * np.sqrt(particle_area / np.pi))
    ring_thickness = 2

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        cropped_frame = frame[y_start:y_end, x_start:x_end]
        resized_frame = cv2.resize(cropped_frame, (crop_size * magnification_factor, crop_size * magnification_factor))
        ring_canvas = np.zeros_like(resized_frame)

        center_x = int(magnification_factor * (x - x_start))
        center_y = int(magnification_factor * (y - y_start))

        # Use false positive color (red) for the ring
        cv2.circle(ring_canvas, (center_x, center_y), ring_radius, type_colors[3], ring_thickness)
        result_frame = cv2.addWeighted(resized_frame, 1, ring_canvas, 1, 0)

        cv2.imshow("Cropped Video with Ring", result_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Video cropping and display with ring finished")

def on_button_click(option, root):
    root.option = option
    root.destroy()

def get_user_input():
    """Modified to show only 3 classification options"""
    root = tk.Tk()
    root.title("Please classify the particle:")
    
    screen_width = root.winfo_screenwidth()
    screen_height = root.winfo_screenheight()
    window_width = 300
    window_height = 200
    x_position = screen_width - window_width - 300
    y_position = screen_height - window_height - 300
    root.geometry(f"{window_width}x{window_height}+{x_position}+{y_position}")

    # Only three buttons now
    tk.Button(root, text=ptype1, command=lambda: on_button_click(1, root)).pack()
    tk.Button(root, text=ptype2, command=lambda: on_button_click(2, root)).pack()
    tk.Button(root, text=ptype3, command=lambda: on_button_click(3, root)).pack()

    root.mainloop()
    return root.option if hasattr(root, 'option') else None

def user_categorize_particles(video_path, table_of_particles):
    """Updated with new classification instructions"""
    input("\nNew classification workflow:\n"
          "1. Watch each particle's movement\n"
          "2. Classify as:\n"
          "   - 1: True moving\n"
          "   - 2: True stationary\n"
          "   - 3: False positive\n"
          "Press Enter to start...\n")

    particle_types = []
    for index, row in table_of_particles.iterrows():
        x_coordinate = int(row['X'])
        y_coordinate = int(row['Y'])
        particle_area = int(row['area'])
        crop_and_play_video(video_path, x_coordinate, y_coordinate, particle_area)
        particle_type = get_user_input()
        particle_types.append(particle_type)
        print(f"Particle at index {index} classified as type: {particle_type}")

    df = table_of_particles.copy()
    df['particle_type'] = particle_types
    print(df.head())
    return df

# New false negative detection functions
def get_quadrants(image):
    """
    Divides the image into four overlapping quadrants.
    
    Each quadrant overlaps its adjacent neighbor by 2% of the full image dimension.
    For instance, Q1 (upper left) extends 2% of the full width to the right and 2%
    of the full height downward, thus including a bit of Q2 (upper right) and Q3 
    (lower left). Similar adjustments are made for the other quadrants.
    
    Returns:
        A list of tuples (x1, y1, x2, y2) corresponding to the quadrant coordinates.
    """
    h, w = image.shape[:2]
    # Calculate the overlap in pixels as 2% of the full image dimensions.
    delta_x = int(0.02 * w)
    delta_y = int(0.02 * h)
    
    # Compute the midpoints.
    mid_x = w // 2
    mid_y = h // 2
    
    # Define quadrants with overlap.
    q1 = (0, 0, mid_x + delta_x, mid_y + delta_y)                # Upper left
    q2 = (mid_x - delta_x, 0, w, mid_y + delta_y)                 # Upper right
    q3 = (0, mid_y - delta_y, mid_x + delta_x, h)                 # Lower left
    q4 = (mid_x - delta_x, mid_y - delta_y, w, h)                 # Lower right
    
    return [q1, q2, q3, q4]

def review_quadrant(video_path, quadrant, full_image, categorized_data):
    """
    Reviews a quadrant of the image by animating a sequence of frames from the video,
    then allows the user to mark false negatives. The display windows are sized dynamically
    to maintain the quadrant’s aspect ratio and are centered on a full HD (1920x1080) screen.
    The animation shows frames at the first, 25%, 50%, 75%, and last positions of the video,
    with the first frame shown for 500ms and subsequent frames for 100ms.
    """
    # Validate required column
    if 'particle_type' not in categorized_data.columns:
        raise ValueError("Input DataFrame must contain 'particle_type' column")

    # Unpack quadrant coordinates
    x1, y1, x2, y2 = quadrant

    # Open video capture
    cap = cv2.VideoCapture(video_path)
    false_negatives = []

    # Get total frames and calculate percentage-based frame indices
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_indices = [
        0,  # First frame
        max(1, int(total_frames * 0.25) - 1),
        max(1, int(total_frames * 0.5) - 1),
        max(1, int(total_frames * 0.75) - 1),
        max(1, total_frames - 1)  # Last frame
    ]

    # Create a base quadrant image (used later in marking phase)
    quadrant_base = full_image[y1:y2, x1:x2].copy()
    for _, row in categorized_data.iterrows():
        x = int(row['X'])
        y = int(row['Y'])
        if x1 <= x <= x2 and y1 <= y <= y2:
            radius = int(2 * np.sqrt(row['area'] / np.pi))
            color = type_colors.get(row['particle_type'], (0, 0, 0))
            cv2.circle(quadrant_base, (x - x1, y - y1), radius, color, 2)

    # Compute dynamic display dimensions based on the quadrant's size.
    quadrant_width = x2 - x1
    quadrant_height = y2 - y1
    display_height = 800
    display_width = int((quadrant_width / quadrant_height) * display_height)
    display_size = (display_width, display_height)

    # Compute window position to center on a 1920x1080 screen.
    screen_width, screen_height = 1920, 1080
    pos_x = int((screen_width - display_size[0]) / 2)
    pos_y = int((screen_height - display_size[1]) / 2)

    # ----- Animation Phase: Preview the Quadrant -----
    cv2.namedWindow('Quadrant Review', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Quadrant Review', display_size[0], display_size[1])
    cv2.moveWindow('Quadrant Review', pos_x, pos_y)
    quit_pressed = False

    while not quit_pressed:
        for i, idx in enumerate(frame_indices):
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                continue

            # Crop the frame to the quadrant
            frame_copy = frame[y1:y2, x1:x2].copy()

            # Draw the already-categorized particles onto the cropped frame
            for _, row in categorized_data.iterrows():
                x = int(row['X'])
                y = int(row['Y'])
                if x1 <= x <= x2 and y1 <= y <= y2:
                    radius = int(2 * np.sqrt(row['area'] / np.pi))
                    color = type_colors.get(row['particle_type'], (0, 0, 0))
                    cv2.circle(frame_copy, (x - x1, y - y1), radius, color, 2)

            # Add instructional text
            text_scale = 1.2
            text_thickness = 2
            text_color = (0, 255, 255)  # Yellow
            cv2.putText(frame_copy, "Look for missing particles", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)
            cv2.putText(frame_copy, "Press Q to start marking", (20, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, text_scale, text_color, text_thickness)

            # Set delay: 500ms for first frame, 100ms for others
            delay = 500 if i == 0 else 100
            # Resize the cropped frame to our dynamic display size
            resized = cv2.resize(frame_copy, display_size)
            cv2.imshow('Quadrant Review', resized)

            if cv2.waitKey(delay) & 0xFF == ord('q'):
                quit_pressed = True
                break

        # Allow forced exit if the window is closed
        if cv2.getWindowProperty('Quadrant Review', cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyWindow('Quadrant Review')

    # ----- Marking Phase: Identify False Negatives -----
    # Prepare a marking image with the same base quadrant image
    marking_image = quadrant_base.copy()
    cv2.putText(marking_image, "Click missed particles", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)
    cv2.putText(marking_image, "Press SPACE when done", (20, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2)

    # Resize the marking image to the computed display size for consistency
    resized_marking = cv2.resize(marking_image, display_size)
    cv2.namedWindow('Mark False Negatives', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Mark False Negatives', display_size[0], display_size[1])
    cv2.moveWindow('Mark False Negatives', pos_x, pos_y)
    cv2.imshow('Mark False Negatives', resized_marking)

    def click_handler(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            # Compute scaling factors using the display dimensions
            scale_x = (x2 - x1) / display_size[0]
            scale_y = (y2 - y1) / display_size[1]
            # Convert display coordinates to absolute image coordinates
            abs_x = int(x1 + x * scale_x)
            abs_y = int(y1 + y * scale_y)
            false_negatives.append((abs_x, abs_y))

            # Provide visual feedback by drawing the marker on the resized marking image
            cv2.circle(resized_marking, (x, y), 8, type_colors[4], -1)
            cv2.putText(resized_marking, f"{abs_x},{abs_y}", (x + 10, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow('Mark False Negatives', resized_marking)

    cv2.setMouseCallback('Mark False Negatives', click_handler)

    # Wait until the user presses the SPACE key or closes the window
    while True:
        key = cv2.waitKey(0)
        if key == ord(' '):
            break
        if cv2.getWindowProperty('Mark False Negatives', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    return false_negatives

def detect_false_negatives(video_path, first_frame_path, categorized_data):
    """Updated with input validation"""
    # Validate input DataFrame structure
    required_columns = ['X', 'Y', 'area', 'particle_type']
    missing = [col for col in required_columns if col not in categorized_data.columns]
    
    if missing:
        raise ValueError(f"Input DataFrame missing required columns: {missing}")
    first_frame = cv2.imread(first_frame_path)
    all_quadrants = get_quadrants(first_frame)
    false_negatives = []
    
    cap = None  # Initialize cap to None to avoid reference issues
    try:
        for q_idx, quadrant in enumerate(all_quadrants):
            print(f"Reviewing quadrant {q_idx+1}/4")
            quadrant_fn = review_quadrant(video_path, quadrant, first_frame, categorized_data)
            false_negatives.extend(quadrant_fn)

            # Update visualization with escape check
            for x, y in quadrant_fn:
                cv2.circle(first_frame, (x, y), 10, type_colors[4], 3)
            
            cv2.imshow('Progress Review', cv2.resize(first_frame, (800, 600)))
            if cv2.waitKey(500) == ord('q'):
                break
            cv2.destroyAllWindows()
    finally:
        if cap is not None:  # Only release if cap was successfully created
            cap.release()
        cv2.destroyAllWindows()

    # Save results
    output_path = first_frame_path.replace('.jpg', '_final.jpg')
    cv2.imwrite(output_path, first_frame)
    
    fn_df = pd.DataFrame(false_negatives, columns=['X', 'Y'])
    fn_df['particle_type'] = 4
    fn_df['area'] = 10
    
    return pd.concat([categorized_data, fn_df], ignore_index=True)