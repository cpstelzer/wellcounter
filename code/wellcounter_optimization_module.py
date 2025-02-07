# -*- coding: utf-8 -*-
"""
Wellcounter optimization module

This software is part of the following publication:
"Wellcounter: Automated High-Throughput Phenotyping for Aquatic Microinvertebrates"
Methods in Ecology and Evolution

The latest version can be found at https://github.com/cpstelzer/wellcounter


Description:
This module is designed to automate the optimization of imaging parameters 
for detecting microorganisms recorded with the WELLCOUNTER. 

The module includes functions for reading and writing the WELLCOUNTER 
configuration file, backing up and restoring configurations, calculating 
performance metrics, and performing batch optimization of imaging parameters. 

Note: Optimization parameters, e.g., the path to the training data, have to be 
set in the WELLCOUNTER configuration file (wellcounter_config.yml)

Note: Portions of the code in this file were generated using ChatGPT v4.0.
      All AI-generated content has been rigorously validated and tested by the 
      authors. The corresponding author accepts full responsibility for the 
      AI-assisted portions of the code.

Author: Claus-Peter Stelzer
Date: 2025-02-07

"""

import pandas as pd
import os
import numpy as np
import yaml
import tempfile
import shutil
import wellcounter_imaging_module as wim
import logging
from datetime import date   

# Get the current date
current_date = date.today().strftime("%Y%m%d")

# Set up logging
logging.basicConfig(filename=f'optimize_imaging_parameters_{current_date}_logfile.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

def read_config(config_path="wellcounter_config.yml"):
    """
    Function to read the config file.
    """
    try:
        with open(config_path, "r") as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        return config
    except Exception as e:
        logging.error(f"Error reading config file: {e}")
        raise

def write_config(config, config_path="wellcounter_config.yml"):
    """
    Function to write the config file.
    """
    try:
        with open(config_path, "w") as config_file:
            yaml.dump(config, config_file)
    except Exception as e:
        logging.error(f"Error writing config file: {e}")
        raise

def backup_config(config_path="wellcounter_config.yml"):
    try:
        config = read_config(config_path)
        # Open the temporary file in text mode
        with tempfile.NamedTemporaryFile(mode='w+', delete=False) as temp_file:
            yaml.dump(config, temp_file)
        return temp_file.name
    except Exception as e:
        logging.error(f"Error backing up config: {e}")
        raise


def restore_config(backup_file, config_path="wellcounter_config.yml"):
    try:
        shutil.copy(backup_file, config_path)
        os.remove(backup_file)
    except Exception as e:
        logging.error(f"Error restoring config: {e}")
        raise
        
   

def calculate_performance_metrics(df):
    """
    Calculate various performance metrics of particle detection in an image sample.
    
    Note: These calculations do not include true negatives and all derived measures 
    (e.g., accuracy) since true negatives are not relevant to the used dataset (WELLCOUNTER).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing all particles in a sample, detected by a certain method (query) 
        compared to the actual particles (reference). Must contain 'in_ref', 'in_query', 
        and 'particle_type' columns with Boolean values indicating particle detection status.

    Returns
    -------
    pandas.DataFrame
        DataFrame containing performance metrics like confusion matrix (excluding TN), sensitivity, etc.
    """
    
    if not {'in_ref', 'in_query'}.issubset(df.columns):
        raise ValueError("DataFrame must contain 'in_ref', 'in_query' columns")

    
    # Calculate confusion matrix elements, excluding true negatives
    true_positives = np.sum((df['in_ref']) & (df['in_query']))
    false_positives = np.sum(~df['in_ref'] & df['in_query'])
    false_negatives = np.sum(df['in_ref'] & ~df['in_query'])

    # Handle cases with zero denominators
    sensitivity = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0

    # Create a DataFrame for metrics, excluding TN, specificity, and npv
    df_metrics = pd.DataFrame({
        'TP': [true_positives],
        'FP': [false_positives],
        'FN': [false_negatives],
        'sensitivity': [sensitivity],
        'precision': [precision],
        'f1_score': [f1_score]
    })

    print("Performance_metrics:")
    print(df_metrics)
    print("")
    print("-----------------")
    print("")

    return df_metrics


def evaluate_imaging_parameters_OLD(training_path='E:/'):
    """
    Assess the effects of two imaging parameters (microorganism_threshold, min_microorganism_area) on 
    female detection performance metrics. Iterates through a curated training dataset.

    Parameters
    ----------
    microorganism_threshold: int
        Intensity threshold value of pixel (range: 0-255; default: 30)
    min_microorganism_area: int
        Contour area of detected particle (default: 150 pixels)
    training_path: str
        Main path to the dataset
    config_path: str
        Path to the configuration file

    Returns
    -------
    DataFrame
        A DataFrame containing the performance metrics.
    """
    
    
    
    
    try:
               
        # Load training data
        random_sample = pd.read_csv(os.path.join(training_path, 'random_sample_of_combined_data.csv'))
        comparison_df = pd.DataFrame()

        for _, row in random_sample.iterrows():
            # Extract values from the DataFrame
            experiment, date, batch, plate, well = row['experiment'], row['date'], row['batch'], row['plate'], row['well']
            
            # Construct path to the video
            sample_path = os.path.join(training_path, 'trainingdata', experiment, f'{date}_batch{batch}_plate{plate}_well{well}')
            video_path = os.path.join(sample_path, 'raw', f'{date}_batch{batch}_plate{plate}_well{well}.mp4')
            
            # Perform image analysis with the current parameter settings
            table_of_particles = wim.image_analysis_of_sample(video_path)
            
            # Read list of true number of females in current sample
            validated_particles = pd.read_csv(os.path.join(sample_path, 'joined_particle_list', 'all_females.csv'))
                        
            # Match particles detected by image analysis to "true" particles by their position
            comparison_particles = wim.compare_detected_particles(validated_particles, table_of_particles)
            
            # Collect data across different samples
            comparison_df = pd.concat([comparison_df, comparison_particles], ignore_index=True) if not comparison_df.empty else comparison_particles
            
            print(comparison_df)
             
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(comparison_df)
        
        # Generate screen output during run
        print("Checkpoint_1:")
        print(performance_metrics)
        
        # Read the config file
        config = read_config()
        print("Checkpoint_2:")
        print(config)
        
        particle_detection_params = config['particle_detection']
        
        print("Checkpoint_3:")
        print(particle_detection_params)
        
        performance_metrics['microorganism_threshold'] = particle_detection_params['microorganism_threshold']
        performance_metrics['min_microorganism_area'] = particle_detection_params['min_microorganism_area']
        
        
 
        return performance_metrics, comparison_df

    except Exception as e:
        logging.error(f"Error in evaluate_imaging_parameters: {e}")
        raise

def evaluate_imaging_parameters(training_path='E:/'):
    """
    Assess the effects of two imaging parameters (microorganism_threshold, min_microorganism_area) on 
    female detection performance metrics. Iterates through a curated training dataset.

    Parameters
    ----------
    microorganism_threshold: int
        Intensity threshold value of pixel (range: 0-255; default: 30)
    min_microorganism_area: int
        Contour area of detected particle (default: 150 pixels)
    training_path: str
        Main path to the dataset
    config_path: str
        Path to the configuration file

    Returns
    -------
    DataFrame
        A DataFrame containing the performance metrics.
    """
    
    
    
    
    try:
               
        # Get a list of all .mp4 video files in the directory
        video_files = [
            f for f in os.listdir(training_path) 
            if os.path.isfile(os.path.join(training_path, f)) and f.endswith('.mp4')
        ]
        
        comparison_df = pd.DataFrame()

       # Iterate through the list of video files
        for video_file in video_files:
            # Construct the full path to the video
            video_path = os.path.join(training_path, video_file)
            
            # Perform image analysis with the current parameter settings
            table_of_particles = wim.image_analysis_of_sample(video_path)
            
            # Navigate to and read the ground truth particle data
            # Remove the .mp4 extension from the video file name
            video_name = os.path.splitext(video_file)[0]     
            folder_name = f"{video_name}_particle_detection"
            
            # Construct the path to 'all_females.csv' within the folder
            csv_path = os.path.join(training_path, folder_name, 'all_females.csv')
            
            # Check if the file exists
            if not (os.path.exists(csv_path)):
                print(f"File not found: {csv_path}")
          
            validated_particles = pd.read_csv(csv_path)
                        
            # Match particles detected by image analysis to "true" particles by their position
            comparison_particles = wim.compare_detected_particles(validated_particles, table_of_particles)
            
            # Collect data across different samples
            comparison_df = pd.concat([comparison_df, comparison_particles], ignore_index=True) if not comparison_df.empty else comparison_particles
            
            #print(comparison_df)
             
        # Calculate performance metrics
        performance_metrics = calculate_performance_metrics(comparison_df)
        
       
        # Read the config file
        config = read_config()
        
        particle_detection_params = config['particle_detection']
        
        performance_metrics['microorganism_threshold'] = particle_detection_params['microorganism_threshold']
        performance_metrics['min_microorganism_area'] = particle_detection_params['min_microorganism_area']
        
        
 
        return performance_metrics, comparison_df

    except Exception as e:
        logging.error(f"Error in evaluate_imaging_parameters: {e}")
        raise
    
def batch_optimizer(config_path="wellcounter_config.yml"):
    
    """
    Optimizes imaging parameters.
    
    Reads optimization parameters from the config file and sequentially executes 
    `optimize_parameters` for different threshold values across a shared range of area values.
    
    Parameters
    ----------
    config_path : str
        Path to the configuration file.
    """
    
    # Function to append configuration to log
    def append_config_to_log():
        # Log a separator and some whitespace for readability
        logging.info("\n" + '-'*40 + "\n")
        
        # Convert the config dictionary to a YAML-formatted string
        formatted_config = yaml.dump(config, default_flow_style=False, sort_keys=False)
        
        # Log the formatted configuration
        logging.info("Configuration Settings:\n" + formatted_config)
        
        
    
    temp_file_path = backup_config()
    
    # Read the config file
    config = read_config(config_path)
    
    # Extract optimization parameters
    optimize_config = config.get('optimize', {})
    training_path = optimize_config.get('training_path', 'E:/')
    output_folder = optimize_config.get('output_folder', 'C:/')
    threshold_from = optimize_config.get('threshold_from', 4)
    threshold_to = optimize_config.get('threshold_to', 80)
    threshold_step = optimize_config.get('threshold_step', 2)
    area_from = optimize_config.get('area_from', 30)
    area_to = optimize_config.get('area_to', 400)
    area_step = optimize_config.get('area_step', 10)

    # Prepare ranges for optimization
    threshold_range = range(threshold_from, threshold_to + 1, threshold_step)
    area_range = range(area_from, area_to + 1, area_step)

    total_iterations = len(threshold_range) * len(area_range)
    iterations_completed = 0

    # Open or create the output files for writing
    outpath_csv_query = os.path.join(output_folder, f'optimize_imaging_parameters_{current_date}.csv')
    
    # Ensure the DataFrame is written to the file as it is being updated
    with open(outpath_csv_query, 'w') as outfile:
       

        for threshold in threshold_range:
            for area in area_range:
                try:
                    
                    # Load and update the config
                    config = read_config(config_path)
                    config["particle_detection"]["microorganism_threshold"] = threshold
                    config["particle_detection"]["min_microorganism_area"] = area
                    config["outputs"]["particle_detection"] = False         # Suppress particle detection outputs during optimization
                    write_config(config, config_path)
                    
                    print("Current image analysis parameters are: ")
                    print("Pixel threshold: ", threshold)
                    print("Microorganism area: ", area)
                    
                    # Evaluate parameters
                    performance_metrics_query, _ = evaluate_imaging_parameters(training_path) 
                    
                    # Determine whether to write headers: write if file does not exist or is empty
                    if not os.path.exists(outpath_csv_query) or os.path.getsize(outpath_csv_query) == 0:
                        header = True
                    else:
                        header = False
                    
                    # Open file, write data, and ensure headers are correctly managed
                    with open(outpath_csv_query, 'a', newline='') as outfile:  # 'a' mode for appending
                        performance_metrics_query.to_csv(outfile, index=False, header=header)
                    
                    iterations_completed += 1
                    progress_percentage = (iterations_completed / total_iterations) * 100
                    logging.info(f"Batch optimization progress: {progress_percentage:.2f}% completed")

                except Exception as e:
                    logging.error(f"Error in optimizing parameters at threshold {threshold} and area {area}: {e}")
                    print(f"Error in optimizing parameters at threshold {threshold} and area {area}: {e}")

    logging.info("Batch optimization completed and results saved.")    
    restore_config(temp_file_path)
    
    # Save wellcounter-configuration file to log
    append_config_to_log()   
    
if __name__ == "__main__":
    batch_optimizer(config_path="wellcounter_config.yml")
    pass
