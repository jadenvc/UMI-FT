import os 
import numpy as np
import pandas as pd

def read_and_save_force_torque_data(input_file_path, output_file_path=None):
    """
    Read force/torque sensor data from CSV file, process it, and save to a new CSV file
    
    Parameters:
    input_file_path (str): Path to the input CSV file
    output_file_path (str): Path for the output CSV file. If None, will create a name based on input file
    
    Returns:
    pandas DataFrame: Processed force/torque data
    """
    # Read CSV file
    df = pd.read_csv(input_file_path)
    
    # Rename columns for clarity
    df.columns = ['timestamp', 'force_x', 'force_y', 'force_z', 
                 'torque_x', 'torque_y', 'torque_z']
    
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    
    # Calculate time elapsed in seconds from start
    df['time_elapsed'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    
    # If no output path specified, create one based on input filename
    if output_file_path is None:
        # Get the directory and filename from the input path
        input_dir = os.path.dirname(input_file_path)
        input_filename = os.path.basename(input_file_path)
        # Create new filename by adding '_processed' before the extension
        base_name = os.path.splitext(input_filename)[0]
        output_file_path = os.path.join(input_dir, f"{base_name}_processed.csv")
    
    # Save to new CSV file
    df.to_csv(output_file_path, index=False)
    print(f"Processed data saved to: {output_file_path}")
    
    return df

