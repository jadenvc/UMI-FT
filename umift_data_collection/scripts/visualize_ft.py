import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os
from umift.utils.print_utils import color_print
from umift.utils.time_utils import convert_timestamp_to_iso

def read_force_torque_data(file_path):
    """
    Read force/torque sensor data from CSV file
    Returns a pandas DataFrame with processed timestamps and measurements
    """
    # Read CSV file
    df = pd.read_csv(file_path)
    
    # Rename columns for clarity
    df.columns = ['timestamp', 'force_x', 'force_y', 'force_z', 
                 'torque_x', 'torque_y', 'torque_z']
    
    # Convert timestamp to datetime
    # df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    
    df['timestamp'] = convert_timestamp_to_iso(df['timestamp'])

    color_print("\nFirst 5 timestamps:")
    print(df['timestamp'].head())
    
    exit()
        
    # Calculate time elapsed in seconds from start
    df['time_elapsed'] = (df['timestamp'] - df['timestamp'].iloc[0]).dt.total_seconds()
    
    return df

def analyze_force_torque(df):
    """
    Perform basic analysis on force/torque measurements
    """
    analysis = {
        'force': {
            'x': {
                'mean': df['force_x'].mean(),
                'std': df['force_x'].std(),
                'max': df['force_x'].max(),
                'min': df['force_x'].min()
            },
            'y': {
                'mean': df['force_y'].mean(),
                'std': df['force_y'].std(),
                'max': df['force_y'].max(),
                'min': df['force_y'].min()
            },
            'z': {
                'mean': df['force_z'].mean(),
                'std': df['force_z'].std(),
                'max': df['force_z'].max(),
                'min': df['force_z'].min()
            }
        },
        'torque': {
            'x': {
                'mean': df['torque_x'].mean(),
                'std': df['torque_x'].std(),
                'max': df['torque_x'].max(),
                'min': df['torque_x'].min()
            },
            'y': {
                'mean': df['torque_y'].mean(),
                'std': df['torque_y'].std(),
                'max': df['torque_y'].max(),
                'min': df['torque_y'].min()
            },
            'z': {
                'mean': df['torque_z'].mean(),
                'std': df['torque_z'].std(),
                'max': df['torque_z'].max(),
                'min': df['torque_z'].min()
            }
        },
        'sampling': {
            'duration': df['time_elapsed'].max(),
            'samples': len(df),
            'avg_sampling_rate': len(df) / df['time_elapsed'].max()
        }
    }
    
    return analysis

def plot_measurements(df):
    """
    Create plots of force and torque measurements over time
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot forces
    ax1.plot(df['time_elapsed'], df['force_x'], label='Force X')
    ax1.plot(df['time_elapsed'], df['force_y'], label='Force Y')
    ax1.plot(df['time_elapsed'], df['force_z'], label='Force Z')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Force')
    ax1.legend()
    ax1.grid(True)
    ax1.set_title('Force Measurements Over Time')
    
    # Plot torques
    ax2.plot(df['time_elapsed'], df['torque_x'], label='Torque X')
    ax2.plot(df['time_elapsed'], df['torque_y'], label='Torque Y')
    ax2.plot(df['time_elapsed'], df['torque_z'], label='Torque Z')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Torque')
    ax2.legend()
    ax2.grid(True)
    ax2.set_title('Torque Measurements Over Time')
    
    plt.tight_layout()
    return fig

# Example usage:
if __name__ == "__main__":
    # Read the data
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--coinFT_csv_path', type=str, default='result_241104_172718_first_multimodal.csv', help='csv file name of coinFT raw data')
    parser.add_argument('--coinFT_csv_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/data/data_wired_coinFT/coinFT-20241104', help='coinFT data directory')
    parser.add_argument('--output_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/output/coinFT_wired_vis', help='camera trajectory file (.csv)')
    parser.add_argument('--show', type=bool, default=True, help='show intermediate plot on machine or not, default: True')
    args = parser.parse_args()
    
    file_path = os.path.join(args.coinFT_csv_dir, args.coinFT_csv_path)
    df = read_force_torque_data(file_path)
    
    demo_header = '_'.join(args.coinFT_csv_path.split('.')[0].split('_')[1:3])
    color_print(f"Processing coinFT data for {demo_header}", color='cyan')
 
    # Perform analysis
    analysis = analyze_force_torque(df)
    
    # Print summary statistics
    color_print("\nForce/Torque Data Analysis", color='yellow', style='back')
    color_print("==========================",  color='yellow', style='fore')
    color_print(f"Duration: {analysis['sampling']['duration']:.2f} seconds",  color='yellow', style='fore')
    color_print(f"Number of samples: {analysis['sampling']['samples']}",  color='yellow', style='fore')
    color_print(f"Average sampling rate: {analysis['sampling']['avg_sampling_rate']:.2f} Hz",  color='yellow', style='fore')
    
    color_print("\nForce Statistics (X, Y, Z):")
    
    for axis in ['x', 'y', 'z']:
        color_print(f"\n{axis.upper()} axis:",  color='green', style='fore')
        color_print(f"  Mean: {analysis['force'][axis]['mean']:.6f}", color='green', style='fore')
        color_print(f"  Std:  {analysis['force'][axis]['std']:.6f}", color='green', style='fore')
        color_print(f"  Max:  {analysis['force'][axis]['max']:.6f}", color='green', style='fore')
        color_print(f"  Mi:  {analysis['force'][axis]['min']:.6f}", color='green', style='fore')
    
    # Create and show plots
    plot_measurements(df)
    plt.savefig(os.path.join(args.output_dir, f"ft_{demo_header}.png"))
    if args.show:
        plt.show()
    plt.close()