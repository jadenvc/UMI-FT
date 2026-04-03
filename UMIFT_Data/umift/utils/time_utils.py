# Time format conversion
# Author: Chuer Pan

from datetime import datetime, timezone, timedelta
import matplotlib.pyplot as plt
import numpy as np
import os, av
from typing import Union
from fractions import Fraction
 
def array_convert_timestamp_to_iso(timestamp_array):
    """
    Converts an array of ISO 8601 strings to an array of float timestamps.
    
    Args:
        iso_strings (list of str): Array of timestamps in ISO 8601 string format.

    Returns:
        list of float: Array of timestamps as float.
    """
    return [
        datetime.fromtimestamp(float(timestamp) if isinstance(timestamp, str) else timestamp).astimezone(timezone.utc).replace(tzinfo=None).isoformat()
        for timestamp in timestamp_array
    ]

# Assumption: right now, all time is preserved up to microseconds (6 decimal places) precision, if need more precision, look into float -> Decimal conversion
def convert_timestamp_to_iso_processed(timestamp):
    """
    ouptut format is the same as in processed_{}.csv
    2024-11-05T02:33:20.681833 
    Args:
        ntp_time: str or float, NTP time in string format
    """
  
    if isinstance(timestamp, str):
        timestamp = float(timestamp)
    return datetime.fromtimestamp(timestamp).astimezone(timezone.utc).replace(tzinfo=None).isoformat()


def convert_timestamp_to_iso_z_format(timestamp):
    """
    Converts float timestamp to ISO8601 string with milliseconds and 'Z' UTC suffix.
    """
    if isinstance(timestamp, str):
        timestamp = float(timestamp)
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat(timespec='milliseconds').replace('+00:00', 'Z')



def isostringformat_to_timestamp(iso_format_time):
    """
    Convert ARKit time to timestamp (iso_format_time)
    Args:
        iso_format_time: str, ARKit time in ISO format
    """
    # Handle 'Z' suffix (UTC indicator) which fromisoformat doesn't support before Python 3.11
    if iso_format_time.endswith('Z'):
        iso_format_time = iso_format_time[:-1] + '+00:00'
    return datetime.fromisoformat(iso_format_time).timestamp()

def ntp_time_to_timestamp(ntp_time):
    """
    Convert NTP time to timestamp with datetime conversion and consistent precision
    Args:
        ntp_time: str or float, NTP time in string format
    Returns:
        float: timestamp with consistent precision
    """
    if isinstance(ntp_time, str):
        ntp_time = float(ntp_time)
    # Convert to datetime and back, maintaining precision
    # print(
    #     'ntp_time: ', ntp_time
    # )

    dt = datetime.fromtimestamp(ntp_time)
    return float(format(dt.timestamp(), '.18f'))

def ntp_time_to_timestamp_direct(ntp_time):
    """
    Convert NTP time to timestamp with consistent precision;
    eg: "1.730769830135123968e+09"
    Args:
        ntp_time: str or float, NTP time in string format
    Returns:
        float: timestamp with consistent precision
    """
    if isinstance(ntp_time, str):
        # Convert string to float with full precision
        ntp_time = format(float(ntp_time), '.18f')  # maintain 18 decimal places
        return float(ntp_time)
    else:
        # If it's already a float, format it to maintain precision
        return float(format(ntp_time, '.18f'))

def array_isostringformat_to_timestamp(X):
    '''
    Convert an array of ISO formatted string to timestamp
    '''
    return np.array([isostringformat_to_timestamp(x) for x in X])

def array_ntp_time_to_timestamp(X):
    '''
    Convert an array of NTP time to timestamp
    '''
    return np.array([ntp_time_to_timestamp(x) for x in X])

def plot_time_sequences(times, labels=None, output_path=None, img_file_name='timestamp_comparison'):
    """
    Plot multiple sequences of timestamps as 1D scatter plots
    Args:
        times: list of lists, where each inner list contains timestamps in ISO format
        labels: list of labels for legend, one for each sequence
        output_path: path to save the figure
        img_file_name: name of the output image file
    Returns:
        fig, ax: matplotlib figure and axis objects
    """
    # Convert ISO times to timestamps for each sequence
    timestamps = [[isostringformat_to_timestamp(t) for t in seq] for seq in times]
    
    # Create default labels if none provided
    if labels is None:
        labels = [f'Sequence {i+1}' for i in range(len(times))]
    
    # Create figure and axis
    fig, ax = plt.subplots(figsize=(15, 3))
    
    # Calculate y positions for each sequence
    num_sequences = len(times)
    y_positions = np.linspace(0, 1, num_sequences)
    
    # Plot each sequence
    for i, (ts, label) in enumerate(zip(timestamps, labels)):
        ax.scatter(ts, np.ones_like(ts) * y_positions[i], 
                  alpha=0.6, label=label)
    
    # Format x-axis to show readable times
    plt.gcf().autofmt_xdate()
    
    # Set y-axis limits and hide y-ticks
    ax.set_ylim(-0.1, 1.1)
    ax.set_yticks([])
    
    # Add legend
    ax.legend()
    
    # Add grid for better readability
    ax.grid(True, axis='x', alpha=0.3)
    
    # Set title and labels
    ax.set_title('Timestamp Comparison')
    ax.set_xlabel('Time')
    
    if output_path:
        plt.savefig(os.path.join(output_path, f'{img_file_name}.png'), 
                    dpi=300, bbox_inches='tight')
    plt.close()
 

def find_timestamp_key(data_dict, demo_idx):
    if 'data' not in data_dict.keys():
        demo_keys = data_dict['data']['left'][demo_idx].keys() # handle the left right for gripper
    else:
        demo_keys = data_dict['data'][demo_idx].keys()
    
    
    # Find the key that ends with 'TimeStamp'
    timestamp_keys = [key for key in demo_keys if key.endswith('TimeStamp')]
    
    if timestamp_keys:
        return timestamp_keys[0]  # Return the first matching key
    return None

def plot_time_sequence_data_dict(data_dict_list, labels=None, demo_idx = 0, output_path=None, img_file_name='timestamp_comparison'):
    times_list = []
    for data_dict in data_dict_list:
        
        timestamp_key_name = find_timestamp_key(data_dict, demo_idx)
        if 'data' not in data_dict.keys():
            times_list.append(data_dict['data']['left'][demo_idx][timestamp_key_name])
        else:   
            times_list.append(data_dict['data'][demo_idx][timestamp_key_name])
    img_file_name=f'{img_file_name}_{demo_idx}'
    return plot_time_sequences(times_list, labels, output_path, img_file_name)


def timecode_to_seconds(
        timecode: str, frame_rate: Union[int, float, Fraction]
        ) -> Union[float, Fraction]:
    """
    Convert non-skip frame timecode into seconds since midnight
    """
    # calculate whole frame rate, rounded up to the nearest integer
    # Eg. 29.97 -> 30, 59.94 -> 60
    int_frame_rate = round(frame_rate)

    # parse timecode string
    h, m, s, f = [int(x) for x in timecode.split(':')]

    # calculate frames assuming whole frame rate (i.e. non-drop frame)
    frames = (3600 * h + 60 * m + s) * int_frame_rate + f

    # convert to seconds
    seconds = frames / frame_rate
    return seconds


def stream_get_start_datetime(stream: av.stream.Stream) -> datetime:
    """
    Combines creation time and timecode to get high-precision
    time for the first frame of a video.
    """
    # read metadata
    frame_rate = stream.average_rate
    tc = stream.metadata['timecode']
    creation_time = stream.metadata['creation_time']
    
    # get time within the day
    seconds_since_midnight = float(timecode_to_seconds(timecode=tc, frame_rate=frame_rate))
    delta = timedelta(seconds=seconds_since_midnight)
    
    # get dates
    create_datetime = datetime.strptime(creation_time, r"%Y-%m-%dT%H:%M:%S.%fZ")
    create_datetime = create_datetime.replace(hour=0, minute=0, second=0, microsecond=0)
    start_datetime = create_datetime + delta
    return start_datetime


def mp4_get_start_datetime(mp4_path: str) -> datetime:
    with av.open(mp4_path) as container:
        stream = container.streams.video[0]
        return stream_get_start_datetime(stream=stream)
