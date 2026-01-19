import json
import numpy as np
import os
from pathlib import Path
import glob

from umift.utils.print_utils import color_print, info_print

class NumpyEncoder(json.JSONEncoder):
    # Encoder to handle numpy arrays and other non-serializable types
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)

def save_ft_data(dic, output_path, stage='pre_trim'):
    """
    Args:
        dic: dictionary containing ftTimestamp, ftData, and fileName
        output_path: path of dir where to save the JSON file
    """
    # Use the filename as part of the output file name
    json_filename = f"ft_data_{dic['meta']['session_name']}_{stage}.json"
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, json_filename)
    
    with open(full_path, 'w') as f:
        json.dump(dic, f, cls=NumpyEncoder, indent=4)
        
def save_campose_data(dic, output_path, stage='pre_trim'):
    """
    Args:
        dic: dictionary containing ftTimestamp, ftData, and fileName
        output_path: path of dir where to save the JSON file
    """
    # Use the filename as part of the output file name
    info_print(f"Saving data to: {output_path}")
    json_filename = f"cam_pose_data_{dic['meta']['session_name']}_{stage}.json"
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, json_filename)
    
    with open(full_path, 'w') as f:
        json.dump(dic, f, cls=NumpyEncoder, indent=4)
        
def save_img_data(dic, output_path, stage='pre_trim'):
    """
    Args:
        dic: dictionary containing ftTimestamp, ftData, and fileName
        output_path: path of dir where to save the JSON file
    """
    # Use the filename as part of the output file name
    info_print(f"Saving data to: {output_path}")
    json_filename = f"img_data_{dic['meta']['session_name']}_{stage}.json"
    os.makedirs(output_path, exist_ok=True)
    full_path = os.path.join(output_path, json_filename)
    
    with open(full_path, 'w') as f:
        json.dump(dic, f, cls=NumpyEncoder, indent=4)
    
def save_json(output_dir, data_dic, filename):
    """
    Args:
        output_dir: path to directory to save JSON file
        data_dic: dictionary containing data to save
        filename: name of the file to save
    Returns:
        None
    """
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {key: convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in list(obj)]
        return obj

    data_dic_converted = convert_numpy_types(data_dic)
    
    with open(os.path.join(output_dir, f'{filename}.json'), 'w') as f:
        json.dump(data_dic_converted, f, indent=4)

def load_processed_json(image_data_dir, side='right'):
   """
   Args:
       image_data_dir: path to directory containing processed_*.json files
       side: which side to load ('right' or 'left')
   Returns:
       json_data: dictionary containing loaded json data
       found: boolean indicating if the file was found and loaded successfully
   """
   # Convert to Path object and validate side parameter
   data_path = Path(image_data_dir)
   if side not in ['right', 'left']:
       raise ValueError("side must be either 'right' or 'left'")
   
   # Construct the filename
   target_file = data_path / f'{side}.json'
   
   # Check if file exists
   if not target_file.exists():
       print(f"No processed_{side}.json found in {image_data_dir}")
       return None, False
   # Load the json file
   try:
       with open(target_file, 'r') as f:
           json_data = json.load(f)
       print(f"Successfully loaded {side}.json")
       return json_data, True
   except json.JSONDecodeError as e:
       print(f"Error decoding {side}.json: {e}")
       return None, False
   except Exception as e:
       print(f"Unexpected error loading {side}.json: {e}")
       return None, False

def load_visual_data_json(image_data_dir, side='right'):
    """
    Args:
        image_data_dir: path to directory containing visual_data_*.json files
        side: which side to load ('right' or 'left')
    Returns:
        json_data: dictionary containing loaded json data
        found: boolean indicating if the file was found and loaded successfully
    Function: Load processed_*.json data after umi_day post processing
    """
    data, found = load_processed_json(image_data_dir, side) # data.keys(): dict_keys(['times', 'poseTransforms'])
    if found:
        # Process the data
        color_print(f"Processing {side} side data...", color='green')
        # Your processing code here
    else:
        color_print(f"Cannot process {side} side data - file not found or error loading", color='red')
        raise FileNotFoundError(f"Cannot process {side} side data - file not found or error loading")
    return data, found

def get_demonstration_dirs(base_dir):
   """
   Get all subdirctories that end with '_demonstration'
   Args:
       base_dir: str or Path, base directory to search in
   Returns:
       list of Path objects fore directories ending with '_demonstration'
   """
   base_path = Path(base_dir)
   
   # Find all subdirs ending with '_demonstration'
   demo_dirs = list(base_path.glob('**/*_demonstration'))
   
   # Filter to ensure we only get directories (not files)
   demo_dirs = [d for d in demo_dirs if d.is_dir()]
   
   print(f"Found {len(demo_dirs)} demonstration directories:")
   for demo_dir in demo_dirs:
       print(f"  - {demo_dir}")
   
   return demo_dirs