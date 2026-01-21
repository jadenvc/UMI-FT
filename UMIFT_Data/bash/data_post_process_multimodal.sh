#!/bin/bash
# NOTE: conda activate umiFT
# Usage: <umift_data_parent_directory>$ bash bash/data_post_process_multimodal.sh

# get the current directory
umift_data_parent_dir=$(pwd)

###### TODO: CHANGE THESE VARS FOR LOCAL SETUP ######
raw_umi_data_dir=/store/real/hjchoi92/data/real/umift # (USER TODO) change this to the directory where the raw umi iphone data are saved on your computer
session_name="zucchini-wild-b1" # (USER TODO) change this to the session name of the raw umi iphone data
gripper_side="left" # (USER TODO) change this to the gripper side used in the session. left is default.
###### TODO: CHANGE THESE VARS FOR LOCAL SETUP ######


umi_data_folder=$raw_umi_data_dir/$session_name 
processed_output_dir_session=$umi_data_folder/processed_data/iphone # this is where the processed gopro and iphone data will be saved
multimodal_data_output_dir=$umi_data_folder/processed_data/all # this is where the multimodal final data will be saved
intermediate_umi_output_dir=$umi_data_folder/processed_data/temp # this is where the intermediate umi data will be saved for gripper width and gripper calibration
umi_submodule_dir=$umift_data_parent_dir/submodules/universal_manipulation_interface
umi_day_submodule_dir=$umift_data_parent_dir/submodules/umi_day
calibration_dir=$umift_data_parent_dir/gripper_calibration

# Folder Struction Assumption:
# <raw_umi_data_dir>/
# ├── <session_name>/
# │   ├── UMI_iphone/
# │   │   ├── export_<YYYY-MM-DD_TIME>/
# │   │   │   ├── <TIME>_<side>.json
# │   │   │   ├── ...
# │   ├── coinft/
# │   │   ├── YYYY-MM-DD/
# │   │   │   ├── <session_name>_<TIME>_LF.csv
# │   │   │   ├── ...
## (output from data processing):
# │   ├── processed_data/ (output from data processing)
# │   │   ├── iphone/
# │   │   ├── all/

iphone_data_dir=$umi_data_folder/UMI_iPhone
ft_data_dir=$umi_data_folder/coinft

echo 'iphone_dir' $iphone_data_dir
echo 'out_dir' $processed_output_dir_session
echo 'session_name' $session_name

# assert the directory exists
if [ ! -d $raw_umi_data_dir ]; then
    echo "Error: $raw_umi_data_dir does not exist"
    exit 1
fi

if [ ! -d $iphone_data_dir ]; then
    echo "Error: iphone data dir $iphone_data_dir does not exist"
    exit 1
fi

export CUDA_VISIBLE_DEVICES=0
# Step 2: extract the data from  the iphone and gopro
export PYTHONPATH=$(pwd)/../umi_day:$PYTHONPATH
export PYTHONPATH=$(pwd)/../PyriteUtility:$PYTHONPATH
python scripts/process_multimodal.py \
    --ft_data_dir $ft_data_dir \
    --visual_data_dir $processed_output_dir_session \
    --intermediate_umi_session_dir $intermediate_umi_output_dir \
    --umi_submodule_dir $umi_submodule_dir \
    --umi_day_submodule_dir $umi_day_submodule_dir\
    --gripper_side $gripper_side \
    --output_dir $multimodal_data_output_dir \
    --calibration_dir $calibration_dir \
    --output_rgb_w 224 \
    --output_rgb_h 224 \
    --plot \
    --plot_horizontal
 