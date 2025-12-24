#!/bin/bash
# NOTE: conda activate umi_day
# Usage: <umift_parent_directory>$ bash bash/data_post_process_gopro_iphone.sh

umift_parent_dir=$(pwd)

###### TODO: CHANGE THESE VARS FOR LOCAL SETUP ######
raw_umi_data_dir=/store/real/hjchoi92/data/real/umift # (USER TODO) change this to the directory where the raw umi gopro+iphone data are saved on your computer
session_name="WBW-iph-b5" # (USER TODO) change this to the session name of the raw umi gopro+iphone data
skip_stages="visualize" # (USER TODO) option \in {"visualize", "", "group", "gopro_timesync", "align"}. change this to "visualize" if you want to skip the visualization stage, or "" to run all stages including visualize 
flatten_coinft_folder_structure=True # (USER TODO) option \in {True, False}. change this to "True" if this is the first time you are running this script for this session, or "False" if you already done so
###### TODO: CHANGE THESE VARS FOR LOCAL SETUP ######

 
umi_data_folder=$raw_umi_data_dir/$session_name 
processed_output_dir_session=$umi_data_folder/processed_data/iphone # this is where the processed gopro and iphone data will be saved
multimodal_data_output_dir=$umi_data_folder/processed_data/all # this is where the multimodal final data will be saved

# Folder Struction Assumption:
# <raw_umi_data_dir>/
# ├── <session_name>/
# │   ├── UMI_iphone/
# │   │   ├── export_<YYYY-MM-DD_TIME>/
# │   │   │   ├── <TIME>_<side>.json
# │   │   │   ├── ...
# │   ├── DCIM/
# │   │   ├── 100GOPRO/
# │   │   │   ├── <GOPRO_FILE>.MP4
# │   │   │   ├──  ...
# │   ├── coinft/
# │   │   ├── YYYY-MM-DD/
# │   │   │   ├── <session_name>_<TIME>_LF.csv
# │   │   │   ├── ...
# │   ├── processed_data/
# │   │   ├── gopro_iphone/
# │   │   ├── all/

iphone_data_dir=$umi_data_folder/UMI_iphone
ft_data_dir=$umi_data_folder/coinft

# # Move all files from the subdirectories to the parent directory
# # turn 
# # # │   ├── coinft/
# # │   │   ├── YYYY-MM-DD/
# # │   │   │   ├── <session_name>_<TIME>_LF.csv
# # │   │   │   ├── ...
# # into
# # # │   ├── coinft/
# # │   │   ├── <session_name>_<TIME>_LF.csv
# # │   │   ├── ...

if [ "$flatten_coinft_folder_structure" = True ]; then
    for subdir in "$ft_data_dir"/*/; do
        # Move all files from the subdirectory to $data_parent_dir
        mv "$subdir"* "$ft_data_dir"
        rm -rf "$subdir"
    done
fi
 

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


cd $umift_parent_dir/submodules/umi_day/umi_day/demonstration_processing
export PYTHONPATH=$(pwd)/../..:$(pwd)/../../..:
#$(pwd)/../../../universal_manipulation_interface
# export PYTHONPATH=$PYTHONPATH:$(pwd)
# export PYTHONPATH=$PYTHONPATH:$(pwd)/..
# cd $umift_parent_dir
# export PYTHONPATH=$PYTHONPATH:$(pwd)
if [ -z "$skip_stages" ]; then
    python process_demos_iphone.py \
        group.iphone_dir=$iphone_data_dir \
        demonstrations_dir=$processed_output_dir_session \
        filters.session_name=$session_name \
        overwrite=true
else 
    echo "Skipping stages: $skip_stages"
    python process_demos_iphone.py \
        group.iphone_dir=$iphone_data_dir \
        demonstrations_dir=$processed_output_dir_session \
        filters.session_name=$session_name \
        overwrite=true \
        skip_stages=[$skip_stages]
fi
 
