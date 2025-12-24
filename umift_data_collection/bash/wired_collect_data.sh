#!/bin/bash
# conda activate wired_coinft_i2c # conda_env/wired_coinft_i2c.yml
# Usage: <umift_parent_directory>$ bash bash/wired_collect_data.sh

# checklist for iphone data collection
# [ ] make sure the iphone is connected to wifi, otherwise the iphone clock will drift
# [ ] make sure the gopro has timesynced with rolling qrcode with the checkmark activated
# [ ] make sure you have collected a video of the rolling qrcode without the checkmark activated for [time] tab
# [ ] run this bash script first, then start iphone data collection under [demo] tab

umift_parent_dir=$(pwd)

###### TODO: CHANGE THESE VARS FOR LOCAL SETUP ######
raw_umi_data_dir=$umift_parent_dir/data/umift_data  # (USER TODO) change this to the directory where the raw umi gopro+iphone data are saved on your computer
session_name="WBW300-b13" # (USER TODO) change this to the session name of the raw umi gopro+iphone data
time=35 # time in seconds for the ft data collection
###### TODO: CHANGE THESE VARS FOR LOCAL SETUP ######

umi_data_folder=$raw_umi_data_dir/$session_name 
ft_data_dir=$umi_data_folder/coinft

cd ./wired_collection/Python
python umift_ft_timestampped_I2C.py $session_name $time $ft_data_dir
cd ../../
