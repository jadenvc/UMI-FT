#!/bin/bash
# conda activate umift_datacollection # conda_env/umift_datacollection.yml
# Usage: <umift_parent_directory>$ bash bash/wired_collect_data.sh

# checklist for iphone data collection
# [ ] make sure the iphone is connected to wifi, otherwise the iphone clock will drift
# [ ] run this bash script first, then start iphone data collection under [demo] tab. 
#     Once demonstration is over, end iphone data collection before the coinft script ends.

umift_data_parent_dir=$(pwd)

###### TODO: CHANGE THESE VARS FOR LOCAL SETUP ######
raw_umi_data_dir=/Users/jadenclark/Documents/UMI-FT/UMI-FT_Data/data
session_name="greyred" # set the exact same session name as in your iPhone
time=50 # how long the CoinFT data collection lasts. CoinFT data collection should start earlier than the iPhone data collection, and end later.
# PORT="/dev/tty.usbmodem179386601" # serial port to the CoinFTs
PORT="/dev/tty.usbmodem181377401" # serial port to the CoinFTs
WINDOW=10 # Applies moving average smoothing to the saved Force/Torque data
MODELS="/Users/jadenclark/Documents/UMI-FT/UMIFT_Data/NFT5_MLP_5L_norm_L2.onnx /Users/jadenclark/Documents/UMI-FT/UMIFT_Data/NFT4_MLP_5L_norm_L2.onnx"   # CoinFT calibration model paths. they later become [model_0_path, model_1_path]
NORMS="/Users/jadenclark/Documents/UMI-FT/UMIFT_Data/NFT5_norm.json /Users/jadenclark/Documents/UMI-FT/UMIFT_Data/NFT4_norm.json" # CoinFT calibration normalization constants.

# Set these to "--live_plot", "--static_plot", or "--debug_wrench" to enable, or leave empty "" to disable.
LIVE_PLOT="" # [important] live plot is only for debugging and visualization. Never enable it when collecting data. Plot rendering takes time and messes up timestamps.
STATIC_PLOT="" # static plot later displays the history of F/T data after data collection
DEBUG_WRENCH="" # this outputs resultant force/torque on the terminal every n samples.
###### END CONFIG ######

umi_data_folder=$raw_umi_data_dir/$session_name 
ft_data_dir=$umi_data_folder/coinft

cd ./wired_collection/Python

#Pass all new variables as arguments
python umift_ft_timestampped_UART.py \
    "$session_name" \
    "$time" \
    "$ft_data_dir" \
    --port "$PORT" \
    --window "$WINDOW" \
    --models $MODELS \
    --norms $NORMS \
    $LIVE_PLOT \
    $STATIC_PLOT \
    $DEBUG_WRENCH

cd ../../