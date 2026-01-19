#!/bin/bash
# conda activate umift_datacollection # conda_env/umift_datacollection.yml
# Usage: <umift_parent_directory>$ bash bash/wired_collect_data.sh

# checklist for iphone data collection
# [ ] make sure the iphone is connected to wifi, otherwise the iphone clock will drift
# [ ] run this bash script first, then start iphone data collection under [demo] tab. 
#     Once demonstration is over, end iphone data collection before the coinft script ends.

umift_parent_dir=$(pwd)


###### TODO: CHANGE THESE VARS FOR LOCAL SETUP ######
raw_umi_data_dir=$umift_parent_dir/data/umift_data 
session_name="test" 
time=30 #[s]

# --- NEW KNOBS ---
PORT="/dev/tty.usbmodem179386601"
WINDOW=10 # Applies moving average smoothing to the saved Force/Torque data
MODELS="NFT5_MLP_5L_norm_L2.onnx NFT4_MLP_5L_norm_L2.onnx"   # they later become [model_0_path, model_1_path]
NORMS="NFT5_norm.json NFT4_norm.json"

# Set these to "--live_plot", "--static_plot", or "--debug_wrench" to enable, or leave empty "" to disable.
LIVE_PLOT="--live_plot" # [important] live plot is only for debugging and visualization. Never enable it when collecting data. Plot rendering takes time and messes up timestamps.
STATIC_PLOT="--static_plot"
DEBUG_WRENCH=""
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