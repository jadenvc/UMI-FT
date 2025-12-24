#!/usr/bin/env python3
"""
Filename: umift_ft_timestampped_I2C.py
Author:   Hojung Choi
Date:     2025/01/09

Description:
    This script provides a timestamped FT data capture from multiple CoinFT sensors
    that share a single serial interface (I2C-over-serial). It uses .onnx models
    to convert raw sensor readings into 6-axis force/torque.

    Output columns:
    [Timestamp,Fx,Fy,Fz,Mx,My,Mz,C1,C2,...,C12]
"""

import time
import psocScanner as psoc
import struct
import numpy as np
import os
import scipy.io
import datetime
import matplotlib.pyplot as plt
import sys
import ntplib
import onnxruntime as ort
import copy

# ------------------------------------------------------
# Fixed-Constants
# -----------------------------------------------------
NUM_RAW_CHANNELS = 12

# We will store: 
#   [timestamp, Fx, Fy, Fz, Mx, My, Mz, raw1, raw2, ..., raw12]
# That makes 1 + 6 + 12 = 19 columns total.
NUM_DATA_ENTRY = 1 + 6 + NUM_RAW_CHANNELS  

# ------------------------------------------------------
# User-Configurable Constants & Variables
# ------------------------------------------------------
SAMPLING_RATE       = 360            # Approx. sampling rate (Hz)
SENSOR_ADDRESSES    = [8, 9]         # I2C addresses (upper nibble in end byte)
INITIAL_SAMPLES     = 200            # Number of initial samples to compute offset
IGNORED_SAMPLES     = 10             # Number of initial samples to ignore before computing offset
WINDOW              = 1              # Moving average window for force/moment

PORT_NAME = "/dev/tty.usbmodem101"

FILE_NAME_ANNOTATIONS = ['LF', 'RF']
FT_SCALING = [1, 30]

do_static_plot = True  
do_live_plot   = False
debug_wrench   = True

debug_wrench_every = 250

# Throttle updates every N frames to reduce overhead
PLOT_EVERY = 25

# --- NEW: Time history for live plot in seconds ---
TIME_HISTORY = 10.0  # Show the last 10 seconds of data in the live plot

# Make sure the length of MODEL_PATHS matches the length of SENSOR_ADDRESSES.
MODEL_PATHS = ['UFT9_MLP_4L_scl_1_30.onnx', 'UFT10_MLP_4L_scl_1_30.onnx']  

# Output directory for results
RESULTS_DIR = r'/Users/hojungchoi/Desktop/code_hub/umift_overall/UMI-FT/wired_collection/Data' # changed to be a 
 
def get_ntp_time():
    """
    Attempt to get time from an NTP server for a more accurate global timestamp.
    Fallback to local time on failure.
    """
    try:
        client = ntplib.NTPClient()
        response = client.request('pool.ntp.org')
        return response.tx_time  # NTP server time
    except Exception as e:
        print("Failed to get NTP time:", e)
        return time.time()       # Use local time as fallback


def mainLoopMultiSensors(savingFileName, run_duration, result_dir):
    """
    Main data capture loop for N sensors over a single serial port.
    
    :param savingFileName: String label for output filename
    :param run_duration:   Duration of data capture in seconds
    """

    # create result dir if not exists
    os.makedirs(result_dir, exist_ok=True)
    
    # ------------------------------------------------------
    # Load .onnx Models (One per sensor)
    # ------------------------------------------------------
    if len(SENSOR_ADDRESSES) != len(MODEL_PATHS):
        raise ValueError("SENSOR_ADDRESSES and MODEL_PATHS length mismatch. "
                         "Each sensor address must have a corresponding .onnx model.")
    
    # Create a list of inference sessions, one per sensor.
    sessions = []
    for model_path in MODEL_PATHS:
        sess = ort.InferenceSession(model_path)
        sessions.append(sess)

    # ------------------------------------------------------
    # Serial / TactileSensor Setup
    # ------------------------------------------------------
    print(f"Opening TactileSensor on port: {PORT_NAME}")
    ts = psoc.TactileSensor(port=PORT_NAME)
    
    # Flush input buffers & idle
    ts.ser.flushInput()
    ts.sendChar("i")  # idle
    time.sleep(0.01)
    
    # Query sensor for packet size
    ts.sendChar("q")
    time.sleep(0.01)

    ts.packet_size = ord(ts.ser.read(1)) - 1
    ts.num_sensors = int((ts.packet_size - 1) / 2)
    ts.unpackFormat = '<' + 'B' * ts.packet_size

    print(f"Packet size (excluding STX): {ts.packet_size}, # of sensor pads: {ts.num_sensors}")

    # ------------------------------------------------------
    # Start streaming
    # ------------------------------------------------------
    print("Sending stream command...")
    ts.sendChar("s")
    time.sleep(0.01)

    # ------------------------------------------------------
    # Initial Sampling for Offsets
    # ------------------------------------------------------
    num_devices = len(SENSOR_ADDRESSES)

    # Initialize offset arrays and data buffers for offset collection
    offsets        = [np.zeros(ts.num_sensors, dtype=float) for _ in range(num_devices)]
    init_data      = [np.zeros((INITIAL_SAMPLES, ts.num_sensors), dtype=float) for _ in range(num_devices)]
    init_counts    = [0]*num_devices
    ma_buffers     = [[] for _ in range(num_devices)]

    print(f"Taring..")

    def all_offsets_collected():
        return all(init_counts[i] >= INITIAL_SAMPLES for i in range(num_devices))

    while not all_offsets_collected():
        # Wait for STX
        if ord(ts.ser.read(1)) != ts.STX:
            continue
        
        raw_packet = ts.ser.read(ts.packet_size)
        packet     = struct.unpack(ts.unpackFormat, raw_packet)
        end_byte   = packet[-1]
        
        address = end_byte >> 4
        etx     = end_byte & 0x0F
        
        # Check ETX
        if etx != ts.ETX:
            print("Bad end framing byte")
            continue
        
        # Convert raw data to np.array
        sensorVals = []
        for i in range(ts.num_sensors):
            low_byte  = packet[2*i]
            high_byte = packet[2*i+1]
            sensorVals.append(low_byte + 256*high_byte)
        sensorVals = np.array(sensorVals, dtype=float)

        # Identify which sensor
        if address in SENSOR_ADDRESSES:
            idx = SENSOR_ADDRESSES.index(address)
            if init_counts[idx] < INITIAL_SAMPLES:
                init_data[idx][init_counts[idx], :] = sensorVals
                init_counts[idx] += 1
        else:
            print(f"Wrong address. Address: {address}")

    # Now compute offsets
    for i in range(num_devices):
        # Exclude first IGNORED_SAMPLES for offset
        offsets[i] = np.mean(init_data[i][IGNORED_SAMPLES:], axis=0)

    # ------------------------------------------------------
    # Prepare Data Buffers for streaming
    # ------------------------------------------------------
    max_samples = int((SAMPLING_RATE + 10) * (run_duration + 5))
    # data_bufs[i] shape: (max_samples, 19):
    # [timestamp, Fx, Fy, Fz, Mx, My, Mz, raw1, raw2, ..., raw12]
    data_bufs = [np.zeros((max_samples, NUM_DATA_ENTRY), dtype=float) for _ in range(num_devices)]
    counts    = [0]*num_devices

    # NTP-based global start time
    start_ntp_time   = get_ntp_time()
    start_local_time = time.time()

    print("Beginning main acquisition loop...")

    # ------------------------------------------------------
    # LIVE PLOTTING SETUP
    # ------------------------------------------------------
    if do_live_plot:
        plt.ion()  # Enable interactive mode

        # We'll create a figure for each sensor, each with two subplots for Force and Moment
        figs = []
        force_axes = []
        moment_axes = []
        force_lines = []
        moment_lines = []

        # Data buffers for live plotting
        time_data = [[] for _ in range(num_devices)]
        fx_data   = [[] for _ in range(num_devices)]
        fy_data   = [[] for _ in range(num_devices)]
        fz_data   = [[] for _ in range(num_devices)]
        mx_data   = [[] for _ in range(num_devices)]
        my_data   = [[] for _ in range(num_devices)]
        mz_data   = [[] for _ in range(num_devices)]

        # We'll track how many frames we've plotted for each sensor
        sample_counts = [0]*num_devices

        for i in range(num_devices):
            fig, (ax_f, ax_m) = plt.subplots(2, 1, figsize=(8, 6))
            fig.suptitle(f"Sensor {i} (addr={SENSOR_ADDRESSES[i]})")

            # Force lines
            lf, = ax_f.plot([], [], label='Fx')
            ly, = ax_f.plot([], [], label='Fy')
            lz, = ax_f.plot([], [], label='Fz')
            ax_f.legend()
            ax_f.set_ylabel("Force [N]")
            ax_f.grid()

            # Moment lines
            lmx, = ax_m.plot([], [], label='Mx')
            lmy_, = ax_m.plot([], [], label='My')
            lmz_, = ax_m.plot([], [], label='Mz')
            ax_m.legend()
            ax_m.set_ylabel("Moment [Nm]")
            ax_m.set_xlabel("Time [s]")
            ax_m.grid()

            figs.append(fig)
            force_axes.append(ax_f)
            moment_axes.append(ax_m)

            force_lines.append((lf, ly, lz))
            moment_lines.append((lmx, lmy_, lmz_))

    # ------------------------------------------------------
    # Main Acquisition Loop
    # ------------------------------------------------------
    while (time.time() - start_local_time) < run_duration:
        # Wait for STX
        if ord(ts.ser.read(1)) != ts.STX:
            continue
        
        raw_packet = ts.ser.read(ts.packet_size)
        packet     = struct.unpack(ts.unpackFormat, raw_packet)
        end_byte   = packet[-1]

        address = end_byte >> 4
        etx     = end_byte & 0x0F
        
        if etx != ts.ETX:
            # Bad framing
            print("Bad end framing byte")
            continue

        # Convert data
        sensorVals = []
        for i in range(ts.num_sensors):
            low_byte  = packet[2*i]
            high_byte = packet[2*i+1]
            sensorVals.append(low_byte + 256*high_byte)
        sensorVals = np.array(sensorVals, dtype=float)

        # Check address
        if address not in SENSOR_ADDRESSES:
            print(f"Wrong address. Address: {address}")
            continue

        idx = SENSOR_ADDRESSES.index(address)
        if counts[idx] >= max_samples:
            print(f"Buffer overflow in data!")
            continue

        # Compute global timestamp
        current_time = start_ntp_time + (time.time() - start_local_time)

        # Subtract offset
        offset_data = sensorVals - offsets[idx]

        # Prepare the input for ONNX
        x_input = offset_data.astype(np.float32).reshape(1, 12)

        # Inference
        session = sessions[idx]
        input_name = session.get_inputs()[0].name
        ft_cal = session.run(None, {input_name: x_input})[0].flatten()  # shape (6,)

        # Scale force & moment
        ft_cal[:3] /= FT_SCALING[0]  # Fx, Fy, Fz
        ft_cal[3:] /= FT_SCALING[1]  # Mx, My, Mz

        ma_buffers[idx].append(ft_cal)
        if len(ma_buffers[idx]) > WINDOW:
            ma_buffers[idx].pop(0)
        averaged_ft_cal = np.mean(ma_buffers[idx], axis=0)

        # Store in that sensor's buffer
        row = data_bufs[idx][counts[idx], :]
        row[0]     = current_time
        row[1:7]   = averaged_ft_cal  # Fx..Mz
        row[7:19]  = offset_data  # raw offset data
        counts[idx] += 1

        # ----------------------------------
        # Print debug wrench
        # ----------------------------------
        if debug_wrench and idx == 1 and counts[idx] % debug_wrench_every == 0:
            # import pdb; pdb.set_trace()
            # The wrenches here are expressed in the left CoinFT frame. 
            wrench_f_left  = copy.deepcopy(data_bufs[0][counts[idx] - 1, 1:4])
            wrench_f_right = copy.deepcopy(data_bufs[1][counts[idx] - 1, 1:4])
            wrench_f_right[1] = -wrench_f_right[1]
            wrench_f_right[2] = -wrench_f_right[2]
            wrench_f = wrench_f_left + wrench_f_right
            print(f"Left: {wrench_f_left}, Right(leftCFTframe): {wrench_f_right}, Sum: {wrench_f}")
            wrench_norm = np.linalg.norm(wrench_f)
            print(f"Norm: {wrench_norm}")


        # ----------------------------------
        # Throttled Live Plot Update
        # ----------------------------------
        if do_live_plot:
            sample_counts[idx] += 1
            if sample_counts[idx] % PLOT_EVERY == 0:
                rel_time = current_time - start_ntp_time  # relative time for plotting x-axis

                # Append new data
                time_data[idx].append(rel_time)
                fx_data[idx].append(averaged_ft_cal[0])
                fy_data[idx].append(averaged_ft_cal[1])
                fz_data[idx].append(averaged_ft_cal[2])
                mx_data[idx].append(averaged_ft_cal[3])
                my_data[idx].append(averaged_ft_cal[4])
                mz_data[idx].append(averaged_ft_cal[5])

                # --- NEW: ENFORCE TIME_HISTORY (e.g., 10 seconds) ---
                # We'll drop old data points if the range exceeds TIME_HISTORY
                while (time_data[idx][-1] - time_data[idx][0]) > TIME_HISTORY:
                    time_data[idx].pop(0)
                    fx_data[idx].pop(0)
                    fy_data[idx].pop(0)
                    fz_data[idx].pop(0)
                    mx_data[idx].pop(0)
                    my_data[idx].pop(0)
                    mz_data[idx].pop(0)

                # Update line data for forces
                lf, ly, lz = force_lines[idx]
                lf.set_data(time_data[idx], fx_data[idx])
                ly.set_data(time_data[idx], fy_data[idx])
                lz.set_data(time_data[idx], fz_data[idx])

                # Update line data for moments
                lmx, lmy_, lmz_ = moment_lines[idx]
                lmx.set_data(time_data[idx], mx_data[idx])
                lmy_.set_data(time_data[idx], my_data[idx])
                lmz_.set_data(time_data[idx], mz_data[idx])

                # Adjust axes
                force_axes[idx].relim()
                force_axes[idx].autoscale_view()
                moment_axes[idx].relim()
                moment_axes[idx].autoscale_view()

                plt.pause(0.001)  # allow the plot to update

    # ------------------------------------------------------
    # Stop streaming & close port
    # ------------------------------------------------------
    print("Stopping acquisition.")
    ts.sendChar("i")
    time.sleep(0.01)
    ts.closePort()

    # Slice buffers to keep only the valid portion. 
    trimmed_count = min(counts)
    for i in range(num_devices):
        data_bufs[i] = data_bufs[i][:trimmed_count, :]

    print("Finished collecting data.")
    for i in range(num_devices):
        print(f"Sensor addr={SENSOR_ADDRESSES[i]}: {counts[i]} samples")
    print(f"Sensor data length was trimmed to {trimmed_count}")

    # ------------------------------------------------------
    # Final Static Plots
    # ------------------------------------------------------
    plt.ioff()
    if do_static_plot:
        for i in range(num_devices):
            sensor_label = f"Sensor_{i}_addr{SENSOR_ADDRESSES[i]}"
            # Force plot
            plt.figure()
            plt.plot(data_bufs[i][:, 1], label='Fx')
            plt.plot(data_bufs[i][:, 2], label='Fy')
            plt.plot(data_bufs[i][:, 3], label='Fz')
            plt.ylabel('Force [N]')
            plt.xlabel('Sample')
            plt.title(f"{sensor_label} - Forces (static)")
            plt.grid()
            plt.legend()

            # Moment plot
            plt.figure()
            plt.plot(data_bufs[i][:, 4], label='Mx')
            plt.plot(data_bufs[i][:, 5], label='My')
            plt.plot(data_bufs[i][:, 6], label='Mz')
            plt.ylabel('Moment [Nm]')
            plt.xlabel('Sample')
            plt.title(f"{sensor_label} - Moments (static)")
            plt.grid()
            plt.legend()

        plt.show()
    

    # ------------------------------------------------------
    # Save to CSV
    # ------------------------------------------------------
    currDateOnlyString = datetime.datetime.now().strftime("%Y-%m-%d")
    directory = os.path.join(result_dir, currDateOnlyString)
    if not os.path.exists(directory):
        os.makedirs(directory)

    currDateTimeString = datetime.datetime.now().strftime("%y%m%d_%H%M%S_")

    # CSV header (19 columns):
    header_str = "Timestamp,Fx,Fy,Fz,Mx,My,Mz," + ",".join([f"C{i+1}" for i in range(NUM_RAW_CHANNELS)])

    # Save each sensor's data to a separate CSV
    for i in range(num_devices):
        filename = f'UMIFT_data_{currDateTimeString}{savingFileName}_{FILE_NAME_ANNOTATIONS[i]}.csv'
        out_file = os.path.join(directory, filename)
        
        np.savetxt(
            out_file,
            data_bufs[i],
            delimiter=",",
            header=header_str,
            comments=''
        )
        print(f"Saved Sensor {i} (addr={SENSOR_ADDRESSES[i]}) data to: {out_file}")

    print("Data Collection and Saving Complete!.")


if __name__ == '__main__':
    # Command-line usage
    if len(sys.argv) < 3:
        print("Usage: python umift_ft_timestampped_I2C.py <SavingFileName> <RunDuration> <ResultDir>")
        sys.exit(1)

    savingFileName = sys.argv[1]
    run_duration   = float(sys.argv[2])  # in seconds
    result_dir = str(sys.argv[3])

    mainLoopMultiSensors(savingFileName, run_duration, result_dir)

