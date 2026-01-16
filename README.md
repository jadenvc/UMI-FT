# UMI-FT
Official code base for UMI-FT.


# Data collection
UMI-FT lets you collect forceful manipulation data without a robot.
Please refer to [TBD](https://github.com/real-stanford/UMI-FT) for instructions on building UMI-FT.


1. **Rename the session name on the iPhone app**  
   Use the naming convention:  
   `task-YYMMDD-location-demonstrator-<other-info>`  

2. **Time Sync** ⌛
   - Open the time sync page:
     ```bash
     open wired_collection/iphone_gopro_timesync/index.html
     ```
   - A browser pop-up will display a QR code for time sync. 
   - Sync the GoPro with the checker on, after seeing the check on gopro, turn the checker off.
   - Mount the gopro on the robot, then select [time] on the iphone to start record a timesync video on iphone.  

     Refer to the [Notion Instructions](https://www.notion.so/Data-collection-pipeline-with-iPhone-ARKit-128892d6672580589667e319629f798a) for details.

3. **Record Gripper Calibration Video**
    - select [gripper] on the iphone to start record the gripper calibration video (open and closing the gripper for > 5 times) 

4. **Record Demos**
    - Set script constants for coinft data collection
        Set the `$raw_umi_data_dir`, `$session_name` and `$time`in the script, `bash/wired_collect_data.sh`
        - `$time`: the duration in seconds for the data collection, make sure to be longer than demo.
        - `$session_name`: Should match the session name set on the UMI_day app.
        - `$raw_umi_data_dir`: Specifies the data path relative to `<UMIFT_REPO_ROOT>` where the multimodal data is stored.

   - Start the FT recording script:
     ```bash
     conda activate wired_coinft_i2c
     bash bash/wired_collect_data.sh
     ```
     - The FT data will be saved to:  
        `<UMIFT_REPO_ROOT>/data/umift_data/<session_name>/coinft`

   - Record GoPro demos using the "Start Recording" button on the UMI_day iPhone app. 🎥
   - Stop the GoPro recording after the demo is complete.
   - Wait for coinFT data collection to finish.

5. **Export Demos from SD Card**  
   Save the data to the following folders:
   - **GoPro Demos**:  
     `<SD_CARD_DIR>/DCIM/100GOPRO` → `<UMIFT_REPO_ROOT>/data/umift_data/<session_name>/DCIM/100GOPRO`
   - **ARKit Demos**:  
     `<SD_CARD_DIR>/UMI_iPhone/export_<SESSION_TIME>` → `<UMIFT_REPO_ROOT>/data/umift_data/<session_name>/UMI_iPhone`

### Recommended Folder Structure
```
<raw_umi_data_dir>/ (:=<UMIFT_REPO_ROOT>/data/umift_data)
├── <session_name>/
│   ├── UMI_iphone/
│   │   ├── export_<YYYY-MM-DD_TIME>/
│   │   │   ├── <TIME>_<side>.json
│   │   │   ├── ...
│   ├── DCIM/
│   │   ├── 100GOPRO/
│   │   │   ├── <GOPRO_FILE>.MP4
│   │   │   ├──  ...
│   ├── coinft/
│   │   ├── YYYY-MM-DD/
│   │   │   ├── <COINFT_FILE>_LF.csv
│   │   │   ├── <COINFT_FILE>_RF.csv
│   │   │   ├── ...
```

---

# Data Processing


## 🔬 Data Postprocessing Instructions 

### 1. Set Constants in Postprocessing Scripts
Set the `$raw_umi_data_dir` and `$session_name` in the following scripts for your collected data session:
    - `bash/data_post_process_gopro_iphone.sh`
    - `bash/data_post_process_multimodal.sh`

- `$session_name`: Should match the session name set on the UMI_day app.
- `$raw_umi_data_dir`: Specifies the data path relative to `<UMIFT_REPO_ROOT>` where the multimodal data is stored.

### 2. Folder Assumptions 📁
```
<raw_umi_data_dir>/ (:=<UMIFT_REPO_ROOT>/data/umift_data)
├── <session_name>/
│   ├── UMI_iphone/
│   │   ├── export_<YYYY-MM-DD_TIME>/
│   │   │   ├── <TIME>_<side>.json
│   │   │   ├── ...
│   ├── DCIM/
│   │   ├── 100GOPRO/
│   │   │   ├── <GOPRO_FILE>.MP4
│   │   │   ├──  ...
│   ├── coinft/
│   │   ├── YYYY-MM-DD/
│   │   │   ├── <COINFT_FILE>_LF.csv
│   │   │   ├── <COINFT_FILE>_RF.csv
│   │   │   ├── ...
│   ├── processed_data/ (output from data processing)
│   │   ├── gopro_iphone/
│   │   ├── all/
```

---

### 3. Run Postprocessing Commands

#### Postprocess the GoPro + iPhone Data
```bash
conda activate umi_day
bash bash/data_post_process_gopro_iphone.sh
```

#### Postprocess All Other Data + Time Sync + Visualization
```bash
conda activate umift
bash bash/data_post_process_multimodal.sh
```

### 4. Output Zarr Data Format 📦

Example zarr data tree for two demos:
```
data.tree()
/
 ├── data
 │   ├── episode_0
 │   │   ├── gripper_0 (580, 1) float64
 │   │   ├── gripper_time_stamps_0 (580, 1) float64
 │   │   ├── rgb_0 (580, 202, 270, 3) uint8
 │   │   ├── rgb_time_stamps_0 (580, 1) float64
 │   │   ├── robot_time_stamps_0 (580, 1) float64
 │   │   ├── ts_pose_fb_0 (580, 7) float64
 │   │   ├── wrench_concat_0 (3600, 12) float64
 │   │   ├── wrench_concat_coinft_0 (3600, 12) float64
 │   │   ├── wrench_left_0 (3600, 6) float64
 │   │   ├── wrench_left_coinft_0 (3600, 6) float64
 │   │   ├── wrench_right_0 (3600, 6) float64
 │   │   ├── wrench_right_coinft_0 (3600, 6) float64
 │   │   ├── wrench_time_stamps_left_0 (3600, 1) float64
 │   │   └── wrench_time_stamps_right_0 (3600, 1) float64
 │   └── episode_1
 │       ├── gripper_0 (903, 1) float64
 │       ├── gripper_time_stamps_0 (903, 1) float64
 │       ├── rgb_0 (903, 202, 270, 3) uint8
 │       ├── rgb_time_stamps_0 (903, 1) float64
 │       ├── robot_time_stamps_0 (903, 1) float64
 │       ├── ts_pose_fb_0 (903, 7) float64
 │       ├── wrench_concat_0 (5404, 12) float64
 │       ├── wrench_concat_coinft_0 (5404, 12) float64
 │       ├── wrench_left_0 (5404, 6) float64
 │       ├── wrench_left_coinft_0 (5404, 6) float64
 │       ├── wrench_right_0 (5404, 6) float64
 │       ├── wrench_right_coinft_0 (5404, 6) float64
 │       ├── wrench_time_stamps_left_0 (5404, 1) float64
 │       └── wrench_time_stamps_right_0 (5404, 1) float64
 └── meta
     ├── episode_gripper0_len (2,) int64
     ├── episode_rgb0_len (2,) int64
     ├── episode_robot0_len (2,) int64
     └── episode_wrench0_len (2,) int64
```

Wrench data order assumption: [Fx, Fy, Fz, Mx, My, Mz]

Wrench body frame and world frame transformation reference from [Modern Robotics, Lynch & Park](https://hades.mech.northwestern.edu/images/7/7f/MR.pdf#page=126.68) 




# Policy Training



# Evaluation

