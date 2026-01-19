
import numpy as np
import pickle, argparse, av, os, json, collections, math, pathlib, subprocess
from umift.utils.time_utils import mp4_get_start_datetime
from umift.processing.wired_time_sync import get_session_name
from exiftool import ExifToolHelper
import pandas as pd
from umift.processing.interpolate_utils import get_interp1d, get_gripper_calibration_interpolator
from umift.processing.preproc import gripper_width_determin_optimal_alignment, get_gripper_width
from umift.utils.print_utils import info_print

def get_bool_segments(bool_seq):
    bool_seq = np.array(bool_seq, dtype=bool)
    segment_ends = (np.nonzero(np.diff(bool_seq))[0] + 1).tolist()
    segment_bounds = [0] + segment_ends + [len(bool_seq)]
    segments = list()
    segment_type = list()
    for i in range(len(segment_bounds) - 1):
        start = segment_bounds[i]
        end = segment_bounds[i+1]
        this_type = bool_seq[start]
        segments.append(slice(start, end))
        segment_type.append(this_type)
    segment_type = np.array(segment_type, dtype=bool)
    return segments, segment_type

def get_number_of_cameras(video_meta_df):
    serial_count = video_meta_df['camera_serial'].value_counts()
    n_cameras = len(serial_count)
    return n_cameras

def get_camera_serial(file_path):
    """Retrieve the camera serial number from metadata using exiftool."""
    try:
        # Run exiftool with JSON output for the specific tag
        result = subprocess.run(
            ["exiftool", "-j", "-QuickTime:CameraSerialNumber", str(file_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        # Parse the JSON output
        metadata = json.loads(result.stdout)
        # Return the CameraSerialNumber if it exists
        return metadata[0].get("QuickTime:CameraSerialNumber", None)
    except subprocess.CalledProcessError as e:
        print(f"ExifTool error: {e.stderr}")
        return None

def find_gripper_calibration_path(demo_path):
    """
    Finds the subdirectory named 'gripper_calibration*' in the given demo_path.

    Args:
        demo_path (str): The path where to search for the subdirectory.

    Returns:
        str: The full path to the gripper_calibration directory if found, else None.
    """
    for item in os.listdir(demo_path):  # List items in the directory
        full_path = os.path.join(demo_path, item)  # Get full path
        if os.path.isdir(full_path) and item.startswith("gripper_calibration"):
            return full_path
    return None

def main(args):
    session_name = get_session_name(args)
    input_path = pathlib.Path(os.path.join(args.intermediate_umi_session_dir, session_name))
    fps = None
    rows = list()
    demos_dir = input_path.joinpath('demos')
    video_dirs = sorted([x.parent for x in demos_dir.glob('demo_*/raw_video.mp4')])
    info_print(f"demos_dir: {demos_dir}")
    # load gripper calibration
    gripper_id_gripper_cal_map = dict()
    # cam_serial_gripper_cal_map = dict()
    info_print('Entering first ExifToolHelper')
    # with ExifToolHelper() as et:
    # assuming just one gripper calibration per session
    gripper_cal_path = pathlib.Path(os.path.join(find_gripper_calibration_path(demos_dir), "gripper_range.json"))

    info_print(f"Processing {gripper_cal_path}")
 
    mp4_path = gripper_cal_path.parent.joinpath('raw_video.mp4')
    # meta = list(et.get_metadata(str(mp4_path)))[0]
    # cam_serial = meta['QuickTime:CameraSerialNumber']

    gripper_range_data = json.load(gripper_cal_path.open('r'))
    gripper_id = gripper_range_data['gripper_id']
    max_width = gripper_range_data['max_width']
    min_width = gripper_range_data['min_width']
    gripper_cal_data = {
        'aruco_measured_width': [min_width, max_width],
        'aruco_actual_width': [min_width, max_width]
    }
    gripper_cal_interp = get_gripper_calibration_interpolator(**gripper_cal_data)
    gripper_id_gripper_cal_map[gripper_id] = gripper_cal_interp
    # cam_serial_gripper_cal_map[cam_serial] = gripper_cal_interp
    info_print(f"Done procesing {gripper_cal_path}")

    info_print('Entering second ExifToolHelper')
 
    for video_dir in video_dirs:            
        mp4_path = video_dir.joinpath('raw_video.mp4')
        start_date = mp4_get_start_datetime(str(mp4_path))
        start_timestamp = start_date.timestamp()
        info_print('Entering third ExifToolHelper')
        with ExifToolHelper() as et:
            meta = list(et.get_metadata(str(mp4_path)))[0]
        cam_serial = meta['QuickTime:CameraSerialNumber']
        
        info_print(f"Processing {cam_serial}")
        
        pkl_path = video_dir.joinpath('tag_detection.pkl')
        if not pkl_path.is_file():
            print(f"Ignored {video_dir.name}, no tag_detection.pkl")
            continue
        
        with av.open(str(mp4_path), 'r') as container:
            stream = container.streams.video[0]
            n_frames = stream.frames
            if fps is None:
                fps = stream.average_rate
            else:
                if fps != stream.average_rate:
                    print(f"Inconsistent fps: {float(fps)} vs {float(stream.average_rate)} in {video_dir.name}")
                    exit(1)
        duration_sec = float(n_frames / fps)
        end_timestamp = start_timestamp + duration_sec
        
        rows.append({
            'video_dir': video_dir,
            'camera_serial': cam_serial,
            'start_date': start_date,
            'n_frames': n_frames,
            'fps': fps,
            'start_timestamp': start_timestamp,
            'end_timestamp': end_timestamp
        })
        
    if len(rows) == 0:
        print("No valid videos found!")
        exit(1)

 
    video_meta_df = pd.DataFrame(data=rows)
    
    n_cameras = get_number_of_cameras(video_meta_df)
    
    info_print('Populating events list')
    # populate events list
    events = list()
    for vid_idx, row in video_meta_df.iterrows():
        events.append({
            'vid_idx': vid_idx,
            'camera_serial': row['camera_serial'],
            't': row['start_timestamp'],
            'is_start': True
        })
        events.append({
            'vid_idx': vid_idx,
            'camera_serial': row['camera_serial'],
            't': row['end_timestamp'],
            'is_start': False
        })
    
    events = sorted(events, key=lambda x: x['t'])

    # populate demo_data_list 
    on_videos = set()
    on_cameras = set()
    demo_data_list = list()
    for _, event in enumerate(events):

        if event['is_start']:
            on_videos.add(event['vid_idx'])
            on_cameras.add(event['camera_serial'])
        else:
            on_videos.remove(event['vid_idx'])
            on_cameras.remove(event['camera_serial'])
        assert len(on_videos) == len(on_cameras)
        
        if len(on_cameras) == n_cameras:
            # start demo episode where all cameras are recording
            t_demo_start = event['t']
        elif t_demo_start is not None:
            # demo already started, but one camera stopped
            # stopping episode
            assert not event['is_start']
            
            t_start = t_demo_start
            t_end = event['t']
            
            # undo state update to get full set of videos
            demo_vid_idxs = set(on_videos)
            demo_vid_idxs.add(event['vid_idx'])
 
            
            demo_data_list.append({
                "video_idxs": sorted(demo_vid_idxs),
                "start_timestamp": t_start,
                "end_timestamp": t_end
            })
            t_demo_start = None
            
    # Stage 3
    # identify gripper id (hardware) using aruco
    # output: 
    # add video_meta_df['gripper_hardware_id'] column
    # cam_serial_gripper_hardware_id_map Dict[str, int]
    finger_tag_det_th = 0.8
    vid_idx_gripper_hardware_id_map = dict()
    cam_serial_gripper_ids_map = collections.defaultdict(list)
    for vid_idx, row in video_meta_df.iterrows():
        video_dir = row['video_dir']
        pkl_path = video_dir.joinpath('tag_detection.pkl')
        if not pkl_path.is_file():
            vid_idx_gripper_hardware_id_map[vid_idx] = -1
            continue
        tag_data = pickle.load(pkl_path.open('rb'))
        n_frames = len(tag_data)
        tag_counts = collections.defaultdict(lambda: 0)
        for frame in tag_data:
            for key in frame['tag_dict'].keys():
                tag_counts[key] += 1
        tag_stats = collections.defaultdict(lambda: 0.0)
        for k, v in tag_counts.items():
            tag_stats[k] = v / n_frames
            
        # classify gripper by tag
        # tag 0, 1 are reserved for gripper 0
        # tag 6, 7 are reserved for gripper 1
        max_tag_id = np.max(list(tag_stats.keys()))
        tag_per_gripper = 6
        max_gripper_id = max_tag_id // tag_per_gripper
        
        gripper_prob_map = dict()
        for gripper_id in range(max_gripper_id+1):
            left_id = gripper_id * tag_per_gripper
            right_id = left_id + 1
            left_prob = tag_stats[left_id]
            right_prob = tag_stats[right_id]
            gripper_prob = min(left_prob, right_prob)
            if gripper_prob <= 0:
                continue
            gripper_prob_map[gripper_id] = gripper_prob
        
        gripper_id_by_tag = -1
        if len(gripper_prob_map) > 0:
            gripper_probs = sorted(gripper_prob_map.items(), key=lambda x:x[-1])
            gripper_id = gripper_probs[-1][0]
            gripper_prob = gripper_probs[-1][1]
            if gripper_prob >= finger_tag_det_th:
                gripper_id_by_tag = gripper_id

        cam_serial_gripper_ids_map[row['camera_serial']].append(gripper_id_by_tag)
        vid_idx_gripper_hardware_id_map[vid_idx] = gripper_id_by_tag
    # stage 3 ends 

 
    all_gripper_widths = list()
    all_is_valid = list()
 
    # TODO: populate demo_video_meta_df 
    for demo_idx, demo_data in enumerate(demo_data_list):
        # descritize timestamps for all videos
        video_idxs = demo_data['video_idxs']
        start_timestamp = demo_data['start_timestamp']
        end_timestamp = demo_data['end_timestamp']
        
        # select relevant video data
        demo_video_meta_df = video_meta_df.loc[video_idxs].copy()
        demo_video_meta_df.set_index('camera_idx', inplace=True)
        demo_video_meta_df.sort_index(inplace=True)
        
        start_timestamp = gripper_width_determin_optimal_alignment(demo_video_meta_df, start_timestamp)

        cam_start_frame_idxs = list()
        n_frames = int((end_timestamp - start_timestamp) / dt)
        
        # to get start_frame_idx for each camera
        for cam_idx, row in demo_video_meta_df.iterrows():
            dt = 1 / row['fps']
            video_start_frame = math.ceil((start_timestamp - row['start_timestamp']) / dt)
            video_n_frames = math.floor((row['end_timestamp'] - start_timestamp) / dt) - 1
            if video_start_frame < 0:
                video_n_frames += video_start_frame
                video_start_frame = 0
            cam_start_frame_idxs.append(video_start_frame)
            n_frames = min(n_frames, video_n_frames)
        demo_timestamps = np.arange(n_frames) * float(dt) + start_timestamp
        
        all_gripper_widths = list()
        for cam_idx, row in demo_video_meta_df.iterrows():
            start_frame_idx = cam_start_frame_idxs[cam_idx]
            video_dir = row['video_dir']
            
           # load SLAM data
            csv_path = video_dir.joinpath('camera_trajectory.csv')
            if not csv_path.is_file():
                print(f"Skipping {video_dir.name}, no camera_trajectory.csv.")
                continue            
            
            csv_df = pd.read_csv(csv_path)
            # select aligned frames
            df = csv_df.iloc[start_frame_idx: start_frame_idx+n_frames]
            is_tracked = (~df['is_lost']).to_numpy()
            
            n_frames_valid = is_tracked.sum()
            if n_frames_valid < 60:
                print(f"Skipping {video_dir.name}, only {n_frames_valid} frames are valid.")
                continue
            is_step_valid = is_tracked.copy()
            
            # get gripper data
            pkl_path = video_dir.joinpath('tag_detection.pkl')
            assert pkl_path.is_file(), f"tag_detection.pkl not found in {video_dir}"
     
            tag_detection_results = pickle.load(open(pkl_path, 'rb'))
            tag_detection_results = tag_detection_results[start_frame_idx: start_frame_idx+n_frames]
            video_timestamps = np.array([x['time'] for x in tag_detection_results])

            ghi = row['gripper_hardware_id']
            if ghi < 0:
                print(f"Skipping {video_dir.name}, invalid gripper hardware id {ghi}")
                continue
            
            left_id = 6 * ghi
            right_id = left_id + 1

            gripper_cal_interp = None
            
            if ghi in gripper_id_gripper_cal_map:
                gripper_cal_interp = gripper_id_gripper_cal_map[ghi]
            # elif row['camera_serial'] in cam_serial_gripper_cal_map:
                # gripper_cal_interp = cam_serial_gripper_cal_map[row['camera_serial']]
                # print(f"Gripper id {ghi} not found in gripper calibrations {list(gripper_id_gripper_cal_map.keys())}. Falling back to camera serial map.")
            else:
                raise RuntimeError("Gripper calibration not found.")

            gripper_timestamps = list()
            gripper_widths = list()
            for td in tag_detection_results:
                width = get_gripper_width(td['tag_dict'], 
                    left_id=left_id, right_id=right_id, 
                    nominal_z=args.nominal_z)
                if width is not None:
                    gripper_timestamps.append(td['time'])
                    gripper_widths.append(gripper_cal_interp(width))
            gripper_interp = get_interp1d(gripper_timestamps, gripper_widths)
            this_gripper_widths = gripper_interp(video_timestamps)
            all_gripper_widths.append(this_gripper_widths)
            
            assert len(is_step_valid) == n_frames
 
       
            all_is_valid.append(is_step_valid)
    
        all_is_valid = np.array(all_is_valid)
        is_step_valid = np.all(all_is_valid, axis=0)
        
        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for s, is_valid_segment in zip(segment_slices, segment_type):
            start = s.start
            end = s.stop
            if not is_valid_segment:
                continue
            if (end - start) < args.min_episode_length:
                is_step_valid[start:end] = False
  
        # finally, generate one episode for each valid segment
        segment_slices, segment_type = get_bool_segments(is_step_valid)
        for s, is_valid in zip(segment_slices, segment_type):
            if not is_valid:
                continue
            start = s.start
            end = s.stop

            total_used_time += float((end - start) * dt)
            
            grippers = list()

            for cam_idx, row in demo_video_meta_df.iterrows():
                # gripper cam
                grippers.append({
                    "gripper_width": all_gripper_widths[cam_idx][start:end],
                })
    return grippers

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft_data_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/wired_collection/Python/Data/241119', help='csv file name of coinFT raw data')
    parser.add_argument('--visual_data_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/data/umi_day_data/coinFT_20241119_meeting_room', help='coinFT data directory')
    parser.add_argument('--output_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/output/coinFT_20241119_meeting_room', help='camera trajectory file (.csv)')
    parser.add_argument('--img_downscale', type=float, default=0.25, help='downscale factor for image processing')
    parser.add_argument('--calibration_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/gripper_calibration', help='Path to calibration')
    parser.add_argument('--intermediate_umi_session_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/intermediate_umi_output/coinft_umi_folder', help='Path to calibration')
    parser.add_argument('--umi_submodule_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/submodules/universal_manipulation_interface', help='Path to calibration')
    parser.add_argument('--umi_day_submodule_dir', type=str, default='/Users/chuerpan/Documents/repo/umiFT/submodules/umi_day', help='Path to calibration')
    parser.add_argument('--plot', action='store_true', help='plot the time sequences')
    parser.add_argument('-nz', '--nominal_z', type=float, default=0.072, help="nominal Z value for gripper finger tag")
    parser.add_argument('-ml', '--min_episode_length', type=int, default=24)
    args = parser.parse_args()
    grippers = main(args)
    print("Done")
    