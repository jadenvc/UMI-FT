from argparse import ArgumentParser
import json
import os
import shutil
from datetime import datetime, timedelta, timezone
from umi_day.common.timecode_util import mp4_get_start_datetime
from av.error import InvalidDataError
import cv2
from umi_day.common.trajectory_util import sample_poses_at_times, plot_video_aligned_trajectory
from umi_day.common.transform_util import pose_4x4_to_quat_xyzw
from umi_day.common.generic_util import symlink_absolute
import numpy as np
import math
from collections import OrderedDict
import pandas as pd
from glob import glob
import re
import os
import tqdm
from statistics import stdev
import yaml
from colorama import init, Fore, Back, Style
init()


MIN_FRAME_COUNT = 60

def demonstration_to_string(demonstration_dir, side=None):
    if side is None:
        side = ''
        note = ''
    else:
        with open(os.path.join(demonstration_dir, f'{side}.json'), 'r') as f:
            demonstration_json = json.load(f)
        note = (f" \"{demonstration_json['note']}\"") if 'note' in demonstration_json and demonstration_json['note'] else ''
    return f'[{demonstration_json["sessionName"]} {demonstration_dir} {side}{note}]'

def write_demonstration_metadata(demonstration_dir, metadata_dict):
    metadata_dict = {**read_demonstration_metadata(demonstration_dir), **metadata_dict}

    metadata_path = os.path.join(demonstration_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata_dict, f)

def color(msg, color=Fore.BLUE):
    return color + msg + Style.RESET_ALL

def read_demonstration_metadata(demonstration_dir):
    metadata_path = os.path.join(demonstration_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            yaml.safe_dump({}, f)
    
    with open(metadata_path, 'r') as f:
        metadata_dict = yaml.safe_load(f)
    return metadata_dict

def keep_demonstration(demonstration_title, demonstration_json_path, specific_demonstrations, demonstrations_filter, session_name, type=None):
    if type is not None and not demonstration_title.endswith(f'_{type}'):
        return False

    if specific_demonstrations is not None and demonstration_title not in specific_demonstrations:
        return False
    
    if demonstrations_filter is not None and not re.search(demonstrations_filter, demonstration_title):
        return False
    
    if session_name:
        with open(demonstration_json_path, 'r') as f:
            demonstration_json = json.load(f)
            if 'sessionName' not in demonstration_json:
                return False
            elif demonstration_json['sessionName'] != session_name:
                return False
    
    return True

def iterate_demonstrations(demonstrations_dir, specific_demonstrations=None, demonstrations_filter=None, session_name=None, type=None):
    for demonstration_dir in sorted(os.listdir(demonstrations_dir)):
        full_demonstration_dir_path = os.path.join(demonstrations_dir, demonstration_dir)
        if not os.path.isdir(full_demonstration_dir_path):
            continue

        demonstration_json_path = os.path.join(demonstrations_dir, demonstration_dir, f'left.json')
        if not os.path.exists(demonstration_json_path):
            demonstration_json_path = os.path.join(demonstrations_dir, demonstration_dir, f'right.json')
        assert os.path.exists(demonstration_json_path)
        if not keep_demonstration(demonstration_dir, demonstration_json_path, specific_demonstrations, demonstrations_filter, session_name, type):
            continue

        yield full_demonstration_dir_path

def get_gopro_start_video_time(path, gopro_latency_correction_ms):
    return mp4_get_start_datetime(path) + timedelta(milliseconds=gopro_latency_correction_ms)

def group_iphone_gopro_data(iphone_dir, left_gopro_dir, right_gopro_dir, out_dir, symlink, delete_previous_output, overwrite, specific_demonstrations, demonstrations_filter, session_name, gopro_latency_correction):
    """Given data from iPhone and GoPro cameras (potentially paired left right data or just one side), group them together by time (finding GoPro clips associated with iPhone recording)."""
    if delete_previous_output and os.path.exists(out_dir):
        shutil.rmtree(out_dir)

    new_demonstrations = set()
    existing_demonstrations = set()
    failure_demonstrations = set()

    # Load demonstrations from iPhone
    export_dirs = os.listdir(iphone_dir)
    export_dirs.sort(reverse=True)
    export_dirs = [x for x in export_dirs if os.path.isdir(os.path.join(iphone_dir, x))]
    export_dirs.append(iphone_dir) # look for files in the root directory as well
    
    for export_dir_name in export_dirs:
        export_dir = os.path.join(iphone_dir, export_dir_name) if export_dir_name != iphone_dir else iphone_dir
        demonstration_files = os.listdir(export_dir)
        demonstration_files.sort()

        for demonstration_name in demonstration_files:
            demonstration_path = os.path.join(export_dir, demonstration_name)

            if not demonstration_name.endswith('.json'):
                continue
            demonstration_name = demonstration_name.replace('.json', '')

            # add this JSON file as a demonstration if it doesn't exist already
            split = demonstration_name.split('_')
            if len(split) != 4:
                continue

            demonstration_time_str, demonstration_randomizer, demonstration_side, recording_type = split
            demonstration_title = f'{demonstration_time_str}_{demonstration_randomizer}'

            demonstration_json_path = os.path.join(export_dir, demonstration_name + '.json')

            # skip this demonstration under certain conditions
            if not keep_demonstration(demonstration_title, demonstration_json_path, specific_demonstrations, demonstrations_filter, session_name):
                continue

            gopro_dir = left_gopro_dir if demonstration_side == 'left' else right_gopro_dir
            if gopro_dir is None:
                continue
            
            # start processing the demonstration
            demonstration_time_str_iso8601 = demonstration_time_str[:demonstration_time_str.index('T')] + demonstration_time_str[demonstration_time_str.index('T'):].replace('-', ':')
            demonstration_time = datetime.fromisoformat(demonstration_time_str_iso8601)

            demonstration_out_dir = os.path.join(out_dir, f'{demonstration_title}_{recording_type}')
            pose_out_path = os.path.join(demonstration_out_dir, demonstration_side + '.json')
            gopro_out_path = os.path.join(demonstration_out_dir, demonstration_side + '.mp4')

            if os.path.exists(demonstration_out_dir) and overwrite:
                shutil.rmtree(demonstration_out_dir)

            if not os.path.exists(demonstration_out_dir):
                new_demonstrations.add(demonstration_title)
                os.makedirs(demonstration_out_dir)
            elif demonstration_title not in new_demonstrations:
                existing_demonstrations.add(demonstration_title)

            # check if outputs already exist
            if os.path.exists(pose_out_path) and os.path.exists(gopro_out_path):
                print(f'Skipping {demonstration_to_string(demonstration_out_dir, demonstration_side)} because output already exists')
                continue

            # find the corresponding GoPro video
            gopro_files = [x for x in glob(os.path.join(gopro_dir, "**/**")) if x.endswith('.MP4') and not x.startswith('.')]
            gopro_files.sort(reverse=True, key=lambda x: os.path.getmtime(x))

            assert len(gopro_files) > 0, f'No GoPro files found in {gopro_dir} for {demonstration_side} side'

            closest_gopro_file = ""
            closest_gopro_time_diff = float('inf')
            closest_gopro_time = datetime.min
            for gopro_file in gopro_files:
                try:
                    gopro_timestamp = get_gopro_start_video_time(gopro_file, gopro_latency_correction) # get GoPro video start time using accurate timecode metadata stored in the mp4
                except IndexError:
                    # some mp4 files are corrupted so just skip them
                    continue
                except InvalidDataError:
                    # some mp4 files are corrupted so just skip them
                    continue

                time_diff = gopro_timestamp.timestamp() - demonstration_time.timestamp()

                if abs(time_diff) < closest_gopro_time_diff:
                    closest_gopro_time_diff = time_diff
                    closest_gopro_file = gopro_file
                    closest_gopro_time = gopro_timestamp

            closest_gopro_local_time = closest_gopro_time.astimezone() # local timezone
            demonstration_local_time = demonstration_time.astimezone() # local timezone
            
            print(f'[{demonstration_name}] Closest GoPro file was {closest_gopro_time_diff} seconds after the iPhone. GoPro ({demonstration_side}): {closest_gopro_local_time} ({os.path.basename(closest_gopro_file)}) and iPhone: {demonstration_local_time}')

            # TODO: we make the assumption that the GoPro video associated with the iPhone pose recording is the one that has the closest to the iPhone recording based on start time. Note that we use the GoPro start video time BEFORE calibrating with the QR code. This assumption is only reasonable if we assume the GoPro is only off from the QR code on the order of ~1 second
            if abs(closest_gopro_time_diff) > 2:
                print(f'--- WARNING: Previous demonstration ({demonstration_name}) had time error ({closest_gopro_time_diff}) larger than 2 seconds between iPhone and GoPro. Skipping this demonstration ---')
                shutil.rmtree(demonstration_out_dir)
                new_demonstrations.remove(demonstration_title)
                failure_demonstrations.add(demonstration_title)
                continue

            # Copy the GoPro video to the output directory if it doesn't already exist
            if not os.path.exists(gopro_out_path):
                if symlink:
                    symlink_absolute(closest_gopro_file, gopro_out_path)
                else:
                    shutil.copyfile(closest_gopro_file, gopro_out_path)

            # copy the pose data to the output directory if it doesn't exist
            if not os.path.exists(pose_out_path):
                if symlink:
                    symlink_absolute(demonstration_path, pose_out_path)
                else:
                    shutil.copyfile(demonstration_path, pose_out_path)

            # write metadata
            write_demonstration_metadata(demonstration_out_dir, {f'{demonstration_side}_closest_gopro_time_diff': closest_gopro_time_diff, f'{demonstration_side}_gopro_start_time': closest_gopro_time.isoformat(), f'{demonstration_side}_demonstration_start_time': demonstration_time.isoformat()})

    print(f'\nWrote {len(new_demonstrations)} new demonstrations to {out_dir} ({len(existing_demonstrations)} already existed)')
    print(f'Had {len(failure_demonstrations)} demonstrations that failed to be grouped{f': {failure_demonstrations}' if len(failure_demonstrations) > 0 else ""}')


def compute_gopro_ahead_of_qr_ms(video_path, max_detections=10):
    video_start_time = mp4_get_start_datetime(video_path).astimezone(timezone.utc)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    detector = cv2.QRCodeDetector()

    progress_bar = tqdm.tqdm(total=max_detections)
    i = 0
    num_detections = 0
    last_detection = ""
    ms_gopro_ahead_of_qr = 0
    video_cur_time = video_start_time
    all_ms_deltas = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        decoded_text, _, _ = detector.detectAndDecodeCurved(frame)
        if decoded_text is not None and decoded_text != '' and last_detection != decoded_text:
            # ignore frames which have the same QR code as the last frame
            num_detections += 1
            qr_time = datetime.fromisoformat(decoded_text).astimezone(timezone.utc)
            msDelta = (video_cur_time - qr_time).total_seconds() * 1000
            all_ms_deltas.append(msDelta)
            ms_gopro_ahead_of_qr = (ms_gopro_ahead_of_qr * (num_detections-1) + msDelta) / num_detections

            last_detection = decoded_text
            progress_bar.update(1)

        i += 1
        stdev_txt = f'{stdev(all_ms_deltas):.2f}' if len(all_ms_deltas) > 1 else 'N/A'
        progress_bar.set_description(f'{num_detections}/{i} had QR detected. Last detect on frame {i}. GoPro ahead of QR by {int(ms_gopro_ahead_of_qr)}ms with stdev {stdev_txt}ms')
        video_cur_time += timedelta(seconds=1/fps)

        if num_detections >= max_detections:
            break

    cap.release()
    cv2.destroyAllWindows()

    return ms_gopro_ahead_of_qr

def gopro_timesync(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, overwrite):
    for demonstration_dir in iterate_demonstrations(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, 'qrcalibration'):
        left_video_path = os.path.join(demonstration_dir, 'left.mp4')
        right_video_path = os.path.join(demonstration_dir, 'right.mp4')

        def process_side(video_path, side):
            if os.path.exists(video_path):
                key_name = f'msGoProAheadOfQR'

                out_path = os.path.join(demonstration_dir, f'processed_{side}.json')
                if os.path.exists(out_path) and not overwrite:
                    with open(out_path, 'r') as f:
                        data = json.load(f)
                    with open(os.path.join(demonstration_dir, f'{side}.json'), 'r') as f:
                        iphone_data = json.load(f)
                    print(f'Skipping {demonstration_to_string(demonstration_dir, side)} because output already exists. GoPro latency: {data[key_name]}ms. iPhone latency: {iphone_data["timeIPhoneAheadOfQRinMS"]}ms')
                    return

                print(f'[{demonstration_dir}] Computing GoPro QR alignment for {side} side')
                gopro_ahead_of_qr_ms = int(compute_gopro_ahead_of_qr_ms(video_path))
                with open(out_path, 'w') as f:
                    json.dump({'msGoProAheadOfQR': gopro_ahead_of_qr_ms}, f)

                write_demonstration_metadata(demonstration_dir, {f'{side}_{key_name}': gopro_ahead_of_qr_ms})

        process_side(left_video_path, 'left')
        process_side(right_video_path, 'right')

def align_iphone_gopro_data(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, gopro_latency_correction, overwrite):
    num_processed = 0
    for demonstration_dir in iterate_demonstrations(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, 'demonstration'):
        # align the pose data with the video
        left_pose_path = os.path.join(demonstration_dir, 'left.json')
        right_pose_path = os.path.join(demonstration_dir, 'right.json')
        left_video_path = os.path.join(demonstration_dir, 'left.mp4')
        right_video_path = os.path.join(demonstration_dir, 'right.mp4')
        left_present = os.path.exists(left_pose_path) and os.path.exists(left_video_path)
        right_present = os.path.exists(right_pose_path) and os.path.exists(right_video_path)

        # check if we have already performed the alignment
        finished = True
        def check_finished(side):
            nonlocal finished
            processed_path = os.path.join(demonstration_dir, f'processed_{side}.json')
            finished = finished and os.path.exists(processed_path)

            camera_trajectory_path = os.path.join(demonstration_dir, f'camera_trajectory_{side}.csv')
            finished = finished and os.path.exists(camera_trajectory_path)

        if left_present:
            check_finished('left')
        if right_present:
            check_finished('right')
        
        if finished and not overwrite:
            print(f'Skipping {demonstration_to_string(demonstration_dir)} because iPhone/GoPro alignment already performed')
            continue
        else:
            print(f'[{demonstration_dir}] Beginning to process')
            num_processed += 1
        
        # load times from pose data
        def load_pose_data(pose_path):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            return pose_data
            
        def load_pose_times(pose_data):
            times = pose_data['poseTimes']
            times = [datetime.fromisoformat(x).timestamp() for x in times]
            assert len(times) == len(pose_data['poseTransforms'])

            return times
        
        left_pose_data = load_pose_data(left_pose_path) if left_present else None
        right_pose_data = load_pose_data(right_pose_path) if right_present else None

        left_pose_times = load_pose_times(left_pose_data) if left_present else []
        right_pose_times = load_pose_times(right_pose_data) if right_present else []

        left_pose_start_time = left_pose_times[0] if left_present else -1
        right_pose_start_time = right_pose_times[0] if right_present else -1
        left_pose_end_time = left_pose_times[-1] if left_present else -1
        right_pose_end_time = right_pose_times[-1] if right_present else -1

        # load times from video metadata
        def load_gopro_video_times(video_path, side):
            """Only works on GoPro mp4 files which have special metadata for start time"""
            # Load the QR latency relative to the GoPro video
            pose_data = left_pose_data if side == 'left' else right_pose_data
            calibration_demonstration_name = pose_data['qrCalibrationRunName'].replace('_right', '').replace('_left', '')
            qr_calibration_path = os.path.join(demonstrations_dir, calibration_demonstration_name, f'processed_{side}.json')
            with open(qr_calibration_path, 'r') as f:
                qr_calibration_data = json.load(f)
                ms_gopro_ahead_of_qr = qr_calibration_data["msGoProAheadOfQR"]

            start_time = get_gopro_start_video_time(video_path, gopro_latency_correction - ms_gopro_ahead_of_qr).timestamp()

            cap = cv2.VideoCapture(video_path)
            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = frame_count / fps
            cap.release()

            end_time = start_time + duration
            return start_time, end_time, fps, frame_count
        
        left_video_start_time, left_video_end_time, _, left_frame_count = load_gopro_video_times(left_video_path, 'left') if left_present else (-1, -1, -1, -1)
        right_video_start_time, right_video_end_time, _, right_frame_count = load_gopro_video_times(right_video_path, 'right') if right_present else (-1, -1, -1, -1)

        if (left_frame_count != -1 and left_frame_count < MIN_FRAME_COUNT) or (right_frame_count != -1 and right_frame_count < MIN_FRAME_COUNT):
            print(f'--- WARNING: Skipping {demonstration_dir} because one of the GoPro videos has less than {MIN_FRAME_COUNT} frames ---')
            with open(os.path.join(demonstration_dir, 'skip.txt'), 'w') as f:
                f.write('One of the GoPro videos has less than 60 frames')
            continue

        # determine the start and end pose by finding where all the data overlaps
        valid_start_times = []
        valid_end_times = []
        if left_pose_times:
            valid_start_times.extend([left_pose_start_time, left_video_start_time])
            valid_end_times.extend([left_pose_end_time, left_video_end_time])
        if right_pose_times:
            valid_start_times.extend([right_pose_start_time, right_video_start_time])
            valid_end_times.extend([right_pose_end_time, right_video_end_time])
        
        valid_start_time = max(valid_start_times)
        valid_end_time = min(valid_end_times)

        def sample_poses_at_video_frames(video_path, side, poses, pose_times, start_time, end_time):
            video_start_time, video_end_time, fps, frame_count = load_gopro_video_times(video_path, side)
            
            # cut frames that happen before start_time
            time_cut_from_start = start_time - video_start_time
            start_frames_to_cut_float = time_cut_from_start * fps
            start_frames_to_cut = math.ceil(start_frames_to_cut_float)
            assert start_frames_to_cut >= 0
            extra_time_cut_from_start = (start_frames_to_cut - start_frames_to_cut_float) / fps # this is the tiny bit of time (less than ~1/60 of a second) that is cut before the first video frame we use

            # cut frames that happen after end_time
            video_duration = video_end_time - video_start_time - time_cut_from_start - extra_time_cut_from_start
            valid_duration = end_time - start_time
            end_frames_to_cut = math.ceil((video_duration + extra_time_cut_from_start - valid_duration) * fps)
            assert end_frames_to_cut >= 0

            # sample poses at the same times as the video frames
            resample_dt = 1 / fps
            num_frames_to_keep = frame_count - start_frames_to_cut - end_frames_to_cut
            sample_times = np.arange(num_frames_to_keep) * resample_dt + start_time + extra_time_cut_from_start
            assert sample_times[0] >= pose_times[0] and sample_times[-1] <= pose_times[-1]
            pose_samples = sample_poses_at_times(poses, pose_times, sample_times)

            # save some metadata
            write_demonstration_metadata(demonstration_dir, {f'{side}_time_cut_from_start': time_cut_from_start, f'{side}_extra_time_cut_from_start': extra_time_cut_from_start, f'{side}_start_frames_to_cut': start_frames_to_cut, f'{side}_end_frames_to_cut': end_frames_to_cut, f'{side}_num_frames_to_keep': num_frames_to_keep, f'{side}_raw_frame_count': frame_count})
            
            return num_frames_to_keep, pose_samples, sample_times, start_frames_to_cut, end_frames_to_cut
        
        # save poses and frames
        def save_sampled(num_frames_to_keep, poses, times, original_pose_data, start_frames_to_cut, end_frames_to_cut):
            side = original_pose_data['side']
            out_poses_path = os.path.join(demonstration_dir, f'processed_{side}.json')

            I_T_G = np.array([[np.cos(np.pi),0,np.sin(np.pi),0.039711],
                              [0, 1, 0, -0.008423],
                              [-np.sin(np.pi), 0, np.cos(np.pi), -0.039907],
                              [0,0,0,1]]) # geometrically the pose of the GoPro in the iPhone frame; involves a 180 rotation about Y axis and a translation. For iPhone 15 Pro
            
            # the poses from the iPhone (in `poses`) are respect to the world frame, so they are geometrically the transform from the world frame (which has y axis opposite of gravity) to the iPhone frame.
            # Thus the poses are W_T_I
            # We want to convert these poses to be in the GoPro frame, so we need to find the transform from the GoPro frame to the iPhone frame, which is I_T_G
            # Then we can convert the poses from the iPhone frame to the GoPro frame by W_T_G = W_T_I @ I_T_G
            poses = np.array([W_T_I @ I_T_G for W_T_I in poses]) # now becomes W_T_G, which is geometrically the transform from the world frame to the GoPro frame

            # output the poses as json
            with open(out_poses_path, 'w') as f:
                output_pose_data = {} 
                output_pose_data['times'] = [datetime.fromtimestamp(x).astimezone(timezone.utc).replace(tzinfo=None).isoformat() for x in times]
                output_pose_data['poseTransforms'] = poses.tolist()
                json.dump(output_pose_data, f)

            """Now save the poses in the camera_trajectory.csv format the UMI pipeline expects. This entails no longer working with poses on the trimmed video, but instead poses for the entire video. This means we need to pad the poses of the video (but setting is_lost to true so they are invalid) and later on UMI pipeline will handle selecting out the overlapping regions."""

            # pad poses to reach the full video length
            pos_quat_xyzw = pose_4x4_to_quat_xyzw(poses)
            pos_quat_xyzw = np.concatenate([np.full((start_frames_to_cut, 7), 0), pos_quat_xyzw], axis=0) # start pad poses
            pos_quat_xyzw = np.concatenate([pos_quat_xyzw, np.full((end_frames_to_cut, 7), 0)], axis=0) # end pad poses

            # pad times to reach the full video length
            delta_t = times[1] - times[0]
            times = [times[0] - delta_t * (start_frames_to_cut - i) for i in range(start_frames_to_cut)] + list(times) # start pad times
            times = times + [times[-1] + delta_t * (i + 1) for i in range(end_frames_to_cut)] # end pad times
            times = np.array(times)

            # output the poses as csv (matching the format in UMI pipeline)
            csv_data = OrderedDict({
                'frame_idx': np.arange(len(times)),
                'timestamp': [time - times[0] for time in times],
                'state': [2] * len(times),
                'is_lost': [True] * start_frames_to_cut + [False] * num_frames_to_keep + [True] * end_frames_to_cut,
                'is_keyframe': [False] * len(times),
                'x': pos_quat_xyzw[:, 0],
                'y': pos_quat_xyzw[:, 1],
                'z': pos_quat_xyzw[:, 2],
                'q_x': pos_quat_xyzw[:, 3],
                'q_y': pos_quat_xyzw[:, 4],
                'q_z': pos_quat_xyzw[:, 5],
                'q_w': pos_quat_xyzw[:, 6],
            })
            df = pd.DataFrame(csv_data)

            formats = {'timestamp': '{:.6f}'}
            for key in ['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']:
                formats[key] = '{:.9f}'

            for col, f in formats.items():
                df[col] = df[col].map(lambda x: f.format(x))

            out_csv_path = os.path.join(demonstration_dir, f'camera_trajectory_{side}.csv')
            df.to_csv(out_csv_path, index=False)
            
        if left_present:
            left_poses = np.array(left_pose_data['poseTransforms'])
            left_num_frames, left_poses, left_times, left_start_frames_to_cut, left_end_frames_to_cut = sample_poses_at_video_frames(left_video_path, 'left', left_poses, left_pose_times, valid_start_time, valid_end_time)
        if right_present:
            right_poses = np.array(right_pose_data['poseTransforms'])
            right_num_frames, right_poses, right_times, right_start_frames_to_cut, right_end_frames_to_cut = sample_poses_at_video_frames(right_video_path, 'right', right_poses, right_pose_times, valid_start_time, valid_end_time)

        if left_present and right_present:
            assert left_num_frames == right_num_frames
            assert len(left_poses) == len(right_poses)
            time_error_ignore = left_times[0] - right_times[0]
            assert time_error_ignore <= 1/60
            print(f'Difference between left and right times that we ignore: {time_error_ignore}')

            right_times = left_times # force align them (the error is less than 1/60 of a second so it shouldn't matter too much). This force align is required because there is no way we ensure the RGB frames are perflectly aligned between the two GoPro cameras.

        if left_present:
            save_sampled(left_num_frames, left_poses, left_times, left_pose_data, left_start_frames_to_cut, left_end_frames_to_cut)
        if right_present:
            save_sampled(right_num_frames, right_poses, right_times, right_pose_data, right_start_frames_to_cut, right_end_frames_to_cut)

    print(f'\nAligned {num_processed} demonstrations')


def visualize_iphone_gopro_data(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, overwrite, visualize_til_frame=-1):
    num_processed = 0
    for demonstration_dir in iterate_demonstrations(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, 'demonstration'):
        def visualize_side(side):
            nonlocal num_processed
            pose_path = os.path.join(demonstration_dir, f'processed_{side}.json')

            if not os.path.exists(pose_path):
                return
            
            video_out_path = f'{demonstration_dir}/combined_pose_and_video_{side}.mp4'
            if os.path.exists(video_out_path) and not overwrite:
                print(f'Skipping {demonstration_to_string(demonstration_dir, side)} because output visualization already exists')
                return
            print(f'[{demonstration_dir}] Beginning to visualize on {side} side')
            
            # get the start and end frames to cut
            metadata = read_demonstration_metadata(demonstration_dir)
            start_frames_to_cut = metadata[f'{side}_start_frames_to_cut']
            end_frames_to_cut = metadata[f'{side}_end_frames_to_cut']
            
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)

            poses = np.array(pose_data['poseTransforms'])
            
            video_path = os.path.join(demonstration_dir, f'{side}.mp4')
            plot_video_aligned_trajectory(video_path, poses, video_out_path, max_frames=visualize_til_frame, skip_start_frames=start_frames_to_cut, skip_end_frames=end_frames_to_cut)
            print(f'[{demonstration_dir}] Wrote to {video_out_path}')
            num_processed += 1

        visualize_side('left')
        visualize_side('right')

    print(f'\nVisualized {num_processed} demonstrations')


if __name__ == '__main__':
    all_stages = ['group', 'gopro_timesync', 'align', 'visualize']

    parser = ArgumentParser()
    parser.add_argument('--iphone_dir', type=str, default='/Volumes/iPhone/UMI_iPhone', help='Directory name for iPhone data')
    parser.add_argument('--left_gopro_dir', type=str, default='/Volumes/GoProLeft/DCIM', help='Directory name for left GoPro data')
    parser.add_argument('--right_gopro_dir', type=str, default='/Volumes/GoProRight/DCIM', help='Directory name for right GoPro data')
    parser.add_argument('--out_dir', type=str, default='tmp_demonstrations', help='Directory name for output demonstrations')
    parser.add_argument('--symlink', action='store_true', help='Create symlinks to demonstration data instead of copy it')
    parser.add_argument('--delete_previous_output', action='store_true', help='WARNING DESCRUCTIVE: Delete the previous output directory before processing. Does NOT delete demonstrations from their original directories.')
    parser.add_argument('--stages', default=all_stages, nargs='+', choices=all_stages, help='Stages to run')
    parser.add_argument('--skip_stages', default=[], nargs='+', choices=all_stages, help='Stages to skip')
    parser.add_argument('--skip_left', action='store_true', help='Skip processing left gripper data')
    parser.add_argument('--skip_right', action='store_true', help='Skip processing right gripper data')
    parser.add_argument('--overwrite', action='store_true', help='If demonstration already exists in output directory, overwrite it')
    parser.add_argument('--demonstrations', type=str, nargs='+', help='Specific demonstrations to process. If not specified, then process all demonstrations')
    parser.add_argument('--demonstrations_filter', help='Regex filter to only process demonstrations that match this filter')
    parser.add_argument('--session_name', default='', help='Process only demonstrations with this exact session name')
    parser.add_argument('--gopro_latency_correction', type=int, default=0, help='You should not need to use this with the new GoPro QR alignment strategy! Manual latency offset (in milliseconds) applied to the GoPro to handle "unmeasured" latency. Positive means shift GoPro time forward. Note that an additional latency offset will be applied according to the QR code synchronization. Use this when you manually want to perform a correction between GoPro and iPhone.') # TODO: potentially remove
    parser.add_argument('--visualize_til_frame', type=int, default=-1, help='Visualize until this frame in the video. -1 for all frames')
    args = parser.parse_args()

    stages = [stage for stage in args.stages if stage not in args.skip_stages]

    if 'group' in stages:
        print(color("--- GROUPING STAGE ---"))
        group_iphone_gopro_data(args.iphone_dir, None if args.skip_left else args.left_gopro_dir, None if args.skip_right else args.right_gopro_dir, args.out_dir, args.symlink, args.delete_previous_output, args.overwrite, args.demonstrations, args.demonstrations_filter, args.session_name, args.gopro_latency_correction)

    if 'gopro_timesync' in stages:
        print(color("\n--- GOPRO TIMESYNC STAGE ---"))
        gopro_timesync(args.out_dir, args.demonstrations, args.demonstrations_filter, args.session_name, args.overwrite)

    if 'align' in stages:
        print(color("\n--- ALIGN STAGE ---"))
        align_iphone_gopro_data(args.out_dir, args.demonstrations, args.demonstrations_filter, args.session_name, args.gopro_latency_correction, args.overwrite)

    if 'visualize' in stages:
        print(color("\n--- VISUALIZE STAGE ---"))
        visualize_iphone_gopro_data(args.out_dir, args.demonstrations, args.demonstrations_filter, args.session_name, args.overwrite, visualize_til_frame=args.visualize_til_frame)