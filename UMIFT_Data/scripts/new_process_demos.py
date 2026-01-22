import tqdm, yaml, os, re, cv2, math, shutil, json
from argparse import ArgumentParser
from datetime import datetime, timedelta, timezone
from umi_day.common.timecode_util import mp4_get_start_datetime
from av.error import InvalidDataError
from umi_day.common.trajectory_util import sample_poses_at_times, plot_video_aligned_trajectory, save_trajectory_umi_format
from umi_day.common.transform_util import pose_4x4_to_quat_xyzw, pos_quat_xyzw_to_4x4
from umi_day.common.generic_util import symlink_absolute
import numpy as np
import pandas as pd
from glob import glob
from statistics import stdev
from colorama import init, Fore, Back, Style
init()

MIN_FRAME_COUNT = 60

def color(msg, color=Fore.BLUE):
    return color + msg + Style.RESET_ALL

def get_demonstration_dir(demonstrations_dir, demonstration_name):
    demonstration_time_str_iso8601 = demonstration_name[:demonstration_name.index('T')]
    demonstration_time = datetime_fromisoformat(demonstration_time_str_iso8601)
    demonstration_mdy = demonstration_time.strftime('%Y-%m-%d')
    path = os.path.join(demonstrations_dir, demonstration_mdy, demonstration_name)
    return path

def datetime_fromisoformat(date_str):
    # prior to python 3.11, the fromisoformat method doesn't support the 'Z' suffix to represent UTC time
    if date_str.endswith('Z'):
        date_str = date_str.replace('Z', '+00:00')
    return datetime.fromisoformat(date_str)

def demonstration_to_string(demonstration_dir, side=None):
    if side is None:
        return f'[{demonstration_dir}]'
    else:
        with open(os.path.join(demonstration_dir, f'{side}.json'), 'r') as f:
            demonstration_json = json.load(f)
        note = (f" \"{demonstration_json['note']}\"") if 'note' in demonstration_json and demonstration_json['note'] else ''
        return f'[{demonstration_json["sessionName"]} {demonstration_dir} {side}{note}]'
    

def write_demonstration_metadata(demonstration_dir, metadata_dict, overwrite_all=False):
    if not overwrite_all:
        metadata_dict = {**read_demonstration_metadata(demonstration_dir), **metadata_dict}

    metadata_path = os.path.join(demonstration_dir, 'metadata.yaml')
    with open(metadata_path, 'w') as f:
        yaml.safe_dump(metadata_dict, f)

def read_demonstration_metadata(demonstration_dir):
    metadata_path = os.path.join(demonstration_dir, 'metadata.yaml')
    if not os.path.exists(metadata_path):
        with open(metadata_path, 'w') as f:
            yaml.safe_dump({}, f)
    
    with open(metadata_path, 'r') as f:
        metadata_dict = yaml.safe_load(f)
    return metadata_dict

def remove_key_demonstration_metadata(demonstration_dir, key):
    metadata = read_demonstration_metadata(demonstration_dir)
    if key in metadata:
        del metadata[key]
    write_demonstration_metadata(demonstration_dir, metadata, overwrite_all=True)

def keep_demonstration(demonstration_title, demonstration_json_path, specific_demonstrations, demonstrations_filter, session_name, type=None):
    if type is not None and not demonstration_title.endswith(f'_{type}'):
        return False

    if specific_demonstrations is not None and demonstration_title not in specific_demonstrations:
        return False
    
    if demonstrations_filter is not None and not re.search(demonstrations_filter, demonstration_title):
        return False
    
    if session_name:
        split = demonstration_title.split('_')
        if len(split) == 4:
            demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type = split
            if session_name != demonstration_session_name:
                return False
        else:
            # TODO: eventually remove this legacy format
            with open(demonstration_json_path, 'r') as f:
                demonstration_json = json.load(f)
                if 'sessionName' not in demonstration_json:
                    return False
                elif demonstration_json['sessionName'] != session_name:
                    return False
    
    return True

def iterate_demonstrations(demonstrations_dir, specific_demonstrations=None, demonstrations_filter=None, session_name=None, type=None):
    for demonstration_dir in sorted(list(glob(demonstrations_dir + '/*/*'))):
        if not os.path.isdir(demonstration_dir):
            continue

        demonstration_json_path = os.path.join(demonstration_dir, f'left.json')
        if not os.path.exists(demonstration_json_path):
            demonstration_json_path = os.path.join(demonstration_dir, f'right.json')
        assert os.path.exists(demonstration_json_path)
        if not keep_demonstration(os.path.basename(demonstration_dir), demonstration_json_path, specific_demonstrations, demonstrations_filter, session_name, type):
            continue

        yield demonstration_dir

def get_gopro_start_video_time(path, gopro_latency_correction_ms=0):
    # contains additional logic to handle GoPro videos which have local time stored in metadata instead of UTC time (if we are using the old QR code method before it was moved to UTC time)
    # and logic to add correction offset to align GoPro with QR code
    global args

    date = mp4_get_start_datetime(path) + timedelta(milliseconds=gopro_latency_correction_ms)

    if args.gopro_local_timezone:
        # this is data using the old QR code method which is in local time so we need to convert it to UTC time
        date = date + timedelta(hours=-args.gopro_local_timezone) # convert from local to UTC time so negate 

    # at this point we know this data is in UTC time so attach UTC timezone to the timestamp
    date = date.replace(tzinfo=timezone.utc)

    return date

def group_iphone_gopro_data(iphone_dir, left_gopro_dir, right_gopro_dir, out_dir, symlink, overwrite, specific_demonstrations, demonstrations_filter, session_name):
    """Given data from iPhone and GoPro cameras (potentially paired left right data or just one side), group them together by time (finding GoPro clips associated with iPhone recording)."""
    new_demonstrations = set()
    existing_demonstrations = set()
    failure_demonstrations = set()
    demonstration_title_to_type = {}

    # Load demonstrations from iPhone
    demonstration_files = list(glob(os.path.join(iphone_dir, "**/*.json"), recursive=True))
    demonstration_files.sort(reverse=True)

    for demonstration_json_path in demonstration_files:
        demonstration_name = os.path.basename(demonstration_json_path).replace('.json', '')

        # add this JSON file as a demonstration if it doesn't exist already
        split = demonstration_name.split('_')
        if len(split) == 5:
            demonstration_time_str, demonstration_randomizer, demonstration_session_name, recording_type, demonstration_side = split
            demonstration_title = f'{demonstration_time_str}_{demonstration_randomizer}_{demonstration_session_name}_{recording_type}'
        elif len(split) == 4:
            # TODO: eventually remove this legacy format
            demonstration_time_str, demonstration_randomizer, demonstration_side, recording_type = split

            with open(demonstration_json_path, 'r') as f:
                json_data = json.load(f)
            demonstration_session_name = json_data['sessionName'] if 'sessionName' in json_data else 'no-session'
            demonstration_title = f'{demonstration_time_str}_{demonstration_randomizer}_{recording_type}'
        else:
            print(color(f"Skipping demonstration with unknown or old name format: {demonstration_name}", Fore.RED))
            continue

        # skip this demonstration under certain conditions
        if not keep_demonstration(demonstration_title, demonstration_json_path, specific_demonstrations, demonstrations_filter, session_name):
            continue

        gopro_dir = left_gopro_dir if demonstration_side == 'left' else right_gopro_dir
        if gopro_dir is None:
            continue
        
        # start processing the demonstration
        demonstration_time_str_iso8601 = demonstration_time_str[:demonstration_time_str.index('T')] + demonstration_time_str[demonstration_time_str.index('T'):].replace('-', ':')
        demonstration_time = datetime_fromisoformat(demonstration_time_str_iso8601)

        demonstration_out_dir = get_demonstration_dir(out_dir, demonstration_title)
        pose_out_path = os.path.join(demonstration_out_dir, demonstration_side + '.json')
        gopro_out_path = os.path.join(demonstration_out_dir, demonstration_side + '.mp4')

        if os.path.exists(demonstration_out_dir) and overwrite:
            shutil.rmtree(demonstration_out_dir)

        demonstration_title_to_type[demonstration_title] = recording_type

        if not os.path.exists(demonstration_out_dir):
            new_demonstrations.add(demonstration_title)
            os.makedirs(demonstration_out_dir)
        elif demonstration_title not in new_demonstrations:
            existing_demonstrations.add(demonstration_title)

        # check if outputs already exist
        if os.path.exists(pose_out_path) and os.path.exists(gopro_out_path):
            continue

        # find the corresponding GoPro video
        gopro_files = glob(os.path.join(gopro_dir, "**/*.MP4"), recursive=True)
        gopro_files.sort(reverse=True, key=lambda x: os.path.getmtime(x))

        assert len(gopro_files) > 0, f'No GoPro files found in {gopro_dir} for {demonstration_side} side'

        closest_gopro_file = ""
        closest_gopro_time_diff = float('inf')
        closest_gopro_time = datetime.min
        for gopro_file in gopro_files:
            try:
                gopro_timestamp = get_gopro_start_video_time(gopro_file) # get GoPro video start time using accurate timecode metadata stored in the mp4
            except IndexError:
                # some mp4 files are corrupted so just skip them
                continue
            except InvalidDataError:
                # some mp4 files are corrupted so just skip them
                continue

            time_diff = gopro_timestamp.timestamp() - demonstration_time.timestamp()

            if abs(time_diff) < abs(closest_gopro_time_diff):
                closest_gopro_time_diff = time_diff
                closest_gopro_file = gopro_file
                closest_gopro_time = gopro_timestamp
        
        print(f'[{demonstration_name}] Closest GoPro file was {closest_gopro_time_diff} seconds after the iPhone. GoPro ({demonstration_side}): {closest_gopro_time} ({os.path.basename(closest_gopro_file)}) and iPhone: {demonstration_time}')

        # TODO: we make the assumption that the GoPro video associated with the iPhone pose recording is the one that has the closest to the iPhone recording based on start time. Note that we use the GoPro start video time BEFORE calibrating with the QR code. This assumption is only reasonable if we assume the GoPro is only off from the QR code on the order of ~1 second
        if abs(closest_gopro_time_diff) > 2:
            print(color(f'--- ERROR: Previous demonstration ({demonstration_name}) had time error ({closest_gopro_time_diff}) seconds larger than 2 seconds between iPhone and GoPro. Skipping this demonstration ---', Fore.RED))
            shutil.rmtree(demonstration_out_dir)
            if demonstration_title in new_demonstrations:
                new_demonstrations.remove(demonstration_title)
            if demonstration_title in existing_demonstrations:
                existing_demonstrations.remove(demonstration_title)
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
                symlink_absolute(demonstration_json_path, pose_out_path)
            else:
                shutil.copyfile(demonstration_json_path, pose_out_path)

        # write metadata
        write_demonstration_metadata(demonstration_out_dir, {f'{demonstration_side}_closest_gopro_time_diff': closest_gopro_time_diff, f'{demonstration_side}_gopro_start_time': closest_gopro_time.isoformat(), f'{demonstration_side}_demonstration_start_time': demonstration_time.isoformat()})

    def count_demonstrations_of_type(demonstrations_set, type):
        count = 0
        for demonstration in demonstrations_set:
            if type == demonstration_title_to_type[demonstration]:
                count += 1
        return count

    print(f'\nNew: {count_demonstrations_of_type(new_demonstrations, "demonstration")} demonstrations, {count_demonstrations_of_type(new_demonstrations, "qrcalibration")} QR calibrations, and {count_demonstrations_of_type(new_demonstrations, "grippercalibration")} gripper calibrations to {out_dir}')
    print(f'Existing: {count_demonstrations_of_type(existing_demonstrations, "demonstration")} demonstrations, {count_demonstrations_of_type(existing_demonstrations, "qrcalibration")} QR calibrations, and {count_demonstrations_of_type(existing_demonstrations, "grippercalibration")} gripper calibrations')
    print(f'Failed: {count_demonstrations_of_type(failure_demonstrations, "demonstration")} demonstrations, {count_demonstrations_of_type(failure_demonstrations, "qrcalibration")} QR calibrations, and {count_demonstrations_of_type(failure_demonstrations, "grippercalibration")} gripper calibrations. Failure names: {failure_demonstrations if len(failure_demonstrations) > 0 else "{}"}')


def compute_gopro_ahead_of_qr_ms(video_path, max_detections=10):
    video_start_time = get_gopro_start_video_time(video_path)

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
            qr_time = datetime_fromisoformat(decoded_text)
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
    num_processed = 0
    num_already_processed = 0
    for demonstration_dir in iterate_demonstrations(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, 'qrcalibration'):
        left_video_path = os.path.join(demonstration_dir, 'left.mp4')
        right_video_path = os.path.join(demonstration_dir, 'right.mp4')

        def process_side(video_path, side):
            nonlocal num_processed, num_already_processed
            if os.path.exists(video_path):
                key_name = f'msGoProAheadOfQR'

                out_path = os.path.join(demonstration_dir, f'processed_{side}.json')
                if os.path.exists(out_path) and not overwrite:
                    with open(out_path, 'r') as f:
                        data = json.load(f)
                    with open(os.path.join(demonstration_dir, f'{side}.json'), 'r') as f:
                        iphone_data = json.load(f)
                    print(f'Skipping {demonstration_to_string(demonstration_dir, side)} because output already exists. GoPro latency: {data[key_name]}ms. iPhone latency: {iphone_data["timeIPhoneAheadOfQRinMS"]}ms')
                    num_already_processed += 1
                    return

                print(f'[{demonstration_dir}] Computing GoPro QR alignment for {side} side')
                gopro_ahead_of_qr_ms = int(compute_gopro_ahead_of_qr_ms(video_path))
                with open(out_path, 'w') as f:
                    json.dump({'msGoProAheadOfQR': gopro_ahead_of_qr_ms}, f)

                write_demonstration_metadata(demonstration_dir, {f'{side}_{key_name}': gopro_ahead_of_qr_ms})
                num_processed += 1

        process_side(left_video_path, 'left')
        process_side(right_video_path, 'right')
    
    print(f'\nPerformed timesync on {num_processed} demonstrations')
    print(f'Already had performed timesync on {num_already_processed} demonstrations')

def align_iphone_gopro_data(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, overwrite):
    num_processed = 0
    num_already_processed = 0
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
            camera_trajectory_path = os.path.join(demonstration_dir, f'camera_trajectory_gopro_{side}.csv')
            finished = finished and os.path.exists(camera_trajectory_path)

        if left_present:
            check_finished('left')
        if right_present:
            check_finished('right')
        
        if finished and not overwrite:
            num_already_processed += 1
            continue
        else:
            print(f'[{demonstration_dir}] Aligning...')
            num_processed += 1
        
        # load times from pose data
        def load_pose_data(pose_path):
            with open(pose_path, 'r') as f:
                pose_data = json.load(f)
            return pose_data
            
        def load_pose_times(pose_data):
            times = pose_data['poseTimes']
            times = [datetime_fromisoformat(x).timestamp() for x in times]
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
            demonstration_dir = get_demonstration_dir(demonstrations_dir, calibration_demonstration_name)
            qr_calibration_path = os.path.join(demonstration_dir, f'processed_{side}.json')
            with open(qr_calibration_path, 'r') as f:
                qr_calibration_data = json.load(f)
                ms_gopro_ahead_of_qr = qr_calibration_data["msGoProAheadOfQR"]

            start_time = get_gopro_start_video_time(video_path, -ms_gopro_ahead_of_qr).timestamp()

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
            print(color(f'--- ERROR: Skipping {demonstration_dir} because one of the GoPro videos has less than {MIN_FRAME_COUNT} frames ---', Fore.RED))
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
        def save_sampled(num_frames_to_keep, iphone_cam_poses, times, original_pose_data, start_frames_to_cut, end_frames_to_cut):
            side = original_pose_data['side']

            """Now save the poses in the csv format that the UMI pipeline expects. This entails no longer working with poses on the trimmed video, but instead poses for the entire video. This means we need to pad the poses of the video (but setting is_lost to true so they are invalid) and later on UMI pipeline will handle selecting out the valid regions."""

            # pad timestamps to reach the full video length
            delta_t = times[1] - times[0]
            times = [times[0] - delta_t * (start_frames_to_cut - i) for i in range(start_frames_to_cut)] + list(times) # start pad timestamps
            times = times + [times[-1] + delta_t * (i + 1) for i in range(end_frames_to_cut)] # end pad timestamps
            times = np.array(times)

            # is_lost
            is_lost = [True] * start_frames_to_cut + [False] * num_frames_to_keep + [True] * end_frames_to_cut

            def save_camera_trajectory(poses, suffix=None):
                # set world frame to have the same origin as the first pose
                pos_quat_xyzw = pose_4x4_to_quat_xyzw(poses)

                # pad poses to reach the full video length
                empty_7d_pos = np.array([0,0,0,0,0,0,1]) # pos quat xyzw
                pos_quat_xyzw = np.concatenate([np.repeat(np.expand_dims(empty_7d_pos, 0), start_frames_to_cut, axis=0), pos_quat_xyzw], axis=0) # start pad poses
                pos_quat_xyzw = np.concatenate([pos_quat_xyzw, np.repeat(np.expand_dims(empty_7d_pos, 0), end_frames_to_cut, axis=0)], axis=0) # end pad poses

                poses = pos_quat_xyzw_to_4x4(pos_quat_xyzw)
                
                if suffix is None:
                    suffix = ""
                else:
                    suffix = f"_{suffix}"

                save_trajectory_umi_format(os.path.join(demonstration_dir, f'camera_trajectory{suffix}_{side}.csv'), poses, times, is_lost)

            # iPhone local frame has y up, x right and z toward you when looking at the screen with charger port on the right
            # GoPro local frame has y down, x right and z away from you when looking at the screen
            # thus converting between the two consists of a 180 degree rotation about the x axis and a translation
            I_T_G = np.array([[1,0,0,0.039711],
                              [0,-1,0,-0.008423],
                              [0,0,-1,-0.039907],
                              [0,0,0,1]]) # geometrically the pose of the GoPro in the iPhone frame; involves a 180 rotation about X axis and a translation. For iPhone 15 Pro
            
            # the poses from the iPhone (in `poses`) are respect to the world frame, so they are geometrically the transform from the world frame (which has y axis opposite of gravity) to the iPhone frame.
            # Thus the poses are W_T_I
            # We want to convert these poses to be in the GoPro frame, so we need to find the transform from the GoPro frame to the iPhone frame, which is I_T_G
            # Then we can convert the poses from the iPhone frame to the GoPro frame by W_T_G = W_T_I @ I_T_G
            gopro_cam_poses = np.array([W_T_I @ I_T_G for W_T_I in iphone_cam_poses]) # now becomes W_T_G, which is geometrically the transform from the world frame to the GoPro frame

            # save poses in two styles (both have the same ARKit world origin):
            # 1) in ARKit pose format which preserves the iPhone local coordinate frame convention from ARKit and stores the iPhone pose instead of the GoPro pose. This is useful if you want to work with iPhone poses.
            save_camera_trajectory(iphone_cam_poses, 'iphone')
            # 2) in UMI pose format which uses the GoPro/TCP local coordinate frame convention. This is useful if you want to work with GoPro poses and/or train an UMI policy.
            save_camera_trajectory(gopro_cam_poses, 'gopro')

            # save the exact timestamps of each of the GoPro frames and associated poses to the metadata
            time_strs = [datetime.fromtimestamp(x).replace(tzinfo=timezone.utc).isoformat() for x in times]
            write_demonstration_metadata(demonstration_dir, {f'{side}_gopro_frame_times': time_strs})
            
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
    print(f'Already had aligned {num_already_processed} demonstrations')


def visualize_iphone_gopro_data(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, overwrite, visualize_til_frame=-1):
    num_processed = 0
    num_already_processed = 0
    for demonstration_dir in iterate_demonstrations(demonstrations_dir, specific_demonstrations, demonstrations_filter, session_name, 'demonstration'):
        def visualize_side(side):
            nonlocal num_processed, num_already_processed
            pose_path = os.path.join(demonstration_dir, f'camera_trajectory_gopro_{side}.csv')

            if not os.path.exists(pose_path):
                return
            
            df = pd.read_csv(pose_path)
            is_lost = df[['is_lost']].values
            poses = df[['x', 'y', 'z', 'q_x', 'q_y', 'q_z', 'q_w']].values
            poses = pos_quat_xyzw_to_4x4(poses)
            
            video_out_path = f'{demonstration_dir}/combined_gopro_pose_and_video_{side}.mp4'
            if os.path.exists(video_out_path) and not overwrite:
                num_already_processed += 1
                return
            print(f'[{demonstration_dir}] Beginning to visualize on {side} side')
            
            # Load the pose data
            pd.read_csv(pose_path)
            
            # Visualize the frames and poses
            video_path = os.path.join(demonstration_dir, f'{side}.mp4')
            plot_video_aligned_trajectory(video_path, video_out_path, poses, is_lost, max_frames=visualize_til_frame)
            print(f'[{demonstration_dir}] Wrote to {video_out_path}')
            num_processed += 1

        visualize_side('left')
        visualize_side('right')

    print(f'\nVisualized {num_processed} demonstrations')
    print(f'Already visualized {num_already_processed} demonstrations')


if __name__ == '__main__':
    all_stages = ['group', 'gopro_timesync', 'align', 'visualize']

    parser = ArgumentParser()
    parser.add_argument('--iphone_dir', type=str, default='/Volumes/iPhone/UMI_iPhone', help='Directory name for iPhone data')
    parser.add_argument('--left_gopro_dir', type=str, default='/Volumes/GoProLeft/DCIM', help='Directory name for left GoPro data')
    parser.add_argument('--right_gopro_dir', type=str, default='/Volumes/GoProRight/DCIM', help='Directory name for right GoPro data')
    parser.add_argument('--out_dir', type=str, default='tmp_demonstrations', help='Directory name for output demonstrations')
    parser.add_argument('--symlink', action='store_true', help='Create symlinks to demonstration data instead of copy it')
    parser.add_argument('--stages', default=all_stages, nargs='+', choices=all_stages, help='Stages to run')
    parser.add_argument('--skip_stages', default=[], nargs='+', choices=all_stages, help='Stages to skip')
    parser.add_argument('--skip_left', action='store_true', help='Skip processing left gripper data')
    parser.add_argument('--skip_right', action='store_true', help='Skip processing right gripper data')
    parser.add_argument('--overwrite', action='store_true', help='If demonstration already exists in output directory, overwrite it')
    parser.add_argument('--demonstrations', type=str, nargs='+', help='Specific demonstrations to process. If not specified, then process all demonstrations')
    parser.add_argument('--demonstrations_filter', help='Regex filter to only process demonstrations that match this filter')
    parser.add_argument('--session_name', default='', help='Process only demonstrations with this exact session name')
    parser.add_argument('--gopro_local_timezone', required=False, type=int, default=None, help='Assume GoPro time is in specified timezone offset in hours instead of UTC. For example you could pass "-8" to indicate that you are in PST time. This is to support backward compatibility with the previous UMI Day setup where we set the GoPro to use local time instead of UTC time. If you are using the new GoPro QR code with UTC time, you should not pass this flag.') # TODO: eventually we can delete this once we know longer use data from before the GoPro QR code was changed to UTC
    parser.add_argument('--visualize_til_frame', type=int, default=-1, help='Visualize until this frame in the video. -1 for all frames')
    args = parser.parse_args()

    stages = [stage for stage in args.stages if stage not in args.skip_stages]

    if 'group' in stages:
        print(color("--- GROUPING STAGE ---"))
        group_iphone_gopro_data(args.iphone_dir, None if args.skip_left else args.left_gopro_dir, None if args.skip_right else args.right_gopro_dir, args.out_dir, args.symlink, args.overwrite, args.demonstrations, args.demonstrations_filter, args.session_name)

    if 'gopro_timesync' in stages:
        print(color("\n--- GOPRO TIMESYNC STAGE ---"))
        gopro_timesync(args.out_dir, args.demonstrations, args.demonstrations_filter, args.session_name, args.overwrite)

    if 'align' in stages:
        print(color("\n--- ALIGN STAGE ---"))
        align_iphone_gopro_data(args.out_dir, args.demonstrations, args.demonstrations_filter, args.session_name, args.overwrite)

    if 'visualize' in stages:
        print(color("\n--- VISUALIZE STAGE ---"))
        visualize_iphone_gopro_data(args.out_dir, args.demonstrations, args.demonstrations_filter, args.session_name, args.overwrite, visualize_til_frame=args.visualize_til_frame)