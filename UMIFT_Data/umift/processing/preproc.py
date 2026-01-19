import numpy as np
def gripper_width_determin_optimal_alignment(demo_video_meta_df, start_timestamp):
    # determine optimal alignment
    dt = None
    alignment_costs = list()
    for cam_idx, row in demo_video_meta_df.iterrows():
        dt = 1 / row['fps']
        this_alignment_cost = list()
        for other_cam_idx, other_row in demo_video_meta_df.iterrows():
            # what's the delay for previous frame
            diff = other_row['start_timestamp'] - row['start_timestamp']
            remainder = diff % dt
            this_alignment_cost.append(remainder)
        alignment_costs.append(this_alignment_cost)
    align_cam_idx = np.argmin([sum(x) for x in alignment_costs])

    # rewrite start_timestamp to be integer multiple of dt
    align_video_start = demo_video_meta_df.loc[align_cam_idx]['start_timestamp']
    start_timestamp += dt - ((start_timestamp - align_video_start) % dt)
    return start_timestamp
    
def get_gripper_width(tag_dict, left_id, right_id, nominal_z=0.072, z_tolerance=0.008):
    zmax = nominal_z + z_tolerance
    zmin = nominal_z - z_tolerance

    left_x = None
    if left_id in tag_dict:
        tvec = tag_dict[left_id]['tvec']
        # check if depth is reasonable (to filter outliers)
        if zmin < tvec[-1] < zmax:
            left_x = tvec[0]

    right_x = None
    if right_id in tag_dict:
        tvec = tag_dict[right_id]['tvec']
        if zmin < tvec[-1] < zmax:
            right_x = tvec[0]

    width = None
    if (left_x is not None) and (right_x is not None):
        width = right_x - left_x
    elif left_x is not None:
        width = abs(left_x) * 2
    elif right_x is not None:
        width = abs(right_x) * 2
    return width
