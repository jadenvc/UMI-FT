import os
import shutil
import numpy as np
import zarr
from umift.processing.imagecodecs_numcodecs import JpegXl
from umift.processing.imagecodecs_numcodecs import register_codecs
register_codecs()

task_to_perform = 'merge_zarr'
# task_to_perform = 'remove_episode'
# task_to_perform = 'keep_only'

def copy_episode_group(src_group, dst_group):
    """
    Recursively copy all arrays/subgroups from one Zarr group (episode) to another.
    """
    # Copy arrays in the current group
    for arr_name, src_arr in src_group.arrays():
        # Create a corresponding array in the destination group
        dst_arr = dst_group.zeros(
            name=arr_name,
            shape=src_arr.shape,
            chunks=src_arr.chunks,
            dtype=src_arr.dtype
        )
        # Copy the data over
        dst_arr[:] = src_arr[:]
    
    # Recursively handle subgroups
    for subgrp_name, src_subgrp in src_group.groups():
        dst_subgrp = dst_group.require_group(subgrp_name)
        copy_episode_group(src_subgrp, dst_subgrp)

def merge_zarr_datasets(input_zarr_paths, output_zarr_path):
    """
    Merge multiple .zarr datasets with the same structure into one.

    Each input store has:
        /data
           episode_0/
           episode_1/
           ...
        /meta
           episode_gripper0_len/
           episode_rgb0_len/
           ...
           (each is a 1D array with shape (#episodes_in_this_store,))
    
    Steps:
      1. Overwrite or create the output zarr store.
      2. Copy all episodes from each input under /data in the output, renaming them consecutively.
      3. Concatenate 1D meta arrays (like 'episode_gripper0_len') across inputs, 
         producing a single array under /meta in the output.
    """
    # If the output directory already exists, remove it to start fresh
    if os.path.exists(output_zarr_path):
        print(f"[Info] Output directory {output_zarr_path} already exists. Removing it.")
        shutil.rmtree(output_zarr_path)

    # Create (or overwrite) the output store
    out_store = zarr.open(output_zarr_path, mode='w')
    out_data_group = out_store.create_group("data")
    out_meta_group = out_store.create_group("meta")

    # We'll store partial meta arrays from each input, then concatenate them
    meta_accumulator = {}

    # Keep track of how many episodes we have written so far
    total_episode_count = 0

    # Process each input Zarr file
    for in_path in input_zarr_paths:
        print(f"Merging from: {in_path}")
        in_store = zarr.open(in_path, mode='r')

        # --- Copy episodes from /data ---
        in_data_group = in_store["data"]
        # Typically, these are named "episode_0", "episode_1", etc.
        episode_names = sorted(in_data_group.group_keys())

        for ep_name in episode_names:
            print(f"In episode: {ep_name}")
            src_ep_group = in_data_group[ep_name]
            # Create a new episode group in the output store
            dst_ep_name = f"episode_{total_episode_count}"
            dst_ep_group = out_data_group.create_group(dst_ep_name)

            copy_episode_group(src_ep_group, dst_ep_group)
            total_episode_count += 1

        # --- Accumulate meta arrays for later concatenation ---
        in_meta_group = in_store["meta"]
        # Each meta array is a 1D array, shape = (# episodes in this dataset,)
        # e.g. "episode_gripper0_len"
        for arr_name in in_meta_group.array_keys():
            arr_data = in_meta_group[arr_name][:]  # read into memory
            if arr_name not in meta_accumulator:
                meta_accumulator[arr_name] = []
            meta_accumulator[arr_name].append(arr_data)

        print(f"  Copied {len(episode_names)} episodes from {in_path}\n")

    # --- Concatenate and write out meta arrays ---
    print("Concatenating meta arrays...")
    for arr_name, list_of_arrays in meta_accumulator.items():
        # For example, if the first dataset had shape (5,) and the second had shape (4,),
        # we get a list [array_of_length_5, array_of_length_4] => total length 9.
        merged_array = np.concatenate(list_of_arrays, axis=0)

        # Create a new dataset in the output meta group
        out_meta_group.create_dataset(
            name=arr_name,
            data=merged_array,
            shape=merged_array.shape,
            # optionally define chunks
            chunks=(merged_array.shape[0],),  # 1D chunk
            dtype=merged_array.dtype
        )
        print(f"  - {arr_name} => shape {merged_array.shape}")

    print("\nAll done!")
    print(f"Merged dataset is available at: {output_zarr_path}")


def remove_episodes(zarr_path, episodes_to_remove):
    """
    Remove selected episodes from a Zarr dataset *in-place*.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        episodes_to_remove (list[int]): List of episode indices to remove.
    """
    store = zarr.open(zarr_path, mode='a')  # Open in read/write mode
    data_group = store['data']
    meta_group = store['meta']

    print(f"[Info] Removing episodes: {episodes_to_remove}")

    # 1. Remove episode groups from /data
    for ep_idx in episodes_to_remove:
        ep_name = f"episode_{ep_idx}"
        if ep_name in data_group:
            print(f"  - Deleting {ep_name}")
            del data_group[ep_name]
        else:
            print(f"  - Warning: {ep_name} not found in data/")

    # 2. Update meta arrays by deleting entries
    for arr_name in meta_group.array_keys():
        meta_arr = meta_group[arr_name]
        print(f"  - Updating meta array: {arr_name}")

        # Read into memory, remove entries, then overwrite
        arr_data = meta_arr[:]
        updated_data = np.delete(arr_data, episodes_to_remove, axis=0)

        # Delete and re-create array in-place
        del meta_group[arr_name]
        meta_group.create_dataset(
            name=arr_name,
            data=updated_data,
            shape=updated_data.shape,
            chunks=(updated_data.shape[0],),
            dtype=updated_data.dtype
        )

    print(f"[Success] Removed {len(episodes_to_remove)} episodes from {zarr_path}")

def reindex_episodes(zarr_path):
    """
    Rename episode groups in /data to ensure continuous naming (episode_0, episode_1, ...).

    Args:
        zarr_path (str): Path to the Zarr store.
    """
    store = zarr.open(zarr_path, mode='a')
    data_group = store['data']

    # Get current episode names and their numeric indices
    old_names = sorted(data_group.group_keys(), key=lambda x: int(x.split('_')[1]))

    # Only rename if needed
    for new_idx, old_name in enumerate(old_names):
        new_name = f"episode_{new_idx}"
        if old_name != new_name:
            print(f"Renaming {old_name} → {new_name}")
            data_group.move(old_name, new_name)

    print(f"[Success] Episodes have been reindexed from 0 to {len(old_names)-1}")

def keep_only_episodes(zarr_path, episodes_to_keep):
    """
    Keep only the specified episodes from a Zarr dataset *in-place*.
    All other episodes and corresponding metadata entries will be removed.

    Args:
        zarr_path (str): Path to the Zarr dataset.
        episodes_to_keep (list[int]): List of episode indices to keep.
    """
    store = zarr.open(zarr_path, mode='a')
    data_group = store['data']
    meta_group = store['meta']

    all_episode_names = sorted(data_group.group_keys(), key=lambda x: int(x.split('_')[1]))
    all_episode_indices = [int(name.split('_')[1]) for name in all_episode_names]

    episodes_to_remove = sorted(set(all_episode_indices) - set(episodes_to_keep))
    print(f"[Info] Keeping episodes: {sorted(episodes_to_keep)}")
    print(f"[Info] Removing episodes: {episodes_to_remove}")

    # Reuse your existing function to delete the unwanted ones
    remove_episodes(zarr_path, episodes_to_remove)
    reindex_episodes(zarr_path)

    print(f"[Success] Now only {len(episodes_to_keep)} episodes remain in {zarr_path}")


if __name__ == "__main__":
    
    if task_to_perform == 'merge_zarr':
        input_zarrs = [
            "/store/real/hjchoi92/data/real/umift/zucchini-wild-b0/processed_data/all/acp_replay_buffer_gripper.zarr",
            "/store/real/hjchoi92/data/real/umift/zucchini-wild-b1/processed_data/all/acp_replay_buffer_gripper.zarr"
        ]
        output_zarr = "/store/real/hjchoi92/data/real_processed/umift/zucchini-wild-test-for-coderelease/acp_replay_buffer_gripper.zarr"

        print(f"Merging {len(input_zarrs)} input Zarr datasets into one...")

        merge_zarr_datasets(input_zarrs, output_zarr)
    elif task_to_perform == 'remove_episode':

        zarr_path = "/store/real/hjchoi92/data/real_processed/umift/lightbulb-cleanRelease-full-mirror/acp_replay_buffer_gripper.zarr"
        episodes_to_remove = [] # for [3, 30, 32] lightbulb-cleanRelease

        remove_episodes(zarr_path, episodes_to_remove)
        reindex_episodes(zarr_path)
    elif task_to_perform == 'keep_only':
        zarr_path = "/store/real/hjchoi92/data/real_processed/umift/WBW2-TEST/acp_replay_buffer_gripper.zarr"
        episodes_to_keep = [0, 5, 24, 29, 34, 39, 44, 49, 54, 59, 64, 69, 74, 79, 84, 89, 94, 134, 139, 144, 149, 154, 159, 164, 169, 174, 179, 184, 189, 194, 199, 204, 209, 214, 219, 224, 229, 234, 239, 244, 249, 254, 259, 264, 269, 274, 279, 284, 289, 294, 299, 304, 309, 314, 319, 324, 329, 334, 339, 344, 349, 354, 359, 364]

        keep_only_episodes(zarr_path, episodes_to_keep)

    else:
        print("Specify available task")
