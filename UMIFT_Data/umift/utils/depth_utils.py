# Depth utilities
import numpy as np
import cv2

def load_depth(depth_path, depth_shape=(192, 256), dtype=np.float16) -> np.ndarray:
    """Returns np array of depth data of dtype `dtype` and shape (T, depth_shape[0], depth_shape[1])"""
    # Load the raw file as a numpy array
    with open(depth_path, 'rb') as f:
        depth_data = np.frombuffer(f.read(), dtype=dtype)

    # Reshape the data to the specified shape
    depth_array = depth_data.reshape((-1, *depth_shape))
    return depth_array

def get_depth_transform_with_border(in_res, out_res):
    """
    Transform depth image to bordered square and resize while keeping float16
    Assumes input image shape: (H, W)
    """
    ih, iw = in_res  # width, height
    oh, ow = out_res  # should both be 224

    def transform(depth_img: np.ndarray):
        assert depth_img.shape == (ih, iw)
        assert depth_img.dtype == np.float16

        # Compute padding
        size = max(iw, ih)
        top = (size - ih) // 2
        bottom = size - ih - top
        left = (size - iw) // 2
        right = size - iw - left

        # Pad with 0 (assumes 0 = invalid depth)
        depth_padded = np.pad(depth_img, ((top, bottom), (left, right)), mode='constant', constant_values=0)

        # Resize using float32 interpolation (cv2 doesn't support float16 directly)
        depth_padded = depth_padded.astype(np.float32)
        resized = cv2.resize(depth_padded, dsize=out_res, interpolation=cv2.INTER_LINEAR)

        return resized.astype(np.float16)

    return transform

def stack_depth_channels(depth_array: np.ndarray) -> np.ndarray:
    """
    Stack a (N, H, W) depth array across 3 channels to get shape (N, H, W, 3)

    Args:
        depth_array (np.ndarray): input array of shape (N, H, W), dtype float16

    Returns:
        np.ndarray: stacked array of shape (N, H, W, 3), dtype float16
    """
    assert depth_array.ndim == 3, f"Expected input of shape (N, H, W), got {depth_array.shape}"
    assert depth_array.dtype == np.float16, f"Expected float16 dtype, got {depth_array.dtype}"
    
    return np.repeat(depth_array[..., np.newaxis], 3, axis=-1)
