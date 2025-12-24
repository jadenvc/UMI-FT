import numpy as np
from scipy.spatial.transform import Rotation

def fm2wrench(force_moment):
    """
    Convert a 6D [fx, fy, fz, mx, my, mz] to [mx, my, mz, fx, fy, fz]
    """
    wrench = np.array(force_moment)[3:].tolist() + np.array(force_moment)[:3].tolist()
    wrench = np.array(wrench)
    return wrench

def wrench2fm(wrench):
    """
    Convert a 6D [mx, my, mz, fx, fy, fz] to [fx, fy, fz, mx, my, mz]
    """
    force_moment = np.array(wrench)[3:].tolist() + np.array(wrench)[:3].tolist()
    force_moment = np.array(force_moment)
    return force_moment

def skew_symmetric(vector):
    """
    Args:
        vector: 3x1 vector
    Compute the skew symmetric matrix of a 3x1 vector, 3x3 matrix
    """
    skew_symmetric_mat = np.zeros((3,3))
    skew_symmetric_mat[0,1] = -vector[2]
    skew_symmetric_mat[0,2] = vector[1]
    skew_symmetric_mat[1,0] = vector[2]
    skew_symmetric_mat[1,2] = -vector[0]
    skew_symmetric_mat[2,0] = -vector[1]
    skew_symmetric_mat[2,1] = vector[0]
    assert np.allclose(skew_symmetric_mat, -skew_symmetric_mat.T)
    return skew_symmetric_mat

def adjoint(transform):
    """
    Args:
        transform: 4x4 transformation matrix
        # rotation: 3x3 rotation matrix
        # translation: 3x1 translation vector
    Compute the adjoint of a 6x6 adjoint matrix
    """
    rotation = transform[:3,:3]
    translation = transform[:3,3]
    adjoint = np.zeros((6,6))
    adjoint[:3,:3] = rotation
    adjoint[3:,3:] = rotation
    adjoint[3:,:3] = skew_symmetric(translation)@rotation
    
    return adjoint

CONST_TRANSFORM = {'gopro2tcp': np.array([[1, 0, 0, 0], 
                                          [0, 1, 0, 0.086], 
                                          [0, 0, 1, 0.14465], 
                                          [0, 0, 0, 1]]),
                   'left_coinft2tcp': np.array([[1, 0, 0, 0], 
                                                 [0, 1, 0, 0], 
                                                 [0, 0, 1, 0], 
                                                 [0, 0, 0, 1]]),
                    'right_coinft2tcp': np.array([[1, 0, 0, 0], 
                                                  [0, 1, 0, 0], 
                                                  [0, 0, 1, 0], 
                                                  [0, 0, 0, 1]]),
                    'dist_coinft2gopro_along_cam_z': 0.152935,
                    'dist_coinft2gopro_along_cam_y': 0.086 #TODO
                    }


def transform_coinft_l2tcp(gripper_left_x):
    """
    Args:
        gripper_left_x: distance, abs
    """
    R_left_ft2gopro = Rotation.from_euler('y', -90, degrees=True).as_matrix()
    T_left_ft2gopro = np.eye(4)
    T_left_ft2gopro[:3, :3] = R_left_ft2gopro
    T_left_ft2gopro[:3, 3] = np.array([CONST_TRANSFORM['dist_coinft2gopro_along_cam_z'], 
                                       -1*CONST_TRANSFORM['dist_coinft2gopro_along_cam_y'], 
                                       gripper_left_x])
    T_coinft_left2tcp = T_left_ft2gopro  
    return T_coinft_left2tcp

def transform_coinft_r2cp(gripper_right_x):
    """
    Args:
        gripper_right_x: distance, abs
    """
    R_right_ft2gopro = Rotation.from_euler('ZY', [180,90], degrees=True).as_matrix()
    T_right_ft2gopro = np.eye(4)
    T_right_ft2gopro[:3, :3] = R_right_ft2gopro
    T_right_ft2gopro[:3, 3] = np.array([CONST_TRANSFORM['dist_coinft2gopro_along_cam_z'], 
                                       CONST_TRANSFORM['dist_coinft2gopro_along_cam_y'], 
                                       gripper_right_x])
    T_coinft_right2tcp = T_right_ft2gopro 
    return T_coinft_right2tcp