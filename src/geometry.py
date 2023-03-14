import cv2
import numpy as np
import rospy
import tf2_ros
from scipy.spatial.transform import Rotation

'''
Contains useful functions for working with geometric transforms.
'''

def external_calibration_with_tf2(image, aruco_data, camera_matrix, distortion, base_frame_name, aruco_frame_name):
    '''
    Performs external calibration using both and ArUco marker and a pair of tf2 transforms.
    Args:
        image: numpy array containing pixel information.
        aruco_data: list containing four elements:
            0) aruco_dict: dictionary of ArUcos to use, for example cv2.aruco.DICT_5X5_250
            1) aruco_params: parameters for the aruco detector
            2) marker_length: length of the marker side in m
            3) marker_id: int representing the ID of the marker to estimate the pose of
        camera_matrix: 3x3 numpy array containing the camera parameters
        distortion matrix: 1x3, 1x5 or 1x8 numpt array containing camera distortion parameters
        base_frame_name: name of the tf2 transform that will be used as a base frame
        aruco_frame_name: name of the tf2 transform corresponding to the ArUco marker.
    Returns:
        pose: 4x4 numpy array containing a homogeneous transform. If all parts were successful, this is the transform from the
                camera to the base frame, otherwise this is None.
    '''
    
    camera_to_aruco = get_pose_from_aruco(image, aruco_data, camera_matrix, distortion)

    base_to_tcp = get_transform(base_frame_name, aruco_frame_name)

    if camera_to_aruco is not None and base_to_tcp is not None:
        return np.matmul(camera_to_aruco, np.linalg.inv(base_to_tcp))
    
    return None

def get_pose_from_aruco(image, aruco_data, camera_matrix, distortion):
    '''
    Given an image and information on an ArUco marker and camera, returns the pose of the marker in the camera reference.
    Args:
        image: numpy array containing pixel information.
        aruco_data: list containing four elements:
            0) aruco_dict: dictionary of ArUcos to use, for example cv2.aruco.DICT_5X5_250
            1) aruco_params: parameters for the aruco detector
            2) marker_length: length of the marker side in m
            3) marker_id: int representing the ID of the marker to estimate the pose of
        camera_matrix: 3x3 numpy array containing the camera parameters
        distortion matrix: 1x3, 1x5 or 1x8 numpt array containing camera distortion parameters
    Returns:
        pose: 4x4 numpy array containing a homogeneous transform. If all parts were successful, this is the transform from the
                camera to the ArUco marker, otherwise this is None.
    '''

    # Parsing args
    aruco_dict = aruco_data[0]
    aruco_params= aruco_data[1]
    marker_length = aruco_data[2]
    marker_id = aruco_data[3]

    # Detecting markers
    (corners, ids, _) = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    if ids is not None:
        # Estimating poses
        rvects, tvects, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, distortion)

        # Extracting the desired marker from the identified one
        if marker_id in ids:
            indexes = np.where(ids == marker_id)

            # If multiple of the same marker are present, only the first is considered.
            rotation, _ = cv2.Rodrigues(rvects[indexes[0]][0])
            translation = np.squeeze(tvects[indexes[0]][0])

            return make4x4matrix(rotation, translation)

    # If no markers are detected, or the desired marker is not present, returns empty arrays.
    return None

def get_transform(origin_frame_name, frame_name, timeout = 5):
    '''
    Gets the trasform between two tf2 frames.
    Args:
        origin_frame_name: name of the tf2 transform corresponding to the origin frame
        frame_name: name of the tf2 transform corresponding to the destination frame.
    Returns:
        pose: 4x4 numpy array containing a homogeneous transform. If all parts were successful, this is the transform from the
                origin frame to the destination frame, otherwise this is None.
    '''

    print("Getting robot TCP position...")
    tfbuffer = tf2_ros.Buffer()
    tf2_ros.TransformListener(tfbuffer)

    try:
        tcp = tfbuffer.lookup_transform(origin_frame_name, frame_name, rospy.Time(0), rospy.Duration(timeout))
    except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
        print("Robot TCP not found.")
        return None
    
    translation = np.array([tcp.transform.translation.x, tcp.transform.translation.y, tcp.transform.translation.z])
    quaternion = [tcp.transform.rotation.x, tcp.transform.rotation.y, tcp.transform.rotation.z, tcp.transform.rotation.w]

    rotation = Rotation.from_quat(quaternion).as_matrix()

    return make4x4matrix(rotation, translation)    


def change_reference_frame(rotation, translation, pose):
    """
    Transforms the given rotation and translation into a new reference frame.
    Args:
        rotation - 3x3 numpy array containing the rotation matrix
        translation - 1x3 numpy array containing the translation vector
        pose -  4x4 numpy array containing the homogeneous transform to the new reference
    returns:
        rotation, translation
    """
    transform_object = make4x4matrix(rotation, translation)
    new_transform = np.matmul(np.linalg.inv(pose), transform_object)

    return unmake4x4matrix(new_transform)

def make4x4matrix(rotation, translation):
    """
    Function that turns a rotation matrix and translation  vector into a homogeneous transform.
    Args:
        rotation - 3x3 numpy array containing the rotation matrix
        translation - 1x3 numpy array containing the components of the translation vector
    Returns
        mat - 4x4 numpy array containing the homogeneous transform matrix
    """

    mat = np.append(rotation, np.transpose(np.expand_dims(translation, axis=0)), axis=1)
    mat = np.append(mat, np.array([[0., 0., 0., 1.]], dtype=np.float32), axis=0)

    return mat

def unmake4x4matrix(matrix):
    """
    Function that extracts rotation and translation from a homogeneous transform.
    Args:
        matrix - 4x4 numpy array containing the homogeneous transform
    Returns:
        rotation - x3 numpy array containing the rotation matrix
        translation - 1x3 numpy array containing the components of the translation vector
    """

    rotation = np.transpose(np.transpose(matrix[:3])[:3])
    translation = np.transpose(matrix[:3])[3]

    return rotation, translation