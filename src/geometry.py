import cv2
import numpy as np

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
        rotation: 3x3 numpy array containing the rotation matrix to the ArUco frame, or None if no ArUcos with ID equal to marker_id were found.
        translation: 1x3 numpy array containing the translation the the ArUco frame, or None if no ArUcos with ID equal to marker_id were found.
    '''

    # Parsing args
    aruco_dict = aruco_data[0]
    aruco_params= aruco_data[1]
    marker_length = aruco_data[2]
    marker_id = aruco_data[3]

    # Detecting markers
    (corners, ids, _) = cv2.aruco.detectMarkers(image, aruco_dict, parameters=aruco_params)

    if ids.any():
        # Estimating poses
        rvects, tvects, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, camera_matrix, distortion)

        # Extracting the desired marker from the identified one
        if marker_id in ids:
            indexes = np.where(ids == marker_id)

            # If multiple of the same marker are present, only the first is considered.
            rotation, _ = cv2.Rodrigues(rvects[indexes[0]][0])
            translation = np.squeeze(tvects[indexes[0]][0])

            return rotation, translation

    # If no markers are detected, or the desired marker is not present, returns empty arrays.
    return [], []

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