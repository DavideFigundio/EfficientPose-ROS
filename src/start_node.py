#!./env/bin/python3.7


IMAGE_TOPIC_NAME = "/rgb/image_raw"
PUBLISH_TOPIC_NAME = "/poses"
ARUCO_ENABLED = True

import rospy
from sensor_msgs.msg import Image
from geometry_msgs.msg import Transform
from efficientpose_ros.msg import ObjectPoses
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import os
import math
import tensorflow as tf

from model import build_EfficientPose
from utils import preprocess_image


def start_node():
    
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    allow_gpu_growth_memory()
    bridge = CvBridge()

    # Model input parameters
    phi = 0
    path_to_weights = "weights.h5"
    class_to_name = {0 : "2-slot", 1 : "3-slot", 2 : "mushroombutton", 3 : "arrowbutton", 4 : "redbutton", 5 : "unknownbutton"}
    score_threshold = 0.5
    translation_scale_norm = 1.0     # conversion factor: 1 for m, 1000 for mm 

    num_classes = len(class_to_name)
    camera_matix, dist = get_camera_params()
    
    # Build model and load weights
    session = tf.Session()
    tf.python.keras.backend.set_session(session)
    graph = tf.get_default_graph()
    model, image_size = build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights)

    # ArUco detection parameters
    if ARUCO_ENABLED:
        aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
        aruco_params = cv2.aruco.DetectorParameters_create()
        marker_length = 0.067 # Length of marker in m
        marker_ID = 100
        aruco_data = [aruco_dict, aruco_params, marker_length, marker_ID]

    # Starting rospy node
    pub = rospy.Publisher(PUBLISH_TOPIC_NAME, ObjectPoses, queue_size=1)
    callback_args = [model, camera_matix, dist, image_size, translation_scale_norm, score_threshold, class_to_name, pub, bridge, graph, session, ARUCO_ENABLED, aruco_data]
    rospy.Subscriber(IMAGE_TOPIC_NAME, Image, inference, callback_args, queue_size=1)
    rospy.init_node("efficientpose", anonymous=True)

    rospy.spin()


def inference(image_message, args):

    # Parsing args
    model = args[0]
    camera_matrix = args[1]
    dist = args[2]
    image_size = args[3]
    translation_scale_norm = args[4]
    score_threshold = args[5]
    class_to_name = args[6]
    pub = args[7]
    bridge = args[8]
    graph = args[9]
    session = args[10]
    aruco_enabled = args[11]
    aruco_data = args[12]

    # Turning ROS image message into cv2 image, then into numpy array.
    color_image = np.asarray(bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough'))

    print("Inferencing...")
    # Removing alpha channel, remove if input image is pure RGB
    color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)
    
    base_rot = None
    base_trans = None
    if aruco_enabled:
        base_rot, base_trans = get_pose_from_aruco(color_image, aruco_data, camera_matrix, dist)

    # Undistorting the imgae
    color_image = cv2.undistort(color_image, camera_matrix, dist)
        
    # Preprocessing
    input_list, scale = preprocess(color_image, image_size, camera_matrix, translation_scale_norm)
        
    # Pose inference with EfficientPose
    with session.as_default():
        with graph.as_default():
            _, scores, labels, rotations, translations = model.predict_on_batch(input_list)
        
    # Postprocessing
    scores, labels, rotations, translations = postprocess(scores, labels, rotations, translations, scale, score_threshold)
    
    transforms = []
    for i in range(len(rotations)):
        # Turning rotation vectors into matrices
        rotation, _ = cv2.Rodrigues(rotations[i])
        
        # Transforming to base frame if detected 
        if base_rot.any() and base_trans.any():
            print("Base ArUco detected at position: " + str(base_trans))
            rotation, translation = change_reference_frame(rotation, translations[i], base_rot, base_trans)
        
        transforms.append(Transform(translation, Rotation.from_matrix(rotation).as_quat()))

    # Publishing to topic
    labels = [class_to_name[label] for label in labels]

    msg = ObjectPoses(labels, transforms)
    rospy.loginfo(msg)
    pub.publish(msg)


def get_pose_from_aruco(image, aruco_data, camera_matrix, distortion):
    '''
    Given an image and information on an ArUco marker and camera, returns the pose of the marker in the camera reference.
    Args:
        -image: numpy array containing pixel information.
        -aruco_data: list containing four elements:
            0) aruco_dict: dictionary of ArUcos to use, for example cv2.aruco.DICT_5X5_250
            1) aruco_params: parameters for the aruco detector
            2) marker_length: length of the marker side in m
            3) marker_id: int representing the ID of the marker to estimate the pose of
        -camera_matrix: 3x3 numpy array containing the camera parameters
        -distortion matrix: 1x3, 1x5 or 1x8 numpt array containing camera distortion parameters
    Returns:
        -rotation: 3x3 numpy array containing the rotation matrix to the ArUco frame, or None if no ArUcos with ID equal to marker_id were found.
        -translation: 1x3 numpy array containing the translation the the ArUco frame, or None if no ArUcos with ID equal to marker_id were found.
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

    # If no markers are detected, or the desired marker is not present, returns None.
    return None, None

def change_reference_frame(rotation, translation, rot_dest, trans_dest):
    transform_object = make4x4matrix(rotation, translation)
    transform_destination = make4x4matrix(rot_dest, trans_dest)

    new_transform = np.matmul(np.linalg.inv(transform_destination), transform_object)

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


def allow_gpu_growth_memory():
    """image_size
        Set allow growth GPU memory to true

    """
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    _ = tf.Session(config = config)


def get_camera_params(print_params=False):
    """
    Gets camera parameters from current Azure calibration settings.
    Args:
        calibration: A calibration object from an Azure Kinect camera.
        print: bool, if true prints the calibration parameters when executed.
    Returns:
        mat: 3x3 camera intrinsic matrix of the type:
            |fx  0   cx|
            |0   fy  cy|
            |0   0   1 |
        dist: camera distortion parameters in the form:
            [k1, k2, p1, p2, k3, k4, k5, k6]
    """

    cam_mat = np.array([[612.6460571289062, 0.0, 638.0296020507812], 
                    [0.0, 612.36376953125, 367.6560363769531],
                    [0.0, 0.0, 1.0]], dtype=np.float32)

    dist = np.array([0.5059323906898499, -2.6153206825256348, 0.000860791013110429, -0.0003529376117512584, 1.4836950302124023, 0.3840336799621582, -2.438732385635376, 1.4119256734848022], dtype = np.float32)

    return cam_mat, dist


def get_3d_bboxes():
    """
    Returns:
        name_to_3d_bboxes: Dictionary with the object names as keys and the cuboids as values

    """
    name_to_model_info = {"2-slot":  {"diameter": 137.11, "min_x": -34.0, "min_y": -52.8, "min_z": -27.5, "size_x": 68.0, "size_y": 105.6, "size_z": 55.0},
                          "3-slot":  {"diameter": 161.36, "min_x": -34.0, "min_y": -67.8, "min_z": -27.5, "size_x": 68.0, "size_y": 135.6, "size_z": 55.0},
                          "mushroombutton":  {"diameter": 21.4, "min_x": -25.92, "min_y": -19.97, "min_z": -19.99, "size_x": 51.87, "size_y": 39.92, "size_z": 39.96},
                          "arrowbutton":  {"diameter": 37.2738, "min_x": -13.32, "min_y": -14.25, "min_z": -14.25, "size_x": 26.65, "size_y": 28.5, "size_z": 28.5},
                          "redbutton":  {"diameter": 37.2738, "min_x": -13.32, "min_y": -14.25, "min_z": -14.25, "size_x": 26.65, "size_y": 28.5, "size_z": 28.5},
                          "unknownbutton":  {"diameter": 37.2738, "min_x": -13.32, "min_y": -14.25, "min_z": -14.25, "size_x": 26.65, "size_y": 28.5, "size_z": 28.5}}

    name_to_3d_bboxes = {name: convert_bbox_3d(model_info) for name, model_info in name_to_model_info.items()}
    
    return name_to_3d_bboxes


def convert_bbox_3d(model_dict):
    """
    Converts the 3D model cuboids from the Linemod format (min_x, min_y, min_z, size_x, size_y, size_z) to the (num_corners = 8, num_coordinates = 3) format
    Args:
        model_dict: Dictionary containing the cuboid information of a single Linemod 3D model in the Linemod format
    Returns:
        bbox: numpy (8, 3) array containing the 3D model's cuboid, where the first dimension represents the corner points and the second dimension contains the x-, y- and z-coordinates.

    """
    #get infos from model dict
    min_point_x = model_dict["min_x"]
    min_point_y = model_dict["min_y"]
    min_point_z = model_dict["min_z"]
    
    size_x = model_dict["size_x"]
    size_y = model_dict["size_y"]
    size_z = model_dict["size_z"]
    
    bbox = np.zeros(shape = (8, 3))
    #lower level
    bbox[0, :] = np.array([min_point_x, min_point_y, min_point_z])
    bbox[1, :] = np.array([min_point_x + size_x, min_point_y, min_point_z])
    bbox[2, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z])
    bbox[3, :] = np.array([min_point_x, min_point_y + size_y, min_point_z])
    #upper level
    bbox[4, :] = np.array([min_point_x, min_point_y, min_point_z + size_z])
    bbox[5, :] = np.array([min_point_x + size_x, min_point_y, min_point_z + size_z])
    bbox[6, :] = np.array([min_point_x + size_x, min_point_y + size_y, min_point_z + size_z])
    bbox[7, :] = np.array([min_point_x, min_point_y + size_y, min_point_z + size_z])
    
    return bbox


def build_model_and_load_weights(phi, num_classes, score_threshold, path_to_weights):
    """
    Builds an EfficientPose model and init it with a given weight file
    Args:
        phi: EfficientPose scaling hyperparameter
        num_classes: The number of classes
        score_threshold: Minimum score threshold at which a prediction is not filtered out
        path_to_weights: Path to the weight file
        
    Returns:
        efficientpose_prediction: The EfficientPose model
        image_size: Integer image size used as the EfficientPose input resolution for the given phi

    """
    print("\nBuilding model...\n")
    _, efficientpose_prediction, _ = build_EfficientPose(phi,
                                                         num_classes = num_classes,
                                                         num_anchors = 9,
                                                         freeze_bn = True,
                                                         score_threshold = score_threshold,
                                                         num_rotation_parameters = 3,
                                                         print_architecture = False)
    
    print("\nDone!\n\nLoading weights...")
    efficientpose_prediction.load_weights(path_to_weights, by_name=True)
    print("Done!")
    
    image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
    image_size = image_sizes[phi]
    
    return efficientpose_prediction, image_size


def preprocess(image, image_size, camera_matrix, translation_scale_norm):
    """
    Preprocesses the inputs for EfficientPose
    Args:
        image: The image to predict
        image_size: Input resolution for EfficientPose
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_list: List containing the preprocessed inputs for EfficientPose
        scale: The scale factor of the resized input image and the original image

    """
    image = image[:, :, ::-1]
    image, scale = preprocess_image(image, image_size)
    camera_input = get_camera_parameter_input(camera_matrix, scale, translation_scale_norm)
    
    image_batch = np.expand_dims(image, axis=0)
    camera_batch = np.expand_dims(camera_input, axis=0)
    input_list = [image_batch, camera_batch]
    
    return input_list, scale


def postprocess(scores, labels, rotations, translations, scale, score_threshold):
    """
    Filter out detections with low confidence scores and rescale the outputs of EfficientPose
    Args:
        scores: numpy array [batch_size = 1, max_detections] containing the confidence scores
        labels: numpy array [batch_size = 1, max_detections] containing class label
        rotations: numpy array [batch_size = 1, max_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [batch_size = 1, max_detections, 3] containing the translation vectors
        scale: The scale factor of the resized input image and the original image
        score_threshold: Minimum score threshold at which a prediction is not filtered out
    Returns:
        scores: numpy array [num_valid_detections] containing the confidence scores
        labels: numpy array [num_valid_detections] containing class label
        rotations: numpy array [num_valid_detections, 3] containing the axis angle rotation vectors
        translations: numpy array [num_valid_detections, 3] containing the translation vectors

    """
    scores, labels, rotations, translations = np.squeeze(scores), np.squeeze(labels), np.squeeze(rotations), np.squeeze(translations)
    #rescale rotations
    rotations *= math.pi
    #filter out detections with low scores
    indices = np.where(scores[:] > score_threshold)
    # select detections
    scores = scores[indices]
    rotations = rotations[indices]
    translations = translations[indices]
    labels = labels[indices]
    
    return scores, labels, rotations, translations


def get_camera_parameter_input(camera_matrix, image_scale, translation_scale_norm):
    """
    Return the input vector for the camera parameter layer
    Args:
        camera_matrix: numpy 3x3 array containing the intrinsic camera parameters
        image_scale: The scale factor of the resized input image and the original image
        translation_scale_norm: factor to change units. EfficientPose internally works with meter and if your dataset unit is mm for example, then you need to set this parameter to 1000
        
    Returns:
        input_vector: numpy array [fx, fy, px, py, translation_scale_norm, image_scale]

    """
    #input_vector = [fx, fy, px, py, translation_scale_norm, image_scale]
    input_vector = np.zeros((6,), dtype = np.float32)
    
    input_vector[0] = camera_matrix[0, 0]
    input_vector[1] = camera_matrix[1, 1]
    input_vector[2] = camera_matrix[0, 2]
    input_vector[3] = camera_matrix[1, 2]
    input_vector[4] = translation_scale_norm
    input_vector[5] = image_scale
    
    return input_vector

if __name__ == "__main__":
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass