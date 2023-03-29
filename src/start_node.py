#!/home/davide/envs/efficientpose-ros/bin/python3.7

import rospy
import cv2
from efficientpose import EfficientPoseROS

'''
Creates and starts a new EfficientPose ROS node.
'''

def start_node():
    mode = 1 # 0 for continuous, 1 for asynchronous

    # Model input parameters
    phi = 0
    path_to_weights = "./src/EfficientPose-ROS/src/weights.h5"
    class_to_name = {0 : "2-slot", 1 : "3-slot", 2 : "mushroombutton", 3 : "arrowbutton", 4 : "redbutton", 5 : "unknownbutton"}
    score_threshold = 0.5
    translation_scale_norm = 1.0     # conversion factor: 1 for m, 100 for cm, 1000 for mm

    # ArUco parameters
    aruco_calibration_mode = 3 # 0: No calibration | 1: Only at startup | 2: Continuous | 3: Startup with tf2
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    marker_length = 0.067 # Length of marker in m
    marker_ID = 100
    aruco_data = [aruco_dict, aruco_params, marker_length, marker_ID]

    # Reference tf2 frames
    base_frame_name = "base_0"
    aruco_frame_name = "gripper_reference"

    # Topics
    image_topic_name = "/rgb/image_raw"
    calibration_topic_name = "/rgb/camera_info"
    publish_topic_name = "/poses"
    service_name = "/get_poses"

    # Creation of the node
    EfficientPoseROS(mode,
                     phi, 
                     path_to_weights, 
                     class_to_name, 
                     score_threshold, 
                     translation_scale_norm, 
                     image_topic_name, 
                     calibration_topic_name, 
                     publish_topic_name=publish_topic_name,
                     service_name=service_name, 
                     aruco_calibration_mode=aruco_calibration_mode, 
                     arucodata=aruco_data, 
                     base_frame_name=base_frame_name, 
                     tcp_frame_name=aruco_frame_name)

    rospy.spin()

if __name__ == "__main__":
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass