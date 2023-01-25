#!./env/bin/python3.7

import rospy
import cv2
from efficientpose import EfficientPoseROS

'''
Creates and starts a new EfficientPose ROS node.
'''

def start_node():
    # Model input parameters
    phi = 0
    path_to_weights = "weights.h5"
    class_to_name = {0 : "2-slot", 1 : "3-slot", 2 : "mushroombutton", 3 : "arrowbutton", 4 : "redbutton", 5 : "unknownbutton"}
    score_threshold = 0.5
    translation_scale_norm = 1.0     # conversion factor: 1 for m, 100 for cm, 1000 for mm

    # ArUco parameters
    do_aruco_calibration = True
    aruco_dict = cv2.aruco.Dictionary_get(cv2.aruco.DICT_5X5_250)
    aruco_params = cv2.aruco.DetectorParameters_create()
    marker_length = 0.067 # Length of marker in m
    marker_ID = 100
    aruco_data = [aruco_dict, aruco_params, marker_length, marker_ID]

    # Topics
    image_topic_name = "/rgb/image_raw"
    calibration_topic_name = "/rgb/camera_info"
    publish_topic_name = "/poses"

    # Creation of the node
    EfficientPoseROS(phi, path_to_weights, class_to_name, score_threshold, translation_scale_norm, image_topic_name, calibration_topic_name, publish_topic_name, aruco_calibration=do_aruco_calibration, arucodata=aruco_data)

    rospy.spin()

if __name__ == "__main__":
    try:
        start_node()
    except rospy.ROSInterruptException:
        pass