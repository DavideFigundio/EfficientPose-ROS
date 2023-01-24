import rospy
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Transform
from efficientpose_ros.msg import ObjectPoses
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import os
import math
import tensorflow as tf
import geometry as geo

from model import build_EfficientPose
from utils import preprocess_image

class EfficientPoseROS:

    def __init__(self, phi, path_to_weights, class_to_name, score_threshold, translation_scale_norm, image_topic_name, calibration_topic_name, publish_topic_name, aruco_calibration = False, arucodata = None):
        rospy.init_node("efficientpose", anonymous=True)
        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.session = self.allow_gpu_growth_memory()
        tf.python.keras.backend.set_session(self.session)
        self.graph = tf.compat.v1.get_default_graph()
        self.bridge = CvBridge()

        # Model input parameters
        self.phi = phi
        self.class_to_name = class_to_name
        self.score_threshold = score_threshold
        self.translation_scale_norm = translation_scale_norm     # conversion factor: 1 for m, 1000 for mm 

        # Getting camera parameters
        self.camera_matrix, self.distortion = self.get_camera_params_from_topic(calibration_topic_name)
    
        # Build model and load weights
        self.model, self.image_size = self.build_model_and_load_weights(path_to_weights)

        # Extrinsic calibration using ArUco markers
        if aruco_calibration:
            self.base_pose = self.get_base_pose_from_aruco(arucodata, image_topic_name)

        self.publisher = rospy.Publisher(publish_topic_name, ObjectPoses, queue_size=1)
        rospy.Subscriber(image_topic_name, Image, self.inference, queue_size=1)
        

    
    def inference(self, image_message):

        # Turning ROS image message into cv2 image, then into numpy array.
        color_image = np.asarray(self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough'))

        print("Inferencing...")
        # Removing alpha channel, remove if input image is pure RGB
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        # Undistorting the imgae
        color_image = cv2.undistort(color_image, self.camera_matrix, self.distortion)
            
        # Preprocessing
        input_list = self.preprocess(color_image)
            
        # Pose inference with EfficientPose
        with self.session.as_default():
            with self.graph.as_default():
                _, scores, labels, rotations, translations = self.model.predict_on_batch(input_list)
            
        # Postprocessing
        scores, labels, rotations, translations = self.postprocess(scores, labels, rotations, translations)
        
        transforms = []
        for i in range(len(rotations)):
            # Turning rotation vectors into matrices
            rotation, _ = cv2.Rodrigues(rotations[i])
            
            # Transforming to base frame if detected 
            if self.base_pose is not None:
                rotation, translation = geo.change_reference_frame(rotation, translations[i], self.base_pose)
            else:
                translation = translations[i]
            
            transforms.append(Transform(translation, Rotation.from_matrix(rotation).as_quat()))

        # Publishing to topic
        labels = [self.class_to_name[label] for label in labels]

        msg = ObjectPoses(labels, transforms)
        rospy.loginfo(msg)
        self.publisher.publish(msg)


    def get_base_pose_from_aruco(self, aruco_data, image_topic_name):

        try:
            print("Performing extrinsic calibration:\n Searching for calibration image on topic " + image_topic_name)

            image_message = rospy.wait_for_message(image_topic_name, Image, timeout=5)
            print(" Found image on topic. Searching for base frame...")
            image = np.asarray(self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough'))

            # Removing alpha channel, remove if input image is pure RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            base_rot, base_trans = geo.get_pose_from_aruco(image, aruco_data, self.camera_matrix, self.distortion)

            if base_rot.any() and base_trans.any():
                print("Found marker at position " + str(base_trans))
                return geo.make4x4matrix(base_rot, base_trans)

            else:
                print("Unable to find marker for extrinsic calibration. Poses will be in camera reference.")
                return None

        except rospy.ROSException:
            print("Unable to get image for extrinsic calibration. Poses will be in camera reference.")
            return None


    def build_model_and_load_weights(self, path_to_weights):
        """
        Builds an EfficientPose model and init it with a given weight file
        Args:
            path_to_weights: Path to the weight file
            
        Returns:
            efficientpose_prediction: The EfficientPose model
            image_size: Integer image size used as the EfficientPose input resolution for the given phi

        """
        print("\nBuilding model...\n")
        num_classes = len(self.class_to_name)
        _, efficientpose_prediction, _ = build_EfficientPose(self.phi,
                                                            num_classes = num_classes,
                                                            num_anchors = 9,
                                                            freeze_bn = True,
                                                            score_threshold = self.score_threshold,
                                                            num_rotation_parameters = 3,
                                                            print_architecture = False)
        
        print("\nDone!\n\nLoading weights...")
        efficientpose_prediction.load_weights(path_to_weights, by_name=True)
        print("Done!")
        
        image_sizes = (512, 640, 768, 896, 1024, 1280, 1408)
        image_size = image_sizes[self.phi]
        
        return efficientpose_prediction, image_size


    def allow_gpu_growth_memory(self):
        """
            Sets allow growth GPU memory to true.

        """
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        return tf.compat.v1.Session(config = config)


    def get_camera_params_from_topic(self, topic_name):
        """
        Gets camera parameters.
        Args:
            topic_name: string, name of the topic where the camera intrinsics are published.
        Returns:
            mat: 3x3 camera intrinsic matrix of the type:
                |fx  0   cx|
                |0   fy  cy|
                |0   0   1 |
            dist: camera distortion parameters in the form:
                [k1, k2, p1, p2, k3, k4, k5, k6]
        """

        print("Waiting for camera intrinsics on topic " + topic_name)
        try:
            data = rospy.wait_for_message('/rgb/camera_info/', CameraInfo, timeout=10)
            cam_mat = np.reshape(np.array(data.K), (3, 3))
            dist = np.array(data.D)
            print("Camera matrix:\n" + str(cam_mat))
            print("Distortion parameters:\n" + str(dist))
        
        except rospy.ROSException:
            print("Unable to get data from topic. Using default values (Azure Kinect camera).")

            cam_mat = np.array([[612.6460571289062, 0.0, 638.0296020507812], 
                            [0.0, 612.36376953125, 367.6560363769531],
                            [0.0, 0.0, 1.0]], dtype=np.float32)

            dist = np.array([0.5059323906898499, -2.6153206825256348, 0.000860791013110429, -0.0003529376117512584, 1.4836950302124023, 0.3840336799621582, -2.438732385635376, 1.4119256734848022], dtype = np.float32)
        
        return cam_mat, dist


    def preprocess(self, image):
        """
        Preprocesses the inputs for EfficientPose
        Args:
            image: The image to predict
            
        Returns:
            input_list: List containing the preprocessed inputs for EfficientPose

        """
        image = image[:, :, ::-1]
        image, scale = preprocess_image(image, self.image_size)
        camera_input = self.get_camera_parameter_input(scale)
        
        image_batch = np.expand_dims(image, axis=0)
        camera_batch = np.expand_dims(camera_input, axis=0)
        input_list = [image_batch, camera_batch]
        
        return input_list


    def postprocess(self, scores, labels, rotations, translations):
        """
        Filter out detections with low confidence scores and rescale the outputs of EfficientPose
        Args:
            scores: numpy array [batch_size = 1, max_detections] containing the confidence scores
            labels: numpy array [batch_size = 1, max_detections] containing class label
            rotations: numpy array [batch_size = 1, max_detections, 3] containing the axis angle rotation vectors
            translations: numpy array [batch_size = 1, max_detections, 3] containing the translation vectors
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
        indices = np.where(scores[:] > self.score_threshold)
        # select detections
        scores = scores[indices]
        rotations = rotations[indices]
        translations = translations[indices]
        labels = labels[indices]
        
        return scores, labels, rotations, translations


    def get_camera_parameter_input(self, image_scale):
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
        
        input_vector[0] = self.camera_matrix[0, 0]
        input_vector[1] = self.camera_matrix[1, 1]
        input_vector[2] = self.camera_matrix[0, 2]
        input_vector[3] = self.camera_matrix[1, 2]
        input_vector[4] = self.translation_scale_norm
        input_vector[5] = image_scale
        
        return input_vector