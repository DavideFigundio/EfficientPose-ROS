import rospy
import geometry_msgs
from sensor_msgs.msg import Image, CameraInfo
from efficientpose_ros.msg import ObjectPoses
from efficientpose_ros.srv import GetPoses
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation
import cv2
import numpy as np
import os
import math
import tensorflow as tf
import geometry as geo
import tf2_ros as tf2

from model import build_EfficientPose
from utils import preprocess_image

class EfficientPoseROS:
    '''
    Class encapsulating the functions and features of a ROS node running EfficientPose for inferencing.
    '''

    def __init__(self, mode, phi, path_to_weights, class_to_name, score_threshold, translation_scale_norm, image_topic_name,  calibration_topic_name, publish_topic_name = None, service_name = None, aruco_calibration_mode = 0, arucodata = None, base_frame_name = None, aruco_frame_name = None):
        '''
        Creates and initializes a new EfficientPose ROS node.
        Args:
            mode - (int) determines whether to run in continuous (0) or asynchronous (1) mode.
            phi - (int) hyperparameter that sets dimensions of the neural network.
            path_to_weights - (str) path to a .h5 file containing trained weights for the network.
            class_to_name - dict[int -> str] associates object classes to their names
            score_threshold - (double) minimum confidence score required to identify object
            translation_scale_norm - (double) sets the output value's scale unit. 1=m, 100=cm, 1000=mm.
            image_topic_name - (str) name of the topic where the node will search for images to inference.
            calibration_topic_name - (str) name of the topic where the node will search for camera calibration parameters.
            publish_topic_name - (str) name of the topic where the node will publish inference results in continuous mode.
            service_name - (str) name of the service used to call the network in asynchronous mode.
            aruco_calibration_mode - (int) selects an extrinsic calibration method using an ArUco marker. Options are:
                0 - no external calibration. Poses will be given in camera reference.
                1 - one-time calibration at startup. Base pose will not be updated afterwards.
                2 - continuous calibration. Base pose will be updated for each image.
                3 - one-time calibration at startup using additional tf2 references.
            arucodata - [list] data used for calibration using ArUco markers. default = None, otherwise must contain in order:
                aruco_dict: the dictionary of ArUcos to use, for example cv2.aruco.DICT_5X5_250
                aruco_params: parameters for the aruco detector
                marker_length: length of the marker side in m
                marker_id: int representing the ID of the marker to estimate the pose of
            base_frame_name - (str) name of the frame corresponding to the base in tf2. Only used in calibration mode 3.
            aruco_frame_name - (str) name of the frame corresponding to the ArUco marker in tf2. Only used in calibration mode 3.
        '''

        rospy.init_node("efficientpose", anonymous=True)

        os.environ['CUDA_VISIBLE_DEVICES'] = '0'
        self.session = self.allow_gpu_growth_memory()
        tf.python.keras.backend.set_session(self.session)
        self.graph = tf.compat.v1.get_default_graph()
        self.bridge = CvBridge()
        self.image_topic_name = image_topic_name
        self.mode = mode

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
        if aruco_calibration_mode == 0:
            self.base_pose = None
            self.update_base_pose = False
        elif aruco_calibration_mode == 1:
            self.set_base_pose_from_aruco(arucodata, image_topic_name)
            self.update_base_pose = False
        elif aruco_calibration_mode == 2:
            self.base_pose = None
            self.update_base_pose = True
            self.aruco_data = arucodata
        elif aruco_calibration_mode == 3:
            self.set_base_pose_with_frames(arucodata, image_topic_name, base_frame_name, aruco_frame_name)
            self.update_base_pose = False
        else:
            print("Invalid ArUco calibration mode. Defaulting to mode 0.")
            self.base_pose = None
            self.update_base_pose = False

        # Setting up rospy communication channels
        self.broadcaster = tf2.TransformBroadcaster()

        # Continuous mode
        if self.mode == 0:
            self.publisher = rospy.Publisher(publish_topic_name, ObjectPoses, queue_size=1)
            rospy.Subscriber(image_topic_name, Image, self.inference, queue_size=1)

        # Asynchronous mode
        elif self.mode == 1:
            self.service = rospy.Service(service_name, GetPoses, self.get_image_and_inference)
            rospy.spin()

        else:
            print("Error! Invalid mode given. Must be either 0 for continuous mode or 1 for asynchronous mode.")
        
    def get_image_and_inference(self, arg):
        '''
        Handler for the service.
        Args:
            arg - (string) dummy argument that can be modified and used freely to give commands to the node.
        returns:
            names - [string] list of identified objects
            poses - [geometry_msgs.msg.Transform] list of corresponding poses for the identified objects.
        '''

        image_message = rospy.wait_for_message(self.image_topic_name, Image, 5)
        names, poses =  self.inference(image_message)
        return names, poses
    
    def inference(self, image_message):
        '''
        Used to perform inferencing.
        In synchronous mode, is called whenever a new image is found, and publishes on the self.publisher channel.
        In asynchronous mode, is called by the service handler and returns the necessary data.
        In both modes, broadcasts using tf2.
        Args:
            image_message -  (sensor_msgs.msg.Image) object containing the image.
        Returns (asynchronous mode only):
            labels - [string] list of identified objects
            transforms - [geometry_msgs.msg.Transform] list of corresponding poses for the identified objects.
        '''

        # Turning ROS image message into numpy array.
        color_image = np.asarray(self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough'))

        # Removing alpha channel, unnecessary if input image is pure RGB
        color_image = cv2.cvtColor(color_image, cv2.COLOR_BGRA2BGR)

        # Updating base pose if enabled
        if self.update_base_pose:
            self.update_base_pose_from_aruco(self.aruco_data, color_image)

        # Undistorting the image
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
            translation = translations[i]
            
            # Transforming to base frame if detected 
            if self.base_pose is not None:
                rotation, translation = geo.change_reference_frame(rotation, translation, self.base_pose)
            
            # Creating a Transform object
            name = self.class_to_name[labels[i]]
            transform_stamped = self.create_transform_stamped(rotation, translation, name)

            # Broadcasting
            self.broadcaster.sendTransform(transform_stamped)

            transforms.append(self.create_transform(rotation, translation))

        # Publishing to topic
        labels = [self.class_to_name[label] for label in labels]

        msg = ObjectPoses(labels, transforms)
        rospy.loginfo(msg)

        if self.mode == 0:
            self.publisher.publish(msg)
        elif self.mode == 1:
            return labels, transforms


    def set_base_pose_from_aruco(self, aruco_data, image_topic_name, timeout=5):
        '''
        Sets the pose of the world frame. Searches for an image on a topic, then looks for an ArUco marker in the image to use as reference
        Args:
            aruco_data - list of four elements:
                0) aruco_dict: dictionary of ArUcos to use, for example cv2.aruco.DICT_5X5_250
                1) aruco_params: parameters for the aruco detector
                2) marker_length: length of the marker side in m
                3) marker_id: int representing the ID of the marker to estimate the pose of
            image_topic_name - string, name of the topic where the function will search for the image.
            timeout - double, timeout for searching for an image.
        '''

        try:
            print("Performing extrinsic calibration:\n Searching for calibration image on topic " + image_topic_name)
             # Getting the image from the topic 
            image_message = rospy.wait_for_message(image_topic_name, Image, timeout)
            print(" Found image on topic. Searching for base frame...")
            image = np.asarray(self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough'))

            # Removing alpha channel, remove if input image is pure RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            self.base_pose = geo.get_pose_from_aruco(image, aruco_data, self.camera_matrix, self.distortion)
            
            if self.base_pose is None:
                print("Unable to find marker for extrinsic calibration. Poses will be in camera reference.")
            else:
                print("\nFound marker with pose:\n" + str(self.base_pose))

        except rospy.ROSException:
            # If no image is found, the world frame is considered to be the camera reference.
            print("Unable to get image for extrinsic calibration. Poses will be in camera reference.")
            return None

    def set_base_pose_with_frames(self, aruco_data, image_topic_name, base_frame_name, aruco_frame_name, timeout=5):
        '''
        Sets the pose of the world frame. Searches for an image on a topic, then looks for an ArUco marker in the image to use as reference
        Args:
            aruco_data - list of four elements:
                0) aruco_dict: dictionary of ArUcos to use, for example cv2.aruco.DICT_5X5_250
                1) aruco_params: parameters for the aruco detector
                2) marker_length: length of the marker side in m
                3) marker_id: int representing the ID of the marker to estimate the pose of
            image_topic_name - string, name of the topic where the function will search for the image.
            base_frame_name - (str) name of the frame corresponding to the base in tf2.
            aruco_frame_name - (str) name of the frame corresponding to the ArUco marker in tf2.
            timeout - double, timeout for searching for an image.
        '''

        try:
            print("Performing extrinsic calibration:\n Searching for calibration image on topic " + image_topic_name)
             # Getting the image from the topic 
            image_message = rospy.wait_for_message(image_topic_name, Image, timeout)
            print(" Found image on topic. Searching for base frame...")
            image = np.asarray(self.bridge.imgmsg_to_cv2(image_message, desired_encoding='passthrough'))

            # Removing alpha channel, remove if input image is pure RGB
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
            self.base_pose = geo.external_calibration_with_tf2(image, aruco_data, self.camera_matrix, self.distortion, base_frame_name, aruco_frame_name)
            
            if self.base_pose is None:
                print("Unable to find marker for extrinsic calibration. Poses will be in camera reference.")
            else:
                print("\nFound marker with pose:\n" + str(self.base_pose))

        except rospy.ROSException:
            # If no image is found, the world frame is considered to be the camera reference.
            print("Unable to get image for extrinsic calibration. Poses will be in camera reference.")
            return None

    def update_base_pose_from_aruco(self, aruco_data, image):
        '''
        Updates the pose of the world frame by looking for an ArUco marker in the provided image.
            aruco_data - list of four elements:
                0) aruco_dict: dictionary of ArUcos to use, for example cv2.aruco.DICT_5X5_250
                1) aruco_params: parameters for the aruco detector
                2) marker_length: length of the marker side in m
                3) marker_id: int representing the ID of the marker to estimate the pose of
            image - numpy array containing pixel data in RGB format.
        '''

        self.base_pose = geo.get_pose_from_aruco(image, aruco_data, self.camera_matrix, self.distortion)

        if self.base_pose is not None:
            print("\nFound marker with pose:\n" + str(self.base_pose))

        else:
            print("Unable to find marker for extrinsic calibration. Pose will be in camera reference.")


    def create_transform_stamped(self, rotation, translation, name):
        '''
        Creates a ros geometry transform time-stamped object from data.
        Args:
            rotation - 3x3 numpy array containing a rotation matrix
            translation - 1x3 numpy array containing a transalation vector
            name = string with the name of the object
        Returns:
            t - TransformStamped object containing position and temporal data.
        '''

        t = geometry_msgs.msg.TransformStamped()
        t.header.frame_id = 'world'
        t.child_frame_id = name
        t.header.stamp = rospy.Time.now()

        t.transform.translation.x = translation[0]
        t.transform.translation.y = translation[1]
        t.transform.translation.z = translation[2]

        quaternion = Rotation.from_matrix(rotation).as_quat()

        t.transform.rotation.x = quaternion[0]
        t.transform.rotation.y = quaternion[1]
        t.transform.rotation.z = quaternion[2]
        t.transform.rotation.w = quaternion[3]

        return t

    def create_transform(self, rotation, translation):
        '''
        Creates a ros geometry transform object from data.
        Args:
            rotation - 3x3 numpy array containing a rotation matrix
            translation - 1x3 numpy array containing a transalation vector
        Returns:
            t - Transform object containing a pose.
        '''
        t = geometry_msgs.msg.Transform()

        t.translation.x = translation[0]
        t.translation.y = translation[1]
        t.translation.z = translation[2]

        quaternion = Rotation.from_matrix(rotation).as_quat()

        t.rotation.x = quaternion[0]
        t.rotation.y = quaternion[1]
        t.rotation.z = quaternion[2]
        t.rotation.w = quaternion[3]

        return t


    def build_model_and_load_weights(self, path_to_weights):
        """
        Builds an EfficientPose model and initializes it with a given weight file.
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
            Sets allow growth GPU memory to true. Sets the current session.
        """
        config = tf.compat.v1.ConfigProto()
        config.gpu_options.allow_growth = True

        return tf.compat.v1.Session(config = config)


    def get_camera_params_from_topic(self, topic_name):
        """
        Gets camera parameters from a topic. If no parameters are found, will return default values.
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

        print("\nSearching for camera intrinsics on topic " + topic_name)
        try:
            data = rospy.wait_for_message('/rgb/camera_info/', CameraInfo, timeout=10)
            cam_mat = np.reshape(np.array(data.K), (3, 3))
            dist = np.array(data.D)
            print("\nFound data on topic:\n\nCamera matrix: \n" + str(cam_mat))
            print("\nDistortion parameters:\n" + str(dist))
        
        except rospy.ROSException:
            print("Unable to get data from topic. Using default values (Azure Kinect camera):")

            cam_mat = np.array([[612.6460571289062, 0.0, 638.0296020507812], 
                            [0.0, 612.36376953125, 367.6560363769531],
                            [0.0, 0.0, 1.0]], dtype=np.float32)

            dist = np.array([0.5059323906898499, -2.6153206825256348, 0.000860791013110429, -0.0003529376117512584, 1.4836950302124023, 0.3840336799621582, -2.438732385635376, 1.4119256734848022], dtype = np.float32)

            print("\nCamera matrix:\n" + str(cam_mat))
            print("\nDistortion parameters:\n" + str(dist))
        
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
        Filters out detections with low confidence scores and rescales the outputs of EfficientPose
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
        
        input_vector = np.zeros((6,), dtype = np.float32)
        
        input_vector[0] = self.camera_matrix[0, 0]
        input_vector[1] = self.camera_matrix[1, 1]
        input_vector[2] = self.camera_matrix[0, 2]
        input_vector[3] = self.camera_matrix[1, 2]
        input_vector[4] = self.translation_scale_norm
        input_vector[5] = image_scale
        
        return input_vector