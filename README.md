# EfficientPose-ROS
Implementation for running the EfficientPose 6D pose estimation network in a ROS node. The original implementation of EfficientPose can be found [here](https://github.com/ybkscht/EfficientPose).

## Requirements
This implementation has been written for [ROS Noetic](http://wiki.ros.org/noetic), which primarily targets Ubuntu 20.04. It has not been tested on other OS.
Instructions for installing ROS Noetic can be found [here](http://wiki.ros.org/noetic/Installation/Ubuntu).

EfficientPose-ROS requires Python 3.7 and venv:
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7-dev
sudo apt install python3.7-venv
```

## Installation Instructions
EfficientPose-ROS is built using catkin.
1. Navigate to your workspaces's `src` (ususally `~/catkin_ws/src`) and clone this repo using `git clone https://github.com/DavideFigundio/EfficientPose-ROS.git`.
2. Create a python 3.7 environment wherever is most convenient, then activate it.
```
python3.7 -m venv /path/to/env
source /path/to/env/bin/activate
```
To make sure that the interpreter for this environment is used by the node, change the first line in `EfficientPose-ROS/src/start_node.py` to:
```
#!/path/to/env/bin/python3.7
```
3. Install required libraries using `pip install -r EfficientPose-ROS/requirements.txt`.
4. Build cython modules using `python EfficientPose-ROS/src/setup.py build_ext --inplace`.
5. Build using `catkin build` or `catkin_make`.

## Configuration
Configuration options can be found and set inside the `start_node.py` script. Here are some of the available options:
- `phi` - the hyperparameter used to set the network's dimensions. Check the [original paper](https://arxiv.org/abs/2011.04307) for more information.
- `path_to_weights` - the locations where the pre-trained weights are stored. EfficientPose-ROS does not contain any functions for training. For that purpose, check the [original implementation](https://github.com/ybkscht/EfficientPose).
- `translation_scale_norm` - rescales the network output according to required measurment units. 1 corresponds to metres, 1000 to millimetres.
- `aruco_calibration_mode` - optional extrinsic camera calibration using [ArUco](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html) markers. Accepts three values:
    + `0` (default) - Disabled. No calibration is performed. All poses will be given in camera reference.
    + `1` - Single mode. Extrinsic calibration will be attempted only at startup. If a base frame is found, all inferences will then be given in that frame's reference.
    + `2` - Continuous mode. Extrinsic calibration will be attempted for each individual image.
Information on the marker must be given in the `aruco_dict`, `marker_length`, and `marker_ID` fields. If no marker is found, the node by default will give poses in camera reference.
- `image_topic_name`, `calibration_topic_name` - names of the ROS topics that the node will use to get images and camera intrinsics. Set these to your convenience. If you don't publish the intrinsics, you can change the camera matrix and distortion parameters' default values inside the `get_camera_params_from_topic` function in `efficientpose.py`.
- `publish_topic_name` - the name of the topic where the node will publish poses. Poses are published using a custom ObjectPoses message, and are also broadcasted using [tf2](http://wiki.ros.org/tf2).

## Launching the node
Assuming you have already launched a master using `roscore` or `roslaunch`:
1. Open a fresh terminal, navigate to your catkin workspace and activate ROS:
```
cd /path/to/workspace/
source /opt/ros/noetic/setup.bash # You can add this to your .bashrc and avoid inserting it every time.
source devel/setup.bash
```
2. If you haven't already done so, set `start_node.py` to executable
```
chmod +x src/EfficientPose-Ros/src/start_node.py
```
3. Launch the node using `rosrun`.
```
rosrun efficientpose_ros start_node.py
```
