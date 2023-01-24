# EfficientPose-ROS
Implementation for running the EfficientPose 6D pose estimation network in a ROS node. The original repo for EfficientPose can be found [here](https://github.com/ybkscht/EfficientPose).

## Requirements
This implementation has been written for [ROS Noetic](http://wiki.ros.org/noetic), which primarily targets Ubuntu 20.04. It has not been tested on other OS.
Instructions for installing ROS Noetic can be found [here](http://wiki.ros.org/noetic/Installation/Ubuntu).

EfficientPose requires Python 3.7:
```
sudo apt update
sudo apt install software-properties-common
sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt install python3.7
```

## Installation Instructions
1. Navigate to your workspaces's `src` (`/path/to/workspace/src`) and clone this repo.
2. Navigate to the repo source and create a python 3.7 environment, then activate it.
```
cd EfficientPose-ROS/src
python3.7 -m venv env
source env/bin/activate
```
3. Install required libraries with pip:
```
pip install -r ../requirements.txt
```


