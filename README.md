# My_project_work_ws

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
Welcome to **My_project_work_ws** – the repository for our project in **The 4th International University Student EV Autonomous Driving Competition**.  
This project focuses on developing an advanced autonomous driving system for electric vehicles, incorporating state-of-the-art algorithms for camera vision system, tunnel detection system and decision making(judgement).  
**Developed using ROS1 Noetic and tested on Ubuntu 20.04.**

## Repository Structure
Below is an overview of the repository structure, including the `camera_lane_segmentation` package and its subfolders: My_project_work_ws/ ├── src/ # Source code of the project (ROS packages) │ ├── camera_lane_segmentation/ # Lane segmentation package │ │ ├── scripts/ # Python scripts for lane detection, etc. │ │ └── utils/ # Utility modules and helper functions │ ├── [other_packages]/ # Other ROS packages (if any)└── README.md # Project overview and guidelines (this file)

## Features
- **Autonomous Navigation:** Advanced algorithms for reliable autonomous driving.
- **Real-Time Control:** Dynamic control systems that adapt to varying driving scenarios.
- **Modular Design:** Clean code architecture designed for scalability and ease of maintenance.
- **Data-Driven Improvements:** Continuous performance enhancements through collected sensor data analysis.

## Installation and Setup

### Prerequisites
- Ubuntu 20.04(Only Tested but can be runned former or later versions)
- ROS1 Noetic installed and properly configured
- Necessary libraries and dependencies as specified in individual package manifests

### Installation Steps
Clone the repository:
```bash
git clone https://github.com/Highsky7/My_project_work_ws.git
cd My_project_work_ws
 ### Run the Project
rosrun (package_name) ~.py
