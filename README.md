# My_project_work_ws

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## Overview
Welcome to **My_project_work_ws** – the repository of my works in **The 4th International University Student EV Autonomous Driving Competition**.  
This project focuses on developing an advanced autonomous driving system for electric vehicles, incorporating state-of-the-art algorithms for camera vision system, tunnel detection system with lidar sensor and decision making(judgement).  
**Developed using ROS1 Noetic and tested on Ubuntu 20.04.**

## Features
- **Camera Vision System** Using Open Source Model YOLOPv2 and modifying it to run in real time and to integrate with ROS
- **Lidar Using Tunnel Detection System** Making prototype modules for tunnel detection systems with VLP-16 Lidar
- **Modular Design:** Clean code architecture designed for scalability and ease of maintenance.

## Installation and Setup

### Prerequisites
- Ubuntu 20.04(Only Tested but can run former or later versions)
- ROS1 Noetic installed and properly configured
- Necessary libraries and dependencies as specified in individual package manifests

### Installation Steps
Clone the repository:
```bash
git clone https://github.com/Highsky7/My_project_work_ws.git
cd My_project_work_ws
```
### Running and Testing
How to Run the Codes with ROS, use the following command:
```bash
rosrun (package_name) ~.py
```
