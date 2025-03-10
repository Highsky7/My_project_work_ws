cmake_minimum_required(VERSION 3.0.2)
project(wall_detection)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  pcl_conversions
  pcl_ros
  roscpp
  sensor_msgs
  std_msgs
  visualization_msgs
)

catkin_package(
  CATKIN_DEPENDS geometry_msgs pcl_conversions pcl_ros roscpp sensor_msgs std_msgs visualization_msgs
)

include_directories(
  ${catkin_INCLUDE_DIRS}
)

# 실행 파일 추가
add_executable(two_walls_detection_node src/two_walls_detection_node.cpp)

# 의존성 링크
target_link_libraries(two_walls_detection_node
  ${catkin_LIBRARIES}
)