cmake_minimum_required(VERSION 2.8.3)
project(vpd)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)


catkin_package(

)


include_directories(
include
  ${catkin_INCLUDE_DIRS}
)

find_package(OpenCV 3.1.0 REQUIRED)
add_library(undistor_lib src/undistor_tool.cpp)
target_link_libraries(undistor_lib  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_library(detection_lib src/detection.cpp)
target_link_libraries(detection_lib  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(lpd_node src/main.cpp)

target_link_libraries(lpd_node  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} undistor_lib detection_lib)
