cmake_minimum_required(VERSION 2.8.3)
project(avm)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
)
find_package(OpenCV 3.1.0 REQUIRED)
find_package(Ceres REQUIRED)
# find_package(eigen3 REQUIRED)

catkin_package(
#  INCLUDE_DIRS include
#  LIBRARIES avm
#  CATKIN_DEPENDS roscpp rospy std_msgs
#  DEPENDS system_lib
)
set( CMAKE_BUILD_TYPE Release )
include_directories(
  eigen3
  include
  ${catkin_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  
)

add_library(image_process_lib src/image_process.cpp)
add_library(lane src/lane.cpp)
# add_library(lane_fit src/lane_fit.cpp)
target_link_libraries(lane  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})
# target_link_libraries(lane_fit  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} ${CERES_LIBRARIES})

add_executable(avm_node src/main.cpp)
target_link_libraries(avm_node  ${catkin_LIBRARIES} ${OpenCV_LIBRARIES} image_process_lib lane)