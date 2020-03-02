#include <ros/ros.h>
#include <iostream>
#include <vector>
#include "detection.h"
#include "undistor_tool.h"
std::string inner_parameter = "/home/rzc/my_ws/data/vehicle_position_detect.avi";
std::string perspect_matrix_path = "/home/rzc/my_ws/src/vpd/src/perspective.yaml";

int main(int argc, char**argv){
    ros::init(argc, argv, "avm_node");
    ros::NodeHandle h_node;
    cv::Mat input = cv::imread("/home/rzc/my_ws/src/vpd/detection_sample/435.bmp", 1);
    // distort 
    // Undistor::Ptr undistor_ptr = std::make_shared<Undistor>(input);
    // auto result = undistor_ptr->distirt_image();
    // detection
    Detection::Ptr detector_ptr = std::make_shared<Detection>(perspect_matrix_path);
    detector_ptr->detection_vposition(input);
    return 0;
    // cv::imshow("hahha", result);
    // cvWaitKey(0);
}