#include <ros/ros.h>
#include <iostream>
#include <vector>
#include "image_process.h"
// #include "lane.h"
#include "lane_fit.h"
std::string video_path = "/home/rzc/my_ws/data/project_video.mp4";

void avm_start(const std::string path){
    std::cout<<path<<std::endl;

}

int main(int argc, char**argv){
    ros::init(argc, argv, "avm_node");
    ros::NodeHandle h_node;
    ImageProcess ::Ptr img_process_ptr = std::make_shared<ImageProcess>(video_path);
    bool result = img_process_ptr->read_video();
    if (result){
        std::cout<<"handle success"<<std::endl;
    }
    else
    {
        std::cout<<"fail handle"<<std::endl;
    }
    // std::vector<std::pair<double, double>> vdata;
    // int N=100;                          // 数据点
    // double w_sigma=1.0;                 // 噪声Sigma值
    // cv::RNG rng; 
    // double a=10.0, b=1.0, c=1.0;      
    // for ( int i=0; i<N; i++ )
    // {
    //     double x = i/100.0;
    //     vdata.push_back(std::make_pair(x, ( a*x*x + b*x + c ) + rng.gaussian ( w_sigma )) );
    // }   

    // Lane::Ptr lane_ptr = std::make_shared<Lane>();
    // avm_start(video_path);
    // double *result;
    // result = fit_lane(vdata);
       

    return 1;
}

