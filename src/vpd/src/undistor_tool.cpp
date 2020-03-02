#include "undistor_tool.h"

Undistor::Undistor(cv::Mat input){
    input_image_ = input;
    output_ = input_image_;
   
   bool load_result = load_parameter();
   if (load_result){
       std::cout<<"畸变矫正中"<<std::endl;
   }else
   {
       std::cerr<<"加载参数失败"<<std::endl;
   }

}

bool Undistor::load_parameter(){

    cv::FileStorage fs("/home/rzc/my_ws/src/vpd/src/parameter.yml", cv::FileStorage::READ);
   
	fs["mtx"]>>mtx_;
    fs["dist"]>>dist_;
    return (mtx_.data)&&(dist_.data);

}

cv::Mat Undistor::distirt_image(){
    cv::Mat view, rview, map1, map2;
    auto new_matrix = cv::getOptimalNewCameraMatrix(mtx_, dist_, input_image_.size(), 1, input_image_.size());
    std::cout<<new_matrix<<std::endl;
    cv::initUndistortRectifyMap(mtx_, dist_, cv::Mat(),
                                 cv::getOptimalNewCameraMatrix(mtx_, dist_, input_image_.size(), 1, input_image_.size(), 0),
                                 input_image_.size(), CV_16SC2, map1, map2);
    remap(input_image_, output_, map1, map2, cv::INTER_LINEAR);
    cv::Mat output_image(output_,cv::Rect(500, 400, 500, 380));
    return output_image;
}
