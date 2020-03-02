#include <iostream>
#include <future>
#include <opencv2/opencv.hpp>     
// #include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
// #include <opencv2/imgproc/imgproc.hpp>
class Undistor
{
    public:
        typedef std::shared_ptr<Undistor> Ptr;
        Undistor(cv::Mat input);
        ~Undistor(){}
    
    public:
        std::string inner_parameter_path_ = "inner_parameter.yaml";
        cv::Mat input_image_, output_;
        cv::Mat mtx_, dist_;

    public:
        bool load_parameter();
        cv::Mat distirt_image();
};

