#include <iostream>
#include <future>
#include <opencv2/opencv.hpp>    
#include <opencv2/highgui/highgui.hpp> 
#include <vector>
#include <cmath>                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                           
class Detection
{
    public:
        typedef std::shared_ptr<Detection> Ptr;
        Detection(std::string perspect_matrix_path);
        ~Detection(){}

    public:
        cv::Mat detection_vposition(cv::Mat input);
        std::string perspect_matrix_path_ ;
        cv::Mat pespective_matrix_,input_;
        std::map<int, int> pixel_hist_;

    private:
        void image_process(const cv::Mat input);
        void proj_by_x(const cv::Mat edge);

        std::vector<cv::Point2d> compute_cross_point(std::vector<std::pair<cv::Point, cv::Point>> 
                                    vertical_vp, std::pair<float, float> horizontal_vp);
        
        std::pair<float, float> simplify_horizontal_line(const std::vector<std::pair<float, float>> kbs);
        
        std::vector<std::pair<cv::Point, cv::Point>>  simplify_vertical_line(
                                    const std::vector<std::pair<cv::Point, cv::Point>> );

        
};
