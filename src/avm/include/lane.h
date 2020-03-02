#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <future>

class Lane
{
   
    public:
        typedef std::shared_ptr<Lane> Ptr;
        Lane();
        ~Lane();

    public:
        bool detected_ = false;
        std::vector<cv::Point> cur_left_points_fitted, cur_right_points_fitted;
        std::vector<cv::Point> pre_left_points_fitted, pre_right_points_fitted;
        cv::Mat cur_left_lane_parameter, cur_right_lane_parameter;
        cv::Mat pre_left_lane_parameter, pre_right_lane_parameter;
};
