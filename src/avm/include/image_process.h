#include <iostream>
#include <future>
#include <vector>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <map>
#include <set>
#include "lane.h"

class ImageProcess
{
    public:
        typedef std::shared_ptr<ImageProcess> Ptr;
        ImageProcess(std::string file_name);
        ~ImageProcess();
        bool read_video();
        Lane ::Ptr lane_ptr_;

    private:
        double depart_lane_distance_;
        IplImage* process_image(IplImage* img);  
        std::string file_name_;
        IplImage* hls_img_;
        
        int width_ ;
        int height_ ;
        int half_width_ ;
        int half_height_ ;

        IplImage* h_channel_img_;
        IplImage* l_channel_img_;
        IplImage* s_channel_img_;
        IplImage* yaw_image_;
        IplImage* edge_img_;
        IplImage* input_image_;
        IplImage* x_edge_img_;
        IplImage* y_edge_img_;
        IplImage* edge_direction_img_;

        IplImage* gray_img_;
        IplImage* eye_img_;
        IplImage* s_binary_img_;

        IplImage* hist_img_;
        CvCapture* capture_;
        IplImage*  resized_eye_img_;

        IplImage* x_eye_img_;
        IplImage* s_eye_img_;
        IplImage* sobel_eye_img_;
        // Lane::lane
        
        CvPoint2D32f srcTri[4], dstTri[4];
        CvMat* warp_mat_ = cvCreateMat( 3, 3, CV_32FC1 );
        void find_lane(std::map<int, int> fore_count_hist, std::map<int, int> after_count_hist);
        bool lane_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A);
        bool track();
        void fill_lane(std::vector<cv::Point> left_lane, std::vector<cv::Point> right_lane);
        void compute_direction_edge();
        void compute_hls_binary(IplImage* img);
        void compute_sobelx_and_direction_binary(IplImage* img);
        void compute_direction_binary();
        void fusion_map();

};
