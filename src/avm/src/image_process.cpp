#include "image_process.h"
#include <numeric>

ImageProcess::ImageProcess(std::string file_name):file_name_(file_name)
{
    lane_ptr_ = std::make_shared<Lane>();

    srcTri[0].x = 194;
    srcTri[0].y = 719;
    srcTri[1].x = 1117;
    srcTri[1].y = 719;
    srcTri[2].x = 705;
    srcTri[2].y = 461;
    srcTri[3].x = 575;
    srcTri[3].y = 461;

    dstTri[0].x = 290;
    dstTri[0].y = 719;
    dstTri[1].x = 990;
    dstTri[1].y = 719;
    dstTri[2].x = 990;
    dstTri[2].y = 0;
    dstTri[3].x = 290;
    dstTri[3].y = 0;
    cvGetPerspectiveTransform( srcTri, dstTri, warp_mat_ ); 

    capture_ = cvCreateFileCapture("/home/rzc/my_ws/data/project_video.mp4");
    IplImage* img = cvQueryFrame(capture_);
     // init
    width_ = img->width;
    height_ = img->height;
    half_width_ = img->width/4;
    half_height_ = img->height/4;


    input_image_ = cvCreateImage(cvSize(half_width_, half_height_), IPL_DEPTH_8U, 3);//cvsize(列，行),跟mat不同
    hls_img_ = cvCreateImage(cvSize(half_width_, half_height_), IPL_DEPTH_8U, img->nChannels);//cvsize(列，行),跟mat不同
    // split three channel hls
    h_channel_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    l_channel_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    s_channel_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    // the channel s binary pic 
    s_binary_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);

    gray_img_ = cvCreateImage(cvSize(half_width_, half_height_), IPL_DEPTH_8U, 1);//cvsize(列，行),跟mat不同
   
   
    edge_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    eye_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    hist_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);

    resized_eye_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    
   
    x_edge_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    y_edge_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);


    x_eye_img_= cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    s_eye_img_= cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    sobel_eye_img_= cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);

    edge_direction_img_ = cvCreateImage(cvSize(half_width_, half_height_),IPL_DEPTH_8U,1);
    
}

ImageProcess::~ImageProcess()
{

}

bool ImageProcess::read_video(){
    cvNamedWindow("s_binary_img_", 0);
    CvCapture* capture = cvCreateFileCapture("/home/rzc/my_ws/data/project_video.mp4");

    IplImage * frame;
    yaw_image_ = frame;
    while (1)
    {
        frame = cvQueryFrame(capture_);
        if (!frame) break;
        process_image(frame);
        cvShowImage("s_binary_img_", resized_eye_img_);
        cvWaitKey(0);
    }

    cvReleaseCapture(&capture);
    return true;  
}
// image y原始图像
IplImage* ImageProcess::process_image(IplImage* img){

    cvWarpPerspective(img, img, warp_mat_ );  //对图像做仿射变换
    cvResize(img, input_image_);
    
    //  hsl detection
    cvCvtColor(input_image_, hls_img_, CV_BGR2HLS);
    cvSplit(hls_img_, h_channel_img_, l_channel_img_, s_channel_img_, NULL);
    cvThreshold(s_channel_img_, s_eye_img_, 120, 255,CV_THRESH_BINARY);

    // edge detection
    cvCvtColor(input_image_, gray_img_, CV_BGR2GRAY);//CV_BGR2HLS
    cvSobel(gray_img_, sobel_eye_img_,1,0,3);
    // cvConvertScale(x_edge_img_,x_edge_img_);
    // cvSobel(gray_img_, y_edge_img_,0,1,3);
    // cvConvertScale(y_edge_img_,y_edge_img_);
    // cvAddWeighted(x_edge_img_, 0.5, y_edge_img_, 0.5, 0, edge_img_);

    // fusion
    cvThreshold(sobel_eye_img_, sobel_eye_img_, 50, 255,CV_THRESH_BINARY);
    cvThreshold(s_eye_img_, s_eye_img_, 120, 255,CV_THRESH_BINARY);
    cvAddWeighted(sobel_eye_img_, 0.5, s_eye_img_, 0.5,0, eye_img_);
    cvThreshold(eye_img_, eye_img_, 50, 255,CV_THRESH_BINARY);
   
   
    // cvShowImage("hahhs", eye_img_);
    // cvWaitKey(0);
    

    
    int value[]={1,1,1,
		         1,1,1,
				 1,1,1
	};

	IplConvKernel* kernel=cvCreateStructuringElementEx(3,3,1,1,CV_SHAPE_CUSTOM,value);
    cvMorphologyEx(eye_img_,eye_img_,NULL,kernel,CV_MOP_OPEN,1); //开运算
    cvResize(eye_img_, resized_eye_img_, 1);
    
    // histogram statistics
    std::map<int, int> fore_count_hist, after_count_hist;
    for (int i = 0; i < resized_eye_img_->height; i++)
    {
        for (int j = 0; j < resized_eye_img_->width; j++)
        {
            if(((uchar*)(resized_eye_img_->imageData + i*resized_eye_img_->widthStep))[j] >100 ){
                if(j<resized_eye_img_->width/2){
                    fore_count_hist[j] +=1;
                }else
                {
                    after_count_hist[j] += 1;
                }
                
            }
        }
    }
    // for debug
    // std::map<int, int>::iterator iter;
    // iter = fore_count_hist.find(0);
    // for (iter = fore_count_hist.begin(); iter != fore_count_hist.end(); iter++)
    //     std::cout<<(*iter).first<<","<<(*iter).second <<std::endl;
    if (lane_ptr_->detected_){
        track();
    }else
    {
        find_lane(fore_count_hist, after_count_hist);
    }
    
    lane_ptr_->pre_left_points_fitted = lane_ptr_->cur_left_points_fitted;
    lane_ptr_->pre_right_points_fitted = lane_ptr_->cur_right_points_fitted;
    lane_ptr_->pre_left_lane_parameter = lane_ptr_->cur_left_lane_parameter;
    lane_ptr_->pre_right_lane_parameter = lane_ptr_->cur_right_lane_parameter;
    
    fore_count_hist.clear();
    after_count_hist.clear();
    return hls_img_;
}
void ImageProcess::compute_hls_binary(IplImage* img){

    cvCvtColor(img, hls_img_, CV_BGR2HLS);//CV_BGR2HLS
    cvSplit(hls_img_, h_channel_img_, l_channel_img_, s_channel_img_, NULL);
    cvThreshold(s_channel_img_, s_binary_img_, 120, 255,CV_THRESH_BINARY);

}
void ImageProcess::fusion_map(){
    // eye_img_;s_eye_img_;
    cvThreshold(sobel_eye_img_, sobel_eye_img_, 50, 255,CV_THRESH_BINARY);
    cvThreshold(s_eye_img_, s_eye_img_, 120, 255,CV_THRESH_BINARY);
    cvAddWeighted(sobel_eye_img_, 0.5, s_eye_img_, 0.5,0, eye_img_);
    cvThreshold(eye_img_, eye_img_, 50, 255,CV_THRESH_BINARY);

}


void ImageProcess::compute_direction_edge(){
    for (int j =0 ; j <edge_img_->height ; j++)
    {
        for (int i = 0 ; i < edge_img_->width; i++)
        {
             ((uchar*)(edge_direction_img_->imageData + j*edge_direction_img_->widthStep))[i] = 
                                atan2(((uchar*)(y_edge_img_->imageData + j*y_edge_img_->widthStep))[i], 
                                ((uchar*)(x_edge_img_->imageData + j*x_edge_img_->widthStep))[i]);
        
        }
    }
    
}

void ImageProcess::find_lane(std::map<int, int> fore_count_hist, std::map<int, int> after_count_hist){
    std::cout<<"start detection"<<std::endl;

    int mid_lane_x = int(resized_eye_img_->width/2);
    auto left_lane_index = (std::max_element(fore_count_hist.begin(), fore_count_hist.end(),
                        [](const std::pair<int, int> &p1, const std::pair<int, int> &p2){
                            return p1.second<p2.second;
                        }))->first;
    auto right_lane_index = (std::max_element(after_count_hist.begin(), after_count_hist.end(),
                        [](const std::pair<int, int> &p1, const std::pair<int, int> &p2){
                            return p1.second<p2.second;
                        }))->first;

    int nwindow = 9;
    int window_height = int(resized_eye_img_->height/nwindow);
    int offset = 25;
    int min_pix = 15;
    
    // std::cout<<"window_height"<<window_height<<std::endl;
    // std::cout<<"mid_x"<< mid_lane_x<<std::endl;
    // std::cout<<"left_lane_index:"<< left_lane_index<<std::endl;
    // std::cout<<"right_max_index:"<< right_max_index<<std::endl;
    // std::cout<<"right_window_left_bound:"<< right_window_left_bound<<std::endl;
    // std::cout<<"right_window_right_bound:"<< right_window_right_bound<<std::endl;

    std::vector<int> left_inner_point_x, left_inner_point_y;
    std::vector<int> right_inner_point_x, right_inner_point_y;

    for (int i = 0; i < 9; i++)
    {
        //  std::cout<<i<<"times"<<"------------------------------------------------:"<<std::endl;
            // std::cout << "height:" << window_height<< std::endl;

        int left_window_left_bound = std::max(0, left_lane_index - offset);
        int left_window_right_bound = std::min(left_lane_index + offset, mid_lane_x);
        int right_window_left_bound = std::max(mid_lane_x, right_lane_index - offset);
        int right_window_right_bound = std::min(right_lane_index + offset, edge_img_->width);
      
        int window_y_low = resized_eye_img_->height - (i+1) * window_height;
        int window_y_high = resized_eye_img_->height - i * window_height;
        

        
        std::vector<int> left_good_inner_point_x, left_good_inner_point_y;
        std::vector<int>  right_good_inner_point_x, right_good_inner_point_y;

        for (int j = window_y_low; j <window_y_high ; j++)
        {
            for (int k = left_window_left_bound; k < left_window_right_bound; k++)
            {
                if(((uchar*)(resized_eye_img_->imageData + j*resized_eye_img_->widthStep))[k] >00 ){
                    left_good_inner_point_x.push_back(k);
                    left_good_inner_point_y.push_back(j);
                    left_inner_point_x.push_back(k);
                    left_inner_point_y.push_back(j);
                    // std::cout << "point_x:" << k<<"-------------"<<"point_y:"<<j << std::endl;
                }
            }
            for (int k = right_window_left_bound; k < right_window_right_bound; k++)
            {
                if(((uchar*)(resized_eye_img_->imageData + j*resized_eye_img_->widthStep))[k] >100 ){
                    right_good_inner_point_x.push_back(k);
                    right_good_inner_point_y.push_back(j);
                    right_inner_point_x.push_back(k);
                    right_inner_point_y.push_back(j);
                }
            }
        }
        
        // compute mean value
        if (left_good_inner_point_x.size() > min_pix)
        {
            // std::cout<<"inner point size:"<< left_inner_point_x.size()<<std::endl;
            int sum = std::accumulate(left_good_inner_point_x.begin(), left_good_inner_point_x.end(), 0);
            int mean = sum/left_good_inner_point_x.size();
            left_lane_index = mean;
            // std::cout << "mean" << mean << std::endl;
        }

        if (right_good_inner_point_x.size() > min_pix)
        {
            // std::cout<<"inner point size:"<< right_inner_point_x.size()<<std::endl;
            int sum = std::accumulate(right_good_inner_point_x.begin(), right_good_inner_point_x.end(), 0);
            int mean = sum/(right_good_inner_point_x.size());
            right_lane_index = mean;
            // std::cout << "mean" << mean << std::endl;

        }
    }
    
    std::vector<cv::Point> left_lane, right_lane;
    for (int i = 0; i < left_inner_point_x.size(); i++)
    {
        left_lane.push_back(cv::Point(left_inner_point_y[i], left_inner_point_x[i]));
       
    }
    //   std::cout<<"left coodinate --------"<<left_lane<<std::endl;
    for (int i = 0; i < right_inner_point_x.size(); i++)
    {
        right_lane.push_back(cv::Point(right_inner_point_y[i], right_inner_point_x[i]));
    }
        //  std::cout<<"right coodinate --------"<<right_lane<<std::endl;

    fill_lane(left_lane, right_lane);
   

}

bool ImageProcess::lane_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
	//Number of key points
	int N = key_point.size();
 
	//构造矩阵X
	cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int j = 0; j < n + 1; j++)
		{
			for (int k = 0; k < N; k++)
			{
				X.at<double>(i, j) = X.at<double>(i, j) +
					std::pow(key_point[k].x, i + j);
			}
		}
	}
 
	//构造矩阵Y
	cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	for (int i = 0; i < n + 1; i++)
	{
		for (int k = 0; k < N; k++)
		{
			Y.at<double>(i, 0) = Y.at<double>(i, 0) +
				std::pow(key_point[k].x, i) * key_point[k].y;
		}
	}
 
	A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
	//求解矩阵A
	cv::solve(X, Y, A, cv::DECOMP_LU);
	return true;
}
//pick bettween equation twoside point fit
bool ImageProcess::track(){
    std::cout<<"start track"<<std::endl;
    std::vector<cv::Point> left_lane, right_lane;

   for (int x = 0; x < resized_eye_img_->height; x++)
	{   
       
		int left_y = int(lane_ptr_->cur_left_lane_parameter.at<double>(0, 0) + lane_ptr_->cur_left_lane_parameter.at<double>(1, 0) * x +
                                                    lane_ptr_->cur_left_lane_parameter.at<double>(2, 0)*std::pow(x, 2));

        int right_y = int(lane_ptr_->cur_right_lane_parameter.at<double>(0, 0) + lane_ptr_->cur_right_lane_parameter.at<double>(1, 0) * x +
                                                    lane_ptr_->cur_right_lane_parameter.at<double>(2, 0)*std::pow(x, 2));
        // std::cout<<"fit coodinate --------"<<left_y<<","<<right_y<<std::endl;
        
        for (int i = -15; i < 15; i++)
        {
            if(((uchar*)(resized_eye_img_->imageData + x*resized_eye_img_->widthStep))[left_y+i] >100 ){
                left_lane.push_back(cv::Point(x, left_y+i));
                //  std::cout<<"left coodinate --------"<<left_y + i<<","<<x<<std::endl;
            }
            if(((uchar*)(resized_eye_img_->imageData + x*resized_eye_img_->widthStep))[right_y+i] >100 ){
                right_lane.push_back(cv::Point(x, right_y+i));
                //  std::cout<<"right coodinate --------"<<right_y + i<<","<<x<<std::endl;


            }
           
        }
	}
    fill_lane(left_lane, right_lane);
    
}

void ImageProcess::fill_lane(std::vector<cv::Point> left_lane, std::vector<cv::Point> right_lane){

    // std::cout<<"left coodinate --------"<<left_lane<<std::endl;
    // std::cout<<"right coodinate --------"<<right_lane<<std::endl;
    
    cv::Mat left_lane_parameter, right_lane_parameter;
    bool result1 = ImageProcess::lane_curve_fit(left_lane, 2, left_lane_parameter);
    bool result2 = ImageProcess::lane_curve_fit(right_lane, 2, right_lane_parameter);
    // std::cout<<"left lane parameter --------"<<left_lane_parameter<<std::endl;
    
    // for compute  coordinate
    std::vector<cv::Point> left_points_fitted, right_points_fitted;

    std::vector<int> v_gap; 
    int sum =0;
	for (int x = 0; x < resized_eye_img_->height; x++)
	{   
       
		double left_y = left_lane_parameter.at<double>(0, 0) + left_lane_parameter.at<double>(1, 0) * x +
                                                    left_lane_parameter.at<double>(2, 0)*std::pow(x, 2);
        // std::cout<<"left_y--------"<<left_y<<std::endl;
	
    	left_points_fitted.push_back(cv::Point(left_y, x));

        double right_y = right_lane_parameter.at<double>(0, 0) + right_lane_parameter.at<double>(1, 0) * x +
                                                    right_lane_parameter.at<double>(2, 0)*std::pow(x, 2);
        // std::cout<<"right_y--------"<<right_y<<std::endl;

		right_points_fitted.push_back(cv::Point(right_y, x));
        // compute gap that left and right
        sum += (right_y - left_y);
        v_gap.push_back(right_y - left_y);
        
	}
        std::cout<<"sum--------"<<sum<<std::endl;

        double mean = sum/resized_eye_img_->height;
        std::cout<<"mean--------"<<mean<<std::endl;

        double accum = 0;
        std::for_each(std::begin(v_gap), std::end(v_gap),[&](const int d){
            accum += (d-mean)*(d-mean);
        });
        double stdev = sqrt(accum/(v_gap.size()-1));
        std::cout<<"std--------"<<stdev<<std::endl;
        if (stdev <15 ){
            lane_ptr_->cur_left_points_fitted = left_points_fitted;
            lane_ptr_->cur_right_points_fitted = right_points_fitted; 
            lane_ptr_->cur_left_lane_parameter = left_lane_parameter;
            lane_ptr_->cur_right_lane_parameter = right_lane_parameter;
            lane_ptr_->detected_ = true;
        }else
        {
            lane_ptr_->detected_ = false;
        }
   
    cv::Mat image = cv::Mat::zeros(resized_eye_img_->height, resized_eye_img_->width, CV_8UC3);
	image.setTo(cv::Scalar(100, 0, 0));
    cv::polylines(image, right_points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
    cv::polylines(image, left_points_fitted, false, cv::Scalar(0, 255, 255), 1, 8, 0);
    // left_points_fitted
    double actual_center =  (left_points_fitted.back().x + right_points_fitted.back().x)/2;
    double image_center = resized_eye_img_->width/2;
    // theory - actual positive car is located right  negtive is left 
    depart_lane_distance_= image_center - actual_center;
    std::cout<<"image_center--------"<<image_center<<std::endl;
    std::cout<<"actual_center--------"<<actual_center<<std::endl;
    std::cout<<"depart_lane_distance_--------"<<depart_lane_distance_<<std::endl;

    left_lane.clear();
    right_lane.clear();
    left_lane_parameter.release();
    right_lane_parameter.release();
    cv::imshow("image", image);
	cv::waitKey(0);
    
}
