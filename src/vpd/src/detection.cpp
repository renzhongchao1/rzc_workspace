#include"detection.h"
Detection::Detection(std::string perspect_matrix_path)
                    :perspect_matrix_path_(perspect_matrix_path){
    
    cv::FileStorage fs(perspect_matrix_path_, cv::FileStorage::READ);
	fs["pespective_matrix"]>>pespective_matrix_;
    std::cout<<"load pespective_matrix:"<< pespective_matrix_<<std::endl;

}

cv::Mat Detection::detection_vposition(cv::Mat input){
    input_ = input;
    image_process(input);
    return input;
}

void Detection::image_process(const cv::Mat input){
  
    cv::Mat tmp_mat = input;
    cv::warpPerspective( tmp_mat, tmp_mat,
                                   pespective_matrix_, tmp_mat.size());
    cv::GaussianBlur(tmp_mat, tmp_mat, cv::Size(5,5), 2, 2);
    cv::Mat grad_x, abs_grad_x, dstImage;
    // compute x gradient
    // cv::Sobel(tmp_mat, grad_x, CV_16S, 1, 0, 3, 1, 1,cv::BORDER_DEFAULT);
    // cv::convertScaleAbs(grad_x, abs_grad_x);
    // cv::cvtColor(abs_grad_x, abs_grad_x, CV_BGR2GRAY);//CV_BGR2HLS
    // cv::adaptiveThreshold(abs_grad_x, edge, 255, CV_ADAPTIVE_THRESH_MEAN_C, CV_THRESH_BINARY, 31, 10);//自适应阈值分割
    cv::Canny(tmp_mat, abs_grad_x, 15, 45, 3);
    // for debug
    cv::cvtColor(abs_grad_x, dstImage, CV_GRAY2BGR);//转化边缘

    std::vector<cv::Vec4f> lines;
    // output line
    cv::HoughLinesP( abs_grad_x, lines, 1, CV_PI/180, 80, 30, 10 );
    std::vector<std::pair<cv::Point, cv::Point> > vertical_vp;
    std::vector<std::pair<float, float> > horizontal_vp;

    for (size_t i = 0; i < lines.size(); i++)
    {
        cv::Vec4i I=lines[i];
        cv::Point pt1, pt2;

        pt1.x = I[0];
        pt1.y = I[1];
        pt2.x = I[2];
        pt2.y = I[3];
            
        double angle_rad = cv::fastAtan2((pt2.y - pt1.y),(pt2.x - pt1.x));

        if (angle_rad >= 180){
            angle_rad -= 180;
        }
        //    horizontal 
        if (((angle_rad<10) || (angle_rad>170))&& abs(pt1.x - pt2.x)>100){
            cv::line(dstImage, pt1, pt2, cv::Scalar(55, 100, 195), 1, CV_AA);
            // float k = tan(angle_rad/180);
            float k = float(pt1.y - pt2.y)/float(pt1.x -pt2.x);
            // float k = (pt1.y - pt2.y);
            // float k = (pt1.x -pt2.x);
            // k = 1/2;
            float b = pt1.y - k*pt1.x;
           
            horizontal_vp.push_back(std::make_pair(k, b));
            std::cout<<"pt1点"<<pt1<<std::endl;
            std::cout<<"pt2点"<<pt2<<std::endl;
            std::cout<<"角度"<<angle_rad<<std::endl;

            std::cout<<"水平斜率"<<k<<std::endl;
            std::cout<<"水平  b "<< b <<std::endl;
        }
        
        // vertical 
        if ((angle_rad>80) && (angle_rad<100)){
            cv::line(dstImage, pt1, pt2, cv::Scalar(55, 100, 195), 1, CV_AA);
            float k = float(pt1.y - pt2.y)/float(pt1.x -pt2.x);
            float b = pt1.y - k*pt1.x;
            vertical_vp.push_back(std::make_pair(pt1, pt2));
        }
    }

    // std::vector<std::pair<cv::Point, float> > vertical_vp;
    // std::vector<std::pair<cv::Point, float> > horizontal_vp;
    std::pair<float, float> simply_horizontal_vp = simplify_horizontal_line(horizontal_vp);
    // simplify_vertical_line(vertical_vp);
    auto cross_point = compute_cross_point(vertical_vp, simply_horizontal_vp);
    std::cout<<"pt1点"<<cross_point<<std::endl;

    cv::imshow("x orientation soble", abs_grad_x);
    cv::imshow("x orientation s", dstImage);
    cv::waitKey(0);
    
}
void Detection::proj_by_x(const cv::Mat edge){
    cv::Mat tmp_edge = edge;
    // histogram statistics
    for(int i = 0; i < tmp_edge.cols; i++)  //lie循环，可根据需要换成rowNumber
	{
		for(int j = 0; j < tmp_edge.rows; j++)  //hang循环，同理
		{
            uchar* data = tmp_edge.ptr<uchar>(j);  
			int intensity = data[i];
			// std::cout << intensity << std::endl ;
            if(intensity < 100 ){
                    pixel_hist_[i] +=1;
            }
		}
	}
    std::map<int, int>::iterator iter;
    iter = pixel_hist_.find(0);
    for (iter = pixel_hist_.begin(); iter != pixel_hist_.end(); iter++)
        
        
    std::cout<<(*iter).first<<","<<(*iter).second <<std::endl;
    // auto left_lane_index = (std::max_element(pixel_hist_.begin(), pixel_hist_.end(),
    //                     [](const std::pair<int, int> &p1, const std::pair<int, int> &p2){
    //                         return p1.second<p2.second;
    //                     }))->first;
}


std::vector<cv::Point2d> Detection::compute_cross_point(std::vector<std::pair<cv::Point, cv::Point>> 
                                    vertical_vp, std::pair<float, float> horizontal_vp){
    float horizontal_k = horizontal_vp.first;
    float horizontal_b = horizontal_vp.second;
    cv::Point pt;
    std::vector<cv::Point2d> cross_point;
    for (size_t i = 0; i < vertical_vp.size(); i++)
    {
        cv::Point pt1 = vertical_vp[i].first;
        cv::Point pt2 = vertical_vp[i].second;
        float vertical_k = (pt2.y - pt1.y)/(pt2.x - pt1.x);
        float vertical_b = pt1.y - vertical_k * pt1.x;
        pt.x = (vertical_b - horizontal_b)/(horizontal_k - vertical_k);
        pt.y = vertical_k * pt.x + vertical_b;
        cross_point.push_back(pt);
    }
    return cross_point;
}

std::pair<float, float> Detection::simplify_horizontal_line(
                                    const std::vector<std::pair<float, float>> horizontal_vp){
    float b = 10000000;
    std::pair<float, float> horizontal;
    for (size_t i = 0; i < horizontal_vp.size(); i++)
    {
        float tmp_b = horizontal_vp[i].second;
        if (tmp_b < b && tmp_b > 200){
            b = tmp_b;
            horizontal = std::make_pair(horizontal_vp[i].first, b);
        }
    }
    std::cout<<horizontal.first<<"-----------"<<horizontal.second<<std::endl;
    return horizontal;
}

std::vector<std::pair<cv::Point, cv::Point>>  simplify_vertical_line(
                                    const std::vector<std::pair<cv::Point, cv::Point>> vertical_vp){
    // std::vector<std::pair<cv::Point, cv::Point>> simply_vertical_vp = vertical_vp;
    // std::vector<int> vindex;
    // float a = 0;
    // for (size_t i = 0; i < vertical_vp.size(); i++)
    // {

    //     if (vertical_vp[i].second.x ){

    //     }
    // }    
}
        