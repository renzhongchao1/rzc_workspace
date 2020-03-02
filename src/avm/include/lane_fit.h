
#include <iostream>
#include <vector>
#include <future>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <future>
// struct CURVE_FITTING_COST
// {
//     CURVE_FITTING_COST ( double x, double y ) : _x ( x ), _y ( y ) {}
//     // 残差的计算
//     template <typename T>
//     bool operator() (
//             const T* const abc,     // 模型参数，有3维
//             T* residual ) const     // 残差
//     {
//         residual[0] = T ( _y ) - (abc[0]*T ( _x ) *T ( _x ) + abc[1]*T ( _x ) + abc[2]) ; // y-(ax^2+bx+c)
//         return true;
//     }
//     const double _x, _y;    // x,y数据
// };

double lane_pameter[3] = {0, 0, 0};    
double* fit_lane(std::vector<std::pair<double, double>> lane_points);