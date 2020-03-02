#include <ceres/ceres.h>
#include <chrono>
#include "lane_fit.h"
using namespace std;


double* fit_lane(std::vector<std::pair<double, double>> lane_points){

    int N = lane_points.size();
    ceres::Problem problem;
    for ( int i=0; i<N; i++ )
    {
        problem.AddResidualBlock (     // 向问题中添加误差项
                // 使用自动求导，模板参数：误差类型，输出维度，输入维度，维数要与前面struct中一致
            new ceres::AutoDiffCostFunction<CURVE_FITTING_COST, 1, 3> (
            new CURVE_FITTING_COST ( lane_points[i].first, lane_points[i].second)
        ),
                nullptr,            // 核函数，这里不使用，为空
                lane_pameter                 // 待估计参数
        );
    }

    // 配置求解器
    ceres::Solver::Options options;     // 这里有很多配置项可以填
    options.linear_solver_type = ceres::DENSE_QR;  // 增量方程如何求解
    options.minimizer_progress_to_stdout = true;   // 输出到cout

    ceres::Solver::Summary summary;                // 优化信息
    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    ceres::Solve ( options, &problem, &summary );  // 开始优化
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
    cout<<"solve time cost = "<<time_used.count()<<" seconds. "<<endl;
    // 输出结果
    cout<<summary.BriefReport() <<endl;
    for ( auto a:lane_pameter ) cout<<a<<" ";
    cout<<endl;

    return lane_pameter;

}
