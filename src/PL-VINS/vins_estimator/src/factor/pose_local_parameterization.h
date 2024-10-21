#pragma once

#include <eigen3/Eigen/Dense>
#include <ceres/ceres.h>
#include "../utility/utility.h"

class PoseLocalParameterization : public ceres::LocalParameterization
{
    // 必须实现的方法，实现参数更新逻辑
    virtual bool Plus(const double *x, const double *delta, double *x_plus_delta) const;
    // 计算雅可比
    virtual bool ComputeJacobian(const double *x, double *jacobian) const;
    // 返回全局参数大小
    virtual int GlobalSize() const { return 7; };
    // 返回局部参数大小
    virtual int LocalSize() const { return 6; };
};
