//
// Created by jemin on 3/1/22.
//
#pragma once

#include <Eigen/Core>

#include "raisim/World.hpp"
#include "raisim/RaisimServer.hpp"


using Dtype=float;
using EigenRowMajorMat=Eigen::Matrix<Dtype, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<Dtype, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;

extern int threadCount;
