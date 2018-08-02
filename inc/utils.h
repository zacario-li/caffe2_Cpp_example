//
// Created by lzj on 18-7-27.
//

#ifndef CAFFE2_CPP_TEST_UTILS_H
#define CAFFE2_CPP_TEST_UTILS_H

#include <iostream>
#include <caffe2/core/init.h>
#include <caffe2/core/operator.h>
#include <caffe2/core/operator_gradient.h>
#include <caffe2/core/context_gpu.h>

#include <caffe2/core/net.h>
#include <caffe2/utils/proto_utils.h>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace caffe2{
    void print(const Blob* blob, const std::string& name);
    void caffe2_pretrained_run();
    TensorCPU prepareMatImgData(cv::Mat& img);
}

#endif //CAFFE2_CPP_TEST_UTILS_H
