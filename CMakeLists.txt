cmake_minimum_required(VERSION 3.10)
project(caffe2_cpp_test)

set(CMAKE_CXX_STANDARD 11)

find_library(CAFFE2_LIB caffe2)
find_library(CAFFE2_GPU_LIB caffe2_gpu)
find_library(GLOG_LIB glog)
find_library(GFLAGS_LIB gflags)
find_library(GTEST_LIB gtest)
find_library(NCCL_LIB nccl)
find_library(PROTOBUF_LIB protobuf)
find_package(OpenCV REQUIRED)

add_executable(caffe2_cpp_test main.cpp)
target_link_libraries(caffe2_cpp_test ${OpenCV_LIBS} ${CAFFE2_LIB} ${CAFFE2_GPU_LIB} ${GLOG_LIB} ${GFLAGS_LIB} ${PROTOBUF_LIB} )