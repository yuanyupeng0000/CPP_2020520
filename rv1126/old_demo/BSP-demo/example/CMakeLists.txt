#
# Copyright 2019 Fuzhou Rockchip Electronics Co., Ltd. All rights reserved.
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#

# vi: set noexpandtab syntax=cmake:
cmake_minimum_required(VERSION 3.4.1)
project(rkmedia_examples)
set(TARGET_NAME "rk_sample_test")

set(CMAKE_SYSTEM_NAME Linux)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")


add_definitions(-g -O0 -ggdb -gdwarf -funwind-tables -rdynamic)

add_definitions(-DDEBUG)

add_definitions(-DRKAIQ) ############

aux_source_directory(common COMMON_SRC)

#--------------------------
# rkmedia_vi_venc_test
#--------------------------
link_directories(${CMAKE_SOURCE_DIR}/../lib) ############
link_directories(${CMAKE_SOURCE_DIR}/../librknn_api/lib) ############

add_executable(${TARGET_NAME} rk_sample.cpp ${COMMON_SRC} librknn.hpp)
include_directories(${CMAKE_SOURCE_DIR})
include_directories(${CMAKE_SOURCE_DIR}/../include)
include_directories(${CMAKE_SOURCE_DIR}/../example/common)
include_directories(${CMAKE_SOURCE_DIR}/../include/rkmedia)
include_directories(${CMAKE_SOURCE_DIR}/../include/rkaiq/common)
include_directories(${CMAKE_SOURCE_DIR}/../include/rkaiq/xcore)
include_directories(${CMAKE_SOURCE_DIR}/../include/rkaiq/uAPI)
include_directories(${CMAKE_SOURCE_DIR}/../include/rkaiq/algos)
include_directories(${CMAKE_SOURCE_DIR}/../include/rkaiq/iq_parser)
target_link_libraries(${TARGET_NAME} -lpthread -leasymedia -ldrm -lrockchip_mpp -lavformat -lavcodec -lswresample -lavutil -lasound -lv4l2 -lv4lconvert -lrga -lRKAP_ANR -lRKAP_Common -lRKAP_AEC -lmd_share -lrkaiq -lod_share -lrknn_yolo -lrknn_api)


