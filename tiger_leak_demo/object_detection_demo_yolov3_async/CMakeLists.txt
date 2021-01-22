# SPDX-License-Identifier: Apache-2.0
#
# --------------------------------------------------------------------------------------------------------------------
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3")

### openvino 2018 R5
#include_directories(/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/include)
#include_directories(/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/samples/common)
#include_directories(/opt/intel/computer_vision_sdk/deployment_tools/inference_engine/src/extension)
#link_directories(/opt/intel/computer_vision_sdk_2018.5.445/deployment_tools/inference_engine/lib/ubuntu_16.04/intel64/)
#link_directories(/home/yyp/inference_engine_samples_build/intel64/Release/lib/)
#link_directories(/opt/intel/computer_vision_sdk_2018.5.445/deployment_tools/inference_engine/external/omp/lib/)

### openvino 2019 R2
#include_directories(/opt/intel/openvino/deployment_tools/inference_engine/include)
#include_directories(/opt/intel/openvino/deployment_tools/inference_engine/samples/common)
#include_directories(/opt/intel/openvino/deployment_tools/inference_engine/src/extension)
#link_directories(/opt/intel/openvino/deployment_tools/inference_engine/lib/intel64/)

### openvino 2020 R4/R3
include_directories(/opt/intel/openvino_2021/deployment_tools/inference_engine/include)
include_directories(/opt/intel/openvino_2021/deployment_tools/inference_engine/samples/cpp/common/)
include_directories(/opt/intel/openvino_2021/deployment_tools/inference_engine/src/extension)
link_directories(/opt/intel/openvino_2021/deployment_tools/inference_engine/lib/intel64/)

# ---------------------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.0)

set(TARGET_NAME "object_detection_yolov3_async")

# Find OpenCV components if exist
#find_package(OpenCV COMPONENTS highgui QUIET)
#if(NOT(OpenCV_FOUND))
#    message(WARNING "OPENCV is disabled or not found, " ${TARGET_NAME} " skipped")
#    return()
#endif()

file (GLOB MAIN_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp
        )

file (GLOB MAIN_HEADERS
        ${CMAKE_CURRENT_SOURCE_DIR}/*.h
        )

# Create named folders for the sources within the .vcproj
# Empty name lists them directly under the .vcproj
source_group("src" FILES ${MAIN_SRC})
source_group("include" FILES ${MAIN_HEADERS})

link_directories(${LIB_FOLDER})

# Create library file from sources.
add_executable(${TARGET_NAME} ${MAIN_SRC} ${MAIN_HEADERS})
###add_library(${TARGET_NAME} SHARED ${MAIN_SRC} ${MAIN_HEADERS})

##add_dependencies(${TARGET_NAME} gflags)  ###why to complie gflags?

set_target_properties(${TARGET_NAME} PROPERTIES "CMAKE_CXX_FLAGS" "${CMAKE_CXX_FLAGS} -fPIE"
        COMPILE_PDB_NAME ${TARGET_NAME})


#target_link_libraries(${TARGET_NAME} IE::ie_cpu_extension ${InferenceEngine_LIBRARIES} gflags ${OpenCV_LIBRARIES})
#target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} clDNNPlugin GNAPlugin inference_engine myriadPlugin MKLDNNPlugin HeteroPlugin HDDLPlugin cpu_extension_avx2 cpu_extension_sse4 gflags ${OpenCV_LIBRARIES} dl)
#target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} clDNNPlugin GNAPlugin inference_engine myriadPlugin MKLDNNPlugin HeteroPlugin HDDLPlugin cpu_extension_avx2 cpu_extension_sse4 gflags dl)
target_link_libraries(${TARGET_NAME} ${InferenceEngine_LIBRARIES} clDNNPlugin GNAPlugin inference_engine_legacy inference_engine myriadPlugin MKLDNNPlugin HeteroPlugin HDDLPlugin gflags dl)
target_link_libraries(${TARGET_NAME} -L/usr/local/lib/ -lopencv_video -lopencv_videoio -lopencv_imgproc -lopencv_core -lopencv_highgui -lopencv_imgcodecs -lopencv_ml)
target_link_libraries(${TARGET_NAME} -L/usr/lib/x86_64-linux-gnu/ -lgflags)

if(UNIX)
    target_link_libraries( ${TARGET_NAME} ${LIB_DL} pthread)
endif()
