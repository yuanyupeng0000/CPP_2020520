#ifndef __YOLO_DETECTOR_H__
#define __YOLO_DETECTOR_H__
#include "darknet.h"
#include <stdbool.h>
#include<sys/time.h>
#include <opencv2/opencv.hpp>
using namespace cv;
#ifndef __cplusplus
#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#include "opencv2/imgproc/imgproc_c.h"
#include "opencv2/core/version.hpp"
#endif
#endif
typedef struct 
{
	network* net;//检测网络
	char** names;//检测类别名
	int classes_num;//检测类别数
	float thresh;//检测阈值
	int handle_index;//cublas handle index
}NET_PARAMS;

#ifdef __cplusplus
extern "C"{
#endif
    bool LoadNetParams(NET_PARAMS* net_params, int gpu_index);//加载网络参数
    int free_yolo_network_params(NET_PARAMS* net_params, int handle_index);//释放检测网络参数
#ifdef OPENCV
	int YoloArithDetect(IplImage* img, NET_PARAMS* net_params, int* result);//yolo检测
#endif
#ifdef __cplusplus
}
#endif

#endif