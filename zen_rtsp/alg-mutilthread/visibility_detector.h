#ifndef __VISIBILITY_DETECTOR_H__
#define __VISIBILITY_DETECTOR_H__
#include "m_arith.h"
//统计每行sobel边缘个数，确定能见度的位置
Uint16 cal_edge_num(Uint8* img, int width, int height);
//计算每行对比度，当对比度大于0.05时，返回行位置
Uint16 cal_visibility(Uint8* img, int calibration_point[][2], int width, int height);
//分区域计算对比度，当对比度大于0.05时，返回行位置
Uint16 cal_region_visibility(Uint8* img, int calibration_point[][2], int width, int height);
Uint16 DayVisibilityDetection(Uint8* img, int calibration_point[][2], int width, int height);//白天能见度计算
float NightVisibilityDetection(Uint8* img, int calibration_point[][2], int width, int height, float l1, float l2);//晚上能见度计算
#endif