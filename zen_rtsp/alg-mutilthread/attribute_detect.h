#ifndef __ATTRIBUTE_DETECT_H__
#define __ATTRIBUTE_DETECT_H__
#ifdef DETECT_PERSON_ATTRIBUTE
#include "m_arith.h"
#include <opencv2/opencv.hpp>
using namespace cv;
//行人属性识别
void attri_init();//初始化python,初始化全局变量
#ifdef USE_PYTHON
void py_attri_init();//调用python文件进行检测
#else
void LoadAttriNet(int gpu_idx);//采用caffe c++进行检测
///////////////////////////////////////////////////////////////////////////采用caffe c++进行检测
//deploy_file 网络配置文件
//trained_file 检测网络文件
//gpu_idx gpu ID
//net_idx net ID
extern void LoadAttriNet(const char* deploy_file,
						 const char* trained_file,
						 int gpu_idx, int net_idx);//加载检测网络

//img 检测图像
//net_idx net ID
//result 属性值
extern void AttriDetect(unsigned char* imgdata, int w, int h, int net_idx, int* result);//检测行人属性
#endif

bool HumanAttributeInit(ALGCFGS *pCfgs);//行人属性初始化
HumanAttribute HumanAttributeRecognition(IplImage* imgROI, ALGCFGS* pCfgs);//行人属性识别
void HumanAttributeDetect(ALGCFGS *pCfgs, IplImage* img);//行人属性检测分析

//单车属性识别
bool BicycleAttributeInit(ALGCFGS *pCfgs);//单车属性初始化
BicycleAttribute BicycleAttributeRecognition(IplImage* imgROI, ALGCFGS* pCfgs);//单车属性识别
void BicycleAttributeDetect(ALGCFGS *pCfgs, IplImage* img);//单车属性检测分析
#endif

#endif