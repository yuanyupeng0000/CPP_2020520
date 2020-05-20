#ifndef __NCS_DETECTOR_H__
#define __NCS_DETECTOR_H__
#include <sys/time.h>
#include "DSPARMProto.h"
#include "m_arith.h"
#include "Python.h"
#include <numpy/arrayobject.h>
#include <opencv2/opencv.hpp>
using namespace cv;
////////////////////////////////////////////////////////////
void py_init();//加载python,用NCS进行检测
void py_free();//释放python内存
int get_ncs_id();//给相机分配NCS id
void free_ncs_id(int NCS_ID);//释放相机的NCS id
extern int NCSArithDetect(Mat BGRImage, ALGCFGS* pCfgs, int* rst);

#endif



