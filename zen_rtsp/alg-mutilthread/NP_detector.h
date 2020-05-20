#ifndef __NP_DETECTOR_H__
#define __NP_DETECTOR_H__
#include "m_arith.h"
Uint16 NPDetector(Mat img, NonMotorInfo* NPDetectInfo, ALGCFGS *pCfgs);//进行非机动车多乘员检测
Uint16 AnalysisNPDetect(ALGCFGS* pCfgs, NonMotorInfo* NPDetectInfo);//分析检测结果
bool detect_riderNum(CRect nonMotorBox, CRect* riderBox, int boxNum);//检测骑行人数
bool detect_helmet(CRect nonMotorBox, CRect helmetBox);//是否带帽
#endif