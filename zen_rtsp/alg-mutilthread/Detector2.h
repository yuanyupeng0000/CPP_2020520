

#ifndef _DETECTOR_H_
#define _DETECTOR_H_

#include "cascadedetect.h"
extern int InitHaarParam(int flag);

extern void alg_opencv_processPC(unsigned char * srcImg,unsigned char* maskImg,int w, int h,Rect* obj_rect,int  *obj_num,int flag,bool& initial_param_ok); 

#endif /*_DETECTOR_H_*/
