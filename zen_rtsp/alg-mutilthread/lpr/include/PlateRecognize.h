//
// Created by Jack Yu on 21/10/2017.
//

#ifndef SWIFTPR_PLATERECOGNIZE_H
#define SWIFTPR_PLATERECOGNIZE_H
#include <opencv2/opencv.hpp>

typedef enum 
{ 
	BLUE, YELLOW, WHITE, GREEN, BLACK,UNKNOWN
}PlateColor ;//车牌颜色
typedef enum  
{
	CHINESE,LETTER,LETTER_NUMS,INVALID
}CharType;//车牌字符类型
#ifndef __ALG_RECT__
#define __ALG_RECT__
typedef struct
{
	int x;
	int y;
	int width;
	int height;
}CRect;
#endif
typedef struct 
{
	char plateName[50];//车牌名
	CRect plateRect;//车牌区域
	PlateColor plateType;//车牌颜色
	float confidence;//车牌置信度
}PlateInfo;

int LoadPlateNet();//加载车牌网络
int FreePlateNet(int flag);//释放车牌网络
int PlateDetectandRecognize(cv::Mat image, const int segmentation_method, PlateInfo*  plateInfo, int flag);//车牌识别
PlateInfo PlateRecognizeOnly(cv::Mat plateROI, const int segmentation_method, int flag);//输入车牌区域图像，进行识别


#endif //SWIFTPR_CNNRECOGNIZER_H
