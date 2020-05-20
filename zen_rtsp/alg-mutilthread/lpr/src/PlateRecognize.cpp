#include "../include/PlateRecognize.h"
#include "../include/Pipeline.h"
using namespace std;
typedef pr::PipelinePR* pPipelinePR;
#define PLATE_RECOGNIZE_NET_NUM 8//最大车牌检测网络数
pPipelinePR prc[PLATE_RECOGNIZE_NET_NUM];
int plate_init_flag[PLATE_RECOGNIZE_NET_NUM] = { 0 };
/*pr::PipelinePR prc("model/cascade.xml",
				   "model/HorizonalFinemapping.prototxt","model/HorizonalFinemapping.caffemodel",
				   "model/Segmentation.prototxt","model/Segmentation.caffemodel",
				   "model/CharacterRecognization.prototxt","model/CharacterRecognization.caffemodel",
				   "model/SegmenationFree-Inception.prototxt","model/SegmenationFree-Inception.caffemodel"
				   );*/
int LoadPlateNet()//加载车牌网络
{
	int flag = -1;
	int i = 0;
	for(i = 0; i < PLATE_RECOGNIZE_NET_NUM; i++)
	{
		if(plate_init_flag[i] == 0)//加载网络
		{
			prc[i] = new pr::PipelinePR("model/cascade.xml",
				"model/HorizonalFinemapping.prototxt","model/HorizonalFinemapping.caffemodel",
				"model/Segmentation.prototxt","model/Segmentation.caffemodel",
				"model/CharacterRecognization.prototxt","model/CharacterRecognization.caffemodel",
				"model/SegmenationFree-Inception.prototxt","model/SegmenationFree-Inception.caffemodel"
				);
			plate_init_flag[i] = 1;
			flag = i;
			return flag;
		}
	}
	return flag;
}
int FreePlateNet(int flag)//释放车牌网络
{
	prc[flag]->~PipelinePR();
	if(plate_init_flag[flag] == 1)
	{
		plate_init_flag[flag] = 0;//将标志设置为0
	}
	return 0;
}
int PlateDetectandRecognize(cv::Mat image, const int segmentation_method, PlateInfo*  plateInfo, int flag)//车牌识别
{
	int i = 0;
	int plateNum = 0;
	std::vector<pr::PlateInfo> res = prc[flag]->RunPiplineAsImage(image, segmentation_method);
	plateNum = res.size();
	for(i = 0; i < plateNum; i++)
	{
		cv::Rect rct = res[i].getPlateRect();//车牌区域
		cv::String str = res[i].getPlateName();//车牌号
		plateInfo[i].plateRect.x = rct.x;
		plateInfo[i].plateRect.y = rct.y;
		plateInfo[i].plateRect.width = rct.width;
		plateInfo[i].plateRect.height = rct.height;
		strcpy(plateInfo[i].plateName, str.c_str());
		plateInfo[i].plateType = (PlateColor)(res[i].getPlateType());//车牌颜色
		plateInfo[i].confidence = res[i].confidence;//置信度

	}
	return plateNum;
}
PlateInfo PlateRecognizeOnly(cv::Mat plateROI, const int segmentation_method, int flag)//车牌识别
{
	int i = 0;
	PlateInfo plateInfo;
	pr::PlateInfo res = prc[flag]->RunPiplineAsPlate(plateROI, segmentation_method);

	cv::Rect rct = res.getPlateRect();//车牌区域
	cv::String str = res.getPlateName();//车牌号
	plateInfo.plateRect.x = rct.x;
	plateInfo.plateRect.y = rct.y;
	plateInfo.plateRect.width = rct.width;
	plateInfo.plateRect.height = rct.height;
	strcpy(plateInfo.plateName, str.c_str());
	plateInfo.plateType = (PlateColor)(res.getPlateType());//车牌颜色
	plateInfo.confidence = res.confidence;//置信度
	printf("plate %s %f\n",plateInfo.plateName,plateInfo.confidence);
	return plateInfo;
}
