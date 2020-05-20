//#include "stdafx.h"
#include "pthread.h"
#include "m_arith.h"
#include "visibility_detector.h"
#ifdef DETECT_GPU//GPU
#include "yolo_detector.h"
#else//ncs
#include "NCS_detector.h"
#endif
#ifdef DETECT_PERSON_ATTRIBUTE//检测行人属性
#include "attribute_detect.h"
#endif
#ifdef DETECT_PLATE//检测车牌
#include "lpr/include/PlateRecognize.h"
#define DETECT_IMAGE_WIDTH 3000
#define DETECT_IMAGE_HEIGHT 3000
#endif
using namespace std;
//////////////////////////////////////////////////////////////////////////////////////

#define	MAX_SPEEDDETECTOR_DOTS	768*576
#define MaxDotsInDetect 768*576
//#define FULL_COLS  					(720)
//#define FULL_ROWS  					(576)

clock_t current_time;
double timer;
//#define  SAVE_VIDEO 
#ifdef SAVE_VIDEO
#define SAVE_VIDEO_WIDTH 640
#define SAVE_VIDEO_HEIGHT 480
cv::Mat img;
#define  SAVE_FRAMES  5000
cv::VideoWriter writer("VideoResult.avi", CV_FOURCC('X', 'V', 'I', 'D'), 15, Size(SAVE_VIDEO_WIDTH, SAVE_VIDEO_HEIGHT)); 
#endif
//char LABELS[][50] = {"background", "bus", "car", "truck", "bicycle", "motorbike", "person"};
//char LABELS[][50] = {"bus","car", "truck", "motorbike", "bicycle","person"};
char LABELS[][50] = {"bus","car", "truck", "motorbike", "bicycle", "person", "plate", "bottle", "cup",\
	"helmet", "rider", "2_person", "3_person", "Reserve0", "Reserve1", "Reserve2", "Reserve3", "Reserve4", "Reserve5",\
    "js", "kw", "ps", "lf"};

int alg_mem_malloc(m_args *p)
{
	//printf("malloc.......\n");
	int ret = -1;
	int size;
	int i = 0;
	//输出内存分配

	//	ALGPARAMS *Params;
	p->p_outbuf = NULL;
	p->p_outbuf = (OUTBUF*) malloc(sizeof(OUTBUF));
	if (p->p_outbuf == NULL) {
		printf("alg malloc err\n");
	}
	memset(p->p_outbuf, 0, sizeof(OUTBUF));

	//参数配置内存分配
	p->pParams = NULL;
	p->pParams = (ALGPARAMS*) malloc(sizeof(ALGPARAMS));
	if (p->pParams == NULL) {
		printf("alg malloc err\n");
	}
	memset(p->pParams, 0, sizeof(ALGPARAMS));
	p->pParams->CurrQueueImage = NULL;

	p->pParams->CurrQueueImage = (Uint8*) malloc(
		DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 11 * sizeof(Uint8));
	if (p->pParams->CurrQueueImage == NULL) {
		printf("alg malloc err\n");
	}
	p->pParams->MaskEventImage = NULL;
	p->pParams->MaskEventImage = (Uint32*)malloc(
		DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * sizeof(Uint32));
	if (p->pParams->CurrQueueImage == NULL || p->pParams->MaskEventImage == NULL) {
		printf("alg malloc err\n");
	}

	memset(p->pParams->CurrQueueImage, 0,
		DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 11 * sizeof(Uint8));
	memset(p->pParams->MaskEventImage, 0,
		DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX  * sizeof(Uint32));

#ifdef DETECT_PLATE//检测车牌
	p->pParams->PreFullImage = NULL;
	p->pParams->PreFullImage = (Uint8*) malloc(DETECT_IMAGE_WIDTH * DETECT_IMAGE_HEIGHT * 2 * sizeof(Uint8));
	memset(p->pParams->PreFullImage, 0, DETECT_IMAGE_WIDTH * DETECT_IMAGE_HEIGHT * 2 * sizeof(Uint8));
#endif

	//配置内存分配
	p->pCfgs = NULL;
	p->pCfgs = (ALGCFGS*) malloc(sizeof(ALGCFGS));
	if (p->pCfgs == NULL) {
		printf("alg malloc err\n");
	}else{
		printf("alg malloc ok \n");
	}

	memset(p->pCfgs, 0, sizeof(ALGCFGS));
	p->pCfgs->names = NULL;
#ifdef DETECT_GPU
	p->pCfgs->net_params = (NET_PARAMS *) malloc(sizeof(NET_PARAMS));
	p->pCfgs->net_params->net = NULL;
	p->pCfgs->net_params->names = NULL;
#else//NCS
	p->pCfgs->NCS_ID = get_ncs_id();
#endif
#ifdef DETECT_PERSON_ATTRIBUTE//加载行人属性识别网络
	attri_init();
#endif
#ifdef DETECT_PLATE//加载车牌识别网络
	p->pCfgs->plate_flag = LoadPlateNet();
	if(p->pCfgs->plate_flag < 0)
		printf("no load plate recognize net\n");
#endif
	//printf("(pCfgs malloc %x,%x\n",p->pCfgs,pthread_self());fflush(NULL);
	//printf("(pCfgs malloc %x\n",p->pCfgs->CameraCfg);
	return 0;
}

int alg_mem_free(m_args *arg_arg)
{
	//printf("free.......\n");
#ifdef DETECT_GPU
	if(arg_arg->pCfgs->net_params)
	{
		free_yolo_network_params(arg_arg->pCfgs->net_params, arg_arg->pCfgs->net_params->handle_index);
		free(arg_arg->pCfgs->net_params);
		arg_arg->pCfgs->net_params = NULL;
	}
#else
	if(arg_arg->pCfgs->NCS_ID >= 0)
	{
		free_ncs_id(arg_arg->pCfgs->NCS_ID);//设置此计算棒不能运行
	}
	//py_free();
#endif
	for(int i = 0; i < arg_arg->pCfgs->classes; i++)
	{
		if(arg_arg->pCfgs->names[i])
		{
			free(arg_arg->pCfgs->names[i]);
			arg_arg->pCfgs->names[i] = NULL;
		}
	}
	if(arg_arg->pCfgs->names)
	{
		free(arg_arg->pCfgs->names);
		arg_arg->pCfgs->names = NULL;
	}
#ifdef DETECT_PLATE//释放车牌识别网络
	if(arg_arg->pCfgs->plate_flag >= 0)
	{
		FreePlateNet(arg_arg->pCfgs->plate_flag);
	}
#endif
		free(arg_arg->pCfgs);
		arg_arg->pCfgs = NULL;

	if ( arg_arg->pParams) {
		if (arg_arg->pParams->CurrQueueImage) {
			free(arg_arg->pParams->CurrQueueImage);
			arg_arg->pParams->CurrQueueImage = NULL;
		}
		if((arg_arg)->pParams->MaskEventImage) {
			free((arg_arg)->pParams->MaskEventImage);
			(arg_arg)->pParams->MaskEventImage = NULL;
		}
#ifdef DETECT_PLATE//检测车牌
		if((arg_arg)->pParams->PreFullImage){
			free((arg_arg)->pParams->PreFullImage);
			(arg_arg)->pParams->PreFullImage = NULL;
		}
#endif

		free(arg_arg->pParams);
		arg_arg->pParams = NULL;
	}
	if (arg_arg->p_outbuf) {
		free(arg_arg->p_outbuf);
		arg_arg->p_outbuf = NULL;
	}
	return 0;
}

#define max(a,b) (((a)>(b)) ? (a):(b))
#define min(a,b) (((a)>(b)) ? (b):(a))

//设置pParams->MaskDetectImage为行人检测区域
bool MaskDetectImage(ALGCFGS *pCfgs, ALGPARAMS *pParams, int imgW, int imgH)
{
	Int32	i, j, k;
	CPoint	ptCorner[4];
	Uint8* p;
	CPoint pt;
	//标记检测区域,如果图像大小大于640*480，resize到640*480
	memset(pParams->MaskDetectImage, 0, pCfgs->m_iWidth * pCfgs->m_iHeight);
	//传入行人检测区域
	pCfgs->uDetectRegionNum = pCfgs->DownDetectCfg.PersonDetectArea.num;
	for(i = 0; i < pCfgs->uDetectRegionNum; i++)	
	{
		ptCorner[0].x = pCfgs->DownDetectCfg.PersonDetectArea.area[i].realcoordinate[0].x;
		ptCorner[0].y = pCfgs->DownDetectCfg.PersonDetectArea.area[i].realcoordinate[0].y;
		ptCorner[1].x = pCfgs->DownDetectCfg.PersonDetectArea.area[i].realcoordinate[1].x;
		ptCorner[1].y = pCfgs->DownDetectCfg.PersonDetectArea.area[i].realcoordinate[1].y;
		ptCorner[2].x = pCfgs->DownDetectCfg.PersonDetectArea.area[i].realcoordinate[2].x;
		ptCorner[2].y = pCfgs->DownDetectCfg.PersonDetectArea.area[i].realcoordinate[2].y;
		ptCorner[3].x = pCfgs->DownDetectCfg.PersonDetectArea.area[i].realcoordinate[3].x;
		ptCorner[3].y = pCfgs->DownDetectCfg.PersonDetectArea.area[i].realcoordinate[3].y;
		printf("person region = %d,[%d,%d,%d,%d,%d,%d,%d,%d]\n",i,ptCorner[0].x,ptCorner[0].y,ptCorner[1].x,ptCorner[1].y,ptCorner[2].x,ptCorner[2].y,ptCorner[3].x,ptCorner[3].y);

		//按照顺时针方向矫正4点顺序
		CorrectRegionPoint(ptCorner);

		//将坐标限制在[0 pCfgs->m_iWidth - 1]和[0 pCfgs->m_iHeight-1]
		for(j = 0; j < 4; j++)
		{
			ptCorner[j].x = ptCorner[j].x * pCfgs->m_iWidth / imgW;
			ptCorner[j].y = ptCorner[j].y * pCfgs->m_iHeight / imgH;
		}

		//对pParams->MaskDetectImage进行初始化代表不同的检测区域
		for(j = 0; j < pCfgs->m_iHeight; j++)
		{
			p = pParams->MaskDetectImage + j * pCfgs->m_iWidth;
			for(k = 0; k < pCfgs->m_iWidth; k++)
			{
				pt.x = k;
				pt.y = j;
				if(isPointInRect(pt, ptCorner[3], ptCorner[0], ptCorner[1], ptCorner[2]))
				{
					p[k] =  i + 1;
					//p[k] = 255;
				}
			}
		}
	}
	/*IplImage* mask = cvCreateImage(cvSize(pCfgs->m_iWidth, pCfgs->m_iHeight), IPL_DEPTH_8U, 1);
	memcpy(mask->imageData, pParams->MaskDetectImage, pCfgs->m_iWidth * pCfgs->m_iHeight);
	cvSaveImage("mask.jpg", mask, 0);
	cvReleaseImage(&mask);*/

	return	TRUE;

}

//设置pParams->MaskLaneImage为车道区域
bool MaskLaneImage(ALGCFGS *pCfgs, ALGPARAMS *pParams, int imgW, int imgH)
{
	Int32	i, j, k;
	CPoint	ptCorner[4];
	Uint8* p;
	CPoint pt;
	//标记检测区域,如果图像大小大于640*480，resize到640*480
	memset(pParams->MaskLaneImage, 0, pCfgs->m_iWidth * pCfgs->m_iHeight);
	//车道区域
	for(i = 0; i < pCfgs->LaneAmount; i++)	
	{
		memcpy( (void*)ptCorner, (void*)pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion, 4 * sizeof(CPoint) );
		printf("lane region = %d,[%d,%d,%d,%d,%d,%d,%d,%d]\n", i, ptCorner[0].x,ptCorner[0].y, ptCorner[1].x, ptCorner[1].y, ptCorner[2].x, ptCorner[2].y,ptCorner[3].x,ptCorner[3].y);
		//按照顺时针方向矫正4点顺序
		CorrectRegionPoint(ptCorner);
		//车道坐标不在检测区域内，返回
		/*for(j = 0; j < 4; j++)
		{
			if(ptCorner[j].x < 0 || ptCorner[j].x > imgW || ptCorner[j].y < 0 || ptCorner[j].y > imgH)
			{
				printf("detect lane region Point err\n");
				return FALSE;
			}		
		}*/
		//将坐标限制在[0 pCfgs->m_iWidth - 1]和[0 pCfgs->m_iHeight-1]
		for(j = 0; j < 4; j++)
		{
			ptCorner[j].x = ptCorner[j].x * pCfgs->m_iWidth / imgW;
			ptCorner[j].y = ptCorner[j].y * pCfgs->m_iHeight / imgH;
		}
		//对pParams->MaskLaneImage进行初始化代表不同的车道区域
		for(j = 0; j < pCfgs->m_iHeight; j++)
		{
			p = pParams->MaskLaneImage + j * pCfgs->m_iWidth;
			for(k = 0; k < pCfgs->m_iWidth; k++)
			{
				pt.x = k;
				pt.y = j;
				if(isPointInRect(pt, ptCorner[3], ptCorner[0], ptCorner[1], ptCorner[2]))
				{
					p[k] = i + 1;
				}
			}
		}
	}
	/*CPoint points[10];
	float polyX[7] = {20, 100, 300, 500, 600, 300, 100};
	float polyY[7] = {10, 30, 300, 60,  200, 400, 460};
	for(j = 0; j < pCfgs->m_iHeight; j++)
	{
		p = pParams->MaskLaneImage + j * pCfgs->m_iWidth;
		for(k = 0; k < pCfgs->m_iWidth; k++)
		{
			float x = k;
			float y = j;
			if(pointInPolygon(7, polyX, polyY, x, y))
			{
				p[k] = 255;
			}
		}
	}*/

	/*IplImage* mask = cvCreateImage(cvSize(pCfgs->m_iWidth, pCfgs->m_iHeight), IPL_DEPTH_8U, 1);
	memcpy(mask->imageData, pParams->MaskLaneImage, pCfgs->m_iWidth * pCfgs->m_iHeight);
	cvSaveImage("masklane.jpg", mask, 0);
	cvReleaseImage(&mask);*/

	return	TRUE;

}
//对图像进行标定
void get_calibration_data(ALGCFGS *pCfgs, int imgW, int imgH)
{
	int i = 0, j = 0, k = 0;
	int min_value = 1000;
	int idx = 0;
	CPoint pt1[2];
	int max_calibration_height = min(MAX_IMAGE_HEIGHT, imgH);//标定高度
	int max_calibration_width = min(MAX_IMAGE_WIDTH, imgW);//标定宽度
	if(imgW > MAX_IMAGE_WIDTH)//图像宽度太大，进行缩放
	{
		pCfgs->scale_x = (float)imgW / (float)MAX_IMAGE_WIDTH;
	}
	else
	{
		pCfgs->scale_x = 1.0;
	}
	if(imgH > MAX_IMAGE_HEIGHT)//图像高度太大，进行缩放
	{
		pCfgs->scale_y = (float)imgH / (float)MAX_IMAGE_HEIGHT;
	}
	else
	{
		pCfgs->scale_y = 1.0;
	}
	//图像标定
	//if((pCfgs->calibration_point[2][1] > pCfgs->calibration_point[0][1]) && (pCfgs->calibration_point[3][1] > pCfgs->calibration_point[1][1]))
	camera_calibration(pCfgs->base_line, pCfgs->base_length, pCfgs->calibration_point, pCfgs->near_point_length, pCfgs->LaneAmount, pCfgs, imgW, imgH);
	//camera_calibration(pCfgs->base_line, pCfgs->base_length, pCfgs->calibration_point, pCfgs->near_point_length, pCfgs->LaneAmount, pCfgs, 720, 576);
	//camera_calibration_transform(pCfgs->base_line, pCfgs->base_length, pCfgs->calibration_point, pCfgs->near_point_length, pCfgs, 720, 576);
	//camera_calibration_transform(pCfgs->base_line, pCfgs->base_length, pCfgs->calibration_point, pCfgs->near_point_length, pCfgs, imgW, imgH);
	//得到远近线圈的实际长度
	for(i = 0; i < pCfgs->LaneAmount; i++)
	{
		pt1[0].y = (pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[0].y + pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[1].y) / 2;
		pt1[1].y = (pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[2].y + pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[3].y) / 2;
		pt1[0].y = max(0, pt1[0].y);
		pt1[0].y = min(int(pt1[0].y / pCfgs->scale_y), max_calibration_height - 1);
		pt1[1].y = max(0, pt1[1].y);
		pt1[1].y = min(int(pt1[1].y / pCfgs->scale_y), max_calibration_height - 1);
		pCfgs->uActualTailLength[i] = (abs(pCfgs->actual_distance[i][pt1[0].y] - pCfgs->actual_distance[i][pt1[1].y]) + 0.5);
		pt1[0].y = (pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[0].y + pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[1].y) / 2;
		pt1[1].y = (pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[2].y + pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[3].y) / 2;
		pt1[0].y = max(0, pt1[0].y);
		pt1[0].y = min(int(pt1[0].y / pCfgs->scale_y), max_calibration_height - 1);
		pt1[1].y = max(0, pt1[1].y);
		pt1[1].y = min(int(pt1[1].y / pCfgs->scale_y), max_calibration_height - 1);
		pCfgs->uActualDetectLength[i] = (abs(pCfgs->actual_distance[i][pt1[0].y] - pCfgs->actual_distance[i][pt1[1].y]) + 0.5);
		printf("actual length = [%d,%d]\n",pCfgs->uActualTailLength[i], pCfgs->uActualDetectLength[i]);
	}
	printf("coil actual length = %d,%d\n",pCfgs->uActualTailLength[0], pCfgs->uActualDetectLength[1]);
	if(pCfgs->LaneAmount)
	{
		//得到刻度线点
		//y方向刻度线，10个值显示一个刻度点
		if(pCfgs->near_point_length < 0)
			j = ((int)(pCfgs->near_point_length) / 10) * 10;
		else
			j = ((int)(pCfgs->near_point_length) / 10 + 1) * 10;
		for(i = 0; i < 20; i++)
		{
			pCfgs->degreepointY[i][0] = 0;
			pCfgs->degreepointY[i][1] = 0;
		}
		k = 0;
		for(i = max_calibration_height - 2; i >= 0; i--)
		{
			if((pCfgs->actual_distance[pCfgs->LaneAmount - 1][i] - j) * (pCfgs->actual_distance[pCfgs->LaneAmount - 1][i + 1] - j) <= 0)//两点之间为j
			{
				pCfgs->degreepointY[k][0] = int(i * pCfgs->scale_y);
				pCfgs->degreepointY[k][1] = j;
				//printf("degree_y=%d,len=%d\n", pCfgs->degreepointY[k][0], pCfgs->degreepointY[k][1]);
				j = j + 10;
				k++;

			}
			if(k == 20)
			{
				break;
			} 
		}
		printf("y max = %f, min = %f\n", pCfgs->actual_distance[pCfgs->LaneAmount - 1][0], pCfgs->actual_distance[pCfgs->LaneAmount - 1][max_calibration_height - 1]);
		//x方向刻度线，5个值显示一个刻度点
		if(pCfgs->image_actual[max_calibration_height - 1][max_calibration_width - 1][0] < 0)
			j = ((int)(pCfgs->image_actual[max_calibration_height - 1][max_calibration_width - 1][0]) / 5) * 5;
		else
			j = ((int)(pCfgs->image_actual[max_calibration_height - 1][max_calibration_width - 1][0]) / 5 + 1) * 5;
		printf("degreepointX = ");
		k = 0;
		for(i = 0; i < 10; i++)
		{
			pCfgs->degreepointX[i][0] = 0;
			pCfgs->degreepointX[i][1] = 0;
		}
		for(i = max_calibration_width - 2; i >= 0; i--)
		{
			if((pCfgs->image_actual[max_calibration_height - 1][i][0] - j) * (pCfgs->image_actual[max_calibration_height - 1][i + 1][0] - j) <= 0)//两点之间为j
			{
				pCfgs->degreepointX[k][0] = int(i * pCfgs->scale_x);
				pCfgs->degreepointX[k][1] = j;
				printf("[%d, %d] ", pCfgs->degreepointX[k][0], pCfgs->degreepointX[k][1]);
				j = j + 5;
				k++;
			}
			if(k == 10)
			{
				break;
			} 
		}
		printf("\n");
		printf("x max = %f, min = %f\n", pCfgs->image_actual[max_calibration_height - 1][0][0], pCfgs->image_actual[max_calibration_height - 1][max_calibration_width - 1][0]);
		//pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uDegreeLength=pCfgs->actual_degree_length;
	}

}
//初始化参数
bool ArithInit(Uint16 ChNum, CFGINFOHEADER *pCfgHeader, SPEEDCFGSEG *pDetectCfgSeg, ALGCFGS *pCfgs, ALGPARAMS *pParams, int gpu_index)
{
	int i = 0;
	bool bInit = FALSE;
	bInit = CfgStructParse(ChNum, pCfgHeader, pDetectCfgSeg, pCfgs, pParams);//加载参数，参数初始化
#ifdef DETECT_GPU
	//加载权重、类名、检测阈值
	if(pCfgs->net_params->net == NULL)
	{
		LoadNetParams(pCfgs->net_params, gpu_index);
	}
#ifdef DETECT_PERSON_ATTRIBUTE
#ifndef USE_PYTHON
	//加载行人属性网络
	LoadAttriNet(gpu_index);
	printf("load attri net\n");
#endif
#endif
#endif
#ifndef DETECT_GPU
	if(pCfgs->names == NULL)//加载检测类别名
	{
		pCfgs->names = get_labels("FP16/labels.txt", pCfgs->classes);

	}
#endif
	if(pCfgs->names == NULL)//没有加载成功
	{
		pCfgs->classes = sizeof(LABELS) / sizeof(LABELS[0]);//检测类别数
		pCfgs->names = (char**)malloc(pCfgs->classes * sizeof(char*));//分配内存
		for(i = 0; i < pCfgs->classes; i++)
		{
			pCfgs->names[i] = (char*)malloc(strlen(LABELS[i]) * sizeof(char));
			strcpy(pCfgs->names[i], LABELS[i]);
		}
	}
	printf("classes num = %d\n", pCfgs->classes);
	//类别名
	for( i = 0; i < pCfgs->classes; i++)
	{
		strcpy(pCfgs->detClasses[i].names, pCfgs->names[i]);
		printf("%s, %d\n", pCfgs->names[i], strlen(pCfgs->names[i]));
	}
   //交通事件初始化
   CfgEventRegion(pDetectCfgSeg->uSegData->EventDetectArea, pCfgs, pParams);
   //行人属性初始化
#ifdef DETECT_PERSON_ATTRIBUTE
   HumanAttributeInit(pCfgs);
   BicycleAttributeInit(pCfgs);
#endif
	return bInit;
}

//配置参数
bool CfgStructParse(Uint16 ChNum, CFGINFOHEADER *pCfgHeader, SPEEDCFGSEG *pDetectCfgSeg, ALGCFGS *pCfgs, ALGPARAMS *pParams)//,CPoint m_ptend[],CPoint LineUp[]
{
	Int32	i, j, k, idx;
	float min_value = 1000;
	ZENITH_SPEEDDETECTOR 		*pDownSpeedDetect = NULL;
	CPoint	ptFlowCorner[4];//流量区域坐标点
	CPoint	ptMiddleCoil[4];//占有线圈坐标点 
	CPoint	ptFrontCoil[4];//占位线圈坐标点
	CPoint  ptLaneRegion[4];//车道区域坐标点
	CPoint  pt1[MAX_REGION_NUM][2];

	pDownSpeedDetect = (ZENITH_SPEEDDETECTOR*)pDetectCfgSeg->uSegData;

	//加载参数
	pCfgs->LaneAmount = pDownSpeedDetect->uLaneTotalNum;
	pCfgs->bAuto = pDownSpeedDetect->uEnvironment; //TRUE;

	//分配内存
	pParams->PreQueueImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX;//前一帧图像
	pParams->PrePreQueueImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 2;//前两帧图像
	pParams->PrePrePreQueueImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 3;//保存中间结果图像
	pParams->MaskLaneImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 4;//车道掩模图像
    pParams->MaskDetectImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 5;//行人检测区域图像
#ifdef DETECT_PLATE//检测车牌
	pParams->PrePreFullImage = (Uint8 *)pParams->PreFullImage + DETECT_IMAGE_WIDTH * DETECT_IMAGE_HEIGHT;
#endif
	////////////////////////////////////////////////////////////////
	//得到每个车道参数，并对图像区域进行矫正
	for(i=0; i<pCfgs->LaneAmount; i++)
	{
		memcpy( (void*)ptFlowCorner, (void*)pDownSpeedDetect->SpeedEachLane[i].RearCoil, 4 * sizeof(CPoint));
		memcpy( (void*)ptMiddleCoil, (void*)pDownSpeedDetect->SpeedEachLane[i].MiddleCoil, 4 * sizeof(CPoint));
		memcpy( (void*)ptFrontCoil, (void*)pDownSpeedDetect->SpeedEachLane[i].FrontCoil, 4 * sizeof(CPoint));
		memcpy( (void*)ptLaneRegion, (void*)pDownSpeedDetect->SpeedEachLane[i].LaneRegion, 4 * sizeof(CPoint));

		//判断坐标是否越界
		for(j = 0; j < 4; j++)
		{
			if(ptLaneRegion[j].x < 0 || ptLaneRegion[j].x >= MAX_IMAGE_WIDTH || ptLaneRegion[j].y < 0 || ptLaneRegion[j].y >= MAX_IMAGE_HEIGHT)
			{
				printf("Lane Point err \n");
				//return FALSE;
			}
			if(ptFlowCorner[j].x < 0 || ptFlowCorner[j].x >= MAX_IMAGE_WIDTH || ptFlowCorner[j].y < 0 || ptFlowCorner[j].y >= MAX_IMAGE_HEIGHT)
			{
				printf("flow Point err \n");
				//return FALSE;
			}
			if(ptMiddleCoil[j].x < 0 || ptMiddleCoil[j].x >= MAX_IMAGE_WIDTH || ptMiddleCoil[j].y < 0 || ptMiddleCoil[j].y >= MAX_IMAGE_HEIGHT)
			{
				printf("Far Point err \n");
				//return FALSE;
			}
			if(ptFrontCoil[j].x < 0 || ptFrontCoil[j].x >= MAX_IMAGE_WIDTH || ptFrontCoil[j].y < 0 || ptFrontCoil[j].y >= MAX_IMAGE_HEIGHT)
			{
				printf("Far Point err \n");
				//return FALSE;
			}
		}

		CorrectRegionPoint(ptFlowCorner);//矫正流量区域
		CorrectRegionPoint(ptMiddleCoil);//矫正占有区域
		CorrectRegionPoint(ptFrontCoil);//矫正占位区域
		CorrectRegionPoint(ptLaneRegion);//矫正车道区域
		
			
		memcpy( (void*)pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil, (void*)ptFlowCorner, 4 * sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil, (void*)ptMiddleCoil, 4 * sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil, (void*)ptFrontCoil, 4 * sizeof(CPoint) );
		memcpy( (void*)pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion, (void*)ptLaneRegion, 4 * sizeof(CPoint) );

		printf("lane region = [%d,%d,%d,%d,%d,%d,%d,%d]\n",pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[0].x, pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[0].y, pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[1].x, pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[1].y, pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[2].x, pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[2].y, pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[3].x, pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[3].y);
		printf("front region = [%d,%d,%d,%d,%d,%d,%d,%d]\n",pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[0].x, pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[0].y, pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[1].x, pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[1].y, pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[2].x, pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[2].y, pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[3].x, pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[3].y);
		printf("middle region = [%d,%d,%d,%d,%d,%d,%d,%d]\n",pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[0].x, pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[0].y, pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[1].x, pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[1].y, pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[2].x, pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[2].y, pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[3].x, pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[3].y);
		printf("rear region = [%d,%d,%d,%d,%d,%d,%d,%d]\n",pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[0].x, pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[0].y, pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[1].x, pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[1].y, pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[2].x, pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[2].y, pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[3].x, pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[3].y);

	}
	
	//得到工作模式 白天或晚上
	if(pCfgs->bAuto == 2)
	{
		pCfgs->bNight = TRUE;
	}
	else
	{	
		pCfgs->bNight = FALSE;
	}

	//得到4个标定点和基准线端点
	for(i = 0; i < 4; i++)
	{
		pCfgs->calibration_point[i][0] = pDownSpeedDetect->ptimage[i].x;
		pCfgs->calibration_point[i][1] = pDownSpeedDetect->ptimage[i].y;

	}
	for(i = 4; i < 8; i++)
	{
		pCfgs->base_line[i - 4][0] = pDownSpeedDetect->ptimage[i].x;
		pCfgs->base_line[i - 4][1] = pDownSpeedDetect->ptimage[i].y;
	}
	pCfgs->base_length[0] = pDownSpeedDetect->base_length[0] / 100.0;//垂直基准线长
	pCfgs->base_length[1] = pDownSpeedDetect->base_length[1] / 100.0;//水平基准线长
	pCfgs->near_point_length = pDownSpeedDetect->near_point_length / 100.0;//最近点离相机的距离
	pCfgs->cam2stop = pDownSpeedDetect->cam2stop / 100.0;//相机到停止线的距离
	/*pCfgs->calibration_point[0][0] = 12;
	pCfgs->calibration_point[0][1] = 570;
	pCfgs->calibration_point[1][0] = 49;
	pCfgs->calibration_point[1][1] = 196;
	pCfgs->calibration_point[2][0] = 112;
	pCfgs->calibration_point[2][1] = 195;
	pCfgs->calibration_point[3][0] = 233;
	pCfgs->calibration_point[3][1] = 570;
	pCfgs->base_line[0][0] = 101;
	pCfgs->base_line[0][1] = 351;
	pCfgs->base_line[1][0] = 109;
	pCfgs->base_line[1][1] = 398;
	pCfgs->base_line[2][0] = 156;
	pCfgs->base_line[2][1] = 339;
	pCfgs->base_line[3][0] = 300;
	pCfgs->base_line[3][1] = 323;
	pCfgs->base_length[0] = 5;
	pCfgs->base_length[1] = 5;
	pCfgs->near_point_length = 0;*/
	printf("calibration point = [%d,%d],[%d,%d],[%d,%d],[%d,%d]\n",pCfgs->calibration_point[0][0], pCfgs->calibration_point[0][1], pCfgs->calibration_point[1][0], pCfgs->calibration_point[1][1], pCfgs->calibration_point[2][0], pCfgs->calibration_point[2][1], pCfgs->calibration_point[3][0], pCfgs->calibration_point[3][1]);
	printf("base line = [%d,%d],[%d,%d],[%d,%d],[%d,%d]\n",pCfgs->base_line[0][0],pCfgs->base_line[0][1], pCfgs->base_line[1][0], pCfgs->base_line[1][1], pCfgs->base_line[2][0], pCfgs->base_line[2][1], pCfgs->base_line[3][0], pCfgs->base_line[3][1]);
    printf("base length = %f,%f,near_point_length =%f,cam2stop = %f\n",pCfgs->base_length[0],pCfgs->base_length[1], pCfgs->near_point_length,pCfgs->cam2stop);

	//加载行人检测参数
	pCfgs->DownDetectCfg.PersonDetectArea = pDownSpeedDetect->PersonDetectArea;//行人检测区域
	pCfgs->uDetectRegionNum = pCfgs->DownDetectCfg.PersonDetectArea.num;//行人区域数
	//行人检测线
	for(i = 0; i < pCfgs->uDetectRegionNum; i++)
	{
		pt1[i][0].x = pDownSpeedDetect->PersonDetectArea.area[i].detectline[0].x;
		pt1[i][0].y = pDownSpeedDetect->PersonDetectArea.area[i].detectline[0].y;
		pt1[i][1].x = pDownSpeedDetect->PersonDetectArea.area[i].detectline[1].x;
		pt1[i][1].y = pDownSpeedDetect->PersonDetectArea.area[i].detectline[1].y;
		memset(pCfgs->uPersonDirNum[i], 0, MAX_DIRECTION_NUM * sizeof(Uint16));//行人方向数
	}
	//求行人检测线的斜率和截距
	SetLine(pCfgs, pt1);
	pCfgs->person_id = 1;//行人目标ID
	pCfgs->objPerson_size = 0;//行人目标数
	memset(pCfgs->uRegionPersonNum, 0, MAX_REGION_NUM * sizeof(Uint16));//区域行人数
 
	pCfgs->gThisFrameTime = 0; 
	////////////////////////////////////////////////////detect params初始化参数
	pCfgs->target_id = 1;
	pCfgs->targets_size = 0;
	memset(pCfgs->targets, 0, MAX_TARGET_NUM * sizeof(CTarget));
	memset(pCfgs->detClasses, 0, MAX_CLASSES * sizeof(CDetBox));
	for( i = 0; i < MAX_LANE; i++)
	{
		memset(pCfgs->detBoxes, 0 , MAX_LANE_TARGET_NUM * sizeof(CRect));
	}
	memset(pCfgs->detNum, 0, MAX_LANE * sizeof(Uint16));
	//memset(pCfgs->uStatPersonNum, 0, 4 * sizeof(Uint16));
	memset(pCfgs->detTargets, 0, MAX_TARGET_NUM * sizeof(CTarget));
	pCfgs->detTarget_id = 1;
	pCfgs->detTargets_size = 0;

	for(i = 0; i < MAX_LANE; i++)
	{
		pCfgs->uDetectVehicleSum[i] = 0;
		memset(pCfgs->uStatVehicleSum[i], 0, 4 * sizeof(Uint16));
		memset(pCfgs->uStatQuePos[i], 0, 6 * sizeof(Uint16));
		pCfgs->uDetectVehicleFrameNum[i] = 0;
		pCfgs->uRearIntervalNum[i] = 0;//后线圈两目标之间间隔
		pCfgs->existFrameNum[i][0] = 0;
		pCfgs->existFrameNum[i][1] = 0;
	}
	pCfgs->bMaskDetectImage = FALSE;//未设置检测区域掩模图像
	pCfgs->bMaskLaneImage = FALSE;//未设置车道掩模图像
	pCfgs->bCalibrationImage = FALSE;//未进行标定
	return	TRUE;

}

inline void CalTargetSpeed(CTarget target, ALGCFGS *pCfgs, int flag, int* speed)//计算目标实际速度， flag为0代表person flag为1代表车辆
{
	if(target.trajectory_num < 2)//目标开始出现，速度设为0
		return;
	float uSpeedX = 0, uSpeedY = 0;
	/*int pos1_x = target.trajectory[0].x;//目标开始出现的位置
	int pos1_y = target.trajectory[0].y;
	int pos2_x = target.box.x + target.box.width / 2;//当前帧位置
	int pos2_y = target.box.y + target.box.height / 2;*/
	int trajectory_num = target.trajectory_num;
    int start_num = trajectory_num - 50;
	start_num = (start_num < 0)? 0 : start_num;
	int pos1_x = target.trajectory[start_num].x;
	int pos1_y = target.trajectory[start_num].y;
	int pos2_x = target.box.x + target.box.width / 2;
	int pos2_y = target.box.y + target.box.height / 2;
	pos1_x = max(0, int(pos1_x / pCfgs->scale_x));
	pos1_x = min(pos1_x, min(MAX_IMAGE_WIDTH, pCfgs->img_width) - 1);
	pos2_x = max(0, int(pos2_x / pCfgs->scale_x));
	pos2_x = min(pos2_x, min(MAX_IMAGE_WIDTH, pCfgs->img_width) - 1);
	pos1_y = max(0, int(pos1_y / pCfgs->scale_y));
	pos1_y = min(pos1_y, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
	pos2_y = max(0, int(pos2_y / pCfgs->scale_y));
	pos2_y = min(pos2_y, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
	float len_x = pCfgs->image_actual[pos2_y][pos2_x][0] - pCfgs->image_actual[pos1_y][pos1_x][0];//x方向运动距离
	//len_x = (len_x > 0)? len_x : -1 * len_x;
	//uSpeedX = len_x * 3.6 / (pCfgs->currTime - target.start_time + 1e-6);
	uSpeedX = len_x * 3.6 / (pCfgs->currTime - target.trajectory_time[start_num] + 1e-6);
	float len_y = pCfgs->image_actual[pos2_y][pos2_x][1] - pCfgs->image_actual[pos1_y][pos1_x][1];//y方向运动距离
	//len_y = (len_y > 0)? len_y : -1 * len_y;
	//uSpeedY = len_y * 3.6 / (pCfgs->currTime - target.start_time + 1e-6);
	uSpeedY = len_y * 3.6 / (pCfgs->currTime - target.trajectory_time[start_num] + 1e-6);
	//printf("[%d,%d,%d],len = %f, t = %f,speed = %f,[%f,%f]\n", target.continue_num, pos1, pos2, len, target.end_time - target.start_time, uVehicleSpeed,target.start_time,target.end_time);
	uSpeedX = (uSpeedX < 0)? -uSpeedX : uSpeedX;
	uSpeedY = (uSpeedX < 0)? -uSpeedY : uSpeedY;
	if(flag == 0)//行人
	{
		uSpeedX = (uSpeedX > 10)? (7 + rand() % 3) : uSpeedX;
		uSpeedY = (uSpeedY > 10)? (7 + rand() % 3) : uSpeedY;
	}
	else//车辆
	{
		int LaneID = target.lane_id;
		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleQueueLength)//对速度进行限制
		{
			uSpeedX = (uSpeedY > 15)? 15 : uSpeedX;
			uSpeedY = (uSpeedY > 15)? 15 : uSpeedY;
		}
		else
		{
			uSpeedX = (uSpeedX > 150)? (145 + rand() % 5) : uSpeedX;
			uSpeedY = (uSpeedY > 150)? (145 + rand() % 5) : uSpeedY;
		}

	}
	//
	speed[0] = (len_x < 0)? (int)(-1.0 * uSpeedX) : (int)(uSpeedX);
	speed[1] = (len_y < 0)? (int)(-1.0 * uSpeedY) : (int)(uSpeedY);
}

inline Uint16 CalLaneTargetSpeedY(CTarget target, int laneID, ALGCFGS *pCfgs)//计算车道内目标实际速度
{
	if(target.continue_num < 1)//目标开始出现，速度设为0
		return 0;
	float uVehicleSpeed = 0;
	int pos1 = target.trajectory[0].y;//目标开始出现的位置
	//int pos1 = target.trajectory[target.trajectory_id[idx]].y;
	int pos2 = target.box.y + target.box.height / 2;//当前帧位置
	pos1 = max(0, int(pos1 / pCfgs->scale_y));
	pos1 = min(pos1, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
	pos2 = max(0, int(pos2 / pCfgs->scale_y));
	pos2 = min(pos2, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
	float len = pCfgs->actual_distance[laneID][pos2] - pCfgs->actual_distance[laneID][pos1];//目标运动距离
	len = (len > 0)? len : -1 * len;
	uVehicleSpeed = len * 3.6 / (pCfgs->currTime - target.start_time + 1e-6);
	//printf("[%d,%d,%d],len = %f, t = %f,speed = %f,[%f,%f]\n", target.continue_num, pos1, pos2, len, target.end_time - target.start_time, uVehicleSpeed,target.start_time,target.end_time);
/*#ifdef SAVE_VIDEO
	if(pCfgs->NCS_ID == 0)
	{
		char str[10];
		sprintf(str, "speed:%d", (int)uVehicleSpeed);
		putText(img, str, Point(target.box.x + 50,max(0,target.box.y - 10)), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(255, 255,0 ), 2);
	}
#endif*/
	if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[laneID].SpeedDetectInfo1.uVehicleQueueLength)//对速度进行限制
		uVehicleSpeed = (uVehicleSpeed > 15)? 15 : uVehicleSpeed;
	else
		uVehicleSpeed = (uVehicleSpeed > 150)? 150 : uVehicleSpeed;
	if(uVehicleSpeed <= 0)
		//uVehicleSpeed = rand() % 5 + 1;
		uVehicleSpeed = 0;
	return (Uint16)uVehicleSpeed;
}
inline Uint16 CalTargetLength(CTarget target, int laneID, ALGCFGS *pCfgs)//计算目标实际长度
{
	int pos1 = target.box.y;
	int pos2 = target.box.y + target.box.height ;
	pos1 = max(0, int(pos1 / pCfgs->scale_y));
	pos1 = min(pos1, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
	pos2 = max(0, int(pos2 / pCfgs->scale_y));
	pos2 = min(pos2, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
	float len = pCfgs->actual_distance[laneID][pos2] - pCfgs->actual_distance[laneID][pos1];
	len = (len > 0)? len : -1 * len;
	len = len * 10;
	if(strcmp(target.names, "bicycle") == 0 || strcmp(target.names, "motorbike") == 0)
	{
		len = (len > 20)? (rand() % 5 + 16) : len;
	}
	else if(strcmp(target.names, "car") == 0)
	{
		len = (len < 30)? (rand() % 10 + 30) : len;
		len = (len > 50)? (rand() % 10 + 40) : len;
	}
	else if(strcmp(target.names, "truck") == 0)
	{
		len = (len < 50)? (rand() % 10 + 40) : len;
		len = (len > 90)? (rand() % 10 + 80) : len;
	}
	else
	{
		len = (len < 90)? (rand() % 10 + 80) : len;
		len = (len > 120)? (rand() % 10 + 110) : len;
	}
	return (Uint16)len;
}
inline Uint16 CalTargetWidth(CTarget target, ALGCFGS *pCfgs)//计算目标的宽度
{
	int pos_x1 = 0, pos_x2 = 0, pos_y = 0;
	float w = 0;
	int width = 0;
	int max_calibration_height = min(MAX_IMAGE_HEIGHT, pCfgs->img_height);//标定高度
	int max_calibration_width = min(MAX_IMAGE_WIDTH, pCfgs->img_width);//标定宽度
	pos_x1 = max(0, int((float)target.box.x / pCfgs->scale_x));
	pos_x1 = min(pos_x1, max_calibration_width - 1);
	pos_x2 = max(0, int((float)(target.box.x + target.box.width) / pCfgs->scale_x));
	pos_x2 = min(pos_x2, max_calibration_width - 1);
	pos_y = max(0, int((float)(target.box.y + target.box.height) / pCfgs->scale_y));
	pos_y = min(pos_y, max_calibration_height - 1);
	w = pCfgs->image_actual[pos_y][pos_x1][0] - pCfgs->image_actual[pos_y][pos_x2][0];
	w = (w > 0)? w : -1 * w;
	width = (int)(w + 0.5);
	if(width < 2)
		width = 2;
	else if(width > 3)
		width = 3;
	return width;
}
//车辆进入线圈，设置参数
void obj_in_region(ALGCFGS *pCfgs, int LaneID, int idx)//0为流量线圈，1为后线圈
{
	pCfgs->headtime[LaneID] = pCfgs->currTime - pCfgs->jgtime[LaneID][idx];
	pCfgs->jgtime[LaneID][idx] = pCfgs->currTime;//用于计算头间距
	pCfgs->headtime[LaneID] = (pCfgs->headtime[LaneID] < 0)? 0 : pCfgs->headtime[LaneID];
	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].uVehicleHeadtime = pCfgs->headtime[LaneID];//车头时距
	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].calarflag = 1;//车在线圈内
	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].DetectInSum++;
}
//车辆离开线圈,连续车辆时，前一车辆不删除目标但是出车，将calarflag设为2
void obj_out_region(ALGCFGS *pCfgs, int LaneID, CTarget* target, int idx)//0为流量线圈，1为后线圈
{
	if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].calarflag == 1)//车出线圈时，calarflag设置为2
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].calarflag = 2;
	if(target->cal_speed == FALSE)//未计算速度，计算车型、车速、车长
	{
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].DetectOutSum++;
		//pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].uVehicleType = target.class_id + 1;
		//clock_t end_time = clock(); 
		//target.end_time = (float)end_time / CLOCKS_PER_SEC;
		target->end_time = pCfgs->currTime;
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].uVehicleSpeed  = CalLaneTargetSpeedY(*target, LaneID, pCfgs);
		target->cal_speed = TRUE;
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.CoilAttribute[idx].uVehicleLength  = CalTargetLength(*target, LaneID, pCfgs);

	}
}

void ProcessDetectBox(ALGCFGS* pCfgs, int* result, int nboxes)//对检测框进行处理
{
	int i = 0, j = 0;
	int classes_num[MAX_CLASSES] = { 0 };
	int class_id = 0;
	memset(pCfgs->detClasses, 0, MAX_CLASSES * sizeof(CDetBox));//初始化检测框
	memset(classes_num, 0, MAX_CLASSES * sizeof(int));//初始化检测框数
	for( i = 0; i < pCfgs->classes; i++)
	{
		strcpy(pCfgs->detClasses[i].names, pCfgs->names[i]);
	}
	//把检测结果按照类别归类
	for(i = 0; i < nboxes; i++)
	{
		class_id = result[i * 6];
#ifdef DETECT_GPU
		//根据检测类别名找到LABLES对应类别
		for(j = 0; j < pCfgs->classes; j++)
		{
			if(strcmp(pCfgs->net_params->names[class_id], pCfgs->names[j]) == 0)//找到对应的类别名
			{
				class_id = j;
				break;
			}
		}
		if(j == pCfgs->classes)//舍弃不需要的类别
			continue;	
#endif
		//去掉不符合条件的检测框
		if(result[i * 6 + 4] > 1000 || result[i * 6 + 5] > 1000 || result[i * 6 + 4] <= 0 || result[i * 6 + 5] <= 0)
			continue;
		pCfgs->detClasses[class_id].class_id = class_id;
		pCfgs->detClasses[class_id].prob[classes_num[class_id]] = result[i * 6 + 1];
		pCfgs->detClasses[class_id].box[classes_num[class_id]].x = result[i * 6 + 2];
		pCfgs->detClasses[class_id].box[classes_num[class_id]].y = result[i * 6 + 3];
		pCfgs->detClasses[class_id].box[classes_num[class_id]].width = result[i * 6 + 4];
		pCfgs->detClasses[class_id].box[classes_num[class_id]].height = result[i * 6 + 5];
		pCfgs->detClasses[class_id].lane_id[classes_num[class_id]] = -1;
#ifdef SAVE_VIDEO
		if(pCfgs->NCS_ID == 0)
		{
			cv::rectangle(img, cv::Rect(pCfgs->detClasses[class_id].box[classes_num[class_id]].x,pCfgs->detClasses[class_id].box[classes_num[class_id]].y,pCfgs->detClasses[class_id].box[classes_num[class_id]].width,pCfgs->detClasses[class_id].box[classes_num[class_id]].height), cv::Scalar(255, 255 ,255), 3, 8, 0 );
			putText(img, pCfgs->detClasses[class_id].names, cv::Point(pCfgs->detClasses[class_id].box[classes_num[class_id]].x + 30,max(0,pCfgs->detClasses[class_id].box[classes_num[class_id]].y - 10)), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2);
		}
#endif
		classes_num[class_id]++;
		pCfgs->detClasses[class_id].classes_num = classes_num[class_id];
	}
	//对检测框进行后处理
	post_process_box(pCfgs, 50);////对不同类别进行处理
	post_process_box_same(pCfgs, 50);//对相同类别进行处理
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////分析行人检测框
void ProcessPersonBox(ALGCFGS* pCfgs, ALGPARAMS *pParams, int width, int height)
{
	int i, j, k;
	int left = 0, right = 0, top = 0, bottom = 0;
	int dis1 = 0, dis2 = 0;
	int x = 0, y = 0;
	int x0 = 0, y0 = 0;
	float val1 = 0, val2 = 0;
	int nPersonNum = 0;
	Uint16 uRegionPersonNum[MAX_REGION_NUM] = { 0 };
	int match_object[MAX_TARGET_NUM] = { 0 };
	int match_rect[MAX_TARGET_NUM] = { 0 };
	int match_success = -1;
	bool isIntersect = FALSE;
	int region_idx = 0;
	for(i = 0; i < pCfgs->uDetectRegionNum; i++)
	{
		memset(pCfgs->uPersonDirNum[i], 0, MAX_DIRECTION_NUM * sizeof(Uint16));//初始化区域方向数
	}
	for( i = 0; i < pCfgs->objPerson_size; i++)
	{
		pCfgs->objPerson[i].detected = FALSE;
	}
	//分析行人检测框
	for( i = 0; i < pCfgs->classes; i++)
	{
		if(strcmp(pCfgs->detClasses[i].names, "person") != 0 )
			continue;
		if(pCfgs->detClasses[i].classes_num)
		{
			match_object_rect(pCfgs->objPerson, pCfgs->objPerson_size, pCfgs->detClasses, i, match_object, match_rect, 5);
			for( j = 0; j < pCfgs->detClasses[i].classes_num; j++)
			{
				bool inRegion = FALSE;
				//判断行人是否在行人检测区域内
				for(k = 0; k < pCfgs->uDetectRegionNum; k++)
				{
					int isInRegion = RectInRegion(pParams->MaskDetectImage, pCfgs, width, height, pCfgs->detClasses[i].box[j], k + 1);
					if(isInRegion > 10)//在检测区域内
					{
						inRegion = TRUE;//在区域内
						region_idx = k;
						//pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum] = pCfgs->detClasses[i].box[j];
						/*pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].x = pCfgs->detClasses[i].box[j].x;
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].y = pCfgs->detClasses[i].box[j].y;
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].w = pCfgs->detClasses[i].box[j].width;
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].h = pCfgs->detClasses[i].box[j].height;
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].label = pCfgs->detClasses[i].class_id + 1;
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].confidence = pCfgs->detClasses[i].prob[j];
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].id = 0;
						int pos_x = max(0, pCfgs->detClasses[i].box[j].x + pCfgs->detClasses[i].box[j].width / 2);
						pos_x = min(int(pos_x / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, width) - 1);
						int pos_y =max(0, pCfgs->detClasses[i].box[j].y + pCfgs->detClasses[i].box[j].height / 2);
						pos_y = min(int(pos_y / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, height) - 1);
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].distance[0] = pCfgs->image_actual[pos_y][pos_x][0];//目标与相机的水平距离
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].distance[1] = pCfgs->image_actual[pos_y][pos_x][1];//目标与相机的垂直距离
						pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].laneid = region_idx;//检测区域ID

#ifdef SAVE_VIDEO
	                    if(pCfgs->NCS_ID == 0)
						{
						    cv::rectangle(img, cv::Rect(pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].x, pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].y, pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].width, pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].height), cv::Scalar(0, 0,255), 1, 8, 0 );
						}
#endif
						nPersonNum++;*/
						uRegionPersonNum[k]++;//相应区域行人数加1
						break;
					}
				}
				if(pCfgs->bDetPersonFlow == FALSE)//不检测行人方向流量
					continue;
				//判断行人是否与检测线相交
				left = pCfgs->detClasses[i].box[j].x;
				right = pCfgs->detClasses[i].box[j].x + pCfgs->detClasses[i].box[j].width;
				top = pCfgs->detClasses[i].box[j].y;
				bottom = pCfgs->detClasses[i].box[j].y + pCfgs->detClasses[i].box[j].height;
				/*CTarget* t;
				t = find_nearest_rect(pCfgs->detClasses[i].box[j], pCfgs->detClasses[i].class_id, pCfgs->objPerson, pCfgs->objPerson_size);
				if(t)//与已有目标匹配成功
				{
					t->box = pCfgs->detClasses[i].box[j];
					t->prob = pCfgs->detClasses[i].prob[j];
					t->class_id = pCfgs->detClasses[i].class_id;
					t->detected = TRUE;
				}
				else */
				match_success = -1;
				for( k = 0; k < pCfgs->objPerson_size; k++)
				{
					if(match_object[j] == k && match_rect[k] == j)//与目标匹配
					{
						match_success = 1;
						pCfgs->objPerson[k].box = pCfgs->detClasses[i].box[j];
						pCfgs->objPerson[k].prob = pCfgs->detClasses[i].prob[j];
						pCfgs->objPerson[k].class_id = pCfgs->detClasses[i].class_id;
						pCfgs->objPerson[k].detected = TRUE;
						break;
					}
				}
				if(match_success < 0)//未匹配
				{
					if(inRegion && pCfgs->objPerson_size < MAX_TARGET_NUM)//在检测区域内，加入到新目标
					{
						CTarget nt; 
						Initialize_target(&nt);
						nt.box = pCfgs->detClasses[i].box[j];
						nt.class_id = pCfgs->detClasses[i].class_id;
						nt.prob = pCfgs->detClasses[i].prob[j];
						nt.detected = TRUE;
						nt.target_id = pCfgs->person_id++;
						nt.region_idx = region_idx;//行人与哪个检测线相交
						nt.start_time = pCfgs->currTime;
						if(pCfgs->person_id > 5000)
							pCfgs->person_id = 1;
						strcpy(nt.names, pCfgs->detClasses[i].names);
						//判断检测框是否与检测线相交
						nt.pass_detect_line = isLineIntersectRect(pCfgs->detLineParm[region_idx].pt[0], pCfgs->detLineParm[region_idx].pt[1], nt.box);
						pCfgs->objPerson[pCfgs->objPerson_size] = nt;
						pCfgs->objPerson_size++;
						continue;
					}
				}
			}
		}
	}
	//分析行人目标
	for(i = 0; i < pCfgs->objPerson_size; i++)
	{
		region_idx = pCfgs->objPerson[i].region_idx;
		//保存轨迹，轨迹数小于3000，直接保存，大于3000，去除旧的
		int trajectory_num = pCfgs->objPerson[i].trajectory_num;
		if(trajectory_num < 3000)
		{

			pCfgs->objPerson[i].trajectory[trajectory_num].x = pCfgs->objPerson[i].box.x + pCfgs->objPerson[i].box.width / 2;
			pCfgs->objPerson[i].trajectory[trajectory_num].y = pCfgs->objPerson[i].box.y + pCfgs->objPerson[i].box.height / 2;
			pCfgs->objPerson[i].trajectory_time[trajectory_num] = pCfgs->currTime;
			pCfgs->objPerson[i].trajectory_num++;
		}
		else
		{
			for(j = 0; j < trajectory_num - 1; j++)
			{
				pCfgs->objPerson[i].trajectory[j] = pCfgs->objPerson[i].trajectory[j + 1];
				pCfgs->objPerson[i].trajectory_time[j] = pCfgs->objPerson[i].trajectory_time[j + 1];
			}
			pCfgs->objPerson[i].trajectory[trajectory_num - 1].x = pCfgs->objPerson[i].box.x + pCfgs->objPerson[i].box.width / 2;
			pCfgs->objPerson[i].trajectory[trajectory_num - 1].y = pCfgs->objPerson[i].box.y + pCfgs->objPerson[i].box.height / 2;
			pCfgs->objPerson[i].trajectory_time[trajectory_num - 1] = pCfgs->currTime;
		}

		//检测到，并更新速度
		if(pCfgs->objPerson[i].detected)
		{
			pCfgs->objPerson[i].lost_detected = 0;
            //get_speed(&pCfgs->objPerson[i]);
		}
		else//未检测到
		{
			pCfgs->objPerson[i].lost_detected++;
			//pCfgs->objPerson[i].box.x += pCfgs->objPerson[i].vx;
			//pCfgs->objPerson[i].box.y += pCfgs->objPerson[i].vy;
		}

		//判断是否计数
		x = pCfgs->objPerson[i].box.x + pCfgs->objPerson[i].box.width / 2 ;
		y = pCfgs->objPerson[i].box.y + pCfgs->objPerson[i].box.height / 2 ;
		x0 = pCfgs->objPerson[i].trajectory[0].x;
		y0 = pCfgs->objPerson[i].trajectory[0].y;
		if(pCfgs->objPerson[i].cal_flow == FALSE)
		{
			val1 = pCfgs->detLineParm[region_idx].k * x + pCfgs->detLineParm[region_idx].b - y;
			val2 = pCfgs->detLineParm[region_idx].k * x0 + pCfgs->detLineParm[region_idx].b - y0;
			if(val1 * val2 < 0)//从检测线一侧运动到另一侧
			{
				printf("get person obj\n");
				get_object_num(pCfgs, i, region_idx);
				pCfgs->objPerson[i].cal_flow = TRUE;
				//printf("%d,count 1\n",pCfgs->gThisFrameTime);
			}
			if(pCfgs->objPerson[i].cal_flow == FALSE && pCfgs->objPerson[i].pass_detect_line)//目标最初在检测线上
			{
				if(pCfgs->objPerson[i].continue_num > 5)
				{
					if(isLineIntersectRect(pCfgs->detLineParm[region_idx].pt[0], pCfgs->detLineParm[region_idx].pt[1], pCfgs->objPerson[i].box) == FALSE)//与检测线不相交时，计数
					{
						get_object_num(pCfgs, i, region_idx);
						pCfgs->objPerson[i].cal_flow = TRUE;
					}
				}
			}
		}
		left = pCfgs->objPerson[i].box.x;
		right = pCfgs->objPerson[i].box.x + pCfgs->objPerson[i].box.width;
		top = pCfgs->objPerson[i].box.y;
		bottom = pCfgs->objPerson[i].box.y + pCfgs->objPerson[i].box.height;
		dis1 = min(pCfgs->detLineParm[region_idx].detRight + 5, right) - max(pCfgs->detLineParm[region_idx].detLeft - 5, left);
		dis2 = min(pCfgs->detLineParm[region_idx].detBottom + 5, bottom) - max(pCfgs->detLineParm[region_idx].detTop - 5, top);
		//val1 = pCfgs->detLineParm[region_idx].k * left + pCfgs->detLineParm[region_idx].b - top;
		//val2 = pCfgs->detLineParm[region_idx].k * right + pCfgs->detLineParm[region_idx].b - bottom;
		//去除不在检测区域的目标
		if(dis1 >= 0 && dis2 >= 0)
		//if(dis1 >= 0 && val1 * val2 <= 0)
		{
			;
		}
		else if(pCfgs->objPerson[i].continue_num > 5)//目标存在大于5帧，防止检测框位置跳动
		{
	       /* if(pCfgs->objPerson[i].cal_flow == FALSE)//这里统计容易多检
			{
				get_object_num(pCfgs, i);
				pCfgs->objPerson[i].cal_flow = TRUE;
				//printf("%d,count 2\n",pCfgs->gThisFrameTime);
			}*/
			//DeleteTarget(&pCfgs->objPerson_size, &i, pCfgs->objPerson);
			//continue;
		}
		//保存行人目标框
		if(pCfgs->objPerson[i].detected)
		{
			CRect box = pCfgs->objPerson[i].box;
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].x = box.x;
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].y = box.y;
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].w = box.width;
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].h = box.height;
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].label = 6;//行人
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].confidence = pCfgs->objPerson[i].prob;
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].id =pCfgs->objPerson[i].target_id;//目标id
			int pos_x = min(max(0, (box.x + box.width / 2) / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, width) - 1);
			int pos_y = min(max(0, (box.y + box.height) / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, height) - 1);
			float actual_pos_x = pCfgs->image_actual[pos_y][pos_x][0];
			float actual_pos_y = pCfgs->image_actual[pos_y][pos_x][1];
			//actual_pos_x = (actual_pos_x < 0)? (actual_pos_x - 0.5) : (actual_pos_x + 0.5);
			//actual_pos_y = (actual_pos_y < 0)? (actual_pos_y - 0.5) : (actual_pos_y + 0.5);
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].distance[0] = actual_pos_x;//目标与相机的水平距离
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].distance[1] = actual_pos_y;//目标与相机的垂直距离
			//左上
			pos_x = min(max(0, box.x / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, width) - 1);
			pos_y = min(max(0, box.y / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, height) - 1);
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].border_distance[0][0] = pCfgs->image_actual[pos_y][pos_x][0];
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].border_distance[0][1] = pCfgs->image_actual[pos_y][pos_x][1];
			//右上
			pos_x = min(max(0, (box.x + box.width) / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, width) - 1);
			pos_y = min(max(0, box.y / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, height) - 1);
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].border_distance[1][0] = pCfgs->image_actual[pos_y][pos_x][0];
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].border_distance[1][1] = pCfgs->image_actual[pos_y][pos_x][1];
			//左下
			pos_x = min(max(0, box.x / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, width) - 1);
			pos_y = min(max(0, (box.y + box.height) / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, height) - 1);
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].border_distance[2][0] = pCfgs->image_actual[pos_y][pos_x][0];
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].border_distance[2][1] = pCfgs->image_actual[pos_y][pos_x][1];
			//右下
			pos_x = min(max(0, (box.x + box.width) / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, width) - 1);
			pos_y = min(max(0, (box.y + box.height) / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, height) - 1);
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].border_distance[3][0] = pCfgs->image_actual[pos_y][pos_x][0];
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].border_distance[3][1] = pCfgs->image_actual[pos_y][pos_x][1];
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].length = CalTargetLength(pCfgs->objPerson[i], 0, pCfgs);//长度
			//pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].width = 0;//宽度
			int speed[2] = { 0 };
			CalTargetSpeed(pCfgs->objPerson[i], pCfgs, 0, speed);//计算目标速度
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].speed_Vx = speed[0];//x方向速度
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].speed = speed[1];//y方向速度
			//pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].laneid = pCfgs->objPerson[i].region_idx;//检测区域ID
			pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].laneid = 1;//视频检测结果
			/*if(pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].w <= 0 || pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].h <= 0 || pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].w > width || pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].h > height)
			{
				prt(info," person box width = %d, height = %d",pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].w, pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].h);
			}*/
			//prt(info," person box width = %d, height = %d",pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].w, pCfgs->ResultMsg.uResultInfo.udetPersonBox[nPersonNum].h);
			nPersonNum++;
		}
		int isInRegion = RectInRegion(pParams->MaskDetectImage, pCfgs, width, height, pCfgs->objPerson[i].box, region_idx + 1);
		//当目标在视频存在时间太长或长时间没有检测到或离开图像，删除目标
		if(isInRegion <= 2 || pCfgs->objPerson[i].continue_num > 5000 || pCfgs->objPerson[i].lost_detected > 5 || ((left < 10 || top < 10 || right > (width - 10)  || bottom > (height - 10))&& pCfgs->objPerson[i].lost_detected > 2))
		{
			/*if(pCfgs->objPerson[i].cal_flow == FALSE)
			{
				get_object_num(pCfgs, i);
			    pCfgs->objPerson[i].cal_flow = TRUE;
			}*/
			/*if(pCfgs->objPerson[i].cal_flow == FALSE && pCfgs->objPerson[i].pass_detect_line)//目标最初在检测线上,此处会导致多计数
			{
				get_object_num(pCfgs, i, region_idx);
				pCfgs->objPerson[i].cal_flow = TRUE;
			}*/
            DeleteTarget(&pCfgs->objPerson_size, &i, pCfgs->objPerson);
			continue;

		}

		pCfgs->objPerson[i].continue_num++;
/*#ifdef SAVE_VIDEO
	    if(pCfgs->NCS_ID == 0)
		{
		    cv::rectangle(img, cv::Rect(pCfgs->objPerson[i].box.x - 2, pCfgs->objPerson[i].box.y - 2, pCfgs->objPerson[i].box.width + 4, pCfgs->objPerson[i].box.height + 4), cv::Scalar(0, 255, 0), 1, 8, 0 );
		    char str[10];
		    sprintf(str, "%d", pCfgs->objPerson[i].target_id);
		    putText(img, str, cv::Point(pCfgs->objPerson[i].box.x, max(0, pCfgs->objPerson[i].box.y - 10)), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0 ), 2);
		}
#endif*/

	}
	pCfgs->ResultMsg.uResultInfo.udetPersonNum = nPersonNum;
	printf("person num = %d\n",nPersonNum);
	//统计行人
	nPersonNum = 0;
	for(i = 1; i >= 0; i--)
	{
		pCfgs->uStatPersonNum[i + 1] = pCfgs->uStatPersonNum[i];
	}
	pCfgs->uStatPersonNum[0] = pCfgs->ResultMsg.uResultInfo.udetPersonNum;
	if(pCfgs->uStatPersonNum[3] < 3)
	{
		pCfgs->uStatPersonNum[3]++;
	}
	for(i = 0; i < pCfgs->uStatPersonNum[3]; i++)
	{
		nPersonNum = nPersonNum + pCfgs->uStatPersonNum[i];
	}
	pCfgs->ResultMsg.uResultInfo.udetStatPersonNum = (float)nPersonNum / pCfgs->uStatPersonNum[3];
#ifdef SAVE_VIDEO
	if(pCfgs->NCS_ID == 0)
	{
		char str1[10];
		sprintf(str1, "%d", pCfgs->uPersonDirNum[0][0]);
		putText(img, str1, cv::Point(100, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0 ), 2);
		char str2[10];
		sprintf(str2, "%d", pCfgs->uPersonDirNum[0][1]);
		putText(img, str2, cv::Point(150, 30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255, 0 ), 2);
	}
#endif
	printf("person obj =%d, direction num = [%d,%d]\n",pCfgs->objPerson_size, pCfgs->uPersonDirNum[0][0], pCfgs->uPersonDirNum[0][1]);
	memcpy(pCfgs->uRegionPersonNum, uRegionPersonNum, MAX_REGION_NUM * sizeof(Uint16));//区域行人数
}
#ifdef DETECT_PLATE
//处理车牌检测框
void ProcessPlateBox(Mat BGRImage, ALGCFGS* pCfgs, ALGPARAMS *pParams, int width, int height)
{
	int i = 0, j = 0, k = 0;
	Mat PrePreImage;//计算棒检测的前两帧的检测框
	PrePreImage.create(BGRImage.rows, BGRImage.cols, CV_8UC3);
	memcpy(PrePreImage.data, pParams->PrePreFullImage, BGRImage.rows * BGRImage.cols * 3);
	PlateInfo plateInfo[MAX_PLATE_NUM];
	int plate_num = 0;
	for(i = 0; i < pCfgs->classes; i++)
	{
		if(strcmp(pCfgs->detClasses[i].names, "plate") != 0 || pCfgs->detClasses[i].classes_num <= 0)
			continue;
		for(j = 0; j < pCfgs->detClasses[i].classes_num; j++)
		{
			if((pCfgs->detClasses[i].box[j].y + pCfgs->detClasses[i].box[j].height) < BGRImage.rows / 2 - 1)
				continue;
			cv::Rect rct(pCfgs->detClasses[i].box[j].x, pCfgs->detClasses[i].box[j].y, pCfgs->detClasses[i].box[j].width, pCfgs->detClasses[i].box[j].height);
			//扩展车牌区域
			int minX = rct.x - rct.width * 0.15;
			int maxX = rct.x + rct.width + rct.width * 0.30;
			int minY = rct.y - rct.height / 2;
			int maxY = rct.y + 2 * rct.height;
			minX = max(0, minX);
			maxX = min(maxX, BGRImage.cols - 1);
			minY = max(0, minY);
			maxY = min(maxY, BGRImage.rows - 1);
			rct = Rect(minX, minY, maxX - minX + 1, maxY - minY + 1);
			Mat PlateROI = PrePreImage(rct);
			plateInfo[plate_num]= PlateRecognizeOnly(PlateROI, 0, pCfgs->plate_flag);
			//printf("num = %d, name = %s, type = %d, confidence = %f\n", plate_num, plateInfo[plate_num].plateName, plateInfo[plate_num].plateType, plateInfo[plate_num].confidence);

			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].x = pCfgs->detClasses[i].box[j].x;
			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].y = pCfgs->detClasses[i].box[j].y;
			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].w = pCfgs->detClasses[i].box[j].width;
			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].h = pCfgs->detClasses[i].box[j].height;
			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].confidence = plateInfo[plate_num].confidence;
			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].id = 0;//目标ID
			for( k = 0; k < pCfgs->LaneAmount; k++)//计算与车道相交值
			{
				int overlapNum = RectInRegion(pParams->MaskLaneImage, pCfgs, width, height, pCfgs->detClasses[i].box[j], k + 1);
				if(overlapNum > 50)
					break;
			}
			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].landid = k;//目标所在车道
			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].colour = plateInfo[plate_num].plateType;//车牌颜色
			pCfgs->ResultMsg.uResultInfo.car_number[plate_num].type =  0;//车牌类型
			strcpy(pCfgs->ResultMsg.uResultInfo.car_number[plate_num].car_number, plateInfo[plate_num].plateName);

			/*cv::rectangle(PrePreImage, rct, cv::Scalar(255, 255 ,255), 1, 8, 0 );
			cv::imwrite("result.jpg",PrePreImage);*/
			plate_num++;
		}
	}
	pCfgs->ResultMsg.uResultInfo.udetPlateNum = plate_num;
	memcpy(pParams->PrePreFullImage, pParams->PreFullImage, BGRImage.rows * BGRImage.cols * 3);
	memcpy(pParams->PreFullImage, BGRImage.data, BGRImage.rows *BGRImage.cols * 3);
	PrePreImage.release();
}
#endif
//处理车辆检测框，得到车流量
//void get_lane_params(ALGCFGS *pCfgs, ALGPARAMS *pParams, int laneNum, int imgW, int imgH)
//{
//	//printf("get_target start...................................\n");
//	CPoint ptCorner[MAX_LANE][16];
//	int i = 0, j = 0, k = 0;
//	int left = 0,right = 0, top = 0, bottom = 0;
//	int x1 = 0, x2 = 0, x3 = 0;
//	int maxValue = 0;
//	int maxLane = 0;
//	int dis = 0;
//	int lane_id = 0;
//	int cal_lane_id[2] = { 0 };
//	int nboxes1 = 0;
//	int vehicle_num[MAX_LANE] = { 0 };
//	int vehicle_num1[MAX_LANE] = {0};
//	bool obj_lost[MAX_LANE] = { FALSE };
//	int match_object[MAX_TARGET_NUM] = { 0 };
//	int match_rect[MAX_TARGET_NUM] = { 0 };
//	int match_success = -1, match_obj_idx = 0;
//	int overlap_x = 0, overlap_y = 0;
//	int det_match_object[MAX_TARGET_NUM] = { 0 };
//	int det_match_rect[MAX_TARGET_NUM] = { 0 };
//	int det_match_success = -1, det_match_obj_idx = 0;
//	float sum = 0.0;
//	int dis_x = 0, dis_y = 0;
//	int laneStatus[MAX_LANE][2] = { 0 };//车道占有状态
//	//车道坐标信息
//	for( i = 0; i < laneNum; i++)
//	{
//		ptCorner[i][0] = pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[0];
//		ptCorner[i][1] = pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[1];
//		ptCorner[i][2] = pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[3];
//		ptCorner[i][3] = pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[2];
//		ptCorner[i][4] = pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[0];
//	    ptCorner[i][5] = pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[1];
//		ptCorner[i][6] = pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[3];
//		ptCorner[i][7] = pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[2];
//		ptCorner[i][8] = pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[0];
//		ptCorner[i][9] = pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[1];
//		ptCorner[i][10] = pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[3];
//		ptCorner[i][11] = pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[2];
//		ptCorner[i][12] = pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[0];
//		ptCorner[i][13] = pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[1];
//		ptCorner[i][14] = pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[3];
//		ptCorner[i][15] = pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[2];
//		laneStatus[i][0] = 0;
//		laneStatus[i][1] = 0;
//		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag)
//			pCfgs->existFrameNum[i][0]++;
//		else
//			pCfgs->existFrameNum[i][0] = 0;
//		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag)
//			pCfgs->existFrameNum[i][1]++;
//		else
//			pCfgs->existFrameNum[i][1] = 0;
//		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag == 2)//流量线圈上一帧为出车状态，先置为0
//		{
//			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag = 0;
//		}
//		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag == 2)//后线圈上一帧为出车状态，先置为0
//		{
//			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag = 0;
//		}
//      pCfgs->Tailposition[i] = 0;//头车位置
//      pCfgs->Headposition[i] = 10000;//末车位置
//	}
//	//memset(pCfgs->uDetectVehicleSum, 0, laneNum * sizeof(Uint32));
//	memset(pCfgs->IsCarInTail, 0, laneNum * sizeof(bool));//尾部占有状态
//
//	for( i = 0; i < pCfgs->targets_size; i++)//设置未检测
//	{
//		pCfgs->targets[i].detected = FALSE;
//	}
//
//	//分析车辆检测框
//	for( i = 0; i < pCfgs->classes; i++)
//	{
//		if(strcmp(pCfgs->detClasses[i].names, "car") != 0 && strcmp(pCfgs->detClasses[i].names, "bus") != 0 && strcmp(pCfgs->detClasses[i].names, "truck") != 0)
//			continue;
//		if(pCfgs->detClasses[i].classes_num)
//		{
//			//目标和检测框进行匹配
//			match_object_rect(pCfgs->targets, pCfgs->targets_size, pCfgs->detClasses, i, match_object, match_rect, 10);
//			for( j = 0; j < pCfgs->detClasses[i].classes_num; j++)
//			{
//				//目标和检测框匹配，更新目标信息
//				match_success = -1;
//				for( k = 0; k < pCfgs->targets_size; k++)
//				{
//					if(match_object[j] == k && match_rect[k] == j)
//					{
//						match_success = 1;
//						pCfgs->targets[k].box = pCfgs->detClasses[i].box[j];
//						pCfgs->targets[k].prob = pCfgs->detClasses[i].prob[j];
//						pCfgs->targets[k].class_id = pCfgs->detClasses[i].class_id;
//						strcpy(pCfgs->targets[k].names, pCfgs->detClasses[i].names);
//						pCfgs->targets[k].detected = TRUE;
//						det_match_obj_idx = k;
//						break;
//					}
//				}
//				int overlapNum[MAX_LANE] = {-1};
//				left = max(0, pCfgs->detClasses[i].box[j].x);
//				right = min(pCfgs->detClasses[i].box[j].x + pCfgs->detClasses[i].box[j].width, imgW - 1);
//				top = max(0, pCfgs->detClasses[i].box[j].y);
//				bottom = min(pCfgs->detClasses[i].box[j].y + pCfgs->detClasses[i].box[j].height, imgH - 1);
//				for( k = 0; k < laneNum; k++)//计算与车道相交值
//				{
//					/*x1 = (float)((top + bottom) / 2 - ptCorner[k][0].y) * (float)(ptCorner[k][2].x - ptCorner[k][0].x) / (float)(ptCorner[k][2].y - ptCorner[k][0].y) + ptCorner[k][0].x;
//					x2 = (float)((top + bottom) / 2 - ptCorner[k][1].y) * (float)(ptCorner[k][3].x - ptCorner[k][1].x) / (float)(ptCorner[k][3].y - ptCorner[k][1].y) + ptCorner[k][1].x;
//					x3 = min(x2, right)-max(x1, left);
//					overlapNum[k] = x3;*/
//					overlapNum[k] = RectInRegion(pParams->MaskLaneImage, pCfgs, imgW, imgH, pCfgs->detClasses[i].box[j], k + 1);
//				}
//				//找出相交最大车道
//				maxValue = overlapNum[0];
//				maxLane = 0;
//				for( k = 1; k < laneNum; k++)
//				{
//					if(maxValue < overlapNum[k])
//					{
//						maxValue = overlapNum[k];
//						maxLane = k;
//					}
//				}
//				//if(/*maxValue > 0*/maxValue >= (right - left) / 4)//(right - left) / 4
//				if(maxValue > 10)//检测框10%之上在车道内
//				{
//				    dis = min(max(ptCorner[maxLane][2].y, ptCorner[maxLane][3].y), bottom) - max(ptCorner[maxLane][0].y, top);
//					if(match_success == 1)
//					{
//						pCfgs->targets[det_match_obj_idx].lane_id = maxLane;
//					}
//					if( dis > 10 )//在占有区域到车道区域下端进行计数
//					{
//                      CRect box = pCfgs->detClasses[i].box[j];
//						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].x = box.x;
//						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].y = box.y;
//						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w = box.width;
//						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h = box.height;
//						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].label = pCfgs->detClasses[i].class_id + 1;
//						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].confidence = pCfgs->detClasses[i].prob[j];
//#ifdef SAVE_VIDEO
//      if(pCfgs->NCS_ID == 0)
//		{
//          cv::rectangle(img, cv::Rect(pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].x,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].y,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h), cv::Scalar(0, 0 ,255), 1, 8, 0 );
//      }
//#endif
//						nboxes1++;
//						//vehicle_num[maxLane]++;
//						//pCfgs->uDetectVehicleSum[maxLane]++;//区域车辆数
//						//未匹配，增加新的目标
//						if(match_success < 0 && pCfgs->targets_size < MAX_TARGET_NUM)
//						{
//							CTarget nt; 
//							Initialize_target(&nt);
//							nt.box = pCfgs->detClasses[i].box[j];
//							nt.class_id = pCfgs->detClasses[i].class_id;
//							nt.prob = pCfgs->detClasses[i].prob[j];
//							nt.detected = TRUE;
//							nt.target_id = pCfgs->target_id++;
//							nt.lane_id = maxLane;
//							if(pCfgs->target_id > 5000)
//								pCfgs->target_id = 1;
//							//nt.start_time = pCfgs->currTime;
//							strcpy(nt.names, pCfgs->detClasses[i].names);
//							pCfgs->targets[pCfgs->targets_size] = nt;
//							pCfgs->targets_size++;
//						}
//					}
//					pCfgs->detClasses[i].lane_id[j] = maxLane;
//					vehicle_num1[maxLane]++;
//					pCfgs->Headposition[maxLane] = (pCfgs->Headposition[maxLane] > top)? top : pCfgs->Headposition[maxLane];//头车位置
//					pCfgs->Tailposition[maxLane] = (pCfgs->Tailposition[maxLane] < bottom)? bottom : pCfgs->Tailposition[maxLane];//末车位置
//					if(pCfgs->IsCarInTail[maxLane] == FALSE)
//					{
//						if(min(bottom,  ptCorner[maxLane][6].y) - max(top, ptCorner[maxLane][4].y) > 5 && min(right, ptCorner[maxLane][5].x) - max(left, ptCorner[maxLane][4].x) > 5)//尾部占有状态
//						{
//							pCfgs->IsCarInTail[maxLane] = TRUE;
//							//pCfgs->ResultMsg.uResultInfo.uEachLaneData[maxLane].SpeedDetectInfo1.CoilAttribute[1].calarflag = 1;
//
//						}
//					}
//				}
//				else
//				{
//					pCfgs->detClasses[i].lane_id[j] = -1;
//				}
//			}
//		}
//	}
//	//pCfgs->ResultMsg.uResultInfo.udetNum = nboxes1;    
//	//分析目标
//	for(i = 0;i < pCfgs->targets_size; i++)
//	{
//		//保存轨迹
//		pCfgs->targets[i].trajectory[pCfgs->targets[i].trajectory_num].x = pCfgs->targets[i].box.x + pCfgs->targets[i].box.width / 2;
//		pCfgs->targets[i].trajectory[pCfgs->targets[i].trajectory_num].y = pCfgs->targets[i].box.y + pCfgs->targets[i].box.height / 2;
//		pCfgs->targets[i].trajectory_num++;
//		//检测到，并更新速度
//		if(pCfgs->targets[i].detected)
//		{
//			pCfgs->targets[i].lost_detected = 0;
//            get_speed(&pCfgs->targets[i]);
//		}
//		else//未检测到
//		{
//			pCfgs->targets[i].lost_detected++;
//			pCfgs->targets[i].box.x += pCfgs->targets[i].vx;
//			pCfgs->targets[i].box.y += pCfgs->targets[i].vy;
//		}
//
//		lane_id = pCfgs->targets[i].lane_id;
//		left = pCfgs->targets[i].box.x;
//		right = pCfgs->targets[i].box.x + pCfgs->targets[i].box.width;
//		top = pCfgs->targets[i].box.y;
//		bottom = pCfgs->targets[i].box.y + pCfgs->targets[i].box.height;
//		dis_x = min(max(ptCorner[lane_id][13].x, ptCorner[lane_id][15].x) + 5, right) - max(min(ptCorner[lane_id][12].x, ptCorner[lane_id][14].x) - 5, left);
//		dis_y = min(max(ptCorner[lane_id][14].y, ptCorner[lane_id][15].y) + 5, bottom) - max(min(ptCorner[lane_id][12].y, ptCorner[lane_id][13].y) - 5, top);//计算与流量区域是否相交
//		if(dis_x > 10 && dis_y > 10 && pCfgs->targets[i].detected && pCfgs->targets[i].cal_flow == FALSE)//与流量检测区域相交
//		{
//			if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//此车道存在车
//			{
//				dis = min(ptCorner[lane_id][14].y, bottom) - max(ptCorner[lane_id][12].y, top);
//				if(dis > (ptCorner[lane_id][14].y - ptCorner[lane_id][12].y) / 2 )//当检测框在流量线圈占线圈一半以上，不删除目标，但加入新目标
//				//if(bottom > (ptCorner[lane_id][12].y + ptCorner[lane_id][14].y) / 2)//不删除目标，但加入新目标
//				{
//					for( k = 0; k < pCfgs->targets_size; k++)
//					{
//						if(pCfgs->targets[k].target_id == pCfgs->currFore_target_id[lane_id])
//						{
//							break;
//						}
//					}
//					if(k < pCfgs->targets_size)//先出车，再进车
//					{
//						obj_out_region(pCfgs, maxLane, &pCfgs->targets[k], 0);//流量线圈出车
//					}
//					if(k == pCfgs->targets_size)//此线圈不存在车，但是线圈还是红色
//					{
//						pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].calarflag = 2;//此帧先出车，下帧进入
//						pCfgs->currFore_target_id[lane_id] = pCfgs->targets[i].target_id;
//					}
//				}
//				else//防止误检，不增加新目标
//					continue;
//			}
//			else
//			{
//				pCfgs->currFore_target_id[lane_id] = pCfgs->targets[i].target_id;
//			}
//
//		}
//		//目标进入流量线圈进行计数
//		if(pCfgs->currFore_target_id[lane_id] == pCfgs->targets[i].target_id && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].calarflag == 0 && pCfgs->targets[i].cal_flow == FALSE)//车刚入线圈
//		{
//			obj_in_region(pCfgs, lane_id, 0);//流量线圈车入
//			if(strcmp(pCfgs->targets[i].names, "car") == 0)//car
//			{
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uCarFlow++;
//			}
//			if(strcmp(pCfgs->targets[i].names, "bus") == 0)//bus
//			{
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uBusFlow++;
//			}
//			if(strcmp(pCfgs->targets[i].names, "truck") == 0)//truck
//			{
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uTruckFlow++;
//			}
//			if(strcmp(pCfgs->targets[i].names, "bicycle") == 0)//bicycle
//			{
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uBicycleFlow++;
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.nVehicleFlow++;
//			}
//			if(strcmp(pCfgs->targets[i].names, "motorbike") == 0)//motorbike
//			{
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uMotorbikeFlow++;
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.nVehicleFlow++;
//			}
//			pCfgs->targets[i].cal_flow = TRUE;
//			pCfgs->targets[i].start_time[0] = pCfgs->currTime;//车进入流量线圈
//			pCfgs->targets[i].trajectory_id[0] = pCfgs->targets[i].trajectory_num - 1;
//			pCfgs->targets[i].cal_lane_id[0] = lane_id;//此车道进行了流量线圈流量计数
//			pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].uVehicleType = pCfgs->targets[i].class_id + 1;
//		}
//		//计算车速、车长
//		cal_lane_id[0] = pCfgs->targets[i].cal_lane_id[0];//计算流量所在的车道ID
//		if(pCfgs->targets[i].cal_speed[0] == FALSE && pCfgs->targets[i].cal_flow)//车已经进入流量线圈
//		{
//			dis = min(ptCorner[cal_lane_id[0]][14].y, bottom) - max(ptCorner[cal_lane_id[0]][12].y, top);
//			if(dis >= min(pCfgs->targets[i].box.height, ptCorner[cal_lane_id[0]][14].y - ptCorner[cal_lane_id[0]][12].y - 2))//当车辆到达流量区域下边缘
//			{
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[0]].SpeedDetectInfo1.CoilAttribute[0].DetectOutSum++;
//				//pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[0]].SpeedDetectInfo1.CoilAttribute[0].uVehicleType = pCfgs->targets[i].class_id + 1;
//				//clock_t end_time = clock();
//				//pCfgs->targets[i].end_time = (float)end_time / CLOCKS_PER_SEC;
//				pCfgs->targets[i].end_time[0] = pCfgs->currTime;
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[0]].SpeedDetectInfo1.CoilAttribute[0].uVehicleSpeed  = CalLaneTargetSpeedY(pCfgs->targets[i], cal_lane_id[0], pCfgs);
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[0]].SpeedDetectInfo1.CoilAttribute[0].uVehicleLength  = CalTargetLength(pCfgs->targets[i], cal_lane_id[0], pCfgs);
//				pCfgs->targets[i].cal_speed[0] = TRUE;
//			}
//		}
//#ifdef SAVE_VIDEO
//      if(pCfgs->NCS_ID == 0)
//		{
//          cv::rectangle(img, cv::Rect(pCfgs->targets[i].box.x,pCfgs->targets[i].box.y,pCfgs->targets[i].box.width,pCfgs->targets[i].box.height), cv::Scalar(255, 255 ,255), 1, 8, 0 );
//		    char str[10];
//		    sprintf(str, "%d", pCfgs->targets[i].target_id);
//		    putText(img, str, cv::Point(pCfgs->targets[i].box.x,max(0,pCfgs->targets[i].box.y - 10)), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2);
//		    char str1[10];
//		    sprintf(str1, "%d", pCfgs->currFore_target_id[lane_id]);
//		    putText(img, str1, cv::Point(pCfgs->targets[i].box.x + 30,max(0,pCfgs->targets[i].box.y - 10)), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2);
//     }
//#endif
//		//判断与当前车道流量线圈是否相交
//		dis_x = min(max(ptCorner[cal_lane_id[0]][13].x, ptCorner[cal_lane_id[0]][15].x) + 5, right) - max(min(ptCorner[cal_lane_id[0]][12].x, ptCorner[cal_lane_id[0]][14].x) - 5, left);
//		dis_y = min(max(ptCorner[cal_lane_id[0]][14].y, ptCorner[cal_lane_id[0]][15].y) + 5, bottom) - max(min(ptCorner[cal_lane_id[0]][12].y, ptCorner[cal_lane_id[0]][13].y) - 5, top);
//		//去除不在检测区域的目标
//		if(dis_x > 0 && dis_y > 0)
//		{
//			laneStatus[cal_lane_id[0]][0] = 1;
//		}
//		else if(pCfgs->targets[i].continue_num > 3)
//		{
//			if(pCfgs->currFore_target_id[cal_lane_id[0]] == pCfgs->targets[i].target_id && pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[0]].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//线圈是红色出车
//			{
//				obj_out_region(pCfgs, cal_lane_id[0], &pCfgs->targets[i], 0);//车出
//			}
//		}
//
//		//判断与当前车道是否相交
//		dis_x = min(max(ptCorner[cal_lane_id[0]][1].x, ptCorner[cal_lane_id[0]][3].x), right) - max(min(ptCorner[cal_lane_id[0]][0].x, ptCorner[cal_lane_id[0]][2].x), left);
//		dis_y = min(max(ptCorner[cal_lane_id[0]][2].y, ptCorner[cal_lane_id[0]][3].y), bottom) - max(min(ptCorner[cal_lane_id[0]][0].y, ptCorner[cal_lane_id[0]][1].y), top);
//		if(dis_x < 0 || dis_y < 0)//与车道线圈不相交
//		{
//			if(pCfgs->targets[i].target_id == pCfgs->currRear_target_id[cal_lane_id[0]] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[0]].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//后线圈是红色
//			{
//				obj_out_region(pCfgs, cal_lane_id[0], &pCfgs->targets[i], 0);//流量线圈出车
//				obj_lost[cal_lane_id[0]] = TRUE;
//			}
//		}
//
//		//判断此目标是否进入占位线圈
//		if(min(bottom, ptCorner[lane_id][10].y) - max(top, ptCorner[lane_id][8].y) > 5 && pCfgs->targets[i].target_id != pCfgs->currRear_target_id[lane_id] && pCfgs->targets[i].cal_flow == FALSE)
//		{
//			//满足时间间隔条件，进行判断
//			if((pCfgs->gThisFrameTime - pCfgs->uRearIntervalNum[lane_id]) > 5 || pCfgs->uRearIntervalNum[lane_id] == 0)
//			{
//				//线圈已经有车辆存在
//				if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag)
//				{
//					for( k = 0; k < pCfgs->targets_size; k++)
//					{
//						if(pCfgs->targets[k].target_id == pCfgs->currRear_target_id[lane_id])
//						{
//							break;
//						}
//					}
//					//存在车辆
//					if(k < pCfgs->targets_size)//先出车，再进车
//					{
//						obj_out_region(pCfgs, lane_id, &pCfgs->targets[k], 1);//后线圈出车
//					}
//					//不存在车辆
//					if(k == pCfgs->targets_size)//此线圈不存在车，但是线圈还是红色
//					{
//						pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag = 0;
//					}
//
//				}
//				//printf("lane_id = %d, intervalnum = %d,calarflag = %d,curr_id = %d,%d\n",lane_id,pCfgs->uRearIntervalNum[lane_id],pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag, pCfgs->targets[k].target_id,pCfgs->targets[i].target_id);
//				pCfgs->currRear_target_id[lane_id] = pCfgs->targets[i].target_id;//赋予新的target_id
//				pCfgs->uRearIntervalNum[lane_id] = pCfgs->gThisFrameTime;
//			}
//		}
//		if(pCfgs->targets[i].target_id == pCfgs->currRear_target_id[lane_id] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].calarflag == 0 && pCfgs->targets[i].cal_flow == FALSE)
//		{
//			obj_in_region(pCfgs, lane_id, 1);//后线圈车入
//			pCfgs->targets[i].cal_flow = TRUE;
//			pCfgs->targets[i].start_time[1] = pCfgs->currTime;//车进占位线圈
//			pCfgs->targets[i].trajectory_id[1] = pCfgs->targets[i].trajectory_num - 1;
//			pCfgs->targets[i].cal_lane_id[1] = lane_id;//此车道进行了占位线圈流量计数
//			pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].uVehicleType = pCfgs->targets[i].class_id + 1;
//		}
//
//		cal_lane_id[1] = pCfgs->targets[i].cal_lane_id[1];//计算流量所在的车道ID
//		if(pCfgs->targets[i].cal_speed[1] == FALSE && pCfgs->targets[i].cal_flow)//车已经进入占位线圈
//		{
//			//计算后线圈的流量、车速、车长
//			dis = min(ptCorner[cal_lane_id[1]][10].y, bottom) - max(ptCorner[cal_lane_id[1]][8].y, top);
//			if(dis >= min(pCfgs->targets[i].box.height, ptCorner[cal_lane_id[1]][10].y - ptCorner[cal_lane_id[1]][8].y - 2))//当车辆到达流量区域下边缘
//			{
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[1]].SpeedDetectInfo1.CoilAttribute[1].DetectOutSum++;
//				//pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].uVehicleType = pCfgs->targets[i].class_id + 1;
//				pCfgs->targets[i].end_time[1] = pCfgs->currTime;
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[1]].SpeedDetectInfo1.CoilAttribute[1].uVehicleSpeed  = CalLaneTargetSpeedY(pCfgs->targets[i], cal_lane_id[1], pCfgs);
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[1]].SpeedDetectInfo1.CoilAttribute[1].uVehicleLength = CalTargetLength(pCfgs->targets[i], cal_lane_id[1], pCfgs);
//				pCfgs->targets[i].cal_speed[1] = TRUE;
//			}
//		}
//
//		//判断与当前车道占位线圈是否相交
//		dis_x = min(max(ptCorner[cal_lane_id[1]][9].x,ptCorner[cal_lane_id[1]][11].x) + 5, right) - max(min(ptCorner[cal_lane_id[1]][8].x,ptCorner[cal_lane_id[1]][10].x) - 5, left);
//		dis_y = min(max(ptCorner[cal_lane_id[1]][10].y,ptCorner[cal_lane_id[1]][11].y) + 5, bottom) - max(min(ptCorner[cal_lane_id[1]][8].y,ptCorner[cal_lane_id[1]][9].y) - 5, top);
//		if(dis_x > 0 && dis_y > 0)//与占位线圈相交
//		{
//			laneStatus[cal_lane_id[1]][1] = 1;
//
//		}
//		else//与占位线圈不相交
//		{
//			if(pCfgs->targets[i].target_id == pCfgs->currRear_target_id[cal_lane_id[1]] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[1]].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//线圈是红色
//			{
//				obj_out_region(pCfgs, cal_lane_id[1], &pCfgs->targets[i], 1);//后线圈出车
//			}
//
//		}
//
//		//判断与当前车道是否相交
//		dis_x = min(max(ptCorner[cal_lane_id[1]][1].x, ptCorner[cal_lane_id[1]][3].x), right) - max(min(ptCorner[cal_lane_id[1]][0].x, ptCorner[cal_lane_id[1]][2].x), left);
//		dis_y = min(max(ptCorner[cal_lane_id[1]][2].y, ptCorner[cal_lane_id[1]][3].y), bottom) - max(min(ptCorner[cal_lane_id[1]][0].y, ptCorner[cal_lane_id[1]][1].y), top);
//		if(dis_x < 0 || dis_y < 0)//与车道线圈不相交
//		{
//			if(pCfgs->targets[i].target_id == pCfgs->currRear_target_id[cal_lane_id[1]] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[1]].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//后线圈是红色
//			{
//				obj_out_region(pCfgs, cal_lane_id[1], &pCfgs->targets[i], 1);//后线圈出车
//			}
//		}
//		//当目标在视频存在时间太长或长时间没有检测到或离开图像，删除目标
//		bool isLost = FALSE;
//		isLost = pCfgs->targets[i].lost_detected > 5 && (strcmp(pCfgs->targets[i].names, "bus") != 0 &&  strcmp(pCfgs->targets[i].names, "truck") != 0 );
//		isLost = pCfgs->targets[i].lost_detected > 20 && (strcmp(pCfgs->targets[i].names, "bus") == 0 ||  strcmp(pCfgs->targets[i].names, "truck") == 0 );
//		//当目标在视频存在时间太长或长时间没有检测到或离开图像，删除目标
//		if(pCfgs->targets[i].continue_num > 5000 || isLost || (pCfgs->targets[i].lost_detected > 5 && (left < 5 || top < 5 || right >= (imgW - 5)  || bottom >= (imgH - 5))))
//		//if(pCfgs->targets[i].continue_num > 5000 /*|| (pCfgs->targets[i].lost_detected > 5 && dis <= 0)*/ || isLost/*pCfgs->targets[i].lost_detected > 5*/ || dis_x < 0 || dis_y < 0)//10
//		//if(pCfgs->targets[i].continue_num > 5000 || (pCfgs->targets[i].lost_detected > 5 && pCfgs->targets[i].box.y > ptCorner[lane_id][14].y)||(pCfgs->targets[i].lost_detected > 10 && (pCfgs->targets[i].box.y> ptCorner[lane_id][12].y))|| (pCfgs->targets[i].box.x < 0 || pCfgs->targets[i].box.y < 0 || (pCfgs->targets[i].box.x + pCfgs->targets[i].box.width) > imgW  ||((pCfgs->targets[i].box.y + pCfgs->targets[i].box.height) > imgH)))
//		{
//			if(pCfgs->currFore_target_id[cal_lane_id[0]] == pCfgs->targets[i].target_id && pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[0]].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//线圈是红色
//			{
//				obj_out_region(pCfgs, cal_lane_id[0], &pCfgs->targets[i], 0);//流量线圈出车
//			    obj_lost[cal_lane_id[0]] = TRUE;
//			}
//			if(pCfgs->targets[i].target_id == pCfgs->currRear_target_id[cal_lane_id[1]] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[cal_lane_id[1]].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//线圈是红色
//			{
//				obj_out_region(pCfgs, cal_lane_id[1], &pCfgs->targets[i], 1);//后线圈出车
//			}
//			DeleteTarget(&pCfgs->targets_size, &i, pCfgs->targets);
//			continue;
//
//		}
//
//		//得到跟踪框
//		dis = min(max(ptCorner[lane_id][2].y, ptCorner[lane_id][3].y), bottom) - max(min(ptCorner[lane_id][4].y, ptCorner[lane_id][5].y), top);
//		if(dis > 0)
//		{
//			//计算区域内的车辆数
//			vehicle_num[lane_id]++;
//			//将跟踪框代替检测框
//          //CRect box = pCfgs->targets[i].box;
//			//pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1++] = box;
//			/*pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].x = box.x;
//			pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].y = box.y;
//			pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w = box.width;
//			pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h = box.height;
//			pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].label = pCfgs->targets[i].class_id + 1;
//			pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].confidence = pCfgs->targets[i].prob;
//			nboxes1++;*/
//		}
//		pCfgs->targets[i].continue_num++;
//
//	}
//	pCfgs->ResultMsg.uResultInfo.udetNum = nboxes1; 
//
//	//防止车辆跳变对区域车辆数进行处理
//	for( i = 0; i < laneNum; i++)
//	{
//		//判断是否与线圈相交
//		for(j = 0; j < 2; j++)
//		{
//			if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[j].calarflag == 1 && laneStatus[i][j] == 0 && pCfgs->existFrameNum[i][j] > 3)//没有目标在线圈内
//			{
//				if(j == 0)//流量线圈
//				{
//					for(k = 0; k < pCfgs->targets_size; k++)
//					{
//						if(pCfgs->currFore_target_id[i] == pCfgs->targets[k].target_id)
//							obj_out_region(pCfgs, i, &pCfgs->targets[k], 0);
//					}
//					if(k == pCfgs->targets_size)//没有目标
//					{
//						if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//车出线圈时，calarflag设置为2
//							pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag = 2;
//					}
//				}
//				if(j == 1)//后线圈
//				{
//					for(k = 0; k < pCfgs->targets_size; k++)
//					{
//						if(pCfgs->currRear_target_id[i] == pCfgs->targets[k].target_id)
//							obj_out_region(pCfgs, i, &pCfgs->targets[k], 1);
//					}
//					if(k == pCfgs->targets_size)//没有目标
//					{
//						if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//车出线圈时，calarflag设置为2
//							pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag = 2;
//					}
//
//				}
//				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[j].calarflag = 2;
//				pCfgs->existFrameNum[i][j] = 0;
//			}
//		}
//		//printf("laneID = %d, vehicle num = %d\n",i,vehicle_num1[i]);
//		if(pCfgs->gThisFrameTime < 10)//对最初几帧不进行处理
//		{
//			pCfgs->uDetectVehicleSum[i] = vehicle_num[i];
//		}
//		else
//		{
//			int curr_num = vehicle_num[i];
//			//printf("lane id = %d,curr num = %d,old num =%d\n",i,curr_num,pCfgs->uDetectVehicleSum[i]);
//			if(pCfgs->uDetectVehicleSum[i] != curr_num)
//			{
//				if(pCfgs->uDetectVehicleFrameNum[i] > 10)//最短10帧才发生车辆数的变化
//				{
//					curr_num = (pCfgs->uDetectVehicleSum[i] > curr_num)? (pCfgs->uDetectVehicleSum[i] - 1) : (pCfgs->uDetectVehicleSum[i] + 1);
//				}
//				else if(pCfgs->uDetectVehicleSum[i] >  curr_num && obj_lost[i])//有车出
//				{
//					curr_num = pCfgs->uDetectVehicleSum[i] - 1;
//				}
//				else
//				{
//					curr_num = pCfgs->uDetectVehicleSum[i];
//				}
//				if(vehicle_num[i] == 0 && curr_num != 0)//当前帧没有区域车辆数，快速将车辆数降下
//				{
//					if(pCfgs->uDetectVehicleFrameNum[i] > 5)
//						curr_num = pCfgs->uDetectVehicleSum[i] - 1;
//				}
//			}
//			//printf("lane id1 = %d,curr num = %d,old num =%d interval_num =%d\n",i,curr_num,pCfgs->uDetectVehicleSum[i],pCfgs->uDetectVehicleFrameNum[i]);
//			if(pCfgs->uDetectVehicleSum[i] != curr_num)//前后两帧的车辆数不相同时，重新计数
//				pCfgs->uDetectVehicleFrameNum[i] = 0;
//			if(pCfgs->uDetectVehicleSum[i] == curr_num)//前后两帧车辆数相同
//				pCfgs->uDetectVehicleFrameNum[i]++;
//			pCfgs->uDetectVehicleSum[i] = curr_num;//将处理后的数量赋给当前车辆数
//		}
//		if(pCfgs->uDetectVehicleSum[i] == 0 || vehicle_num1[i] == 0)//车道区域内无车
//		{
//			pCfgs->Tailposition[i] = min(ptCorner[i][2].y, ptCorner[i][3].y);
//			pCfgs->Headposition[i] = min(ptCorner[i][2].y, ptCorner[i][3].y);
//		}
//	}
//
//	/*for( i = 0; i < laneNum; i++)
//	{
//		for( j = 1;j >= 0; j--)
//		{
//			pCfgs->uStatVehicleSum[i][j + 1] = pCfgs->uStatVehicleSum[i][j];
//		}
//		pCfgs->uStatVehicleSum[i][0] = vehicle_num[i];
//		if(pCfgs->uStatVehicleSum[i][3] < 3)
//			pCfgs->uStatVehicleSum[i][3] = pCfgs->uStatVehicleSum[i][3] + 1;
//
//		sum = 0.0;
//		for(j = 0; j < pCfgs->uStatVehicleSum[i][3];j++)
//			sum +=  pCfgs->uStatVehicleSum[i][j];
//		sum = sum / pCfgs->uStatVehicleSum[i][3];
//		//防止区域车辆数跳变
//		if(pCfgs->uDetectVehicleSum[i] > vehicle_num[i] && obj_lost[i])//有车出，数量减1
//		{
//			pCfgs->uDetectVehicleSum[i] = pCfgs->uDetectVehicleSum[i] - 1;
//		} 
//		else if(vehicle_num[i] > pCfgs->uDetectVehicleSum[i])
//		{
//			//pCfgs->uDetectVehicleSum[i] = vehicle_num[i];
//			pCfgs->uDetectVehicleSum[i] = sum + 0.5;
//		} 
//		else if(pCfgs->uDetectVehicleSum[i] > vehicle_num[i] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag == 0)//流量区域无车，采用实际检测数
//		{
//			//pCfgs->uDetectVehicleSum[i] = vehicle_num[i];//实际车辆数
//			//防止车辆数跳变
//			pCfgs->uDetectVehicleSum[i] = sum + 0.5;
//		}
//		if(vehicle_num[i] == 0)//车道区域内无车
//		{
//		pCfgs->Tailposition[i] = min(ptCorner[i][2].y, ptCorner[i][3].y);
//		pCfgs->Headposition[i] = min(ptCorner[i][2].y, ptCorner[i][3].y);
//		}
//#ifdef SAVE_VIDEO
//      if(pCfgs->NCS_ID == 0)
//		{
//          char str[10];
//		    sprintf(str, "%d", pCfgs->uDetectVehicleSum[i]);
//		    putText(img, str, Point(10 + 30 * i,10), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2);
//		    char str1[10];
//		    sprintf(str1, "%d", pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].car_out);
//		    putText(img, str1, Point(10 + 30 * i,30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2);
//       }
//#endif
//	}*/
//	//printf("get_target end...................................\n");
//}
//处理车辆检测框，得到车流量
void get_lane_params(ALGCFGS *pCfgs, ALGPARAMS *pParams, int laneNum, int imgW, int imgH)
{
	//printf("get_target start...................................\n");
	CPoint ptCorner[MAX_LANE][16];
	int i = 0, j = 0, k = 0;
	int left = 0,right = 0, top = 0, bottom = 0;
	int x1 = 0, x2 = 0, x3 = 0;
	int maxValue = 0;
	int maxLane = 0;
	int dis = 0;
	int lane_id = 0;
	int nboxes1 = 0;
	int vehicle_num[MAX_LANE] = { 0 };
	int vehicle_num1[MAX_LANE] = {0};
	bool obj_lost[MAX_LANE] = { FALSE };
	//int match_object[MAX_TARGET_NUM] = { 0 };
	//int match_rect[MAX_TARGET_NUM] = { 0 };
	int match_object[MAX_CLASSES][MAX_DETECTION_NUM] = { 0 };
	int match_rect[MAX_TARGET_NUM][3] = { 0 };
	int match_success = -1, match_obj_idx = 0;
	int overlap_x = 0, overlap_y = 0;
	//int det_match_object[MAX_TARGET_NUM] = { 0 };
	//int det_match_rect[MAX_TARGET_NUM] = { 0 };
	int det_match_object[MAX_CLASSES][MAX_DETECTION_NUM] = { 0 };
	int det_match_rect[MAX_TARGET_NUM][3] = { 0 };
	int det_match_success = -1, det_match_obj_idx = 0;
	float sum = 0.0;
	int dis_x = 0, dis_y = 0;
	int laneStatus[MAX_LANE][2] = { 0 };//车道占有状态
	int lane_type = 0;//0   竖向车道   1 横向车道
	//车道坐标信息
	for( i = 0; i < laneNum; i++)
	{
		ptCorner[i][0] = pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[0];
		ptCorner[i][1] = pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[1];
		ptCorner[i][2] = pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[3];
		ptCorner[i][3] = pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[2];
		ptCorner[i][4] = pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[0];
		ptCorner[i][5] = pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[1];
		ptCorner[i][6] = pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[3];
		ptCorner[i][7] = pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[2];
		ptCorner[i][8] = pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[0];
		ptCorner[i][9] = pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[1];
		ptCorner[i][10] = pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[3];
		ptCorner[i][11] = pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[2];
		ptCorner[i][12] = pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[0];
		ptCorner[i][13] = pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[1];
		ptCorner[i][14] = pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[3];
		ptCorner[i][15] = pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[2];
		laneStatus[i][0] = 0;
		laneStatus[i][1] = 0;
		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag)
			pCfgs->existFrameNum[i][0]++;
		else
			pCfgs->existFrameNum[i][0] = 0;
		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag)
			pCfgs->existFrameNum[i][1]++;
		else
			pCfgs->existFrameNum[i][1] = 0;
		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag == 2)//流量线圈上一帧为出车状态，先置为0
		{
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag = 0;
		}
		if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag == 2)//后线圈上一帧为出车状态，先置为0
		{
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag = 0;
		}
		pCfgs->Tailposition[i] = 0;//头车位置
		pCfgs->Headposition[i] = 10000;//末车位置
	}
	//memset(pCfgs->uDetectVehicleSum, 0, laneNum * sizeof(Uint32));
	memset(pCfgs->IsCarInTail, 0, laneNum * sizeof(bool));//尾部占有状态

	for( i = 0; i < pCfgs->targets_size; i++)//设置未检测
	{
		pCfgs->targets[i].detected = FALSE;
	}
	for( i = 0; i < pCfgs->detTargets_size; i++)//设置未检测
	{
		pCfgs->detTargets[i].detected = FALSE;
	}
	match_object_rect1(pCfgs, pCfgs->targets, pCfgs->targets_size, match_object, match_rect, 15);
	match_object_rect1(pCfgs, pCfgs->detTargets, pCfgs->detTargets_size, det_match_object, det_match_rect, 5);
	//分析车辆检测框
	for( i = 0; i < pCfgs->classes; i++)
	{
		if(strcmp(pCfgs->detClasses[i].names, "car") != 0 && strcmp(pCfgs->detClasses[i].names, "bus") != 0 && strcmp(pCfgs->detClasses[i].names, "truck") != 0 \
			&& strcmp(pCfgs->detClasses[i].names, "motorbike") != 0 && strcmp(pCfgs->detClasses[i].names, "bicycle") != 0)
			continue;
		if(pCfgs->detClasses[i].classes_num)
		{
			//目标和检测框进行匹配
			//match_object_rect(pCfgs->targets, pCfgs->targets_size, pCfgs->detClasses, i, match_object, match_rect, 15);
			//match_object_rect(pCfgs->detTargets, pCfgs->detTargets_size, pCfgs->detClasses, i, det_match_object, det_match_rect, 5);
			for( j = 0; j < pCfgs->detClasses[i].classes_num; j++)
			{
				//目标和检测框匹配，更新目标信息
				det_match_success = -1;
				for( k = 0; k < pCfgs->detTargets_size; k++)
				{
					/*if(det_match_object[j] == k && det_match_rect[k] == j)
					{
						det_match_success = 1;
						break;
					}*/
					if(det_match_object[i][j] == k && det_match_rect[k][0] == i && det_match_rect[k][1] == j)
					{
						det_match_success = 1;
						break;
					}
					if(det_match_object[i][j] == k && (det_match_rect[k][0] != i || det_match_rect[k][1] != j))//两个检测框都匹配一个目标框，没有匹配上，不加入新目标
						det_match_success = 0;

				}
				if(det_match_success == 1)
				{
					pCfgs->detTargets[k].box = pCfgs->detClasses[i].box[j];
					pCfgs->detTargets[k].prob = pCfgs->detClasses[i].prob[j];
					pCfgs->detTargets[k].class_id = pCfgs->detClasses[i].class_id;
					strcpy(pCfgs->detTargets[k].names, pCfgs->detClasses[i].names);
					pCfgs->detTargets[k].detected = TRUE;
					det_match_obj_idx = k;
				}
				match_success = -1;
				for( k = 0; k < pCfgs->targets_size; k++)
				{
					if(match_object[i][j] == k && match_rect[k][0] == i && match_rect[k][1] == j)
					{
						match_success = 1;
						pCfgs->targets[k].box = pCfgs->detClasses[i].box[j];
						pCfgs->targets[k].prob = pCfgs->detClasses[i].prob[j];
						pCfgs->targets[k].class_id = pCfgs->detClasses[i].class_id;
						strcpy(pCfgs->targets[k].names, pCfgs->detClasses[i].names);
						pCfgs->targets[k].detected = TRUE;
						break;
					}
					if(match_object[i][j] == k && (match_rect[k][0] != i || match_rect[k][1] != j) && match_rect[k][2] > 20)//两个检测框都匹配一个目标框，没有匹配上，不加入新目标
						match_success = 0;
					/*if(match_object[j] == k && match_rect[k] == j)
					{
						match_success = 1;
						pCfgs->targets[k].box = pCfgs->detClasses[i].box[j];
						pCfgs->targets[k].prob = pCfgs->detClasses[i].prob[j];
						pCfgs->targets[k].class_id = pCfgs->detClasses[i].class_id;
						strcpy(pCfgs->targets[k].names, pCfgs->detClasses[i].names);
						pCfgs->targets[k].detected = TRUE;
						break;
					}*/
				}
				int overlapNum[MAX_LANE] = {-1};
				left = max(0, pCfgs->detClasses[i].box[j].x);
				right = min(pCfgs->detClasses[i].box[j].x + pCfgs->detClasses[i].box[j].width, imgW - 1);
				top = max(0, pCfgs->detClasses[i].box[j].y);
				bottom = min(pCfgs->detClasses[i].box[j].y + pCfgs->detClasses[i].box[j].height, imgH - 1);
				for( k = 0; k < laneNum; k++)//计算与车道相交值
				{
					/*x1 = (float)((top + bottom) / 2 - ptCorner[k][0].y) * (float)(ptCorner[k][2].x - ptCorner[k][0].x) / (float)(ptCorner[k][2].y - ptCorner[k][0].y) + ptCorner[k][0].x;
					x2 = (float)((top + bottom) / 2 - ptCorner[k][1].y) * (float)(ptCorner[k][3].x - ptCorner[k][1].x) / (float)(ptCorner[k][3].y - ptCorner[k][1].y) + ptCorner[k][1].x;
					x3 = min(x2, right)-max(x1, left);
					overlapNum[k] = x3;*/
					overlapNum[k] = RectInRegion(pParams->MaskLaneImage, pCfgs, imgW, imgH, pCfgs->detClasses[i].box[j], k + 1);
				}
				//找出相交最大车道
				maxValue = overlapNum[0];
				maxLane = 0;
				for( k = 1; k < laneNum; k++)
				{
					if(maxValue < overlapNum[k])
					{
						maxValue = overlapNum[k];
						maxLane = k;
					}
				}
				//if(/*maxValue > 0*/maxValue >= (right - left) / 4)//(right - left) / 4
				if(maxValue > 10)//检测框10%之上在车道内
				{
					dis_x = min(max(ptCorner[maxLane][1].x, ptCorner[maxLane][3].x), right) - max(min(ptCorner[maxLane][0].x, ptCorner[maxLane][2].x), left);
					dis_y = min(max(ptCorner[maxLane][2].y, ptCorner[maxLane][3].y), bottom) - max(min(ptCorner[maxLane][0].y, ptCorner[maxLane][1].y), top);
					if(det_match_success == 1)
					{
						pCfgs->detTargets[det_match_obj_idx].lane_id = maxLane;
					}
					if( dis_x  > 10 && dis_y > 10)//在占有区域到车道区域下端进行计数
					{
						/*CRect box = pCfgs->detClasses[i].box[j];
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].x = box.x;
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].y = box.y;
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w = box.width;
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h = box.height;
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].label = pCfgs->detClasses[i].class_id + 1;
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].confidence = pCfgs->detClasses[i].prob[j];
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].id = 0;//目标ID
						int pos_x = min(max(0, (box.x + box.width / 2) / pCfgs->scale_x, min(MAX_IMAGE_WIDTH, width) - 1);
						int pos_y = min(max(0, (box.y + box.height / 2) / pCfgs->scale_y, min(MAX_IMAGE_HEIGHT, height) - 1);
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].distance[0] = pCfgs->image_actual[pos_y][pos_x][0];//目标与相机的水平距离
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].distance[1] = pCfgs->image_actual[pos_y][pos_x][1];//目标与相机的垂直距离
						pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].laneid = 0;//车道号
#ifdef SAVE_VIDEO
					    if(pCfgs->NCS_ID == 0)
		                {
						    cv::rectangle(img, cv::Rect(pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].x,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].y,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h), cv::Scalar(0, 0 ,255), 1, 8, 0 );
						}
#endif
						nboxes1++;*/
						//vehicle_num[maxLane]++;
						//pCfgs->uDetectVehicleSum[maxLane]++;//区域车辆数
						//未匹配，增加新的目标
						if(det_match_success < 0 && pCfgs->detTargets_size < MAX_TARGET_NUM)
						{
							CTarget nt; 
							Initialize_target(&nt);
							nt.box = pCfgs->detClasses[i].box[j];
							nt.class_id = pCfgs->detClasses[i].class_id;
							nt.prob = pCfgs->detClasses[i].prob[j];
							nt.detected = TRUE;
							nt.target_id = pCfgs->detTarget_id++;
							nt.lane_id = maxLane;
							if(pCfgs->detTarget_id > 5000)
								pCfgs->detTarget_id = 1;
							nt.start_time = pCfgs->currTime;
							strcpy(nt.names, pCfgs->detClasses[i].names);
							pCfgs->detTargets[pCfgs->detTargets_size] = nt;
							pCfgs->detTargets_size++;
						}
					}
					pCfgs->detClasses[i].lane_id[j] = maxLane;
					vehicle_num1[maxLane]++;
					pCfgs->Headposition[maxLane] = (pCfgs->Headposition[maxLane] > top)? top : pCfgs->Headposition[maxLane];//头车位置
					pCfgs->Tailposition[maxLane] = (pCfgs->Tailposition[maxLane] < bottom)? bottom : pCfgs->Tailposition[maxLane];//末车位置
					if(pCfgs->IsCarInTail[maxLane] == FALSE)
					{
						if(min(bottom,  ptCorner[maxLane][6].y) - max(top, ptCorner[maxLane][4].y) > 5 && min(right, ptCorner[maxLane][5].x) - max(left, ptCorner[maxLane][4].x) > 5)//尾部占有状态
						{
							pCfgs->IsCarInTail[maxLane] = TRUE;
							//pCfgs->ResultMsg.uResultInfo.uEachLaneData[maxLane].SpeedDetectInfo1.CoilAttribute[1].calarflag = 1;

						}
					}
					//与流量区域相交
					dis_x = min(ptCorner[maxLane][13].x, right) - max(ptCorner[maxLane][12].x, left);
					dis_y = min(ptCorner[maxLane][14].y, bottom) - max(ptCorner[maxLane][12].y, top);
					//if(min(ptCorner[maxLane][14].y, bottom) - max(ptCorner[maxLane][12].y, top) > 0)
					{
						if(match_success >= 0)//匹配成功
						{
							;
						}
						else if(strcmp(pCfgs->detClasses[i].names, "motorbike") == 0 || strcmp(pCfgs->detClasses[i].names, "bicycle") == 0)//非机动车不处理
						{
							;
						}
						else if(dis_x > 10 && dis_y > 10 && pCfgs->targets_size < MAX_TARGET_NUM)//检测框与流量相交，增加新目标
						{
							//if(top > max(ptCorner[maxLane][12].y, ptCorner[maxLane][2].y - 100))//防止误检,车top已经进入线圈，不再加入到目标中   加上此限制条件，车尾入车慢
							//	continue;
							/*for( k = 0; k < pCfgs->targets_size; k++)
							{
								if(pCfgs->targets[k].target_id == pCfgs->currFore_target_id[maxLane])
								{
									if(pCfgs->targets[k].cal_flow == FALSE)
									{
										break;
									}
								}
							}
							if(k < pCfgs->targets_size)//当前车道的目标ID还没有进行流量统计
								continue;*/
							if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[maxLane].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//此车道存在车
							{
								if((lane_type == 0 && dis_y > (ptCorner[maxLane][14].y - ptCorner[maxLane][12].y) / 2) || (lane_type == 1 && dis_x > (ptCorner[maxLane][13].x - ptCorner[maxLane][12].x) / 2))//当检测框在流量线圈占线圈一半以上，不删除目标，但加入新目标
								//if(bottom > (ptCorner[maxLane][12].y + ptCorner[maxLane][14].y) / 2)//不删除目标，但加入新目标
								{
									for( k = 0; k < pCfgs->targets_size; k++)
									{
										if(pCfgs->targets[k].target_id == pCfgs->currFore_target_id[maxLane])
										{
											break;
										}
									}
									if(k < pCfgs->targets_size)//先出车，再进车
									{
										obj_out_region(pCfgs, maxLane, &pCfgs->targets[k], 0);//流量线圈出车
									}
									if( k == pCfgs->targets_size)
									{
										pCfgs->ResultMsg.uResultInfo.uEachLaneData[maxLane].SpeedDetectInfo1.CoilAttribute[0].calarflag = 2;
									}
								}
								else//防止误检，不增加新目标
									continue;
							}
							//增加新的目标
							CTarget nt; 
							Initialize_target(&nt);
							nt.box = pCfgs->detClasses[i].box[j];
							nt.class_id = pCfgs->detClasses[i].class_id;
							nt.prob = pCfgs->detClasses[i].prob[j];
							nt.detected = TRUE;
							nt.target_id = pCfgs->target_id++;
							nt.lane_id = maxLane;
							//clock_t start_time = clock();
							//nt.start_time = (float)start_time / CLOCKS_PER_SEC;
							nt.start_time = pCfgs->currTime;
							if(pCfgs->target_id > 5000)
								pCfgs->target_id = 1;
							strcpy(nt.names, pCfgs->detClasses[i].names);
							//printf("%s\n",pCfgs->detClasses[i].names);
							pCfgs->targets[pCfgs->targets_size] = nt;
							pCfgs->targets_size++;
							//obj_in_region(pCfgs, maxLane, 0);//车入
							pCfgs->currFore_target_id[maxLane] = nt.target_id;

						}
					}
				}
				else
				{
					pCfgs->detClasses[i].lane_id[j] = -1;
				}
			}
		}
	}
	//pCfgs->ResultMsg.uResultInfo.udetNum = nboxes1;    
	//分析目标
	for(i = 0;i < pCfgs->targets_size; i++)
	{
	
		//保存轨迹，轨迹数小于3000，直接保存，大于3000，去除旧的
		int trajectory_num = pCfgs->targets[i].trajectory_num;
		if(trajectory_num < 3000)
		{

			pCfgs->targets[i].trajectory[trajectory_num].x = pCfgs->targets[i].box.x + pCfgs->targets[i].box.width / 2;
			pCfgs->targets[i].trajectory[trajectory_num].y = pCfgs->targets[i].box.y + pCfgs->targets[i].box.height / 2;
			pCfgs->targets[i].trajectory_time[trajectory_num] = pCfgs->currTime;
			pCfgs->targets[i].trajectory_num++;
		}
		else
		{
			for(j = 0; j < trajectory_num - 1; j++)
			{
				pCfgs->targets[i].trajectory[j] = pCfgs->targets[i].trajectory[j + 1];
				pCfgs->targets[i].trajectory_time[j] = pCfgs->targets[i].trajectory_time[j + 1];
			}
			pCfgs->targets[i].trajectory[trajectory_num - 1].x = pCfgs->targets[i].box.x + pCfgs->targets[i].box.width / 2;
			pCfgs->targets[i].trajectory[trajectory_num - 1].y = pCfgs->targets[i].box.y + pCfgs->targets[i].box.height / 2;
			pCfgs->targets[i].trajectory_time[trajectory_num - 1] = pCfgs->currTime;
		}
		//检测到，并更新速度
		if(pCfgs->targets[i].detected)
		{
			pCfgs->targets[i].lost_detected = 0;
            get_speed(&pCfgs->targets[i]);
		}
		else//未检测到
		{
			pCfgs->targets[i].lost_detected++;
			pCfgs->targets[i].box.x += pCfgs->targets[i].vx;
			pCfgs->targets[i].box.y += pCfgs->targets[i].vy;
		}

		lane_id = pCfgs->targets[i].lane_id;
		left = pCfgs->targets[i].box.x;
		right = pCfgs->targets[i].box.x + pCfgs->targets[i].box.width;
		top = pCfgs->targets[i].box.y;
		bottom = pCfgs->targets[i].box.y + pCfgs->targets[i].box.height;
		dis_x = min(max(ptCorner[lane_id][13].x, ptCorner[lane_id][15].x) + 5, right) - max(min(ptCorner[lane_id][12].x, ptCorner[lane_id][14].x) - 5, left);
		dis_y = min(max(ptCorner[lane_id][14].y, ptCorner[lane_id][15].y) + 5, bottom) - max(min(ptCorner[lane_id][12].y, ptCorner[lane_id][13].y) - 5, top);
		//目标进入流量线圈进行计数
		if(pCfgs->currFore_target_id[lane_id] == pCfgs->targets[i].target_id && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].calarflag == 0 && pCfgs->targets[i].cal_flow == FALSE)//车刚入线圈
		{
			obj_in_region(pCfgs, lane_id, 0);//流量线圈车入
			if(strcmp(pCfgs->targets[i].names, "car") == 0)//car
			{
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uCarFlow++;
			}
			if(strcmp(pCfgs->targets[i].names, "bus") == 0)//bus
			{
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uBusFlow++;
			}
			if(strcmp(pCfgs->targets[i].names, "truck") == 0)//truck
			{
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uTruckFlow++;
			}
			if(strcmp(pCfgs->targets[i].names, "bicycle") == 0)//bicycle
			{
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uBicycleFlow++;
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.nVehicleFlow++;
			}
			if(strcmp(pCfgs->targets[i].names, "motorbike") == 0)//motorbike
			{
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.uMotorbikeFlow++;
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.nVehicleFlow++;
			}
			pCfgs->targets[i].cal_flow = TRUE;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].uVehicleType = pCfgs->targets[i].class_id + 1;
		}
		//计算车速、车长
		if(pCfgs->currFore_target_id[lane_id] == pCfgs->targets[i].target_id && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//车已经进入线圈
		{
			//计算车速
			bool isTrue = FALSE;
			//lane_type = pCfgs->DownDetectCfg.FvdDetectCfg.EachLaneCfg[lane_id].LaneType;//0：竖向 1：横向
			isTrue = (dis_y >= min(pCfgs->targets[i].box.height, ptCorner[lane_id][14].y - ptCorner[lane_id][12].y - 2) && lane_type == 0) || (dis_x >= min(pCfgs->targets[i].box.width, ptCorner[lane_id][13].x - ptCorner[lane_id][12].x - 2) && lane_type == 1);
			dis = min(ptCorner[lane_id][14].y, bottom) - max(ptCorner[lane_id][12].y, top);
			if(isTrue && pCfgs->targets[i].cal_speed == FALSE && pCfgs->targets[i].continue_num)//当车辆到达流量区域下边缘
				//if(pCfgs->targets[i].box.y + pCfgs->targets[i].box.height > (ptCorner[lane_id][14].y + ptCorner[lane_id][15].y)/2 && pCfgs->targets[i].cal_speed == FALSE && pCfgs->targets[i].continue_num)//当车辆到达流量区域下边缘
			{
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].DetectOutSum++;
				//pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].uVehicleType = pCfgs->targets[i].class_id + 1;
				//clock_t end_time = clock();
				//pCfgs->targets[i].end_time = (float)end_time / CLOCKS_PER_SEC;
				pCfgs->targets[i].end_time = pCfgs->currTime;
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].uVehicleSpeed  = CalLaneTargetSpeedY(pCfgs->targets[i], lane_id, pCfgs);
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].uVehicleLength  = CalTargetLength(pCfgs->targets[i], lane_id, pCfgs);
				pCfgs->targets[i].cal_speed = TRUE;
			}
		}

		//判断与当前车道流量线圈是否相交
		if(dis_x > 0 && dis_y > 0)
		{
			laneStatus[lane_id][0] = 1;
		}
		else if(pCfgs->targets[i].continue_num > 3)
		{
			if(pCfgs->currFore_target_id[lane_id] == pCfgs->targets[i].target_id && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//线圈是红色出车
				obj_out_region(pCfgs, lane_id, &pCfgs->targets[i], 0);//车出
			obj_lost[lane_id] = TRUE;
			//DeleteTarget(&pCfgs->targets_size, &i, pCfgs->targets);//这里删除目标，容易导致跳动框删除目标，然后多计数
			//continue;
		}
		//判断与当前车道是否相交
		//dis_x = min(max(ptCorner[lane_id][1].x, ptCorner[lane_id][3].x), right) - max(min(ptCorner[lane_id][0].x, ptCorner[lane_id][2].x), left);
		//dis_y = min(max(ptCorner[lane_id][2].y, ptCorner[lane_id][3].y), bottom) - max(min(ptCorner[lane_id][0].y, ptCorner[lane_id][1].y), top);
		int ratio = RectInRegion(pParams->MaskLaneImage, pCfgs, imgW, imgH, pCfgs->targets[i].box, lane_id + 1);
		//当目标在视频存在时间太长或长时间没有检测到或离开图像，删除目标
		bool isLost = FALSE;
		/*if(strcmp(pCfgs->targets[i].names, "bus") == 0 ||  strcmp(pCfgs->targets[i].names, "truck") == 0)//公交车或卡车
		{
			isLost = (pCfgs->targets[i].lost_detected > 10);
		}
		else
		{
			isLost = (pCfgs->targets[i].lost_detected > 5);
		}*/
		isLost = (pCfgs->targets[i].lost_detected > 20);
		if(pCfgs->targets[i].continue_num > 5000 || isLost/*pCfgs->targets[i].lost_detected > 5*/ || ratio <= 0)//10
		//if(pCfgs->targets[i].continue_num > 5000 || (pCfgs->targets[i].lost_detected > 5 && pCfgs->targets[i].box.y > ptCorner[lane_id][14].y)||(pCfgs->targets[i].lost_detected > 10 && (pCfgs->targets[i].box.y> ptCorner[lane_id][12].y))|| (pCfgs->targets[i].box.x < 0 || pCfgs->targets[i].box.y < 0 || (pCfgs->targets[i].box.x + pCfgs->targets[i].box.width) > imgW  ||((pCfgs->targets[i].box.y + pCfgs->targets[i].box.height) > imgH)))
		{
			if(pCfgs->currFore_target_id[lane_id] == pCfgs->targets[i].target_id && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//线圈是红色
				obj_out_region(pCfgs, lane_id, &pCfgs->targets[i], 0);//车出
			obj_lost[lane_id] = TRUE;
			DeleteTarget(&pCfgs->targets_size, &i, pCfgs->targets);
			continue;

		}
#ifdef SAVE_VIDEO
		if(pCfgs->NCS_ID == 0)
		{
			cv::rectangle(img, cv::Rect(pCfgs->targets[i].box.x,pCfgs->targets[i].box.y,pCfgs->targets[i].box.width,pCfgs->targets[i].box.height), cv::Scalar(0, 0 ,255), 3, 8, 0 );
			char str[10];
			sprintf(str, "%d", pCfgs->targets[i].target_id);
			putText(img, str, cv::Point(pCfgs->targets[i].box.x,max(0,pCfgs->targets[i].box.y - 10)), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255,0 ), 2);
			char str1[10];
			sprintf(str1, "%d", pCfgs->currFore_target_id[lane_id]);
			putText(img, str1, cv::Point(pCfgs->targets[i].box.x + 50,max(0,pCfgs->targets[i].box.y - 10)), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 255,0 ), 2);
		}
#endif
		pCfgs->targets[i].continue_num++;

	}

	//分析检测目标，并进行区域内的车辆数统计
	for(i = 0;i < pCfgs->detTargets_size; i++)
	{
		//保存轨迹，轨迹数小于3000，直接保存，大于3000，去除旧的
		int trajectory_num = pCfgs->detTargets[i].trajectory_num;
		if(trajectory_num < 3000)
		{

			pCfgs->detTargets[i].trajectory[trajectory_num].x = pCfgs->detTargets[i].box.x + pCfgs->detTargets[i].box.width / 2;
			pCfgs->detTargets[i].trajectory[trajectory_num].y = pCfgs->detTargets[i].box.y + pCfgs->detTargets[i].box.height / 2;
			pCfgs->detTargets[i].trajectory_time[trajectory_num] = pCfgs->currTime;
			pCfgs->detTargets[i].trajectory_num++;
		}
		else
		{
			for(j = 0; j < trajectory_num - 1; j++)
			{
				pCfgs->detTargets[i].trajectory[j] = pCfgs->detTargets[i].trajectory[j + 1];
				pCfgs->detTargets[i].trajectory_time[j] = pCfgs->detTargets[i].trajectory_time[j + 1];
			}
			pCfgs->detTargets[i].trajectory[trajectory_num - 1].x = pCfgs->detTargets[i].box.x + pCfgs->detTargets[i].box.width / 2;
			pCfgs->detTargets[i].trajectory[trajectory_num - 1].y = pCfgs->detTargets[i].box.y + pCfgs->detTargets[i].box.height / 2;
			pCfgs->detTargets[i].trajectory_time[trajectory_num - 1] = pCfgs->currTime;
		}

		//检测到，并更新速度
		if(pCfgs->detTargets[i].detected)
		{
			pCfgs->detTargets[i].lost_detected = 0;
			//get_speed(&pCfgs->detTargets[i]);
		}
		else//未检测到
		{
			pCfgs->detTargets[i].lost_detected++;
			//pCfgs->detTargets[i].box.x += pCfgs->detTargets[i].vx;
			//pCfgs->detTargets[i].box.y += pCfgs->detTargets[i].vy;
		}

		lane_id = pCfgs->detTargets[i].lane_id;
		left = pCfgs->detTargets[i].box.x;
		right = pCfgs->detTargets[i].box.x + pCfgs->detTargets[i].box.width;
		top = pCfgs->detTargets[i].box.y;
		bottom = pCfgs->detTargets[i].box.y + pCfgs->detTargets[i].box.height;
		dis_x = min(max(ptCorner[lane_id][9].x,ptCorner[lane_id][11].x) + 5, right) - max(min(ptCorner[lane_id][8].x,ptCorner[lane_id][10].x) - 5, left);
		dis_y = min(max(ptCorner[lane_id][10].y,ptCorner[lane_id][11].y) + 5, bottom) - max(min(ptCorner[lane_id][8].y,ptCorner[lane_id][9].y) - 5, top);
		//判断此目标是否进入占位线圈
		if(dis_x > 10 && dis_y > 10 && pCfgs->detTargets[i].target_id != pCfgs->currRear_target_id[lane_id] && pCfgs->detTargets[i].cal_flow == FALSE)
		{
			//满足时间间隔条件，进行判断
			if((pCfgs->gThisFrameTime - pCfgs->uRearIntervalNum[lane_id]) > 5 || pCfgs->uRearIntervalNum[lane_id] == 0)
			{
				//线圈已经有车辆存在
				if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag)
				{
					for( k = 0; k < pCfgs->detTargets_size; k++)
					{
						if(pCfgs->detTargets[k].target_id == pCfgs->currRear_target_id[lane_id])
						{
							break;
						}
					}
					//存在车辆
					if(k < pCfgs->detTargets_size)//先出车，再进车
					{
						obj_out_region(pCfgs, lane_id, &pCfgs->detTargets[k], 1);//后线圈出车
					}
					if(k == pCfgs->detTargets_size)
					{
						pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag = 2;
					}
				}
				//printf("lane_id = %d, intervalnum = %d,calarflag = %d,curr_id = %d,%d\n",lane_id,pCfgs->uRearIntervalNum[lane_id],pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag, pCfgs->detTargets[k].target_id,pCfgs->detTargets[i].target_id);
				pCfgs->currRear_target_id[lane_id] = pCfgs->detTargets[i].target_id;//赋予新的target_id
				pCfgs->uRearIntervalNum[lane_id] = pCfgs->gThisFrameTime;

			}
		}

		if(pCfgs->detTargets[i].target_id == pCfgs->currRear_target_id[lane_id] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag == 0 && pCfgs->detTargets[i].cal_flow == FALSE)
		{
			obj_in_region(pCfgs, lane_id, 1);//后线圈车入
			pCfgs->detTargets[i].cal_flow = TRUE;
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].uVehicleType = pCfgs->detTargets[i].class_id + 1;
		}

		//判断与当前车道占位线圈是否相交
		if(dis_x > 0 && dis_y > 0)//与线圈相交
		{
			laneStatus[lane_id][1] = 1;

		}
		else//与线圈不相交
		{
			if(pCfgs->detTargets[i].target_id == pCfgs->currRear_target_id[lane_id] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//线圈是红色
			{
				obj_out_region(pCfgs, lane_id, &pCfgs->detTargets[i], 1);//后线圈出车
			}

		}

		if(pCfgs->detTargets[i].target_id == pCfgs->currRear_target_id[lane_id] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)
		{
			//计算后线圈的流量、车速、车长
			//计算车速
			bool isTrue = FALSE;
			//lane_type = pCfgs->DownDetectCfg.FvdDetectCfg.EachLaneCfg[lane_id].LaneType;//0：竖向 1：横向
			isTrue = (dis_y >= min(pCfgs->detTargets[i].box.height, ptCorner[lane_id][10].y - ptCorner[lane_id][8].y - 2) && lane_type == 0) || (dis_x >= min(pCfgs->detTargets[i].box.width, ptCorner[lane_id][9].x - ptCorner[lane_id][8].x - 2) && lane_type == 1);
			if(isTrue && pCfgs->detTargets[i].cal_speed == FALSE && pCfgs->detTargets[i].continue_num)//当车辆到达流量区域下边缘
			{
			    pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].DetectOutSum++;
				//pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].uVehicleType = pCfgs->detTargets[i].class_id + 1;
				pCfgs->detTargets[i].end_time = pCfgs->currTime;
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].uVehicleSpeed  = CalLaneTargetSpeedY(pCfgs->detTargets[i], lane_id, pCfgs);
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].uVehicleLength = CalTargetLength(pCfgs->detTargets[i], lane_id, pCfgs);
				pCfgs->detTargets[i].cal_speed = TRUE;
			}
		}
		//判断与当前车道是否相交
		int ratio = RectInRegion(pParams->MaskLaneImage, pCfgs, imgW, imgH, pCfgs->detTargets[i].box, lane_id + 1);
		if(ratio > 0)//与当前车道相交
		{
			//计算区域内的车辆数
			vehicle_num[lane_id]++;
			//将跟踪框代替检测框
			//pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1++] = pCfgs->detTargets[i].box;
			//if(pCfgs->detTargets[i].detected)
			{
				CRect box = pCfgs->detTargets[i].box;
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].x = box.x;
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].y = box.y;
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w = box.width;
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h = box.height;
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].label = pCfgs->detTargets[i].class_id + 1;
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].confidence = pCfgs->detTargets[i].prob;
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].id = pCfgs->detTargets[i].target_id;//目标ID
				int pos_x = min(max(0, (box.x + box.width / 2) / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, imgW) - 1);
				int pos_y = min(max(0, (box.y + box.height / 2) / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, imgH) - 1);
				//float actual_pos_x = pCfgs->image_actual[pos_y][pos_x][0];
				//float actual_pos_y = pCfgs->image_actual[pos_y][pos_x][1];
				//actual_pos_x = (actual_pos_x < 0)? (actual_pos_x - 0.5) : (actual_pos_x + 0.5);
				//actual_pos_y = (actual_pos_y < 0)? (actual_pos_y - 0.5) : (actual_pos_y + 0.5);
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].distance[0] = pCfgs->image_actual[pos_y][pos_x][0];//目标与相机的水平距离
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].distance[1] = pCfgs->image_actual[pos_y][pos_x][1];//目标与相机的垂直距离
				//左上
				pos_x = min(max(0, box.x / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, imgW) - 1);
				pos_y = min(max(0, box.y / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, imgH) - 1);
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].border_distance[0][0] = pCfgs->image_actual[pos_y][pos_x][0];
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].border_distance[0][1] = pCfgs->image_actual[pos_y][pos_x][1];
				//右上
				pos_x = min(max(0, (box.x + box.width) / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, imgW) - 1);
				pos_y = min(max(0, box.y / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, imgH) - 1);
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].border_distance[1][0] = pCfgs->image_actual[pos_y][pos_x][0];
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].border_distance[1][1] = pCfgs->image_actual[pos_y][pos_x][1];
				//左下
				pos_x = min(max(0, box.x / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, imgW) - 1);
				pos_y = min(max(0, (box.y + box.height) / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, imgH) - 1);
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].border_distance[2][0] = pCfgs->image_actual[pos_y][pos_x][0];
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].border_distance[2][1] = pCfgs->image_actual[pos_y][pos_x][1];
				//右下
				pos_x = min(max(0, (box.x + box.width) / pCfgs->scale_x), min(MAX_IMAGE_WIDTH, imgW) - 1);
				pos_y = min(max(0, (box.y + box.height) / pCfgs->scale_y), min(MAX_IMAGE_HEIGHT, imgH) - 1);
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].border_distance[3][0] = pCfgs->image_actual[pos_y][pos_x][0];
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].border_distance[3][1] = pCfgs->image_actual[pos_y][pos_x][1];
				int laneid = pCfgs->detTargets[i].lane_id;
				lane_id = (laneid < 0)? 0 :((laneid >= pCfgs->LaneAmount)? (pCfgs->LaneAmount - 1) : laneid);
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].length = CalTargetLength(pCfgs->detTargets[i], laneid, pCfgs);//长度
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].width = CalTargetWidth(pCfgs->detTargets[i], pCfgs);//宽度
				int speed[2] = { 0 };
				CalTargetSpeed(pCfgs->detTargets[i], pCfgs, 1, speed);//计算目标速度
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].speed_Vx = speed[0];//x方向速度
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].speed = speed[1];//y方向速度
			    //pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].speed = CalLaneTargetSpeedY(pCfgs->detTargets[i], laneid, pCfgs);//速度
				//pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].laneid = pCfgs->detTargets[i].lane_id;//车道号
				pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].laneid = 1;//视频检测结果
/*#ifdef SAVE_VIDEO
	            if(pCfgs->NCS_ID == 0)
				{
				    cv::rectangle(img, cv::Rect(pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].x,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].y,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w,pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h), cv::Scalar(0, 0 ,255), 1, 8, 0 );
				}
#endif*/
				/*if(pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w <= 0 || pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h <= 0 || pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w > imgW || pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h > imgH)
				{
					prt(info," box width = %d, height = %d",pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w, pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h);
				}*/
				//prt(info," box width = %d, height = %d",pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].w, pCfgs->ResultMsg.uResultInfo.udetBox[nboxes1].h);
				nboxes1++;
			}
		}
		else if(pCfgs->detTargets[i].continue_num > 5)//不相交删除目标
		{
			if(pCfgs->detTargets[i].target_id == pCfgs->currRear_target_id[lane_id] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//线圈是红色
			{
				obj_out_region(pCfgs, lane_id, &pCfgs->detTargets[i], 1);//后线圈出车
			}
			DeleteTarget(&pCfgs->detTargets_size, &i, pCfgs->detTargets);
			continue;
		}
		//当目标在视频存在时间太长或长时间没有检测到或离开图像，删除目标
		if(pCfgs->detTargets[i].continue_num > 5000 || pCfgs->detTargets[i].lost_detected > 5 ||(pCfgs->detTargets[i].lost_detected > 0 && (left < 10 || top < 10 || right >= (imgW - 10)  || bottom >= (imgH - 10))))
		{
			if(pCfgs->detTargets[i].target_id == pCfgs->currRear_target_id[lane_id] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[lane_id].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//线圈是红色
			{
				obj_out_region(pCfgs, lane_id, &pCfgs->detTargets[i], 1);//后线圈出车
			}
			DeleteTarget(&pCfgs->detTargets_size, &i, pCfgs->detTargets);
			continue;

		}
/*#ifdef SAVE_VIDEO
	    if(pCfgs->NCS_ID == 0)
		{
		    cv::rectangle(img, cv::Rect(pCfgs->detTargets[i].box.x,pCfgs->detTargets[i].box.y,pCfgs->detTargets[i].box.width,pCfgs->detTargets[i].box.height), cv::Scalar(0, 0 ,255), 1, 8, 0 );
		}
#endif*/
		pCfgs->detTargets[i].continue_num++;

	}
	pCfgs->ResultMsg.uResultInfo.udetNum = nboxes1; 

	//防止车辆跳变对区域车辆数进行处理
	for( i = 0; i < laneNum; i++)
	{
		//判断是否与线圈相交
		for(j = 0; j < 2; j++)//变换车道，车辆出车
		{
			if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[j].calarflag == 1 && laneStatus[i][j] == 0 && pCfgs->existFrameNum[i][j] > 3)//没有目标在线圈内
			{
				if(j == 0)//流量线圈
				{
					for(k = 0; k < pCfgs->targets_size; k++)
					{
						if(pCfgs->currFore_target_id[i] == pCfgs->targets[k].target_id)
							obj_out_region(pCfgs, i, &pCfgs->targets[k], 0);
					}
					if(k == pCfgs->targets_size)//没有目标
					{
						if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag == 1)//车出线圈时，calarflag设置为2
							pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag = 2;
					}
				}
				if(j == 1)//后线圈
				{
					for(k = 0; k < pCfgs->detTargets_size; k++)
					{
						if(pCfgs->currRear_target_id[i] == pCfgs->detTargets[k].target_id)
						{
							obj_out_region(pCfgs, i, &pCfgs->detTargets[k], 1);
						}
					}
					if(k == pCfgs->detTargets_size)//没有目标
					{
						if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag == 1)//车出线圈时，calarflag设置为2
						{
							pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag = 2;
						}
					}

				}
				pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[j].calarflag = 2;
				pCfgs->existFrameNum[i][j] = 0;
			}
		}
		//printf("laneID = %d, vehicle num = %d\n",i,vehicle_num1[i]);
		if(pCfgs->gThisFrameTime < 10)//对最初几帧不进行处理
		{
			pCfgs->uDetectVehicleSum[i] = vehicle_num[i];
		}
		else
		{
			int curr_num = vehicle_num[i];
			//printf("lane id = %d,curr num = %d,old num =%d\n",i,curr_num,pCfgs->uDetectVehicleSum[i]);
			if(pCfgs->uDetectVehicleSum[i] != curr_num)
			{
				if(pCfgs->uDetectVehicleFrameNum[i] > 10)//最短10帧才发生车辆数的变化
				{
					curr_num = (pCfgs->uDetectVehicleSum[i] > curr_num)? (pCfgs->uDetectVehicleSum[i] - 1) : (pCfgs->uDetectVehicleSum[i] + 1);
				}
				else if(pCfgs->uDetectVehicleSum[i] >  curr_num && obj_lost[i])//有车出
				{
					curr_num = pCfgs->uDetectVehicleSum[i] - 1;
				}
				else
				{
					curr_num = pCfgs->uDetectVehicleSum[i];
				}
				if(vehicle_num[i] == 0 && curr_num != 0)//当前帧没有区域车辆数，快速将车辆数降下
				{
					if(pCfgs->uDetectVehicleFrameNum[i] > 5)
						curr_num = pCfgs->uDetectVehicleSum[i] - 1;
				}
			}
			//printf("lane id1 = %d,curr num = %d,old num =%d interval_num =%d\n",i,curr_num,pCfgs->uDetectVehicleSum[i],pCfgs->uDetectVehicleFrameNum[i]);
			if(pCfgs->uDetectVehicleSum[i] != curr_num)//前后两帧的车辆数不相同时，重新计数
				pCfgs->uDetectVehicleFrameNum[i] = 0;
			if(pCfgs->uDetectVehicleSum[i] == curr_num)//前后两帧车辆数相同
				pCfgs->uDetectVehicleFrameNum[i]++;
			pCfgs->uDetectVehicleSum[i] = curr_num;//将处理后的数量赋给当前车辆数
		}
		if(pCfgs->uDetectVehicleSum[i] == 0 || vehicle_num1[i] == 0)//车道区域内无车
		{
			pCfgs->Tailposition[i] = min(ptCorner[i][2].y, ptCorner[i][3].y);
			pCfgs->Headposition[i] = min(ptCorner[i][2].y, ptCorner[i][3].y);
		}
	}

	/*for( i = 0; i < laneNum; i++)
	{
		for( j = 1;j >= 0; j--)
		{
			pCfgs->uStatVehicleSum[i][j + 1] = pCfgs->uStatVehicleSum[i][j];
		}
		pCfgs->uStatVehicleSum[i][0] = vehicle_num[i];
		if(pCfgs->uStatVehicleSum[i][3] < 3)
			pCfgs->uStatVehicleSum[i][3] = pCfgs->uStatVehicleSum[i][3] + 1;

		sum = 0.0;
		for(j = 0; j < pCfgs->uStatVehicleSum[i][3];j++)
			sum +=  pCfgs->uStatVehicleSum[i][j];
		sum = sum / pCfgs->uStatVehicleSum[i][3];
		//防止区域车辆数跳变
		if(pCfgs->uDetectVehicleSum[i] > vehicle_num[i] && obj_lost[i])//有车出，数量减1
		{
			pCfgs->uDetectVehicleSum[i] = pCfgs->uDetectVehicleSum[i] - 1;
		} 
		else if(vehicle_num[i] > pCfgs->uDetectVehicleSum[i])
		{
			//pCfgs->uDetectVehicleSum[i] = vehicle_num[i];
			pCfgs->uDetectVehicleSum[i] = sum + 0.5;
		} 
		else if(pCfgs->uDetectVehicleSum[i] > vehicle_num[i] && pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag == 0)//流量区域无车，采用实际检测数
		{
			//pCfgs->uDetectVehicleSum[i] = vehicle_num[i];//实际车辆数
			//防止车辆数跳变
			pCfgs->uDetectVehicleSum[i] = sum + 0.5;
		}
		if(vehicle_num[i] == 0)//车道区域内无车
		{
		pCfgs->Tailposition[i] = min(ptCorner[i][2].y, ptCorner[i][3].y);
		pCfgs->Headposition[i] = min(ptCorner[i][2].y, ptCorner[i][3].y);
		}
#ifdef SAVE_VIDEO
		if(pCfgs->NCS_ID == 0)
		{
		    char str[10];
		    sprintf(str, "%d", pCfgs->uDetectVehicleSum[i]);
		    putText(img, str, Point(10 + 30 * i,10), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2);
		    char str1[10];
		    sprintf(str1, "%d", pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].car_out);
		    putText(img, str1, Point(10 + 30 * i,30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2);
		}
#endif
	}*/
	//printf("get_target end...................................\n");
}
Uint16 ArithProc(Uint16 ChNum, unsigned char *pInFrameBuf, unsigned char *pInuBuf, unsigned char *pInvBuf, int FrameWidth, int FrameHeight,\
				 RESULTMSG* outBuf, Int32 outSize, ALGCFGS *pCfgs, ALGPARAMS *pParams, mRadarRTObj* objRadar, int objRadarNum)
{
	Int32 i, j, k;
	float temp = 0;

	CPoint m_ptend[16];
	CPoint LineUp[2];
	CPoint LineUp1[2];
	int x1 = 0, x2 = 0, x3 = 0, x4 = 0;
	int thr = 10;
	int result[6 * MAX_DETECTION_NUM] = { 0 };
	int nboxes = 0;

	//pthread_self();
	//printf("(pCfgs %x,%x\n",pCfgs,pthread_self());fflush(NULL);
	//printf("(pCfgs %x\n",pCfgs->CameraCfg);
	//将图像限制在[0, FULL_COLS),[0, FULL_ROWS)
	if(FrameWidth <= 0 || FrameHeight <= 0)//没有图像数据
	{
		printf("img cannot be zero!\n");
		return 0;
	}
	//处理数据大小
	pCfgs->m_iHeight = (FrameHeight > FULL_ROWS)? FULL_ROWS : FrameHeight;
	pCfgs->m_iWidth = (FrameWidth > FULL_COLS)? FULL_COLS : FrameWidth;
	pCfgs->img_height = FrameHeight;//图像的宽高
	pCfgs->img_width = FrameWidth;
	//printf("process,%d,%d,%d,%d\n",FrameHeight,FrameWidth,pCfgs->m_iHeight,pCfgs->m_iWidth);
	if(pCfgs->gThisFrameTime % 199 == 1)
	{
		//能见度计算
		thr = 8;
		pCfgs->up_visib_value++;
		pCfgs->fuzzydegree = fuzzy(pInFrameBuf, FrameWidth, FrameHeight);
		for (j = VISIB_LENGTH - 1; j > 0; j--)
		{
			pCfgs->visib_value[j] = pCfgs->visib_value[j - 1];
		}
		pCfgs->visib_value[0] = (int)(pCfgs->fuzzydegree);
		if (pCfgs->up_visib_value > VISIB_LENGTH)
		{
			pCfgs->visibility = visible_judge(pCfgs->visib_value, VISIB_LENGTH, thr);
		} 
		else
		{
			pCfgs->visibility = FALSE;
		}
		printf("fuzzy degree = %d\n",(int)(pCfgs->fuzzydegree));
		//视频异常计算
		if(Color_deviate(pInuBuf, pInvBuf, FrameWidth / 4, FrameHeight / 4))
			pCfgs->abnormal_time++;
		else
			pCfgs->abnormal_time = 0;	   
		pCfgs->fuzzyflag = (pCfgs->abnormal_time > 5)? TRUE : FALSE;
	}

	//设置车道掩模图像
	if(pCfgs->bMaskLaneImage == FALSE)
	{
		MaskLaneImage(pCfgs, pParams, FrameWidth, FrameHeight);
		pCfgs->bMaskLaneImage = TRUE;
	}
	//设置行人检测区域掩模图像
	if(pCfgs->bMaskDetectImage == FALSE)
	{
		MaskDetectImage(pCfgs, pParams, FrameWidth, FrameHeight);
		pCfgs->bMaskDetectImage = TRUE;
	}
	//标定图像
	if(pCfgs->bCalibrationImage == FALSE)
	{
		get_calibration_data(pCfgs, FrameWidth, FrameHeight);
		pCfgs->bCalibrationImage = TRUE;
	}
//#define  DETECT_VISIBILITY
#ifdef DETECT_VISIBILITY
	float vis_actual= 0;
	if(pCfgs->gThisFrameTime % 100 == 0)//能见度检测
	{
		if(pCfgs->bAuto == 1)
		{
			int vis = DayVisibilityDetection(pInFrameBuf, pCfgs->calibration_point, FrameWidth, FrameHeight);
			vis_actual = pCfgs->image_actual[vis][FrameWidth / 2][1];
		}
		else 
		{
			float l1 = pCfgs->image_actual[FrameHeight - 300][FrameWidth / 2][1];
			float l2 = pCfgs->image_actual[FrameHeight - 100][FrameWidth / 2][1];
			vis_actual = NightVisibilityDetection(pInFrameBuf, pCfgs->calibration_point, FrameWidth, FrameHeight, l1, l2);
		}
	}
#endif
	gettimeofday(&pCfgs->time_end, NULL);
	if(pCfgs->gThisFrameTime == 0)
		pCfgs->currTime = 0;
	else
	pCfgs->currTime += (pCfgs->time_end.tv_sec - pCfgs->time_start.tv_sec) + (pCfgs->time_end.tv_usec - pCfgs->time_start.tv_usec)/1000000.0;
	gettimeofday(&pCfgs->time_start, NULL);

	//记录150帧的帧时间，程序开始时间为0
	for(i = 1; i < 150; i++)
	{
		pCfgs->uStatFrameTime[i - 1] = pCfgs->uStatFrameTime[i];
	}
	pCfgs->uStatFrameTime[149] = pCfgs->currTime;
	pCfgs->gThisFrameTime++;
	////////////////////////////////////////////////
	//将yuv转化成bgr
	Mat YUVImage, BGRImage;
	int size = FrameWidth * FrameHeight;
	//yuv420 to bgr
	YUVImage.create(FrameHeight * 3 / 2, FrameWidth, CV_8UC1);
	memcpy(YUVImage.data, pInFrameBuf, size);
	memcpy(YUVImage.data + size, pInuBuf, size / 4);
	memcpy(YUVImage.data + size + size / 4, pInvBuf, size / 4);
	cvtColor(YUVImage, BGRImage, CV_YUV2BGR_I420);
	YUVImage.release();
	if(pCfgs->NCS_ID == 0)
	{

#ifdef SAVE_VIDEO
		BGRImage.copyTo(img);
#endif // SAVE_VIDEO
		for(i = 0; i < objRadarNum; i++)
		{
#ifdef SAVE_VIDEO
			char str[200];
			sprintf(str, "%f, %f, %f, %f, %f", objRadar[i].x_Point,objRadar[i].y_Point,objRadar[i].Speed_x,objRadar[i].Speed_y,objRadar[i].Obj_Len);
			putText(img, str, Point(10, pCfgs->img_height - 30 - 30 * i), cv::FONT_HERSHEY_DUPLEX, 1, cv::Scalar(0, 0,255), 2);
#endif
		}
	}
#ifdef DETECT_GPU  
	/*IplImage img = IplImage(BGRImage);
	//printf("image size = %d,%d,[%d,%d]\n",img.width,img.height,pCfgs->net_params->net->h,pCfgs->net_params->net->w);
	if(pCfgs->net_params->net)
	{
		nboxes = YoloArithDetect(&img, pCfgs->net_params, result);//yolo检测
	}*/
	IplImage* image = cvCreateImage(cvSize(FrameWidth, FrameHeight), IPL_DEPTH_8U, 3);
	memcpy(image->imageData, BGRImage.data, FrameHeight * FrameWidth * 3);
	//printf("image size = %d,%d,[%d,%d]\n",image.width,image.height,pCfgs->net_params->net->h,pCfgs->net_params->net->w);
	if(pCfgs->net_params->net)
	{
		nboxes = YoloArithDetect(image, pCfgs->net_params, result);//yolo检测
	}
	cvReleaseImage(&image);
#else
	nboxes = NCSArithDetect(BGRImage, pCfgs, result);
#endif
	//////////////////////////////////////////////////////
	memset((void *)&pCfgs->ResultMsg, 0, sizeof(pCfgs->ResultMsg));
	memcpy((void *)&pCfgs->ResultMsg, (void *)outBuf, outSize);
	//分析检测结果
	ProcessDetectBox(pCfgs, result, nboxes);
	//分析行人检测框
	ProcessPersonBox(pCfgs, pParams, FrameWidth, FrameHeight);
	//分析车辆检测框，得到车道检测参数
	get_lane_params(pCfgs, pParams, pCfgs->LaneAmount, FrameWidth, FrameHeight);

	//交通事件检测
	EventDetectProc(pCfgs, pParams, FrameWidth, FrameHeight);

#ifdef SAVE_VIDEO
	if(pCfgs->NCS_ID == 0)
	{
		for( i = 0; i < pCfgs->uNoPersonAllowNum; i++)
		{
			cv::line(img, cv::Point(pCfgs->NoPersonAllowBox[i].EventBox[0].x, pCfgs->NoPersonAllowBox[i].EventBox[0].y), cv::Point(pCfgs->NoPersonAllowBox[i].EventBox[1].x, pCfgs->NoPersonAllowBox[i].EventBox[1].y), cv::Scalar(0, 255 ,0), 2, 8, 0);
			cv::line(img, cv::Point(pCfgs->NoPersonAllowBox[i].EventBox[1].x, pCfgs->NoPersonAllowBox[i].EventBox[1].y), cv::Point(pCfgs->NoPersonAllowBox[i].EventBox[2].x, pCfgs->NoPersonAllowBox[i].EventBox[2].y), cv::Scalar(0, 255 ,0), 2, 8, 0);
			cv::line(img, cv::Point(pCfgs->NoPersonAllowBox[i].EventBox[2].x, pCfgs->NoPersonAllowBox[i].EventBox[2].y), cv::Point(pCfgs->NoPersonAllowBox[i].EventBox[3].x, pCfgs->NoPersonAllowBox[i].EventBox[3].y), cv::Scalar(0, 255 ,0), 2, 8, 0);
			cv::line(img, cv::Point(pCfgs->NoPersonAllowBox[i].EventBox[3].x, pCfgs->NoPersonAllowBox[i].EventBox[3].y), cv::Point(pCfgs->NoPersonAllowBox[i].EventBox[0].x, pCfgs->NoPersonAllowBox[i].EventBox[0].y), cv::Scalar(0, 255 ,0), 2, 8, 0);
			putText(img, "pe", Point(pCfgs->NoPersonAllowBox[i].EventBox[0].x, pCfgs->NoPersonAllowBox[i].EventBox[0].y - 30), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 255,0 ), 2);
		}
		for( i = 0; i < pCfgs->uNonMotorAllowNum; i++)
		{
			cv::line(img, cv::Point(pCfgs->NonMotorAllowBox[i].EventBox[0].x, pCfgs->NonMotorAllowBox[i].EventBox[0].y), cv::Point(pCfgs->NonMotorAllowBox[i].EventBox[1].x, pCfgs->NonMotorAllowBox[i].EventBox[1].y), cv::Scalar(255, 255 ,0), 2, 8, 0);
			cv::line(img, cv::Point(pCfgs->NonMotorAllowBox[i].EventBox[1].x, pCfgs->NonMotorAllowBox[i].EventBox[1].y), cv::Point(pCfgs->NonMotorAllowBox[i].EventBox[2].x, pCfgs->NonMotorAllowBox[i].EventBox[2].y), cv::Scalar(255, 255 ,0), 2, 8, 0);
			cv::line(img, cv::Point(pCfgs->NonMotorAllowBox[i].EventBox[2].x, pCfgs->NonMotorAllowBox[i].EventBox[2].y), cv::Point(pCfgs->NonMotorAllowBox[i].EventBox[3].x, pCfgs->NonMotorAllowBox[i].EventBox[3].y), cv::Scalar(255, 255 ,0), 2, 8, 0);
			cv::line(img, cv::Point(pCfgs->NonMotorAllowBox[i].EventBox[3].x, pCfgs->NonMotorAllowBox[i].EventBox[3].y), cv::Point(pCfgs->NonMotorAllowBox[i].EventBox[0].x, pCfgs->NonMotorAllowBox[i].EventBox[0].y), cv::Scalar(255, 255 ,0), 2, 8, 0);
			putText(img, "motor", Point(pCfgs->NonMotorAllowBox[i].EventBox[0].x, pCfgs->NonMotorAllowBox[i].EventBox[0].y - 30), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 255,0 ), 2);
		}
		for(i = 0; i < pCfgs->event_targets_size; i++)
		{
			cv::rectangle(img, cv::Rect(pCfgs->event_targets[i].box.x,pCfgs->event_targets[i].box.y,pCfgs->event_targets[i].box.width,pCfgs->event_targets[i].box.height), cv::Scalar(0, 0 ,255), 3, 8, 0 );
			char str1[10];
			sprintf(str1, "%d%s", pCfgs->event_targets[i].target_id, pCfgs->event_targets[i].names);

			putText(img, str1, Point(pCfgs->event_targets[i].box.x, pCfgs->event_targets[i].box.y - 20), cv::FONT_HERSHEY_DUPLEX, 1.2, cv::Scalar(0, 255,0 ), 2);
		}
	}
#endif

#ifdef DETECT_PERSON_ATTRIBUTE
	HumanAttributeDetect(pCfgs, &img);//进行行人属性识别
	//BicycleAttributeDetect(pCfgs, img);//进行单车属性识别
#endif

#ifdef DETECT_PLATE//车牌识别
	/*PlateInfo plateInfo[MAX_PLATE_NUM];
	if(pCfgs->plate_flag >= 0)
	{
		int plateNum = PlateDetectandRecognize(BGRImage, 0, plateInfo, pCfgs->plate_flag);
		printf("plate num = %d\n",plateNum);
	}*/
	//给出车牌区域进行识别
	if(pCfgs->plate_flag >= 0 && BGRImage.cols <= DETECT_IMAGE_WIDTH && BGRImage.rows <= DETECT_IMAGE_HEIGHT)//加载车牌识别网络成功
	{
		ProcessPlateBox(pCfgs, pParams, FrameWidth, FrameHeight);
	}
#endif
	BGRImage.release();

	pCfgs->uCongestionNum = 0;//拥堵数初始化
	pCfgs->ResultMsg.uResultInfo.LaneSum = pCfgs->LaneAmount;//车道数

	//缩放图像
	Mat grayImage, resizeImage;
	grayImage.create(FrameHeight, FrameWidth, CV_8UC1);
	memcpy(grayImage.data, pInFrameBuf, FrameHeight * FrameWidth);
	if(pCfgs->m_iWidth != FrameWidth || pCfgs->m_iHeight != FrameHeight)
		resize(grayImage, resizeImage, Size(pCfgs->m_iWidth, pCfgs->m_iHeight), 0, 0, INTER_LINEAR);//缩放
	else
		grayImage.copyTo(resizeImage);//复制
	grayImage.release();
	memcpy((void *)pParams->CurrQueueImage, (void *)resizeImage.data, pCfgs->m_iWidth * pCfgs->m_iHeight);
	memset(pCfgs->CongestionBox, 0, MAX_LANE * sizeof(EVENTBOX));//初始化拥堵框

	//计算车道内的排队长度
	iSubStractImage(pParams->CurrQueueImage, pParams->PrePrePreQueueImage, pParams->MaskLaneImage, 15, 0, pCfgs->m_iWidth, pCfgs->m_iHeight);//隔三帧帧差
	for( i = 0; i < pCfgs->LaneAmount; i++ )
	{

		m_ptend[0]= pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[0]; 
		m_ptend[1]= pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[1];
		m_ptend[2]= pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[2];  
		m_ptend[3]= pCfgs->DownDetectCfg.SpeedEachLane[i].LaneRegion[3];//车道区域
		m_ptend[4]= pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[0];  
		m_ptend[5]= pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[1];
		m_ptend[6]= pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[2];  
		m_ptend[7]= pCfgs->DownDetectCfg.SpeedEachLane[i].FrontCoil[3]; //占位线圈		
		m_ptend[8]= pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[0];  
		m_ptend[9]= pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[1];
		m_ptend[10]= pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[2];  
		m_ptend[11]= pCfgs->DownDetectCfg.SpeedEachLane[i].MiddleCoil[3];//占有线圈
		m_ptend[12]= pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[0];  
		m_ptend[13]= pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[1];
		m_ptend[14]= pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[2];  
		m_ptend[15]= pCfgs->DownDetectCfg.SpeedEachLane[i].RearCoil[3];//流量线圈

		QueLengthCaculate( i, pCfgs, pParams, m_ptend, FrameWidth, FrameHeight, FALSE);
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.getQueback_flag = 1;
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.IsCarInTailFlag = pCfgs->IsCarInTail[i];
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uDetectRegionVehiSum = pCfgs->uDetectVehicleSum[i];


		LineUp[0].x = LineUp[0].y = LineUp[1].x = LineUp[1].y = 0;
		LineUp1[0].x = LineUp1[0].y =LineUp1[1].x = LineUp1[1].y = 0;
		//末车位置

		if(m_ptend[0].y != m_ptend[2].y && m_ptend[1].y != m_ptend[3].y)
		{
			LineUp[0].y = pCfgs->Headposition[i];
			LineUp[1].y = pCfgs->Headposition[i];
			if(m_ptend[0].x == m_ptend[3].x)//垂直车道线
			{
				LineUp[0].x = m_ptend[0].x;
			}
			else
			{
				LineUp[0].x = (LineUp[0].y - m_ptend[0].y) * (m_ptend[3].x - m_ptend[0].x) / (m_ptend[3].y - m_ptend[0].y) + m_ptend[0].x;
			}
			if(m_ptend[1].x == m_ptend[2].x)//垂直车道线
			{
				LineUp[1].x = m_ptend[1].x;
			}	
			else
			{
				LineUp[1].x = (LineUp[1].y - m_ptend[1].y) * (m_ptend[2].x - m_ptend[1].x) / (m_ptend[2].y - m_ptend[1].y) + m_ptend[1].x;
			}
		}
		else
		{
			printf("detect point err\n");
		}
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0] = LineUp[0];
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1] = LineUp[1];

		//头车位置、末车位置、 头车速度、末车速度
		if(pCfgs->uDetectVehicleSum[i] == 0)
		{
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uHeadVehiclePos = 0;//头车位置
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uLastVehiclePos = 0;//末车位置
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uHeadVehicleSpeed = 0;//头车速度
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uLastVehicleSpeed = 0;//末车速度
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleDensity = 0;//车辆密度
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uLastVehicleLength = 0;//最后一辆车位置
		}
		else
		{	
			int pos1 = pCfgs->Tailposition[i];
			int pos2 = pCfgs->Headposition[i];
			pos1 = max(0, int(pos1 / pCfgs->scale_y));
			pos1 = min(pos1, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
			pos2 = max(0, int(pos2 / pCfgs->scale_y));
			pos2 = min(pos2, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uHeadVehiclePos = pCfgs->actual_distance[i][pos1];//头车位置
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uLastVehiclePos =pCfgs->actual_distance[i][pos2];//末车位置
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uHeadVehicleSpeed = pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].uVehicleSpeed;//头车速度
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uLastVehicleSpeed = pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].uVehicleSpeed;//末车速度
			//最后一辆车的位置
			temp = pCfgs->actual_distance[i][pos2];
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uLastVehicleLength = temp + pCfgs->cam2stop;//加上相机到停止线的距离

		}

		//排队长度
		if(m_ptend[0].y != m_ptend[3].y && m_ptend[1].y != m_ptend[2].y)
		{
			LineUp1[0].y = min(m_ptend[2].y, m_ptend[3].y);
			//LineUp1[0].x = (m_ptend[2].x + m_ptend[3].x) / 2 ;
			//LineUp1[0].y = ( m_ptend[2].y + m_ptend[3].y) / 2 ;

			LineUp1[1].y = pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength;
			if(m_ptend[0].x == m_ptend[3].x)//垂直车道线
			{
				x1 = m_ptend[0].x;
				x3 = m_ptend[0].x;
			}
			else
			{
				x1 = (LineUp1[1].y - m_ptend[0].y) * (m_ptend[3].x - m_ptend[0].x) / (m_ptend[3].y - m_ptend[0].y) + m_ptend[0].x;
				x3 = (LineUp1[0].y - m_ptend[0].y) * (m_ptend[3].x - m_ptend[0].x) / (m_ptend[3].y - m_ptend[0].y) + m_ptend[0].x;
			}
			if(m_ptend[1].x == m_ptend[2].x)//垂直车道线
			{
				x2 = m_ptend[1].x;
				x4 = m_ptend[1].x;
			}	
			else
			{
				x2 = (LineUp1[1].y - m_ptend[1].y) * (m_ptend[2].x - m_ptend[1].x) / (m_ptend[2].y - m_ptend[1].y) + m_ptend[1].x;
				x4 = (LineUp1[0].y - m_ptend[1].y) * (m_ptend[2].x - m_ptend[1].x) / (m_ptend[2].y - m_ptend[1].y) + m_ptend[1].x;
			}
			LineUp1[1].x = (x1 + x2) / 2;
			LineUp1[0].x = (x3 + x4) / 2;
		}
		else
		{
			printf("detect point err\n");
		} 
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.QueLine[0] = LineUp1[0];//队首
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.QueLine[1] = LineUp1[1];//队尾
		if(abs(LineUp1[0].y - LineUp1[1].y) < 5)//没有排队	
		{	
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uQueueHeadDis = 0;//队首距离
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uQueueTailDis = 0;//队尾距离
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength = 0;//排队长度

		}
		else
		{
			int pos = LineUp1[0].y;
			pos = max(0, int(pos / pCfgs->scale_y));
			pos = min(pos, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
			temp = pCfgs->actual_distance[i][pos];	
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uQueueHeadDis = temp;//队首距离
			pos = LineUp1[1].y;
			pos = max(0, int(pos / pCfgs->scale_y));
			pos = min(pos, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
			temp = pCfgs->actual_distance[i][pos];	
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uQueueTailDis = temp;//队尾距离
			pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength = pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uQueueTailDis;//从停车线开始
			//pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uVehicleQueueLength = abs(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uQueueTailDis - pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.uQueueHeadDis);//队首减队尾  排队长度
		}
#ifdef SAVE_VIDEO
		if(pCfgs->NCS_ID == 0)
		{
			//cv::line(img, cv::Point(pCfgs->detLineParm[0].pt[0].x,pCfgs->detLineParm[0].pt[0].y),cv::Point(pCfgs->detLineParm[0].pt[1].x,pCfgs->detLineParm[0].pt[1].y), cv::Scalar(255, 0 ,0), 1, 8, 0 );
			cv::line(img, cv::Point(m_ptend[0].x,m_ptend[0].y),cv::Point(m_ptend[3].x,m_ptend[3].y), cv::Scalar(255, 0 ,0), 3, 8, 0 );
			cv::line(img, cv::Point(m_ptend[1].x,m_ptend[1].y),cv::Point(m_ptend[2].x,m_ptend[2].y), cv::Scalar(255, 0 ,0), 3, 8, 0 );
			if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].calarflag)
			{
				cv::line(img, cv::Point(m_ptend[12].x,m_ptend[12].y),cv::Point(m_ptend[13].x,m_ptend[13].y), cv::Scalar(0, 0 ,255), 3, 8, 0 );
				cv::line(img, cv::Point(m_ptend[14].x,m_ptend[14].y),cv::Point(m_ptend[15].x,m_ptend[15].y), cv::Scalar(0, 0 ,255), 3, 8, 0 );
			}
			else
			{
				cv::line(img, cv::Point(m_ptend[12].x,m_ptend[12].y),cv::Point(m_ptend[13].x,m_ptend[13].y), cv::Scalar(255, 0 ,0), 3, 8, 0 );
				cv::line(img, cv::Point(m_ptend[14].x,m_ptend[14].y),cv::Point(m_ptend[15].x,m_ptend[15].y), cv::Scalar(255, 0 ,0), 3, 8, 0 );
			}
			if(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].calarflag)
			{
				cv::line(img, cv::Point(m_ptend[8].x,m_ptend[8].y),cv::Point(m_ptend[9].x,m_ptend[9].y), cv::Scalar(0, 0 ,255), 3, 8, 0 );
				cv::line(img, cv::Point(m_ptend[10].x,m_ptend[10].y),cv::Point(m_ptend[11].x,m_ptend[11].y), cv::Scalar(0, 0 ,255), 3, 8, 0 );
			}
			else
			{
				cv::line(img, cv::Point(m_ptend[8].x,m_ptend[8].y),cv::Point(m_ptend[9].x,m_ptend[9].y), cv::Scalar(255, 0 ,0), 1, 8, 0 );
				cv::line(img, cv::Point(m_ptend[10].x,m_ptend[10].y),cv::Point(m_ptend[11].x,m_ptend[11].y), cv::Scalar(255, 0 ,0), 1, 8, 0 );
			}

			cv::line(img, cv::Point(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.QueLine[0].x,pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.QueLine[0].y),cv::Point(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.QueLine[1].x,pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.QueLine[1].y), cv::Scalar(0, 255 ,0), 3, 8, 0 );
			//cv::line(img, cv::Point(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].x,pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[0].y),cv::Point(pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].x,pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.LineUp[1].y), cv::Scalar(255, 0 ,0), 3, 8, 0 );
			if(pCfgs->IsCarInTail[i])
			{
				cv::line(img, cv::Point(m_ptend[4].x,m_ptend[4].y),cv::Point(m_ptend[5].x,m_ptend[5].y), cv::Scalar(0, 0 ,255), 1, 8, 0);
				cv::line(img, cv::Point(m_ptend[6].x,m_ptend[6].y),cv::Point(m_ptend[7].x,m_ptend[7].y), cv::Scalar(0, 0 ,255), 1, 8, 0);
			}
			else
			{
				cv::line(img, cv::Point(m_ptend[4].x,m_ptend[4].y),cv::Point(m_ptend[5].x,m_ptend[5].y), cv::Scalar(255, 0 ,0), 1, 8, 0);
				cv::line(img, cv::Point(m_ptend[6].x,m_ptend[6].y),cv::Point(m_ptend[7].x,m_ptend[7].y), cv::Scalar(255, 0 ,0), 1, 8, 0);
			}
			char str[10];
			sprintf(str, "%d", pCfgs->gThisFrameTime);
			putText(img, str, cv::Point(320,30), cv::FONT_HERSHEY_DUPLEX, 0.5, cv::Scalar(0, 255,0 ), 2);
			char str1[10];
			sprintf(str1, "%d", pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].DetectInSum);
			putText(img, str1, cv::Point(10 + 50 * i, 50), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255,0 ), 2);
			char str2[10];
			sprintf(str2, "%d", pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[0].DetectOutSum);
			putText(img, str2, cv::Point(10 + 50 * i, 150), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255,0 ), 2);
			char str3[10];
			sprintf(str3, "%d", pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].DetectInSum);
			putText(img, str3, cv::Point(10 + 50 * i, 250), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255,0 ), 2);
			char str4[10];
			sprintf(str4, "%d", pCfgs->ResultMsg.uResultInfo.uEachLaneData[i].SpeedDetectInfo1.CoilAttribute[1].DetectOutSum);
			putText(img, str4, cv::Point(10 + 50 * i, 350), cv::FONT_HERSHEY_DUPLEX, 1.5, cv::Scalar(0, 255,0 ), 2);
		}

#endif
	}
	memcpy((void *)pParams->PrePrePreQueueImage, (void *)pParams->PrePreQueueImage, pCfgs->m_iWidth * pCfgs->m_iHeight);
	memcpy((void *)pParams->PrePreQueueImage, (void *)pParams->PreQueueImage, pCfgs->m_iWidth * pCfgs->m_iHeight);
	memcpy((void *)pParams->PreQueueImage, (void *)pParams->CurrQueueImage, pCfgs->m_iWidth * pCfgs->m_iHeight);
#ifdef SAVE_VIDEO
	if(pCfgs->NCS_ID == 0)
	{
		//writer << img;
		if(pCfgs->gThisFrameTime <= SAVE_FRAMES)
		{
			cv::Mat img_resize;
			if(img.cols != SAVE_VIDEO_WIDTH || img.rows != SAVE_VIDEO_HEIGHT)
				resize(img, img_resize, Size(SAVE_VIDEO_WIDTH, SAVE_VIDEO_HEIGHT), 0, 0, INTER_LINEAR);//缩放
			else
				img.copyTo(img_resize);//复制
			writer << img_resize;
		}
		if(pCfgs->gThisFrameTime > SAVE_FRAMES)
			writer.release();
		//imshow("img",img);
		//waitKey(1);
	}
#endif 
	 BGRImage.release();
	pCfgs->ResultMsg.uResultInfo.uEnvironmentStatus = pCfgs->bAuto; //added by david 20131014
	memcpy((void *)outBuf, (void *)&pCfgs->ResultMsg, outSize);
	return 1;
}
//按照位置顺序对检测框进行排序
void sort_obj(int obj_pos[][2], int obj_num)
{
	int temp[2];
	int i = 0, j = 0;
	for(i = 0; i < obj_num - 1; i++)

	{
		for(j = i + 1; j < obj_num; j++)

        {
			if(obj_pos[i][0] < obj_pos[j][0])
			{
				temp[0] = obj_pos[i][0];
				temp[1] = obj_pos[i][1];
				obj_pos[i][0] = obj_pos[j][0];
				obj_pos[i][1] = obj_pos[j][1];
				obj_pos[j][0] = temp[0];
				obj_pos[j][1] = temp[1];
			}
		}
	}

}
//是否排队
bool IsVehicleQueue(Uint8* puSubImage, Uint8* puMaskImage, Uint16 laneID, Uint16 tail_line, Uint16 head_line, ALGCFGS *pCfgs)
{
	int nRow, nCol;
	int subnum = 0, num = 0;
	float ratio = 0.0;
	Uint32 offset = 0;
	unsigned char* subPtr;
	unsigned char* maskPtr;
	//计算运动点数
	for( nRow =tail_line ; nRow < head_line; nRow += 2)
	{
		offset = nRow * pCfgs->m_iWidth;
		subPtr = puSubImage + offset;
		maskPtr = puMaskImage + offset;
		for (nCol = 0 ; nCol < pCfgs->m_iWidth; nCol += 2)
		{
			if(maskPtr[nCol] == (laneID + 1))
			{
				if (subPtr[nCol])
				{
					subnum++;
				}
				num++;
			}
		}
	}
	ratio = (num > 0)? (float)subnum / (float)num : 0;
	if(ratio < 0.15)//0.15  小于阈值，则认为此区域不运动
	{
		return true;
	} 
	else
	{
		return false;
	}
}
Uint16 sort_median(Uint16* arr, int num)//取中间值
{
	int i = 0, j = 0;
	Uint16 temp;
	Uint16 array_temp[10] = {0};
	for(i = 0; i < num; i++)
	{
		array_temp[i] = arr[i];
	}
	for(i = 0; i <= num / 2; i++)
	{
		for(j = i + 1; j < num; j++)
		{
			if(array_temp[i] > array_temp[j])
			{
				temp = array_temp[i];
				array_temp[i] = array_temp[j];
				array_temp[j] = temp;
			}
		}
	}
	return array_temp[num / 2];
}
//计算车道是否拥堵
void CongestionDetect(Uint16 LaneID, ALGCFGS *pCfgs, int obj_pos[][2], int obj_num, CPoint m_ptend[], float* obj_interval, bool* IsVehStatic)
{
	int i = 0;
	int lane_top = MIN(m_ptend[0].y, m_ptend[1].y);
	int lane_bottom = MIN(m_ptend[2].y, m_ptend[3].y);
	int start_congestion = lane_bottom;
	int end_congestion = lane_bottom;
	bool bCongestion = FALSE, bDriveSlow =	FALSE;
	int cal_congestion = 0;//开始计算拥堵
	float sum = 0.0;
	int num = 0;
	int tolerate_num = 0;//允许车辆区域间隔大的数目
	int vehicle_static_num = 0;//静止车辆数量
	int x1 = 0, x2 = 0, x3 = 0, x4 = 0;
	int congestion_num = 0, totalnum = 0;
	//判断拥堵情况
	/*for( i = obj_num - 1; i >= 0; i--)//从上向下判断
	{
		//根据目标的距离判断拥堵情况
		if(obj_interval[i] < 0.3 && cal_congestion == 0)
		{
			start_congestion = obj_pos[i][1];
			cal_congestion = 1;
			sum = 0;
			num = 0;
			tolerate_num = 0;
		}
		else if(obj_interval[i] > 1 && cal_congestion)
		{
			tolerate_num++;
		}
		if(obj_interval[i] > 2)
		{
			break;
		}
		if(cal_congestion)
		{
			sum += (obj_interval[i] < 0) ? 0 : obj_interval[i];
			num++;
			if(sum /num < 0.5)
			{
				end_congestion = obj_pos[i][0];
			}
			if(sum / num > 0.5 || tolerate_num >= 2)
			{
				if((end_congestion - start_congestion) >= (lane_bottom - lane_top) / 3)//此段已经达到拥堵条件,不再继续分析
				{
					cal_congestion = 0;
					break;
				}
				else//重新判断
				{
					start_congestion = obj_pos[i][0];
					end_congestion = obj_pos[i][0];
					cal_congestion = 0;
				}
			}

		}
	}
	if((end_congestion - start_congestion) >= (lane_bottom - lane_top) / 3 && obj_num >=  4)//此帧为拥堵
	{
		bCongestion = TRUE;
	}
	pCfgs->CongestionBox[LaneID].uNewEventFlag = 0;
	//printf("congestion:[%d,%d],%d,%d,%d\n",end_congestion,start_congestion,obj_num,bCongestion,pCfgs->uStatCongestionNum[LaneID]);
	if(pCfgs->bStatCongestion[LaneID][0])
		pCfgs->uStatCongestionNum[LaneID]--;
	if(bCongestion)
		pCfgs->uStatCongestionNum[LaneID]++;
	for(i = 1; i < 150; i++)//将拥堵情况加入统计数组中
	{
		pCfgs->bStatCongestion[LaneID][i - 1] = pCfgs->bStatCongestion[LaneID][i];
	}
	pCfgs->bStatCongestion[LaneID][149] = bCongestion;
	if(pCfgs->uStatCongestionNum[LaneID] > 140 && pCfgs->bCongestion[LaneID] == FALSE && bCongestion)//当拥堵数达到一定数目时，为拥堵
	{
		pCfgs->CongestionBox[LaneID].uNewEventFlag = 1;
	}
	else if(pCfgs->uStatCongestionNum[LaneID] < 50 && pCfgs->bCongestion[LaneID] == TRUE && bCongestion == FALSE)//当拥堵数降为一定数目，为不拥堵
	{
		pCfgs->bCongestion[LaneID] = FALSE;
	}
	if(pCfgs->CongestionBox[LaneID].uNewEventFlag == 1)//新产生的拥堵事件 
	{
		pCfgs->bCongestion[LaneID] = TRUE;
		pCfgs->CongestionBox[LaneID].uEventID = pCfgs->eventID++;
	}*/
	if(obj_num >= 4)//此车道有4个目标以上，才进行拥堵判断
	{
		int idx = 0;
		for(i = 0; i < obj_num; i++)//求出均值
		{
			sum += (obj_pos[i][0] + obj_pos[i][1]) / 2; 
		}
		sum = sum / obj_num;
		//找出均值所在的框位置
		for(i = 0; i < obj_num; i++)
		{
			if(sum > obj_pos[i][1])
			{
				idx = i;
				break;
			}
		}
		if((obj_pos[idx][0] + obj_pos[idx][1]) < (lane_top + lane_bottom))//从车道上往下判断
		{
			for(i = obj_num - 1; i >= 0; i--)
			{
				//根据目标的距离判断拥堵情况
				if(obj_interval[i] < 0.3 && cal_congestion == 0)
				{
					end_congestion = obj_pos[i][1];
					cal_congestion = 1;
					sum = 0;
					num = 0;
					tolerate_num = 0;
				}
				else if(obj_interval[i] > 1 && cal_congestion)
				{
					tolerate_num++;
				}
				/*if(obj_interval[i] > 2)
				{
					break;
				}*/
				if(IsVehStatic[i] && cal_congestion)//统计静止车辆数
				{
					vehicle_static_num++;
				}
				if(cal_congestion)
				{
					//sum += (obj_interval[i] < 0) ? 0 : obj_interval[i];
					sum += obj_interval[i];
					num++;
					if(sum /num < 0.5)
					{
						start_congestion = obj_pos[i][0];
					}
					if(sum / num > 0.5 || tolerate_num >= 2)
					{
						if((start_congestion - end_congestion) >= (lane_bottom - lane_top) / 3)//此段已经达到拥堵条件,不再继续分析
						{
							cal_congestion = 0;
							break;
						}
						else//重新判断
						{
							start_congestion = obj_pos[i][0];
							end_congestion = obj_pos[i][0];
							cal_congestion = 0;
							vehicle_static_num = 0;
						}
					}

				}
			}
		}
		else
		{
			for( i = 0; i < obj_num; i++)//从下向上判断
			{
				//根据目标的距离判断拥堵情况
				if(obj_interval[i] < 0.3 && cal_congestion == 0)
				{
					start_congestion = obj_pos[i][0];
					cal_congestion = 1;
					sum = 0;
					num = 0;
					tolerate_num = 0;
				}
				else if(obj_interval[i] > 1 && cal_congestion)
				{
					tolerate_num++;
				}
				/*if(obj_interval[i] > 2)
				{
					break;
				}*/
				if(IsVehStatic[i] && cal_congestion)//统计静止车辆数
				{
					vehicle_static_num++;
				}
				if(cal_congestion)
				{
					//sum += (obj_interval[i] < 0) ? 0 : obj_interval[i];
					sum += obj_interval[i];
					num++;
					if(sum /num < 0.5)
					{
						end_congestion = obj_pos[i][1];
					}
					if(sum / num > 0.5 || tolerate_num >= 2)
					{
						if((start_congestion - end_congestion) >= (lane_bottom - lane_top) / 3)//此段已经达到拥堵条件,不再继续分析
						{
							cal_congestion = 0;
							break;
						}
						else//重新判断
						{
							start_congestion = obj_pos[i][0];
							end_congestion = obj_pos[i][0];
							cal_congestion = 0;
							vehicle_static_num++;
						}
					}

				}
			}
		}
		/*for(i = 0; i < obj_num; i++)
		{
			printf("[%f, %d, %d, %d] ", obj_interval[i],IsVehStatic[i],obj_pos[i][0],obj_pos[i][1]);
		}
		printf("\nlane id = %d, obj_num = %d, static_num = %d, start_congestion = %d, end_congestion = %d,lane_top = %d,lane_bottom = %d\n", LaneID, obj_num, vehicle_static_num, start_congestion, end_congestion,lane_top,lane_bottom);*/
		if(vehicle_static_num >= 4 && abs(end_congestion - start_congestion) >= (lane_bottom - lane_top) / 3)//有4辆以上的车辆静止，并且拥堵达到一定长度,则此帧拥堵
			bCongestion = TRUE;
		/*if(obj_num >= 4 && abs(end_congestion - start_congestion) >= (lane_bottom - lane_top) / 2 && no_vehicle_ratio < 0.3)//缓行判断
		{
		bDriveSlow = TRUE;
		//printf("drive slowly\n");
		}*/
	}
	//对拥堵情况进行分析
	pCfgs->CongestionBox[LaneID].uNewEventFlag = 0;

	for(i = 1; i < 150; i++)//将拥堵情况加入统计数组中,保存每帧实际时间
	{
		pCfgs->bStatCongestion[LaneID][i - 1] = pCfgs->bStatCongestion[LaneID][i];
	}
	pCfgs->bStatCongestion[LaneID][149] = bCongestion;

	if(pCfgs->bCongestion[LaneID] == FALSE && bCongestion)//判断是否满足拥堵条件
	{
		congestion_num = 0;
		totalnum = 0;
		for( i = 149; i >= 0; i--)
		{
			if(pCfgs->bStatCongestion[LaneID][i])
				congestion_num++;
			totalnum++;
			if((pCfgs->currTime - pCfgs->uStatFrameTime[i]) > pCfgs->uCongestionThreshTime)//当拥堵达到一定时间间隔，认为拥堵
			{
				if((float)congestion_num / (float)totalnum > 0.95)
				{
					pCfgs->CongestionBox[LaneID].uNewEventFlag = 1;
				}
				printf("start congestion, interval time = %d,[%d,%d]\n", 149 - i, congestion_num, totalnum);
				break;
			}
		}
		if(i == -1)//统计数组没有达到统计时间间隔
		{
			if((float)congestion_num / (float)totalnum > 0.95)
			{
				pCfgs->CongestionBox[LaneID].uNewEventFlag = 1;
			}
		}
	}
	if(pCfgs->bCongestion[LaneID] == TRUE && bCongestion == FALSE)//判断是否结束拥堵条件
	{
		congestion_num = 0;
		totalnum = 0;
		for( i = 149; i >= 0; i--)
		{
			if(pCfgs->bStatCongestion[LaneID][i])
				congestion_num++;
			totalnum++;
			if((pCfgs->currTime - pCfgs->uStatFrameTime[i]) > 2)//当达到不拥堵条件时，设置此车道不拥堵
			{
				if((float)congestion_num / (float)totalnum < 0.5)
				{
					pCfgs->bCongestion[LaneID] = FALSE;
				    printf("end congestion, interval time = %d,[%d,%d]\n", 149 - i, congestion_num, totalnum);
				}
				break;
			}
		}
	}
	if(pCfgs->CongestionBox[LaneID].uNewEventFlag == 1 && (pCfgs->currTime - pCfgs->uCongestionTime[LaneID]) > (pCfgs->EventDetectCfg.ReportInterval[CONGESTION] * 60))//新产生的拥堵事件，并且达到前后拥堵的时间阈值
	{
		printf("congestion event\n");
		pCfgs->bCongestion[LaneID] = TRUE;
		pCfgs->CongestionBox[LaneID].uEventID = pCfgs->eventID++;
		pCfgs->uCongestionTime[LaneID] = pCfgs->currTime;//上一次拥堵时间
	}
	/*if(bCongestion)
	{
		printf("Congestion = %d %d,%d\n",bCongestion, pCfgs->bCongestion[LaneID],pCfgs->uStatCongestionNum[LaneID]);
	}*/
	//计算拥堵线
	if(pCfgs->bCongestion[LaneID])
	{
		pCfgs->uCongestionNum++;
		if(m_ptend[0].y != m_ptend[2].y && m_ptend[1].y != m_ptend[3].y)
		{
			if(m_ptend[0].x == m_ptend[3].x)
			{
				x1 = m_ptend[0].x;
				x3 = m_ptend[0].x;
			}
			else
			{
				x1 = (start_congestion - m_ptend[0].y) * (m_ptend[3].x - m_ptend[0].x) / (m_ptend[3].y - m_ptend[0].y) + m_ptend[0].x;
				x3 = (end_congestion- m_ptend[0].y) * (m_ptend[3].x - m_ptend[0].x) / (m_ptend[3].y - m_ptend[0].y) + m_ptend[0].x;
			}
			if(m_ptend[1].x == m_ptend[2].x)
			{
				x2 = m_ptend[1].x;
				x4 = m_ptend[1].x;
			}	
			else
			{
				x2 = (start_congestion - m_ptend[1].y) * (m_ptend[2].x - m_ptend[1].x) / (m_ptend[2].y - m_ptend[1].y) + m_ptend[1].x;
				x4 = (end_congestion - m_ptend[1].y) * (m_ptend[2].x - m_ptend[1].x) / (m_ptend[2].y - m_ptend[1].y) + m_ptend[1].x;
			}
			pCfgs->CongestionBox[LaneID].EventBox[0].x = x1;
			pCfgs->CongestionBox[LaneID].EventBox[0].y = start_congestion;
			pCfgs->CongestionBox[LaneID].EventBox[1].x = x2;
			pCfgs->CongestionBox[LaneID].EventBox[1].y = start_congestion;
			pCfgs->CongestionBox[LaneID].EventBox[2].x = x4;
			pCfgs->CongestionBox[LaneID].EventBox[2].y = end_congestion;
			pCfgs->CongestionBox[LaneID].EventBox[3].x = x3;
			pCfgs->CongestionBox[LaneID].EventBox[3].y = end_congestion;
			bool inRegion = FALSE;
			for(i = 0; i < pCfgs->EventDetectCfg.uEventRegionNum; i++)
			{ 
				CPoint ptCorner[4];
				memcpy((void*)ptCorner, (void*)pCfgs->EventDetectCfg.EventRegion[i].detRegion, 4 * sizeof(CPoint));
				//判断拥堵区域顶点是否在事件区域内
				inRegion = FALSE;
				if(isPointInRect(pCfgs->CongestionBox[LaneID].EventBox[0], ptCorner[3], ptCorner[0], ptCorner[1], ptCorner[2]))
				{
					inRegion = TRUE;
					break;
				}
				if(isPointInRect(pCfgs->CongestionBox[LaneID].EventBox[1], ptCorner[3], ptCorner[0], ptCorner[1], ptCorner[2]))
				{
					inRegion = TRUE;
					break;
				}
				if(isPointInRect(pCfgs->CongestionBox[LaneID].EventBox[2], ptCorner[3], ptCorner[0], ptCorner[1], ptCorner[2]))
				{
					inRegion = TRUE;
					break;
				}
				if(isPointInRect(pCfgs->CongestionBox[LaneID].EventBox[3], ptCorner[3], ptCorner[0], ptCorner[1], ptCorner[2]))
				{
					inRegion = TRUE;
					break;
				}
			}
			if(inRegion)
				pCfgs->CongestionBox[LaneID].uRegionID = pCfgs->EventDetectCfg.EventRegion[i].uRegionID;
			else
				pCfgs->CongestionBox[LaneID].uRegionID = pCfgs->EventDetectCfg.EventRegion[0].uRegionID;
			if(i == pCfgs->EventDetectCfg.uEventRegionNum)
			{
				pCfgs->CongestionBox[LaneID].uRegionID = pCfgs->EventDetectCfg.EventRegion[0].uRegionID;
			}
		}
		if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
		{
			pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = pCfgs->CongestionBox[LaneID].uNewEventFlag;
			pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->CongestionBox[LaneID].uRegionID;//事件区域ID
			if(pCfgs->CongestionBox[LaneID].uNewEventFlag == 1)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
			}
			memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, pCfgs->CongestionBox[LaneID].EventBox, 4 * sizeof(CPoint));
			pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = CONGESTION;
			pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
		}
	}

}
//计算排队长度
void QueLengthCaculate(Uint16 LaneID, ALGCFGS *pCfgs, ALGPARAMS	*pParams, CPoint m_ptend[], int width, int height, bool detRear)
{
	int i = 0, j = 0;
	int top = 0, bottom = 0;
	int obj_pos[MAX_DETECTION_NUM][2];//车辆框位置
	int obj_num = 0;//车辆框数量
	int lane_top = min(m_ptend[0].y, m_ptend[1].y);//车道最上端
	int lane_bottom = min(m_ptend[2].y, m_ptend[3].y);//车道最下端
	int que_pos = lane_bottom, que_pos1 = 0;
	float obj_interval[MAX_DETECTION_NUM] = { 0 };//框间距
	int no_interval_num = 0;
	bool IsVehStatic[MAX_DETECTION_NUM]; //车辆位置是否静止
	float sum = 0.0;
	int stat_num = 0, num = 0;
	int QueVehicleNum = 0;//排队区域内车辆数量
	int actual_dis = 0;
	//根据运动情况和车辆数进行排队长度计算
	//iSubStractImage(pParams->CurrQueueImage, pParams->PreQueueImage, 15, 0, pCfgs->team_height, pCfgs->team_width, pCfgs->team_height);
	//iSubStractImage(pParams->CurrQueueImage, pParams->PrePrePreQueueImage, pParams->MaskLaneImage, 15, LaneID, pCfgs->m_iWidth, pCfgs->m_iHeight);//隔三帧帧差
	/*//根据检测框进行排队长度分析
	for( i = 0; i < pCfgs->classes; i++)
	{
		if(pCfgs->detClasses[i].classes_num)
		{
			for( j = 0; j < pCfgs->detClasses[i].classes_num; j++)
			{
				if(pCfgs->detClasses[i].lane_id[j] == LaneID)
				{
					bottom = min(pCfgs->detClasses[i].box[j].y + pCfgs->detClasses[i].box[j].height, lane_bottom);
					top = min(pCfgs->detClasses[i].box[j].y, lane_bottom);
					if(bottom - top > 5)
					{
						if(obj_num < MAX_LANE_TARGET_NUM)
						{
							pCfgs->detBoxes[LaneID][obj_num] = pCfgs->detClasses[i].box[j];
						}//save
						obj_pos[obj_num][0] = bottom;
						obj_pos[obj_num][1] = top;
						obj_num++;
					}
				}
			}
		}
	}*/
	////根据跟踪框进行排队长度分析
	for( i = 0; i < pCfgs->detTargets_size; i++)
	{
		if(pCfgs->detTargets[i].lane_id == LaneID)
		{
			bottom = min(pCfgs->detTargets[i].box.y + pCfgs->detTargets[i].box.height, lane_bottom);
			top = min(pCfgs->detTargets[i].box.y, lane_bottom);
			if(bottom - top > 5)
			{
				if(obj_num < MAX_LANE_TARGET_NUM)
				{
					pCfgs->detBoxes[LaneID][obj_num] = pCfgs->detTargets[i].box;
				}//save
				obj_pos[obj_num][0] = bottom;
				obj_pos[obj_num][1] = top;
				obj_num++;
			}
		}
	}

	pCfgs->detNum[LaneID] = (obj_num > MAX_LANE_TARGET_NUM)? MAX_LANE_TARGET_NUM : obj_num;
	sort_obj(obj_pos, obj_num);//排序，从下向上
	for( i = 0; i < obj_num; i++)//计算每个框的间隔值和运动情况
	{
		int last_obj_bottom = lane_bottom;
		if(i == 0)
			last_obj_bottom = lane_bottom;
		else if(i == 1)
			last_obj_bottom = obj_pos[i - 1][1];
		else
			last_obj_bottom = min(obj_pos[i - 2][1], obj_pos[i - 1][1]);
		obj_interval[i] = (float)(last_obj_bottom - obj_pos[i][0])/(float)(obj_pos[i][0] - obj_pos[i][1]);//目标间隔比
		//sum += obj_interval[i];
		sum += (last_obj_bottom - obj_pos[i][0] < 0)? 0 : (last_obj_bottom - obj_pos[i][0]);//不是车辆的行数
		if(obj_interval[i] <= 0)
			no_interval_num++;
		else
			no_interval_num = 0;
		if(i != 0)
		{
			int pos1 = obj_pos[i][0];
			pos1 = max(0, int(pos1 / pCfgs->scale_y));
			pos1= min(pos1, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
			int pos2 = last_obj_bottom;
			pos2 = max(0, int(pos2 / pCfgs->scale_y));
			pos2= min(pos2, min(MAX_IMAGE_HEIGHT, pCfgs->img_height) - 1);
			actual_dis += (pCfgs->actual_distance[LaneID][pos1] - pCfgs->actual_distance[LaneID][pos2]) * (pCfgs->actual_distance[LaneID][pos1] - pCfgs->actual_distance[LaneID][pos2]);//车辆之间距离方差
		}
		if(i == obj_num - 1)
		{
			sum += (obj_pos[i][1] - lane_top < 0)? 0 :(obj_pos[i][1] - lane_top);//不是车辆的行数
		}
		IsVehStatic[i] = IsVehicleQueue(pParams->PrePrePreQueueImage, pParams->MaskLaneImage, LaneID, max(0, obj_pos[i][1] * pCfgs->m_iHeight / height), max(0, min(pCfgs->m_iHeight - 1, obj_pos[i][0] * pCfgs->m_iHeight / height)), pCfgs);
	}
	actual_dis = (obj_num > 1)? actual_dis / (obj_num - 1) : 0;
	//分析排队情况
	for( i = 0; i < obj_num; i++)
	{
		if(obj_interval[i] > 2)
		{
			break;
		}
		if(i == 1 && IsVehStatic[i] == FALSE && IsVehStatic[i - 1] == FALSE)
		{
			break;
		}
		if((IsVehStatic[i] && obj_interval[i] < 0.2 && (i == 0)) || (IsVehStatic[i] && IsVehStatic[i - 1] && obj_interval[i] < 0.5 && (i != 0)))// 0.1  0.5
		{
			que_pos = obj_pos[i][1];
		}
	}
	//printf("id =%d,vehicle_num = %d\n",LaneID,obj_num);

	//sum = (obj_num)? sum / obj_num : 0;
	/*if(obj_num > 3 && sum < 0.1 && no_interval_num > 0)
	{
		que_pos = obj_pos[obj_num - 1][1];
	}*/
	//统计得到排队长度
	stat_num = pCfgs->uStatQuePos[LaneID][5];
	if(stat_num < 5)
	{
		pCfgs->uStatQuePos[LaneID][stat_num] = que_pos;
		pCfgs->uStatQuePos[LaneID][5] = pCfgs->uStatQuePos[LaneID][5] + 1;
		stat_num = stat_num + 1;
	}
	else
	{
		for(num = 1; num < 5; num++)
		{
			pCfgs->uStatQuePos[LaneID][num - 1] = pCfgs->uStatQuePos[LaneID][num];
		}
		pCfgs->uStatQuePos[LaneID][4] = que_pos;

	}
	//防止排队长度闪烁
	que_pos1 = sort_median(pCfgs->uStatQuePos[LaneID], stat_num);
	//printf("[%d,%d,%d,%d,%d,%d],%d\n",pCfgs->uStatQuePos[LaneID][0],pCfgs->uStatQuePos[LaneID][1],pCfgs->uStatQuePos[LaneID][2],pCfgs->uStatQuePos[LaneID][3],pCfgs->uStatQuePos[LaneID][4],pCfgs->uStatQuePos[LaneID][5],que_pos);

	if(obj_num == 0)
	{
		que_pos1 = lane_bottom;
	}
	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleQueueLength = que_pos1;//排队时最后的位置
	//计算排队区域内车辆数
	if(que_pos1 < lane_bottom)//有排队
	{
		for( i = 0; i < obj_num; i++)
		{
			if(obj_pos[i][0] > que_pos1)
				QueVehicleNum++;
		}
	}
	else
	{
		QueVehicleNum = 0;
	}
	//分布情况
	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDistribution = (actual_dis > 254)? 254 : actual_dis;
	//得到车辆密度
	if(sum == 0)
	{
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDensity = 0;
	}
	else
	{
		pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDensity = (lane_bottom - lane_top - sum) * 100 /(lane_bottom - lane_top);
	}
	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDensity = (pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDensity < 0)? 0 : pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDensity;
	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDensity = (pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDensity > 100)? 100 : pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uVehicleDensity;
	//通道内排队数量
	pCfgs->ResultMsg.uResultInfo.uEachLaneData[LaneID].SpeedDetectInfo1.uQueueVehiSum = QueVehicleNum;//通道内排队数量
	//排队长度
	memcpy((void *)pParams->PreQueueImage, (void *)pParams->CurrQueueImage, pCfgs->m_iWidth * pCfgs->m_iHeight);
	//判断车道是否拥堵
	if(pCfgs->bDetCongestion == TRUE)
		CongestionDetect(LaneID, pCfgs, obj_pos, obj_num, m_ptend, obj_interval, IsVehStatic);
}
//帧差
/*void iSubStractImage(Uint8 *puSourceImage,Uint8 *puTargetImage, Uint32 nThreshold, Int16 nFromLine, Int16 nToLine, Int16 width, Int16 height)
{
	Int32 iRow,iCol,nCompareResult;

	for( iRow = nFromLine; iRow < nToLine; iRow++ )
	{
		for( iCol = 0; iCol < width; iCol++ )
		{
			nCompareResult =  *( puSourceImage + iCol + width * iRow )  -  *( puTargetImage + iCol + width * iRow ) ;
			if( abs(nCompareResult) < nThreshold )
				nCompareResult = 0;
			*( puTargetImage + iCol + width * iRow ) = (Uint16)abs(nCompareResult);
		}
	}
}*/
void iSubStractImage(Uint8 *puSourceImage,Uint8 *puTargetImage, Uint8 *puMaskImage, Uint32 nThreshold, Int16 laneID, Int16 width, Int16 height)
{
	Int32 iRow, iCol, nCompareResult;

	for( iRow = 0; iRow < height; iRow++ )
	{
		for( iCol = 0; iCol < width; iCol++ )
		{
			if( *(puMaskImage + iCol + width * iRow))
			{
				nCompareResult =  *( puSourceImage + iCol + width * iRow )  -  *( puTargetImage + iCol + width * iRow ) ;
				if( abs(nCompareResult) < nThreshold )
					nCompareResult = 0;
				*( puTargetImage + iCol + width * iRow ) = (Uint8)abs(nCompareResult);
			}
		}
	}
}

///////////////////////////////////////////////
float fuzzy(unsigned char* puNewImage, int nWidth, int nHight)//计算视频对比度
{
	float degree = 0.0;
	int i, j;
	unsigned char x1, x2, x3;
	float temp = 0.0;
	int count = 0;
	for(i = 100; i < nHight - 100;i += 4)  
	{  
		for(j = 0; j < nWidth; j += 4)  
		{  
			x1 = *(puNewImage + i * nWidth + j);
			x2 = *(puNewImage + (i + 1) * nWidth + j);
			x3 = *(puNewImage + i * nWidth + j + 1);
			degree = (x2 - x1) * (x2 - x1) + (x3 - x1) * (x3 - x1);
			temp += sqrt(degree);
			temp += abs(x2 - x1) + abs(x3 - x1);
			count++;
		}  
	}  
	degree = temp / count;
	return degree;
}

//计算视频图像颜色异常
bool Color_deviate(unsigned char* uImage, unsigned char* vImage, int width, int height)
{	
	float ave_a = 0, ave_b = 0, std_a = 0, std_b = 0;
	int x = 0, y = 0;
	float color_deviate = 0;
	int pixelnum = 0;
	int temp_a, temp_b;
	//pixelnum = width * height;
	for (y = 100; y < height - 100; y += 4)
	{
		for (x = 0; x < width; x += 4)
		{
			ave_a += *(uImage + y * width + x) - 128;
			ave_b += *(vImage + y * width + x) - 128;
			pixelnum++;
		}
	} 
	ave_a /= pixelnum;
	ave_b /= pixelnum;


	for (y = 100; y < height - 100; y += 4)
	{
		for (x = 0; x < width; x += 4)
		{
			temp_a = *(uImage + y * width + x) - 128;
			std_a += (temp_a - ave_a) * (temp_a - ave_a);
			temp_b = *(vImage + y * width + x) - 128;
			std_b += (temp_b - ave_b) * (temp_b - ave_b);
		}
	}
	std_a /= pixelnum;
	std_b /= pixelnum;
	color_deviate = sqrt(ave_a * ave_a + ave_b * ave_b) / sqrt(std_a + std_b);
	//printf("\ncolor deviate is:%f\n,",color_deviate*10);

	if (color_deviate >= 5 || color_deviate < 0.05)
	{
		return TRUE;

	} 
	else
	{
		return FALSE; 
	}

}
//统计能见度
bool visible_judge(Uint16 *a, int visib_length, int threshold)
{
	int i = 0, num = 0;
	for (i = 0; i < visib_length; i++)
	{
		if (a[i] < threshold)
		{
			num++;
		}
		else
		{
			break;
		}

	}
	//当能见度大于数组长度一半以上，才认为是能见度高
	if (num > (visib_length / 2))
	{
		return TRUE;
	} 
	else
	{
		return FALSE;
	}
}
