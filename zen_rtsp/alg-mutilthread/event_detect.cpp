#include "m_arith.h"
#ifndef MIN
#define MIN(a,b)  ((a) > (b) ? (b) : (a))
#endif

#ifndef MAX
#define MAX(a,b)  ((a) < (b) ? (b) : (a))
#endif
//配置交通事件区域
bool CfgEventRegion(mEventInfo	EventDetectCfg, ALGCFGS *pCfgs, ALGPARAMS *pParams)
{
	int i = 0, j = 0, k = 0;
	int uEventRegionNum = 0;
	CPoint ptCorner[4];
	/*pParams->MaskIllegalParkImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 4;
	pParams->MaskOppositeDirDriveImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 5;
	pParams->MaskOffLineImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 6;
	pParams->MaskNoPersonAllowImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 7;
	pParams->MaskNonMotorAllowImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 8;
	pParams->MaskPersonFallImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 9;
	pParams->MaskDropImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 10;
	pParams->MaskTrafficAccidentImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 11;*/
	pParams->CurrBackImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 6;
	pParams->BufferBackImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 7;
	pParams->ForeImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 8;
	pParams->MaskOppositeDirDriveImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 9;
	pParams->MaskEventIDImage = (Uint8 *)pParams->CurrQueueImage + DETECTRECT_WIDTH_MAX * DETECTRECT_HEIGHT_MAX * 10;
	//先设置不检测拥堵
	pCfgs->bDetCongestion = FALSE;
	pCfgs->uCongestionThreshTime = 10;//设置拥堵阈值10s
	//加载事件检测区域
	for(i = 0; i < EventDetectCfg.eventAreaNum; i++)
	{
		Uint16 areaNum = EventDetectCfg.eventArea[i].areaNum;//区域编号
		mSelectType  eventType = EventDetectCfg.eventArea[i].eventType;//事件类型
		for(j = 0; j < 4; j++)
		{
			ptCorner[j].x = EventDetectCfg.eventArea[i].realcoordinate[j].x;
			ptCorner[j].y = EventDetectCfg.eventArea[i].realcoordinate[j].y;
		}
		CorrectRegionPoint(ptCorner);//校正区域坐标
		for(j = 0; j < MAX_EVENT_TYPE; j++)
		{
			if(eventType.type & (1 << j))
			{
				pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].uRegionID = areaNum;//区域编号
				memcpy((void*)pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].detRegion, (void*)ptCorner, 4 * sizeof(CPoint));
				pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].direction = EventDetectCfg.eventArea[i].direction;//区域方向
				pCfgs->EventDetectCfg.ReportInterval[j + 1] = EventDetectCfg.eventArea[i].report[j];//传入差一位
				pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].eventType = (enum eventType)(j + 1);
		        printf("[%d,%d,%d,%d],%d\n",pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].detRegion[0].x,pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].detRegion[0].y,pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].detRegion[1].x,pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].detRegion[1].y,pCfgs->EventDetectCfg.EventRegion[uEventRegionNum].eventType);
				uEventRegionNum++;
			}
		}
	}
	pCfgs->EventDetectCfg.uEventRegionNum = uEventRegionNum;//事件检测区域数
	//判断是否检测拥堵
	for(i = 0; i < uEventRegionNum; i++)
	{
		if(pCfgs->EventDetectCfg.EventRegion[i].eventType == CONGESTION)
		{
			pCfgs->bDetCongestion = TRUE;
			break;
		}
	}
	memset(pCfgs->uStatCongestionNum, 0, MAX_LANE * sizeof(Uint16));//统计拥堵数量
	printf("ReportInterval time REVERSE_DRIVE:%d,STOP_INVALID:%d,PERSON:%d,DRIVE_AWAY:%d,CONGESTION:%d,DROP:%d,PERSONFALL:%d,NONMOTORFALL:%d,NONMOTOR:%d,ACCIDENTTRAFFIC:%d,GREENWAYDROP:%d]\n",pCfgs->EventDetectCfg.ReportInterval[REVERSE_DRIVE],pCfgs->EventDetectCfg.ReportInterval[STOP_INVALID],pCfgs->EventDetectCfg.ReportInterval[NO_PEDESTRIANTION],pCfgs->EventDetectCfg.ReportInterval[DRIVE_AWAY],pCfgs->EventDetectCfg.ReportInterval[CONGESTION],pCfgs->EventDetectCfg.ReportInterval[DROP],pCfgs->EventDetectCfg.ReportInterval[PERSONFALL],pCfgs->EventDetectCfg.ReportInterval[NONMOTORFALL],pCfgs->EventDetectCfg.ReportInterval[NONMOTOR],pCfgs->EventDetectCfg.ReportInterval[ACCIDENTTRAFFIC],pCfgs->EventDetectCfg.ReportInterval[GREENWAYDROP]);
	//交通事件检测信息初始化
	memset(pCfgs->event_targets, 0, MAX_TARGET_NUM * sizeof(CTarget));//交通事件检测区域目标
	pCfgs->event_target_id = 1;
	pCfgs->event_targets_size = 0;
	pCfgs->eventID = 1;//交通事件ID初始化为1
	pCfgs->uIllegalParkNum = 0;
	memset(pCfgs->IllegalParkBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//禁止停车
	pCfgs->uIllegalParkTime = 0;//前一停车时间
	pCfgs->uIllegalParkID = 0;//前一停车事件ID
	pCfgs->uOppositeDirDriveNum = 0;
	memset(pCfgs->OppositeDirDriveBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//禁止逆行
	pCfgs->uOppositeDirDriveTime = 0;//前一逆行时间
	pCfgs->uOppositeDirDriveID = 0;//前一逆行事件ID
	memset(pCfgs->direction, 0, MAX_REGION_NUM * sizeof(Uint16));//区域运行方向
    //memset(pCfgs->bCongestion, FALSE, MAX_LANE * sizeof(bool));//车道区域拥堵
	pCfgs->uCongestionNum = 0;
	memset(pCfgs->CongestionBox, 0, MAX_LANE * sizeof(EVENTBOX));
	pCfgs->uOffLaneNum = 0;
	memset(pCfgs->OffLaneBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//偏离车道
	pCfgs->uOffLaneTime = 0;//前一驶离时间
	pCfgs->uOffLaneID = 0;//前一驶离事件ID
	pCfgs->uNoPersonAllowNum = 0;
	memset(pCfgs->NoPersonAllowBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//违法行人
	pCfgs->uPersonEventTime = 0;//前一行人事件时间
	pCfgs->uCurrentPersonID = 0;//前一行人事件ID
	pCfgs->uNonMotorAllowNum = 0;
	memset(pCfgs->NonMotorAllowBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//违法非机动车
	pCfgs->uNonMotorEventTime = 0;//前一非机动车事件时间
	pCfgs->uCurrentNonMotorID = 0;//前一非机动车事件ID
	pCfgs->uDropNum = 0;
	memset(pCfgs->DropBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//抛弃物
	pCfgs->uPersonFallNum = 0;//行人跌倒数
	memset(pCfgs->PersonFallBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//行人跌倒
	pCfgs->uPersonFallEventTime = 0;//前一行人跌倒事件时间
	pCfgs->uCurrentPersonFallID = 0;//前一行人跌倒事件ID
	pCfgs->uNonMotorFallNum = 0;//非机动车跌倒数
	memset(pCfgs->NonMotorFallBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//非机动车跌倒
	pCfgs->uNonMotorFallEventTime = 0;//前一非机动车跌倒事件时间
	pCfgs->uCurrentNonMotorFallID = 0;//前一非机动车跌倒事件ID
	pCfgs->uGreenwayDropNum = 0;//绿道抛弃物
	memset(pCfgs->GreenwayDropBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//绿道抛弃物框
	pCfgs->uGreenwayDropEventTime = 0;//前一绿道抛弃物事件时间
	pCfgs->uCurrentGreenwayDropID = 0;//前一绿道抛弃物事件ID
	pCfgs->uTrafficAccidentNum = 0;
	memset(pCfgs->TrafficAccidentBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));//交通事故
	pCfgs->uTrafficAccidentTime = 0;//前一交通事故时间
	pCfgs->uTrafficAccidentID = 0;//前一交通事故ID
	pCfgs->bMaskEventImage = FALSE;//交通事件掩模图像
	pCfgs->CurrCandidateROINum = 0;//用于抛弃物检测
	memset(pCfgs->CurrCandidateROI, 0, 50 * sizeof(CRect));//用于存储抛弃物候选区域
	memset(pCfgs->abandoned_targets, 0, 10 * sizeof(CTarget));//用于存储抛弃物目标
	pCfgs->abandoned_targets_id = 1;
	pCfgs->abandoned_targets_size = 0;
	pCfgs->gThisFrameTime = 0;
	memset(pCfgs->EventInfo, 0 , MAX_EVENT_NUM * sizeof(EVENTINFO));//初始化事件信息
	pCfgs->EventNum = 0;//设置事件数量为0  
	pCfgs->video_fps = 0;
	pCfgs->EventState = 0;
	pCfgs->HaveEvent = FALSE;
	pCfgs->first_update = 0;
	pCfgs->EventBeginTime = 0;
	pCfgs->EventEndTime = 0;
	//道路事件检测信息初始化
	memset(pCfgs->road_event_targets, 0, MAX_ROAD_TARGET_NUM * sizeof(CTarget));//道路事件检测区域目标
	pCfgs->road_event_target_id = 1;
	pCfgs->road_event_targets_size = 0;
	return TRUE;

}
//pParams->MaskEventImage的不同值代表检测不同的交通事件区域
bool MaskEventImage(ALGCFGS *pCfgs, ALGPARAMS *pParams, int imgW, int imgH)
{
	Int32	i, j, k;
	CPoint	ptDetectCorner[4];
	CPoint pt;
	//标记事件检测区域
	memset(pParams->MaskEventImage, 0, pCfgs->m_iWidth * pCfgs->m_iHeight * sizeof(Uint32));
	memset(pParams->MaskOppositeDirDriveImage, 0, pCfgs->m_iWidth * pCfgs->m_iHeight);
	for(i = 0; i < pCfgs->m_iHeight; i++)//事件区域ID初始化为255
	{
		for(j = 0; j < pCfgs->m_iWidth; j++)
		{
			*(pParams->MaskEventIDImage + i * pCfgs->m_iWidth + j) = 255;
		}
	}
	//设置事件检测区域
	for(i = 0; i < pCfgs->EventDetectCfg.uEventRegionNum; i++)
	{
		(void*)memcpy(ptDetectCorner, (void*)pCfgs->EventDetectCfg.EventRegion[i].detRegion, 4 * sizeof(CPoint));
		//resize 到640 * 480
		for(j = 0; j < 4; j++)
		{
			ptDetectCorner[j].x = ptDetectCorner[j].x * pCfgs->m_iWidth / imgW;
			ptDetectCorner[j].y = ptDetectCorner[j].y * pCfgs->m_iHeight / imgH;

		}
		EventType eventType = pCfgs->EventDetectCfg.EventRegion[i].eventType;
		if(eventType == REVERSE_DRIVE)//逆行区域有方向,单独设置
		{
			Uint8* p;
			for(j = 0; j < pCfgs->m_iHeight; j++)
			{
				p = pParams->MaskOppositeDirDriveImage + j * pCfgs->m_iWidth;
				for(k = 0; k < pCfgs->m_iWidth; k++)
				{
					pt.x = k;
					pt.y = j;
					if(isPointInRect(pt, ptDetectCorner[3], ptDetectCorner[0], ptDetectCorner[1], ptDetectCorner[2]))
					{
						if(pCfgs->EventDetectCfg.EventRegion[i].direction)//0 为向下，1为向上
						{
							p[k] = 255;
						}
						else
						{
							p[k] = 128;
						}
					}
				}
			}
		}
		else //其他事件
		{
			Uint32* p;
			int num = 0;
			printf("type = %d,[%d,%d,%d,%d]\n",eventType,ptDetectCorner[0].x,ptDetectCorner[0].y,ptDetectCorner[1].x,ptDetectCorner[1].y);
			for(j = 0; j < pCfgs->m_iHeight; j++)
			{
				p = pParams->MaskEventImage + j * pCfgs->m_iWidth;
				for(k = 0; k < pCfgs->m_iWidth; k++)
				{
					pt.x = k;
					pt.y = j;
					if(isPointInRect(pt, ptDetectCorner[3], ptDetectCorner[0], ptDetectCorner[1], ptDetectCorner[2]))
					{
						p[k] += (1 << eventType);//将相应事件位设为1
						num++;
					}
				}
			}
		}
		//设置事件区域ID
		Uint16 uRegionID = pCfgs->EventDetectCfg.EventRegion[i].uRegionID;
		for(j = 0; j < pCfgs->m_iHeight; j++)
		{
			Uint8* p = pParams->MaskEventIDImage + j * pCfgs->m_iWidth;
			for(k = 0; k < pCfgs->m_iWidth; k++)
			{
				pt.x = k;
				pt.y = j;
				if(isPointInRect(pt, ptDetectCorner[3], ptDetectCorner[0], ptDetectCorner[1], ptDetectCorner[2]))
				{
					p[k] = uRegionID;
				}
			}
		}

	}
	//保存掩模图像
	/*IplImage* mask = cvCreateImage(cvSize(pCfgs->m_iWidth, pCfgs->m_iHeight), IPL_DEPTH_8U, 1);
	memcpy(mask->imageData, pParams->MaskOppositeDirDriveImage, pCfgs->m_iWidth * pCfgs->m_iHeight);
	cvSaveImage("maskEvent1.jpg", mask, 0);
	cvReleaseImage(&mask);*/

	return	TRUE;

}
//判断检测框是否在检测区域内，用于逆行检测
int RectInRegion0(unsigned char* maskImage, int width, int height, CRect rct, EventType type)
{
	int isInRegion = 0;
	int i = 0, j = 0;
	int num = 0;
	int val = 0;
	unsigned char* p;
	float ratio = 0;
	for(i = rct.y; i < (rct.y + rct.height); i++)
	{
		p = maskImage + i * width;
		for(j = rct.x; j < (rct.x + rct.width); j++)
		{
			int val0 = *(p + j);
			if(val0)
			{
				num++;
				val = *(p + j);
			}
		}
	}
	ratio = (float)num / (float)(rct.width * rct.height);
	if(ratio > 0.2)//大于阈值，在区域内
	{
		if(type == REVERSE_DRIVE)
		{
			if(val == 128)//加入方向信息，128代表direction为0，255代表direction为1
			{
				isInRegion = 1;
			}
			else
			{
				isInRegion = 2;
			}
		}

	}
	return isInRegion;
}
//判断检测框是否在检测区域内
int RectInRegion1(Uint32* maskImage, int width, int height, CRect rct, EventType type)
{
	int isInRegion = 0;
	int i = 0, j = 0;
	int num = 0;
	int val = 0;
	Uint32* p;
	float ratio = 0;
	for(i = rct.y; i < (rct.y + rct.height); i++)
	{
		p = maskImage + i * width;
		for(j = rct.x; j < (rct.x + rct.width); j++)
		{
			val = *(p + j);
			if(val & (1 << type))
			{
				num++;
			}
		}
	}
	ratio = (float)num / (float)(rct.width * rct.height);
	if(ratio > 0.2)//大于阈值，在区域内
	{
		isInRegion = 1;
	}
	return isInRegion;
}
void calc_fore(unsigned char *current, unsigned char *back, unsigned char* mask, unsigned char *fore, int width, int height, ALGPARAMS *pParams)//根据两个背景图像进行前景检测
{
	int i, j;
	unsigned char* p1 = current;
	unsigned char* p2 = back;
	unsigned char* p3 = fore;
	unsigned char* mask1 = mask;
	Uint32* mask2 = pParams->MaskEventImage;//抛弃物检测区域
	memset(fore, 0, width * height);
	for (i = 0; i < height; i++)
	{
		p1 = current + i * width;
		p2 = back + i * width;
		p3 = fore + i * width;
		mask1 = mask + i * width;
		mask2 = pParams->MaskEventImage + i * width;
		for (j = 0;j < width; j++)
		{
			if (p1[j] - p2[j] > 90 && mask1[j] == 0 && (mask2[j] & (1 << DROP)))
			{
				p3[j] = 255;//foreground
			}
		}
	}
}

void update_currentback(unsigned char *current, unsigned char *curr_back, int width, int height)//更新curr背景
{
	int i, j;
	unsigned char* p1 = current;
	unsigned char* p2 = curr_back;
	int val1, val2;
	for (i = 0; i < height; i++)
	{
		p1 = current + i * width;
		p2 = curr_back + i * width;
		for (j = 0; j < width; j++)
		{
			val1 = p1[j];
			val2 = p2[j];
			if (val1 >= val2)
			{
				if (val2 == 255)
				{
					val2 = 254;
				}
				p2[j] = val2 + 1;
			}else
			{
				if (val2 == 0)
				{
					val2 = 1;
				}
				p2[j] = val2 - 1;
			}
		}
	}
}

void update_bufferedback(unsigned char *curr_back, unsigned char *buf_back, unsigned char *abandon, int width, int height, ALGCFGS* pCfgs)//更新buffer背景
{
	int i, j;
	int leave_flag = 0;
	unsigned char* p1 = curr_back;
	unsigned char* p2 = buf_back;
	unsigned char* p3 = abandon;
	if (pCfgs->first_update == 0)
	{
		for ( i = 0; i < height; i++)
		{
			p1 = curr_back + i * width;
			p2 = buf_back + i * width;
			p3 = abandon + i * width;
			for (j = 0; j < width; j++)
			{
				if (abs(p1[j] - p2[j]) <= 50 )
				{
					p3[j] = 0;//background
				}else
				{
					p3[j] = p1[j];//foreground
					pCfgs->first_update = 1;
				}
			}
		}
		return;
	}

	//物体离开判断
	for ( i = 0; i < height; i++)
	{
		p1 = curr_back + i * width;
		p3 = abandon + i * width;
		for (j = 0; j < width; j++)
		{
			if ( p3[j] != 0)
			{
				if (p3[j] != p1[j])
				{
					leave_flag = 1;	//物体掩膜处之前背景与当前的不一致，1：物体未离开，0：物体离开
				}
			}
		}
	}

	if(leave_flag == 0)	//物体离开
	{
		memcpy(curr_back, buf_back, width * height);
		for ( i = 0; i < height; i++)
		{
			p1 = curr_back + i * width;
			p3 = abandon + i * width;
			for (j = 0; j < width; j++)
			{
				if (p3[j] != 0)
				{
					p3[j] = p1[j];
				}
			}
		}
	}else//物体未离开
	{
		for ( i = 0; i < height; i++)
		{
			p2 = buf_back + i * width;
			p3 = abandon + i * width;
			for (j = 0; j < width; j++)
			{
				if (p3[j] != 0)
				{
					p2[j] = p3[j];
				}
			}
		}
	}	

}
void DropDetect(ALGCFGS *pCfgs, ALGPARAMS *pParams, int event_idx, int width, int height)//抛弃物检测
{
	int i, j;
	CRect rct[50];
	int rct_num = 0;
	int match_rct[50] = {0};
	IplImage* foreImg = cvCreateImage(cvSize(pCfgs->m_iWidth, pCfgs->m_iHeight), IPL_DEPTH_8U, 1);//前景图像
	IplImage* maskImg = cvCreateImage(cvSize(pCfgs->m_iWidth, pCfgs->m_iHeight), IPL_DEPTH_8U, 1);//检测框的掩模图像
	unsigned char* fore = (unsigned char *)foreImg->imageData;//前景图像
	unsigned char* mask = (unsigned char *)maskImg->imageData;//检测框的掩模图像
	memset(mask, 0, pCfgs->m_iWidth * pCfgs->m_iHeight);
	if (pCfgs->gThisFrameTime == 0)//两个背景
	{
		//初始化背景模版
		memcpy((void *)pParams->CurrBackImage, (void *)pParams->CurrQueueImage, pCfgs->m_iWidth  * pCfgs->m_iHeight);
		memcpy((void *)pParams->BufferBackImage, (void *)pParams->CurrQueueImage, pCfgs->m_iWidth  * pCfgs->m_iHeight);
	}
	if(pCfgs->gThisFrameTime > 0)
	{
		//得到mask图像
		for(i = 0; i < pCfgs->classes; i++)
		{
			if(pCfgs->detClasses[i].classes_num == 0)
				continue;
			for(j = 0; j < pCfgs->detClasses[i].classes_num; j++)
			{
				CRect rct = pCfgs->detClasses[i].box[j];
				//对框进行缩放
				rct.x = rct.x * pCfgs->m_iWidth / width;
				rct.width = rct.width * pCfgs->m_iWidth /width;
				rct.y = rct.y * pCfgs->m_iHeight / height;
				rct.height = rct.height * pCfgs->m_iHeight / height;
				rct.width = ((rct.x + rct.width) > (pCfgs->m_iWidth - 1))? (pCfgs->m_iWidth - 1 - rct.x) : rct.width;//防止越界
				rct.height = ((rct.y + rct.height) > (pCfgs->m_iHeight - 1))? (pCfgs->m_iHeight - 1 - rct.y) : rct.height;
				//将框位置数据置1
				for(int k = rct.y; k <= (rct.y + rct.height); k++)
				{
					memset(mask + k * pCfgs->m_iWidth + rct.x, 1, rct.width);
				}
			}
		}
		//计算前景掩膜
		calc_fore(pParams->CurrBackImage, pParams->BufferBackImage, mask, fore, pCfgs->m_iWidth, pCfgs->m_iHeight, pParams);
		//更新跟踪背景
		update_currentback(pParams->CurrQueueImage, pParams->CurrBackImage, pCfgs->m_iWidth, pCfgs->m_iHeight);
		if (pCfgs->gThisFrameTime % 20 == 0)
		{
			update_bufferedback(pParams->CurrBackImage, pParams->BufferBackImage, pParams->ForeImage, pCfgs->m_iWidth, pCfgs->m_iHeight, pCfgs);
		}
		//对前景图像进行腐蚀膨胀以减少干扰
		cvDilate(foreImg, foreImg, 0, 2);//膨胀
		cvErode(foreImg, foreImg, 0, 2);//腐蚀
		//找最外接矩形
		CvSeq* pContour = NULL;
		CvMemStorage *pStorage = NULL;
		pStorage = cvCreateMemStorage(0);
		cvFindContours(foreImg, pStorage, &pContour, sizeof(CvContour), CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0,0));
		for (; pContour != NULL; pContour = pContour->h_next)   
        {   
			CvRect r = ((CvContour*)pContour)->rect;  
			int size =r.width * r.height;
			if(size > 50 && size < 10000)//满足面积条件，才认为是抛弃物区域
			{
				rct[rct_num].x = r.x * width / pCfgs->m_iWidth;
				rct[rct_num].y = r.y * height / pCfgs->m_iHeight;
				rct[rct_num].width = r.width * width / pCfgs->m_iWidth;
				rct[rct_num].height = r.height * height / pCfgs->m_iHeight;
				rct_num++;
			}
        }
		if(pStorage)
		{
			cvReleaseMemStorage(&pStorage);   
			pStorage = NULL;  
		}
		for(i = 0; i < pCfgs->abandoned_targets_size; i++)
		{
			pCfgs->abandoned_targets[i].detected = FALSE;
		}
		if(pCfgs->CurrCandidateROINum == 0)//上一帧没有候选区域
		{
			memcpy(pCfgs->CurrCandidateROI, rct, rct_num * sizeof(CRect));
			pCfgs->CurrCandidateROINum = rct_num;
		}
		else//上一帧有候选区域,进行匹配
		{
			for(j = 0; j < rct_num; j++)
			{
				match_rct[j] = 0;
				for(i = 0; i < pCfgs->CurrCandidateROINum; i++)
				{
					if(overlapRatio(pCfgs->CurrCandidateROI[i], rct[j]) > 50)//两个框匹配大于50
					{
						match_rct[j] = 1;
						break;
					}
				}
				//跟踪框和检测框进行匹配
				if(match_rct[j] == 1)
				{
					int match_sucess = -1;
					if(pCfgs->abandoned_targets_size)
					{
						for(i =  0; i < pCfgs->abandoned_targets_size; i++)
						{
							if(overlapRatio(pCfgs->abandoned_targets[i].box, rct[j]) > 20)//两个框匹配上
							{
								pCfgs->abandoned_targets[i].detected = TRUE;
								pCfgs->abandoned_targets[i].box = rct[j];
								match_sucess = 1;
							}
						}
					}
					if(match_sucess < 0 && pCfgs->abandoned_targets_size < 10)//没有匹配上，加入abandoned_targets中
					{
						CTarget nt; 
						Initialize_target(&nt);
						nt.box = rct[j];
						nt.detected = TRUE;
						memset(nt.event_continue_num, 0, MAX_EVENT_TYPE * sizeof(int));//初始化事件持续帧数
						memset(nt.event_flag, 0, MAX_EVENT_TYPE * sizeof(int));//初始化事件标记
						memset(nt.cal_event, FALSE, MAX_EVENT_TYPE * sizeof(bool));//初始化各类事件为未计算
						memset(nt.sign_event, 0, MAX_EVENT_TYPE * sizeof(int));//初始化为未标记的事件
						nt.target_id = (pCfgs->abandoned_targets_id > 5000)? 1 : pCfgs->abandoned_targets_id++;
						pCfgs->abandoned_targets[pCfgs->abandoned_targets_size] = nt;
						pCfgs->abandoned_targets_size++;
					}
				}
			}
		}
	}
	if(foreImg)
	{
		cvReleaseImage(&foreImg);
		foreImg = NULL;
	}
	if(maskImg)
	{
		cvReleaseImage(&maskImg);
		maskImg = NULL;
	}
	//分析抛弃物跟踪目标
	for(i = 0; i < pCfgs->abandoned_targets_size; i++)
	{
		//检测到，并更新速度
		if(pCfgs->abandoned_targets[i].detected)
		{
			pCfgs->abandoned_targets[i].lost_detected = 0;
			pCfgs->abandoned_targets[i].event_continue_num[event_idx]++;
		}
		else//未检测到
		{
			pCfgs->abandoned_targets[i].lost_detected++;
			pCfgs->abandoned_targets[i].box.x += pCfgs->abandoned_targets[i].vx;
			pCfgs->abandoned_targets[i].box.y += pCfgs->abandoned_targets[i].vy;
			pCfgs->abandoned_targets[i].event_continue_num[event_idx] = 0;
		}
		if(pCfgs->abandoned_targets[i].cal_event[event_idx] == FALSE && pCfgs->abandoned_targets[i].event_continue_num[event_idx] > 10)//抛弃物检测加入到事件中
		{
			pCfgs->abandoned_targets[i].cal_event[event_idx] = TRUE;
			pCfgs->abandoned_targets[i].event_flag[event_idx] = 1;
			/*pCfgs->EventInfo[pCfgs->EventNum].uEventID = pCfgs->abandoned_targets[i].target_id;
			pCfgs->EventInfo[pCfgs->EventNum].begin_time = pCfgs->gThisFrameTime;
			pCfgs->EventInfo[pCfgs->EventNum].type = DROP;
			pCfgs->EventInfo[pCfgs->EventNum].flag = 0;
			pCfgs->EventNum++;
			if(pCfgs->EventState == 0)//事件开始时间
			{
				pCfgs->EventState = 1;
				pCfgs->EventBeginTime = pCfgs->gThisFrameTime;
			}*/
		}
		//printf("%d,[%d,%d,%d,%d]\n",pCfgs->abandoned_targets[i].detected, pCfgs->abandoned_targets[i].box.x,pCfgs->abandoned_targets[i].box.y,pCfgs->abandoned_targets[i].box.width,pCfgs->abandoned_targets[i].box.height);
		//当目标在视频存在时间太长或长时间没有检测到或离开图像，删除目标
		if(pCfgs->abandoned_targets[i].continue_num > 3000 || pCfgs->abandoned_targets[i].lost_detected > 20 ||((pCfgs->abandoned_targets[i].box.x < 10 || pCfgs->abandoned_targets[i].box.y < 10 || (pCfgs->abandoned_targets[i].box.x + pCfgs->abandoned_targets[i].box.width) > (width - 10) || (pCfgs->abandoned_targets[i].box.y + pCfgs->abandoned_targets[i].box.height) > (height - 10))&& pCfgs->abandoned_targets[i].lost_detected > 5))
		{
			/*for(j = 0; j < pCfgs->EventNum; j++)//如果没有设置此事件结束，设置
			{
				if(pCfgs->EventInfo[j].flag == 0)
				{
					if(pCfgs->EventInfo[j].uEventID == pCfgs->abandoned_targets[i].target_id)
					{
						//pCfgs->EventInfo[j].end_time = pCfgs->gThisFrameTime;
						pCfgs->EventInfo[j].flag = 1;
						break;
					}
				}
			}*/
			DeleteTarget(&pCfgs->abandoned_targets_size, &i, pCfgs->abandoned_targets);
			continue;
		}
		pCfgs->abandoned_targets[i].continue_num++;
	}
}
void IllegalParkDetect(ALGCFGS *pCfgs, int target_idx, int event_idx, int targetDisXY[][3], int width, int height)//禁止停车检测
{
	int j = 0;
	int disX = 0, disY = 0;
	int thr = 20;//阈值
	int num = 0;
	/*int continue_num = pCfgs->event_targets[target_idx].trajectory_num - 100;
	continue_num = (continue_num < 0)? 0 : continue_num;
	int dx = pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 1].x - pCfgs->event_targets[target_idx].trajectory[continue_num].x;
	int dy = pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 1].y - pCfgs->event_targets[target_idx].trajectory[continue_num].y;
	dx = (dx < 0)? -dx : dx;
	dy = (dy < 0)? -dy : dy;*/
	//运动距离
	int dx = targetDisXY[target_idx][0];
	int dy = targetDisXY[target_idx][1];
	int id = targetDisXY[target_idx][2];//记录运动时间间隔的位置

	//未标记为停车事件
	if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		thr = pCfgs->event_targets[target_idx].box.height / 2;
		thr = (thr > 15)? 15 : thr;
		thr = (thr < 5)? 5 : thr;
		//if(dx < width / 30 && dy < height / 30 )//静止
		if(dx < width / 30 && dy < thr && ((pCfgs->currTime - pCfgs->uStatFrameTime[id]) > 10 || id == 0))//10s之上才算停车
		{
			//printf("id = %d,dx = [%d,%d,%d],[%d,%d],thr = %d\n", pCfgs->event_targets[target_idx].target_id, dx, dy,id, pCfgs->event_targets[target_idx].box.width, pCfgs->event_targets[target_idx].box.height, thr);
			//printf("dx = %d,dy =%d,time = %d,%f\n",dx, dy, id,pCfgs->currTime - pCfgs->uStatFrameTime[0][id]);
			/*if(pCfgs->event_targets[target_idx].detected)
			{
				pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
			}*/
			if(pCfgs->event_targets[target_idx].lost_detected > 20)//长时间没有检测到，重新计数，防止误检
			{
				pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;
			}
			if(pCfgs->event_targets[target_idx].detected)
			{
				//printf("may be IllegalPark\n");
				num = 0;
				//判断目标周围有无缓慢运行的目标
				for(j = 0; j < pCfgs->event_targets_size; j++)
				{
					if(j == target_idx)
						continue;
					disX = MIN(pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width, pCfgs->event_targets[j].box.x + pCfgs->event_targets[j].box.width) - MAX(pCfgs->event_targets[target_idx].box.x, pCfgs->event_targets[j].box.x);
					disY = MIN(pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height, pCfgs->event_targets[j].box.y + pCfgs->event_targets[j].box.height) - MAX(pCfgs->event_targets[target_idx].box.y, pCfgs->event_targets[j].box.y);
					if(disX > -width / 6 && disY > -height / 6)//10
					{
						thr = pCfgs->event_targets[j].box.height / 4 * 3;
						thr = (thr > 100)? 100 : thr;
						thr = (thr < 10)? 10 : thr;
						//printf("[%d,%d,%d,%d], [%d,%d],thr = %d\n", pCfgs->event_targets[j].target_id, targetDisXY[j][0], targetDisXY[j][1], targetDisXY[j][2], pCfgs->event_targets[j].trajectory[0].y, pCfgs->event_targets[j].box.y + pCfgs->event_targets[j].box.height / 2,thr);
						if(targetDisXY[j][0] < width / 15 && targetDisXY[j][1] < thr)//有缓慢运行的目标
						//if(targetDisXY[j][0] < width / 20 && targetDisXY[j][1] < height / 20)
						{
							num++;
						}
					}
					if(num > 0)
						break;
				}
				//满足条件达到10帧以上，才报停车事件
				if(num < 1)
				{
					pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
				}
				else
				{
					pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;
				}
				//pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
			}
			if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 10)//开始进行停车事件
			{
				printf("illegal park ok\n");
				if(pCfgs->event_targets[target_idx].box.y > 20)//防止误检，图像顶部不认为是停车
				{
					/*if(pCfgs->event_targets[target_idx].target_id == 0)
					pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
					pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
					pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
					//printf("illegal park,%d\n",pCfgs->event_targets[target_idx].target_id);
					/*pCfgs->EventInfo[pCfgs->EventNum].uEventID = pCfgs->event_targets[target_idx].target_id;
					pCfgs->EventInfo[pCfgs->EventNum].begin_time = pCfgs->gThisFrameTime;
					pCfgs->EventInfo[pCfgs->EventNum].type = STOP_INVALID;
					pCfgs->EventInfo[pCfgs->EventNum].flag = 0;
					pCfgs->EventNum++; 
					if(pCfgs->EventState == 0)//事件开始时间
					{
					pCfgs->EventState = 1;
					pCfgs->EventBeginTime = pCfgs->gThisFrameTime - 120;
					}*/
				}

			}
		}
		else//运动
		{
			pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] > 0 && dx < MIN(100, width / 10) && dy < MIN(pCfgs->event_targets[target_idx].box.height * 2, height / 10))//已经标记此类事件，当事件一直存在时，传事件,防止误检，从停车到不停车
	{
		//保存停车事件框
		if(pCfgs->uIllegalParkNum < MAX_EVENT_NUM && pCfgs->event_targets[target_idx].lost_detected < 10)
		{
			//判断是否为新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].uNewEventFlag = 1;
			}
			else
				pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].uNewEventFlag = 0;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->IllegalParkBox[pCfgs->uIllegalParkNum].uEventType = STOP_INVALID;
			pCfgs->uIllegalParkNum++;
		}
	}
	else//车运动后，结束停车
	{
		pCfgs->event_targets[target_idx].event_flag[event_idx] = 0;
		/*for(j = 0; j < pCfgs->EventNum; j++)//如果没有设置此事件结束，设置
		{
			if(pCfgs->EventInfo[j].flag == 0)
			{
				if(pCfgs->EventInfo[j].uEventID == pCfgs->event_targets[target_idx].target_id)
				{
					//pCfgs->EventInfo[j].end_time = pCfgs->gThisFrameTime;
					pCfgs->EventInfo[j].flag = 1;
					break;
				}
			}
		}*/
	}
}
void OppositeDirDriveDetect(ALGCFGS *pCfgs, int target_idx, int event_idx, int InRegionVal, int width, int height)//逆行检测
{
#if 1
	int i = 0, j = 0;
	int dis_x = 0;//距离间隔
	int dis_y = 0;
	int continue_num = 0;
	int startIdx = 0;
	int startIdy = 0;
	int dis_sum_x = 0;
	int dis_sum_y = 0;
	int inverse_type = 0;
	bool inverse_ok = FALSE;
	bool inverse_off = FALSE;
	if(pCfgs->event_targets[target_idx].trajectory_num < 5)//车辆运行小于5帧不进行计算
		return;
	if(pCfgs->event_targets[target_idx].trajectory_num % 5 == 2)//每间隔5帧进行距离计算
	{
		continue_num = pCfgs->event_targets[target_idx].trajectory_num - 1;
		i = continue_num / 5;
		dis_x = (pCfgs->event_targets[target_idx].trajectory[continue_num].x + pCfgs->event_targets[target_idx].trajectory[continue_num - 1].x - pCfgs->event_targets[target_idx].trajectory[continue_num - 5].x - pCfgs->event_targets[target_idx].trajectory[continue_num - 6].x) / 2;
	    dis_y = (pCfgs->event_targets[target_idx].trajectory[continue_num].y + pCfgs->event_targets[target_idx].trajectory[continue_num - 1].y - pCfgs->event_targets[target_idx].trajectory[continue_num - 5].y - pCfgs->event_targets[target_idx].trajectory[continue_num - 6].y) / 2;
		startIdx = pCfgs->event_targets[target_idx].statistic[0];
		startIdy = pCfgs->event_targets[target_idx].statistic[1];
		dis_sum_x = pCfgs->event_targets[target_idx].statistic[2];
		dis_sum_y = pCfgs->event_targets[target_idx].statistic[3];
		inverse_type = pCfgs->event_targets[target_idx].statistic[4];

		/*//按照x方向运动计算
		if(startIdx < 0)//判断车俩是否与设定的方向相反，如果相反就开始进行逆行判断
		{
			if(dis_x < 0 && InRegionVal == 1)//区域方向右行
			{
				startIdx = i;
				dis_sum_x = dis_x;
			}
			if(dis_x > 0 && InRegionVal == 2)//区域方向左行
			{
				startIdx = i;
				dis_sum_x = dis_x;
			}
		}
		else if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)//开始逆行判断
		{
			dis_sum_x += dis_x;
			if((dis_sum_x < MIN(-width / 10, -pCfgs->event_targets[target_idx].box.width) && InRegionVal == 1) || (dis_sum_x >= MAX(width / 10, pCfgs->event_targets[target_idx].box.width) && InRegionVal == 2))//达到逆行条件
			{
				if(pCfgs->event_targets[target_idx].detected)
				{
					pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
					printf("x ok\n");
				}
			}
			else
			{
				pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;//不满足逆行条件，设置为0
				if((dis_sum_x >= 0 && InRegionVal == 1) || (dis_sum_x <= 0 && InRegionVal == 2))//车辆运行方向正确，则再重新进行计算
				{
					startIdx = -1;
				}
			}
			if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] >= 2)//连续7*2帧，满足条件，则认为是逆行
			{
				inverse_type = 1;
				inverse_ok = TRUE;
				dis_sum_x = 0;
				printf("x inverse ok\n");
			}

		}
		else if(pCfgs->event_targets[target_idx].event_flag[event_idx] > 0 && inverse_type == 1)//已经判断逆行，判断是否结束逆行
		{
			if(startIdx < 1000)
			{
				if((dis_x > 0 && InRegionVal == 1) || (dis_x < 0 && InRegionVal == 2))//行驶方向正确后，重新进行统计，是否结束逆行
				{
					dis_sum_x = 0;
					startIdx = 1000;
				}
			}
			else
			{
				dis_sum_x += dis_x;
			}
			if((dis_sum_x >= width / 50 && InRegionVal == 1) || (dis_sum_x <= -width / 50 && InRegionVal == 2))
			{
				inverse_off = TRUE;
			}
		}*/

		//按照y方向运动进行计算
		if(startIdy < 0)//判断车俩是否与设定的方向相反，如果相反就开始进行逆行判断
		{
			if(dis_y < 0 && InRegionVal == 1)//区域方向下行
			{
				startIdy = i;
				dis_sum_y = dis_y;
			}
			if(dis_y > 0 && InRegionVal == 2)//区域方向上行
			{
				startIdy = i;
				dis_sum_y = dis_y;
			}
			//printf("idy =%d, %d, %d\n",startIdy,dis_sum_y,InRegionVal);
		}
		else if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)//开始逆行判断
		{
			dis_sum_y += dis_y;
			if((dis_sum_y < MIN(-height / 6, -pCfgs->event_targets[target_idx].box.height) && InRegionVal == 1) || (dis_sum_y >= MAX(height / 6, pCfgs->event_targets[target_idx].box.height) && InRegionVal == 2))//达到逆行条件
			{
				printf("inverse id = %d, dis_y = %d, dis_sum_y = %d\n", pCfgs->event_targets[target_idx].target_id, dis_y,dis_sum_y);
				//车辆在图像边缘，不计入事件
				int top = pCfgs->event_targets[target_idx].box.y;
				int bottom = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
				if((bottom < (height - 10) && InRegionVal == 1) || (top > 10 && InRegionVal == 2))
				{
					if(pCfgs->event_targets[target_idx].detected)
					{
						pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
						//printf("y ok\n");
					}
				}
			}
			else 
			{
				pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;//不满足逆行条件，设置为0
				if((dis_sum_y >= 0 && InRegionVal == 1) || (dis_sum_y <= 0 && InRegionVal == 2))//车辆运行方向正确，则再重新进行计算
				{
					startIdy = -1;
				}
			}
			if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] >= 2)//连续7*2帧，满足条件，则认为是逆行
			{
				inverse_type = 2;
				inverse_ok = TRUE;
				dis_sum_y = 0;
				//printf("y inverse ok\n");
			}

		}
		else if(pCfgs->event_targets[target_idx].event_flag[event_idx] > 0 && inverse_type == 2)//已经判断逆行，判断是否结束逆行
		{
			if(startIdy < 1000)
			{
				if((dis_y > 0 && InRegionVal == 1) || (dis_y < 0 && InRegionVal == 2))//行驶方向正确后，重新进行统计，是否结束逆行
				{
					dis_sum_y = 0;
					startIdy = 1000;
				}
			}
			else
			{
				dis_sum_y += dis_y;
			}
			if((dis_sum_y >= width / 50 && InRegionVal == 1) || (dis_sum_y <= -width / 50 && InRegionVal == 2))
			{
				inverse_off = TRUE;
			}
		}


		//将计算数据保存起来
		pCfgs->event_targets[target_idx].statistic[0] = startIdx;
		pCfgs->event_targets[target_idx].statistic[1] = startIdy;
		pCfgs->event_targets[target_idx].statistic[2] = dis_sum_x;
		pCfgs->event_targets[target_idx].statistic[3] = dis_sum_y;
		pCfgs->event_targets[target_idx].statistic[4] = inverse_type;
		/*for(j = 0; j < pCfgs->event_targets[target_idx].trajectory_num;j++)
		{
			printf("%d ",pCfgs->event_targets[target_idx].trajectory[j].y);
		}
		printf("\n");
		printf("target_idx= %d,[%d,%d,%d,%d]\n",pCfgs->event_targets[target_idx].target_id,pCfgs->event_targets[target_idx].statistic[0], pCfgs->event_targets[target_idx].statistic[1],pCfgs->event_targets[target_idx].statistic[2],pCfgs->event_targets[target_idx].statistic[3]);*/
		
	}
	//此目标没有被标记为逆行事件
	if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		if(inverse_ok == TRUE)//逆行
		{
			printf("id = %d, OppositeDirDrive\n", pCfgs->event_targets[target_idx].target_id);
			/*if(pCfgs->event_targets[target_idx].target_id == 0)
			pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
			pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
			pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
			/*pCfgs->EventInfo[pCfgs->EventNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->EventInfo[pCfgs->EventNum].begin_time = pCfgs->gThisFrameTime;
			pCfgs->EventInfo[pCfgs->EventNum].type = REVERSE_DRIVE;
			pCfgs->EventInfo[pCfgs->EventNum].flag = 0;
			pCfgs->EventNum++;
			if(pCfgs->EventState == 0)//事件开始时间
			{
				pCfgs->EventState = 1;
				pCfgs->EventBeginTime = pCfgs->gThisFrameTime - 10;

			}*/
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] > 0 && inverse_off == FALSE)//已经标记此类事件，当事件一直存在时，传事件
	{
		//printf("uOppositeDirDriveNum ,%d\n",pCfgs->event_targets[target_idx].target_id);
		//保存停车事件框
		if(pCfgs->uOppositeDirDriveNum < MAX_EVENT_NUM)
		{
			//判断是否为新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uNewEventFlag = 1;
			}
			else
				pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uNewEventFlag = 0;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uEventType = REVERSE_DRIVE;
			pCfgs->uOppositeDirDriveNum++;
		}
	}
	else//车辆不再逆行
	{
		pCfgs->event_targets[target_idx].event_flag[event_idx] = 0;
		/*for(j = 0; j < pCfgs->EventNum; j++)//如果没有设置此事件结束，设置
		{
			if(pCfgs->EventInfo[j].flag == 0)
			{
				if(pCfgs->EventInfo[j].uEventID == pCfgs->event_targets[target_idx].target_id)
				{
					//pCfgs->EventInfo[j].end_time = pCfgs->gThisFrameTime;
					pCfgs->EventInfo[j].flag = 1;
					break;
				}
			}
		}*/
	}
#else
	int j = 0;
	//得到运动距离
	int continue_num = pCfgs->event_targets[target_idx].trajectory_num - 200;//防止车辆运行很慢，间隔帧数为200
	continue_num = (continue_num < 0)? 0 : continue_num;
	int dx = 0;
	int dy = 0;
	if(pCfgs->event_targets[target_idx].trajectory_num == 1)
	{
		dx = pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 1].x - pCfgs->event_targets[target_idx].trajectory[continue_num].x;
		dy = pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 1].y - pCfgs->event_targets[target_idx].trajectory[continue_num].y;
	}
	else
	{
		dx = pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 1].x + pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 2].x - pCfgs->event_targets[target_idx].trajectory[continue_num + 1].x - pCfgs->event_targets[target_idx].trajectory[continue_num].x ;
		dy = pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 1].y + pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 2].y - pCfgs->event_targets[target_idx].trajectory[continue_num + 1].y - pCfgs->event_targets[target_idx].trajectory[continue_num].y;
		dx = dx / 2;
		dy = dy / 2;
	}
	//此目标没有被标记为逆行事件
	if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		if((dy > height / 8 && InRegionVal == 2) || (dy < -height / 8 && InRegionVal == 1))//车辆下行，但区域方向为上行，车辆上行，但区域方向下行
		{
			//printf("may be OppositeDirDrive\n");
			if(pCfgs->event_targets[target_idx].detected)
			{
				pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
			}
			if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 10 && pCfgs->event_targets[target_idx].detected)//当检测目标达到一定帧数时，才认为是事件
			{
				//printf("OppositeDirDrive\n");
				/*if(pCfgs->event_targets[target_idx].target_id == 0)
				pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
				pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
				pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
				/*pCfgs->EventInfo[pCfgs->EventNum].uEventID = pCfgs->event_targets[target_idx].target_id;
				pCfgs->EventInfo[pCfgs->EventNum].begin_time = pCfgs->gThisFrameTime;
				pCfgs->EventInfo[pCfgs->EventNum].type = REVERSE_DRIVE;
				pCfgs->EventInfo[pCfgs->EventNum].flag = 0;
				pCfgs->EventNum++;
				if(pCfgs->EventState == 0)//事件开始时间
				{
					pCfgs->EventState = 1;
					pCfgs->EventBeginTime = pCfgs->gThisFrameTime - 10;

				}*/
			}
		}
		else//不满足条件，设置为0
		{
			pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] > 0 && ((dy > height / 50 && InRegionVal == 2) || (dy < -height / 50 && InRegionVal == 1)))//已经标记此类事件，当事件一直存在时，传事件
	{
		//printf("uOppositeDirDriveNum ,%d\n",pCfgs->event_targets[target_idx].target_id);
		//保存逆行事件框
		if(pCfgs->uOppositeDirDriveNum < MAX_EVENT_NUM)
		{
			//判断是否为新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uNewEventFlag = 1;
			}
			else
				pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uNewEventFlag = 0;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->OppositeDirDriveBox[pCfgs->uOppositeDirDriveNum].uEventType = REVERSE_DRIVE;
			pCfgs->uOppositeDirDriveNum++;
		}
	}
	else//车辆不再逆行
	{
		pCfgs->event_targets[target_idx].event_flag[event_idx] = 0;
		/*for(j = 0; j < pCfgs->EventNum; j++)//如果没有设置此事件结束，设置
		{
			if(pCfgs->EventInfo[j].flag == 0)
			{
				if(pCfgs->EventInfo[j].uEventID == pCfgs->event_targets[target_idx].target_id)
				{
					//pCfgs->EventInfo[j].end_time = pCfgs->gThisFrameTime;
					pCfgs->EventInfo[j].flag = 1;
					break;
				}
			}
		}*/
	}
#endif
}

void OffLaneDetect(ALGCFGS *pCfgs, int target_idx, int event_idx)//车辆驶离检测
{
	//此目标没有被标记为驶离事件
	if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		if(pCfgs->event_targets[target_idx].detected)
		{
			pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
		}
		if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 10 && pCfgs->event_targets[target_idx].detected)//目标在驶离区域，检测达到一定帧数
		{
			//printf("OffLane\n");
			/*if(pCfgs->event_targets[target_idx].target_id == 0)
			pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
			pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
			pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] > 0)//已经标记此类事件，当事件一直存在时，传事件
	{
		//保存驶离事件框
		if(pCfgs->uOffLaneNum < MAX_EVENT_NUM)
		{
			//判断是否是新出现事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->OffLaneBox[pCfgs->uOffLaneNum].uNewEventFlag = 1;
			}
			else
				pCfgs->OffLaneBox[pCfgs->uOffLaneNum].uNewEventFlag = 0;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->OffLaneBox[pCfgs->uOffLaneNum].uEventType = DRIVE_AWAY;
			pCfgs->uOffLaneNum++;
		}
	}
}
void NoPersonAllowDetect(ALGCFGS *pCfgs, int target_idx, int event_idx, int width, int height)//禁止行人检测
{
	//已经标记为非机动车事件，则不进行行人事件判断
	if(pCfgs->event_targets[target_idx].cal_event[NONMOTOR])
		return;
	//未标记为行人事件
	if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		if(pCfgs->event_targets[target_idx].continue_num < 20 && (pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height) > height - 20)
			return;
		//如果行人目标和非机动车目标相交，则认为是非机动车上的行人
		for(int i = 0; i < pCfgs->event_targets_size; i++)
		{
			if(strcmp(pCfgs->event_targets[i].names, "motorbike") != 0 && strcmp(pCfgs->event_targets[i].names, "bicycle") != 0)//只与非机动车进行比较
				continue;
			if(overlapRatio(pCfgs->event_targets[i].box, pCfgs->event_targets[target_idx].box) > 5)//如果相交，退出，不进行后续判断
				return;
		}
		if(pCfgs->event_targets[target_idx].detected && pCfgs->event_targets[target_idx].prob >= 0.5)//未防止误检，将置信度设为0.5
		{
			pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;//发生事件的持续帧数
		}
		if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 15 && pCfgs->event_targets[target_idx].detected)//大于一定帧数才认为禁止行人，防止行人误检
		{
			//printf("no person allow ok\n");
			/*if(pCfgs->event_targets[target_idx].target_id == 0)
			pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
			pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
			pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
			/*pCfgs->EventInfo[pCfgs->EventNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->EventInfo[pCfgs->EventNum].begin_time = pCfgs->gThisFrameTime;
			pCfgs->EventInfo[pCfgs->EventNum].type = NO_PEDESTRIANTION;
			pCfgs->EventInfo[pCfgs->EventNum].flag = 0;
			pCfgs->EventNum++;
			if(pCfgs->EventState == 0)//事件开始时间
			{
				pCfgs->EventState = 1;
				pCfgs->EventBeginTime = pCfgs->gThisFrameTime;
			}*/
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx])//已经标记此类事件，当事件一直存在时，传事件
	{
		//保存行人事件框
		if(pCfgs->uNoPersonAllowNum < MAX_EVENT_NUM)
		{
			//判断是否为新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].uNewEventFlag = 1;
			}
			else
				pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].uNewEventFlag = 0;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->NoPersonAllowBox[pCfgs->uNoPersonAllowNum].uEventType = NO_PEDESTRIANTION;
			pCfgs->uNoPersonAllowNum++;
		}
	}
}
void NonMotorAllowDetect(ALGCFGS *pCfgs, int target_idx, int event_idx)//禁止非机动车检测
{
	//已经标记为非机动车事件，则不进行行人事件判断
	if(pCfgs->event_targets[target_idx].cal_event[NO_PEDESTRIANTION])
		return;
	//未标记为非机动车事件
	if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		//如果行人目标和非机动车目标相交，则已经检测到行人事件，则不进行非机动车事件判断
		for(int i = 0; i < pCfgs->event_targets_size; i++)
		{
			if(strcmp(pCfgs->event_targets[i].names, "person") != 0)//只与行人进行比较
				continue;
			if(pCfgs->event_targets[i].cal_event[NO_PEDESTRIANTION] == FALSE)
				continue;
			if(overlapRatio(pCfgs->event_targets[i].box, pCfgs->event_targets[target_idx].box) > 0)//如果相交，退出，不进行后续判断
				return;
		}
		if(pCfgs->event_targets[target_idx].detected && pCfgs->event_targets[target_idx].prob >= 0.5)//检测
		{
			pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;//发生事件的持续帧数
		}
		if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 10 && pCfgs->event_targets[target_idx].detected)//大于一定帧数才认为禁止非机动车，防止非机动车误检
		{
			/*if(pCfgs->event_targets[target_idx].target_id == 0)
			pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
			pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
			pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx])//已经标记此类事件，当事件一直存在时，传事件
	{
		//保存非机动车事件框
		if(pCfgs->uNonMotorAllowNum < MAX_EVENT_NUM)
		{
			//判断是否是新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].uNewEventFlag = 1;
			}
			else
				pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].uNewEventFlag = 0;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->NonMotorAllowBox[pCfgs->uNonMotorAllowNum].uEventType = NONMOTOR;
			pCfgs->uNonMotorAllowNum++;
		}
	}
}
void PersonFallDetect(ALGCFGS *pCfgs, int target_idx, int event_idx, int width, int height)//行人跌倒检测
{
	int j = 0;
	//未标记为行人跌倒事件
	float ratio = (float) pCfgs->event_targets[target_idx].box.height / (float) pCfgs->event_targets[target_idx].box.width;//当前帧的长宽比  
	//printf("height and width =%f\n",ratio);
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] == 0)
	{
		bool inDetRegion = FALSE;//用于判断行人是否在检测区域内部
		if(pCfgs->event_targets[target_idx].box.x > 10 && pCfgs->event_targets[target_idx].box.y > 10 && (pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width) < width - 10 && (pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height) < height - 10)//在检测区域内部
			inDetRegion = TRUE;
		if(inDetRegion)
		{  
			float ratioMax = 0;
			int personStand = 0;
			for(j = pCfgs->event_targets[target_idx].trajectory_num - 1; j  >= 0; j--)
			{
				float ratio0 = (float) pCfgs->event_targets[target_idx].trajectory[j].height / pCfgs->event_targets[target_idx].trajectory[j].width;
				ratioMax = (ratioMax < ratio0)? ratio0 : ratioMax;
				if(ratio0 > 2.5) //行人站立
					personStand++;
				else if(ratio0 < 2)
					personStand = 0;
				if(personStand > 5)
					break;
			} 
			if(personStand > 5 || ratio < 0.8)//行人站立超过5帧
			{
				//printf("detected =%d,%f,%d\n",pCfgs->event_targets[target_idx].detected,pCfgs->event_targets[target_idx].prob,pCfgs->event_targets[target_idx].event_continue_num[event_idx]);
				if(pCfgs->event_targets[target_idx].detected && pCfgs->event_targets[target_idx].prob >= 0.5 && ((ratioMax / ratio > 2 && ratio < 1.2)|| ratio < 0.8))//未防止误检，将置信度设为0.5
				{
					pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;//发生事件的持续帧数
				}

				if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 20 && pCfgs->event_targets[target_idx].detected)//大于一定帧数才认为行人跌倒
				{
					if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
					{
						/*if(pCfgs->event_targets[target_idx].target_id == 0)
							pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
						pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
					}
					pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
				}
			}
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx])//已经标记此类事件，当事件一直存在时，传事件
	{
		//printf("person down \n");
		//保存行人跌倒事件框
		if(pCfgs->uPersonFallNum < MAX_EVENT_NUM)
		{
			//判断是否是新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->PersonFallBox[pCfgs->uPersonFallNum].uNewEventFlag = 1;
				//printf("new person fall event\n");
			}
			else
				pCfgs->PersonFallBox[pCfgs->uPersonFallNum].uNewEventFlag = 0;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->PersonFallBox[pCfgs->uPersonFallNum].uEventType = PERSONFALL;
			pCfgs->uPersonFallNum++;
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] && ratio > 2)//结束行人跌倒
	{
		pCfgs->event_targets[target_idx].event_flag[event_idx] = 0;
		pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;
	}
}
void NonMotorFallDetect(ALGCFGS *pCfgs, int target_idx, int event_idx, int width, int height)//非机动车跌倒
{
	int j = 0;
	//未标记为非机动车事件
	float ratio = (float) pCfgs->event_targets[target_idx].box.height / (float) pCfgs->event_targets[target_idx].box.width;//当前帧的长宽比  
	//printf("height and width =%f\n",ratio);
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] == 0)
	{
		bool inDetRegion = FALSE;//用于判断非机动车是否在检测区域内部
		if(pCfgs->event_targets[target_idx].box.x > 10 && pCfgs->event_targets[target_idx].box.y > 10 && (pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width) < width - 10 && (pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height) < height - 10)//在检测区域内部
			inDetRegion = TRUE;
		if(inDetRegion)
		{  
			float ratioMax = 0;
			int NonMontorStand = 0;
			for(j = pCfgs->event_targets[target_idx].trajectory_num - 1; j  >= 0; j--)
			{
				float ratio0 = (float) pCfgs->event_targets[target_idx].trajectory[j].height / pCfgs->event_targets[target_idx].trajectory[j].width;
				ratioMax = (ratioMax < ratio0)? ratio0 : ratioMax;
				if(ratio0 > 2.5) //非机动车站立
					NonMontorStand++;
				else if(ratio0 < 2)
					NonMontorStand = 0;
				if(NonMontorStand > 5)
					break;
			} 
			if(NonMontorStand > 5 || ratio < 0.8)//非机动车站立超过5帧
			{
				//printf("detected =%d,%f,%d\n",pCfgs->event_targets[target_idx].detected,pCfgs->event_targets[target_idx].prob,pCfgs->event_targets[target_idx].event_continue_num[event_idx]);
				if(pCfgs->event_targets[target_idx].detected && pCfgs->event_targets[target_idx].prob >= 0.5 && ((ratioMax / ratio > 2 && ratio < 1.2)|| ratio < 0.8))//未防止误检，将置信度设为0.5
				{
					pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;//发生事件的持续帧数
				}

				if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 20 && pCfgs->event_targets[target_idx].detected)//大于一定帧数才认为非机动车跌倒
				{
					if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
					{
						/*if(pCfgs->event_targets[target_idx].target_id == 0)
							pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
						pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
					}
					pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
				}
			}
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx])//已经标记此类事件，当事件一直存在时，传事件
	{
		//printf("person down \n");
		//保存非机动车跌倒事件框
		if(pCfgs->uNonMotorFallNum < MAX_EVENT_NUM)
		{
			//判断是否是新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].uNewEventFlag = 1;
			}
			else
				pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].uNewEventFlag = 0;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->NonMotorFallBox[pCfgs->uNonMotorFallNum].uEventType = NONMOTORFALL;
			pCfgs->uNonMotorFallNum++;
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] && ratio > 2)//结束行人跌倒
	{
		pCfgs->event_targets[target_idx].event_flag[event_idx] = 0;
		pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;
	}
}
void GreenwayDropDetect(ALGCFGS *pCfgs, int target_idx, int event_idx)//纸杯或瓶子
{
	//未标记为绿道抛弃物
	if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		if(pCfgs->event_targets[target_idx].detected && pCfgs->event_targets[target_idx].prob >= 0.5)//检测
		{
			pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;//发生事件的持续帧数
		}
		if(pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 10 && pCfgs->event_targets[target_idx].detected)//大于一定帧数才认为抛弃物，防止误检
		{
			/*if(pCfgs->event_targets[target_idx].target_id == 0)
			pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
			pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
			pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx])//已经标记此类事件，当事件一直存在时，传事件
	{
		//保存绿道抛弃物事件框
		if(pCfgs->uGreenwayDropNum < MAX_EVENT_NUM)
		{
			//判断是否是新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].uNewEventFlag = 1;
			}
			else
				pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].uNewEventFlag = 0;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->GreenwayDropBox[pCfgs->uGreenwayDropNum].uEventType = GREENWAYDROP;
			pCfgs->uGreenwayDropNum++;
		}
	}
}
void TrafficAccidentDetect(ALGCFGS *pCfgs, int target_idx, int event_idx, int targetDisXY[][3], int width, int height)//交通事故检测
{
	int i = 0, j = 0, k = 0;
	int thr = 20;//阈值
	int disX = 0, disY = 0;
	int num = 0;
	/*int continue_num = pCfgs->event_targets[target_idx].trajectory_num - 100;
	continue_num = (continue_num < 0)? 0 : continue_num;
	int dx = pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 1].x - pCfgs->event_targets[target_idx].trajectory[continue_num].x;
	int dy = pCfgs->event_targets[target_idx].trajectory[pCfgs->event_targets[target_idx].trajectory_num - 1].y - pCfgs->event_targets[target_idx].trajectory[continue_num].y;
	dx = (dx < 0)? -dx : dx;
	dy = (dy < 0)? -dy : dy;*/
	int dx = targetDisXY[target_idx][0];
	int dy = targetDisXY[target_idx][1];
	int idx = target_idx;
	//未标记为事故事件
	if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		thr = pCfgs->event_targets[target_idx].box.height / 2;
		thr = (thr > 200)? 200 : thr;
		thr = (thr < 5)? 5 : thr;
		if(dx < width / 20 && dy < thr )//车辆相对静止，周围有车相对静止，并且附近有人，则认为是交通事件
		{
			/*if(pCfgs->event_targets[target_idx].detected)
			{
				pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
			}*/
			if(pCfgs->event_targets[target_idx].lost_detected > 20)//长时间没有检测到，重新计数，防止误检
			{
				pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;
			}
			if(/*pCfgs->event_targets[target_idx].event_continue_num[event_idx] > 20 &&*/ pCfgs->event_targets[target_idx].detected)//停车帧数达到20帧，即认为是停车
			{
				bool hasPerson = FALSE;
				//判断跟踪框周围是否有相对静止的目标
				for(j = 0; j < pCfgs->event_targets_size; j++)
				{
					if( j == target_idx || pCfgs->event_targets[j].cal_event[event_idx] || pCfgs->event_targets[j].continue_num < 50 || pCfgs->event_targets[j].lost_detected)
						continue;
					disX = MIN(pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width,pCfgs->event_targets[j].box.x + pCfgs->event_targets[j].box.width)- MAX(pCfgs->event_targets[target_idx].box.x, pCfgs->event_targets[j].box.x);
					disY = MIN(pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height,pCfgs->event_targets[j].box.y + pCfgs->event_targets[j].box.height)- MAX(pCfgs->event_targets[target_idx].box.y, pCfgs->event_targets[j].box.y);
					//if(disX > -width / 10 && disY > -height / 10)
					if(disX >= 0 && disY >= 0)//两辆车相交
					{
						thr = pCfgs->event_targets[j].box.height / 2;
						thr = (thr > 200)? 200 : thr;
						thr = (thr < 5)? 5 : thr;
						if(targetDisXY[j][0] < width / 20 && targetDisXY[j][1] < thr)
						{
							num++;
						}
					}
					if(num > 0)
					{
						/*if(pCfgs->event_targets[j].box.width > pCfgs->event_targets[target_idx].box.width || pCfgs->event_targets[j].box.height > pCfgs->event_targets[target_idx].box.height)
							idx = target_idx;
						else
							idx = j;*/
						//扩展区域
						CRect rct;
						rct.x = MIN(pCfgs->event_targets[target_idx].box.x, pCfgs->event_targets[j].box.x) - 30;
						rct.y = MIN(pCfgs->event_targets[target_idx].box.y, pCfgs->event_targets[j].box.y) - 30;
						rct.width = MAX(pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width,pCfgs->event_targets[j].box.x + pCfgs->event_targets[j].box.width) - MIN(pCfgs->event_targets[target_idx].box.x, pCfgs->event_targets[j].box.x) + 60;
						rct.height = MAX(pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height,pCfgs->event_targets[j].box.y + pCfgs->event_targets[j].box.height) - MIN(pCfgs->event_targets[target_idx].box.y, pCfgs->event_targets[j].box.y) + 60;
						//判断跟踪框周围是否有行人
						for(i = 0; i < pCfgs->classes; i++)
						{
							if(strcmp(pCfgs->detClasses[i].names, "person") != 0)
								continue;
							for( k = 0; k < pCfgs->detClasses[i].classes_num; k++)
							{
								if(pCfgs->detClasses[i].box[k].width * 2  > pCfgs->detClasses[i].box[k].height)//车里的人除外
									continue;
								if(overlapRatio(pCfgs->detClasses[i].box[k], rct) > 2)
								{
									hasPerson = TRUE;
									break;
								}
							}
							if(hasPerson)
								break;
						}
						break;
					}
				}
				//存在行人，并且周围也有相对静止的目标
				if(hasPerson && num)
				{
					pCfgs->event_targets[target_idx].event_continue_num[event_idx]++;
				}
				if(pCfgs->event_targets[target_idx].cal_event[event_idx] == FALSE && pCfgs->event_targets[target_idx].event_continue_num[event_idx]> 10)//开始进行事故事件
				{
					/*if(pCfgs->event_targets[target_idx].target_id == 0)
						pCfgs->event_targets[target_idx].target_id = pCfgs->eventID++;//给每个事件一个ID*/
					pCfgs->event_targets[target_idx].cal_event[event_idx] = TRUE;
					pCfgs->event_targets[target_idx].event_flag[event_idx] = 1;
				}
			}
		}
		else//车辆运动
		{
			pCfgs->event_targets[target_idx].event_continue_num[event_idx] = 0;
		}
	}
	if(pCfgs->event_targets[target_idx].event_flag[event_idx] == 1 && dx < width / 10 && dy < height / 5)//已经标记此类事件，当事件一直存在时，传事件,防止误检，车辆离开
	{
		/*for(k = 0; k < pCfgs->uTrafficAccidentNum; k++)
		{
			if(pCfgs->TrafficAccidentBox[k].uEventID == pCfgs->event_targets[target_idx].target_id)
			{
				break;
			}
		}*/
		if(pCfgs->uTrafficAccidentNum < MAX_EVENT_NUM /*&& k == pCfgs->uTrafficAccidentNum*/)//之前没有此ID事件框
		{
			//将周围静止的目标设为已检测到交通事故，防止误报
			for(j = 0; j < pCfgs->event_targets_size; j++)
			{
				if( j == target_idx)
					continue;
				if(overlapRatio(pCfgs->event_targets[j].box,pCfgs->event_targets[target_idx].box) > 0)
				{
					thr = pCfgs->event_targets[j].box.height;
					thr = (thr > 200)? 200 : thr;
					thr = (thr < 5)? 5 : thr;
					if(targetDisXY[j][0] < width / 10 && targetDisXY[j][1] < thr)
					{
						pCfgs->event_targets[j].event_flag[event_idx] = pCfgs->event_targets[target_idx].target_id;//标记为车辆事故的ID
						pCfgs->event_targets[j].cal_event[event_idx] = TRUE;
						//pCfgs->event_targets[j].target_id = pCfgs->event_targets[target_idx].target_id;//将此目标的ID设置为相应交通事故的ID

					}
				}
			}

			//判断是否为新出现的事件
			if(pCfgs->event_targets[target_idx].sign_event[event_idx] == 0)
			{
				pCfgs->event_targets[target_idx].sign_event[event_idx] = 1;
				pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].uNewEventFlag = 1;
			}
			else
				pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].uNewEventFlag = 0;

			//保存交通事故事件框
			int minX = pCfgs->event_targets[target_idx].box.x;
			int maxX = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			int minY = pCfgs->event_targets[target_idx].box.y;
			int maxY = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].uRegionID = pCfgs->event_targets[target_idx].region_idx;//事件区域ID
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].uEventID = pCfgs->event_targets[target_idx].target_id;
			/*pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[0].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[0].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[1].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[1].y = pCfgs->event_targets[target_idx].box.y;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[2].x = pCfgs->event_targets[target_idx].box.x + pCfgs->event_targets[target_idx].box.width;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[2].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[3].x = pCfgs->event_targets[target_idx].box.x;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[3].y = pCfgs->event_targets[target_idx].box.y + pCfgs->event_targets[target_idx].box.height;*/
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[0].x = minX;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[0].y = minY;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[1].x = maxX;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[1].y = minY;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[2].x = maxX;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[2].y = maxY;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[3].x = minX;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].EventBox[3].y = maxY;
			pCfgs->TrafficAccidentBox[pCfgs->uTrafficAccidentNum].uEventType = ACCIDENTTRAFFIC;
			pCfgs->uTrafficAccidentNum++;
		}
	}
	else//车运动后，结束事故
	{
		pCfgs->event_targets[target_idx].event_flag[event_idx] = 0;
	}
}
//////////////////////////////////////////////////////////////////////////////交通事件检测
void TrafficEventAnalysis(ALGCFGS *pCfgs, ALGPARAMS *pParams, int width, int height)
{	
	int i = 0, j = 0, k = 0, l = 0;
	int left = 0, right = 0, top = 0, bottom = 0;
	//int match_object[MAX_TARGET_NUM] = { -1 };
	//int match_rect[MAX_TARGET_NUM] = { -1 };
	int match_object[MAX_CLASSES][MAX_DETECTION_NUM];
	int match_rect[MAX_TARGET_NUM][3];
	int match_success = -1;
	bool isInRegion = FALSE;
	CRect targetRect[MAX_TARGET_NUM];
	int  targetInRegion[MAX_TARGET_NUM][MAX_EVENT_TYPE]={ 0 };
	int targetDisXY[MAX_TARGET_NUM][3]={ 0 };
	bool isCycle = FALSE;
	//初始化交通事件检测结果
	pCfgs->uIllegalParkNum = 0;//违法停车
	memset(pCfgs->IllegalParkBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uOppositeDirDriveNum = 0;//逆行
	memset(pCfgs->OppositeDirDriveBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uOffLaneNum = 0;//驶离
	memset(pCfgs->OffLaneBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uNoPersonAllowNum = 0;//禁止行人
	memset(pCfgs->NoPersonAllowBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uDropNum = 0;//抛弃物
	memset(pCfgs->DropBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uNonMotorAllowNum = 0;//禁止非机动车
	memset(pCfgs->NonMotorAllowBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uPersonFallNum = 0;//行人跌倒
	memset(pCfgs->PersonFallBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uNonMotorFallNum = 0;//非机动车倒地
	memset(pCfgs->NonMotorFallBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uGreenwayDropNum = 0;//绿道抛弃物
	memset(pCfgs->GreenwayDropBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uTrafficAccidentNum = 0;//交通事故
	memset(pCfgs->TrafficAccidentBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	//设置目标为未检测到
	for( i = 0; i < pCfgs->event_targets_size; i++)
	{
		pCfgs->event_targets[i].detected = FALSE;
	}
	//检测框和目标框进行匹配
	match_object_rect1(pCfgs, pCfgs->event_targets, pCfgs->event_targets_size, match_object, match_rect, 10);
	//分析检测框
	for( i = 0; i < pCfgs->classes; i++)
	{
		//处理需要的类别
		if(strcmp(pCfgs->detClasses[i].names, "person") != 0 && strcmp(pCfgs->detClasses[i].names, "bicycle") != 0 && strcmp(pCfgs->detClasses[i].names, "motorbike") != 0 && strcmp(pCfgs->detClasses[i].names, "car") != 0 && strcmp(pCfgs->detClasses[i].names, "bus") != 0 && strcmp(pCfgs->detClasses[i].names, "truck") != 0 && strcmp(pCfgs->detClasses[i].names,"cup") != 0 && strcmp(pCfgs->detClasses[i].names,"bottle") != 0)
			continue;
		if(pCfgs->detClasses[i].classes_num)
		{
			//match_object_rect(pCfgs->event_targets, pCfgs->event_targets_size, pCfgs->detClasses, i, match_object, match_rect, 10);

			for( j = 0; j < pCfgs->detClasses[i].classes_num; j++)
			{
				//置信度低于0.5不处理
				//if(pCfgs->detClasses[i].prob[j] < 0.5)
				//	continue;
				/*if(strcmp(pCfgs->detClasses[i].names, "person") == 0)//判断是否骑非机动车的人
				{
					isCycle = FALSE;
					for( k = 0; k < pCfgs->classes; k++)
					{
						if(strcmp(pCfgs->detClasses[k].names, "bicycle") != 0 && strcmp(pCfgs->detClasses[k].names, "motorbike") != 0)//非机动车
							continue;
						for(l = 0; l < pCfgs->detClasses[k].classes_num; l++)
						{
							if(overlapRatio(pCfgs->detClasses[i].box[j], pCfgs->detClasses[k].box[l]) > 5)//骑在非机动车上的行人
							{
								isCycle = TRUE;
								break;
							
							}
						}
						if(isCycle == TRUE)
							break;
					}
				}
				if(isCycle == TRUE)//骑在非机动车上的人
					continue;*/
				//将检测框限制在检测区域内
				left = MAX(0, pCfgs->detClasses[i].box[j].x * pCfgs->m_iWidth / width);
				right = MIN((pCfgs->detClasses[i].box[j].x + pCfgs->detClasses[i].box[j].width) * pCfgs->m_iWidth / width, pCfgs->m_iWidth - 1);
				top = MAX(0, pCfgs->detClasses[i].box[j].y * pCfgs->m_iHeight / height);
				bottom = MIN((pCfgs->detClasses[i].box[j].y + pCfgs->detClasses[i].box[j].height) * pCfgs->m_iHeight / height, pCfgs->m_iHeight - 1);
				CRect rct;
				rct.x = left;
				rct.y = top;
				rct.width = right - left + 1;
				rct.height = bottom - top + 1;
				isInRegion = FALSE;//初始设为未在事件检测区域
				int inRegion[MAX_EVENT_TYPE] = { 0 };
				if(strcmp(pCfgs->detClasses[i].names, "person") == 0)//行人
				{
					inRegion[NO_PEDESTRIANTION] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, NO_PEDESTRIANTION);//禁止行人
					inRegion[PERSONFALL] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, PERSONFALL);//行人跌倒

				}
				else if(strcmp(pCfgs->detClasses[i].names, "bicycle") == 0 || strcmp(pCfgs->detClasses[i].names, "motorbike") == 0)//非机动车
				{
					inRegion[NONMOTOR] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, NONMOTOR);//禁行非机动车
					inRegion[NONMOTORFALL] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, NONMOTORFALL);//非机动车倒地
				}
				else if(strcmp(pCfgs->detClasses[i].names, "cup") == 0 || strcmp(pCfgs->detClasses[i].names, "bottle") == 0)//纸杯或瓶子
				{
					inRegion[GREENWAYDROP] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, GREENWAYDROP);//纸杯或瓶子
				}
				else
				{ 
					inRegion[STOP_INVALID] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, STOP_INVALID);//禁止停车
					inRegion[REVERSE_DRIVE] = RectInRegion0(pParams->MaskOppositeDirDriveImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, REVERSE_DRIVE);//禁止逆行
					inRegion[DRIVE_AWAY] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, DRIVE_AWAY);//驶离
					inRegion[ACCIDENTTRAFFIC] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, ACCIDENTTRAFFIC);//追尾
				}
				for(k = 0; k < MAX_EVENT_TYPE; k++)
				{
					if(inRegion[k])
					{
						isInRegion = TRUE;//在事件检测区域
						break;
					}
				}
				//if(isInRegion)//在事件检测区域内进行跟踪
				{
					match_success = -1;
					for( k = 0; k < pCfgs->event_targets_size; k++)
					{
						/*if(match_object[j] == k && match_rect[k] == j)
						{
							//float ratio = (float)(pCfgs->event_targets[k].box.width * pCfgs->event_targets[k].box.height)/(float)(pCfgs->detClasses[i].box[j].width * pCfgs->detClasses[i].box[j].height);
							//if(ratio > 0.5 && ratio < 1.5)
								match_success = 1;
							break;
						}*/
						if(match_object[i][j] == k && match_rect[k][0] == i && match_rect[k][1] == j)
						{
							//float ratio = (float)(pCfgs->event_targets[k].box.width * pCfgs->event_targets[k].box.height)/(float)(pCfgs->detClasses[i].box[j].width * pCfgs->detClasses[i].box[j].height);
							//if(ratio > 0.5 && ratio < 1.5)
							match_success = 1;
							break;
						}
						if(match_object[i][j] == k && (match_rect[k][0] != i || match_rect[k][1] != j))//两个检测框都匹配一个目标框，没有匹配上，不加入新目标
						{
							match_success = 0;
							break;
						}
					}
					if(match_success < 0)//检测框没有找到匹配的目标框
					{
						//摩托车或自行车上的行人不能当成行人
						for( k = 0; k < pCfgs->event_targets_size; k++)
						{
							//检测框为行人，目标框为机动车，如果两个框相交，认为是机动车上的行人，不加入新的目标
							/*if(strcmp(pCfgs->detClasses[i].names, "person") == 0 && (strcmp(pCfgs->event_targets[k].names, "motorbike") == 0 || strcmp(pCfgs->event_targets[k].names, "bicycle") == 0))//行人
							{
								if(overlapRatio(pCfgs->detClasses[i].box[j], pCfgs->event_targets[k].box) > 10 && match_rect[k][0] < 0 && match_rect[k][1] < 0)//骑在非机动车上的行人
								{
									//printf("person match nonmotor %d\n", pCfgs->event_targets[k].target_id);
									match_success = 0;
									pCfgs->event_targets[k].box = pCfgs->detClasses[i].box[j];
									pCfgs->event_targets[k].prob = pCfgs->detClasses[i].prob[j];
									pCfgs->event_targets[k].detected = TRUE;
									break;
								}
							}*/
							//检测框为非机动车，目标框为行人，如果两个框相交，用检测框更新目标框
							if((strcmp(pCfgs->detClasses[i].names, "motorbike") == 0 || strcmp(pCfgs->detClasses[i].names, "bicycle") == 0) && strcmp(pCfgs->event_targets[k].names, "person") == 0)
							{
								if(overlapRatio(pCfgs->detClasses[i].box[j], pCfgs->event_targets[k].box) > 5 && match_rect[k][0] < 0 && match_rect[k][1] < 0)//将行人目标更新为非机动车目标
								{
									//printf("nonmotor match person %d\n", pCfgs->event_targets[k].target_id);
									match_success = 1;
									break;
								}

							}
						}

					}
					if(match_success > 0)//跟踪到,更新检测框
					{
						//如果目标已经被判为停车事件，应与检测框匹配度高，才认为检测到，防止目标漂移，变为运动目标，后面再出现停车事件
						if(pCfgs->event_targets[k].cal_event[STOP_INVALID])
						{
							if(overlapRatio(pCfgs->detClasses[i].box[j], pCfgs->event_targets[k].box) < 60)
							{
								continue;
							}
						}
						pCfgs->event_targets[k].box = pCfgs->detClasses[i].box[j];
						pCfgs->event_targets[k].prob = pCfgs->detClasses[i].prob[j];
						pCfgs->event_targets[k].class_id = pCfgs->detClasses[i].class_id;
						strcpy(pCfgs->event_targets[k].names, pCfgs->detClasses[i].names);
						pCfgs->event_targets[k].detected = TRUE;
					}
					else if(isInRegion && match_success < 0 && pCfgs->event_targets_size < MAX_TARGET_NUM)//事件区域内，未跟踪到，加入新的目标
					{
						if(strcmp(pCfgs->detClasses[i].names, "person") == 0 && pCfgs->detClasses[i].prob[j] < 0.5)
							continue;						
						CTarget nt; 
						Initialize_target(&nt);//初始化事件目标
						nt.box = pCfgs->detClasses[i].box[j];
						nt.class_id = pCfgs->detClasses[i].class_id;
						nt.prob = pCfgs->detClasses[i].prob[j];
						nt.detected = TRUE;
						nt.target_id = pCfgs->event_target_id++;
						nt.start_time = pCfgs->currTime;//目标开始时间
						nt.region_idx = pCfgs->EventDetectCfg.EventRegion[0].uRegionID;//初始化0区域ID
						if(pCfgs->event_target_id > 5000)
							pCfgs->event_target_id = 1;
						strcpy(nt.names, pCfgs->detClasses[i].names);
						memset(nt.event_continue_num, 0, MAX_EVENT_TYPE * sizeof(int));//初始化事件持续帧数
						memset(nt.event_flag, 0, MAX_EVENT_TYPE * sizeof(int));//初始化事件标记
						memset(nt.cal_event, FALSE, MAX_EVENT_TYPE * sizeof(bool));//初始化各类事件为未计算
						memset(nt.sign_event, 0, MAX_EVENT_TYPE * sizeof(int));//初始化为未标记的事件
						memset(nt.statistic, -1, 5 * sizeof(int));//用于统计运动情况
						pCfgs->event_targets[pCfgs->event_targets_size] = nt;
						pCfgs->event_targets_size++;
					}
				}
			}
		}
	}
	//合并相交大于阈值的非机动车目标和行人目标
	for(i = 0; i < pCfgs->event_targets_size; i++)
	{
		int del_target = 0;
		if(strcmp(pCfgs->event_targets[i].names, "person") != 0)//与行人目标比较
			continue;
		if(pCfgs->event_targets[i].cal_event[NO_PEDESTRIANTION])//如果目标已经检测为行人事件，不合并目标
			continue;
		for(j = 0; j < pCfgs->event_targets_size; j++)
		{
			if(strcmp(pCfgs->event_targets[j].names, "motorbike") != 0 && strcmp(pCfgs->event_targets[j].names, "bicycle") != 0)
				continue;
			if(overlapRatio(pCfgs->event_targets[i].box, pCfgs->event_targets[j].box) > 10)//非机动车和其他目标重合比较大,认为是非机动车
			{
				CRect rct;
				bool detected = pCfgs->event_targets[j].detected;
				float prob = pCfgs->event_targets[j].prob;
				if(pCfgs->event_targets[i].detected && pCfgs->event_targets[j].detected)//目标框都检测到，合并框
				{
					rct.x = MIN(pCfgs->event_targets[i].box.x, pCfgs->event_targets[j].box.x);
					rct.y = MIN(pCfgs->event_targets[i].box.y, pCfgs->event_targets[j].box.y);
					rct.width = MAX(pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width - rct.x, pCfgs->event_targets[j].box.x + pCfgs->event_targets[j].box.width - rct.x);
					rct.height = MAX(pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height - rct.y, pCfgs->event_targets[j].box.y + pCfgs->event_targets[j].box.height - rct.y);
				}
				else if(pCfgs->event_targets[i].detected)//采用行人框
				{
					rct = pCfgs->event_targets[i].box;
					prob = pCfgs->event_targets[i].prob;
					detected = pCfgs->event_targets[i].detected;
				}
				else
					rct = pCfgs->event_targets[j].box;//采用非机动车框
				pCfgs->event_targets[j].box = rct;
				pCfgs->event_targets[j].prob = prob;
				pCfgs->event_targets[j].detected = detected;
				del_target = 1;
				break;
			}

		}
		if(del_target == 1)//删除行人目标
			DeleteTarget(&pCfgs->event_targets_size, &i, pCfgs->event_targets);

	}
	//对重叠大的非机动车进行合并
	for(i = 0; i < pCfgs->event_targets_size; i++)
	{
		int del_target = 0;
		if(strcmp(pCfgs->event_targets[i].names, "motorbike") != 0 && strcmp(pCfgs->event_targets[i].names, "bicycle") != 0)//非机动车目标
			continue;
		if(pCfgs->event_targets[i].cal_event[NONMOTOR])//如果目标已经检测为非机动车事件，不合并目标
			continue;
		for(j = 0; j < pCfgs->event_targets_size; j++)
		{
			if(strcmp(pCfgs->event_targets[j].names, "motorbike") != 0 && strcmp(pCfgs->event_targets[j].names, "bicycle") != 0)//非机动车目标
				continue;
			if(i == j)//相同目标
				continue;
			if(overlapRatio(pCfgs->event_targets[i].box, pCfgs->event_targets[j].box) > 10)//非机动车目标重合比较大,认为是非机动车
			{
				CRect rct;
				bool detected = pCfgs->event_targets[j].detected;
				float prob = pCfgs->event_targets[j].prob;
				if(pCfgs->event_targets[i].detected && pCfgs->event_targets[j].detected)//目标框都检测到，合并框
				{
					rct.x = MIN(pCfgs->event_targets[i].box.x, pCfgs->event_targets[j].box.x);
					rct.y = MIN(pCfgs->event_targets[i].box.y, pCfgs->event_targets[j].box.y);
					rct.width = MAX(pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width - rct.x, pCfgs->event_targets[j].box.x + pCfgs->event_targets[j].box.width - rct.x);
					rct.height = MAX(pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height - rct.y, pCfgs->event_targets[j].box.y + pCfgs->event_targets[j].box.height - rct.y);
				}
				else if(pCfgs->event_targets[i].detected)
				{
					rct = pCfgs->event_targets[i].box;
					prob = pCfgs->event_targets[i].prob;
					detected = pCfgs->event_targets[i].detected;
				}
				else
					rct = pCfgs->event_targets[j].box;
				pCfgs->event_targets[j].box = rct;
				pCfgs->event_targets[j].prob = prob;
				pCfgs->event_targets[j].detected = detected;
				del_target = 1;
				break;
			}

		}
		if(del_target == 1)//删除行人目标
			DeleteTarget(&pCfgs->event_targets_size, &i, pCfgs->event_targets);
	}
	//分析目标
	for(i = 0; i < pCfgs->event_targets_size; i++)
	{

		//轨迹数小于3000，直接保存，大于3000，去除旧的
		if(pCfgs->event_targets[i].trajectory_num < 3000)
		{

			pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num].x = pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width / 2;
			pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num].y = pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height / 2;
			pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num].width = pCfgs->event_targets[i].box.width;
			pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num].height = pCfgs->event_targets[i].box.height;
			pCfgs->event_targets[i].trajectory_num++;
		}
		else
		{
			for(j = 0; j < pCfgs->event_targets[i].trajectory_num - 1; j++)
			{
				pCfgs->event_targets[i].trajectory[j] = pCfgs->event_targets[i].trajectory[j + 1];
			}
			pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num - 1].x = pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width / 2;
			pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num - 1].y = pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height / 2;
			pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num - 1].width = pCfgs->event_targets[i].box.width;
			pCfgs->event_targets[i].trajectory[pCfgs->event_targets[i].trajectory_num - 1].height = pCfgs->event_targets[i].box.height;
		}
		//检测到，并更新速度
		if(pCfgs->event_targets[i].detected)
		{
			/*if(pCfgs->event_targets[i].lost_detected == 0)//前一帧也有检测到，计算速度
			{
			get_speed(&pCfgs->event_targets[i]);
			}*/
			pCfgs->event_targets[i].lost_detected = 0;
			/*for(j = 0; j < pCfgs->EventNum; j++)
			{
				if(pCfgs->EventInfo[j].uEventID == pCfgs->event_targets[i].target_id)
				{
					pCfgs->EventInfo[j].end_time = pCfgs->gThisFrameTime;//记录检测的最后一帧
					if(pCfgs->EventEndTime < pCfgs->gThisFrameTime)
						pCfgs->EventEndTime = pCfgs->gThisFrameTime;
				}
			}*/
		}
		else//未检测到
		{
			pCfgs->event_targets[i].lost_detected++;
			pCfgs->event_targets[i].box.x += pCfgs->event_targets[i].vx;
			pCfgs->event_targets[i].box.y += pCfgs->event_targets[i].vy;
		}

		//判断目标是否还在检测区域内
		left = MAX(0, pCfgs->event_targets[i].box.x * pCfgs->m_iWidth / width);
		right = MIN((pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width)* pCfgs->m_iWidth / width, pCfgs->m_iWidth - 1);
		top = MAX(0, pCfgs->event_targets[i].box.y * pCfgs->m_iHeight / height);
		bottom = MIN((pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height) * pCfgs->m_iHeight / height, pCfgs->m_iHeight - 1);
		CRect rct;
		rct.x = left;
		rct.y = top;
		rct.width = right - left + 1;
		rct.height = bottom - top + 1;
		isInRegion = FALSE;
		if(strcmp(pCfgs->event_targets[i].names, "person") == 0)//行人
		{
			targetInRegion[i][NO_PEDESTRIANTION] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, NO_PEDESTRIANTION);
			targetInRegion[i][PERSONFALL] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, PERSONFALL);
		}
		else if(strcmp(pCfgs->event_targets[i].names, "bicycle") == 0 || strcmp(pCfgs->event_targets[i].names, "motorbike") == 0)//非机动车
		{
			targetInRegion[i][NONMOTOR] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, NONMOTOR);
			targetInRegion[i][NONMOTORFALL] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, NONMOTORFALL);
		}
		else if(strcmp(pCfgs->event_targets[i].names, "cup") == 0 || strcmp(pCfgs->event_targets[i].names, "bottle") == 0)//纸杯或瓶子
		{
			targetInRegion[i][GREENWAYDROP] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, GREENWAYDROP);//纸杯或瓶子
		}
		else//机动车
		{
			targetInRegion[i][STOP_INVALID] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, STOP_INVALID);//违停
			targetInRegion[i][REVERSE_DRIVE] = RectInRegion0(pParams->MaskOppositeDirDriveImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, REVERSE_DRIVE);//逆行
			targetInRegion[i][DRIVE_AWAY] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, DRIVE_AWAY);//驶离
			targetInRegion[i][ACCIDENTTRAFFIC] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, ACCIDENTTRAFFIC);//追尾
		}
		//判断目标属于哪个事件区域
		for(k = 0; k < pCfgs->EventDetectCfg.uEventRegionNum; k++)
		{
			int inRegionRatio = RectInRegion(pParams->MaskEventIDImage, pCfgs, width, height, pCfgs->event_targets[i].box, pCfgs->EventDetectCfg.EventRegion[k].uRegionID);
			if(inRegionRatio > 10)
			{
				pCfgs->event_targets[i].region_idx = pCfgs->EventDetectCfg.EventRegion[k].uRegionID;
				break;
			}
		}
		for(k = 0; k < MAX_EVENT_TYPE; k++)
		{
			if(targetInRegion[i][k])
			{
				isInRegion = TRUE;//在事件检测区域
				break;
			}
		}
		//去除不在检测区域的目标
		if(isInRegion)
		{
			;
		}
		else if(pCfgs->event_targets[i].continue_num > 5)//目标不在事件区域内，并且目标在检测区域内存在5帧以上
		{
			/*for(j = 0; j < pCfgs->EventNum; j++)//如果没有设置此事件结束，设置
			{
				if(pCfgs->EventInfo[j].flag == 0)
				{
					if(pCfgs->EventInfo[j].uEventID == pCfgs->event_targets[i].target_id)
					{
						//pCfgs->EventInfo[j].end_time = pCfgs->gThisFrameTime;
						pCfgs->EventInfo[j].flag = 1;
						break;
					}
				}
			}*/
			//if(pCfgs->event_targets[i].target_id > 0)
				//printf("delete 1\n");
			//DeleteTarget(&pCfgs->event_targets_size, &i, pCfgs->event_targets);
			//continue;
		}
		//当目标在视频存在时间太长或长时间没有检测到或离开图像，删除目标
		if(pCfgs->event_targets[i].continue_num > 5000 || (pCfgs->event_targets[i].lost_detected > 20 && (pCfgs->event_targets[i].event_flag[STOP_INVALID] == 0 && pCfgs->event_targets[i].event_flag[ACCIDENTTRAFFIC] == 0)) || (pCfgs->event_targets[i].lost_detected > 200 && (pCfgs->event_targets[i].event_flag[STOP_INVALID] || pCfgs->event_targets[i].event_flag[ACCIDENTTRAFFIC]))||((pCfgs->event_targets[i].box.x < 10 || pCfgs->event_targets[i].box.y < 10 || (pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width) > (width - 10) || (pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height) > (height - 10))&& pCfgs->event_targets[i].lost_detected > 0))
		{
			/*for(j = 0; j < pCfgs->EventNum; j++)//如果没有设置此事件结束，设置
			{
				if(pCfgs->EventInfo[j].flag == 0)
				{
					if(pCfgs->EventInfo[j].uEventID == pCfgs->event_targets[i].target_id)
					{
						//pCfgs->EventInfo[j].end_time = pCfgs->gThisFrameTime;
						pCfgs->EventInfo[j].flag = 1;
						break;
					}
				}
			}*/
			//if(pCfgs->event_targets[i].target_id > 0)
			//	printf("delete 2,%d,lost_detected =%d,flag = %d\n",pCfgs->event_targets[i].continue_num,pCfgs->event_targets[i].lost_detected, pCfgs->event_targets[i].event_flag[ACCIDENTTRAFFIC]);
			DeleteTarget(&pCfgs->event_targets_size, &i, pCfgs->event_targets);
			continue;

		}
		//将目标框保存起来，用于事件检测
		targetRect[i] = pCfgs->event_targets[i].box;
		//保存目标的运动情况，保存最多10s间隔的运动情况
		int continue_num = pCfgs->event_targets[i].trajectory_num - 150;
		for(j = 0; j < 150; j++)
		{
			continue_num = pCfgs->event_targets[i].trajectory_num - 1 - j;
			if(pCfgs->currTime - pCfgs->uStatFrameTime[149 - j] > 10 || continue_num <= 0)//10s间隔
			{
				break;
			}
		}
		int dx = pCfgs->event_targets[i].box.x + pCfgs->event_targets[i].box.width / 2 - pCfgs->event_targets[i].trajectory[continue_num].x;
		int dy = pCfgs->event_targets[i].box.y + pCfgs->event_targets[i].box.height / 2 - pCfgs->event_targets[i].trajectory[continue_num].y;
		dx = (dx < 0)? -dx : dx;
		dy = (dy < 0)? -dy : dy;
		targetDisXY[i][0] = dx;
		targetDisXY[i][1] = dy;
		targetDisXY[i][2] = ((149 - j) < 0) ? 0 : (149 - j);//记录10s间隔的帧数位置
		pCfgs->event_targets[i].continue_num++;
	}
	//分析目标事件类型
	for(i = 0;i < pCfgs->event_targets_size; i++)
	{

		isInRegion = FALSE;
		for(k = 0; k < MAX_EVENT_TYPE; k++)
		{
			if(targetInRegion[i][k])
			{
				isInRegion = TRUE;//在事件检测区域
			}
			if(targetInRegion[i][k] == 0)
			{
				pCfgs->event_targets[i].event_continue_num[k] = 0;
				pCfgs->event_targets[i].event_flag[k] = 0;//当事件离开此区域后，不再重新计算此事件，防止一个目标计算同类事件多次
			}
		}
		//在检测区域的目标
		if(isInRegion)
		{
			//判断交通事件
			if(targetInRegion[i][STOP_INVALID] && pCfgs->event_targets[i].continue_num > 100)//目标在禁止停车区域
			{
				IllegalParkDetect(pCfgs, i, STOP_INVALID, targetDisXY, width, height);//禁止停车检测
			}
			if(targetInRegion[i][REVERSE_DRIVE])//目标在禁止逆行区域
			{
				OppositeDirDriveDetect(pCfgs, i, REVERSE_DRIVE, targetInRegion[i][REVERSE_DRIVE], width, height);
			}
			if(targetInRegion[i][DRIVE_AWAY])//驶离
			{
				OffLaneDetect(pCfgs, i, DRIVE_AWAY);
			}
			if(targetInRegion[i][NO_PEDESTRIANTION])//行人
			{
				NoPersonAllowDetect(pCfgs, i, NO_PEDESTRIANTION, width, height);//禁止行人检测
			}
			if(targetInRegion[i][NONMOTOR])//非机动车
			{
				NonMotorAllowDetect(pCfgs, i, NONMOTOR);//禁止非机动车检测
			}
			if(targetInRegion[i][PERSONFALL])//行人跌倒
			{
				PersonFallDetect(pCfgs, i, PERSONFALL, width, height);//行人跌倒检测
			}
			if(targetInRegion[i][NONMOTORFALL])//非机动车跌倒
			{
				NonMotorFallDetect(pCfgs, i, NONMOTORFALL, width, height);//非机动车跌倒检测
			}
			if(targetInRegion[i][GREENWAYDROP])//纸杯或瓶子
			{
				GreenwayDropDetect(pCfgs, i, GREENWAYDROP);////纸杯或瓶子
			}
			if(targetInRegion[i][ACCIDENTTRAFFIC] && pCfgs->event_targets[i].continue_num > 100)//目标在事故检测区域内
			{
				TrafficAccidentDetect(pCfgs, i, ACCIDENTTRAFFIC, targetDisXY, width, height);
			}

		}

	}
	//判断是否进行抛弃物检测
	/*bool detDrop = FALSE;
	for(i = 0; i < pCfgs->EventDetectCfg.uEventRegionNum; i++)
	{
		if(pCfgs->EventDetectCfg.EventRegion[i].eventType == DROP)
		{
			detDrop = TRUE;
			break;
		}
	}
	if(detDrop)//进行检测
	{
		DropDetect(pCfgs, pParams, DROP, width, height);
	}*/
	//printf("event person = %d, person fall = %d, nonmotor = %d,greendrop = %d\n",pCfgs->uPersonNum,pCfgs->uPersonFallNum,pCfgs->uNonMotorFallNum,pCfgs->uGreenwayDropNum);
}
void EventDetectProc(ALGCFGS *pCfgs, ALGPARAMS *pParams, int width, int height)
{
	int i = 0, j = 0;
	int eventID = 0, newID = 0, new_event_flag = 0;

	//设置交通事件掩模图像
	if(pCfgs->bMaskEventImage == FALSE)
	{
		MaskEventImage(pCfgs, pParams, width, height);
		pCfgs->bMaskEventImage = TRUE;
	}
	
	//对交通事件进行分析
	TrafficEventAnalysis(pCfgs, pParams, width, height);

	//对道路事件进行分析
	TrafficRoadAnalysis(pCfgs, pParams, width, height);

	memset((void *)&pCfgs->ResultMsg.uResultInfo.eventData, 0, sizeof(pCfgs->ResultMsg.uResultInfo.eventData));//初始化
	//分析停车事件
	if(pCfgs->uIllegalParkNum)//有停车事件
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool IllegalParkEvent = FALSE;
		for(i = 0; i < pCfgs->uIllegalParkNum; i++)
		{
			if(pCfgs->uIllegalParkID == pCfgs->IllegalParkBox[i].uEventID)
			{
				IllegalParkEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->IllegalParkBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的停车事件
		if(IllegalParkEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uIllegalParkTime) > (pCfgs->EventDetectCfg.ReportInterval[STOP_INVALID] * 60) || pCfgs->uIllegalParkID == 0)//出现新的ID,并且前后停车事件间隔7分钟之上
			{
				eventID = newID;
				IllegalParkEvent = TRUE;
				pCfgs->uIllegalParkID = pCfgs->IllegalParkBox[eventID].uEventID;//更新数据
				pCfgs->uIllegalParkTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
				printf("STOP_INVALID new flag\n");
			}
		}
		else//之前已经出现的停车事件
		{
			new_event_flag = 0;
		}
		//只报一个停车事件
		if(IllegalParkEvent)
		{
			int IsIllegalPark = TRUE;
			//判断停车是否在拥堵区域内
			for(i = 0; i < pCfgs->LaneAmount; i++)
			{
				if(pCfgs->bCongestion[i])
				{
					int sum = 0;
					int num = 0;
					for(int k = pCfgs->IllegalParkBox[eventID].EventBox[0].x; k <= pCfgs->IllegalParkBox[eventID].EventBox[1].x; k++)
					{
						for(int l = pCfgs->IllegalParkBox[eventID].EventBox[0].y; l <= pCfgs->IllegalParkBox[eventID].EventBox[3].y; l++)
						{
							CPoint pt;
							pt.x = k;
							pt.y = l;
							if(isPointInRect(pt, pCfgs->CongestionBox[i].EventBox[3], pCfgs->CongestionBox[i].EventBox[0], pCfgs->CongestionBox[i].EventBox[1], pCfgs->CongestionBox[i].EventBox[2]))
								sum++;
							num++;
						}
					}
					//在拥堵区域内
					if((float)sum / (float)(num) > 0.8)
						IsIllegalPark = FALSE;
				}
			}
			if(IsIllegalPark == TRUE)
			{
				if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
					pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->IllegalParkBox[eventID].uRegionID;//事件区域ID
					if(new_event_flag == 1)
					{
						pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
					}
					memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, pCfgs->IllegalParkBox[eventID].EventBox, 4 * sizeof(CPoint));
					pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = STOP_INVALID;
					pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
				}
			}
			else
			{
				printf("STOP_INVALID in congestion region\n");
			}
		}
	}

	//分析逆行事件
	/*for(i = 0; i < pCfgs->uOppositeDirDriveNum; i++)
	{
		if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
		{
			pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = pCfgs->OppositeDirDriveBox[i].uNewEventFlag;
			if(pCfgs->OppositeDirDriveBox[i].uNewEventFlag == 1)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
			}
			memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, pCfgs->OppositeDirDriveBox[i].EventBox, 4 * sizeof(CPoint));
			pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = REVERSE_DRIVE;
			pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
		}
	}*/
	if(pCfgs->uOppositeDirDriveNum)//有逆行事件
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool OppositeDirDriveEvent = FALSE;
		for(i = 0; i < pCfgs->uOppositeDirDriveNum; i++)
		{
			//一直存在逆行事件
			if(pCfgs->uOppositeDirDriveID == pCfgs->OppositeDirDriveBox[i].uEventID)
			{
				OppositeDirDriveEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->OppositeDirDriveBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的逆行事件
		if(OppositeDirDriveEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uOppositeDirDriveTime) > (pCfgs->EventDetectCfg.ReportInterval[REVERSE_DRIVE] * 60) || pCfgs->uOppositeDirDriveID == 0)//出现新的ID,前后逆行超过3000
			{
				eventID = newID;
				OppositeDirDriveEvent = TRUE;
				pCfgs->uOppositeDirDriveID = pCfgs->OppositeDirDriveBox[eventID].uEventID;//更新数据
				pCfgs->uOppositeDirDriveTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(OppositeDirDriveEvent == TRUE)
		{
			//只传一个逆行事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->OppositeDirDriveBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->OppositeDirDriveBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = REVERSE_DRIVE;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}

	//分析驶离事件
	/*for(i = 0; i < pCfgs->uOffLaneNum; i++)
	{
		if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
		{
			pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = pCfgs->OffLaneBox[i].uNewEventFlag;
			pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->OppositeDirDriveBox[eventID].uRegionID;//事件区域ID
			if(pCfgs->OffLaneBox[i].uNewEventFlag == 1)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
			}
			memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, pCfgs->OffLaneBox[i].EventBox, 4 * sizeof(CPoint));
			pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = DRIVE_AWAY;
            pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
		}
	}*/
	if(pCfgs->uOffLaneNum)//有驶离事件
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool OffLaneEvent = FALSE;
		for(i = 0; i < pCfgs->uOffLaneNum; i++)
		{
			//一直存在驶离事件
			if(pCfgs->uOffLaneID == pCfgs->OffLaneBox[i].uEventID)
			{
				OffLaneEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->OffLaneBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的驶离事件
		if(OffLaneEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uOffLaneTime) > (pCfgs->EventDetectCfg.ReportInterval[DRIVE_AWAY] * 60) || pCfgs->uOffLaneID == 0)//出现新的ID,前后驶离超过3000
			{
				eventID = newID;
				OffLaneEvent = TRUE;
				pCfgs->uOffLaneID = pCfgs->OffLaneBox[eventID].uEventID;//更新数据
				pCfgs->uOffLaneTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(OffLaneEvent == TRUE)
		{
			//只传一个驶离事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->OffLaneBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->OffLaneBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = DRIVE_AWAY;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}
	//分析行人事件
	if(pCfgs->uNoPersonAllowNum)//有行人事件
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool PersonEvent = FALSE;
		for(i = 0; i < pCfgs->uNoPersonAllowNum; i++)
		{
			//一直存在的行人事件
			if(pCfgs->uCurrentPersonID == pCfgs->NoPersonAllowBox[i].uEventID)
			{
				PersonEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->NoPersonAllowBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的行人事件
		if(PersonEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uPersonEventTime) > MAX(5, pCfgs->EventDetectCfg.ReportInterval[NO_PEDESTRIANTION] * 60) || pCfgs->uCurrentPersonID == 0)//出现新的ID,并且前后行人事件间隔超过5s
			{
				eventID = newID;
				PersonEvent = TRUE;
				pCfgs->uCurrentPersonID = pCfgs->NoPersonAllowBox[eventID].uEventID;//更新数据
				pCfgs->uPersonEventTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(PersonEvent == TRUE)
		{
			//只传一个行人事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->NoPersonAllowBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
					printf("person new flag, time = %f,%f,%d\n", pCfgs->currTime,pCfgs->uPersonEventTime,pCfgs->EventDetectCfg.ReportInterval[NO_PEDESTRIANTION]);
				}
				memcpy(detectRegion.detRegion, pCfgs->NoPersonAllowBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = NO_PEDESTRIANTION;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}

	//分析行人跌倒
	if(pCfgs->uPersonFallNum)//有行人跌倒
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool PersonFallEvent = FALSE;
		for(i = 0; i < pCfgs->uPersonFallNum; i++)
		{
			//一直存在的行人跌倒
			if(pCfgs->uCurrentPersonFallID == pCfgs->PersonFallBox[i].uEventID)
			{
				PersonFallEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->PersonFallBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的行人跌倒事件
		if(PersonFallEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uPersonFallEventTime) > MAX(5, pCfgs->EventDetectCfg.ReportInterval[PERSONFALL] * 60) || pCfgs->uCurrentPersonFallID == 0)//出现新的ID,前后行人跌倒超过5s
			{
				eventID = newID;
				PersonFallEvent = TRUE;
				pCfgs->uCurrentPersonFallID = pCfgs->PersonFallBox[eventID].uEventID;//更新数据
				pCfgs->uPersonFallEventTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(PersonFallEvent == TRUE)
		{
			//只传一个行人跌倒事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->PersonFallBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				//printf("person fall new flag = [%d,%d]\n",pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID,pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag);
				memcpy(detectRegion.detRegion, pCfgs->PersonFallBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = PERSONFALL;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}

	//分析非机动车事件
	if(pCfgs->uNonMotorAllowNum)//有非机动车事件
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool NonMotorEvent = FALSE;
		for(i = 0; i < pCfgs->uNonMotorAllowNum; i++)
		{
			//一直存在的非机动车事件
			if(pCfgs->uCurrentNonMotorID == pCfgs->NonMotorAllowBox[i].uEventID)
			{
				NonMotorEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->NonMotorAllowBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的非机动车事件
		if(NonMotorEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uNonMotorEventTime) > MAX(5, pCfgs->EventDetectCfg.ReportInterval[NONMOTOR] * 60) || pCfgs->uCurrentNonMotorID == 0)//出现新的ID,并且前后非机动车事件间隔超过5s
			{
				eventID = newID;
				NonMotorEvent = TRUE;
				pCfgs->uCurrentNonMotorID = pCfgs->NonMotorAllowBox[eventID].uEventID;//更新数据
				pCfgs->uNonMotorEventTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(NonMotorEvent == TRUE)
		{
			//只传一个非机动车
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->NonMotorAllowBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->NonMotorAllowBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = NONMOTOR;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}

	//分析非机动车跌倒事件
	if(pCfgs->uNonMotorFallNum)//有非机动车跌倒
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool NonMotorFallEvent = FALSE;
		for(i = 0; i < pCfgs->uNonMotorFallNum; i++)
		{
			//一直存在的非机动车跌倒
			if(pCfgs->uCurrentNonMotorFallID == pCfgs->NonMotorFallBox[i].uEventID)
			{
				NonMotorFallEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->NonMotorFallBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的非机动车跌倒事件
		if(NonMotorFallEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uNonMotorFallEventTime) > MAX(5, pCfgs->EventDetectCfg.ReportInterval[NONMOTORFALL] * 60) || pCfgs->uCurrentNonMotorFallID == 0)//出现新的ID,前后非机动车跌倒超过5s
			{
				NonMotorFallEvent = TRUE;
				pCfgs->uCurrentNonMotorFallID = pCfgs->NonMotorFallBox[eventID].uEventID;//更新数据
				pCfgs->uNonMotorFallEventTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(NonMotorFallEvent == TRUE)
		{
			//只传一个非机动车跌倒事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->NonMotorFallBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->NonMotorFallBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = NONMOTORFALL;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}

	//分析绿道抛弃物事件
	if(pCfgs->uGreenwayDropNum)//有绿道抛弃物
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool GreenwayDropEvent = FALSE;
		for(i = 0; i < pCfgs->uGreenwayDropNum; i++)
		{
			//一直存在绿道抛弃物
			if(pCfgs->uCurrentGreenwayDropID == pCfgs->GreenwayDropBox[i].uEventID)
			{
				GreenwayDropEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->GreenwayDropBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的绿道抛弃物
		if(GreenwayDropEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uGreenwayDropEventTime) > (pCfgs->EventDetectCfg.ReportInterval[GREENWAYDROP] * 60) || pCfgs->uCurrentGreenwayDropID == 0)//出现新的ID,前后绿道抛弃物超过3000
			{
				GreenwayDropEvent = TRUE;
				pCfgs->uCurrentGreenwayDropID = pCfgs->GreenwayDropBox[eventID].uEventID;//更新数据
				pCfgs->uGreenwayDropEventTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(GreenwayDropEvent == TRUE)
		{
			//只传一个绿道抛弃物事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->GreenwayDropBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->GreenwayDropBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = GREENWAYDROP;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}

	//分析交通事故事件
	if(pCfgs->uTrafficAccidentNum)//有交通事故
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool TrafficAccidentEvent = FALSE;
		for(i = 0; i < pCfgs->uTrafficAccidentNum; i++)
		{
			//一直有交通事故
			if(pCfgs->uTrafficAccidentID == pCfgs->TrafficAccidentBox[i].uEventID)
			{
				TrafficAccidentEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->TrafficAccidentBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的交通事故
		if(TrafficAccidentEvent == FALSE)
		{
			if((pCfgs->currTime - pCfgs->uTrafficAccidentTime) > (pCfgs->EventDetectCfg.ReportInterval[ACCIDENTTRAFFIC] * 60) || pCfgs->uTrafficAccidentID == 0)//出现新的ID,前后交通事故超过3000
			{
				eventID = newID;
				TrafficAccidentEvent = TRUE;
				pCfgs->uTrafficAccidentID = pCfgs->TrafficAccidentBox[eventID].uEventID;//更新数据
				pCfgs->uTrafficAccidentTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
				printf("ACCIDENTTRAFFIC new flag\n");
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(TrafficAccidentEvent == TRUE)
		{
			//只传一个交通事故事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->TrafficAccidentBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->TrafficAccidentBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = ACCIDENTTRAFFIC;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}
	printf("new event = %d, event_num = %d\n", pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag, pCfgs->ResultMsg.uResultInfo.eventData.uEventNum);
	//////////////////////////////////////////////////////分析道路事件
	//道路积水
	if(pCfgs->uRoadWaterNum)//有道路积水
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool RoadWaterEvent = FALSE;
		for(i = 0; i < pCfgs->uRoadWaterNum; i++)
		{
			//一直存在积水事件
			if(pCfgs->uRoadWaterID == pCfgs->RoadWaterBox[i].uEventID)
			{
				RoadWaterEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->RoadWaterBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的积水事件
		if(RoadWaterEvent == FALSE)
		{
			{
				eventID = newID;
				RoadWaterEvent = TRUE;
				pCfgs->uRoadWaterID = pCfgs->RoadWaterBox[eventID].uEventID;//更新数据
				pCfgs->uRoadWaterTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(RoadWaterEvent == TRUE)
		{
			//只传一个积水事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->RoadWaterBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->RoadWaterBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = ROADWATER;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}
	//道路坑洼
	if(pCfgs->uRoadHollowNum)//有道路坑洼
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool RoadHollowrEvent = FALSE;
		for(i = 0; i < pCfgs->uRoadHollowNum; i++)
		{
			//一直存在坑洼事件
			if(pCfgs->uRoadHollowID == pCfgs->RoadHollowBox[i].uEventID)
			{
				RoadHollowrEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->RoadHollowBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的坑洼事件
		if(RoadHollowrEvent == FALSE)
		{
			{
				eventID = newID;
				RoadHollowrEvent = TRUE;
				pCfgs->uRoadHollowID = pCfgs->RoadHollowBox[eventID].uEventID;//更新数据
				pCfgs->uRoadHollowTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(RoadHollowrEvent == TRUE)
		{
			//只传一个坑洼事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->RoadHollowBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->RoadHollowBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = ROADHOLLOW;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}
	//道路破损
	if(pCfgs->uRoadDamageNum)//有道路破损
	{
		eventID = 0, newID = 0;
		new_event_flag = 0;
		bool RoadDamagerEvent = FALSE;
		for(i = 0; i < pCfgs->uRoadDamageNum; i++)
		{
			//一直存在破损事件
			if(pCfgs->uRoadDamageID == pCfgs->RoadDamageBox[i].uEventID)
			{
				RoadDamagerEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->RoadDamageBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的破损事件
		if(RoadDamagerEvent == FALSE)
		{
			{
				eventID = newID;
				RoadDamagerEvent = TRUE;
				pCfgs->uRoadDamageID = pCfgs->RoadDamageBox[eventID].uEventID;//更新数据
				pCfgs->uRoadDamageTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(RoadDamagerEvent == TRUE)
		{
			//只传一个破损事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->RoadDamageBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->RoadDamageBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = ROADDAMAGE;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}
	//道路裂缝
	if(pCfgs->uRoadCrackNum)//有道路裂缝
	{
		eventID = 0, eventID = 0;
		new_event_flag = 0;
		bool RoadCrackEvent = FALSE;
		for(i = 0; i < pCfgs->uRoadCrackNum; i++)
		{
			//一直存在裂缝事件
			if(pCfgs->uRoadCrackID == pCfgs->RoadCrackBox[i].uEventID)
			{
				RoadCrackEvent = TRUE;
				eventID = i;
			}
			if(pCfgs->RoadCrackBox[i].uNewEventFlag == 1)
			{
				new_event_flag = 1;
				newID = i;
			}
		}
		//新出现的裂缝事件
		if(RoadCrackEvent == FALSE)
		{
			{
				eventID = newID;
				RoadCrackEvent = TRUE;
				pCfgs->uRoadCrackID = pCfgs->RoadCrackBox[eventID].uEventID;//更新数据
				pCfgs->uRoadCrackTime = pCfgs->currTime;
				//new_event_flag = 1;//新事件
			}
		}
		else
		{
			new_event_flag = 0;
		}
		if(RoadCrackEvent == TRUE)
		{
			//只传一个裂缝事件
			DETECTREGION detectRegion;
			if(pCfgs->ResultMsg.uResultInfo.eventData.uEventNum < MAX_EVENT_NUM)
			{
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventID = new_event_flag;
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uRegionID = pCfgs->RoadCrackBox[eventID].uRegionID;//事件区域ID
				if(new_event_flag == 1)
				{
					pCfgs->ResultMsg.uResultInfo.eventData.uNewEventFlag = 1;
				}
				memcpy(detectRegion.detRegion, pCfgs->RoadCrackBox[eventID].EventBox, 4 * sizeof(CPoint));
				memcpy(pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].EventBox, detectRegion.detRegion, 4 * sizeof(CPoint));
				pCfgs->ResultMsg.uResultInfo.eventData.EventBox[pCfgs->ResultMsg.uResultInfo.eventData.uEventNum].uEventType = ROADCRACK;
				pCfgs->ResultMsg.uResultInfo.eventData.uEventNum++;
			}
		}
	}
}
/////////////////////////////////////////////////////////////////////////////////////////////////对道路事件进行处理
//将相同类别距离近的框进行合并
void post_process_box_road(ALGCFGS* pCfgs, Uint16 ratio_threshold)
{
	Uint16 i = 0, j = 0, k = 0, l = 0;
	Uint16 dis_x = 0, dis_y = 0;
	CRect r0, r1, r2;
	for(i = 0; i < pCfgs->classes; i++)//分类别进行
	{
		if(pCfgs->detClasses[i].classes_num == 0)
			continue;
		if(strcmp(pCfgs->detClasses[i].names, "js") != 0 && strcmp(pCfgs->detClasses[i].names, "kw") != 0 && strcmp(pCfgs->detClasses[i].names, "ps") != 0 && strcmp(pCfgs->detClasses[i].names, "lf") != 0)//其他类别不进行处理
			continue;
		for(j = 0; j < pCfgs->detClasses[i].classes_num; j++)
		{
			r0 = pCfgs->detClasses[i].box[j];
			for(k = j + 1; k < pCfgs->detClasses[i].classes_num; k++)
			{
				//求两框之间距离
				r1 = pCfgs->detClasses[i].box[k];
				dis_x = MIN(r0.x + r0.width, r1.x + r1.width) - MAX(r0.x, r1.x);
				dis_y = MIN(r0.y + r0.height, r1.y + r1.height) - MAX(r0.y, r1.y);
				if(dis_x  > -ratio_threshold && dis_y > -ratio_threshold)//距离小于阈值进行合并
				{
					r2.x = (r0.x < r1.x)? r0.x : r1.x;//取最小值
					r2.y = (r0.y < r1.y)? r0.y : r1.y;
					r2.width = ((r0.x + r0.width) > (r1.x + r1.width))? (r0.x + r0.width - r2.x) : (r1.x + r1.width - r2.x);
					r2.height = ((r0.y + r0.height) > (r1.y + r1.height))? (r0.y + r0.height - r2.y) : (r1.y + r1.height - r2.y);
					r0 = r2;
					for( l = k + 1; l < pCfgs->detClasses[i].classes_num; l++)
					{
						pCfgs->detClasses[i].box[l - 1] = pCfgs->detClasses[i].box[l];
						pCfgs->detClasses[i].prob[l - 1] = pCfgs->detClasses[i].prob[l];
					}
					pCfgs->detClasses[i].classes_num = pCfgs->detClasses[i].classes_num - 1;
					k--;
				}
			}
			pCfgs->detClasses[i].box[j] = r0;
		}
	}
}
int match_object_rect2(CTarget* targets, int targets_size, CDetBox* detClasses, int class_id, int* match_object, int* match_rect, int thresh)
{

	if(targets_size < 1 || detClasses[class_id].classes_num < 1)//没有目标或检测框，返回
	{
		return -1;
	}
	int i = 0, j = 0;
	int dis_min = 1e+9, idx_max = 0;
	CRect r1,r2;
	//初始化匹配值为-1
	memset(match_object, -1, MAX_ROAD_TARGET_NUM * sizeof(int));
	memset(match_rect, -1, MAX_ROAD_TARGET_NUM * sizeof(int));

	for(j = 0; j < detClasses[class_id].classes_num; j++)//匹配目标
	{
		r1 = detClasses[class_id].box[j];
		match_object[j] = -1;
		idx_max = -1;
		dis_min = 1e+9;
		for(i = 0; i < targets_size; i++)
		{
			r2 = targets[i].box;
			if(class_id == targets[i].class_id)
			{
				//计算两个框的中心距离
				int dis = 0;
				dis = (r1.x + r1.width /2 - r2.x - r2.width / 2) * (r1.x + r1.width /2 - r2.x - r2.width / 2) + (r1.y + r1.height / 2 - r2.y - r2.height / 2) * (r1.y + r1.height / 2 - r2.y - r2.height / 2);
				dis =sqrt(dis);
				if(dis < dis_min)//得到最小距离
				{
					dis_min = dis;
					idx_max = i;
				}
			}
		}
		if(dis_min < thresh)//最小距离小于阈值
		{
			match_object[j] = idx_max;
		}
	}

	for(j = 0; j < targets_size; j++)//匹配框
	{
		match_rect[j] = -1;
		idx_max = -1;
		dis_min = 1e+9;
		r1 = targets[j].box;
		for(i = 0; i < detClasses[class_id].classes_num; i++)
		{
			if(class_id == targets[j].class_id)
			{
				r2 = detClasses[class_id].box[i];
				//计算两个框的中心距离
				int dis = 0;
				dis = (r1.x + r1.width /2 - r2.x - r2.width / 2) * (r1.x + r1.width /2 - r2.x - r2.width / 2) + (r1.y + r1.height / 2 - r2.y - r2.height / 2) * (r1.y + r1.height / 2 - r2.y - r2.height / 2);
				dis =sqrt(dis);
				if(dis < dis_min)//得到最小距离
				{
					dis_min = dis;
					idx_max = i;
				}
			}
		}
		if(dis_min < thresh)//最小距离小于阈值
		{
			match_rect[j] = idx_max;
		}
	}

	return 1;

}
void RoadEventDetect(ALGCFGS *pCfgs, int target_idx, int event_idx)//道路积水检测
{
	CPoint pt[4];
	//此目标没有被标记为道路事件
	if(pCfgs->road_event_targets[target_idx].cal_event[event_idx] == FALSE)
	{
		if(pCfgs->road_event_targets[target_idx].detected)
		{
			pCfgs->road_event_targets[target_idx].event_continue_num[event_idx]++;
		}
		if(pCfgs->road_event_targets[target_idx].event_continue_num[event_idx] > 5 && pCfgs->road_event_targets[target_idx].detected)//目标在检测区域内，检测达到一定帧数
		{
			pCfgs->road_event_targets[target_idx].cal_event[event_idx] = TRUE;
			pCfgs->road_event_targets[target_idx].event_flag[event_idx] = 1;
		}
	}
	if(pCfgs->road_event_targets[target_idx].event_flag[event_idx] > 0)//已经标记此类事件，当事件一直存在时，传事件
	{
		pt[0].x = pCfgs->road_event_targets[target_idx].box.x;
		pt[0].y = pCfgs->road_event_targets[target_idx].box.y;
		pt[1].x = pCfgs->road_event_targets[target_idx].box.x + pCfgs->road_event_targets[target_idx].box.width;
		pt[1].y = pCfgs->road_event_targets[target_idx].box.y;
		pt[2].x = pCfgs->road_event_targets[target_idx].box.x + pCfgs->road_event_targets[target_idx].box.width;
		pt[2].y = pCfgs->road_event_targets[target_idx].box.y + pCfgs->road_event_targets[target_idx].box.height;
		pt[3].x = pCfgs->road_event_targets[target_idx].box.x;
		pt[3].y = pCfgs->road_event_targets[target_idx].box.y + pCfgs->road_event_targets[target_idx].box.height;
		if(event_idx == ROADWATER)//保存道路积水框
		{
			if(pCfgs->uRoadWaterNum < MAX_EVENT_NUM)
			{
				printf("road water event\n");
				//判断是否是新出现事件
				if(pCfgs->road_event_targets[target_idx].sign_event[event_idx] == 0)
				{
					pCfgs->road_event_targets[target_idx].sign_event[event_idx] = 1;
					pCfgs->RoadWaterBox[pCfgs->uRoadWaterNum].uNewEventFlag = 1;
				}
				else
					pCfgs->RoadWaterBox[pCfgs->uRoadWaterNum].uNewEventFlag = 0;
				pCfgs->RoadWaterBox[pCfgs->uRoadWaterNum].uEventID = pCfgs->road_event_targets[target_idx].target_id;
				memcpy(pCfgs->RoadWaterBox[pCfgs->uRoadWaterNum].EventBox, pt, 4 * sizeof(CPoint));
				pCfgs->RoadWaterBox[pCfgs->uRoadWaterNum].uEventType = ROADWATER;
				pCfgs->uRoadWaterNum++;
			}

		}
		if(event_idx == ROADHOLLOW)//保存道路坑洼框
		{
			if(pCfgs->uRoadHollowNum < MAX_EVENT_NUM)
			{
				printf("road hollow event\n");
				//判断是否是新出现事件
				if(pCfgs->road_event_targets[target_idx].sign_event[event_idx] == 0)
				{
					pCfgs->road_event_targets[target_idx].sign_event[event_idx] = 1;
					pCfgs->RoadHollowBox[pCfgs->uRoadHollowNum].uNewEventFlag = 1;
				}
				else
					pCfgs->RoadHollowBox[pCfgs->uRoadHollowNum].uNewEventFlag = 0;
				pCfgs->RoadHollowBox[pCfgs->uRoadHollowNum].uEventID = pCfgs->road_event_targets[target_idx].target_id;
				memcpy(pCfgs->RoadHollowBox[pCfgs->uRoadHollowNum].EventBox, pt, 4 * sizeof(CPoint));
				pCfgs->RoadHollowBox[pCfgs->uRoadHollowNum].uEventType = ROADHOLLOW;
				pCfgs->uRoadHollowNum++;
			}

		}
		if(event_idx == ROADDAMAGE)//保存道路破损框
		{
			if(pCfgs->uRoadDamageNum < MAX_EVENT_NUM)
			{
				printf("road damage event\n");
				//判断是否是新出现事件
				if(pCfgs->road_event_targets[target_idx].sign_event[event_idx] == 0)
				{
					pCfgs->road_event_targets[target_idx].sign_event[event_idx] = 1;
					pCfgs->RoadDamageBox[pCfgs->uRoadDamageNum].uNewEventFlag = 1;
				}
				else
					pCfgs->RoadDamageBox[pCfgs->uRoadDamageNum].uNewEventFlag = 0;
				pCfgs->RoadDamageBox[pCfgs->uRoadDamageNum].uEventID = pCfgs->road_event_targets[target_idx].target_id;
				memcpy(pCfgs->RoadDamageBox[pCfgs->uRoadDamageNum].EventBox, pt, 4 * sizeof(CPoint));
				pCfgs->RoadDamageBox[pCfgs->uRoadDamageNum].uEventType = ROADDAMAGE;
				pCfgs->uRoadDamageNum++;
			}

		}
		if(event_idx == ROADCRACK)//保存道路裂缝框
		{
			if(pCfgs->uRoadCrackNum < MAX_EVENT_NUM)
			{
				printf("road crack event\n");
				//判断是否是新出现事件
				if(pCfgs->road_event_targets[target_idx].sign_event[event_idx] == 0)
				{
					pCfgs->road_event_targets[target_idx].sign_event[event_idx] = 1;
					pCfgs->RoadCrackBox[pCfgs->uRoadCrackNum].uNewEventFlag = 1;
				}
				else
					pCfgs->RoadCrackBox[pCfgs->uRoadCrackNum].uNewEventFlag = 0;
				pCfgs->RoadCrackBox[pCfgs->uRoadCrackNum].uEventID = pCfgs->road_event_targets[target_idx].target_id;
				memcpy(pCfgs->RoadCrackBox[pCfgs->uRoadCrackNum].EventBox, pt, 4 * sizeof(CPoint));
				pCfgs->RoadCrackBox[pCfgs->uRoadCrackNum].uEventType = ROADCRACK;
				pCfgs->uRoadCrackNum++;
			}

		}


	}
}
//检测道路事件时，不检测其他交通事件，跟踪目标用event_targets,如果都检测，则不能用event_targets
void TrafficRoadAnalysis(ALGCFGS *pCfgs, ALGPARAMS *pParams, int width, int height)
{	
	int i = 0, j = 0, k = 0, l = 0;
	int left = 0, right = 0, top = 0, bottom = 0;
	int match_object[MAX_ROAD_TARGET_NUM];
	int match_rect[MAX_ROAD_TARGET_NUM];
	int match_success = -1;
	bool isInRegion = FALSE;
	CRect targetRect[MAX_ROAD_TARGET_NUM];
	int targetDisXY[MAX_ROAD_TARGET_NUM][2]={ 0 };
	//初始化道路事件检测结果
	pCfgs->uRoadWaterNum = 0;//道路积水
	memset(pCfgs->RoadWaterBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uRoadHollowNum = 0;//道路坑洼
	memset(pCfgs->RoadHollowBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uRoadDamageNum = 0;//道路破损
	memset(pCfgs->RoadDamageBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	pCfgs->uRoadCrackNum = 0;//道路裂缝
	memset(pCfgs->RoadCrackBox, 0, MAX_EVENT_NUM * sizeof(EVENTBOX));
	//设置目标为未检测到
	for( i = 0; i < pCfgs->road_event_targets_size; i++)
	{
		pCfgs->road_event_targets[i].detected = FALSE;
	}
	//对检测框进行处理，把相同类别离得近的合并
	post_process_box_road(pCfgs, 50);
	//分析检测框
	for( i = 0; i < pCfgs->classes; i++)
	{
		//处理需要的类别
		if(strcmp(pCfgs->detClasses[i].names, "js") != 0 && strcmp(pCfgs->detClasses[i].names, "kw") != 0 && strcmp(pCfgs->detClasses[i].names, "ps") != 0 && strcmp(pCfgs->detClasses[i].names, "lf") != 0)
			continue;
		if(pCfgs->detClasses[i].classes_num)
		{
			match_object_rect2(pCfgs->road_event_targets, pCfgs->road_event_targets_size, pCfgs->detClasses, i, match_object, match_rect, 300);

			for( j = 0; j < pCfgs->detClasses[i].classes_num; j++)
			{
				//将检测框限制在检测区域内
				left = MAX(0, pCfgs->detClasses[i].box[j].x * pCfgs->m_iWidth / width);
				right = MIN((pCfgs->detClasses[i].box[j].x + pCfgs->detClasses[i].box[j].width) * pCfgs->m_iWidth / width, pCfgs->m_iWidth - 1);
				top = MAX(0, pCfgs->detClasses[i].box[j].y * pCfgs->m_iHeight / height);
				bottom = MIN((pCfgs->detClasses[i].box[j].y + pCfgs->detClasses[i].box[j].height) * pCfgs->m_iHeight / height, pCfgs->m_iHeight - 1);
				CRect rct;
				rct.x = left;
				rct.y = top;
				rct.width = right - left + 1;
				rct.height = bottom - top + 1;
				isInRegion = FALSE;//初始设为未在事件检测区域
				int inRegion[MAX_EVENT_TYPE] = { 0 };
				if(strcmp(pCfgs->detClasses[i].names, "js") == 0)//道路坑洼
				{
					inRegion[ROADWATER] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, ROADWATER);
				}
				if(strcmp(pCfgs->detClasses[i].names, "kw") == 0)//道路坑洼
				{
					inRegion[ROADHOLLOW] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, ROADHOLLOW);
				}
				if(strcmp(pCfgs->detClasses[i].names, "ps") == 0)//道路破损
				{
					inRegion[ROADDAMAGE] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, ROADDAMAGE);
				}
				if(strcmp(pCfgs->detClasses[i].names, "lf") == 0)//道路裂缝
				{
					inRegion[ROADCRACK] = RectInRegion1(pParams->MaskEventImage, pCfgs->m_iWidth, pCfgs->m_iHeight, rct, ROADCRACK);
				}
				for(k = 0; k < MAX_EVENT_TYPE; k++)
				{
					if(inRegion[k])
					{
						isInRegion = TRUE;//在事件检测区域
						break;
					}
				}
				//if(isInRegion)//在事件检测区域内进行跟踪
				{
					match_success = -1;
					for( k = 0; k < pCfgs->road_event_targets_size; k++)
					{
						if(match_object[j] == k && match_rect[k] == j)
						{
							match_success = 1;
							break;
						}

					}
					if(match_success > 0)//跟踪到,更新检测框
					{
						pCfgs->road_event_targets[k].box = pCfgs->detClasses[i].box[j];
						pCfgs->road_event_targets[k].prob = pCfgs->detClasses[i].prob[j];
						pCfgs->road_event_targets[k].class_id = pCfgs->detClasses[i].class_id;
						strcpy(pCfgs->road_event_targets[k].names, pCfgs->detClasses[i].names);
						pCfgs->road_event_targets[k].detected = TRUE;
					}
					else if(isInRegion && match_success < 0 && pCfgs->road_event_targets_size < MAX_ROAD_TARGET_NUM)//事件区域内，未跟踪到，加入新的目标
					{	
						CTarget nt; 
						Initialize_target(&nt);
						nt.box = pCfgs->detClasses[i].box[j];
						nt.class_id = pCfgs->detClasses[i].class_id;
						nt.prob = pCfgs->detClasses[i].prob[j];
						nt.detected = TRUE;
						nt.target_id = pCfgs->road_event_target_id++;
						nt.start_time = pCfgs->currTime;//目标开始时间
						nt.region_idx = pCfgs->EventDetectCfg.EventRegion[0].uRegionID;//初始化0区域ID
						if(pCfgs->road_event_target_id > 5000)
							pCfgs->road_event_target_id = 1;
						strcpy(nt.names, pCfgs->detClasses[i].names);
						memset(nt.event_continue_num, 0, MAX_EVENT_TYPE * sizeof(int));//初始化事件持续帧数
						memset(nt.event_flag, 0, MAX_EVENT_TYPE * sizeof(int));//初始化事件标记
						memset(nt.cal_event, FALSE, MAX_EVENT_TYPE * sizeof(bool));//初始化各类事件为未计算
						memset(nt.sign_event, 0, MAX_EVENT_TYPE * sizeof(int));//初始化为未标记的事件
						memset(nt.statistic, -1, 5 * sizeof(int));//用于统计运动情况
						pCfgs->road_event_targets[pCfgs->road_event_targets_size] = nt;
						pCfgs->road_event_targets_size++;
					}
				}
			}
		}
	}
	//分析目标
	for(i = 0; i < pCfgs->road_event_targets_size; i++)
	{

		//轨迹数小于3000，直接保存，大于3000，去除旧的
		if(pCfgs->road_event_targets[i].trajectory_num < 3000)
		{

			pCfgs->road_event_targets[i].trajectory[pCfgs->road_event_targets[i].trajectory_num].x = pCfgs->road_event_targets[i].box.x + pCfgs->road_event_targets[i].box.width / 2;
			pCfgs->road_event_targets[i].trajectory[pCfgs->road_event_targets[i].trajectory_num].y = pCfgs->road_event_targets[i].box.y + pCfgs->road_event_targets[i].box.height / 2;
			pCfgs->road_event_targets[i].trajectory[pCfgs->road_event_targets[i].trajectory_num].width = pCfgs->road_event_targets[i].box.width;
			pCfgs->road_event_targets[i].trajectory[pCfgs->road_event_targets[i].trajectory_num].height = pCfgs->road_event_targets[i].box.height;
			pCfgs->road_event_targets[i].trajectory_num++;
		}
		else
		{
			for(j = 0; j < pCfgs->road_event_targets[i].trajectory_num - 1; j++)
			{
				pCfgs->road_event_targets[i].trajectory[j] = pCfgs->road_event_targets[i].trajectory[j + 1];
			}
			pCfgs->road_event_targets[i].trajectory[pCfgs->road_event_targets[i].trajectory_num - 1].x = pCfgs->road_event_targets[i].box.x + pCfgs->road_event_targets[i].box.width / 2;
			pCfgs->road_event_targets[i].trajectory[pCfgs->road_event_targets[i].trajectory_num - 1].y = pCfgs->road_event_targets[i].box.y + pCfgs->road_event_targets[i].box.height / 2;
			pCfgs->road_event_targets[i].trajectory[pCfgs->road_event_targets[i].trajectory_num - 1].width = pCfgs->road_event_targets[i].box.width;
			pCfgs->road_event_targets[i].trajectory[pCfgs->road_event_targets[i].trajectory_num - 1].height = pCfgs->road_event_targets[i].box.height;
		}
		//检测到，并更新速度
		if(pCfgs->road_event_targets[i].detected)
		{
			pCfgs->road_event_targets[i].lost_detected = 0;
		}
		else//未检测到
		{
			pCfgs->road_event_targets[i].lost_detected++;
			pCfgs->road_event_targets[i].box.x += pCfgs->road_event_targets[i].vx;
			pCfgs->road_event_targets[i].box.y += pCfgs->road_event_targets[i].vy;
		}
		//当目标在视频存在时间太长或长时间没有检测到或离开图像，删除目标
		if(pCfgs->road_event_targets[i].continue_num > 5000 || pCfgs->road_event_targets[i].lost_detected > 5 ||((pCfgs->road_event_targets[i].box.x < 5 || pCfgs->road_event_targets[i].box.y < 5 || (pCfgs->road_event_targets[i].box.x + pCfgs->road_event_targets[i].box.width) > (width - 5) || (pCfgs->road_event_targets[i].box.y + pCfgs->road_event_targets[i].box.height) > (height - 5))&& pCfgs->road_event_targets[i].lost_detected > 2))
		{
			DeleteTarget(&pCfgs->road_event_targets_size, &i, pCfgs->road_event_targets);
			continue;

		}
		//判断目标属于哪个事件区域
		for(k = 0; k < pCfgs->EventDetectCfg.uEventRegionNum; k++)
		{
			int inRegionRatio = RectInRegion(pParams->MaskEventIDImage, pCfgs, width, height, pCfgs->road_event_targets[i].box, pCfgs->EventDetectCfg.EventRegion[k].uRegionID);
			if(inRegionRatio > 10)
			{
				pCfgs->road_event_targets[i].region_idx = pCfgs->EventDetectCfg.EventRegion[k].uRegionID;
				break;
			}
		}
		//分析道路事件
		if(strcmp(pCfgs->road_event_targets[i].names, "js") == 0)//道路坑洼
		{
			RoadEventDetect(pCfgs, i, ROADWATER);
		}
		if(strcmp(pCfgs->road_event_targets[i].names, "kw") == 0)//道路坑洼
		{
			RoadEventDetect(pCfgs, i, ROADHOLLOW);
		}
		if(strcmp(pCfgs->road_event_targets[i].names, "ps") == 0)//道路破损
		{
			RoadEventDetect(pCfgs, i, ROADDAMAGE);
		}
		if(strcmp(pCfgs->road_event_targets[i].names, "lf") == 0)//道路裂缝
		{
			RoadEventDetect(pCfgs, i, ROADCRACK);
		}
		//将目标框保存起来，用于事件检测
		targetRect[i] = pCfgs->road_event_targets[i].box;
		//保存目标的运动情况
		int continue_num = pCfgs->road_event_targets[i].trajectory_num - 100;
		continue_num = (continue_num < 0)? 0 : continue_num;
		int dx = pCfgs->road_event_targets[i].box.x + pCfgs->road_event_targets[i].box.width / 2 - pCfgs->road_event_targets[i].trajectory[continue_num].x;
		int dy = pCfgs->road_event_targets[i].box.y + pCfgs->road_event_targets[i].box.height / 2 - pCfgs->road_event_targets[i].trajectory[continue_num].y;
		dx = (dx < 0)? -dx : dx;
		dy = (dy < 0)? -dy : dy;
		targetDisXY[i][0] = dx;
		targetDisXY[i][1] = dy;
		pCfgs->road_event_targets[i].continue_num++;
	}
}
bool ArithInitEvent(ALGCFGS *pCfgs, mEventInfo	EventDetectCfg, ALGPARAMS *pParams)
{
	bool bInit = FALSE;
#ifdef DETECT_GPU
	//加载网络参数
	if(pCfgs->net_params->net == NULL)
	{
		LoadNetParams(pCfgs->net_params, 0);
	}
#endif
	//交通事件区域初始化
	CfgEventRegion(EventDetectCfg, pCfgs, pParams);
	return bInit;
}
Uint16 ArithProcEvent(IplImage* img, ALGCFGS *pCfgs, ALGPARAMS *pParams, char* videoName, char* resultfile)
{
	Int32 i, j;
	int result[6 * MAX_DETECTION_NUM] = { 0 };
	int nboxes = 0;
	unsigned char* pInFrameBuf;
	if(img->width <= 0 || img->height <= 0)
	{
		printf("img cannot be zero!\n");
		return 0;
	}
	//处理数据大小
	pCfgs->m_iHeight = (img->height > FULL_ROWS)? FULL_ROWS : img->height;
	pCfgs->m_iWidth = (img->width > FULL_COLS)? FULL_COLS : img->width;

	//gbr转yuv420
	IplImage* imgBGR = cvCreateImage(cvSize(pCfgs->m_iWidth, pCfgs->m_iHeight), IPL_DEPTH_8U, 3);
	IplImage* imgYUV = cvCreateImage(cvSize(pCfgs->m_iWidth, pCfgs->m_iHeight * 3 / 2), IPL_DEPTH_8U, 1);
	if(img->width != pCfgs->m_iWidth || img->height != pCfgs->m_iHeight)
	{
		cvResize(img, imgBGR, CV_INTER_LINEAR);
	}
	else
	{
		cvCopy(img, imgBGR, NULL); 
	}
	cvCvtColor(imgBGR, imgYUV, CV_BGR2YUV_I420);
	pInFrameBuf = (unsigned char *)imgYUV->imageData;
	memcpy((void *)pParams->CurrQueueImage, (void *)pInFrameBuf, pCfgs->m_iWidth * pCfgs->m_iHeight);//灰度图像
	//printf("process,%d,%d\n",pCfgs->m_iHeight,pCfgs->m_iWidth);
	//设置交通事件掩模图像
	if(pCfgs->bMaskEventImage == FALSE)
	{
		printf("mask event image\n");
		MaskEventImage(pCfgs, pParams, img->width, img->height);
		pCfgs->bMaskEventImage = TRUE;
	}
	//得到系统时间
	gettimeofday(&pCfgs->time_end, NULL);
	if(pCfgs->gThisFrameTime == 0)
		pCfgs->currTime = 0;
	else
		pCfgs->currTime += (pCfgs->time_end.tv_sec - pCfgs->time_start.tv_sec) + (pCfgs->time_end.tv_usec - pCfgs->time_start.tv_usec)/1000000.0;
	gettimeofday(&pCfgs->time_start, NULL);
	//检测
#ifdef DETECT_GPU
	if(pCfgs->net_params->net)
	{
		nboxes = YoloArithDetect(img, pCfgs->net_params, result);//yolo检测
	}
#endif
	printf("frame = %d,nboxes = %d\n",pCfgs->gThisFrameTime, nboxes);
	//分析检测结果
	ProcessDetectBox(pCfgs, result, nboxes);
	//对交通事件进行分析
	TrafficEventAnalysis(pCfgs, pParams, img->width, img->height);
	pCfgs->gThisFrameTime++;
	if(pCfgs->EventNum == 0 && pCfgs->EventState == 1)//事件结束，写事件到文件中
	{
		//pCfgs->EventEndTime = pCfgs->gThisFrameTime;
		pCfgs->EventState = 0;
		FILE* fp = fopen(resultfile,"a");
		if(fp == NULL)
		{
			perror("fail to read");
			exit (1) ;
		}
		if(fp)
		{
			fputs("{\"file\":\"",fp);
			fprintf(fp,"%s",videoName);
			if(pCfgs->eventType  == NO_PEDESTRIANTION)
				fputs("\",\"catalog\":\"people\",\"event\":true,\"btime\" : ",fp);
			if(pCfgs->eventType  == STOP_INVALID)
				fputs("\",\"catalog\":\"stop\",\"event\":true,\"btime\" : ",fp);
			if(pCfgs->eventType  == REVERSE_DRIVE)
				fputs("\",\"catalog\":\"wrongway\",\"event\":true,\"btime\" : ",fp);

			if(pCfgs->eventType  == DROP)
				fputs("\",\"catalog\":\"drop\",\"event\":true,\"btime\" : ",fp);
			fprintf(fp,"%d",pCfgs->EventBeginTime/ pCfgs->video_fps); 
			fputs(" ,\"etime\" :",fp);
			fprintf(fp,"%d",pCfgs->EventEndTime/ pCfgs->video_fps);
			fputs(" }\n",fp);
		}
		fclose(fp);
	}

	if(pCfgs->EventNum)//存在事件
	{
		pCfgs->HaveEvent = TRUE;
		for( i = 0; i < pCfgs->EventNum; i++)
		{
			if(pCfgs->EventInfo[i].flag == 1)//写数据到文件中
			{
				//删除相应的事件
				for(j = i + 1; j < pCfgs->EventNum; j++)
				{
					pCfgs->EventInfo[j - 1] = pCfgs->EventInfo[j];
				}
				pCfgs->EventNum = pCfgs->EventNum - 1;
			}
		}

	}
	if(imgBGR)
	{
		cvReleaseImage(&imgBGR);
		imgBGR = NULL;
	}
	if(imgYUV)
	{
		cvReleaseImage(&imgYUV);
		imgYUV = NULL;
	}
	return 1;
}
