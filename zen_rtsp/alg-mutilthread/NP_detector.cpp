//非机动车多成员检测
#include "NP_detector.h"
#ifdef DETECT_GPU//GPU
#include "yolo_detector.h"
#else//ncs
#include "NCS_detector.h"
#endif
#define max(a,b) (((a)>(b)) ? (a):(b))
#define min(a,b) (((a)>(b)) ? (b):(a))
//#define SVAE_PIC
#ifdef SVAE_PIC
Mat preImg;
Mat prePreImg;
#endif // SVAE_PIC

Uint16 NPDetector(Mat img, NonMotorInfo* NPDetectInfo, ALGCFGS *pCfgs)
{
	int result[6 * MAX_DETECTION_NUM] = { 0 };
	Uint16 nboxes = 0;
	Uint16 nonMotorNum = 0;
	int width = img.cols;//图像宽度
	int height = img.rows;//图像高度
	int channel = img.channels();//图像通道数
	if(width <= 0 || height <= 0)//没有图像数据
	{
		printf("img cannot be zero!\n");
		return 0;
	}
	//处理数据大小
	pCfgs->m_iHeight = (height > FULL_ROWS)? FULL_ROWS : height;
	pCfgs->m_iWidth = (width > FULL_COLS)? FULL_COLS : width;
	pCfgs->img_height = height;//图像的宽高
	pCfgs->img_width = width;
	pCfgs->gThisFrameTime++;
	//视频对比度检测和异常检测
	if(pCfgs->gThisFrameTime % 50 == 1 && channel == 3)//三通道
	{
		int resize_width = (width / 2) * 2;
		int resize_height = (height / 2) * 2;
		//bgr转为yuv
		Mat imgYUV, resizeImage;
		resize(img, resizeImage, Size(resize_width, resize_height), 0, 0, INTER_LINEAR);//缩放
		cvtColor(resizeImage, imgYUV, CV_BGR2YUV_I420);
		unsigned char* pInFrameBuf = (unsigned char *)imgYUV.data;
		unsigned char* pInuBuf = (unsigned char *)imgYUV.data + resize_width * resize_height;
		unsigned char* pInvBuf = (unsigned char *)imgYUV.data + resize_width * resize_height * 5 / 4;
		//能见度计算
		int thr = 8;//阈值
		pCfgs->up_visib_value++;
		pCfgs->fuzzydegree = fuzzy(pInFrameBuf, resize_width, resize_height);
		for (int j = VISIB_LENGTH - 1; j > 0; j--)
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
		if(Color_deviate(pInuBuf, pInvBuf, resize_width / 4, resize_height / 4))
			pCfgs->abnormal_time++;
		else
			pCfgs->abnormal_time = 0;	   
		pCfgs->fuzzyflag = (pCfgs->abnormal_time > 5)? TRUE : FALSE;
	}

#ifdef DETECT_GPU
	IplImage* image = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, channel);
	memcpy(image->imageData, img.data, width * height * channel);
	if(pCfgs->net_params->net)
	{
		nboxes = YoloArithDetect(image, pCfgs->net_params, result);//yolo检测
	}
	cvReleaseImage(&image);
#else
	nboxes = NCSArithDetect(img, pCfgs, result);//NCS检测
#endif
	//分析检测结果
	ProcessDetectBox(pCfgs, result, nboxes);
#ifdef SVAE_PIC
	if(pCfgs->gThisFrameTime > 2)
	{
		Mat imgCopy;
		prePreImg.copyTo(imgCopy);
		for(int i = 0; i < nboxes; i++)
		{
			cv::rectangle(imgCopy, cv::Rect(result[6 * i + 2], result[6 * i + 3], result[6 * i + 4], result[6 * i + 5]), cv::Scalar(255, 255, 255), 3, 8, 0 );
		}
		for(int i = 0; i < pCfgs->classes; i++)
		{
			for(int j = 0; j < pCfgs->detClasses[i].classes_num; j++)
			{
				if(i == 0)
					cv::rectangle(imgCopy, cv::Rect(pCfgs->detClasses[i].box[j].x, pCfgs->detClasses[i].box[j].y, pCfgs->detClasses[i].box[j].width, pCfgs->detClasses[i].box[j].height), cv::Scalar(255, 255, 255), 3, 8, 0 );
				else if(i == 1)
					cv::rectangle(imgCopy, cv::Rect(pCfgs->detClasses[i].box[j].x, pCfgs->detClasses[i].box[j].y, pCfgs->detClasses[i].box[j].width, pCfgs->detClasses[i].box[j].height), cv::Scalar(255, 0, 0), 3, 8, 0 );
				else if(i == 2)
					cv::rectangle(imgCopy, cv::Rect(pCfgs->detClasses[i].box[j].x, pCfgs->detClasses[i].box[j].y, pCfgs->detClasses[i].box[j].width, pCfgs->detClasses[i].box[j].height), cv::Scalar(0, 255, 0), 3, 8, 0 );
				else if(i == 3)
					cv::rectangle(imgCopy, cv::Rect(pCfgs->detClasses[i].box[j].x, pCfgs->detClasses[i].box[j].y, pCfgs->detClasses[i].box[j].width, pCfgs->detClasses[i].box[j].height), cv::Scalar(0, 0, 255), 3, 8, 0 );
				else
					cv::rectangle(imgCopy, cv::Rect(pCfgs->detClasses[i].box[j].x, pCfgs->detClasses[i].box[j].y, pCfgs->detClasses[i].box[j].width, pCfgs->detClasses[i].box[j].height), cv::Scalar(0, 0, 0), 3, 8, 0 );
			}

		}
		char strname[100];
		sprintf(strname, "pic/%d.jpg", pCfgs->gThisFrameTime);
		cv::imwrite(strname, imgCopy);
	}
	if(pCfgs->gThisFrameTime == 1)
	{
		img.copyTo(preImg);
		img.copyTo(prePreImg);
	}
	else
	{
		preImg.copyTo(prePreImg);
		img.copyTo(preImg);
	}
#endif
	//分析多乘员检测
	nonMotorNum = AnalysisNPDetect(pCfgs, NPDetectInfo);
/*#ifdef SVAE_PIC
	if(pCfgs->gThisFrameTime > 2)
	{
		Mat imgCopy;
		prePreImg.copyTo(imgCopy);
		for(int i = 0; i < nonMotorNum; i++)
		{
			cv::rectangle(imgCopy, cv::Rect(NPDetectInfo[i].nonMotorBox.x, NPDetectInfo[i].nonMotorBox.y, NPDetectInfo[i].nonMotorBox.width, NPDetectInfo[i].nonMotorBox.height), cv::Scalar(255, 255, 255), 3, 8, 0 );
		}
		char strname[100];
		sprintf(strname, "pic/%d.jpg", pCfgs->gThisFrameTime);
		cv::imwrite(strname, imgCopy);
	}
	if(pCfgs->gThisFrameTime == 1)
	{
		img.copyTo(preImg);
		img.copyTo(prePreImg);
	}
	else
	{
		preImg.copyTo(prePreImg);
		img.copyTo(preImg);
	}
#endif*/
	return nonMotorNum;

}
Uint16 AnalysisNPDetect(ALGCFGS* pCfgs, NonMotorInfo* NPDetectInfo)
{
	int i = 0, j = 0, k = 0;
	CDetBox* twoPersonCalss;
	CDetBox* threePersonCalss;
	CDetBox* helmetClass;
	Uint16 helmetNum = 0;
	Uint16 nonMotorNum = 0;
	for(i = 0; i < pCfgs->classes; i++)
	{
		if(strcmp(pCfgs->detClasses[i].names, "2_person") == 0)//2人
		{
			twoPersonCalss = &pCfgs->detClasses[i];
		}
		if(strcmp(pCfgs->detClasses[i].names, "3_person") == 0)//3人
			threePersonCalss = &pCfgs->detClasses[i];
		if(strcmp(pCfgs->detClasses[i].names, "helmet") == 0)//安全帽
			helmetClass = &pCfgs->detClasses[i];
	}
	for(i = 0; i < pCfgs->classes; i++)
	{
		if(strcmp(pCfgs->detClasses[i].names, "rider") != 0)//非机动车
			continue;
		nonMotorNum = pCfgs->detClasses[i].classes_num;
		for(j = 0; j < pCfgs->detClasses[i].classes_num; j++)
		{
			CRect nonMotorBox = pCfgs->detClasses[i].box[j];
			if((nonMotorBox.y + nonMotorBox.height) < pCfgs->img_height / 6)//图像上端的不处理，防止误检
				continue;
			NPDetectInfo[j].nonMotorBox = nonMotorBox;//非机动车检测框
			if(detect_riderNum(nonMotorBox, threePersonCalss->box, threePersonCalss->classes_num))//先检测三人是否相交
			{
				NPDetectInfo[j].riderNum = 3;//3人
			}
			else
			{
				if(detect_riderNum(nonMotorBox, twoPersonCalss->box, twoPersonCalss->classes_num))//检测2个人是否相交
				{
					NPDetectInfo[j].riderNum = 2;//2人
				}
				else
				{
					NPDetectInfo[j].riderNum = 1;//1人
				}
			}
			//判断是否超载
			NPDetectInfo[j].overLoad = (NPDetectInfo[j].riderNum > 2)? TRUE : FALSE;
			//判断是否带帽，初始未带帽
			helmetNum = 0;
			NPDetectInfo[j].hasHelmet = FALSE;
			bool hasHelmet = FALSE;
			for(k = 0; k < helmetClass->classes_num; k++)
			{
				CRect helmetBox = helmetClass->box[k];
				hasHelmet = detect_helmet(nonMotorBox, helmetBox);
				if(hasHelmet)//带帽
					NPDetectInfo[j].helmetBox[helmetNum++] = helmetClass->box[k];//安全帽检测框
			}
			if(helmetNum >= NPDetectInfo[j].riderNum)//带帽数量大于等于骑行人数为带帽
			{
				NPDetectInfo[j].hasHelmet = TRUE;
			}
			NPDetectInfo[j].helmetNum = helmetNum;

		}
	}
	printf("nonMotorNum = %d\n", nonMotorNum);
	return nonMotorNum;
}
bool detect_riderNum(CRect nonMotorBox, CRect* riderBox, int boxNum)//检测骑行人数
{
	int i = 0;
	int nonMotorR = nonMotorBox.x + nonMotorBox.width;
	int nonMotorB = nonMotorBox.y + nonMotorBox.height;
	int riderR, riderB;
	//判断检测框是否相交
	for(i = 0; i < boxNum; i++)
	{
		riderR = riderBox[i].x + riderBox[i].width;
		riderB = riderBox[i].y + riderBox[i].height;
		if((min(nonMotorR, riderR) - max(nonMotorBox.x, riderBox[i].x) > min(nonMotorBox.width, riderBox[i].width) / 2)&& \
			(min(nonMotorB, riderB) - max(nonMotorBox.y, riderBox[i].y) > min(nonMotorBox.height, riderBox[i].height) / 2))//两个检测框相交
			return TRUE;
	}
	return FALSE;

}
bool detect_helmet(CRect nonMotorBox, CRect helmetBox)//是否带帽
{
	int nonMotorR = nonMotorBox.x + nonMotorBox.width;
	int nonMotorB = nonMotorBox.y + nonMotorBox.height;
	int helmetR = helmetBox.x + helmetBox.width;
	int helmetB = helmetBox.y + helmetBox.height;
	if((min(nonMotorR, helmetR) - max(nonMotorBox.x, helmetBox.x) > min(nonMotorBox.width, helmetBox.width) / 2)&& \
		(min(nonMotorB, helmetB) - max(nonMotorBox.y, helmetBox.y) > min(nonMotorBox.height, helmetBox.height) / 2))//两个检测框相交
		return TRUE;
	return FALSE;

}
