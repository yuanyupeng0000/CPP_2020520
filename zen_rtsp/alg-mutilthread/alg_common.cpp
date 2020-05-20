/////basic function 
#include "stdio.h"
#include "stdlib.h" 
#include "m_arith.h"
#include <dirent.h>
#define max(a,b) (((a)>(b)) ? (a):(b))
#define min(a,b) (((a)>(b)) ? (b):(a))
//获得检测类别名
char** get_labels(char* filename, Uint16& classes_num)
{
	char** label_names = (char**)malloc(MAX_CLASSES * sizeof(char*));//类别名
	char buf[1024] = { 0 };
	int len = 0;
	classes_num = 0;
	FILE *file = fopen(filename, "r");
	if(!file)
	{
		printf("load class names failed!");
		return NULL;
	}
	while(fgets(buf, 1024, file))//逐行读取类别名
	{
		len = strlen(buf);
		if(strcmp(buf, "\n") == 0 || strcmp(buf, "\r") == 0)//空行
			continue;
		if(buf[len - 1] == '\n') buf[len - 1] = '\0';
		label_names[classes_num] = (char*)malloc(len * sizeof(char));
		strncpy(label_names[classes_num], buf, len);
		classes_num++;
	}
	fclose(file);
	return label_names;
}
//判断点是否在多边形内部
// Globals which should be set before calling this function:
//
// int    polySides  =  how many cornersthe polygon has
// float  polyX[]    =  horizontalcoordinates of corners
// float  polyY[]    =  verticalcoordinates of corners
// float  x,y       =  point to be tested
//
// (Globals are used in this example for purposes of speed.  Change as
// desired.)
//
//  Thefunction will return YES if the point x,y is inside the polygon, or
//  NOif it is not.  If the point is exactly on the edge of the polygon,
// then the function may return YES or NO.
//
// Note that division by zero is avoided because the division is protected
//  bythe "if" clause which surrounds it.

//bool pointInPolygon(int polySides, float  polyX[], float  polyY[], float  x, float y) 
//{
//	int   i, j = polySides - 1 ;
//	bool  oddNodes = FALSE;
//
//	for (i = 0; i < polySides; i++) 
//	{
//		if(polyY[i] < y && polyY[j] >= y
//			||  polyY[j] < y && polyY[i] >= y) 
//		{
//				if(polyX[i] + (y - polyY[i]) / (polyY[j] - polyY[i]) * (polyX[j] - polyX[i]) < x)
//				{
//					oddNodes =!oddNodes;
//				}
//		}
//		j = i; 
//	}
//
//	return oddNodes; 
//}
bool pointInPolygon(int polySides, float  polyX[], float  polyY[], float  x, float y) 
{
	int   i, j = polySides - 1;
	bool  oddNodes = FALSE;

	for (i = 0; i < polySides; i++) 
	{
		if((polyY[i] < y && polyY[j] >= y
			||   polyY[j] < y && polyY[i] >= y)
			&&  (polyX[i] <= x || polyX[j] <= x))
		{
				/*if(polyX[i] + (y - polyY[i]) / (polyY[j] - polyY[i]) * (polyX[j] - polyX[i]) < x) 
				{
					oddNodes =! oddNodes;
				}*/
                oddNodes ^= (polyX[i] + (y - polyY[i]) / (polyY[j] - polyY[i]) * (polyX[j] - polyX[i]) < x);

		}
		j = i;
	}

	return oddNodes; 
}



//若点a大于点b,即点a在点b顺时针方向,返回true,否则返回false
bool PointCmp(CPoint a,CPoint b,CPoint center)
{
	if (a.x >= 0 && b.x < 0)
		return true;
	if (a.x == 0 && b.x == 0)
		return a.y > b.y;
	//向量OA和向量OB的叉积
	int det = (a.x - center.x) * (b.y - center.y) - (b.x - center.x) * (a.y - center.y);
	if (det < 0)
		return true;
	if (det > 0)
		return false;
	//向量OA和向量OB共线，以距离判断大小
	int d1 = (a.x - center.x) * (a.x - center.x) + (a.y - center.y) * (a.y - center.y);
	int d2 = (b.x - center.x) * (b.x - center.y) + (b.y - center.y) * (b.y - center.y);
	return d1 > d2;
}
//校正区域四个点的坐标顺序，按顺时针 0 1 2 3
void CorrectRegionPoint(CPoint* ptCorner)
{
	int i = 0, j = 0;
	//计算重心
	CPoint center;
	double x = 0, y = 0;
	for (i = 0; i < 4; i++)
	{
		x += ptCorner[i].x;
		y += ptCorner[i].y;
	}
	center.x = (int)x / 4;
	center.y = (int)y / 4;
	//冒泡排序
	for(i = 0; i < 3; i++)
	{
		for (j = 0;j < 3 - i;j++)
		{
			if (PointCmp(ptCorner[j], ptCorner[j+1], center))
			{
				CPoint tmp = ptCorner[j];
				ptCorner[j] = ptCorner[j + 1];
				ptCorner[j + 1] = tmp;
			}
		}
	}
	int sum_y[4];
	int minY = 0, startID = 0;
	CPoint ptNewCorner[4];
	sum_y[0] = ptCorner[0].y + ptCorner[1].y;
	sum_y[1] = ptCorner[1].y + ptCorner[2].y;
	sum_y[2] = ptCorner[2].y + ptCorner[3].y;
	sum_y[3] = ptCorner[3].y + ptCorner[0].y;
	minY = sum_y[0];
	//确定端点位置
	for(i = 1; i < 4; i++)
	{
		if(sum_y[i] < minY)
		{
			startID = i;
			minY = sum_y[i];
		}
	}
	if(startID == 0)//0 - 1 - 2 -3不进行操作
	{
		ptNewCorner[0] = ptCorner[0];
		ptNewCorner[1] = ptCorner[1];
		ptNewCorner[2] = ptCorner[2];
		ptNewCorner[3] = ptCorner[3];
	}
	if(startID == 1)//1 - 2 - 3 - 0
	{
		ptNewCorner[0] = ptCorner[1];
		ptNewCorner[1] = ptCorner[2];
		ptNewCorner[2] = ptCorner[3];
		ptNewCorner[3] = ptCorner[0];
	}
	if(startID == 2)//2 - 3 - 0 - 1
	{
		ptNewCorner[0] = ptCorner[2];
		ptNewCorner[1] = ptCorner[3];
		ptNewCorner[2] = ptCorner[0];
		ptNewCorner[3] = ptCorner[1];
	}
	if(startID == 3)//3 - 0 - 1 - 2
	{
		ptNewCorner[0] = ptCorner[3];
		ptNewCorner[1] = ptCorner[0];
		ptNewCorner[2] = ptCorner[1];
		ptNewCorner[3] = ptCorner[2];
	}
	memcpy( (void*)ptCorner, (void*)ptNewCorner, 4 * sizeof(CPoint) );

}
//校正区域四个点的坐标顺序，按顺时针 0 1 2 3
/*void CorrectRegionPoint(CPoint* ptCorner)
{
	CPoint temp;
	if(ptCorner[0].y > ptCorner[3].y)//左上的y值大于左下
	{
		temp = ptCorner[0];
		ptCorner[0] = ptCorner[3];
		ptCorner[3] = temp;
	}
	if(ptCorner[0].y > ptCorner[2].y)//左上的y值大于右下
	{
		temp = ptCorner[0];
		ptCorner[0] = ptCorner[2];
		ptCorner[2] = temp;
	}
	if(ptCorner[1].y > ptCorner[3].y)//右上的y值大于左下
	{
		temp = ptCorner[1];
		ptCorner[1] = ptCorner[3];
		ptCorner[3] = temp;
	}
	if(ptCorner[1].y > ptCorner[2].y)//右上的y值大于右下
	{
		temp = ptCorner[1];
		ptCorner[1] = ptCorner[2];
		ptCorner[2] = temp;
	}
	if(ptCorner[0].x > ptCorner[1].x)//左上的x值大于右上
	{
		temp = ptCorner[0];
		ptCorner[0] = ptCorner[1];
		ptCorner[1] = temp;
	}
	if(ptCorner[3].x > ptCorner[2].x)//左下的x值大于右下
	{
		temp = ptCorner[3];
		ptCorner[3] = ptCorner[2];
		ptCorner[2] = temp;
	}

}*/
//判断一个点是否在四边形里
bool isPointInRect(CPoint pt, CPoint mLBPoint, CPoint mLTPoint, CPoint mRTPoint, CPoint mRBPoint) 
{
	int a = (mLTPoint.x - mLBPoint.x) * (pt.y - mLBPoint.y) - (mLTPoint.y - mLBPoint.y) * (pt.x - mLBPoint.x);
	int b = (mRTPoint.x - mLTPoint.x) * (pt.y - mLTPoint.y) - (mRTPoint.y - mLTPoint.y) * (pt.x - mLTPoint.x);
	int c = (mRBPoint.x - mRTPoint.x) * (pt.y - mRTPoint.y) - (mRBPoint.y - mRTPoint.y) * (pt.x - mRTPoint.x);
	int d = (mLBPoint.x - mRBPoint.x) * (pt.y - mRBPoint.y) - (mLBPoint.y - mRBPoint.y) * (pt.x - mRBPoint.x);
	if((a > 0 && b > 0 && c > 0 && d > 0) || (a < 0 && b < 0 && c < 0 && d < 0)) 
	{
		return TRUE;
	}

	return FALSE; 
}
bool isLineIntersect(CPoint pt1, CPoint pt2, CPoint pt3, CPoint pt4)
{
	if(min(pt1.x, pt2.x) > max(pt3.x, pt4.x) || min(pt1.y, pt2.y) > max(pt3.y, pt4.y) || 
		max(pt1.x, pt2.x) < min(pt3.x, pt4.x) || max(pt1.y, pt2.y) < min(pt3.y, pt4.y))//矩形排斥实验
		return FALSE;
	int a, b, c, d; 
	a = (pt1.x - pt2.x) * (pt3.y - pt1.y) - (pt1.y - pt2.y) * (pt3.x - pt1.x);//跨立实验
	b = (pt1.x - pt2.x) * (pt4.y - pt1.y) - (pt1.y - pt2.y) * (pt4.x - pt1.x); 
	c = (pt3.x - pt4.x) * (pt1.y - pt3.y) - (pt3.y - pt4.y) * (pt1.x - pt3.x); 
	d = (pt3.x - pt4.x) * (pt2.y - pt3.y) - (pt3.y - pt4.y) * (pt2.x - pt3.x); 
	if(a * b <= 0 && c * d <= 0)
		return TRUE;
	else
		return FALSE;
}
//判断一个线段是否与矩形框相交
bool isLineIntersectRect(CPoint pt1, CPoint pt2, CRect rct)
{
	CPoint mLTPoint, mRTPoint, mRBPoint, mLBPoint;
	mLTPoint.x = rct.x;//左上
	mLTPoint.y = rct.y;
	mRTPoint.x = rct.x + rct.width;//右上
	mRTPoint.y = rct.y;
	mLBPoint.x = rct.x;//左下
	mLBPoint.y = rct.y + rct.height;
	mRBPoint.x = rct.x + rct.width;//右下
	mRBPoint.y = rct.y + rct.height;
	//线段在矩形内部
	if(min(pt1.x, pt2.x) > mLTPoint.x && max(pt1.x, pt2.x) < mRTPoint.x && min(pt1.y, pt2.y) > mLTPoint.y && max(pt1.y, pt2.y) < mLBPoint.y)//线段在矩形内部
		return TRUE;
	if(isLineIntersect(pt1, pt2, mLTPoint, mLBPoint) || isLineIntersect(pt1, pt2, mLBPoint, mRBPoint) || 
		isLineIntersect(pt1, pt2, mRTPoint, mRBPoint) || isLineIntersect(pt1, pt2, mLTPoint, mRTPoint))//判断线段与矩形相交
		return TRUE;
	return FALSE;
}
//得到检测框是否在检测区域
/*int RectInRegion(unsigned char* maskImage, int width, int height, CRect rct, int idx)
{
	int InRegionRatio = 0;
	int i = 0, j = 0;
	int num = 0;
	int val = 0;
	unsigned char* p;
	for(i = rct.y; i < (rct.y + rct.height); i++)
	{
		p = maskImage + i * width;
		for(j = rct.x; j < (rct.x + rct.width); j++)
		{
			val = *(p + j);
			if(val & (1 << idx))
			{
				num++;
			}
		}
	}
	InRegionRatio = (float)(num * 100) / (float) (rct.width * rct.height);
	return InRegionRatio;
}*/
int RectInRegion(unsigned char* maskImage, ALGCFGS *pCfgs, int width, int height, CRect rct, int idx)
{
	int InRegionRatio = 0;
	int i = 0, j = 0;
	int num = 0, totalnum = 0;
	int val = 0;
	unsigned char* p;
	int left = 0, right = 0, top = 0, bottom = 0;
	left = max(0, rct.x * pCfgs->m_iWidth / width);
	right = min(pCfgs->m_iWidth, (rct.x + rct.width) * pCfgs->m_iWidth / width);
	top = max(0, rct.y * pCfgs->m_iHeight / height);
	bottom = min(pCfgs->m_iHeight, (rct.y + rct.height)* pCfgs->m_iHeight / height);
	for(i = top; i < bottom; i++)
	{
		p = maskImage + i * pCfgs->m_iWidth;
		for(j = left; j < right; j++)
		{
			val = *(p + j);
			if(val == idx)
			{
				num++;
			}
			totalnum++;
		}
	}
	if(totalnum > 0)
		InRegionRatio = num * 100 / totalnum;
	return InRegionRatio;
}
//设置检测线
int SetLine(ALGCFGS* pCfgs, CPoint ptDetLine[][2])
{
	int i = 0;
	bool bDetPersonFlow = FALSE;
	//检测线未设置或者错误
	for( i = 0; i < pCfgs->uDetectRegionNum; i++)
	{
		if((ptDetLine[i][0].x <= 0 && ptDetLine[i][0].y <= 0 && ptDetLine[i][1].x <= 0 && ptDetLine[i][1].y <= 0) || (ptDetLine[i][0].x == ptDetLine[i][1].x  && ptDetLine[i][0].y == ptDetLine[i][1].y))
		{
			printf("detect line error\n");
			pCfgs->detLineParm[i].k = 0;
			pCfgs->detLineParm[i].b = 0;
			pCfgs->detLineParm[i].line_vertical = 0;
			continue;
		}
		printf("line %d:[%d,%d],[%d,%d]\n",ptDetLine[i][0].x,ptDetLine[i][0].y,ptDetLine[i][1].x,ptDetLine[i][1].y);
		pCfgs->detLineParm[i].pt[0] = ptDetLine[i][0];
		pCfgs->detLineParm[i].pt[1] = ptDetLine[i][1];
		pCfgs->detLineParm[i].detLeft = min(ptDetLine[i][0].x, ptDetLine[i][1].x);
		pCfgs->detLineParm[i].detTop = min(ptDetLine[i][0].y, ptDetLine[i][1].y);
		pCfgs->detLineParm[i].detRight = max(ptDetLine[i][0].x, ptDetLine[i][1].x);
		pCfgs->detLineParm[i].detBottom = max(ptDetLine[i][0].y, ptDetLine[i][1].y);

		//计算检测线的斜率和截距
		if(abs(ptDetLine[i][0].x - ptDetLine[i][1].x) < 5)//检测线垂直
		{
			pCfgs->detLineParm[i].k = 0;
			pCfgs->detLineParm[i].b = (ptDetLine[i][0].y + ptDetLine[i][1].y) / 2; 
			pCfgs->detLineParm[i].line_vertical = 1;
		}
		else
		{
			pCfgs->detLineParm[i].k = (float)(ptDetLine[i][1].y - ptDetLine[i][0].y) / (float)(ptDetLine[i][1].x - ptDetLine[i][0].x);
			pCfgs->detLineParm[i].b = ptDetLine[i][1].y - pCfgs->detLineParm[i].k * ptDetLine[i][1].x;
			pCfgs->detLineParm[i].line_vertical = 0;
		}
		bDetPersonFlow = TRUE;
	}
	//检测线设置成功，检测行人方向流量数
	pCfgs->bDetPersonFlow = bDetPersonFlow;
	return 1;
}
//分方向得到检测线处的行人数
void get_object_num(ALGCFGS* pCfgs, int target_idx, int region_idx)
{
	int x = 0, y = 0;
	int direction = 0;
	x = pCfgs->objPerson[target_idx].box.x + pCfgs->objPerson[target_idx].box.width / 2 ;
	y = pCfgs->objPerson[target_idx].box.y + pCfgs->objPerson[target_idx].box.height / 2 ;
	//得到行人方向，根据检测线端点的顺序确定正反方向
	if(pCfgs->detLineParm[region_idx].line_vertical)
	{
		if(x > pCfgs->detLineParm[region_idx].pt[0].x)
		{
			//direction = (pCfgs->detLineParm[region_idx].pt[1].y > pCfgs->detLineParm[region_idx].pt[0].y)? 0 : 1;
			direction = 0;
		}
		else
		{
			//direction = (pCfgs->detLineParm[region_idx].pt[1].y > pCfgs->detLineParm[region_idx].pt[0].y)? 1 : 0;
			direction = 1;
		}
	}
	else
	{
		if(pCfgs->detLineParm[region_idx].k * x + pCfgs->detLineParm[region_idx].b - y > 0)
		{
			//direction = (pCfgs->detLineParm[region_idx].pt[1].x > pCfgs->detLineParm[region_idx].pt[0].x)? 0 : 1;
			direction = 1;
		}
		else
		{
			//direction = (pCfgs->detLineParm[region_idx].pt[1].x > pCfgs->detLineParm[region_idx].pt[0].x)? 1 : 0;
			direction = 0;
		}
	}
	//分方向统计
	if(strcmp(pCfgs->objPerson[target_idx].names, "person") == 0)
	{
		pCfgs->uPersonDirNum[region_idx][direction]++;
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////计算两个框的交集
int overlapRatio(const CRect r1,const CRect r2)
{
	int ratio=0;
	int x1   =   max(r1.x, r2.x);
	int y1   =   max(r1.y, r2.y);
	int x2   =   min(r1.x + r1.width, r2.x + r2.width);
	int y2   =   min(r1.y + r1.height, r2.y + r2.height);

	if(x1 < x2 && y1 < y2) //intersect
	{
		int area_r1 = r1.width * r1.height;
		int area_r2 = r2.width * r2.height;
		int area_intersection = (x2 - x1) * (y2 - y1);

		ratio = area_intersection * 100 / (area_r1 + area_r2 - area_intersection);
	}
	return ratio;
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////找到最近框
CTarget* find_nearest_rect(	CRect detectBox, int class_id, CTarget* targets, int targets_size)
{

	if(targets_size < 1)
	{
		return NULL;
	}
	int i = 0;
	int overlap_ratio = 0;
	if(targets_size)
	{
		CTarget* it_max = &targets[0];
		for(i = 0; i < targets_size; i++)
		{
			//if(class_id == targets[i].class_id)
			{
				int ratio = 0;
				ratio = overlapRatio(detectBox, targets[i].box);
				if(ratio > overlap_ratio)
				{
					overlap_ratio = ratio;
					it_max = &targets[i];
				}
			}
		}
		if(overlap_ratio > 5)
			return it_max;
	}
	return NULL;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////检测框和目标进行匹配
int match_object_rect(CTarget* targets, int targets_size, CDetBox* detClasses, int class_id, int* match_object, int* match_rect, int thresh)
{

	if(targets_size < 1 || detClasses[class_id].classes_num < 1)
	{
		return -1;
	}
	int i = 0 , j = 0;
	int overlap_ratio = 0, idx_max = 0;
	memset(match_object, -1, MAX_TARGET_NUM * sizeof(int));
	memset(match_rect, -1, MAX_TARGET_NUM * sizeof(int));

	for(j = 0; j < detClasses[class_id].classes_num; j++)//匹配目标
	{
		match_object[j] = -1;
		idx_max = -1;
		overlap_ratio = 0;
		for(i = 0; i < targets_size; i++)
		{
			//if(class_id == targets[i].class_id)
			{
				int ratio = 0;
				ratio = overlapRatio(detClasses[class_id].box[j], targets[i].box);
				if(ratio > overlap_ratio)
				{
					overlap_ratio = ratio;
					idx_max = i;
				}
			}
		}
		if(overlap_ratio > thresh)
			match_object[j] = idx_max;
	}

	for(j = 0; j < targets_size; j++)//匹配框
	{
		match_rect[j] = -1;
		idx_max = -1;
		overlap_ratio = 0;
		for(i = 0; i < detClasses[class_id].classes_num; i++)
		{
			//if(class_id == targets[j].class_id)
			{
				int ratio = 0;
				ratio = overlapRatio(detClasses[class_id].box[i], targets[j].box);
				if(ratio > overlap_ratio)
				{
					overlap_ratio = ratio;
					idx_max = i;
				}
			}
		}
		if(overlap_ratio > thresh)
			match_rect[j] = idx_max;
	}

	return 1;

}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////检测框和目标进行匹配
int match_object_rect1(ALGCFGS* pCfgs, CTarget* targets, int targets_size, int match_object[][MAX_DETECTION_NUM], int match_rect[][3], int thresh)
{
	if(targets_size < 1)
	{
		return -1;
	}
	int i = 0 , j = 0, k = 0;
	int overlap_ratio = 0, idx_max = 0, idx_class = -1;
	int same_class_match = 0;
	int thresh0 = thresh;//行人和非机动车采用小的阈值
	//初始化匹配值为-1
	for( i = 0; i < pCfgs->classes; i++)
	{
		memset(match_object[i], -1, MAX_DETECTION_NUM * sizeof(int));
	}
	for( i = 0; i < MAX_TARGET_NUM; i++)
	{
		memset(match_rect[i], -1, 3 * sizeof(int));
	}
	//找到每个检测框的最近的目标框
	for(i = 0; i < pCfgs->classes; i++)
	{
		if(pCfgs->detClasses[i].classes_num <= 0)
			continue;
		thresh0 = thresh;
		//行人、摩托车、自行车进行相同类别的匹配
		if(strcmp(pCfgs->detClasses[i].names, "person") == 0)
		{
			same_class_match = 1;
			thresh0 = thresh / 2;
		}
		else if(strcmp(pCfgs->detClasses[i].names, "motorbike") == 0 || strcmp(pCfgs->detClasses[i].names, "bicycle") == 0)
		{
			same_class_match = 2;
			thresh0 = thresh / 2;
		}
		else if(strcmp(pCfgs->detClasses[i].names, "car") == 0 || strcmp(pCfgs->detClasses[i].names, "bus") == 0 || strcmp(pCfgs->detClasses[i].names, "truck") == 0 || strcmp(pCfgs->detClasses[i].names, "train") == 0)
			same_class_match = 3;
		else
			same_class_match = 4;
		for(j = 0; j < pCfgs->detClasses[i].classes_num; j++)//匹配目标
		{
			idx_max = -1;
			overlap_ratio = 0;
			for(k = 0; k < targets_size; k++)
			{
				if(i != targets[k].class_id && same_class_match == 1)//不是同一类的不进行匹配
					continue;
				if(i != targets[k].class_id && same_class_match == 2 && strcmp(targets[k].names, "motorbike") != 0 && strcmp(targets[k].names, "bicycle") != 0)//摩托车和自行车可以进行匹配
					continue;
				if(i != targets[k].class_id && same_class_match == 3 && strcmp(targets[k].names, "car") != 0 && strcmp(targets[k].names, "bus") != 0 && strcmp(targets[k].names, "truck") != 0 && strcmp(targets[k].names, "train") != 0)//car bus truck train可以进行匹配
					continue;
				//if(class_id == targets[i].class_id)
				{
					int ratio = 0;
					ratio = overlapRatio(pCfgs->detClasses[i].box[j], targets[k].box);//计算交集
					if(ratio > overlap_ratio)//得到最大交集
					{
						overlap_ratio = ratio;
						idx_max = k;
					}
				}
			}
			if(overlap_ratio > thresh0)//最大交集大于阈值
			{
				match_object[i][j] = idx_max;
			}
		}

	}
	//找到每个目标框的最近的检测框
	for(k = 0; k < targets_size; k++)//匹配框
	{
		match_rect[k][0] = -1;
		match_rect[k][1] = -1;
		idx_max = -1;
		idx_class = -1;
		overlap_ratio = 0;
		thresh0 = thresh;
		if(strcmp(targets[k].names, "person") == 0 || strcmp(targets[k].names, "motorbike") == 0 || strcmp(targets[k].names, "bicycle") == 0)
			thresh0 = thresh / 2;
		for(i = 0; i < pCfgs->classes; i++)
		{
			if(pCfgs->detClasses[i].classes_num <= 0)
				continue;
			//行人、摩托车、自行车进行相同类别的匹配
			if(strcmp(pCfgs->detClasses[i].names, "person") == 0)
				same_class_match = 1;
			else if(strcmp(pCfgs->detClasses[i].names, "motorbike") == 0 || strcmp(pCfgs->detClasses[i].names, "bicycle") == 0)
				same_class_match = 2;
			else if(strcmp(pCfgs->detClasses[i].names, "car") == 0 || strcmp(pCfgs->detClasses[i].names, "bus") == 0 || strcmp(pCfgs->detClasses[i].names, "truck") == 0 || strcmp(pCfgs->detClasses[i].names, "train") == 0)
				same_class_match = 3;
			else
				same_class_match = 4;
			if(i != targets[k].class_id && same_class_match == 1)//不是同一类的不进行匹配
				continue;
			if(i != targets[k].class_id && same_class_match == 2 && strcmp(targets[k].names, "motorbike") != 0 && strcmp(targets[k].names, "bicycle") != 0)//摩托车和自行车可以进行匹配
				continue;
			if(i != targets[k].class_id && same_class_match == 3 && strcmp(targets[k].names, "car") != 0 && strcmp(targets[k].names, "bus") != 0 && strcmp(targets[k].names, "truck") != 0 && strcmp(targets[k].names, "train") != 0)//car bus truck train可以进行匹配
				continue;
			for(j = 0; j < pCfgs->detClasses[i].classes_num; j++)//匹配目标
			{
				//if(class_id == targets[j].class_id)
				{
					int ratio = 0;
					ratio = overlapRatio(pCfgs->detClasses[i].box[j], targets[k].box);//计算交集
					if(ratio > overlap_ratio)//得到最大交集
					{
						overlap_ratio = ratio;
						idx_class = i;
						idx_max = j;
					}
				}
			}
		}
		if(overlap_ratio > thresh0)//最大交集大于阈值
		{
			match_rect[k][0] = idx_class;
			match_rect[k][1] = idx_max;
			match_rect[k][2] = overlap_ratio;
		}
	}

	return 1;
}
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////合并检测框
int merge_overlapped_box(CRect detRect, int class_id, float prob, ALGCFGS* pCfgs, Uint16 ratio_threshold)
{
	int i = 0, j = 0, k = 0;
	CRect r1;
	float prob1 = 0.0;
	for(j = 0; j < pCfgs->classes; j++)
	{
		if(pCfgs->detClasses[j].classes_num == 0)
			continue;
		if( j != class_id )
		{
			for( i = 0; i < pCfgs->detClasses[j].classes_num; i++)
			{
				r1 = pCfgs->detClasses[j].box[i];
				prob1 = pCfgs->detClasses[j].prob[i];
				if(overlapRatio(r1, detRect) > ratio_threshold)
				{
					if(prob1 > prob)
					{
						for( k = i + 1; k < pCfgs->detClasses[j].classes_num; k++)
						{
							pCfgs->detClasses[j].box[k - 1] = pCfgs->detClasses[j].box[k];
							pCfgs->detClasses[j].prob[k - 1] = pCfgs->detClasses[j].prob[k];
						}
						pCfgs->detClasses[j].classes_num = pCfgs->detClasses[j].classes_num - 1;
						i--;
						//printf("[%d,%d],[%f,%f],%d\n",pCfgs->detClasses[j].class_id,class_id,prob1,prob,overlapRatio(r1, detRect));
					}
					else
					{
						return 1;
					}
				}
			}
		}
	}
	return 0;
}
//对相同类别检测框进行处理
void post_process_box_same(ALGCFGS* pCfgs, Uint16 ratio_threshold)
{
	Uint16 i = 0, j = 0, k = 0, l = 0;
	CRect r0, r1;
	float prob0 = 0.0, prob1 = 0.0;
	for(i = 0; i < pCfgs->classes; i++)//分类别进行
	{
		if(pCfgs->detClasses[i].classes_num == 0)
			continue;
		if(strcmp(pCfgs->detClasses[i].names, "person")  == 0)//防止拥挤行人框进行合并
			continue;
		for(j = 0; j < pCfgs->detClasses[i].classes_num; j++)
		{
			r0 = pCfgs->detClasses[i].box[j];
			prob0 = pCfgs->detClasses[i].prob[j];
			for(k = j + 1; k < pCfgs->detClasses[i].classes_num; k++)
			{
				r1 = pCfgs->detClasses[i].box[k];
				prob1 = pCfgs->detClasses[i].prob[k];
				if(overlapRatio(r0, r1) > ratio_threshold)
				{
					if(prob1 <= prob0)
					{
						for( l = k + 1; l < pCfgs->detClasses[i].classes_num;l++)
						{
							pCfgs->detClasses[i].box[l - 1] = pCfgs->detClasses[i].box[l];
							pCfgs->detClasses[i].prob[l - 1] = pCfgs->detClasses[i].prob[l];
						}
						pCfgs->detClasses[i].classes_num = pCfgs->detClasses[i].classes_num - 1;
						k--;
					}
					else
					{
						for( l = j + 1; l < pCfgs->detClasses[i].classes_num; l++)
						{
							pCfgs->detClasses[i].box[l - 1] = pCfgs->detClasses[i].box[l];
							pCfgs->detClasses[i].prob[l - 1] = pCfgs->detClasses[i].prob[l];
						}
						pCfgs->detClasses[i].classes_num = pCfgs->detClasses[i].classes_num - 1;
						j--;
						continue;

					}
				}
			}

		}
	}
}
//如果行人与非机动车相交，则认为是非机动车删除行人
#ifdef DETECT_GPU
int del_non_person(CRect detRect, int class_id, ALGCFGS* pCfgs,int ratio_threshold)
{
	int i = 0, j = 0;
	CRect r1;
	for(j = 0; j < pCfgs->classes; j++)
	{
		if(strcmp(pCfgs->detClasses[j].names, "motorbike") != 0 && strcmp(pCfgs->detClasses[j].names, "bicycle") != 0)
			continue;
		if(pCfgs->detClasses[j].classes_num == 0)
			continue;
		for( i = 0; i < pCfgs->detClasses[j].classes_num; i++)
		{
			r1 = pCfgs->detClasses[j].box[i];
			if(overlapRatio(r1, detRect) > ratio_threshold)
			{
				return 1;
			}
		}

	}
	return 0;
}
#endif
/////////////////////////////////////////////////////////////////////////////////////////////////////////////////对检测框进行后处理
void post_process_box(ALGCFGS* pCfgs,  Uint16 ratio_threshold)
{
	int i = 0, j = 0, k = 0;
	CRect r;
	float prob = 0.0;
	int class_id = 0;
	int val = 0;
	for(j = 0; j < pCfgs->classes; j++)//分类别进行
	{
		if(pCfgs->detClasses[j].classes_num == 0)
			continue;
		for( i = 0; i < pCfgs->detClasses[j].classes_num; i++)
		{
			r = pCfgs->detClasses[j].box[i];
			prob = pCfgs->detClasses[j].prob[i];
			class_id = pCfgs->detClasses[j].class_id;
			val = merge_overlapped_box(r, class_id, prob, pCfgs, ratio_threshold);
			if(val == 1)
			{
				for( k = i + 1; k < pCfgs->detClasses[j].classes_num; k++)
				{
					pCfgs->detClasses[j].box[k - 1] = pCfgs->detClasses[j].box[k];
					pCfgs->detClasses[j].prob[k - 1] = pCfgs->detClasses[j].prob[k];
				}
				//printf("%d,%f\n", pCfgs->detClasses[j].class_id, prob);
				pCfgs->detClasses[j].classes_num = pCfgs->detClasses[j].classes_num - 1;
				i--;

			}
		}
	}
}
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////初始化目标
bool Initialize_target(CTarget* target)
{
	target->box.x = 0;
	target->box.y = 0;
	target->box.width = 0;
	target->box.height = 0;
	target->prob = 0;
	target->class_id = 0;
	target->detected = FALSE;
	target->target_id = 0;
	target->lost_detected = 0;
	target->trajectory_num = 0;
	target->vx = 0;
	target->vy = 0;
	target->continue_num = 0;
	target->cal_speed = FALSE;
	target->cal_flow = FALSE;
	target->start_time = 0;
	target->end_time = 0;
	target->region_idx = -1;
	target->pass_detect_line = FALSE;
	target->radar_speed = FALSE;//雷达未检测到速度
	return true;
}
void DeleteTarget(Uint16* size, int* startIdx, CTarget* target)//删除目标
{
	Uint16 sz = *size;
	Uint16 idx = *startIdx;
	Uint16 j = 0;
	for(j = idx + 1; j < sz; j++)
	{
		target[j - 1] = target[j];
	}
	*size = sz - 1;
	*startIdx = idx - 1;
}
//分析目标轨迹
int analysis_trajectory(CTarget* target)
{
	int direction = 0;
	int dir_x = 0;
	int dir_y = 0;
	if(target->trajectory_num < 2)
	{
		return -1;
	}
	//根据运动轨迹得到目标方向
	dir_x = target->trajectory[target->trajectory_num - 1].x - target->trajectory[0].x;
	dir_y = target->trajectory[target->trajectory_num - 1].y - target->trajectory[0].y;
	if(dir_y && dir_y > abs(dir_x))//向下运动
		direction = 0;
	else if(dir_y < 0 && dir_y > abs(dir_x))//向上运动
		direction = 1;
	else if(dir_x)//向右运动
		direction = 2;
	else//向左运动
		direction = 3;

	return direction;

}
//得到目标速度
void get_speed(CTarget* target)
{
	int vx = 0, vy = 0;
	int num = 0, idx1 = 0, idx2 = 0;
	num = (target->lost_detected == 0)? 1 : target->lost_detected;
	idx1  = target->trajectory_num - num - 1;
	idx1 = (idx1 < 0)? 0 : idx1;
	idx2 = target->trajectory_num - 1;
	if(target->trajectory_num > 1)
	{
		/*while(num < 3)
		{
			vx += target->trajectory[idx].x - target->trajectory[idx - 1].x;
			vy += target->trajectory[idx].y - target->trajectory[idx - 1].y;
			num++;
			idx--;
			if(idx == 0 || num >= 3)
				break;
		}
		target->vx = vx / num;
		target->vy = vy / num;*/
		vx += (target->trajectory[idx2].x  - target->trajectory[idx1].x) / num;//框中心点x方向的位移
		vy += (target->trajectory[idx2].y - target->trajectory[idx1].y) / num;//框中心点y方向的位移
		//if(vy <= 0)//针对车尾检测
		{
			target->vx = (target->vx + vx) / 2;
			target->vy = (target->vy + vy) / 2;
		}
	}

}
///////////////////////////////////////////////////////////////
//矩阵相乘1203
void matrix_mult( float *result, float *left, float *right, int m, int n, int z )
{
	int i, j, k;
	float sum;
	for( i = 0; i < m; i++ )
		for( j =0; j < z; j++ )
		{
			sum = 0.0;
			for( k = 0; k < n; k++ )
				sum = sum + ( left[ i*n+k ] * right[ k*z+j ] );
			result[i*z+j] = sum;
		}
}
//矩阵转置
void matrix_transport( float *result, float *mat, int m, int n )
{
	int i, j;
	for( i = 0; i < n; i++ )
		for( j =0; j < m; j++ )
			result[i*m+j] = mat[j*n+i];
}
//3*3矩阵求逆
void matrix_inverse(float *R, float *Ri)
{
	float den=(R[0]*R[4]*R[8] + R[1]*R[5]*R[6] + R[2]*R[3]*R[7])-
		(R[2]*R[4]*R[6] + R[1]*R[3]*R[8] + R[0]*R[5]*R[7]);
	int   i;
	if((den<-0.00000000001)||(den>0.00000000001))
	{
		Ri[0]=(R[4]*R[8]-R[7]*R[5])/den;
		Ri[1]=(R[2]*R[7]-R[1]*R[8])/den;
		Ri[2]=(R[1]*R[5]-R[2]*R[4])/den;
		Ri[3]=(R[5]*R[6]-R[3]*R[8])/den;
		Ri[4]=(R[0]*R[8]-R[2]*R[6])/den;
		Ri[5]=(R[2]*R[3]-R[0]*R[5])/den;
		Ri[6]=(R[3]*R[7]-R[4]*R[6])/den;
		Ri[7]=(R[1]*R[6]-R[0]*R[7])/den;
		Ri[8]=(R[0]*R[4]-R[1]*R[3])/den;
	}
	else 
	{
		for(i=0; i<9; i++) 
		{
			Ri[i]=0.0;

		}
	}
}

//??
bool jacobi(float *a, float *eigen_val, float *eigen_vec,int n)
{
	int k = 0,i,j;
	const float e = 0.00001;		//?
	const int l = 10000;			//
	int p, q;
	float max_value =0;
	float cos_2a, sin_2a, cos_a, sin_a;
	float t, z;
	float a_pp;
	float a_qq;
	float a_pq;
	float a_pi;
	float a_qi;
	float r_ip;
	float r_iq;
	for (i = 0; i < n; i++) 
		eigen_vec[i * n + i] = 1;
	while (1)
	{
		max_value = 0;
		for (i = 0; i < n; i++)
			for (j = i + 1; j < n; j++)
				if (fabs(a[i * n + j]) > max_value)
				{
					max_value = fabs(a[i * n + j]);
					p = i;
					q = j;
				}
				if (max_value < e || k > l) break;


				if (fabs(a[p * n + p] - a[q * n + q]) == 0)
				{
					sin_2a = 1;
					cos_2a = 0;
					cos_a = 1 / sqrt(2.0);
					sin_a = 1 / sqrt(2.0);
				}
				else 
				{	
					t = 2 * a[p * n + q] / (a[p * n + p] - a[q * n + q]);
					z = (a[p * n + p] - a[q * n + q]) / (2 * a[p * n + q]);

					if (fabs(t) < 1)
					{
						cos_2a = 1 / sqrt(1 + t * t);
						sin_2a = t / sqrt(1 + t * t);
					}
					else 
					{
						cos_2a = fabs(z) / sqrt(1 + z * z);
						sin_2a = (z > 0 ? 1 : (z < 0 ? -1 :0)) / sqrt(1 + z * z);
					}
					cos_a = sqrt((1 + cos_2a) / 2);
					sin_a = sin_2a / (2 * cos_a);
				}
				a_pp =a[p * n + p];
				a_qq = a[q * n + q];
				a_pq = a[p * n + q];
				a[p * n + p] = a_pp * cos_a * cos_a + a_qq * sin_a * sin_a + 2 * a_pq * cos_a * sin_a;
				a[q * n + q] = a_pp * sin_a * sin_a + a_qq * cos_a * cos_a - 2 * a_pq * cos_a * sin_a;

				for (i = 0; i < n; i++)
					if (i != p && i != q)
					{
						a_pi = a[p * n + i];
						a_qi = a[q * n + i];

						a[p * n + i] = a[i * n + p] = a_pi * cos_a + a_qi * sin_a;
						a[q * n + i] = a[i * n + q] = - a_pi * sin_a + a_qi * cos_a;
					}

					a[p * n + q] = a[q * n + p] = (a_qq - a_pp) * sin_2a / 2 + a_pq * cos_2a;
					//?
					for (i = 0; i < n; i++)
					{
						r_ip = eigen_vec[i * n + p];
						r_iq = eigen_vec[i * n + q];
						eigen_vec[i * n + p] = r_ip * cos_a + r_iq * sin_a;
						eigen_vec[i * n + q] = - r_ip * sin_a + r_iq * cos_a;
					}
					k++;
	}

	for (i = 0; i < n; i++) 
		eigen_val[i] = a[i * n + i];
	return TRUE;
}
//svd?
void svd( float *a, int m, int n,  float *d, float v[] )
{
	int i, j, k;
	float aT[2*CALIBRATION_POINT_NUM*9]={0};
	float aT_a[9*9] ={0};
	float tmp;
	float t[9] = {0};
	matrix_transport( aT, a, m, n );
	matrix_mult( aT_a, aT, a, n, m, n );
	jacobi(aT_a, d, v,n);
	//???

	for( i = 0; i < n-1; i++ )
	{
		tmp = d[ i ];
		for( k = 0; k < n; k++ )
			t[ k ] = v[ k * n + i ];
		for( j = i+1; j < n ; j++ )
		{
			if( d[ j ] > tmp )
			{
				d[ i ] = d[ j ];
				d[ j ] = tmp;
				tmp = d[ i ];
				for( k = 0; k < n; k++ )
				{
					v[ k * n + i ] = v[ k * n + j ];
					v[ k * n + j ] = t[ k ];
				}				
			}
		}
	}

}
//从图像坐标向实际坐标进行映射lhx,20150608
static void img_to_actual(float mapping_matrix[],int start_row,int end_row,int overlap_row1,int overlap_row2,int flag,ALGCFGS *pCfgs)
{
	int row,col;
	float a1=0,a2=0;
	float b1=0,b2=0;
	float c1=0,c2=0;
	int temp;
	float temp1=0;
	float temp2=0;
	if(flag==1)
	{
		if(start_row>=end_row)
			start_row=FULL_ROWS-1;
		else
			start_row=0;
	}
	if(flag==2)
	{
		if(start_row>=end_row)
			end_row=0;
		else	
			end_row=FULL_ROWS-1;
	}
	if(start_row>end_row)
	{
		temp=start_row;
		start_row=end_row;
		end_row=temp;
	}
	if(overlap_row1>overlap_row2)
	{
		temp=overlap_row1;
		overlap_row1=overlap_row2;
		overlap_row2=temp;
	}
	for(row=start_row;row<=end_row;row++)
	{
		for(col=0;col<FULL_COLS;col++)
		{
			a1=mapping_matrix[0]-mapping_matrix[6]*col;
			a2=mapping_matrix[3]-mapping_matrix[6]*row;
			b1=mapping_matrix[1]-mapping_matrix[7]*col;
			b2=mapping_matrix[4]-mapping_matrix[7]*row;
			c1=mapping_matrix[2]-mapping_matrix[8]*col;
			c2=mapping_matrix[5]-mapping_matrix[8]*row;
			temp1=0;
			temp2=0;
			if(abs(a1*b2-b1*a2)>0.00000000001)
			{
				temp1=(a2*c1-a1*c2)/(a1*b2-b1*a2);
				temp2=(b1*c2-b2*c1)/(a1*b2-b1*a2);
				if(row>=overlap_row1&&row<=overlap_row2&&flag!=1)
				{
					pCfgs->image_actual[row][col][0]+=temp1;
					pCfgs->image_actual[row][col][1]+=temp2;
					pCfgs->image_actual[row][col][0]=pCfgs->image_actual[row][col][0]/2;
					pCfgs->image_actual[row][col][1]=pCfgs->image_actual[row][col][1]/2;
				}
				else
				{
					pCfgs->image_actual[row][col][0]=temp1;
					pCfgs->image_actual[row][col][1]=temp2;
				}
			}
			else
			{
				pCfgs->image_actual[row][col][0]=0;
				pCfgs->image_actual[row][col][1]=0;
			}

		}
	}
}
//得到实际点的距离值
static void get_actual_point(float actual_point[2],int row,int col,int limit_line,ALGCFGS *pCfgs)
{
	row=(row<limit_line)? limit_line:row;
	actual_point[0]=pCfgs->image_actual[row][col][0];
	actual_point[1]=pCfgs->image_actual[row][col][1];

}
//计算两点的实际距离
float distance_two(float actual_point1[2],float actual_point2[2])
{
	float dis=0;
	dis=(actual_point1[0]-actual_point2[0])*(actual_point1[0]-actual_point2[0])+(actual_point1[1]-actual_point2[1])*(actual_point1[1]-actual_point2[1]);
	dis=sqrt(dis);
	return dis;

}
void SobelCalculate(unsigned char *puPointNewImage, unsigned char *puPointSobelImage, int threshold, int width, int height)
{
	unsigned char *in;
	unsigned char *out;

	int H, O, V, i, j;
	int i00, i01, i02;
	int i10,      i12;
	int i20, i21, i22;

	in = puPointNewImage;
	out = puPointSobelImage;
	memset(puPointSobelImage, 0, sizeof(unsigned char) * width * height);
	for(i = 1; i < height - 1; i++)
	{
		in = puPointNewImage + i * width;
		out = puPointSobelImage + i * width;
		for(j = 1; j < width - 1; j++)
		{
			i00 = *(in + j - width - 1);
			i01 = *(in + j - width);
			i02 = *(in + j - width + 1);
			i10 = *(in + j - 1);
			i12 = *(in + j + 1);
			i20 = *(in + j + width - 1);
			i21 = *(in + j + width);
			i22 = *(in + j + width + 1);

			H = -   i00 - 2 * i01 -   i02 +
				+   i20 + 2 * i21 +   i22;

			V = -     i00     +     i02
				- 2 * i10     + 2 * i12
				-     i20     +     i22;

			O = abs(H) + abs(V);
			O = (O < threshold) ? 0 : 255;
			*(out + j) = O;

		}

	}
}
void AvgFilter(unsigned char *img, unsigned char *dst_img, int width, int height)
{
	int i = 0, j = 0;
	int val = 0;
	unsigned char* p0;
	unsigned char* p1;
	unsigned char* p2;
	memcpy(dst_img, img, width * height * sizeof(unsigned char));
	for(i = 1; i < height - 1; i++)
	{
		p0 = img + i * width;
		p2 = dst_img + i * width;
		for(j = 1; j < width - 1; j++)
		{
			p1 = p0 + j;
			val = *p1 + *(p1 - 1) + *(p1 + 1);
			val += *(p1 - width) + *(p1 - width - 1) + *(p1 - width + 1);
			val += *(p1 + width) + *(p1 + width - 1) + *(p1 + width + 1);
			*(p2 + j) = val / 9;
		}
	}
}
//计算暗通道图像
void DarkChannel(unsigned char *img, unsigned char *dark_img, int width, int height, int r)
{
	int min_val = 0;
	unsigned char* p;
	int st_row, ed_row;
	int st_col, ed_col;
	for (int i = 0; i < height; i++)
	{
		for (int j = 0; j < width; j++)
		{
			st_row = i - r, ed_row = i + r;
			st_col = j - r, ed_col = j + r;
			st_row = st_row < 0 ? 0 : st_row;
			ed_row = ed_row >= height ? (height - 1) : ed_row;
			st_col = st_col < 0 ? 0 : st_col;
			ed_col = ed_col >= width ? (width - 1) : ed_col;
			min_val = 300;
			//区域内最小值
			for (int m = st_row; m <= ed_row; m++)
			{
				for (int n = st_col; n <= ed_col; n++)
				{
					p = img + m * width * 3 + n * 3;
					//通道内最小值
					min_val = min(min_val, *p);
					min_val = min(min_val, *(p + 1));
					min_val = min(min_val, *(p + 2));
				}
			}
			*(dark_img + i * width + j) = min_val;
		}
	}
}
//二值化,并判断两个位置处是否有车灯
int Threshold(unsigned char* img, unsigned char* dst_img, int width, int height, int thr, int l1, int l2)
{
	unsigned char* p;
	int sum1 = 0, sum2 = 0;
	for(int i = 0; i < height; i++)
	{
		for(int j = 0; j < width; j++)
		{
			if(*(img + i * width + j) >= thr)
			{
				*(dst_img + i * width + j) = *(img + i * width + j);
				if(i >= (l1 - 50) && i <= (l1 + 50))
				{
					sum1++;
				}
				if(i >= (l2 - 50) && i <= (l2 + 50))
				{
					sum2++;
				}
			}
			else
			{
				*(dst_img + i * width + j) = 0;
			}
		}
	}
	if(sum1 > 50 && sum2 > 50)
		return 3;
	else if(sum1 > 50)
		return 1;
	else if(sum2 > 50)
		return 2;
	else 
		return 0;

}