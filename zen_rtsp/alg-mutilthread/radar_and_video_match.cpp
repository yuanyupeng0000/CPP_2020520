#include "m_arith.h"
//雷达检测和视频检测数据融合
//objRadar:雷达检测目标 
//objRadarNum 雷达检测目标数
//objInfo 视频和雷达融合后的目标
//objNum 视频和雷达融合后的目标数
void associate_radar_and_video(ALGCFGS *pCfgs, mRadarRTObj* objRadar, int objRadarNum, OBJECTINFO* objVehicleInfo, Uint16* objVehicleNum, OBJECTINFO* objPersonInfo, Uint16* objPersonNum)
{
	if(objRadarNum <= 0)//没有雷达检测目标,则返回
		return;
	int i = 0, j = 0, k = 0;
	int actual_x = 0, actual_y = 0;//雷达和视频数据正好相反
	int offset_x = 0, offset_y = 0; //x和y方向上雷达和相机坐标系的偏移
	//雷达和视频能匹配上，利用雷达的检测信息
	//视频检测出的目标，雷达没有，利用视频进行计算目标的速度、位置、目标长度宽度
	//雷达检测出的目标，视频没有，需要计算图像位置
	OBJECTINFO* objVehicleVideo = pCfgs->ResultMsg.uResultInfo.udetBox;
	int objVehicleVideoNum = pCfgs->ResultMsg.uResultInfo.udetNum;
	OBJECTINFO* objPersonVideo = pCfgs->ResultMsg.uResultInfo.udetPersonBox;
	int objPersonVideoNum = pCfgs->ResultMsg.uResultInfo.udetPersonNum;
	int objVideoNum = objVehicleVideoNum + objPersonVideoNum;
	Uint16 vehicleNum = 0, personNum = 0;
	int match_video[200] = { -1 };
	int match_radar[200] = { -1 };
	float dis_obj[200][200] = { 1000000 };//用于计算视频和雷达目标之间的距离
	for(i = 0; i < 200; i++)
	{
		for(j = 0; j < 200; j++)
			dis_obj[i][j] = 1000000;
	}
	memset(match_video, -1, 200 * sizeof(int));
	memset(match_radar, -1, 200 * sizeof(int));
	printf("detect vehicle obj = %d, person obj =%d\n", objVehicleVideoNum, objPersonVideoNum);
	for(i = 0; i < objRadarNum; i++)
	{
		//将雷达目标映射到图像
		mRadarRTObj objR = objRadar[i];
		actual_x = int((objRadar[i].y_Point + offset_x - pCfgs->actual_origin[0]) * pCfgs->mapping_ratio[0]);
		actual_y = int((objRadar[i].x_Point + offset_y - pCfgs->actual_origin[1]) * pCfgs->mapping_ratio[1]);
		printf("i = %d, radar = [ %f, %f, %f, %f, %f], [%d,%d]\n", i, objRadar[i].x_Point, objRadar[i].y_Point, objRadar[i].Speed_x, objRadar[i].Speed_y, objRadar[i].Obj_Len, actual_x,actual_y);
		if(actual_x >= 2000 || actual_y >= 2000 || actual_x < 0 || actual_y < 0)//超出实际坐标到图像的映射范围
			continue;
		int img_x = pCfgs->actual_image[actual_y][actual_x][0];
		int img_y = pCfgs->actual_image[actual_y][actual_x][1];
		printf("i = %d img point = [%d,%d]\n", i, img_x, img_y);
		if(img_x < 0 || img_y < 0)//超出视频图像范围
			continue;
		//与视频目标求距离
		for(j = 0; j < objVehicleVideoNum; j++)
		{
			dis_obj[i][j] = abs(objVehicleVideo[j].x + objVehicleVideo[j].w / 2 - img_x) + abs(objVehicleVideo[j].y + objVehicleVideo[j].h / 2 - img_y);
		}
		for(j = 0; j < objPersonVideoNum; j++)
		{
			dis_obj[i][j + objVehicleVideoNum] = abs(objPersonVideo[j].x + objPersonVideo[j].w / 2 - img_x) + abs(objPersonVideo[j].y + objPersonVideo[j].h / 2 - img_y);
		}
	}
	//判断视频目标和雷达目标是否匹配
	for(j = 0; j < objVideoNum; j++)
	{
		float min_dis = 1000000;
		int idx = -1;
		for(i = 0; i < objRadarNum; i++)
		{
			//找到没有匹配的最近的视频目标
			if(dis_obj[i][j] < min_dis && match_radar[i] < 0)
			{
				min_dis = dis_obj[i][j];
				idx = i;

			}
		}
		if(idx >= 0 && min_dis < 100)//100为阈值，匹配成功
		{
			match_radar[idx] = j;
			match_video[j] = idx;
			if(j < objVehicleVideoNum)
			{
				objVehicleInfo[vehicleNum] = objVehicleVideo[j];
				objVehicleInfo[vehicleNum].speed_Vx = (int)(objRadar[idx].Speed_y); 
				objVehicleInfo[vehicleNum].speed = (int)(objRadar[idx].Speed_x);
				objVehicleInfo[vehicleNum].length = (int)(objRadar[idx].Obj_Len + 0.5);
				for(i = 0; i < pCfgs->detTargets_size; i++)
				{
					if(pCfgs->detTargets[i].target_id == objVehicleInfo[vehicleNum].id)
					{
						pCfgs->detTargets[i].vx = objVehicleInfo[vehicleNum].speed_Vx;
						pCfgs->detTargets[i].vy = objVehicleInfo[vehicleNum].speed;
						pCfgs->detTargets[i].radar_speed = TRUE;
						break;
					}
				}
				printf("join vehicle dis = %f,%f, %d, speed = [%f,%d],length = [%f, %d]\n", dis_obj[idx][j], min_dis, idx, objRadar[idx].Speed_x, objVehicleInfo[vehicleNum].speed, objRadar[idx].Obj_Len,objVehicleInfo[vehicleNum].length);
				vehicleNum++;

			}
			else
			{
				objPersonInfo[personNum] = objPersonVideo[j - objVehicleVideoNum];
				objPersonInfo[personNum].speed_Vx = (int)(objRadar[idx].Speed_y);
				objPersonInfo[personNum].speed = (int)(objRadar[idx].Speed_x);  
				objPersonInfo[personNum].length = (int)(objRadar[idx].Obj_Len + 0.5);
				for(i = 0; i < pCfgs->objPerson_size; i++)
				{
					if(pCfgs->objPerson[i].target_id == objPersonInfo[personNum].id)
					{
						pCfgs->objPerson[i].vx = objPersonInfo[personNum].speed_Vx;
						pCfgs->objPerson[i].vy = objPersonInfo[personNum].speed;
						pCfgs->objPerson[i].radar_speed = TRUE;
						break;
					}
				}
				printf("join person dis = %f,%f, %d, speed = [%f,%d],length = [%f, %d]\n", dis_obj[idx][j], min_dis, idx, objRadar[idx].Speed_x, objPersonInfo[personNum].speed, objRadar[idx].Obj_Len,objPersonInfo[personNum].length);
				personNum++;

			}
		}
		if(match_video[j] < 0)//视频目标没有匹配上
		{
			if(j < objVehicleVideoNum)
			{
				objVehicleInfo[vehicleNum++] = objVehicleVideo[j];
			}
			else
			{
				objPersonInfo[personNum++] = objPersonVideo[j - objVehicleVideoNum];
			}
		}
	}

	for(i = 0; i < objRadarNum; i++)//雷达目标没有匹配上
	{
		if(match_radar[i] >= 0)
			continue;
		//将雷达目标映射到图像
		mRadarRTObj objR = objRadar[i];
		if(abs(objRadar[i].Speed_x) < 10)//雷达目标速度太小，认为是误检
			continue;
		int top = 0, bottom = 0;
		actual_x = int((objRadar[i].y_Point + offset_x - pCfgs->actual_origin[0]) * pCfgs->mapping_ratio[0]);
		actual_y = int((objRadar[i].x_Point + offset_y - pCfgs->actual_origin[1]) * pCfgs->mapping_ratio[1]);
		top = int((objRadar[i].x_Point - objRadar[i].Obj_Len / 2 + offset_y - pCfgs->actual_origin[1]) * pCfgs->mapping_ratio[1]);
		bottom = int((objRadar[i].x_Point + objRadar[i].Obj_Len / 2 + offset_y - pCfgs->actual_origin[1]) * pCfgs->mapping_ratio[1]);
		if(actual_x >= 2000 || actual_y >= 2000 || actual_x < 0 || actual_y < 0)//超出实际坐标到图像的映射范围
			continue;
		int x = pCfgs->actual_image[actual_y][actual_x][0];
		int y = pCfgs->actual_image[actual_y][actual_x][1];
		int h = abs(pCfgs->actual_image[top][actual_x][1] - pCfgs->actual_image[bottom][actual_x][1]);
		if(x < 0 || y < 0)//超出视频图像范围
			continue;
		objVehicleInfo[vehicleNum].x = x;
		objVehicleInfo[vehicleNum].y = y;
		objVehicleInfo[vehicleNum].w = h * 2 / 3;
		objVehicleInfo[vehicleNum].h = h;
		objVehicleInfo[vehicleNum].distance[0] = objRadar[i].y_Point + offset_x;
		objVehicleInfo[vehicleNum].distance[1] = objRadar[i].x_Point + offset_y;
		if(objRadar[i].Obj_Len < 5)//car
		{
			objVehicleInfo[vehicleNum].label = 2;
		}
		else if(objRadar[i].Obj_Len < 12)//truck
		{
			objVehicleInfo[vehicleNum].label  = 3;
		}
		else//bus
		{
			objVehicleInfo[vehicleNum].label = 1;

		}
		objVehicleInfo[vehicleNum].speed_Vx = (int)(objRadar[i].Speed_y);
		objVehicleInfo[vehicleNum].speed = (int)(objRadar[i].Speed_x); 
		objVehicleInfo[vehicleNum].length = (int)(objRadar[i].Obj_Len + 0.5);
		vehicleNum++;

	}
	*objVehicleNum = vehicleNum;
	*objPersonNum = personNum;

}
//雷达检测和视频检测数据融合
//objRadar:雷达检测目标 
//objRadarNum 雷达检测目标数
//objInfo 视频和雷达融合后的目标
//objNum 视频和雷达融合后的目标数
void associate_radar_and_video(ALGCFGS *pCfgs, ALGPARAMS *pParams, mRadarRTObj* objRadar, int objRadarNum, OBJECTINFO* objVehicleInfo, Uint16* objVehicleNum)//车辆检测框和雷达融合
{
	if(objRadarNum <= 0)//没有雷达检测目标,则返回
		return;
	int i = 0, j = 0, k = 0;
	int actual_x = 0, actual_y = 0;//雷达和视频数据正好相反
	int offset_x = 0, offset_y = 0; //x和y方向上雷达和相机坐标系的偏移
	//雷达和视频能匹配上，利用雷达的检测信息
	//视频检测出的目标，雷达没有，利用视频进行计算目标的速度、位置、目标长度宽度
	//雷达检测出的目标，视频没有，需要计算图像位置
	OBJECTINFO* objVehicleVideo = pCfgs->ResultMsg.uResultInfo.udetBox;
	int objVehicleVideoNum = pCfgs->ResultMsg.uResultInfo.udetNum;
	Uint16 vehicleNum = 0;
	int match_video[200] = { -1 };
	int match_radar[200] = { -1 };
	float dis_obj[200][200] = { 1000000 };//用于计算视频和雷达目标之间的距离
	for(i = 0; i < 200; i++)
	{
		for(j = 0; j < 200; j++)
			dis_obj[i][j] = 1000000;
	}
	memset(match_video, -1, 200 * sizeof(int));
	memset(match_radar, -1, 200 * sizeof(int));
	printf("detect vehicle obj = %d\n", objVehicleVideoNum);
	for(i = 0; i < objRadarNum; i++)
	{
		//将雷达目标映射到图像
		mRadarRTObj objR = objRadar[i];
		actual_x = int((objRadar[i].y_Point + offset_x - pCfgs->actual_origin[0]) * pCfgs->mapping_ratio[0]);
		actual_y = int((objRadar[i].x_Point + offset_y - pCfgs->actual_origin[1]) * pCfgs->mapping_ratio[1]);
		printf("i = %d, radar = [ %f, %f, %f, %f, %f], [%d,%d]\n", i, objRadar[i].x_Point, objRadar[i].y_Point, objRadar[i].Speed_x, objRadar[i].Speed_y, objRadar[i].Obj_Len, actual_x,actual_y);
		if(actual_x >= 2000 || actual_y >= 2000 || actual_x < 0 || actual_y < 0)//超出实际坐标到图像的映射范围
			continue;
		int img_x = pCfgs->actual_image[actual_y][actual_x][0];
		int img_y = pCfgs->actual_image[actual_y][actual_x][1];
		printf("i = %d img point = [%d,%d]\n", i, img_x, img_y);
		if(img_x < 0 || img_y < 0)//超出视频图像范围
			continue;
		//与视频目标求距离
		for(j = 0; j < objVehicleVideoNum; j++)
		{
			dis_obj[i][j] = abs(objVehicleVideo[j].x + objVehicleVideo[j].w / 2 - img_x) + abs(objVehicleVideo[j].y + objVehicleVideo[j].h / 2 - img_y);
		}
	}
	//判断视频目标和雷达目标是否匹配
	/*for(i = 0; i < objRadarNum; i++)
	{
		float min_dis = 1000000;
		int idx = -1;
		for(j = 0; j < objVehicleVideoNum; j++)
		{
			//找到没有匹配的最近的视频目标
			if(dis_obj[i][j] < min_dis && match_video[j] < 0)
			{
				min_dis = dis_obj[i][j];
				idx = j;

			}
		}
		if(idx >= 0 && min_dis < 100)//100为阈值，匹配成功
		{
			match_radar[i] = idx;
			match_video[idx] = i;
			if(idx < objVehicleVideoNum)
			{
				objVehicleInfo[vehicleNum] = objVehicleVideo[idx];
				objVehicleInfo[vehicleNum].speed_Vx =(int)(objRadar[i].Speed_y);
				objVehicleInfo[vehicleNum].speed = (int)(objRadar[i].Speed_x); 
				objVehicleInfo[vehicleNum].length = (int)(objRadar[i].Obj_Len + 0.5);
				for(j = 0; j < pCfgs->detTargets_size; j++)
				{
					if(pCfgs->detTargets[j].target_id == objVehicleInfo[vehicleNum].id)
					{
						pCfgs->detTargets[j].vx = objVehicleInfo[vehicleNum].speed_Vx;
						pCfgs->detTargets[j].vy = objVehicleInfo[vehicleNum].speed;
						pCfgs->detTargets[i].radar_speed = TRUE;
						break;
					}
				}
				printf("join vehicle dis = %f,%f, %d, speed = [%f,%d],length = [%f, %d]\n", dis_obj[i][idx], min_dis, idx, objRadar[i].Speed_x, objVehicleInfo[vehicleNum].speed, objRadar[i].Obj_Len,objVehicleInfo[vehicleNum].length);
				vehicleNum++;

			}
		}
	}*/

	//判断视频目标和雷达目标是否匹配
	for(j = 0; j < objVehicleVideoNum; j++)
	{
		float min_dis = 1000000;
		int idx = -1;
		for(i = 0; i < objRadarNum; i++)
		{
			//找到没有匹配的最近的视频目标
			if(dis_obj[i][j] < min_dis && match_radar[i] < 0)
			{
				min_dis = dis_obj[i][j];
				idx = i;

			}
		}
		if(idx >= 0 && min_dis < 100)//100为阈值，匹配成功
		{
			match_radar[idx] = j;
			match_video[j] = idx;
			if(idx < objRadarNum)
			{
				objVehicleInfo[vehicleNum] = objVehicleVideo[j];
				objVehicleInfo[vehicleNum].speed_Vx = (int)(objRadar[idx].Speed_y);
				objVehicleInfo[vehicleNum].speed = (int)(objRadar[idx].Speed_x); 
				objVehicleInfo[vehicleNum].length = (int)(objRadar[idx].Obj_Len + 0.5);
				objVehicleInfo[vehicleNum].laneid = 2;//雷达检测结果
				printf("join vehicle dis = %f,%f, %d, speed = [%f,%d],length = [%f, %d]\n", dis_obj[idx][j], min_dis, idx, objRadar[idx].Speed_x, objVehicleInfo[vehicleNum].speed, objRadar[idx].Obj_Len,objVehicleInfo[vehicleNum].length);

			}
		}
		if(match_video[j] < 0)//视频目标没有匹配上
		{
			objVehicleVideo[j].laneid = 1;//视频检测结果
			objVehicleInfo[vehicleNum] = objVehicleVideo[j];
		}
		for(i = 0; i < pCfgs->detTargets_size; i++)
		{
			if(pCfgs->detTargets[i].target_id == objVehicleInfo[vehicleNum].id)
			{
				//对比上一帧和当前帧的速度差别
				if(((pCfgs->detTargets[i].vy - objVehicleInfo[vehicleNum].speed) > 10) && abs(objVehicleInfo[vehicleNum].speed < 5) && (objVehicleInfo[vehicleNum].laneid == 2))//消除雷达检测的速度为0
				{
					objVehicleInfo[vehicleNum].speed_Vx = pCfgs->detTargets[i].vx;
					objVehicleInfo[vehicleNum].speed = pCfgs->detTargets[i].vy;//采用上一帧的速度
				}
			    pCfgs->detTargets[i].vx = objVehicleInfo[vehicleNum].speed_Vx;
				pCfgs->detTargets[i].vy = objVehicleInfo[vehicleNum].speed;//保存当前帧的速度
				//pCfgs->detTargets[i].radar_speed = TRUE;
				break;
			}
		}
		vehicleNum++;
	}

	/*for(i = 0; i < objRadarNum; i++)//雷达目标没有匹配上
	{
		if(match_radar[i] >= 0)
			continue;
		//将雷达目标映射到图像
		mRadarRTObj objR = objRadar[i];
		if(abs(objRadar[i].Speed_x) < 10 || abs(objRadar[i].Speed_y) < 10)//雷达目标速度太小，认为是误检
			continue;
		int top = 0, bottom = 0;
		actual_x = int((objRadar[i].y_Point + offset_x - pCfgs->actual_origin[0]) * pCfgs->mapping_ratio[0]);
		actual_y = int((objRadar[i].x_Point + offset_y - pCfgs->actual_origin[1]) * pCfgs->mapping_ratio[1]);
		top = int((objRadar[i].x_Point - objRadar[i].Obj_Len / 2 + offset_y - pCfgs->actual_origin[1]) * pCfgs->mapping_ratio[1]);
		bottom = int((objRadar[i].x_Point + objRadar[i].Obj_Len / 2 + offset_y - pCfgs->actual_origin[1]) * pCfgs->mapping_ratio[1]);
		if(actual_x >= 2000 || actual_y >= 2000 || actual_x < 0 || actual_y < 0)//超出实际坐标到图像的映射范围
			continue;
		int x = pCfgs->actual_image[actual_y][actual_x][0];
		int y = pCfgs->actual_image[actual_y][actual_x][1];
		int h = abs(pCfgs->actual_image[top][actual_x][1] - pCfgs->actual_image[bottom][actual_x][1]);
		if(x < 0 || y < 0)//超出视频图像范围
			continue;
		//判断是否属于检测车道内
		int overlapNum[MAX_LANE] = {-1};
		int max_value = -1, idx = -1;
		CRect rct;
		rct.x = x;
		rct.y = y;
		rct.width = h * 2 / 3;
		rct.height = h;
		for( j = 0; j < pCfgs->LaneAmount; j++)//计算与车道相交值
		{
			overlapNum[j] = RectInRegion(pParams->MaskLaneImage, pCfgs, pCfgs->img_width, pCfgs->img_height, rct, j);
			if(overlapNum[j] > max_value)
			{
				max_value = overlapNum[j];
				idx = j;
			}
		}
		if(max_value < 10)
			continue;
		objVehicleInfo[vehicleNum].x = x;
		objVehicleInfo[vehicleNum].y = y;
		objVehicleInfo[vehicleNum].w = h * 2 / 3;
		objVehicleInfo[vehicleNum].h = h;
		objVehicleInfo[vehicleNum].distance[0] = objRadar[i].y_Point + offset_x;
		objVehicleInfo[vehicleNum].distance[1] = objRadar[i].x_Point + offset_y;
		if(objRadar[i].Obj_Len < 5)//car
		{
			objVehicleInfo[vehicleNum].label = 2;
		}
		else if(objRadar[i].Obj_Len < 12)//truck
		{
			objVehicleInfo[vehicleNum].label  = 3;
		}
		else//bus
		{
			objVehicleInfo[vehicleNum].label = 1;

		}
		//objVehicleInfo[vehicleNum].laneid = idx;
		objVehicleInfo[vehicleNum].laneid = 2;//雷达检测结果
		objVehicleInfo[vehicleNum].speed_Vx = (int)(objRadar[i].Speed_y);
		objVehicleInfo[vehicleNum].speed = (int)(objRadar[i].Speed_x); 
		objVehicleInfo[vehicleNum].length = (int)(objRadar[i].Obj_Len + 0.5);
		vehicleNum++;

	}*/
	*objVehicleNum = vehicleNum;

}