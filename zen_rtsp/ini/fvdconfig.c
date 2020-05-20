#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <stdlib.h>
#include "fvdconfig.h"
#include "fvdconfig.h"
#include "inifile.h"

//
#define CAMERADIR         "./config/%d_cammer"
//
#define BASEFILE          "./config/base.ini"
#define SYSFILE           "./config/system.ini"
#define NETWORKFILE       "./config/network.ini"
#define CHANGETIMEFILE    "./config/changetime.ini"
#define STATISFILE        "./config/statis.ini"
#define SERIALFILE        "./config/serial.ini"
#define PROTOCOLLFILE     "./config/protocol.ini"
#define NTPFILE           "./config/ntp.ini"
#define CAMERASTATUSFILE  "./config/camera_status.ini"

//区域8个顶点文件
#define AREAFILE           "./config/%d_cammer/area.ini"
//车道IDS
#define LANEIDFILE          "./config/%d_cammer/laneid.ini"
//系统公用的配置数据 包括检测参数,相机属性,算法参数
#define COMMONFILE          "./config/%d_cammer/config.ini"
//占位线圈文件
#define FRONTCOILFILE      "./config/%d_cammer/frontcoil.ini"
//占位线圈文件
#define MIDDLECOILFILE      "./config/%d_cammer/middlecoil.ini"
//后置线圈
#define REARCOILFILE       "./config/%d_cammer/rearcoil.ini"
//虚拟车道线文件
#define VIRLANELINEFILE    "./config/%d_cammer/virlaneline.ini"
//4个标定值坐标点文件
#define STANDARDPOINTFILE  "./config/%d_cammer/standardpoint.ini"
//行人
#define PERSONFILE         "./config/%d_cammer/person.ini"
//other
#define OTHERFILE         "./config/%d_cammer/other.ini"
//#define ALGCAMERAFILE      "./config/%d_cammer/algcame.ini"
#define DETECTPARAMFILE   "./config/%d_cammer/detectparam.ini"
//
#define PERSON_AREAS        "./config/%d_cammer/person_areas.ini"
#define PERSON_AREAS_TIMES  "./config/%d_cammer/person_areas_times.ini"
//
#define EVENTS_FILE "./config/%d_cammer/events.ini"
//
#define POSITION_FILE "./config/%d_cammer/position.ini"



//读基本信息
int ReadBaseParam(IVDDevSets *devsets)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "BASEINFO");
	char file[100] = {0};
	sprintf(file, BASEFILE);

	sprintf(key, "checkaddr");
	read_profile_string(section, key, (char*)devsets->checkaddr, CHECKADDR, "test", file);
	sprintf(key, "devUserNo");
	read_profile_string(section, key, (char*)devsets->devUserNo, DEVUSERNO, "1", file);
	sprintf(key, "overWrite");
	devsets->overWrite = read_profile_int(section, key, 1, file);
	sprintf(key, "loglevel");
	devsets->loglevel = read_profile_int(section, key, 3, file);
	sprintf(key, "autoreset");
	devsets->autoreset = read_profile_int(section, key, 8, file);
	sprintf(key, "timeset");
	devsets->timeset = read_profile_int(section, key, 0, file);
	sprintf(key, "protype");
	devsets->pro_type = read_profile_int(section, key, 0, file);

	return 1;
}


//写基本信息
int WriteBaseParam(IVDDevSets *devsets)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "BASEINFO");
	char file[100] = {0};
	sprintf(file, BASEFILE);


	sprintf(key, "checkaddr");
	write_profile_string(section, key, (char*)devsets->checkaddr, file);

	sprintf(key, "devUserNo");
	write_profile_string(section, key, (char*)devsets->devUserNo, file);

	sprintf(key, "overWrite");
	sprintf(value, "%d", devsets->overWrite);
	write_profile_string(section, key, value, file);

	sprintf(key, "loglevel");
	sprintf(value, "%d", devsets->loglevel);
	write_profile_string(section, key, value, file);

	sprintf(key, "autoreset");
	sprintf(value, "%d", devsets->autoreset);
	write_profile_string(section, key, value, file);

	sprintf(key, "timeset");
	sprintf(value, "%d", devsets->timeset);
	write_profile_string(section, key, value, file);

	sprintf(key, "protype");
	sprintf(value, "%d", devsets->pro_type);
	write_profile_string(section, key, value, file);

	return 1;
}

//读取系统信息
int ReadSysParam(IVDDevInfo *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "SYSTEMINFO");
	char file[100] = {0};
	sprintf(file, SYSFILE);

	sprintf(key, "devicetype");
	read_profile_string(section, key, (char*)info->devicetype, SYS_SIZE, "1.0", file);
	sprintf(key, "firmwareV");
	read_profile_string(section, key, (char*)info->firmwareV, SYS_SIZE, "1.0", file);
	sprintf(key, "webV");
	read_profile_string(section, key, (char*)info->webV, SYS_SIZE, "1.0", file);
	sprintf(key, "libV");
	read_profile_string(section, key, (char*)info->libV, SYS_SIZE, "1.0", file);
	sprintf(key, "hardwareV");
	read_profile_string(section, key, (char*)info->hardwareV, SYS_SIZE, "1.0", file);


	return 1;
}

//读网络信息
int ReadNetWorkParam(IVDNetInfo *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};


	sprintf(section, "NETWORK");
	char file[100] = {0};
	sprintf(file, NETWORKFILE);

	sprintf(key, "strIpaddr");
	read_profile_string(section, key, (char*)info->strIpaddr, NET_SIZE, "", file);
	sprintf(key, "strIpaddr1");
	read_profile_string(section, key, (char*)info->strIpaddr1, NET_SIZE, "1", file);
	sprintf(key, "strIpaddr2");
	read_profile_string(section, key, (char*)info->strIpaddr2, NET_SIZE, "1", file);
	sprintf(key, "strIpaddrIO");
	read_profile_string(section, key, (char*)info->strIpaddrIO, NET_SIZE, "1", file);
	sprintf(key, "strPort");
	info->strPort = read_profile_int(section, key, 10000, file);
	sprintf(key, "strPortIO");
	info->strPortIO = read_profile_int(section, key, 10001, file);
	sprintf(key, "UpServer");
	info->UpServer = read_profile_int(section, key, 1, file);
	sprintf(key, "strNetmask");
	read_profile_string(section, key, (char*)info->strNetmask, NET_SIZE, "1", file);
	sprintf(key, "strGateway");
	read_profile_string(section, key, (char*)info->strGateway, NET_SIZE, "1", file);
	sprintf(key, "strMac");
	read_profile_string(section, key, (char*)info->strMac, NET_SIZE + 2, "1", file);
	sprintf(key, "tcpPort");
	info->tcpPort = read_profile_int(section, key, 1, file);
	sprintf(key, "udpPort");
	info->udpPort = read_profile_int(section, key, 1, file);
	sprintf(key, "maxConn");
	info->maxConn = read_profile_int(section, key, 1, file);
	sprintf(key, "strDNS1");
	read_profile_string(section, key, (char*)info->strDNS1, NET_SIZE, "202.96.128.86", file);
	sprintf(key, "strDNS2");
	read_profile_string(section, key, (char*)info->strDNS2, NET_SIZE, "202.96.128.86", file);

	return 1;
}


//写网络信息
int WriteNetWorkParam(IVDNetInfo *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "NETWORK");
	char file[100] = {0};
	sprintf(file, NETWORKFILE);

	sprintf(key, "strIpaddr");
	write_profile_string(section, key, (char*)info->strIpaddr, file);
	sprintf(key, "strIpaddr1");
	write_profile_string(section, key, (char*)info->strIpaddr1, file);
	sprintf(key, "strIpaddr2");
	write_profile_string(section, key, (char*)info->strIpaddr2, file);
	sprintf(key, "strIpaddrIO");
	write_profile_string(section, key, (char*)info->strIpaddrIO, file);
	sprintf(key, "strPort");
	sprintf(value, "%d", info->strPort);
	write_profile_string(section, key, value, file);
	sprintf(key, "strPortIO");
	sprintf(value, "%d", info->strPortIO);
	write_profile_string(section, key, value, file);
	sprintf(key, "UpServer");
	sprintf(value, "%d", info->UpServer);
	write_profile_string(section, key, value, file);
	sprintf(key, "strNetmask");
	write_profile_string(section, key, (char*)info->strNetmask, file);
	sprintf(key, "strGateway");
	write_profile_string(section, key, (char*)info->strGateway, file);
	//sprintf(key, "strMac");
	//write_profile_string(section, key, (char*)info->strMac, file);
	sprintf(key, "tcpPort");
	sprintf(value, "%d", info->tcpPort);
	write_profile_string(section, key, value, file);
	sprintf(key, "udpPort");
	sprintf(value, "%d", info->udpPort);
	write_profile_string(section, key, value, file);
	sprintf(key, "maxConn");
	sprintf(value, "%d", info->maxConn);
	write_profile_string(section, key, value, file);
	sprintf(key, "strDNS1");
	write_profile_string(section, key, (char*)info->strDNS1, file);
	sprintf(key, "strDNS2");
	write_profile_string(section, key, (char*)info->strDNS2, file);

	return 1;
}

//读取切换时间
int ReadChTimeParam(IVDTimeStatu *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "CHANGETIME");
	char file[100] = {0};
	sprintf(file, CHANGETIMEFILE);

	sprintf(key, "0_time");
	info->timep1 = read_profile_int(section, key, 0, file);
	sprintf(key, "1_time");
	info->timep2 = read_profile_int(section, key, 0, file);
	sprintf(key, "2_time");
	info->timep3 = read_profile_int(section, key, 0, file);
	sprintf(key, "3_time");
	info->timep4 = read_profile_int(section, key, 0, file);

	return 1;
}

//写切换时间
int WriteChTimeParam(IVDTimeStatu *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "CHANGETIME");
	char file[100] = {0};
	sprintf(file, CHANGETIMEFILE);

	sprintf(key, "0_time");
	sprintf(value, "%lu", info->timep1);
	write_profile_string(section, key, value, file);
	sprintf(key, "1_time");
	sprintf(value, "%lu", info->timep2);
	write_profile_string(section, key, value, file);
	sprintf(key, "2_time");
	sprintf(value, "%lu", info->timep3);
	write_profile_string(section, key, value, file);
	sprintf(key, "3_time");
	sprintf(value, "%lu", info->timep4);
	write_profile_string(section, key, value, file);


	return 1;
}


//协议参数配置
int ReadProtocolParam(mThirProtocol *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "PROTOCOL");
	char file[100] = {0};
	sprintf(file, PROTOCOLLFILE);

	sprintf(key, "type");
	info->type = read_profile_int(section, key, 0, file);

	return 1;
}

int WriteProtocolParam(mThirProtocol *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "PROTOCOL");
	char file[100] = {0};
	sprintf(file, PROTOCOLLFILE);

	sprintf(key, "type");
	sprintf(value, "%d", info->type);
	write_profile_string(section, key, value, file);
	return 1;
}



//读取统计配置
int ReadStatisParam(IVDStatisSets *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "STATIS");
	char file[100] = {0};
	sprintf(file, STATISFILE);

	sprintf(key, "period");
	info->period = read_profile_int(section, key, 0, file);
	sprintf(key, "type");
	info->type = read_profile_int(section, key, 0, file);
	sprintf(key, "tiny");
	info->tiny = read_profile_int(section, key, 0, file);
	sprintf(key, "small");
	info->small = read_profile_int(section, key, 0, file);
	sprintf(key, "mediu");
	info->mediu = read_profile_int(section, key, 0, file);
	sprintf(key, "large");
	info->large = read_profile_int(section, key, 0, file);
	sprintf(key, "huge");
	info->huge = read_profile_int(section, key, 0, file);
	return 1;
}


//写统计配置
int WriteStatisParam(IVDStatisSets *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "STATIS");
	char file[100] = {0};
	sprintf(file, STATISFILE);

	sprintf(key, "period");
	sprintf(value, "%d", info->period);
	write_profile_string(section, key, value, file);
	sprintf(key, "type");
	sprintf(value, "%d", info->type);
	write_profile_string(section, key, value, file);
	sprintf(key, "tiny");
	sprintf(value, "%d", info->tiny);
	write_profile_string(section, key, value, file);
	sprintf(key, "small");
	sprintf(value, "%d", info->small);
	write_profile_string(section, key, value, file);
	sprintf(key, "mediu");
	sprintf(value, "%d", info->mediu);
	write_profile_string(section, key, value, file);
	sprintf(key, "large");
	sprintf(value, "%d", info->large);
	write_profile_string(section, key, value, file);
	sprintf(key, "huge");
	sprintf(value, "%d", info->huge);
	write_profile_string(section, key, value, file);
	return 1;
}

//读取串口配置
int ReadSerialParam(RS485CONFIG *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "SERIALINFO");
	char file[100] = {0};
	sprintf(file, SERIALFILE);

	sprintf(key, "uartNo");
	info->uartNo = read_profile_int(section, key, 0, file);
	sprintf(key, "protocol");
	info->protocol = read_profile_int(section, key, 0, file);
	sprintf(key, "buadrate");
	info->buadrate = read_profile_int(section, key, 0, file);
	sprintf(key, "databit");
	info->databit = read_profile_int(section, key, 0, file);
	sprintf(key, "stopbit");
	info->stopbit = read_profile_int(section, key, 0, file);
	sprintf(key, "checkbit");
	info->checkbit = read_profile_int(section, key, 0, file);
	return 1;
}

//写串口配置
int WriteSerialParam(RS485CONFIG *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "SERIALINFO");
	char file[100] = {0};
	sprintf(file, SERIALFILE);

	sprintf(key, "uartNo");
	sprintf(value, "%d", info->uartNo);
	write_profile_string(section, key, value, file);
	sprintf(key, "protocol");
	sprintf(value, "%d", info->protocol);
	write_profile_string(section, key, value, file);
	sprintf(key, "buadrate");
	sprintf(value, "%d", info->buadrate);
	write_profile_string(section, key, value, file);
	sprintf(key, "databit");
	sprintf(value, "%d", info->databit);
	write_profile_string(section, key, value, file);
	sprintf(key, "stopbit");
	sprintf(value, "%d", info->stopbit);
	write_profile_string(section, key, value, file);
	sprintf(key, "checkbit");
	sprintf(value, "%d", info->checkbit);
	write_profile_string(section, key, value, file);
	return 1;
}

/*---------------point--------------------*/
int ReadPoint(mPoint*point, char *file, int index, int keyindex)
{
	int i;
	char section[SECTIONMAX] = {0};
	char key[KEYMAX] = {0};
	sprintf(section, "%d_point", index + 1);
	sprintf(key, "%d_x", keyindex + 1);
	point->x = read_profile_int(section, key, 0, file);
	//printf("%d_x=%d ", keyindex+1, point->x);
	sprintf(key, "%d_y", keyindex + 1);
	point->y = read_profile_int(section, key, 0, file);
	//printf("%d_y=%d ", keyindex+1, point->y);
	return 1;
}


/*---------------读取前置线圈和占位线圈-----------*/
int ReadChannelCoil(mChannelCoil* coil, int camNo, int index)
{
	char file[100] = {0};
	int i;
	for (i = 0; i < COILPOINTMAX; i++) {
		sprintf(file, FRONTCOILFILE, camNo);
		//printf("[FrontCoil[%d]]\n", index);
		ReadPoint(&coil->FrontCoil[i], file, index, i);
		sprintf(file, MIDDLECOILFILE, camNo);
		ReadPoint(&coil->MiddleCoil[i], file, index, i);
		sprintf(file, REARCOILFILE, camNo);
		ReadPoint(&coil->RearCoil[i], file, index, i);
	}
	return 1;
}

int ReadCamDetectLane(mCamDetectLane *DetectLane, int camNo)
{
	int index;

	char file[100] = {0};
	char section[SECTIONMAX] = {0};
	sprintf(file, LANEIDFILE, camNo);
	char key[SECTIONMAX] = {0};

	sprintf(section, "LANEIDS");

	for (index = 0; index < DetectLane->lanenum; index++) {
		sprintf(key, "%d_lane_id", index);
		DetectLane->virtuallane[index].landID  = read_profile_int(section, key, 0, file);
		sprintf(key, "%d_Landtype", index);
		DetectLane->virtuallane[index].Landtype  = read_profile_int(section, key, 0, file);
		ReadChannelCoil(&DetectLane->virtuallane[index], camNo, index);
	}
	return 1;
}

//
int WritePoint(mPoint*point, char *file, int index, int keyindex)
{
	int i;
	char section[SECTIONMAX] = {0};
	char key[KEYMAX] = {0};
	char value[VALUEMAX] = {0};
	sprintf(section, "%d_point", index + 1);
	sprintf(key, "%d_x", keyindex + 1);
	sprintf(value, "%d", point->x);
	//printf("%d_x=%d\n", keyindex+1, point->x);
	write_profile_string(section, key, value, file);

	sprintf(key, "%d_y", keyindex + 1);
	sprintf(value, "%d", point->y);
	//printf("%d_y=%d\n", keyindex+1, point->y);
	write_profile_string(section, key, value, file);
	return 1;
}


/*---------------读取前置线圈和占位线圈-----------*/
int WriteChannelCoil(mChannelCoil* coil, int camNo, int index)
{
	char file[100] = {0};
	int i;
	for (i = 0; i < COILPOINTMAX; i++) {
		sprintf(file, FRONTCOILFILE, camNo);
		WritePoint(&coil->FrontCoil[i], file, index, i);
		sprintf(file, MIDDLECOILFILE, camNo);
		WritePoint(&coil->MiddleCoil[i], file, index, i);
		sprintf(file, REARCOILFILE, camNo);
		WritePoint(&coil->RearCoil[i], file, index, i);
	}
	return 1;
}

int WriteCamDetectLane(mCamDetectLane *DetectLane, int camNo)
{
	int index;
	char file[100] = {0};
	char value[VALUEMAX] = {0};
	char key[SECTIONMAX] = {0};
	char section[SECTIONMAX] = {0};
	sprintf(file, LANEIDFILE, camNo);


	sprintf(section, "LANEIDS");

	for (index = 0; index < DetectLane->lanenum; index++) {
		sprintf(key, "%d_lane_id", index);
		sprintf(value, "%d", DetectLane->virtuallane[index].landID);
		write_profile_string(section, key, value, file);
		sprintf(key, "%d_Landtype", index);
		sprintf(value, "%d", DetectLane->virtuallane[index].Landtype);
		write_profile_string(section, key, value, file);
		WriteChannelCoil(&DetectLane->virtuallane[index], camNo, index);
	}
	return 1;
}

//
int ReadLine(mLine* line, int index, int camNo)
{

	char section[SECTIONMAX] = {0};
	sprintf(section, "%d_point", index + 1);

	char file[100] = {0};
	sprintf(file, VIRLANELINEFILE, camNo);
	line->startx = read_profile_int(section, "1_x", 0, file);
	line->starty = read_profile_int(section, "1_y", 0, file);
	line->endx = read_profile_int(section, "2_x", 0, file);
	line->endy = read_profile_int(section, "2_y", 0, file);
	return 1;
}

/*----------读取车道线的值:最大个数为6条线,5车道----------*/
int ReadVirtualLaneLine(mVirtualLaneLine *virline, int camNo)
{
	int index;

	for (index = 0; index < LANELINEMAX; index++)
		ReadLine(&virline->laneline[index], index, camNo);
	return 1;
}


int WriteLine(mLine* line, int index, int camNo)
{
	char section[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, VIRLANELINEFILE, camNo);

	sprintf(section, "%d_point", index + 1);
	sprintf(value, "%d", line->startx);
	write_profile_string(section, "1_x", value, file);

	sprintf(value, "%d", line->starty);
	write_profile_string(section, "1_y", value, file);

	sprintf(value, "%d", line->endx);
	write_profile_string(section, "2_x", value, file);

	sprintf(value, "%d", line->endy);
	write_profile_string(section, "2_y", value, file);
	return 1;
}


/*----------读取车道线的值:最大个数为6条线,5车道----------*/
int WriteVirtualLaneLine(mVirtualLaneLine *virline, int camNo)
{
	int index;

	for (index = 0; index < LANELINEMAX; index++)
		WriteLine(&virline->laneline[index], index, camNo);
	return 1;
}


/*-------读取四个表定点的坐标和标定值-------*/
int ReadStandardPoint(mStandardPoint* standard, int index, int camNo)
{
	char section[SECTIONMAX] = {0};
	char file[100] = {0};
	sprintf(file, STANDARDPOINTFILE, camNo);
	sprintf(section, "%d_point", index + 1);
	standard->coordinate.x = read_profile_int(section, "x", 0, file);
	standard->coordinate.y = read_profile_int(section, "y", 0, file);
	standard->value = read_profile_int(section, "value", 0, file);
	return 1;
}


/*-------保存四个表定点的坐标和标定值-------*/
int WriteStandardPoint(mStandardPoint* standard, int index, int camNo)
{
	char section[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, STANDARDPOINTFILE, camNo);

	sprintf(section, "%d_point", index + 1);
	sprintf(value, "%d", standard->coordinate.x);
	write_profile_string(section, "x", value, file);

	sprintf(value, "%d", standard->coordinate.y);
	write_profile_string(section, "y", value, file);

	sprintf(value, "%d", standard->value);
	write_profile_string(section, "value", value, file);
	return 1;
}


/*--------读取标定区域的8个顶点的坐标和实际坐标-------*/
int ReadDemDetectArea(mDemDetectArea* Area, int camNo)
{
	int index;
	char section[SECTIONMAX] = {0};

	char file[100] = {0};
	sprintf(file, AREAFILE, camNo);
	for (index = 0; index < DETECT_AREA_MAX; index++) {
		sprintf(section, "%d_point", index + 1);
		//printf("%d_point from [%s]\n", index+1,  file);
		Area->vircoordinate[index].x = read_profile_int(section, "x", 0, file);
		//printf("x=%d\n", Area->vircoordinate[index].x);
		Area->vircoordinate[index].y = read_profile_int(section, "y", 0, file);
		//printf("y=%d\n", Area->vircoordinate[index].y);
		Area->realcoordinate[index].x = read_profile_int(section, "r_x", 0, file);
		//printf("r_x=%d\n", Area->realcoordinate[index].x);
		Area->realcoordinate[index].y = read_profile_int(section, "r_y", 0, file);
		//printf("r_y=%d\n", Area->realcoordinate[index].y);
	}
	return 1;
}


/*--------读取标定区域的8个顶点的坐标和实际坐标-------*/
int WriteDemDetectArea(mDemDetectArea* Area, int camNo)
{
	int index;
	char section[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, AREAFILE, camNo);

	for (index = 0; index < DETECT_AREA_MAX; index++) {
		sprintf(section, "%d_point", index + 1);
		sprintf(value, "%d", Area->vircoordinate[index].x);
		write_profile_string(section, "x", value, file);

		sprintf(value, "%d", Area->vircoordinate[index].y);
		write_profile_string(section, "y", value, file);

		sprintf(value, "%d", Area->realcoordinate[index].x);
		write_profile_string(section, "r_x", value, file);

		sprintf(value, "%d", Area->realcoordinate[index].y);
		write_profile_string(section, "r_y", value, file);
	}
	return 1;
}

int ReadDetectParam(mDetectParam *detect, int index, int camNo)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "%d_DetectParam", index + 1);

	char file[100] = {0};
	sprintf(file, DETECTPARAMFILE, camNo);

	sprintf(key, "uTransFactor");
	detect->uTransFactor = read_profile_int(section, key, 2, file);
	sprintf(key, "uGraySubThreshold");
	detect->uGraySubThreshold = read_profile_int(section, key, 200, file);

	if (0 == index) {
		sprintf(key, "uSpeedCounterChangedThreshold");
		detect->uSpeedCounterChangedThreshold = read_profile_int(section, key, 20, file);
		sprintf(key, "uSpeedCounterChangedThreshold1");
		detect->uSpeedCounterChangedThreshold1 = read_profile_int(section, key, 20, file);
		sprintf(key, "uSpeedCounterChangedThreshold2");
		detect->uSpeedCounterChangedThreshold2 = read_profile_int(section, key, 20, file);
	} else {
		sprintf(key, "uSpeedCounterChangedThreshold");
		detect->uSpeedCounterChangedThreshold = read_profile_int(section, key, 20, file);
		sprintf(key, "uSpeedCounterChangedThreshold1");
		detect->uSpeedCounterChangedThreshold1 = read_profile_int(section, key, 800, file);
		sprintf(key, "uSpeedCounterChangedThreshold2");
		detect->uSpeedCounterChangedThreshold2 = read_profile_int(section, key, 180, file);
	}

	sprintf(key, "uDayNightJudgeMinContiuFrame");
	detect->uDayNightJudgeMinContiuFrame = read_profile_int(section, key, 200, file);
	sprintf(key, "uComprehensiveSens");
	detect->uComprehensiveSens = read_profile_int(section, key, 20, file);
	sprintf(key, "uDetectSens1");
	detect->uDetectSens1 = read_profile_int(section, key, 20, file);
	sprintf(key, "uDetectSens2");
	detect->uDetectSens2 = read_profile_int(section, key, 20, file);
	sprintf(key, "uStatisticsSens1");
	detect->uStatisticsSens1 = read_profile_int(section, key, 15, file);
	sprintf(key, "uStatisticsSens2");
	detect->uStatisticsSens2 = read_profile_int(section, key, 3, file);
	sprintf(key, "uSobelThreshold");
	detect->uSobelThreshold = read_profile_int(section, key, 3, file);

	if (0 == index) {
		sprintf(key, "shutterMax");
		detect->shutterMax = read_profile_int(section, key, 2, file);

		sprintf(key, "shutterMin");
		detect->shutterMin = read_profile_int(section, key, 3, file);
	} else {
		sprintf(key, "shutterMax");
		detect->shutterMax = read_profile_int(section, key, 14, file);

		sprintf(key, "shutterMin");
		detect->shutterMin = read_profile_int(section, key, 14, file);
		//printf("shutterMin:%d\n", detect->shutterMin);
	}
	return 1;
}

int WriteDetectParam(mDetectParam *detect, int index, int camNo)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, DETECTPARAMFILE, camNo);

	sprintf(section, "%d_DetectParam", index + 1);

	sprintf(key, "uTransFactor");
	sprintf(value, "%d", detect->uTransFactor);
	write_profile_string(section, key, value, file);

	sprintf(key, "uGraySubThreshold");
	sprintf(value, "%d", detect->uGraySubThreshold);
	write_profile_string(section, key, value, file);

	sprintf(key, "uSpeedCounterChangedThreshold");
	sprintf(value, "%d", detect->uSpeedCounterChangedThreshold);
	write_profile_string(section, key, value, file);

	sprintf(key, "uSpeedCounterChangedThreshold1");
	sprintf(value, "%d", detect->uSpeedCounterChangedThreshold1);
	write_profile_string(section, key, value, file);

	sprintf(key, "uSpeedCounterChangedThreshold2");
	sprintf(value, "%d", detect->uSpeedCounterChangedThreshold2);
	write_profile_string(section, key, value, file);

	sprintf(key, "uDayNightJudgeMinContiuFrame");
	sprintf(value, "%d", detect->uDayNightJudgeMinContiuFrame);
	write_profile_string(section, key, value, file);

	sprintf(key, "uComprehensiveSens");
	sprintf(value, "%d", detect->uComprehensiveSens);
	write_profile_string(section, key, value, file);

	sprintf(key, "uDetectSens1");
	sprintf(value, "%d", detect->uDetectSens1);
	write_profile_string(section, key, value, file);

	sprintf(key, "uDetectSens2");
	sprintf(value, "%d", detect->uDetectSens2);
	write_profile_string(section, key, value, file);

	sprintf(key, "uStatisticsSens1");
	sprintf(value, "%d", detect->uStatisticsSens1);
	write_profile_string(section, key, value, file);

	sprintf(key, "uStatisticsSens2");
	sprintf(value, "%d", detect->uStatisticsSens2);
	write_profile_string(section, key, value, file);

	sprintf(key, "uSobelThreshold");
	sprintf(value, "%d", detect->uSobelThreshold);
	write_profile_string(section, key, value, file);

	sprintf(key, "shutterMax");
	sprintf(value, "%d", detect->shutterMax);
	write_profile_string(section, key, value, file);

	sprintf(key, "shutterMin");
	sprintf(value, "%d", detect->shutterMin);
	write_profile_string(section, key, value, file);
	return 1;
}

int ReadCamDemarcateParam(mCamDemarcateParam* dempram, int camNo)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	char file[100] = {0};
	sprintf(file, COMMONFILE, camNo);

	sprintf(section, "CamDemarcateParam");
	sprintf(key, "cam2stop");
	dempram->cam2stop = read_profile_int(section, key, 0, file);
	sprintf(key, "camheight");
	dempram->camheight = read_profile_int(section, key, 0, file);
	//printf("|---> Read camheight:[ %d ]\n", dempram->camheight);
	//sprintf(key, "lannum");
	//dempram->lannum = read_profile_int(section, key, 0, file);
	//sprintf(key, "number");
	//printf("|----> Read number:[ %d ]\n", dempram->lannum);

	//dempram->number = read_profile_int(section, key, 0, file);
	sprintf(key, "baselinelen");
	dempram->baselinelen = read_profile_int(section, key, 0, file);
	sprintf(key, "farth2stop");
	dempram->farth2stop = read_profile_int(section, key, 0, file);
	sprintf(key, "recent2stop");
	dempram->recent2stop = read_profile_int(section, key, 0, file);
	sprintf(key, "horizontallinelen");
	dempram->horizontallinelen = read_profile_int(section, key, 0, file);
	return 1;
}


int WriteCamDemarcateParam(mCamDemarcateParam* dempram, int camNo)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, COMMONFILE, camNo);

	sprintf(section, "CamDemarcateParam");
	sprintf(key, "cam2stop");
	sprintf(value, "%d", dempram->cam2stop);
	write_profile_string(section, key, value, file);

	sprintf(key, "camheight");
	sprintf(value, "%d", dempram->camheight);
	write_profile_string(section, key, value, file);
	/*
	sprintf(key, "lannum");
	sprintf(value, "%d", dempram->lannum);
	write_profile_string(section, key, value, file);
	printf("|----> Write number:[ %d ]\n", dempram->lannum);

	sprintf(key, "number");
	sprintf(value, "%d", dempram->number);
	write_profile_string(section, key, value, file);
	*/
	sprintf(key, "baselinelen");
	sprintf(value, "%d", dempram->baselinelen);
	write_profile_string(section, key, value, file);

	sprintf(key, "farth2stop");
	sprintf(value, "%d", dempram->farth2stop);
	write_profile_string(section, key, value, file);

	sprintf(key, "recent2stop");
	sprintf(value, "%d", dempram->recent2stop );
	write_profile_string(section, key, value, file);

	sprintf(key, "horizontallinelen");
	sprintf(value, "%d", dempram->horizontallinelen );
	write_profile_string(section, key, value, file);
	return 1;
}


//行人检测区域
int ReadPersonParam(mPersonDetectArea* per_area, int camNo)
{
	int index;
	char section[SECTIONMAX] = {0};

	char file[100] = {0};
	sprintf(file, PERSONFILE, camNo);
	for (index = 0; index < AREAMAX; index++) {
		sprintf(section, "%d_area", index + 1);
		per_area->area[index].id = read_profile_int(section, "id", 0, file);
		per_area->area[index].realcoordinate[0].x = read_profile_int(section, "1_x", 0, file);
		per_area->area[index].realcoordinate[0].y = read_profile_int(section, "1_y", 0, file);
		per_area->area[index].realcoordinate[1].x = read_profile_int(section, "2_x", 0, file);
		per_area->area[index].realcoordinate[1].y = read_profile_int(section, "2_y", 0, file);
		per_area->area[index].realcoordinate[2].x = read_profile_int(section, "3_x", 0, file);
		per_area->area[index].realcoordinate[2].y = read_profile_int(section, "3_y", 0, file);
		per_area->area[index].realcoordinate[3].x = read_profile_int(section, "4_x", 0, file);
		per_area->area[index].realcoordinate[3].y = read_profile_int(section, "4_y", 0, file);

		per_area->area[index].detectline[0].x = read_profile_int(section, "1_line_x", 0, file);
		per_area->area[index].detectline[0].y = read_profile_int(section, "1_line_y", 0, file);
		per_area->area[index].detectline[1].x = read_profile_int(section, "2_line_x", 0, file);
		per_area->area[index].detectline[1].y = read_profile_int(section, "2_line_y", 0, file);
	}

#if 0
	for (index = 0; index < 2; index++) {
		sprintf(section, "%d_detectline", index + 1);
		per_area->detectline[index].x =  read_profile_int(section, "x", 0, file);
		per_area->detectline[index].y =  read_profile_int(section, "y", 0, file);
	}
#endif

	return 1;
}

//行人检测区域
int WritePersonParam(mPersonDetectArea* per_area, int camNo)
{
	int index;
	char section[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, PERSONFILE, camNo);

	for (index = 0; index < AREAMAX; index++) {
		sprintf(section, "%d_area", index + 1);
		sprintf(value, "%d", per_area->area[index].id);
		write_profile_string(section, "id", value, file);

		sprintf(value, "%d", per_area->area[index].realcoordinate[0].x);
		write_profile_string(section, "1_x", value, file);

		sprintf(value, "%d", per_area->area[index].realcoordinate[0].y);
		write_profile_string(section, "1_y", value, file);

		sprintf(value, "%d", per_area->area[index].realcoordinate[1].x);
		write_profile_string(section, "2_x", value, file);

		sprintf(value, "%d", per_area->area[index].realcoordinate[1].y);
		write_profile_string(section, "2_y", value, file);

		sprintf(value, "%d", per_area->area[index].realcoordinate[2].x);
		write_profile_string(section, "3_x", value, file);

		sprintf(value, "%d", per_area->area[index].realcoordinate[2].y);
		write_profile_string(section, "3_y", value, file);

		sprintf(value, "%d", per_area->area[index].realcoordinate[3].x);
		write_profile_string(section, "4_x", value, file);

		sprintf(value, "%d", per_area->area[index].realcoordinate[3].y);
		write_profile_string(section, "4_y", value, file);

		sprintf(value, "%d", per_area->area[index].detectline[0].x);
		write_profile_string(section, "1_line_x", value,  file);
		sprintf(value, "%d", per_area->area[index].detectline[0].y);
		write_profile_string(section, "1_line_y", value,  file);

		sprintf(value, "%d", per_area->area[index].detectline[1].x);
		write_profile_string(section, "2_line_x", value,  file);
		sprintf(value, "%d", per_area->area[index].detectline[1].y);
		write_profile_string(section, "2_line_y", value,  file);

	}

#if 0
	for (index = 0; index < 2; index++) {
		sprintf(section, "%d_detectline", index + 1);
		sprintf(value, "%d", per_area->detectline[index].x);
		write_profile_string(section, "x", value, file);
		sprintf(value, "%d", per_area->detectline[index].y);
		write_profile_string(section, "y", value, file);
	}
#endif

	return 1;
}


//其他参数
int ReadOtherParam(mOtherparam* param, int camNo)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, OTHERFILE, camNo);

	sprintf(section, "OTHER");

	sprintf(key, "directio");
	param->directio = read_profile_int(section, key, 0, file);
	sprintf(key, "detecttype");
	param->detecttype = read_profile_int(section, key, 0, file);
	sprintf(key, "videotype");
	param->videotype = read_profile_int(section, key, 0, file);
	sprintf(key, "rtsppath");
	read_profile_string(section, key, (char*)param->rtsppath, FILEPATHMAX, "rtsp://", file);
	sprintf(key, "camIp");
	read_profile_string(section, key, (char*)param->camIp, IPADDRMAX, "127.0.0.1", file);
	sprintf(key, "camPort");
	param->camPort = read_profile_int(section, key, 0, file);
	sprintf(key, "username");
	read_profile_string(section, key, (char*)param->username, USERNAMEMAX, "admin", file);
	sprintf(key, "passwd");
	read_profile_string(section, key, (char*)param->passwd, USERNAMEMAX, "admin", file);
	sprintf(key, "detectaccuracy");
	param->detectaccuracy = read_profile_int(section, key, 0, file);
	sprintf(key, "personlimit");
	param->personlimit = read_profile_int(section, key, 0, file);
	sprintf(key, "pensondetecttype");
	param->pensondetecttype = read_profile_int(section, key, 0, file);
	sprintf(key, "camerId");
	param->camerId = read_profile_int(section, key, 0, file);
	sprintf(key, "filepath");
	read_profile_string(section, key, (char*)param->filePath, FILEPATHMAX, " ", file);
	sprintf(key, "camdirection");
	param->camdirection = read_profile_int(section, key, 0, file);
}

//其他参数
int WriteOtherParam(mOtherparam* param, int camNo)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, OTHERFILE, camNo);

	sprintf(section, "OTHER");

	sprintf(key, "directio");
	sprintf(value, "%d", param->directio);
	write_profile_string(section, key, value, file);

	sprintf(key, "detecttype");
	sprintf(value, "%d", param->detecttype);
	write_profile_string(section, key, value, file);

	sprintf(key, "videotype");
	sprintf(value, "%d", param->videotype);
	write_profile_string(section, key, value, file);

	sprintf(key, "camPort");
	sprintf(value, "%d", param->camPort );
	write_profile_string(section, key, value, file);

	sprintf(key, "detectaccuracy");
	sprintf(value, "%d", param->detectaccuracy );
	write_profile_string(section, key, value, file);

	sprintf(key, "personlimit");
	sprintf(value, "%d", param->personlimit );
	write_profile_string(section, key, value, file);

	sprintf(key, "pensondetecttype");
	sprintf(value, "%d", param->pensondetecttype );
	write_profile_string(section, key, value, file);


	sprintf(key, "camIp");
	param->camIp[IPADDRMAX - 1] = '\0';
	printf("camIp %d", strlen(param->camIp) );
	write_profile_string(section, key, param->camIp, file);

	sprintf(key, "username");
	param->username[USERNAMEMAX - 1] = '\0';
	write_profile_string(section, key, param->username, file);

	sprintf(key, "passwd");
	param->passwd[USERNAMEMAX - 1] = '\0';
	write_profile_string(section, key, (const char *)param->passwd, file);

	sprintf(key, "camerId");
	sprintf(value, "%d", param->camerId );
	write_profile_string(section, key, value, file);

	sprintf(key, "filepath");
	param->filePath[FILEPATHMAX - 1] = '\0';
	write_profile_string(section, key, (const char *)param->filePath, file);

	sprintf(key, "rtsppath");

	if (strlen(param->rtsppath) >= FILEPATHMAX ) {
		param->rtsppath[FILEPATHMAX - 1] = '\0';
	} else {
		param->rtsppath[strlen(param->rtsppath)] = '\0';
	}
	write_profile_string(section, key, param->rtsppath, file);

	sprintf(key, "camdirection");
	sprintf(value, "%d", param->camdirection );
	write_profile_string(section, key, value, file);

	return 1;
}

int ReadCamDetectParam(mCamDetectParam* detectparam, int camNo)
{
	printf("*********mCamDetectParam %d*********\n", sizeof(mCamDetectParam));
	int index;
	int value;
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	char file[100] = {0};
	sprintf(file, COMMONFILE, camNo);

	/*
	for(index=0;index<4;index++){
		sprintf(section, "TimeP");
		sprintf(key, "%d_time", index);
		switch(index){
			case 0: value=300;break;
			case 1: value=360;break;
			case 2: value=1080;break;
			case 3: value=1140;break;
		}

		//detectparam->timep[index] =read_profile_int(section, key, value, file);
		//printf("read timep[%d]: %d\n", index, detectparam->timep[index]);
	}
	 */
	sprintf(section, "Person");
	sprintf(key, "num");
	detectparam->personarea.num  = read_profile_int(section, key, value, file);


	sprintf(section, "CamDemarcateParam");
	detectparam->detectlane.lanenum = read_profile_int(section, "lanenum", 0, file);

	sprintf(section, "CamDemarcateParam");
	detectparam->laneline.lanelinenum = read_profile_int(section, "lanelinenum", 0, file);

	ReadCamDetectLane(&detectparam->detectlane, camNo);
	ReadVirtualLaneLine(&detectparam->laneline, camNo);
	for (index = 0; index < STANDPOINT; index++) {
		ReadStandardPoint(&detectparam->standpoint[index], index, camNo);
	}

	ReadDemDetectArea(&detectparam->area, camNo);
	for (index = 0; index < ALGMAX; index++) {
		ReadDetectParam(&detectparam->detectparam[index], index, camNo);
	}

	ReadPersonParam(&detectparam->personarea, camNo);
	ReadOtherParam(&detectparam->other, camNo);
	ReadCamDemarcateParam(&detectparam->camdem, camNo);
	return 1;
}



int WriteCamDetectParam(mCamDetectParam* detectparam, int camNo)
{
	int index;
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, COMMONFILE, camNo);

	/*
	for(index=0;index<4;index++){
		sprintf(section, "TimeP");
		sprintf(key, "%d_time", index);
		sprintf(value, "%d", detectparam->timep[index]);
		write_profile_string(section, key, value, file);
	}
	*/

	sprintf(section, "Person");
	sprintf(key, "num");
	sprintf(value, "%d", detectparam->personarea.num);
	write_profile_string(section, key, value, file);

	sprintf(section, "CamDemarcateParam");
	sprintf(value, "%d", detectparam->detectlane.lanenum);
	write_profile_string(section, "lanenum", value, file);

	sprintf(value, "%d", detectparam->laneline.lanelinenum);
	write_profile_string(section, "lanelinenum", value, file);

	WriteCamDetectLane(&detectparam->detectlane, camNo);
	WriteVirtualLaneLine(&detectparam->laneline, camNo);
	for (index = 0; index < STANDPOINT; index++) {
		WriteStandardPoint(&detectparam->standpoint[index], index, camNo);
	}
	WriteDemDetectArea(&detectparam->area, camNo);

	for (index = 0; index < ALGMAX; index++) {
		WriteDetectParam(&detectparam->detectparam[index], index, camNo);
	}

	WritePersonParam(&detectparam->personarea, camNo);
	WriteOtherParam(&detectparam->other, camNo);
	WriteCamDemarcateParam(&detectparam->camdem, camNo);

	return 1;
}



//协议参数配置
int ReadNtpParam(IVDNTP *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "NTPINFO");
	char file[100] = {0};
	sprintf(file, NTPFILE);

	sprintf(key, "ip");
	read_profile_string(section, key, (char*)info->ipaddr, NET_SIZE, "", file);
	sprintf(key, "port");
	info->port = read_profile_int(section, key, 0, file);
	sprintf(key, "cycle");
	info->cycle = read_profile_int(section, key, 0, file);

	return 1;
}

int WriteNtpParam(IVDNTP *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "NTPINFO");
	char file[100] = {0};
	sprintf(file, NTPFILE);

	sprintf(key, "ip");
	write_profile_string(section, key, (char*)info->ipaddr, file);

	sprintf(key, "port");
	sprintf(value, "%d", info->port);
	write_profile_string(section, key, value, file);

	sprintf(key, "cycle");
	sprintf(value, "%d", info->cycle);
	write_profile_string(section, key, value, file);
	return 1;
}

//相机状态
int ReadCamStatusParam(IVDCAMERANUM *info)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "CAMERASTATUS");
	char file[100] = {0};
	sprintf(file, CAMERASTATUSFILE);

	sprintf(key, "cam_num");
	info->cam_num = read_profile_int(section, key, 0, file);
	for (int i = 0; i < CAM_MAX; i++) {
		sprintf(key, "cam_%d", i);
		info->exist[i] = read_profile_int(section, key, 0, file);
	}
	return 1;
}


int WriteCamStatusParam(unsigned char index, unsigned char status, int cam_num)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "CAMERASTATUS");
	char file[100] = {0};
	sprintf(file, CAMERASTATUSFILE);

	sprintf(key, "cam_%d", index);
	sprintf(value, "%d", status);
	write_profile_string(section, key, value, file);

	sprintf(key, "cam_num");
	sprintf(value, "%d", cam_num);
	write_profile_string(section, key, value, file);

	return 1;
}


bool ConfigFileIsExist(unsigned char index)
{
	bool ret = false;
	struct stat filestat;
	char dir_path[100] = {0};
	char buf[200] = {0};
	sprintf(dir_path, CAMERADIR, index);
	sprintf(buf, "cp -rf ./config/0_cammer ./config/%d_cammer >/dev/null ", index);

	if (stat(dir_path, &filestat) != 0)
	{
		system(buf);
		ret = true;
	} else {
		if (!S_ISDIR(filestat.st_mode)) {
			system(buf);
			ret = true;
		} else
			ret = true;
	}

	return ret;
}

int ReadPersonAreasTimes(mAreaPlanInfo *info, int camNo, int area_i, int tm_i)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	char file[100] = {0};
	sprintf(file, PERSON_AREAS_TIMES, camNo);

	//sprintf(section, "%d_%d_%d_time", camNo, area_i, tm_i);
	sprintf(section, "%d_%d_time", area_i, tm_i);
	sprintf(key, "plannum");
	info->plannum = read_profile_int(section, key, 0, file);
	sprintf(key, "start_hour");
	info->start_hour = read_profile_int(section, key, 0, file);
	sprintf(key, "start_minute");
	info->start_minute = read_profile_int(section, key, 0, file);
	sprintf(key, "personlimit");
	info->personlimit = read_profile_int(section, key, 0, file);
	sprintf(key, "maxWaitTime");
	info->maxWaitTime = read_profile_int(section, key, 0, file);
	sprintf(key, "noPersonTime");
	info->noPersonTime = read_profile_int(section, key, 0, file);
	sprintf(key, "overTime");
	info->overTime = read_profile_int(section, key, 0, file);
	return 1;
}

//行人检测方案参数
int ReadPersonAreas(mPersonPlanInfo *info, int camNo)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	char file[100] = {0};
	sprintf(file, PERSON_AREAS, camNo);

	for (int i = 0; i < PERSON_AREAS_MAX; i++) {
		sprintf(section, "%d_Area", i);
		sprintf(key, "areanum");
		info[i].areaNum = read_profile_int(section, key, 0, file);
		sprintf(key, "planTotal");
		info[i].planTotal = read_profile_int(section, key, 0, file);

		for (int j = 0; j < info[i].planTotal && j < PERSON_AREAS_TIMES_MAX; j++) {
			ReadPersonAreasTimes(&info[i].plan[j], camNo, i, j);
		}
	}
	return 1;
}

int WritePersonAreasTimes(mAreaPlanInfo *info, int camNo, int area_i, int tm_i)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, PERSON_AREAS_TIMES, camNo);

	//sprintf(section, "%d_%d_%d_time", camNo, area_i, tm_i);
	sprintf(section, "%d_%d_time", area_i, tm_i);

	sprintf(key, "plannum");
	sprintf(value, "%d", info->plannum);
	write_profile_string(section, key, value, file);

	sprintf(key, "start_hour");
	sprintf(value, "%d", info->start_hour);
	write_profile_string(section, key, value, file);

	sprintf(key, "start_minute");
	sprintf(value, "%d", info->start_minute);
	write_profile_string(section, key, value, file);

	sprintf(key, "personlimit");
	sprintf(value, "%d", info->personlimit);
	write_profile_string(section, key, value, file);

	sprintf(key, "maxWaitTime");
	sprintf(value, "%d", info->maxWaitTime);
	write_profile_string(section, key, value, file);

	sprintf(key, "noPersonTime");
	sprintf(value, "%d", info->noPersonTime);
	write_profile_string(section, key, value, file);

	sprintf(key, "overTime");
	sprintf(value, "%d", info->overTime);
	write_profile_string(section, key, value, file);

	return 1;
}


//行人检测方案参数
int WritePersonAreas(mPersonPlanInfo *info, int camNo)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, PERSON_AREAS, camNo);

	for (int i = 0; i < PERSON_AREAS_MAX; i++) {
		sprintf(section, "%d_Area", i);
		sprintf(key, "areanum");
		sprintf(value, "%d", info[i].areaNum);
		write_profile_string(section, key, value, file);
		sprintf(key, "planTotal");
		sprintf(value, "%d", info[i].planTotal);
		write_profile_string(section, key, value, file);

		for (int j = 0; j < info[i].planTotal && j < PERSON_AREAS_TIMES_MAX; j++) {
			WritePersonAreasTimes(&info[i].plan[j], camNo, i, j);
		}
	}
	return 1;
}

//事件
int ReadEventParam(mEventInfo *info, int cam_indx)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "EventInfo");
	char file[100] = {0};
	sprintf(file, EVENTS_FILE, cam_indx);

	sprintf(key, "eventAreaNum");
	info->eventAreaNum = read_profile_int(section, key, 0, file);

	for (int i = 0; i < 8; i++) {
		sprintf(section, "%d_EventArea", i + 1);
		sprintf(key, "areaNum");
		info->eventArea[i].areaNum = read_profile_int(section, key, 0, file);
		for (int n = 0; n < 4; n++) {
			sprintf(key, "%d_%d_x_realcoordinate", i + 1, n + 1);
			info->eventArea[i].realcoordinate[n].x = read_profile_int(section, key, 0, file);
			sprintf(key, "%d_%d_y_realcoordinate", i + 1, n + 1);
			info->eventArea[i].realcoordinate[n].y = read_profile_int(section, key, 0, file);
		}
		sprintf(key, "eventType");
		info->eventArea[i].eventType.type = read_profile_int(section, key, 0, file);
		sprintf(key, "reserve");
		info->eventArea[i].reserve = read_profile_int(section, key, 0, file);
		sprintf(key, "direction");
		info->eventArea[i].direction = read_profile_int(section, key, 0, file);

		for (int p = 0; p < 32; p++) {
			sprintf(key, "%d_%d_report", i + 1, p + 1);
			info->eventArea[i].report[p] = read_profile_int(section, key, 0, file);
			sprintf(key, "%d_%d_reserve1", i + 1, p + 1);
			info->eventArea[i].reserve1[p] = read_profile_int(section, key, 0, file);
		}

	}

	return 1;
}


int WriteEventParam(mEventInfo *info, int cam_indx)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	sprintf(section, "EventInfo");
	char file[100] = {0};
	sprintf(file, EVENTS_FILE, cam_indx);


	sprintf(key, "eventAreaNum");
	sprintf(value, "%d", info->eventAreaNum);
	write_profile_string(section, key, value, file);

	for (int i = 0; i < 8; i++) {
		sprintf(section, "%d_EventArea", i + 1);
		sprintf(key, "areaNum");
		sprintf(value, "%d", info->eventArea[i].areaNum);
		write_profile_string(section, key, value, file);

		for (int n = 0; n < 4; n++) {
			sprintf(key, "%d_%d_x_realcoordinate", i + 1, n + 1);
			sprintf(value, "%d", info->eventArea[i].realcoordinate[n].x);
			write_profile_string(section, key, value, file);
			sprintf(key, "%d_%d_y_realcoordinate", i + 1, n + 1);
			sprintf(value, "%d", info->eventArea[i].realcoordinate[n].y);
			write_profile_string(section, key, value, file);
		}

		sprintf(key, "eventType");
		sprintf(value, "%d", info->eventArea[i].eventType.type);
		write_profile_string(section, key, value, file);
		sprintf(key, "reserve");
		sprintf(value, "%d", info->eventArea[i].reserve);
		write_profile_string(section, key, value, file);
		sprintf(key, "direction");
		sprintf(value, "%d", info->eventArea[i].direction);
		write_profile_string(section, key, value, file);

		for (int p = 0; p < 32; p++) {
			sprintf(key, "%d_%d_report", i + 1, p + 1);
			sprintf(value, "%d", info->eventArea[i].report[p]);
			write_profile_string(section, key, value, file);

			sprintf(key, "%d_%d_reserve1", i + 1, p + 1);
			sprintf(value, "%d", info->eventArea[i].reserve1[p]);
			write_profile_string(section, key, value, file);

		}

	}

	return 1;
}

//详细位置信息
void WritePosition(mPositionData *info, int index)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};
	char value[VALUEMAX] = {0};

	char file[100] = {0};
	sprintf(file, POSITION_FILE);

	sprintf(section, "Coordinate");
	sprintf(key, "Longitude");
	sprintf(value, "%f", info->Longitude);
	write_profile_string(section, key, value, file);

	sprintf(key, "Latitude");
	sprintf(value, "%f", info->Latitude);
	write_profile_string(section, key, value, file);

	sprintf(key, "Altitude");
	sprintf(value, "%f", info->Altitude);
	write_profile_string(section, key, value, file);

	sprintf(key, "RelativeHeight");
	sprintf(value, "%f", info->RelativeHeight);
	write_profile_string(section, key, value, file);

	sprintf(key, "YawAngle");
	sprintf(value, "%f", info->YawAngle);
	write_profile_string(section, key, value, file);

}

void ReadPosition(mPositionData *info, int index)
{
	char section[SECTIONMAX] = {0};
	char key[SECTIONMAX] = {0};

	sprintf(section, "Coordinate");
	char file[100] = {0};
	sprintf(file, POSITION_FILE, index);

	sprintf(key, "Longitude");
	info->Longitude = read_profile_float(section, key, 0.0, file);
	sprintf(key, "Latitude");
	info->Latitude = read_profile_float(section, key, 0.0, file);
	sprintf(key, "Altitude");
	info->Altitude = read_profile_float(section, key, 0.0, file);
	sprintf(key, "RelativeHeight");
	info->RelativeHeight = read_profile_float(section, key, 0.0, file);
	sprintf(key, "YawAngle");
	info->YawAngle = read_profile_float(section, key, 0.0, file);


}


