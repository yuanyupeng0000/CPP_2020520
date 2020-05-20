#ifndef __FVD_CONFIG_H_
#define __FVD_CONFIG_H_

#include "../client_obj.h"

//
#define CHECKADDR 50
#define DEVUSERNO 8
//
#define SYS_SIZE 32
//
#define NET_SIZE 16

int ReadBaseParam(IVDDevSets *devsets);
int WriteBaseParam(IVDDevSets *devsets);
int ReadSysParam(IVDDevInfo *info);

int ReadNetWorkParam(IVDNetInfo *info);
int WriteNetWorkParam(IVDNetInfo *info);

int ReadStatisParam(IVDStatisSets *info);
int WriteStatisParam(IVDStatisSets *info);

int ReadCamDetectParam(mCamDetectParam* detectparam, int camNo);
int WriteCamDetectParam(mCamDetectParam* detectparam, int camNo);

int ReadChTimeParam(IVDTimeStatu *info);
int WriteChTimeParam(IVDTimeStatu *info);

int ReadSerialParam(RS485CONFIG *info);
int WriteSerialParam(RS485CONFIG *info);

int ReadProtocolParam(mThirProtocol *info);
int WriteProtocolParam(mThirProtocol *info);

int ReadNtpParam(IVDNTP *info);
int WriteNtpParam(IVDNTP *info);

int ReadCamStatusParam(IVDCAMERANUM *info);
int WriteCamStatusParam(unsigned char index, unsigned char status,int cam_num);
//
bool ConfigFileIsExist(unsigned char index);
//int  ReadPersonAreas(mPersonPlanInfo **info, int camNo);
int ReadPersonAreasTimes(mAreaPlanInfo *info, int camNo, int area_i, int tm_i);
int ReadPersonAreas(mPersonPlanInfo *info, int camNo);
int WritePersonAreasTimes(mAreaPlanInfo *info, int camNo, int area_i, int tm_i);
int WritePersonAreas(mPersonPlanInfo *info, int camNo);

//int WritePersonAreas(mPersonPlanInfo (*info)[PERSON_AREAS_MAX], int camNo);
//
int ReadEventParam(mEventInfo *info, int cam_indx);
int WriteEventParam(mEventInfo *info, int cam_indx);
//
void WritePosition(mPositionData *info, int index);
void ReadPosition(mPositionData *info, int index);

/*
int LoadConfig(mSystemConfig * sysconfig);
int SaveConfig(mSystemConfig * sysconfig);

int WriteCamAttributes(mCamAttributes*camattr, int camNo);
int WriteCammerConfig(mCamParam *camparam, int camNo);
int WriteCamDetectParam(mCamDetectParam* detectparam, int camNo);
int WriteDetectDeviceConfig(mDetectDeviceConfig* devconfig);
*/

#endif
