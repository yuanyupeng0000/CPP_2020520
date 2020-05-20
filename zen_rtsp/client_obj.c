/*
 * protocol.c
 *
 *  Created on: 2016��8��17��
 *      Author: root
 */
#include "common.h"
#include "client_obj.h"

#include "g_define.h"
#include <arpa/inet.h>

typedef struct pointers {
	mCamAttributes *p_mCamAttributes;
	mCommand *p_mCommand;
	short *p_short;
	int *p_int;
    IVDDevSets       *p_mBaseInfo;
    IVDNetInfo       *p_mNetInfo;
    TimeSetInterface *p_mData;
    IVDStatisSets    *p_statisset;
    IVDTimeStatu     *p_timestatu;
    RS485CONFIG      *p_rs485;
    IVDNTP           *p_ntp;
    mEventInfo        *p_event;
    //
	mDetectDeviceConfig *p_mDetectDeviceConfig;
	mCamParam *p_mCamParam;
	mCamDemarcateParam *p_mCamDemarcateParam;
	mChannelVirtualcoil *p_mChannelVirtualcoil;
	mCamDetectParam *p_mCamDetectParam;
	mCamDetectLane *p_mCamDetectLane;
	mChannelCoil *p_mChannelCoil;
	mPoint *p_mPoint;
	mLine *p_mLine;
	mVirtualLaneLine *p_mVirtualLaneLine;
	mStandardPoint *p_mStandardPoint;
	mDemDetectArea *p_mDemDetectArea;
	mDetectParam *p_mDetectParam;
	//
	mPersonPlanInfo *p_mPersonPlanInfo;

} m_pointers;


void net_decode_obj_n(unsigned char *addr,int type,int encode,int num,int size);
int get_obj_len(int class_type)
{
	int len=0;
	switch(class_type){
	case CLASS_NULL:
		break;
	case CLASS_char:
		break;
	case CLASS_short:
		len=2;
		break;
	case CLASS_int:
		len=4;
		break;
	case CLASS_mCommand:
	    len=sizeof(mCommand);
		break;
    case CLASS_mBaseInfo:
        len=sizeof(IVDDevSets);
        break;
    case CLASS_mSysInfo:
        len=sizeof(IVDDevInfo);
        break;
    case CLASS_mNetworkInfo:
        len=sizeof(IVDNetInfo);
        break;
    case CLASS_mAlgInfo:
        len=sizeof(mCamDetectParam);
        break;
    case CLASS_mDate:
        len=sizeof(TimeSetInterface);
        break;
    case CLASS_mStatiscInfo:
        len=sizeof(IVDStatisSets);
        break;
    case CLASS_mChangeTIME:
        len = sizeof(IVDTimeStatu);
        break;
    case CLASS_mSerialInfo:
        len = sizeof(RS485CONFIG);
        break;
    case CLASS_mProtocolInfo:
        len = sizeof(mThirProtocol);
        break;
    case CLASS_mNTP:
        len = sizeof(IVDNTP);
        break;
    case CLASS_mCameraStatus:
        len = sizeof(IVDCAMERANUM);
        break;
    case CLASS_mPersonAreaTimes:
        len = sizeof(mPersonPlanInfo) * PERSON_AREAS_MAX;
        break;
    case CLASS_mEventInfo:
        len = sizeof(mEventInfo);
        break;
   	default:
		prt(info,"not recognize");
		break;
	}
	return len;
}
void net_decode_obj(unsigned char *bf,int type,int encode)
{
	m_pointers pt;

	switch(type){
	case CLASS_NULL:
		break;
	case CLASS_char:
		break;
	case CLASS_short:
	{
	    pt.p_short=(short *)bf;
		if (!encode){
			 *pt.p_short = ntohs(*pt.p_short);
		}
		else {
		    *pt.p_short = htons(*pt.p_short);
		}
	}
		break;
	case CLASS_int:
	{
		pt.p_int = (int *) bf;
		if (!encode)
			*pt.p_int = ntohl(*pt.p_int);
		else
			*pt.p_int = htonl(*pt.p_int);
	}
		break;
	case CLASS_mCommand:
	{
	    pt.p_mCommand=(mCommand *)bf;
		//net_decode_obj((unsigned char *)& pt.p_mCommand->version,CLASS_char,encode);
		//net_decode_obj((unsigned char *)& pt.p_mCommand->prottype,CLASS_char,encode);
		net_decode_obj((unsigned char *)& pt.p_mCommand->objnumber,CLASS_short,encode);
		net_decode_obj((unsigned char *)& pt.p_mCommand->objtype,CLASS_short,encode);
		net_decode_obj((unsigned char *)& pt.p_mCommand->objlen,CLASS_int,encode);
    }
		break;
    case CLASS_mBaseInfo:
    {
		pt.p_mBaseInfo=(IVDDevSets *)bf;
		net_decode_obj((unsigned char *)&pt.p_mBaseInfo->timeset,CLASS_int,encode);
    }
	    break;
    case CLASS_mNetworkInfo:
    {
		pt.p_mNetInfo=(IVDNetInfo *)bf;
		net_decode_obj((unsigned char *)&pt.p_mNetInfo->strPort,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_mNetInfo->strPortIO,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_mNetInfo->tcpPort,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_mNetInfo->udpPort,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_mNetInfo->maxConn,CLASS_short,encode);
    }
        break;
    case CLASS_mAlgInfo:
    {
        pt.p_mCamDetectParam=(mCamDetectParam *)bf;
        for(int i = 0; i < DETECTLANENUMMAX; i++) {

            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectlane.virtuallane[i].landID,CLASS_int,encode);

            for(int r=0;r < COILPOINTMAX; r++) {
                net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectlane.virtuallane[i].RearCoil[r].x,CLASS_short,encode);
                net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectlane.virtuallane[i].RearCoil[r].y,CLASS_short,encode);
                net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectlane.virtuallane[i].MiddleCoil[r].x,CLASS_short,encode);
                net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectlane.virtuallane[i].MiddleCoil[r].y,CLASS_short,encode);
                net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectlane.virtuallane[i].FrontCoil[r].x,CLASS_short,encode);
                net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectlane.virtuallane[i].FrontCoil[r].y,CLASS_short,encode);
            }
        }
        /*
        for(int i = 0; i < DETECTLANENUMMAX; i++) {
            for(int r=0;r < COILPOINTMAX; r++) {
                prt(info, "*********************lane[%d] RearCoil[%d].x:%d  RearCoil[%d].y: %d", i, r,pt.p_mCamDetectParam->detectlane.virtuallane[i].RearCoil[r].x, r, pt.p_mCamDetectParam->detectlane.virtuallane[i].RearCoil[r].y);
            }
        }
        */
        for(int i = 0; i < LANELINEMAX; i++) {
           net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->laneline.laneline[i].startx,CLASS_short,encode);
           net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->laneline.laneline[i].starty,CLASS_short,encode);
           net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->laneline.laneline[i].endx,CLASS_short,encode);
           net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->laneline.laneline[i].endy,CLASS_short,encode);
        }


        for(int i = 0; i < STANDPOINT; i++){
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->standpoint[i].value,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->standpoint[i].coordinate.x,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->standpoint[i].coordinate.y,CLASS_short,encode);
        }

        for(int i = 0; i < DETECT_AREA_MAX; i++){
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->area.vircoordinate[i].x,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->area.vircoordinate[i].y,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->area.realcoordinate[i].x,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->area.realcoordinate[i].y,CLASS_short,encode);
        }

        for(int i = 0; i < ALGMAX; i++){
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uTransFactor,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uGraySubThreshold,CLASS_int,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uSpeedCounterChangedThreshold,CLASS_int,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uSpeedCounterChangedThreshold1,CLASS_int,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uSpeedCounterChangedThreshold2,CLASS_int,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uDayNightJudgeMinContiuFrame,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uComprehensiveSens,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uDetectSens1,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uDetectSens2,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uStatisticsSens1,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uStatisticsSens2,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].uSobelThreshold,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].shutterMax,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->detectparam[i].shutterMin,CLASS_short,encode);
        }

        for(int i = 0; i < AREAMAX; i++){
            for(int j = 0; j < 4;j++) {
                net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->personarea.area[i].realcoordinate[j].x,CLASS_short,encode);
                net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->personarea.area[i].realcoordinate[j].y,CLASS_short,encode);
            }

			for(int j = 0; j < 2; j++) {
				net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->personarea.area[i].detectline[j].x,CLASS_short,encode);
				net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->personarea.area[i].detectline[j].y,CLASS_short,encode);
			}

        }
#if 0
        for(int i = 0; i < 2; i++) {
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->personarea.detectline[i].x,CLASS_short,encode);
            net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->personarea.detectline[i].y,CLASS_short,encode);
        }
#endif
        net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->other.personlimit,CLASS_short,encode);
        net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->other.camPort,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->other.camerId,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->camdem.cam2stop,CLASS_short,encode);
        net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->camdem.camheight,CLASS_short,encode);
        net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->camdem.baselinelen,CLASS_short,encode);
        net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->camdem.farth2stop,CLASS_short,encode);
        net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->camdem.recent2stop,CLASS_short,encode);
		net_decode_obj((unsigned char *)&pt.p_mCamDetectParam->camdem.horizontallinelen,CLASS_short,encode);
    }
        break;

    case CLASS_mDate:
    {
        pt.p_mData=(TimeSetInterface *)bf;
		net_decode_obj((unsigned char *)&pt.p_mData->year,CLASS_short,encode);
    }
        break;
    case CLASS_mStatiscInfo:
    {
        pt.p_statisset=(IVDStatisSets *)bf;
		net_decode_obj((unsigned char *)&pt.p_statisset->period,CLASS_short,encode);
	}
        break;
     case CLASS_mChangeTIME:
     {
        pt.p_timestatu=(IVDTimeStatu *)bf;
        net_decode_obj((unsigned char *)&pt.p_timestatu->timep1,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_timestatu->timep2,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_timestatu->timep3,CLASS_int,encode);
        net_decode_obj((unsigned char *)&pt.p_timestatu->timep4,CLASS_int,encode);
     }
        break;
     case CLASS_mSerialInfo:
     {
        pt.p_rs485 = (RS485CONFIG *)bf;
        net_decode_obj((unsigned char *)&pt.p_rs485->uartNo,CLASS_short,encode);
        net_decode_obj((unsigned char *)&pt.p_rs485->protocol,CLASS_short,encode);
        net_decode_obj((unsigned char *)&pt.p_rs485->buadrate,CLASS_short,encode);
     }
        break;
      case CLASS_mNTP:
      {
        pt.p_ntp=(IVDNTP *)bf;
		net_decode_obj((unsigned char *)&pt.p_ntp->port,CLASS_int,encode);
		net_decode_obj((unsigned char *)&pt.p_ntp->cycle,CLASS_int,encode);
      }
        break;
      case CLASS_mEventInfo:
      {
        pt.p_event=(mEventInfo *)bf;

        for(int i = 0; i < 8; i++) {
            for(int n = 0; n < 4; n++) {
                net_decode_obj((unsigned char *)&pt.p_event->eventArea[i].realcoordinate[n].x,CLASS_short,encode);
                net_decode_obj((unsigned char *)&pt.p_event->eventArea[i].realcoordinate[n].y,CLASS_short,encode);
            }

            net_decode_obj((unsigned char *)&pt.p_event->eventArea[i].eventType.type,CLASS_int,encode);
            net_decode_obj((unsigned char *)&pt.p_event->eventArea[i].reserve,CLASS_int,encode);
        }
      }
        break;
	  case CLASS_mPersonAreaTimes:
	  	{
	  		for(int j = 0; j < PERSON_AREAS_MAX; j++) {
		  		pt.p_mPersonPlanInfo=(mPersonPlanInfo *) (bf + j*sizeof(mPersonPlanInfo));

				for(int i = 0; i < pt.p_mPersonPlanInfo->planTotal; i++) {
					//if (encode)
					//	prt(info,"arear[%d][%d] befor noPersonTime : %d  encode: %d", j, i, pt.p_mPersonPlanInfo->plan[i].noPersonTime, encode);
					net_decode_obj((unsigned char *)&pt.p_mPersonPlanInfo->plan[i].noPersonTime,CLASS_short,encode);
					net_decode_obj((unsigned char *)&pt.p_mPersonPlanInfo->plan[i].overTime,CLASS_short,encode);
					//if (encode)
					//	prt(info,"arear[%d][%d] after noPersonTime : %d  encode: %d", j, i, pt.p_mPersonPlanInfo->plan[i].noPersonTime, encode);
				}
	  		}
	  	}
	  	break;
	default:
		prt(info,"not recognize");
		break;
	}
}
void net_decode_obj_n(unsigned char *addr,int type,int encode,int num,int size)
{
	int i=0;
	for(i=0;i<num;i++){
	//	prt(info,"add  %p,size %d,num %d",(unsigned char *)(addr+i*size),size,num);
		net_decode_obj((unsigned char *)addr+i*size,type,encode);
	}
}

int prepare_pkt(unsigned char *p_start, int head_length,int reply_type, int class_type, int class_length,unsigned char *p_obj)
{
	int total_length;
	mCommand *p_head_obj=(mCommand *)p_start;
	p_head_obj->objtype=reply_type;
	unsigned char *p_obj_start=p_start+head_length;
	memcpy(p_obj_start,p_obj,class_length);
    net_decode_obj(p_obj_start,class_type,1);
    total_length=head_length+class_length;
    p_head_obj->objlen = class_length;
	return total_length;
}
int handle_pkt(unsigned char *p_start, int head_length, int class_type, int class_length)
{
	int total_length;
	mCommand *p_head_obj=(mCommand *)p_start;
//	p_head_obj->objtype=reply_type;
	unsigned char *p_obj_start=p_start+head_length;
	//memcpy(p_obj_start,p_obj,class_length);
    net_decode_obj(p_obj_start,class_type,0);
   // total_length=head_length+class_length;
	return 0;
}
int get_pkt(unsigned char *p_start, int head_length,int reply_type, int class_type, int class_length,unsigned char *p_obj)
{
	int total_length;
	mCommand *p_head_obj=(mCommand *)p_start;
	p_head_obj->objtype=reply_type;
	unsigned char *p_obj_start=p_start+head_length;
	memcpy(p_obj_start,p_obj,class_length);
    net_decode_obj(p_obj_start,class_type,1);
    total_length=head_length+class_length;
	return total_length;
}


