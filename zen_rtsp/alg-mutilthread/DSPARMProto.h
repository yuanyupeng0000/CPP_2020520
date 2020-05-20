///////////////////////////////////////////////////////
//		DSPARMProto.h
//
//		dsp与arm通讯协议
//  	BY DAVID 
//      20130512
//		VER: 1.01.00.00
///////////////////////////////////////////////////////

#ifndef __DSP_PROTO_H__
#define __DSP_PROTO_H__

#include "../client_obj.h"

#define MAX_DETECTOR_TYPES		2			//最大支持两种检测器
#define MAX_DETECTOR_ONETYPE	8			//每种检测器最大支持8个
////20131222
#define MAX_LANE		8	//	8			//每个检测器最大支持8个车道

//#define FULL_COLS  					(720)
#define FULL_COLS  					(640)
#define FULL_ROWS  					(480)
#define  CALIBRATION_POINT_NUM   8  //标定点数2015102
#define MAX_DETECTION_NUM 300//最大检测数
#define MAX_REGION_NUM 6//行人检测区域个数
#define MAX_DIRECTION_NUM 2//行人方向数
#define MAX_EVENT_TYPE 27//事件类型数量
#define MAX_EVENT_NUM 20//发生事件的最多数量
#define MAX_EVENT_REGION_NUM 100//最大事件检测区域数
//////////////////////////////////////////////////
//		结构定义: 	检测器配置信息结构
/////////////////////////////////////////////////

#ifndef __ALG_POINT__
#define __ALG_POINT__

typedef  unsigned int 	Uint32;
typedef  int 			Int32;
typedef  unsigned short Uint16;
typedef  short 			Int16;
typedef  int             BOOL;
typedef  unsigned char	Uint8;

typedef struct 
{
	Uint16 x;
	Uint16 y;
}CPoint;
#endif

#ifndef __ALG_RECT__
#define __ALG_RECT__
typedef struct 
{
	Uint16 x;
	Uint16 y;
	Uint16 width;
	Uint16 height;
}CRect;
#endif

typedef struct tagSPEEDLANE
{
	Uint16				uLaneID; //车道号
	Uint16              LaneType;//车道类型 0：竖向  1：横向
	//detect region
	CPoint FrontCoil[4];//占位线圈
	CPoint RearCoil[4];//流量线圈
    CPoint MiddleCoil[4];//占有线圈
	CPoint LaneRegion[4];//车道区域
	//CPoint				ptFourCorner[6];//四个点的坐标
	//CPoint				ptCornerQ[2];//排队前置线
	//CPoint				ptCornerQA[2];//警戒线
	//CPoint				ptCornerLB[2];//20150918
	//CPoint				ptCornerRB[2];//20150918
	Uint16				uDetectDerection;//方向
	Uint16				ptFrontLine;//钱直线
	Uint16				ptBackLine;//后直线
	Uint16				uReserved0[30];//保留

	//vehicle length and speed transformation params
	Uint16				uTransFactor;//转换系数
	
	//extended params by david 20130904
	Uint32 			uSpeedCounterChangedThreshold;
	Uint32 			uSpeedCounterChangedThreshold1;
	Uint32 			uSpeedCounterChangedThreshold2;
	Uint32				uGraySubThreshold;//灰度差分值域
	
	
	Uint16				uTrackParams[20];		//预留
	Uint16				uReserved1[20];			//预留
}SPEEDLANE;


/*车辆存在检测器定义,start*/
typedef struct tagZENITH_SPEEDDETECTOR {
	
	Uint16				uLaneTotalNum;//车道总数
	SPEEDLANE 			SpeedEachLane[MAX_LANE];
	Uint16				uEnvironmentStatus;			//环境状态, 1－白天  2－白天阴天(傍晚) 3－晚上路灯 4－晚上无路灯 0－其他  //20130930 by david
	//alg params
	Uint16				uDayNightJudgeMinContiuFrame;	//环境转换灵敏度 15
	Uint16				uComprehensiveSens;		//综合灵敏度 60
	Uint16				uDetectSens1;			//检测灵敏度1	 20	
	Uint16				uDetectSens2;			//检测灵敏度2    10
	Uint16				uStatisticsSens1;		//统计灵敏度1   15
	Uint16				uStatisticsSens2;		//统计灵敏度2   3
	Uint16				uSobelThreshold;		//阀值灵敏度    3
	Uint16             uEnvironment;           //20140320,白天 晚上不抽点与抽点的转换
	CPoint					ptactual[8];//标定点
	CPoint					ptimage[8];//标定点	
    //CPoint calibration_point[4];//标定区域点
	//CPoint base_line[2];//标定基准线点
	float base_length[2];//基准线长
	float near_point_length;//最近点距离
	float cam2stop;//相机到停止线的距离
	//CPoint PersonDetectArea[MAX_REGION_NUM][4];
	mPersonDetectArea PersonDetectArea;//行人检测参数
	mEventInfo       EventDetectArea;//事件检测参数
	Uint16				uReserved1[10];			//预留   
}ZENITH_SPEEDDETECTOR;
/*车辆存在检测器定义,end*/


/*车队长度检测器,start*/
typedef struct tagPRESENCELANE {
	Uint16 uReserved[256];
}PRESENCELANE;

typedef struct tagZENITH_PRESENCEDETECTOR {
	
	Uint16			uLaneTotalNum;
	PRESENCELANE		PresenseEachLane[MAX_LANE];
}ZENITH_PRESENCEDETECTOR;
/*车队长度检测器,end*/

//////////////////////////////////////////////////
//		结构定义: 	消息传输 arm ---> dsp
/////////////////////////////////////////////////

typedef struct tagSpeedCfgSeg {
	Uint16				uType;		//检测器类型:1－车辆存在检测器,2－车队长度检测器,3－256 保留,0－无检测器
	Uint16				uNum; 		//检测器个数
	ZENITH_SPEEDDETECTOR	uSegData[1];//同类型所有检测器配置数据区
} SPEEDCFGSEG;

typedef struct tagPresenceCfgSeg {
	Uint16				uType;		//检测器类型:1－车辆存在检测器,2－车队长度检测器,3－256 保留,0－无检测器
	Uint16				uNum; 		//检测器个数
	ZENITH_PRESENCEDETECTOR	uSegData[1];//同类型所有检测器配置数据区
} PRESENCECFGSEG;

//union NormalCfgSeg {
//	SPEEDCFGSEG  		cSpeedCfgSeg;
//	PRESENCECFGSEG		cPresenceCfgSeg;
//};

typedef struct tagMsgHeader {
    Uint16   	uFlag;
    Uint16  	uCmd;
    Uint16  	uMsgLength;		
} MSGHEADER;

typedef struct tagConfigInfoHeader {
    Uint16		uDetectPosition;
    Uint16		uDetectFuncs[2];	//
} CFGINFOHEADER;

typedef struct tagCfgMsg {
	MSGHEADER		uMsgHeader;
	CFGINFOHEADER	uCfgHeader;			
} CFGMSG;
//////////////////////////////////////////////////
//		结构定义: 	消息传输  dsp ---> arm
//////////////////////////////////////////////////
typedef struct 
{
	Uint32 DetectInSum;//入线圈流量数
	Uint32 DetectOutSum;//出线圈流量数
	unsigned int calarflag;//线圈占有状态
	Uint16	uVehicleSpeed;//车辆速度km/h
	Uint16  uVehicleType; //车辆类型
	Uint16  uVehicleDirection;//车辆运行方向
	Uint16	uVehicleLength;//车辆长度
	Uint16	uVehicleHeight;//车辆高度
	Uint16	uVehicleHeadtime; //车头时距
}RegionAttribute;

typedef struct tagSpeedDetectInfo {
	BOOL		bInfoValid;				//检测器结果有效
	Uint16	bVehicleSta;			//车入车出状态
	CPoint		ptVehicleCoordinate;	//车辆位置
	RegionAttribute CoilAttribute[2];//线圈属性
	Uint16  uLastVehicleLength;//最后一辆车的位置
	CPoint	LineUp[2];
	int AlarmLineflag;
	bool     IsCarInTailFlag;    //尾部区域占有标志
	bool     getQueback_flag;	//txl,20160104
	Uint16 uDetectRegionVehiSum; //区域车辆数
	Uint16 uVehicleQueueLength; //排队长度
	CPoint QueLine[2]; //排队长度线
	Uint16 uQueueHeadDis;//队首到停车线的距离，单位：m
	Uint16 uQueueTailDis;//队尾到停车线的距离，单位：m
	Uint16 uQueueVehiSum;//通道排队数量，单位：辆
	Uint16 uVehicleDensity;//空间占有率，单位：百分比
	Uint16 uVehicleDistribution;//车辆分布情况，单位：m
	Uint16 uHeadVehiclePos;//头车位置，单位：m
	Uint16 uHeadVehicleSpeed;//头车速度，单位：km/h
	Uint16 uLastVehiclePos;//末车位置，单位：m
	Uint16 uLastVehicleSpeed;//末车速度，单位：km/h
	Uint32 uBicycleFlow;//自行车流量
	Uint32 uBusFlow;//公交车流量
	Uint32 uCarFlow;//小车流量
	Uint32 uTruckFlow;//货车流量
	Uint32 uMotorbikeFlow;//摩托车流量
	Uint32 nVehicleFlow; //非机动车流量
	Uint16	uReserved[20];			//预留
}SpeedDetectInfo_t;

typedef struct tagPresenceDetectInfo {
	BOOL		bInfoValid;				//检测器结果有效
	Uint16	uMotorCadeLength;		//车队长度
	CPoint		ptLisnceCoordinate[4];	//车牌坐标
	Uint16	uLisnceID[10];			//车牌号码
	Uint16	uLisnceColor;			//车牌颜色
	Uint16	uVehicleColor;			//车身颜色
	Uint16	uVehicleBrand;			//车标类型(车辆品牌)
	Uint16	uSignalLightSta;		//信号灯状态			
	Uint16	uEnvirenmentSta;		//环境状态			
	Uint16	uReserved[100];			//预留
}PresenceDetectInfo_t;

typedef struct tagResultData {
	Uint16			uLaneID;				//车道ID
	Uint16			uReserved0;				//
	SpeedDetectInfo_t		SpeedDetectInfo1;
	PresenceDetectInfo_t PresenceDetectInfo;
	Uint16			uReserved[95];			//预留 
}RESULTDATA;

typedef struct {
	int x;
	int y;
	int w;
	int h;
	int label;//类型标签，1为bus 2为car 3为truck 4为motorbike 5为bicycle 6为person 7为车牌
	int confidence;//置信度
	int id;//目标id
	float distance[2];//目标中心点与相机距离
	float border_distance[4][2];//目标左上 右上、左下、右下与相机的距离
	int laneid;//车道号
	int speed;//目标y方向速度
	int speed_Vx;//目标x方向速度
	Uint16 width;//目标宽度
	Uint16 length;//目标长度
}OBJECTINFO;

typedef enum eventType{
	OVER_SPEED=1,//超速
	REVERSE_DRIVE, //逆行
	STOP_INVALID, //停车
	NO_PEDESTRIANTION, //行人
	DRIVE_AWAY, //驶离
	CONGESTION,//拥堵
	DROP,//抛洒物
	PERSONFALL = 10,//行人跌倒
	NONMOTORFALL = 12,//非机动车倒地
	NONMOTOR = 20,//禁行非机动车
	ACCIDENTTRAFFIC = 21,//交通事故
	GREENWAYDROP = 22,//绿道抛弃物
	ROADWATER = 23,//道路积水
	ROADHOLLOW = 24, //道路坑洼
	ROADDAMAGE = 25,//道路破损
	ROADCRACK = 26//道路裂缝
} EventType;//事件类型

typedef struct{
	Uint16 uRegionID;//事件区域ID
	Uint16 uNewEventFlag;//是否新事件标记
	Uint16 uEventID;//事件ID
	EventType uEventType;//事件类型
	CPoint EventBox[4];//事件框
}EVENTBOX;//事件框信息

typedef struct{
	Uint16 uNewEventFlag;//新产生事件标记
	Uint16 uEventNum;//事件数量
	EVENTBOX EventBox[MAX_EVENT_NUM];//事件框
}EVENTOUTBUF;

#define MAX_HELMET_NUM 5//非机动车上最大的安全帽数量
#define MAX_NONMONTOR_NUM 50//最大非机动车数量
typedef struct{
	CRect nonMotorBox;//非机动车检测框
	CRect helmetBox[MAX_HELMET_NUM];//安全帽检测框
	Uint16 helmetNum;//安全帽数
	Uint16 riderNum;//乘员人数 1 2 3 4 5
	bool hasHelmet;//是否带安全帽 true:带安全帽；false:未带安全帽
	bool overLoad;//是否超载 true:超过2人(3人或3人以上)；false:未超载
}NonMotorInfo;

typedef struct{
	Uint16 uNonMotorNum;//非机动车数量
	NonMotorInfo nonMotorInfo[MAX_NONMONTOR_NUM];//非机动车检测结果
}NPOUTBUF;//非机动车乘员信息
typedef struct tagResultInfo {
	Uint16 		LaneSum;						//
	Uint16			uEnvironmentStatus;		//环境状态, 1－白天  2－白天阴天(傍晚) 3－晚上路灯 4－晚上无路灯 0－其他  //20130930 by david
	//CRect udetBox[100];
	OBJECTINFO udetBox[MAX_DETECTION_NUM];
	Uint16 udetNum;//检测框
	OBJECTINFO udetPersonBox[MAX_DETECTION_NUM];
	Uint16 udetPersonNum;//行人检测框
	Uint16 udetStatPersonNum;//统计行人检测框
	Uint16 uPersonDirNum[MAX_DIRECTION_NUM];//分方向行人数
	Uint16 uBicycleDirNum[MAX_DIRECTION_NUM];//分方向自行车数
	Uint16 uMotorbikeDirNum[MAX_DIRECTION_NUM];//分方向摩托车数
	Uint16 uPersonRegionNum[MAX_REGION_NUM];//区域行人数
	car_objects car_number[MAX_DETECTION_NUM];//车牌检测框
	Uint16 udetPlateNum;//车牌检测框数
	EVENTOUTBUF eventData;
	NPOUTBUF NPData;//多乘员检测信息
    RESULTDATA			uEachLaneData[8];				//包含8个车道所有的检测信息
} RESULTINFO;

typedef struct tagResultMsg {
    MSGHEADER		uMsgHeader;
    RESULTINFO 		uResultInfo;
} RESULTMSG;


/////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////

#endif
