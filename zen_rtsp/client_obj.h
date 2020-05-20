/*
 * protocol.h
 *
 *  Created on: 2016年8月17日
 *      Author: root
 */

#ifndef PROTOCOL_H_
#define PROTOCOL_H_

#include <pthread.h>
#include "g_define.h"
//
#define SET_BASE_CMD 0x1001
#define READ_BASE_CMD 0x1002
#define RETURN_BASE_CMD 0x1003

#define  VERSION       0x01        //版本类型
#define  PROTTYPE      0x02        //协议版本

//网络协议类型
#define  GETSYSPARAM   0x0101
#define  REPSYSPARAM   0x0102
//基本信息
#define  GETBASEPARAM   0x1002      //请求基本信息
#define  SETBASEPARAM   0x1001	   //设置基本信息
#define  REPBASEPARAM   0x1003	   //设置基本信息
//网络参数
#define  GETNETWORKPARAM   0x1008      //请求网络参数信息
#define  SETNETWORKPARAM   0x1007	   //网络参数设置
#define  REPNETWORKPARAM   0x1009	   //回复网络参数信息
//切换时间设置
#define  GETCHTIMEPARAM   0x1035
#define  SETCHTIMEPARAM   0x1034
#define  REPCHTIMEPARAM   0x1036
//串口配置
#define  GETSERIALPARAM   0x1029
#define  SETSERIALPARAM   0x1028
#define  REPSERIALPARAM   0x1030
//统计参数设置
#define  GETSTATISPARAM   0x2017
#define  SETSTATISPARAM   0x2016
#define  REPSTATISPARAM   0x2018
//算法配置
#define  GETALGSPARAM   0x2014
#define  SETALGPARAM    0x2013
#define  REPALGPARAM   0x2015
//时间参数设置
#define  GETDATEPARAM   0x2030
#define  SETDATEPARAM   0x2032
#define  REPDATEPARAM   0x2031
//协议参数配置
//#define  GETPROTOCOLPARAM  0x1042
//#define  SETPROTOCOLPARAM  0x1041
//#define  REPPROTOCOLPARAM  0x1043
//ntp参数配置
#define  GETNTPPARAM       0x7001
#define  SETNTPPARAM       0x7002
#define  REPNTPPARAM       0x7003
//
#define  GETCAMERANUMPARAM    0x6001  //获取相机数量
#define  REPCAMERANUMPARAM    0x6002 //回复
#define  DELCAMERANUMPARAM    0x6004  //删除相机

#define SETPERSONARETIM 0x8001 //设置行人时间区域
#define GETPERSONARETIM 0x8002 //读取行人时间区域
#define REPPERSONARETIM 0x8003 //回复行人时间区域

#define SETEVENTPARAM 0x9001 //事件检测参数设置
#define GETEVENTPARAM 0x9002 //事件检测参数请求
#define REPEVENTPARAM 0x9003 //回复事件检测参数请求
#define EVENTUPDATE 0x9004 //事件数据上传

//设备重启
#define SETDEVREBOOT 0x5003

//
#define AREAMAX 8
//#########################################################
#define  SETDETECTDEVICE    0x0004    //设置检测设备参数命名
#define  GETDETECTDEVICE    0x0005    //获取检测设备参数命名
#define  REPDETECTDEVICE    0x0006    //获取检测设备参数命名回应

#define  SETCHECHPARAM    0x0007    //设置视频画线参数
#define  GETCHECHPARAM    0x0008    //获取视频画线参数
#define  REPCHECHPARAM    0x0009    //获取视频画线参数回应

#define  HEART    0x1000    //心跳包数据协议

#define  SHUTDOWN         0x1050    //关闭命令
#define  STARTREALDATA    0x1051    //开始命令
#define  REALTESTDATA     0x1053    //测试数据回应

#define  FILE_PLAY_START  0x1070
#define  FILE_PLAY_STOP   0x1071


#define  REALDATA    0x1060    //实时数据

#define  REPHEART    0x1002    //心跳包数据协议回应
#define  REBOOTZEN   0x2001   //重启命令
#define  FORKEXIT    0x3001   //程序异常退出

#define  COMMAND_STATUS 0x3060 
///////////////////////////////

#define  HEARTTIME     60
#define  USERNAMEMAX   20
#define  IPADDRMAX     16
#define  DEVNAMEMAX    50
#define  FILEPATHMAX   100

//相机方向代号
#define Z_NONE 0x0
#define Z_NORTH 0x1
#define Z_NORTHEAST 0x2
#define Z_EAST 0x4
#define Z_SOUTHEAST 0x8
#define Z_SOUTH 0x10
#define Z_SOUTHWEST 0x20
#define Z_WEST 0x40
#define Z_NORTHWEST 0x80

#define  COILPOINTMAX        4      //线圈四个顶点
#define  DETECTLANENUMMAX    8      //检测通道个数
#define  LANELINEMAX         2*DETECTLANENUMMAX  //车道线个数
#define  STANDPOINT          4      //标定定8个顶点
#define  STANDARDVAULEMAX    4
#define  ALGMAX              2      //算法参数个数
#define  MAXSUBSYSTEM        4
#define  DETECT_AREA_MAX     8

#define BOX_SIZE 300

#define FILE_PATH_MAX 100

#pragma pack(push)
#pragma pack(1)


typedef union
{
unsigned int type;
struct
{
unsigned char bit0:1;
unsigned char bit1:1;
unsigned char bit2:1;
unsigned char bit3:1;
unsigned char bit4:1;
unsigned char bit5:1;
unsigned char bit6:1;
unsigned char bit7:1;
unsigned char bit8:1;
unsigned char bit9:1;
unsigned char bit10:1;
unsigned char bit11:1;
unsigned char bit12:1;
unsigned char bit13:1;
unsigned char bit14:1;
unsigned char bit15:1;
unsigned char bit16:1;
unsigned char bit17:1;
unsigned char bit18:1;
unsigned char bit19:1;
unsigned char bit20:1;
unsigned char bit21:1;
unsigned char bit22:1;
unsigned char bit23:1;
unsigned char bit24:1;
unsigned char bit25:1;
unsigned char bit26:1;
unsigned char bit27:1;
unsigned char bit28:1;
unsigned char bit29:1;
unsigned char bit30:1;
unsigned char bit31:1;
}bits;
}mSelectType;


typedef struct Command{
	unsigned char version;       //版本类型VERSION
	unsigned char prottype;      //协议版本 PROTTYPE
	unsigned short objnumber;      //对象类型. 下标index
	unsigned short objtype;      //数据类型   例如: SETCAMPARAM
	unsigned int objlen;         //数据体的长度
}mCommand; //网络协议头

typedef struct
{
	uint16 x;
	uint16 y;
	uint16 width;
	uint16 height;
	int label;
    int confidence;
	int id;
	int distance[2];//传输出实际坐标值，distance[0]为水平(X)坐标，distance[1]为垂直(Y)坐标，也表示目标与相机的垂直距离
	int landid;
#if 1 //beijing
	unsigned short  speed;
    unsigned short speed_Vx;//目标横向速度
	unsigned short rl_width;
	unsigned short rl_lenght;
#endif
}IVDCRect;


typedef struct{
    unsigned short x;
    unsigned short y;
}IVDPoint;
/*-----------------实时检测数据-----------------*/
typedef struct{
    unsigned char   state;                    //车道状态   //0出车  1入车
    unsigned char   isCarMid;        //车道第二线圈状态（中间线圈） //0出车  1入车
    unsigned char   isCarInTail;      //车道第一线圈状态（视频最下端线圈） 0出车, 1入
    unsigned short  queueLength;              //车队的长度
    //unsigned int    vehnum;                   //车辆总数
    unsigned int    vehnum1;           //第一线圈车辆总数
    unsigned int    vehnum2;           //第二线圈车辆总数
    unsigned int    vehlength1;           //第一线圈车辆长度
    unsigned int    vehlength2;           //第二线圈车辆长度
    //unsigned int    speed; 					  //车辆的速度
    unsigned int    speed1; 				//第一线圈车辆的速度
    unsigned int    speed2; 				//第二线圈车辆的速度
    unsigned int    existtime1;      //第一线圈存在时间 单位ms
    unsigned int    existtime2;      //第二线圈存在时间 单位ms
    unsigned short  uActualDetectLength;       //虚拟线圈的长度  //近线圈
    unsigned short  uActualTailLength;			//虚拟线圈的长度  //远线圈
    IVDPoint        LineUp[2];                //当前车的实际坐标 起始点和终点
    unsigned int  BicycleFlow;//自行车流量
	unsigned int  BusFlow;//公交车流量
	unsigned int  CarFlow;//小车流量
	unsigned int  TruckFlow;//货车流量
	unsigned int  MotorbikeFlow;//摩托车流量
	unsigned int  nVehicleFlow; //非机动车流量
	unsigned char LaneDirection;//检测方向0表示来车方向为正向（即车头），1表示去车方向为正向（即车尾）
	unsigned int  Headway;//车头间距
}mRealLaneInfo;
typedef struct{
    int x;
    int y;
    int w;
    int h;
    int label;
    int confidense;
	int id;
	int distance[2];//传输出实际坐标值，distance[0]为水平(X)坐标，distance[1]为垂直(Y)坐标，也表示目标与相机的垂直距离
	int landid;
	#if 1 //beijing --0
	unsigned short  speed;
    unsigned short speed_Vx;//目横向速度
	unsigned short width;
	unsigned short lenght;
	#endif
}fvd_objects;

typedef struct{
 	short x;
    short y;
    short w;
	short h;
	char confidence; //置信度
	int id;//目标id
	int landid;//车道号
	char colour;//车牌颜色
	char type;//车牌类型
	char car_number[50];//车牌号
}car_objects; //车牌数据

typedef struct{
unsigned char id;//区域id   
uint16 personNum;//区域行人数量
unsigned int upperson;// 上行行人数据
unsigned int downperson;//下行行人数

}mRealPersonInfo;


typedef struct{
    unsigned char   flag;          //数据标志0xFF
    unsigned int    deviceId;//设备ID
	unsigned int    camId;//相机ID
    unsigned char   laneNum;       //实际车道个数
    unsigned char   curstatus;      //  1 是白天, 2 是夜晚
    unsigned char   fuzzyflag;                //视频异常状态
    unsigned char   visibility;		           //能见度状态
    unsigned short 	uDegreePoint[20][2];      //表定点的坐标. 0:x 1:y
    //unsigned short 	uDegreePoint[4][2];      //表定点的坐标. 0:x 1:y
    mRealLaneInfo   lane[DETECTLANENUMMAX];  //16
    // nanjing....
    unsigned char   area_car_num[DETECTLANENUMMAX];//car amount
    unsigned char   queue_len[DETECTLANENUMMAX];// len
    IVDPoint queue_line[DETECTLANENUMMAX][2];
    int rcs_num;
    fvd_objects rcs[BOX_SIZE];
    //IVDPoint detectline[2];//行人检测线坐标，0起点坐标和1终点坐标

   // unsigned int upperson;// 上行行人数据
   // unsigned int downperson;//下行行人数
    IVDCRect udetPersonBox[BOX_SIZE];//行人检测框
	uint16   udetPersonNum;//行人检测框数
    mRealPersonInfo personRegion[6];//区域行人数
	unsigned short plateNumber;//车牌数量
	car_objects plate_objs[100];//车牌
    unsigned int BicycleFlow1;//自行车正向流量
	unsigned int BicycleFlow2;//自行车反向流量
	unsigned int MotorbikeFlow1;//摩托车正向流量
	unsigned int MotorbikeFlow2;//摩托车反向流量
	unsigned int frame_no;
	unsigned int utc_tm;
	unsigned short 	uHorizontalDegreePoint[10][2]; //水平标定点的坐标. 0:x 1:y
}mRealStaticInfo;

#define CAM_CLOSED_STATUS 0
#define CAM_OPENED_STATUS 1
typedef struct caminfo {

	unsigned char camstatus;
	unsigned char camdirect;
	unsigned char cammerIp[IPADDRMAX];
} m_caminfo;

typedef struct DetectDeviceConfig{
	unsigned int  deviceID;   //检测器ID
	unsigned int  detectport;
	unsigned char camnum;
	unsigned char detectip[IPADDRMAX];
	unsigned char detectname[DEVNAMEMAX];
	m_caminfo cam_info[CAM_NUM_1];

}mDetectDeviceConfig;   //检测器设备参数,包含4个检测相机,

//-----------------界面配置----单个相机相关参数配置-------------
typedef struct CamAttributes{
	unsigned char direction;
	unsigned int  camID;      //相机的ID
	unsigned int  cammerport;
	unsigned int  adjustport;
	unsigned int  signalport;
	unsigned char urlname[USERNAMEMAX];  //为访问的流文件名字
	unsigned char username[USERNAMEMAX];
	unsigned char passwd[USERNAMEMAX];
	unsigned char cammerIp[IPADDRMAX];
	unsigned char adjustIp[IPADDRMAX];
	unsigned char signalIp[IPADDRMAX];
}mCamAttributes; //相机的属性

typedef struct CamDemarcateParam{
	unsigned short cam2stop;
	unsigned short camheight;
	//unsigned short lannum;    //车道数
	//unsigned short number;    //通道编号
	unsigned short baselinelen;
	unsigned short farth2stop;
	unsigned short recent2stop;
	unsigned short horizontallinelen;//水平标定线长
}mCamDemarcateParam; //相机标定参数

typedef struct ChannelVirtualcoil{
	unsigned short number;      //通道编号
	unsigned short farthCoillen;
	unsigned short recentCoillen;
}mChannelVirtualcoil; //通道虚拟线圈的参数

typedef struct CamParam{
	unsigned char coilnum;     //通道数
	mCamAttributes camattr;
	mCamDemarcateParam camdem;
	mChannelVirtualcoil channelcoil[DETECTLANENUMMAX];
}mCamParam;

//---------------线圈配置------单个相机检测参数配置---------
typedef struct mePoint{
    unsigned short x;
    unsigned short y;
}mPoint;//点坐标

typedef struct meLine{
    unsigned short startx;
    unsigned short starty;
    unsigned short endx;
    unsigned short endy;
}mLine; //线坐标

typedef struct RearCoil{
    int landID;//车道ID
    unsigned char Landtype; //杞﹂亾绫诲瀷锛?涓虹珫鍚戯紙鍨傜洿锛夛紝1涓烘í鍚戯紙姘村钩锛?
	mPoint RearCoil[COILPOINTMAX];  //占位检测线圈
	mPoint MiddleCoil[COILPOINTMAX];
	mPoint FrontCoil[COILPOINTMAX]; //前置线圈
}mChannelCoil;  //单通道虚拟线圈的位置

typedef struct CamDetectLane{
	unsigned char lanenum;                 //车道数
	mChannelCoil virtuallane[DETECTLANENUMMAX];
}mCamDetectLane;   //单个相机相关的车道数据, 相机检测检测通道虚拟线圈 //每一个车道包含两个线圈

typedef struct VirtualLaneLine{
	unsigned char lanelinenum;         //
	mLine         laneline[LANELINEMAX];
}mVirtualLaneLine;    //虚拟车道线 最大支持4个车道5根线

typedef struct StandardPoint{
	mPoint  coordinate;
	unsigned short value;
}mStandardPoint; //标定点的坐标和值

typedef struct DemDetectArea{
	mPoint  vircoordinate[DETECT_AREA_MAX];
	mPoint  realcoordinate[DETECT_AREA_MAX];
}mDemDetectArea;  // 标定坐标系8个点,虚拟点和实际坐标点

typedef struct DetectParam{
	unsigned short uTransFactor;
	unsigned int   uGraySubThreshold;
	unsigned int   uSpeedCounterChangedThreshold;
	unsigned int   uSpeedCounterChangedThreshold1;
	unsigned int   uSpeedCounterChangedThreshold2;
	unsigned short  uDayNightJudgeMinContiuFrame;//切换时二值化阈值
	unsigned short  uComprehensiveSens;//取背景的连续帧数
	unsigned short  uDetectSens1;//判断是车头的最小行数
	unsigned short  uDetectSens2;
	unsigned short  uStatisticsSens1;
	unsigned short  uStatisticsSens2;	//by david 20130910 from tagCfgs
	unsigned short  uSobelThreshold;//sobel阈值
	unsigned short  shutterMax;        // 1 2 3 4 5 6 7 8
	unsigned short  shutterMin;        // 1 2 3 4 5 6 7 8
}mDetectParam;


//系统信息
typedef struct{
	char devicetype[32];
	char firmwareV[32];
	char webV[32];
	char libV[32];
	char hardwareV[32];
	//char serial[32];
}IVDDevInfo;

//基本信息设置
typedef struct{
    char checkaddr[50];
    char devUserNo[8];
    char undefine2;
    uint8 autoreset;
    uint8 overWrite;
    uint8 loglevel;
    uint32 timeset;
    uint8  pro_type;
}IVDDevSets;

//时间参数设置
typedef struct{
	unsigned short year;
	unsigned char month;
	unsigned char date;
	unsigned char hour;
	unsigned char minute;
	unsigned char second;
}TimeSetInterface;
//ntp
typedef struct{
	char   ipaddr[16];
	uint32 port;
	uint32 cycle;
}IVDNTP;

typedef struct {
unsigned char cam_num;
char exist[CAM_MAX];  //0--不存在    1---存在
}IVDCAMERANUM;

//2.5.网络参数设置
typedef struct{
    char strIpaddr[16];
    char strIpaddr1[16];
    char strIpaddr2[16];
    char strIpaddrIO[16];
    uint32 strPort;
	uint32 strPortIO;
    char   UpServer;
    char   strNetmask[16];
    char   strGateway[16];
    char   strMac[20];
    uint32 tcpPort;
    uint32 udpPort;
    uint16 maxConn;
    char strDNS1[16];
    char strDNS2[16];
}IVDNetInfo;

//2.6.切换时间设置 //0 1代表凌晨 点1至点2   2 3代表 黄昏 点1至点2
typedef struct{
	unsigned int timep1;
	unsigned int timep2;
	unsigned int timep3;
	unsigned int timep4;
}IVDTimeStatu;

//串口参数设置
typedef struct{
    unsigned short  uartNo;       //uart number  // 0: com0 1:com1……
    unsigned short  protocol;     //protocol type  // 1- 应答 2-主动
    unsigned short  buadrate;     //rate of buad //0
    char   databit;      // data bit 8位
    char   stopbit;      // 1
    char   checkbit;     // 0 无 1-奇 2-偶
}RS485CONFIG;

//统计参数设置
typedef struct{
    uint16 period;
    uint8  type;
    uint8  tiny;
    uint8  small;
    uint8  mediu;
    uint8  large;
    uint8  huge;
}IVDStatisSets;

//协议参数设置
typedef struct{
    uint8  type;
}mThirProtocol;

typedef struct PersonArea{
unsigned char id;//区域id
mPoint  realcoordinate[4];
mPoint detectline[2];//行人检测坐标,0起点坐标和1终点坐标
}mPersonArea;

typedef struct PersonDetectArea{
unsigned char num;//区域总数
mPersonArea area[AREAMAX];
//mPoint detectline[2];//行人检测坐标,0起点坐标和1终点坐标

}mPersonDetectArea; //行人检测区域

typedef struct Otherparam{
unsigned char  directio;//检测方向
unsigned char detecttype;//检测类型
unsigned char videotype;//视频源
char rtsppath[FILEPATHMAX]; //rtsp地址
unsigned char  detectaccuracy;//检测精度
unsigned short personlimit;//行人门限值
unsigned char  pensondetecttype;//行人检测类型
char  camIp[IPADDRMAX];//相机ip
unsigned int   camPort;//相机ip
char  username[USERNAMEMAX];
unsigned char  passwd[USERNAMEMAX];
unsigned int camerId;
unsigned char camdirection;//鐩告満鏂瑰悜锛屾柟鍚戝畾涔夎涓婂畯瀹氫箟
char filePath[FILEPATHMAX];
}mOtherparam ;//类型等其他参数

typedef struct CamDetectParam{
	mCamDetectLane detectlane;        //检测车道参数
	mVirtualLaneLine  laneline;      //车道线
	mStandardPoint standpoint[STANDPOINT];       //标定点和坐标
	mDemDetectArea area;              //标定的区域
	mDetectParam detectparam[ALGMAX];   // 检测参数：0  白天的参数,  1代表晚上参数
    mPersonDetectArea personarea;        //行人检测区域
    mOtherparam other;  //类型参数等其他参数
    mCamDemarcateParam camdem; //相机标定参数
}mCamDetectParam;  //算法所有检测有关参数

typedef struct{
unsigned char id;//区域id   
unsigned int time;//行人等待时间
}mRealPersonTestInfo;


typedef struct
{
unsigned int    lagerVehnum;//大车流量
unsigned int    smallVehnum;//小车流量
unsigned int    Vehnum;//总流量
unsigned int    speed;//实时速度
unsigned int    aveSpeed;//平均速度
unsigned short  queueLength;//排队长度
unsigned int    timedist;//平均车头时距
unsigned int    share;//平均时间占有率
}mRealLaneTestInfo;

typedef struct{
  unsigned char   laneNum;       //实际车道总数
  mRealLaneTestInfo lane[DETECTLANENUMMAX]; //车道数据DETECTLANENUMMAX=8
  mRealPersonTestInfo person[TEST_PERSON_AREA_SIZE];
}mRealTestInfo;


typedef struct
{
unsigned char      areaNum;//区域编号
mPoint  realcoordinate[4];//区域坐标
mSelectType  eventType;//事件类型
unsigned int  reserve;//预留
unsigned char   direction;//正向设置
unsigned char report[32] ;//事件检测时间间隔
unsigned char reserve1[32];//预留
}mEventArea;

typedef struct{
  unsigned char   eventAreaNum; //事件区域总数
  mEventArea  eventArea[8]; //最大8事件检测个区域
}mEventInfo;

typedef struct{
unsigned char ereaId;//区域ID
unsigned int eventId;//事件id
unsigned char eventType;//事件类型
mPoint eventRect[4];//事件目标框坐标
unsigned char picPath[FILE_PATH_MAX];//图片路径
unsigned char videoPath[FILE_PATH_MAX];//视频路径
}mEventPara;

typedef struct{
unsigned char   flag;          //事件数据标志0xFE
unsigned int  deviceId;//设备ID
unsigned int  camId;//相机ID
unsigned char newEventFlag;//新事件标志
unsigned char eventNum;//事件数量
mEventPara eventData[32];
unsigned int time;//utc时间
}mRealEventInfo;


typedef struct{
    mRealStaticInfo static_info[CAM_MAX];
    unsigned int pre_car_num[CAM_MAX][DETECTLANENUMMAX][2];
    unsigned int car_num[CAM_MAX][DETECTLANENUMMAX][2];
    unsigned char reset_flag[CAM_MAX];
    mRealTestInfo real_test_info[CAM_MAX];
    mRealTestInfo real_test_one[CAM_MAX];
    mRealTestInfo real_test_five[CAM_MAX];
    unsigned char real_test_updated[CAM_MAX];
	mRealEventInfo real_event_info[CAM_MAX];
    pthread_mutex_t lock[CAM_MAX];
    pthread_mutex_t real_test_lock[CAM_MAX];
}EX_mRealStaticInfo;

typedef struct {
    mCommand        pack_head;
    mRealStaticInfo static_info;
} real_data_pack_t;

typedef struct {
    mCommand        pack_head;
    mRealTestInfo   real_test_info;
} real_test_data_pack_t;

typedef struct
{
	unsigned char   plannum;//时间段编号
	unsigned char   start_hour;//开始时间-小时
	unsigned char   start_minute;//开始时间-分钟
	unsigned char   personlimit;//等待人数上限阈值
	unsigned char   maxWaitTime;//最大忍耐时间
	unsigned short  noPersonTime;//最大无人时间
	unsigned short  overTime;//超时
}mAreaPlanInfo;

typedef struct{
  unsigned char areaNum; //行人检测区域编号
  unsigned char planTotal; //时间段数
  mAreaPlanInfo plan[48]; //最大时间段数48段
}mPersonPlanInfo;

typedef struct {
    unsigned char perso_num[PERSON_AREAS_MAX]; //实时数据
    unsigned char up_perso_num[PERSON_AREAS_MAX]; //上行数量
    unsigned char down_perso_num[PERSON_AREAS_MAX]; //下行数量
    long long   no_person_time[PERSON_AREAS_MAX]; //没有行人的时间
    long long   gj_no_person_time[PERSON_AREAS_MAX];
    long long   gj_have_person_time[PERSON_AREAS_MAX]; //区域有人时间
    long long   line_no_person_time[PERSON_AREAS_MAX]; //没有行人的时间
    unsigned int  prev_status[PERSON_AREAS_MAX]; //上一个状态，是否有人
    unsigned char work_staus; //视频异常或能见度低
    long long     start_over_person_limit_ms[PERSON_AREAS_MAX];
	long long     start_unover_person_limit_ms[PERSON_AREAS_MAX];
	long long     start_wait_ms[PERSON_AREAS_MAX];//等待时间second
	//unsigned int  prev_status_start_time[PERSON_AREAS_MAX];//上一个需要变化的状态
	unsigned char prev_send_area_status[PERSON_AREAS_MAX];
	unsigned char prev_type[PERSON_AREAS_MAX];
	pthread_mutex_t lock;
}mRealTimePerson;

typedef struct {
    unsigned char area_no;
    unsigned char area_status;
    unsigned char occupt;
    unsigned char person_status;
    unsigned char work_status;
    unsigned char area_person_num;
    unsigned int  wait_sec;
}mPersonCheckData;

/*
typedef struct Point{
    unsigned short x;
    unsigned short y;
}mPoint;
*/

typedef struct {
    double Longitude;
	double Latitude;
	double Altitude;
	double RelativeHeight;
	double YawAngle;
}mPositionData;

typedef struct {
	char WSType[5];
	char DeviceNo[20];
	char Timestamp[25];
	int  DetectorType;
	int  TargetNum;
}mTargetHead;

typedef struct {
	int ID;
	int ParticipantType;
	double XPos;
	double YPos;
	double Speed;
	double Length;
	double Width;
	double Longitude;
	double Latitude;
	double Altitude;
	double RelativeHeight;
	double YawAngle;
}mTargetData;

typedef struct {
	unsigned short data_id:16;
	unsigned char data_len:8;
	unsigned char Mode_Signal:1;
	unsigned short x_Point:13;
	unsigned short y_Point:13;
	unsigned short Speed_x:11;
	unsigned short Speed_y:11;
	unsigned char Object_Length:7;
	unsigned char Object_ID:8;
}mRadarObj;

typedef struct {
	 float x_Point;
	 float y_Point;
	 float Speed_x;
	 float Speed_y;
	 float Obj_Len;
}mRadarRTObj;

typedef struct {
	bool cmd_play;
	int radar_sock;
	//third handle
	int sock_fd;
	pthread_mutex_t third_lock;
	
}mGlobalVAR;

typedef struct {
    mCommand        pack_head;
    mRealEventInfo  real_event_info;
} event_data_pack_t;

typedef struct{
	mCommand pack_head;
	unsigned int index;
	unsigned short type;
	unsigned char state;
}ret_cmd_status_t; //命令返回

#pragma pack(pop)

enum{
	CLASS_NULL,
	CLASS_char,
	CLASS_short,
	CLASS_int,
	CLASS_mCommand,
	CLASS_mBaseInfo,
	CLASS_mAlgInfo,
	CLASS_mDate,
	CLASS_mNTP,
	CLASS_mSysInfo,
	CLASS_mNetworkInfo,
	CLASS_mChangeTIME,
	CLASS_mSerialInfo,
	CLASS_mStatiscInfo,
	CLASS_mProtocolInfo,
	CLASS_mCameraStatus,
	CLASS_mCameraDelete,
    CLASS_mPersonAreaTimes,
    CLASS_mEventInfo
};
void net_decode_obj(unsigned char *bf,int type,int encode);
void net_decode_obj_n(unsigned char *addr,int type,int encode,int num,int size);
int get_obj_len(int class_type);
int prepare_pkt(unsigned char *p_start, int head_length,int reply_type, int class_type, int class_length,unsigned char *p_obj);
int handle_pkt(unsigned char *p_start, int head_length, int class_type, int class_length);
int get_pkt(unsigned char *p_start, int head_length,int reply_type, int class_type, int class_length,unsigned char *p_obj);

#endif /* PROTOCOL_H_ */
