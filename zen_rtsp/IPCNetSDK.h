/*******************************************************************************
Copyright (c) HKVision Tech. Co., Ltd. All rights reserved.
--------------------------------------------------------------------------------

Date Created: 2011-10-25
Author: wuxiaofan
Description: 网络通讯库和数据获取SDK头文件，包括PTZ和监听报警的封装

--------------------------------------------------------------------------------
Modification History
DATE          AUTHOR          DESCRIPTION
--------------------------------------------------------------------------------
YYYY-MM-DD

*******************************************************************************/

#ifndef _INF_SDK_NET_H_
#define _INF_SDK_NET_H_
 
#define DO_NOTHING

#if defined(WIN32)
	#ifndef INLINE
		#define INLINE __inline
	#endif
#else
#define INLINE inline
#endif


/*use for parameter INPUT, *DO NOT Modify the value* */
#define IN
/* use for parameter OUTPUT, the value maybe change when return from the function 
 * the init value is ingore in the function.*/
#define OUT
/*use for parameter INPUT and OUTPUT*/
#define IO

/* --------------------------------  */
#define EXTERN extern
#define STATIC static

#define LOCALVAR static
#define GLOBALVAR extern


/*申明全局变量时*/
#define DECLARE_GLOBALVAR 

/*使用全局变量时, 用此声明*/
#define USE_GLOBALVAR extern
#define LOCALFUNC    static
#define EXTERNFUNC   extern

/*低层次的LOW API*/
#define LAPI  
/*高层次API*/
#define HAPI  
/*Multimedia Frame API*/
#define MMFAPI

/* -------- Standard input/output/err *****/
#define STDIN  stdin
#define STDOUT stdout
#define STDERR stderr
#if 0
//MARCO define
#define SAFE_DELETE(p)  { if(p != NULL) { delete (p);     (p) = NULL; } }   //Delete object by New create 
#define SAFE_DELETEA(p) { if(p != NULL) { delete[] (p);   (p) = NULL; } }   //Delete Arrary
#define SAFE_RELEASE(p) { if(p != NULL) { (p)->Release(); (p) = NULL; } }
#define SAFE_FREE(p)	{ if(p != NULL) { free(p); (p) = NULL; } }
#endif

#define SOCKET_ERROR            (-1)
#define INVALID_SOCKET -1
#define WEAK __attribute__((weak))
#define Packed  __attribute__((aligned(1),packed))

#define MAX_FILENAME_LEN 128
#define AUDIO_SAMPLE_RATE 8000
#define AUDIO_QUANTSIZE 16
#define AUDIO_CHANNELS 1


#ifndef    NULL
#define	   NULL 0
#endif
#define TRUE 1
#define FALSE 0

typedef unsigned long       DWORD;
typedef unsigned short      WORD;
typedef void *LPVOID;
//typedef int BOOL;
#ifndef BOOL
#define BOOL int
#endif
typedef unsigned char BYTE;

typedef unsigned int        UINT32, *PUINT32;

typedef unsigned int UINT;

typedef void VOID;

/******************************************************************************
SDKNET错误码定义，对应NET_IPC_GetLastError接口的返回值
*******************************************************************************/
//通用错误
#define ERR_SUCCEED         0           /**< 执行成功 */
#define ERR_FAIL            -1          /**< 执行失败 */
#define ERR_INVALIDPARAM    0x80000001  /**< 输入参数非法 */
#define ERR_NOMEMORY        0x80000002  /**< 内存分配失败 */
#define ERR_SYSFAIL         0x80000003  /**< 系统通用错误 */
#define ERR_USERNAME        0x80000004  /**< 用户名错误 */
#define ERR_PASSWORD        0x80000005  /**< 密码错误 */
#define ERR_NOINIT          0x80000006  /**< 没有初始化 */
#define ERR_INVALIDCHANNEL  0x80000007  /**< 通道号错误 */
//网络错误
#define ERR_OPENSOCKET      0x80000008  /**< 创建SOCKET错误 */
#define ERR_SEND            0x80000009  /**< 向设备发送网络数据失败 */
#define ERR_RECV            0x80000010  /**< 从设备接收网络数据失败 */
#define ERR_CONNNECT        0x80000011  /**< 连接设备失败，设备不在线、设备忙或网
										*络原因引起的连接超时等 */


//版本
#define IP_VERSION4					4			//IPV4
#define IP_VERSION6					6			//IPV6

//音视频类型
#define DATA_VIDEO					1			//视频流
#define DATA_AUDIO					2			//音频流
#define DATA_AV						3			//复合流

//视频编码类型
#define VIDEO_MPEG4_MAJOR			1			//MPEG4主码流
#define VIDEO_MPEG4_MINOR			2			//MPEG4次码流
#define VIDEO_MJPEG					3			//MJPEG码流
#define VIDEO_H264_MAJOR			4			//H264主码流
#define VIDEO_H264_MINOR			5			//H264次码流

//音频编码类型

#define	AUDIO_G711_A				0x02		//G711_A
#define	AUDIO_G711_U				0x01		//G711_U
#define	AUDIO_ADPCM_A				0x03		//ADPCM_A
#define	AUDIO_G726					0x04		//G726
#define	AUDIO_G711_A_HI				0x05		//HI H264
#define	AUDIO_G711_U_HI				0x06		//HI H264
#define	AUDIO_G726_HI				0x07		//HI H264

//帧类型
#define FRAME_VOL					0xD0		//VOL
#define FRAME_IVOP					0xD1		//I帧
#define FRAME_PVOP					0xD2		//P帧
#define FRAME_AUDIO					0xD3		//音频帧

//登录请求错误码
#define RE_SUCCESS					0			//成功
#define RE_USERNAME_ERROR			1			//用户名错误
#define RE_PASSWORD_ERROR			2			//密码错误


//视频请求错误码
#define RE_AV_SUCCESS               0           //正常
#define RE_AV_FULL_ERROR            1           //连接已满
#define RE_AV_LOST_ERROR            2           //视频丢失
#define RE_AV_TYPE_ERROR            3           //非音视频类型
#define RE_AV_NONSUPPORT_ERROR      4           //不支持 
#define RE_AV_NO_PRIVILEGE          5           //没有权限

#define DEVICE_TYPE_ASIC	0x00
#define DEVICE_TYPE_DSP		0x01
#define DEVICE_TYPE_H264	0x02
#define DEVICE_TYPE_DM355	0x03
#define DEVICE_TYPE_POWERPC	0x04
#define DEVICE_TYPE_HI3510	0x05
#define DEVICE_TYPE_DM365   0x06
#define DEVICE_TYPE_HI3512  0x07
#define DEVICE_TYPE_MG3500  0x08
#define DEVICE_TYPE_DM368	0x09
#define DEVICE_TYPE_3061	0x0A


#define TEXT_LENGTH					32

//音频采样类型
#define PLAY_AUDIO_SAMPLE_POOR		8000
#define PLAY_AUDIO_SAMPLE_LOW		11025
#define PLAY_AUDIO_SAMPLE_NORMAL	22050
#define PLAY_AUDIO_SAMPLE_HIGH		44100

//转发请求客户端类型
#define	CLIENT_DECODER				0x01		//DECODER
#define	CLIENT_LMC					0x02		//LMC
#define	CLIENT_SMT					0x03		//SMT
#define	CLIENT_NVR					0x04		//NVR
#define CLIENT_WEB					0x05		//Web 

/******************************************************************************
SDKNET数据结构定义
*******************************************************************************/
/**
* @struct tagPlayParam
* @brief 编码类型参数
* @attention
*/
typedef enum tagEncodeType
{
	ENCODE_MPEG4 = 1,   /**< MPEG4编码 */
	ENCODE_H264,        /**< H264编码 */
	ENCODE_H264_Hi3510, /**< H264 3510编码 */
	ENCODE_MJPEG,       /**< MJPEG编码，暂无用 */
}E_ENCODE_TYPE;

/**
* @struct tagRealDataInfo
* @brief 实时数据流参数
* @attention
*/
typedef struct tagRealDataInfo
{
	unsigned long lChannel;    /**< 通道号，从0开始 */
	unsigned long lStreamMode; /**< 码流类型，0-主码流，1-子码流 */
	unsigned int eEncodeType; /**< 编码类型 */
}S_REALDATA_INFO;

typedef struct{
	int iChannel;		/**<通道号从0开始*/
	int iAVType;		/**<音视频类型:1~视频流2~音频流3~复合流*/
	int iEncodeType;	/**<编码类型:3~Mjpeg 4~主码流5~副码流*/
}S_REALPLAY_INFO;

/**
* @enum tagRealDataType
* @brief 回调实时流的数据类型
* @attention 无
*/
typedef enum tagRealDataType
{
	REALDATA_HEAD,   /**< 实时流的头数据 */
	REALDATA_VIDEO,  /**< 实时视频流数据（包括复合流和音视频分开的视频流数据） */
	REALDATA_AUDIO,  /**< 实时音频流数据 */
}E_REALDATA_TYPE;

/****************************************************************
** 数据结构名: SnapShotParamInfo
** 功能描述:  抓拍图片参数信息,用于获取和设置
** 作 者:      
** 日 期:      2012-2-9
****************************************************************/
typedef struct tagSnapShotParamInfo
{
	unsigned char	cCmdType;				 //1:MJPEG拍照，2:其他参数设置
	unsigned char	cPhotoNum;               //抓拍张数 1-4
	unsigned char	usOutTouchNum[2];        //usOutTouchNum[0]:外触发延时1   0ms-200ms  usOutTouchNum[1]:外触发延时2   0ms-200ms
	unsigned char	usPhotoSpaceNum;         //拍照间隔      20-100   （*10）ms   该值需要乘10
	unsigned char	cGraspPhotoType;         //1:红灯  2:卡口 3:网络
	unsigned char	cTouchType;				 //1:电平     2:信号沿
	unsigned char   cSignalType;             //1:上升沿   2:下降沿
	unsigned char   cPhotoLampType[4];       //cPhotoLampType[0]:第一张  1:开  2:关  后面的以此类推  
	unsigned char   cLampType;               //1:闪光灯   2:补光灯  
	unsigned char   cShutterModel;           //1:自动 2:手动，默认自动
	unsigned char   cStreamExpTime;          //视频曝光值 0-240 默认240 步长1
	unsigned char   cAGC;                   //自动增益 0-180 默认100 步长1
	unsigned char   cEmpNum;                 //使能延时  0-200   （*10）ms
	unsigned char   cSnapExpTime;       // 抓拍瞬间曝光值 0-240 默认0 步长1
	unsigned char   cFill[2];              //填充字节，凑够4字节
	unsigned short  sDistence[4];            //距离 厘米
	unsigned short  sTriggerDelay;      // 闪光灯延时 0-60000 默认38500 步长600 
	unsigned short  sTriggerWidth;      //脉冲宽度 1-4000 默认1000 步长40
	unsigned char   cReserve[16];              //保留字节，扩展用
}SnapShotParamInfo, *pSnapShotParamInfo;

typedef struct tagSpeedInfo
{
	BYTE ID;
	BYTE CLane;
	BYTE Fill[2];//填充字节
	unsigned short usSpeed[4];
}Packed SpeedInfo;

#define PHOTO_NAME_LEN  64  //图片名称长度

typedef struct tagSnapShotPhotoInfo
{
	SpeedInfo     tSpeedInfo;       //车速信息
	unsigned char cPhotoName[PHOTO_NAME_LEN]; // 图片名称，PHOTO_NAME_LEN必须为4的倍数
	unsigned int  uPhotoDataLen;    //图片大小，字节数
}SnapShotPhotoInfo;

typedef struct tagPhotoData
{
	BYTE              cSaveFlag;      //0:不保存 1:保存，默认值
	BYTE			  cFill[3];		 // 填充
	char *			  cPhotoPath; //图片保存路径 
	SnapShotPhotoInfo tPhotoInfo; //图片信息
	BYTE*			  pPhotoData;              //图片数据指针
}PhotoData; 

/**
* @struct tagTalkParam
* @brief 语音对讲的参数
* @attention
*/
typedef struct tagTalkParam
{
	unsigned int nAudioEncode;    /**< 预留，音频编码类型 */
	unsigned int nSamplesPerSec;  /**< 采样频率，取值为：8000，11025，22050，44100 */
	unsigned int nBitsPerSample;  /**< 预留，采样位数，如：8，16 */
}S_TALK_PARAM;

/**
* @struct tagAlarmerInfo
* @brief 报警源设备信息
* @attention 无
*/
typedef struct tagAlarmerInfo
{   
	char sDeviceIP[128];      /**< 报警源设备的IP地址 */
	unsigned short wLinkPort; /**< 报警源设备的通讯端口 */
}S_ALARMER_INFO;

/**
* @enum tagRealDataType
* @brief 报警类型
* @attention 无
*/
typedef enum tagAlarmType
{
	ALARM_UNKNOWN = 0,/**< 未知类型报警 */
	ALARM_INPUT,      /**< 继电器输入报警 */
	ALARM_MOTION,     /**< 移动侦测报警 */
	ALARM_SHELTER,    /**< 视频遮挡报警 */
	ALARM_VIDEOLOST,  /**< 视频丢失报警 */
	ALARM_DEVICEERR,  /**< 预留，设备异常报警 */
}E_ALARM_TYPE;
/**
* @struct tagAlarmerDeviceInfo
* @brief 报警信息
* @attention 无
*/
typedef struct tagAlarmInfo
{
	E_ALARM_TYPE eAlarmType;    /**< 报警类型 */
	unsigned int nAlarmID;      /**< 报警标识号，设备异常时表示异常类型，从
								*1开始。继电器输入报警：1表示继电器输入号1；
								*移动侦测报警：1表示移动侦测1等。设备异常类
								*型：1-硬盘满，2-硬盘出错，3-网络断开，4-非
								*法访问，5-网络冲突 */ 
	unsigned char cAlarmStatus; /**< 报警状态，0-无报警，1-有报警。继电器
								*输入报警时， 0-报警取消，1-报警触发，2-报
								*警持续 */
	unsigned char cAlarmArea;   /**< 报警区域号，移动侦测和视频遮挡有效，
								*区域号代表哪个区域发生报警 */
}S_ALARM_INFO;

/*justin add 2014.1.14*/
typedef enum tagSerialReqType
{
	SERIAL_SWITCH_TYPE,
	SERIAL_CONNECT_TYPE,
	SERIAL_SET_TYPE,
	SERIAL_SEND_TYPE,
	SERIAL_MAX_TYPE,
}E_REQ_SERIAL_TYPE;

/**
*@enum tagPtzType
*@brief 云台控制类型:  移动+  3D定位
*/
typedef enum tagPtzType
{
	PTZ_MOVE_TYPE,
	PTZ_3D_TYPE,
	END_TYPE,
}E_PTZ_TYPE;

/**
*@enum tagActType
*@brief 3D定位时动作类型
*@从上向下框选为放大，从下向上框选是缩小
*/
typedef enum tagActType
{
	Click = 1,		/**< 不放大*/
	DblClick,			/**< 放大4倍*/
	ZoomIn,			/**< 放大*/
	ZoomOut		/**< 缩小*/
} E_ActionType;

/**
* @enum tagPtzCommand
* @brief 云台控制命令
* @attention 同时描述了NET_IPC_PTZControl接口中2个参数对应的含义和设置，p1表示
*参数iParam1，p2表示参数iParam2
*/
typedef enum tagPtzCommand
{    
	//基本命令
	ZOOM_TELE,      /**< 焦距变大(倍率变大,视野缩小,目标放大),p1速度 */
	ZOOM_WIDE,      /**< 焦距变小(倍率变小,视野放大,目标缩小),p1速度 */
	FOCUS_NEAR,     /**< 焦点前调(目标靠近),p1速度 */
	FOCUS_FAR,      /**< 焦点后调(目标远离),p1速度 */
	IRIS_OPEN,      /**< 光圈扩大,p1速度 */
	IRIS_CLOSE,     /**< 光圈缩小,p1速度 */
	UP,             /**< 上转,p1速度 */
	DOWN,           /**< 下转,p1速度 */
	LEFT,		    /**< 左转,p1速度 */
	RIGHT,		    /**< 右转,p1速度 */
	UP_LEFT,		/**< 左上,p1速度 */
	UP_RIGHT,		/**< 右上,p1速度 */
	DOWN_LEFT,		/**< 左下,p1速度 */
	DOWN_RIGHT,		/**< 右下,p1速度 */

	//预置位操作
	SET_PRESET,     		/**< 设置预置点,p1预置点的序号(1-255) */
	CALL_PRESET,    		/**< 转到预置点,p1预置点的序号  */

	//花样扫描
	START_PATTERN,   	/**< 开始花样扫描,p1花样扫描的序号(1-4) */
	STOP_PATTERN,    	/**< 停止花样扫描,p1花样扫描的序号 */
	RUN_PATTERN,     	/**< 运行花样扫描,p1花样扫描的序号 */

	//自动水平运行
	START_AUTO_PAN, /**< 开始自动水平运行,p1自动水平运行的序号(1-4) */
	STOP_AUTO_PAN,  /**< 停止自动水平运行,p1自动水平运行的序号 */
	RUN_AUTO_PAN,   /**< 运行自动水平运行,p1自动水平运行的序号 */

	AUTO_SCAN,      /**< 自动扫描 */
	FLIP,           /**< 翻转 */
	STOP,           /**< 停止 */
	VECTOR,		/**<矢量控制*/
	PTZ_3D,	/**<3D 定位*/
	CMD_END,
}E_PTZ_COMMAND;

/*justin add 2014.1.14*/
/*justin add 2014.1.14*/
/**
*@ brief - 透明通道请求后返回结果类型值
*/
typedef enum Serialresponsecode_Eum
{
	SerialStatusOK=0,
	SerialReceiveData=1,
	SerialUnsupport,
	SerialDeviceBusy,
	SerialChanelError,
	SerialSendingError,
	SerialBitsError,
	SerialParityError,
	SerialStopBitError,
	SerialBaudrateError,
	SerialFail
}Serialresponsecode_Eum;


#ifdef __cplusplus
extern "C" {
#endif
/******************************************************************************
SDKNET初始化接口
*******************************************************************************/
/**
* 初始化SDK，调用其他SDK函数的前提
* @return 返回如下结果：
* - 成功：true
* - 失败：false
* - 获取错误码调用NET_IPC_GetLastError
* @note 无
*/
BOOL NET_IPC_Init();
/**
* 释放SDK资源，在结束之前最后调用
* @return 返回如下结果：
* - 成功：true
* - 失败：false
* - 获取错误码调用NET_IPC_GetLastError
* @note 无
*/
BOOL NET_IPC_Cleanup();
/******************************************************************************
SDKNET获取错误码接口
*******************************************************************************/
/**
* 获取错误码
* @return 返回值为错误码
* @note 无
*/
long NET_IPC_GetLastError();

/******************************************************************************
SDKNET用户注册接口
*******************************************************************************/
/**
* 用户注册
* @param [IN]   sDevIP    设备IP地址
* @param [IN]   nDevPort  设备端口号
* @param [IN]   sUserName 登录的用户名
* @param [IN]   sPassword 用户密码
* @return 返回如下结果：
* - 失败：-1
* - 其他值：表示返回的用户ID值。该用户ID具有唯一性，后续对设备的操作都需要通过此ID实现
* - 获取错误码调用NET_IPC_GetLastError
* @note 无
*/
long NET_IPC_Login(const char         *sDevIP,
				  const unsigned int nDevPort,
				  const char         *sUserName,
				  const char         *sPassword);
/**
* 用户注销
* @param [IN]   lLoginID 用户ID号，NET_IPC_Login的返回值
* @return 返回如下结果：
* - 成功：true
* - 失败：false
* - 获取错误码调用NET_IPC_GetLastError
* @note 无
*/
BOOL NET_IPC_Logout(long lLoginID);

/******************************************************************************
SDKNET实时流获取接口
*******************************************************************************/

/**
* 开始实时数据获取
* @param [IN]   lLoginID      登陆的ID，NET_IPC_Login的返回值
* @param [IN]   sRealDataInfo 实时数据流的参数结构体
* @param [IN]   fRealData     码流数据回调函数
* @param [IN]   pUserData     用户自定义的数据，回调函数原值返回
* @return 返回如下结果：
* - 失败：-1
* - 其他值：作为NET_IPC_StopRealData等函数的句柄参数
* - 获取错误码调用NET_IPC_GetLastError
* @note 无
*/
typedef void(*CBRealData)(int iStreamID,unsigned char  *pFrameData,int iFrameSize,void *pUserData);


/**
* 停止实时数据获取
* @param [IN]   lRealHandle 登陆的ID，NET_IPC_Login的返回值
* @return 返回如下结果：
* - 成功：true
* - 失败：false
* - 获取错误码调用NET_IPC_GetLastError
* @note 无
*/
BOOL NET_IPC_StopRealData(long lRealHandle,int iStreamID);

/**
* 停止实时数据获取
* @param [IN]   lRealHandle 登陆的ID，NET_IPC_Login的返回值
* @return 返回如下结果：
* - 成功：true
* - 失败：false
* - 获取错误码调用NET_IPC_GetLastError
* @note 无
*/
BOOL NET_IPC_CloseRealDataConnect(long lRealHandle,int iStreamID,int iSockfd);


/******************************************************************************
SDKNET云台控制接口
*******************************************************************************/
/**
* 云台控制接口，不用启动预览时也可以使用
* @param [IN]   lLoginID    登陆的ID，NET_IPC_Login的返回值
* @param [IN]   nChannel    设备通道号， 从0开始
* @param [IN]   ePTZCommand 云台控制命令
* @param [IN]   iParam1     参数1，具体内容跟控制命令有关，详见E_PTZ_COMMAND
* @param [IN]   iParam2     参数2，同上
* @param [IN]   iParam3     参数3，同上
* @param [IN]   iParam4     参数4，同上
* @param [IN]   iParam5     参数5，同上
* @param [IN]   iParam6     参数6，同上
* @param [IN]   iParam7     参数7，同上
* @param [IN]   cRes          保留
* @return 返回如下结果：
* - 成功：true
* - 失败：false
* - 获取错误码调用NET_IPC_GetLastError
* @note 当iParam1表示速度时，范围是1~8
*/
BOOL  NET_IPC_PTZControl(long          lLoginID,
						unsigned int  nChannel,
						E_PTZ_COMMAND ePTZCommand,
						int           iParam1 /*= 0*/,
						int           iParam2/* = 0*/,
						int           iParam3/* = 0*/,
						int           iParam4/* = 0*/,
						int           iParam5/* = 0*/,
						int           iParam6/* = 0*/,
						int           iParam7/* = 0*/,
						char cRes /*= 0*/);


typedef BOOL(*CBTransData)(int result, BYTE *data, int datalen);

/**
* @brief - 透明通道控制接口
* @param[in] lLoginID  		 登陆的ID，NET_IPC_Login的返回值
* @param[in] nChannel		 通道号，默认为1
* @param[in] eSerialReqType透明通道控制命令
* @param[in] iParam1 		 具体内容跟控制命令有关，详见文档说明
* @param[in] iParam2 		 同上
* @param[in] iParam3 		 同上
* @param[in] iParam4 		 同上
* @param[in] data		 通过串口发送的数据
* @param[in] dataLen 		 发送数据的长度
* @param[in] cRes 		 保留
*/
BOOL NET_IPC_TransparantSerialControl(long lLoginID,
	int nChannel,
	E_REQ_SERIAL_TYPE eSerialReqType,
	int iParam1,
	int iParam2,
	int iParam3,
	int iParam4,
	BYTE *data,
	int dataLen,
	CBTransData fTransData,
	char cRes);

/**
 * - 回调函数，用于获取设备报警信息
 * @param[out] alarmInfo 报警信息结构体
 */
typedef BOOL (*fAlarmMsgCallBack)(S_ALARM_INFO *alarmInfo);

/**
 * @brief - NET_IPC_AlarmStartListen
 * - 开始报警监听
 * @param[in] lLoginID 用户ID 号，NET_IPC_Login的返回值
 * @param[in] DataCallback 回调函数，获取设备报警信息
 * @return 返回结果如下:
 * - 成功: true
 * - 失败: false
 * - 获取错误码调用NET_IPC_GetLastError
 */
BOOL NET_IPC_AlarmStartListen(long lLoginID, fAlarmMsgCallBack DataCallback);

/**
 * @brief - NET_IPC_AlarmStopListen
 * - 停止报警监听
 * @return 返回结果如下:
 * - 成功: true
 * - 失败: false
 * - 获取错误码调用NET_IPC_GetLastError
 */ 
BOOL NET_IPC_AlarmStopListen();

/**
 * - 回调函数，用于获取音视频数据
 * @param[out] lRealHandle 实时预览监听句柄，NET_IPC_StartRealPlay 的返回值
 * @param[out] iDataType 音视频数据类型1-视频数据2-音频数据
 * @param[out] pFrameBuf 音视频数据
 * @param[out] iFrameSize 音视频数据长度
 * @param[out] pUser 用户数据，保留
 */
typedef void (*fRealDataCallBack)(long lRealHandle, int iDataType, BYTE cRrameType, BYTE cFrameRate, BYTE *pFrameBuf, unsigned int iFrameSize, unsigned long lTimeStamp, void *pUser);

/**
 * @brief - NET_IPC_StartRealPlay
 * - 开始实时预览
 * @param[in] lLoginID 用户ID 号，NET_IPC_Login的返回值
 * @param[in] sPort 设备端口，默认90
 * @param[in] pRealDataInfo 输入声音数据信息，其中:
 * - iChannel 通道号，默认为0
 * - iAVType 音视频类型，1-视频流2-音频流3-复合流
 * - iEncodeType 音视频编码类型1-G711 3-Mjpeg 4-主码流5-副码流
 * @param[in] fRealData 回调函数，用于获取音视频数据
 * @param[in] pUserData 用户数据
 * @return 返回实时预览监听句柄
 */
long NET_IPC_StartRealPlay(long lLoginID,short sPort,S_REALPLAY_INFO  *pRealDataInfo,fRealDataCallBack fRealData,void *pUserData);

/**
 * @brief - NET_IPC_StopRealPlay
 * - 停止实时预览
 * @param[in] lRealHandle 实时预览监听句柄，NET_IPC_StartRealPlay 的返回值
 * @return 返回结果如下:
 * - 成功: true
 * - 失败: false
 * - 获取错误码调用NET_IPC_GetLastError
 */
BOOL NET_IPC_StopRealPlay(long lRealHandle);

/**
 * @brief - NET_IPC_VoiceFrameInput
 * - 客户端向设备输入声音
 * @param[in] lLoginID 用户ID 号，NET_IPC_Login的返回值
 * @param[in] sPort 发送设备端口，默认90
 * @param[in] pRealDataInfo 输入声音数据信息，其中:
 * - iChannel 通道号，默认为0
 * - iAVType 音视频类型，1-视频流2-音频流3-复合流，此处为2
 * - iEncodeType 音视频编码类型1-G711 3-Mjpeg 4-主码流5-副码流，此处默认为1
 * @param[in] pFrameBuf 编码后的音频帧数据
 * @param[in] iFrameSize 音频帧数据大小，固定长度640
 * @param[in] pUserData 用户数据
 * @return 返回结果如下:
 * - 成功: true
 * - 失败: false
 * - 获取错误码调用NET_IPC_GetLastError
 */
BOOL NET_IPC_VoiceFrameInput(long lLoginID,short sPort,S_REALPLAY_INFO  *pRealDataInfo,BYTE *pFrameBuf, unsigned int iFrameSize,void *pUserData);

/**
 * @brief - NET_IPC_StopVoiceInput
 * - 停止客户端到设备的声音输入
 * @param[in] lLoginID 用户ID 号，NET_IPC_Login的返回值
 * @return 返回结果如下:
 * - 成功: true
 * - 失败: false
 * - 获取错误码调用NET_IPC_GetLastError
 */
BOOL NET_IPC_StopVoiceInput(long lLoginID);

#ifdef __cplusplus
}
#endif //#ifdef __cplusplus

#endif //#ifndef _INF_SDK_NET_H_
