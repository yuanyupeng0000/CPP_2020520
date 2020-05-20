#include "cam_net.h"
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#include "common.h"
#include "Net_param.h"
#include "Net.h"
m_cam_context login_info[CAM_MAX];

int stoped;
INT  LogonNotifyCallback(UINT dwMsgID,UINT ip,UINT port,HANDLE hNotify,void *pPar)
{
	prt(info,"camera Notify CALLBK  id %d ,port %d, %p",dwMsgID,port,hNotify);
	struct in_addr in;
	in.s_addr = ip;
#if 0
	mOperateParam* Run = g_subsys->Run;
	if(Run->g_start) {
		Run->g_start = 0;
		StopCammer(Run);
		Run->g_channel = NULL;
		Run->g_login = NULL;
		Run->g_count = 0;
		Run->g_firstframe=1;
	}
#endif
	return 0;
}

INT  CheckUserPswCallback(const CHAR *pUserName,const CHAR *pPsw)
{
	prt(info,"camera passwd CALLBK");
	return 3;
}
INT	 UpdateFileCallback(INT nOperation,INT hsock,ULONG ip,ULONG port,INT nUpdateType,CHAR *pFileName,CHAR *pFileData,INT nFileLen)
{
	prt(info,"camera file  CALLBK");
	return 0;
}
INT  PreviewStreamCallback(HANDLE hOpenChannel,void *pStreamData,UINT dwClientID,void *pContext,ENCODE_VIDEO_TYPE encodeVideoType)
{
	prt(info,"camera preview CALLBK");
	return 0;
}
INT  StreamWriteCheckCallback(INT nOperation,const CHAR *pUserName,const CHAR *pPsw,ULONG ip,ULONG port,OPEN_VIEWPUSH_INFO viewPushInfo,HANDLE hOpen)
{
	stoped=1;
	prt(info,"camera write back CALLBK opration %d,name %s",nOperation,pUserName);
#if 0
	//struct in_addr in;
	struct in_addr in;
	in.s_addr = ip;
	mOperateParam* Run = g_subsys->Run;
	if(Run->g_start) {
		Run->g_start = 0;
		StopCammer(Run);
		Run->g_channel = NULL;
		Run->g_login = NULL;
		Run->g_count = 0;
		Run->g_firstframe=1;
	}
#endif
	return 0;
}
INT  ServerMsgCmdCallback(ULONG ip,ULONG port,CHAR *pMsgHd)
{
	prt(info,"camera CALLBK");
	return 0;
}
int NetSDKInit(void)
{
	ERR_CODE errCode = ERR_FAILURE;
	errCode = NET_Startup(COMMONPORT, LogonNotifyCallback, CheckUserPswCallback,
			UpdateFileCallback, ServerMsgCmdCallback, StreamWriteCheckCallback,
			(ChannelStreamCallback) PreviewStreamCallback);
	return errCode;
}

HANDLE cam_net_login(char *IpAddr, char*UserName, char * Passwd)
{
	ERR_CODE errCode = ERR_FAILURE;
	HANDLE m_hLogonServer =(HANDLE) NULL;
	CHAR tmp[] = "admin";
	//errCode = NET_LogonServer(g_pIPString, g_lPort, "admin", g_pUseName, g_pPASSWD, 0, &m_hLogonServer);
	errCode = NET_LogonServer(IpAddr, COMMONPORT, tmp, UserName, Passwd, 0,
			&m_hLogonServer);
	if (errCode)
		m_hLogonServer = (HANDLE)NULL;
	return m_hLogonServer;
}

ERR_CODE cam_net_logoff(HANDLE h)
{
	ERR_CODE c;
//	prt(info,"###################log off server %x",h);
	if(h!=NULL)
	c=NET_LogoffServer(h);
	return c;
}
ERR_CODE cam_net_close_channel(HANDLE handle);
ERR_CODE logoff_camera(m_cam_context *p_ctx)
{
 	ERR_CODE c;
 	if(p_ctx->channel_handle!=NULL)
 	cam_net_close_channel(p_ctx->channel_handle);

	return 	cam_net_logoff(p_ctx->server_hangle);;
}
HANDLE cam_net_open_channel(char *IpAddr, char*UserName, char *Pwd, int port ,void*context, ChannelStreamCallback CallBack)
{
	HANDLE m_hOpenChannel = (HANDLE)NULL;
	ERR_CODE errCode = ERR_FAILURE;
	OPEN_CHANNEL_INFO_EX channelInfo;
	channelInfo.dwClientID = 1;
	channelInfo.nOpenChannel = 0;
	channelInfo.nSubChannel = 0x1; //sub stream
	//channelInfo.nSubChannel = 0; //main stream
	channelInfo.protocolType = (NET_PROTOCOL_TYPE) 0;
	channelInfo.funcStreamCallback = (ChannelStreamCallback) CallBack;
	channelInfo.pCallbackContext = context;
	CHAR tmp[] = "admin";
		printf("---------> %s \n",IpAddr);
    errCode = NET_OpenChannel(IpAddr, port, tmp, UserName, Pwd,(OPEN_CHANNEL_INFO_EX*) &channelInfo, &m_hOpenChannel);
	if (errCode)
		m_hOpenChannel = (HANDLE)NULL;
else{
		printf("open chnel ok\n");
}
	return m_hOpenChannel;
}

HANDLE login_camera(char *ip, char*username, char *passwd, int port , void*context, ChannelStreamCallback CallBack)
{
  	m_cam_context *p_ctx=(m_cam_context *)context;
    p_ctx->channel_handle = NULL;
	ERR_CODE errCode = ERR_FAILURE;
	HANDLE m_hLogonServer = (HANDLE) NULL;
//	prt(err,"login cam => : %s , %s  ,%s  %d ",ip,username,passwd,port);
	errCode = NET_LogonServer(ip, port, "admin", username, passwd, 0,
			&m_hLogonServer);
	p_ctx->server_hangle=m_hLogonServer;
//	prt(info,"###########log on server %x",m_hLogonServer);
	if (errCode) {
		m_hLogonServer = (HANDLE) NULL;
		prt(err,"login cam err : %s , %s  ,%s ,code %d ",ip,username,passwd,errCode);
		return m_hLogonServer;
	}else{
		printf("log in server ok\n");
	}
	 p_ctx->channel_handle=cam_net_open_channel(ip,username,passwd,port ,context,CallBack);
	return p_ctx->channel_handle;
}
ERR_CODE cam_net_close_channel(HANDLE handle)
{
	ERR_CODE errCode = NET_CloseChannel(handle);;
	return errCode;
}


HANDLE NetSDKLogin(char *IpAddr, char*UserName, char * Passwd,int port)
{
	ERR_CODE errCode = ERR_FAILURE;
	HANDLE m_hLogonServer =(HANDLE) NULL;
	CHAR tmp[] = "admin";
	//errCode = NET_LogonServer(g_pIPString, g_lPort, "admin", g_pUseName, g_pPASSWD, 0, &m_hLogonServer);
	errCode = NET_LogonServer(IpAddr, port, tmp, UserName, Passwd, 0,
			&m_hLogonServer);
	if (errCode)
		m_hLogonServer = (HANDLE)NULL;

	return m_hLogonServer;
}

HANDLE NetSDKOpenChannel(char *IpAddr, char*UserName, char *Pwd, int port ,void*context, ChannelStreamCallback CallBack)
{
	HANDLE m_hOpenChannel = (HANDLE)NULL;
	ERR_CODE errCode = ERR_FAILURE;
	OPEN_CHANNEL_INFO_EX channelInfo;
	channelInfo.dwClientID = 0;
	channelInfo.nOpenChannel = 0;
	channelInfo.nSubChannel = 0x1;
	channelInfo.protocolType = (NET_PROTOCOL_TYPE) 0;
	channelInfo.funcStreamCallback = (ChannelStreamCallback) CallBack;
	channelInfo.pCallbackContext = context;
	CHAR tmp[] = "admin";
    errCode = NET_OpenChannel(IpAddr, port, tmp, UserName, Pwd,(OPEN_CHANNEL_INFO_EX*) &channelInfo, &m_hOpenChannel);
	if (errCode)
		m_hOpenChannel = (HANDLE)NULL;
	return m_hOpenChannel;
}


//HANDLE NetSDKLogin(char *IpAddr, char*UserName, char * Passwd)
//{
//	ERR_CODE errCode = ERR_FAILURE;
//	HANDLE m_hLogonServer = NULL;
//	//errCode = NET_LogonServer(g_pIPString, g_lPort, "admin", g_pUseName, g_pPASSWD, 0, &m_hLogonServer);
//	errCode = NET_LogonServer(IpAddr, COMMONPORT, "admin", UserName, Passwd, 0, &m_hLogonServer);
//	if(errCode)
//		m_hLogonServer=NULL;
//	printf("--->|\tNET_LogonServer[%s]: %s\n",IpAddr, errCode?"Failed!":"Successful!");
//	return m_hLogonServer;
////	NET_LogoffServer();
//}






int SetNtpServer(HANDLE m_hLogonServer,  char* ntpserver)
{
	ERR_CODE		errCode;
	unsigned int	nSize;
	NTP_CONFIG	    ntpconfig;

	memset((char*)&ntpconfig, 0, sizeof(NTP_CONFIG));
	nSize = sizeof(NTP_CONFIG);
	unsigned int nID = 0;
	errCode	= NET_GetServerConfig(m_hLogonServer, CMD_GET_NTP, (char*)&ntpconfig, &nSize, &nID);
	if (errCode != ERR_SUCCESS){
		prt(err,"ntp");
		return 0;
	}

	prt(err,"ntp");
	if(ntpserver != NULL){
		if(!strcmp("0.0.0.0", ntpserver)
			||(1 == ntpconfig.ntpOpen && !strcmp(ntpconfig.ntpHost, ntpserver)))
			return 0;
		ntpconfig.ntpOpen = 0x1;
		strcpy(ntpconfig.ntpHost, ntpserver);
	}
	errCode = NET_SetServerConfig(m_hLogonServer, CMD_SET_NTP, (char *)&ntpconfig, nSize, 0);
	if (errCode != ERR_SUCCESS){
		prt(err,"ntp");
		return 0;
	}
	return 1;
}
//int ZenithStreamCallBack(HANDLE hOpenChannel,void *pStreamData,UINT dwClientID,void *pContext,ENCODE_VIDEO_TYPE encodeVideoType,ULONG frameno)
//{
//	if (FRAME_FLAG_A == ((FRAME_HEAD *) pStreamData)->streamFlag)
//		return -1;
//	m_cam_context *p=(m_cam_context *)pContext;
//	p->h264_buf=(unsigned char*) pStreamData + (sizeof(EXT_FRAME_HEAD))+ sizeof(FRAME_HEAD);
//	p->h264_buf_len=((FRAME_HEAD *) pStreamData)->nByteNum;
//	p->callback_fun(p);
//	return 0;
//}
int running_flag=0;
int ZenithStreamCallBack(HANDLE hOpenChannel,void *pStreamData,UINT dwClientID,void *pContext,ENCODE_VIDEO_TYPE encodeVideoType,ULONG frameno)
{
	stoped=0;
	running_flag=1;
	m_cam_context *p_ctx=(m_cam_context *) pContext;
//	prt(info,"ip%s",p_ctx->ip);
//	prt(info,"1");
	p_ctx->frame_data=( char*) pStreamData + (sizeof(EXT_FRAME_HEAD))+ sizeof(FRAME_HEAD);
	p_ctx->frame_size=((FRAME_HEAD *) pStreamData)->nByteNum;
    p_ctx->callback_fun(p_ctx->index,p_ctx->frame_data,p_ctx->frame_size); //2019.04.13 by roger ���֡������������
	 //prt(info,"in call back function");
	if (FRAME_FLAG_A == ((FRAME_HEAD *) pStreamData)->streamFlag)
		return -1;

	return 0;
}
#include "IPCNetSDK.h"
m_cam_context *g_p;
int HandleFrameCallBack(int iStreamID,unsigned char  *pFrameData,int iFrameSize)
{
	//printf("##################");
	prt(info,"id %d",iStreamID);
#if 1
	if(iStreamID == VIDEO_MJPEG)
		printf("\n MJPEG FrameSize : %d \n",iFrameSize);
	else if(iStreamID == VIDEO_H264_MAJOR)
		printf("\n H264 Major FrameSize : %d \n",iFrameSize);
	else if(iStreamID == VIDEO_H264_MINOR)
		printf("\n H264 Minor FrameSize : %d \n",iFrameSize);
#endif
	#if 0
	static int count = 0;
	count ++;
	if(count == 10)
	{
		FILE *stream;
		if((stream = fopen("test.jpg","wb")) == NULL)
		{
			printf("fopen test.jpg error !\n");
			return -1;
		}

		fwrite(pFrameData, iFrameSize, 1, stream);
		fclose(stream);
	}
	#endif



//	g_p->ret_data.h264_pkt.data=(unsigned char*) pFrameData;
//	g_p->ret_data.h264_pkt.size=iFrameSize;

	//g_p->callback_fun(&g_p->ret_data);
	return 0;
}
//long lLoginID;


#define AUDIO_REQUEST 	4
#define VIDEO_REQUEST 	6
#define TIME 60*15
int loops=TIME;
void RealPlayHandleFrameGetNew(long lRealHandle, int iDataType, BYTE cFrameType, BYTE cFrameRate, BYTE *pFrameBuf, unsigned int iFrameSize, unsigned long lTimeStamp, void *pUser)
{
  //  prt(info,"calling back ");
	if(!loops--){
		prt(info,"check loop");
		loops=TIME;
	}
	m_cam_context *p=(m_cam_context  *)pUser;
//	prt(info,"get index %d",p->cam_id);
	if(iDataType == AUDIO_REQUEST)
	{
		//printf("lRealHandle: %d get audio data length: %d frameType: %d\n",lRealHandle,iFrameSize,cFrameType);
		//fwrite(pFrameBuf,iFrameSize,1,pfile);
	}
	else if(iDataType == VIDEO_REQUEST)
	{
	//	printf("lRealHandle: %d get video data length: %d frameType: %d frameRate: %d timeStamp: %ld\n",lRealHandle,iFrameSize,cFrameType,cFrameRate,lTimeStamp);
	}
//	p->ret_data.cam_id=p->cam_id;
//	p->ret_data.h264_pkt.data=(unsigned char*) pFrameBuf;
//	p->ret_data.h264_pkt.size=iFrameSize;

	//p->callback_fun(&p->ret_data);
}

//#define HANHUI
//#undef HANHUI


int net_open_camera(int index,char ip[],int port,char name[],char passwd[],void *func,void *pri)
{
   prt(info, "%s %s %s %d\n",ip,name ,passwd,port);
    if (strlen(ip) < 1 || port < 0) {
        prt(info, "ip or port error");
        return -1;
    }

	m_cam_context *p;
	m_cam_context *ctx=&login_info[index];
	ctx->index=index;
	p=ctx;
	ctx->pri=pri;
	ctx->callback_fun=(THREAD_ENTITY1)func;
	ctx->port=port;
	memset(ctx->username,0,NAME_LEN_MAX);
	memset(ctx->ip,0,NAME_LEN_MAX);
	memset(ctx->passwd,0,NAME_LEN_MAX);

	memcpy(ctx->ip,ip,strlen(ip));
	memcpy(ctx->username,name,strlen(name));
	memcpy(ctx->passwd,passwd,strlen(passwd));
#ifdef HANHUI

	g_p=p;
//	const char *pszName = "admin";
//	const char *pszPassword = "admin";
//	char pDevIP[] = "192.168.1.18";
//	int iDevPort = 90;

	char * pszName=p->username;
	char * pszPassword=p->passwd;
	char * pDevIP=p->ip;
	int iDevPort=p->port;

//	IPCNET_Init();
	//	NET_IPC_Init();
	printf(" NET_IPC_Login with %s %d, %s %s\n ",pDevIP, iDevPort, pszName, pszPassword); //閿熸枻鎷烽敓鐭尅鎷烽敓鍙帴鍖℃嫹鍓嶉敓鏂ゆ嫹閿熸枻鎷烽敓閾帮�?
	p->id = NET_IPC_Login(pDevIP, iDevPort, pszName, pszPassword); //閿熸枻鎷烽敓鐭尅鎷烽敓鍙帴鍖℃嫹鍓嶉敓鏂ゆ嫹閿熸枻鎷烽敓閾帮�?
	//lLoginID = IPCNET_Login(pDevIP, iDevPort, pszName, pszPassword); //閿熸枻鎷烽敓鐭尅鎷烽敓鍙帴鍖℃嫹鍓嶉敓鏂ゆ嫹閿熸枻鎷烽敓閾帮�?

//	S_REALDATA_INFO pRealDataInfo;
//	S_REALDATA_INFO pRealDataInfoTmp;
	S_REALPLAY_INFO pRealDataInfoTmp;
//	pRealDataInfo.lChannel = 0; /**< 閫氶敓鏂ゆ嫹閿熻剼锝忔嫹閿熸枻鎷�?0閿熸枻鎷峰 */
//	pRealDataInfo.lStreamMode = 2; /**< 閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓閰碉綇鎷�0-閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹1-閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷� ,2-MJPEG*/
//	pRealDataInfo.eEncodeType = VIDEO_MJPEG; //VIDEO_H264_MAJOR; /**< 閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹 */

//	IPCNET_StartRealData(lLoginID, &pRealDataInfo, NULL);

//	pRealDataInfoTmp.lChannel = 0; /**< 閫氶敓鏂ゆ嫹閿熻剼锝忔嫹閿熸枻鎷�?0閿熸枻鎷峰 */
//	pRealDataInfoTmp.lStreamMode = 1; /**< 閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓閰碉綇鎷�0-閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹1-閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷� ,2-MJPEG*/
//	pRealDataInfoTmp.eEncodeType = VIDEO_H264_MINOR; /**< 閿熸枻鎷烽敓鏂ゆ嫹閿熸枻鎷烽敓鏂ゆ嫹 */
	S_REALPLAY_INFO sRealPlayInfo;
	sRealPlayInfo.iChannel = 0;
	sRealPlayInfo.iAVType = DATA_VIDEO;
	sRealPlayInfo.iEncodeType = VIDEO_H264_MINOR;
	//IPCNET_StartRealData(lLoginID, &pRealDataInfoTmp, NULL);

	long lRealHandleMain = NET_IPC_StartRealPlay(p->id,90,&sRealPlayInfo,(fRealDataCallBack)RealPlayHandleFrameGetNew, (void *)p);
//	prt(info,"id %ld",lLoginID);
	return 0;

#else
//	HANDLE login_camera(char *ip, char*username, char *passwd, int port , void*context, ChannelStreamCallback CallBack)
//	p->ret_data.cam_id=p->cam_id;

	login_camera(p->ip,p->username,p->passwd,p->port,p,	(ChannelStreamCallback) ZenithStreamCallBack);
//	p->timed_func_handle=regist_timed_callback((THREAD_ENTITY)p->callback_fun,1000000);

	if (p->channel_handle == NULL) {
		prt(info, "login fail");
		return -1;
	} else {

		return 0;
	}

#endif
}

void net_close_camera(int index)
{
	m_cam_context *p=&login_info[index];
	prt(info,"closing cam,freeing %p",p);
#ifdef HANHUI
//	IPCNET_Logout(lLoginID);
//	IPCNET_Cleanup();
	prt(info,"closing %ld",p->id);
	NET_IPC_StopRealPlay(p->id);
	NET_IPC_Logout(p->id);
//	NET_IPC_Cleanup();
	prt(info,"id %ld",p->id);

#else
	logoff_camera(p);
//	unregist_timed_callback(p->timed_func_handle);
#endif

	//free(p);
}

int open_sdk(void)
{
#ifdef HANHUI
// 	IPCNET_Init();
 	NET_IPC_Init();
#else
	ERR_CODE errCode = ERR_FAILURE;
	errCode = NET_Startup(5000, LogonNotifyCallback, CheckUserPswCallback,
			UpdateFileCallback, ServerMsgCmdCallback, StreamWriteCheckCallback,
			(ChannelStreamCallback) PreviewStreamCallback);
	return errCode;
#endif
}

void close_sdk(void)
{
#ifdef HANHUI
	NET_IPC_Cleanup();
#else
	NET_Cleanup();
#endif
}

m_cam_context * prepare_camera(char ip[],int port,char name[],char passwd[],void *func,void *pri)
{
	m_cam_context *ctx=(m_cam_context *)malloc(sizeof(m_cam_context));
	ctx->pri=pri;
	ctx->callback_fun=(THREAD_ENTITY1)func;
	ctx->port=port;
	memset(ctx->username,0,NAME_LEN_MAX);
	memset(ctx->ip,0,NAME_LEN_MAX);
	memset(ctx->passwd,0,NAME_LEN_MAX);

	memcpy(ctx->ip,ip,strlen(ip));
	memcpy(ctx->username,name,strlen(name));
	memcpy(ctx->passwd,passwd,strlen(passwd));
//	prt(info,"prepare ip %s,name  %s passwd %s",ctx->ip,ctx->username,ctx->passwd);
	return ctx;
}
int cam_set_shutter(int index,int value)
{
	HANDLE m_hLogonServer=login_info[index].server_hangle;
	prt(info,"set shutter ");
	ERR_CODE		errCode;
	unsigned int	nSize;
	VIDEO_IN_SENSOR_S	addr_config;
	memset((char*)&addr_config, 0, sizeof(VIDEO_IN_SENSOR_S));

	nSize = sizeof(VIDEO_IN_SENSOR_S);
	unsigned int nID = 0;
	errCode	= NET_GetServerConfig(m_hLogonServer, CMD_GET_VI_SENSOR, (char*)&addr_config, &nSize, &nID);
	if (errCode != ERR_SUCCESS){
		prt(camera_msg,"get config err");
		return 0;
	}

	switch(value){
		case 1:
			addr_config.wExpTimeMax = 25;break;
		case 2:
			addr_config.wExpTimeMax = 50;break;
		case 3:
			addr_config.wExpTimeMax = 100;break;
		case 4:
			addr_config.wExpTimeMax = 150;break;
		case 5:
			addr_config.wExpTimeMax = 200;break;
		case 6:
			addr_config.wExpTimeMax = 250;break;
		case 7:
			addr_config.wExpTimeMax = 300;break;
		case 8:
			addr_config.wExpTimeMax = 400;break;
		case 9:
			addr_config.wExpTimeMax = 500;break;
		case 10:
			addr_config.wExpTimeMax = 1000;break;
		case 11:
			addr_config.wExpTimeMax = 2000;break;
		case 12:
			addr_config.wExpTimeMax = 4000;break;
		case 13:
			addr_config.wExpTimeMax = 6000;break;
		case 14:
			addr_config.wExpTimeMax = 8000;break;
		default:
			addr_config.wExpTimeMax = 25;break;
	}
	errCode = NET_SetServerConfig(m_hLogonServer, CMD_SET_VI_SENSOR, (char *)&addr_config, nSize, 0);
	if (errCode != ERR_SUCCESS){
		prt(err,"set shutter fail");
		return 0;
	}

	return 1;
}
