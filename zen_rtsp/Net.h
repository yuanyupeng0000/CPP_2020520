/******************************************************************************
* 绯荤粺鍚嶇О锛歂et
* 鏂囦欢鍚嶇О锛歂et.h
* 鐗堟湰    锛�	V2.0.0.0.2
* 璇存槑	  锛氱綉缁滃紑鍙戝寘鎺ュ彛,鏈琒DK涓哄拰缂栫爜鍣ㄤ氦浜掓彁渚涙帴鍙�
			璇ユ枃浠跺寘鍚叚閮ㄥ垎锛�
			涓�銆佹灇涓剧被鍨嬪畾涔夛紱
			浜屻�佸洖璋冨嚱鏁帮紱
			涓夈�佹帴鍙ｇ粨鏋勭被鍨嬪畾涔夛紱
			鍥涖�侀煶瑙嗛缃戠粶甯уご锛�
			浜斻�佸嚱鏁版帴鍙ｅ畾涔夛紱
* 鍏朵粬璇存槑: 鏃�
******************************************************************************/
#ifndef  __NET_H__
#define  __NET_H__
#include "Net_param.h"
#define PACKED  __attribute__((packed, aligned(1)))
#define PACKED4 __attribute__((packed, aligned(4)))


/*********************************  涓�銆佹灇涓剧被鍨嬪畾涔�  ******************************/

//1锛屾暟瀛楄棰戞湇鍔″櫒鏈哄櫒绫诲瀷
typedef enum	DVS_MACHINE_TYPE_
{
	NONE = 0xff,
}DVS_MACHINE_TYPE;

//2锛岃棰戠紪鐮佹牸寮�									
typedef enum  _ENCODE_VIDEO_TYPE
{
	EV_TYPE_NONE		= 0xFFFF,
}ENCODE_VIDEO_TYPE;

//3锛屽崌绾х被鍨�
typedef enum _UPDATE_TYPE
{
	UPDATE_KERNEL,					//鍗囩骇鍐呮牳
	UPDATE_YUNTAI1,				//鍗囩骇浜戝彴鍗忚1
	UPDATE_YUNTAI2,				//鍗囩骇浜戝彴鍗忚2
	UPDATE_YUNTAI3,				//鍗囩骇浜戝彴鍗忚3
	UPDATE_YUNTAI4,				//鍗囩骇浜戝彴鍗忚4
	UPDATE_YUNTAI5,				//鍗囩骇浜戝彴鍗忚5
	UPDATE_YUNTAI6,				//鍗囩骇浜戝彴鍗忚6
	UPDATE_OCX,						//鍗囩骇鎺т欢
	UPDATE_WEBPAGE,				//鍗囩骇椤甸潰
	UPDATE_PATHFILE,				//鍗囩骇鐗瑰畾鐩綍鏂囦欢
}UPDATE_TYPE;


//5锛岃繛鎺ョ姸鎬�
typedef enum _CONNECT_STATUS
{
	CONNECT_STATUS_NONE,			//鏈繛鎺�
	CONNECT_STATUS_OK,				//宸茬粡杩炴帴
}CONNECT_STATUS;

//6锛岃浆鍙戠被鍨�
typedef enum _RELAY_TYPE
{
	RELAY_LOGON,					//杞彂鐧诲綍璁剧疆
	RELAY_PREVIEW,					//杞彂瑙嗛棰勮鏁版嵁
	RELAY_TALK,						//杞彂瀵硅
	RELAY_AUDIOBRD,				//杞彂璇煶骞挎挱
	RELAY_QUERY						//杞彂鐨勬煡璇�
}RELAY_TYPE;

//7锛岄�氱煡搴旂敤绋嬪簭绐楀彛娑堟伅鍛戒护
typedef enum _MSG_NOTIFY
{

    MSG_CONNECT_CLOSE,             //鐧诲綍杩炴帴鍏抽棴 
    MSG_CHANNEL_CLOSE,             //閫氶亾杩炴帴鍏抽棴 
    MSG_TALK_CLOSE,                //瀵硅杩炴帴鍏抽棴 
    MSG_ALARM_OUTPUT,              //鎶ヨ杈撳嚭                   
    MSG_UPDATE_SEND_PERCENT,       //鍗囩骇绋嬪簭鍙戦�佺櫨鍒嗘瘮 
    MSG_UPDATE_SAVE_PERCENT,       //鍗囩骇鍐欏叆鍙戦�佺櫨鍒嗘瘮
    MSG_VIEWPUSH_CLOSE ,           //瑙ｇ爜鍣ㄨ棰戣緭鍏ュ叧闂� 
   	MSG_BROADCAST_ADD_FAILURE,     //鍔犲叆璇煶骞挎挱缁勫け璐� 
   	MSG_BROADCAST_CLOSE,           //璇煶骞挎挱涓竴涓柇寮� 
   	MSG_SENSOR_CAPTURE,            //鎺㈠ご瑙﹀彂鐨勬姄鎷� 
   	MSG_COM_DATA,                  //涓插彛閲囬泦鏁版嵁 
   	MSG_ALARM_LOST,                //鎶ヨ娑堝け                   
    MSG_ALARM_OUTPUT_NEW,          //鎶ヨ杈撳嚭(鏂�)               
    MSG_ALARM_LOST_NEW,            //鎶ヨ娑堝け(鏂�)               
    MSG_PICCHN_CLOSE,              //鎶撴媿閫氶亾杩炴帴鍏抽棴 
}MSG_NOTIFY;


//8锛岄敊璇爜
typedef enum _ERR_CODE
{
	ERR_SUCCESS,					//鎿嶄綔鎴愬姛
	ERR_FAILURE,					//鎿嶄綔澶辫触
	ERR_REFUSE_REQ,				//璇锋眰琚嫆缁�
	ERR_USER_FULL,					//鐧诲綍鐢ㄦ埛宸叉弧
	ERR_PREVIEW_FULL,				//棰勮鐢ㄦ埛宸叉弧
	ERR_TASK_FULL,					//绯荤粺浠诲姟绻佸繖锛屽緟浼氬皾璇曡繛鎺�
	ERR_CHANNEL_NOT_EXIST,			//瑕佹墦寮�鐨勯�氶亾涓嶅瓨鍦ㄦ垨宸叉弧
	ERR_DEVICE_NAME,				//鎵撳紑鐨勮澶囦笉瀛樺湪
	ERR_IS_TALKING,				//姝ｅ湪瀵硅
	ERR_QUEUE_FAILUE,				//闃熷垪鍑洪敊
	ERR_USER_PASSWORD,				//鐢ㄦ埛鍚嶆垨瀵嗙爜鍜岀郴缁熶笉鍖归厤
	ERR_SHARE_SOCKET,				//socket 閿欒
	ERR_RELAY_NOT_OPEN,			//杞彂璇锋眰鐨勬湇鍔¤繕鏈墦寮�
	ERR_RELAY_MULTI_PORT,			//杞彂澶氭挱绔彛閿欒
	ERR_VIEWPUSH_CHANNEL_USING,	//瑙嗛杈撳叆鐨勯�氶亾宸茬粡琚崰鐢�
	ERR_VIEWPUSH_DECODE_TYPE,		//瑙嗛杈撳叆閫氶亾鐨勮В鐮佹牸寮忛敊璇紝0閫氶亾(4cif,2cif,cif),1閫氶亾(2cif,cif),2閫氶亾(cif),3閫氶亾(cif)
	ERR_AUTO_LINK_FAILURE,			//杞彂鐨勮嚜鍔ㄨ繛鎺ュけ璐�
	ERR_NOT_LOGON,
	ERR_IS_SETTING,
	ERR_COMMAND_FAILURE,
	
	ERR_INVALID_PARAMETER=100,		//杈撳叆鍙傛暟鏃犳晥
	ERR_LOGON_FAILURE,				//鐧诲綍澶辫触
	ERR_TIME_OUT,					//鎿嶄綔瓒呮椂
	ERR_SOCKET_ERR,				//SOCKET閿欒
	ERR_NOT_LINKSERVER,			//杩樻湭杩炴帴鏈嶅姟鍣�
	ERR_BUFFER_EXTCEED_LIMIT,		//浣跨敤缂撳啿瓒呰繃闄愬埗	
	ERR_LOW_PRIORITY,				//鎿嶄綔鏉冮檺涓嶈冻
	ERR_BUFFER_SMALL,				//缂撳啿澶皬
	ERR_IS_BUSY,					//绯荤粺浠诲姟姝ｅ繖
	ERR_UPDATE_FILE,				//鍗囩骇鏂囦欢閿欒
	ERR_UPDATE_UNMATCH,			//鍗囩骇鏂囦欢鍜屾満鍣ㄤ笉鍖归厤
	ERR_PORT_INUSE,				//绔彛琚崰鐢�
	ERR_RELAY_DEVICE_EXIST,		//璁惧鍚嶅凡缁忓瓨鍦�
	ERR_CONNECT_REFUSED,			//杩炴帴鏃惰鎷掔粷
	ERR_PROT_NOT_SURPPORT,			//涓嶆敮鎸佽鍗忚

	ERR_FILE_OPEN_ERR,            //鎵撳紑鏂囦欢澶辫触
	ERR_FILE_SEEK_ERR,            //fseek澶辫触 
	ERR_FILE_WRITE_ERR,           //fwrite澶辫触 
	ERR_FILE_READ_ERR,            //fread澶辫触 
	ERR_FILE_CLOSING,             //姝ｅ湪鍏抽棴鏂囦欢 
	
}ERR_CODE;

//9锛屽弬鏁版搷浣滃懡浠�
typedef enum _CMD_NET
{
	//缂栫爜鍣ㄥ懡浠�
	CMD_GET_ALL_PARAMETER,			//寰楀埌鎵�鏈夌紪鐮佸櫒鍙傛暟
	CMD_SET_DEFAULT_PARAMETER,	//鎭㈠鎵�鏈夌紪鐮佸櫒榛樿鍙傛暟
	CMD_SET_RESTART_DVS,			//閲嶅惎缂栫爜鍣�
	CMD_GET_SYS_CONFIG,			//鑾峰彇绯荤粺璁剧疆
	CMD_SET_SYS_CONFIG,			//璁剧疆绯荤粺璁剧疆
	CMD_GET_TIME,					//鑾峰彇缂栫爜鍣ㄦ椂闂�
	CMD_SET_TIME,					//璁剧疆缂栫爜鍣ㄦ椂闂�
	CMD_GET_AUDIO_CONFIG,			//鑾峰彇闊抽璁剧疆
	CMD_SET_AUDIO_CONFIG,			//璁剧疆闊抽璁剧疆
	CMD_GET_VIDEO_CONFIG,			//鑾峰彇瑙嗛璁剧疆
	CMD_SET_VIDEO_CONFIG,			//璁剧疆瑙嗛璁剧疆
	CMD_GET_VMOTION_CONFIG,		//鑾峰彇绉诲姩渚︽祴璁剧疆
	CMD_SET_VMOTION_CONFIG,		//璁剧疆绉诲姩渚︽祴璁剧疆
	CMD_GET_VMASK_CONFIG,			//鑾峰彇鍥惧儚灞忚斀璁剧疆
	CMD_SET_VMASK_CONFIG,			//璁剧疆鍥惧儚灞忚斀璁剧疆
	CMD_GET_VLOST_CONFIG,			//鑾峰彇瑙嗛涓㈠け璁剧疆
	CMD_SET_VLOST_CONFIG,			//璁剧疆瑙嗛涓㈠け璁剧疆
	CMD_GET_SENSOR_ALARM,			//鑾峰彇鎺㈠ご鎶ヨ渚︽祴璁剧疆
	CMD_SET_SENSOR_ALARM,			//璁剧疆鎺㈠ご鎶ヨ渚︽祴璁剧疆
	CMD_GET_USER_CONFIG,			//鑾峰彇鐢ㄦ埛璁剧疆
	CMD_SET_USER_CONFIG,			//璁剧疆鐢ㄦ埛璁剧疆
	CMD_GET_NET_CONFIG,			//鑾峰彇缃戠粶璁剧疆缁撴瀯
	CMD_SET_NET_CONFIG,			//璁剧疆缃戠粶璁剧疆缁撴瀯
	CMD_GET_COM_CONFIG,			//鑾峰彇涓插彛璁剧疆
	CMD_SET_COM_CONFIG,			//璁剧疆涓插彛璁剧疆
	CMD_GET_YUNTAI_CONFIG,			//鑾峰彇鍐呯疆浜戝彴淇℃伅
	CMD_SET_YUNTAI_CONFIG,			//璁剧疆鍐呯疆浜戝彴淇℃伅
	CMD_GET_VIDEO_SIGNAL_CONFIG,	//鑾峰彇瑙嗛淇″彿鍙傛暟锛堜寒搴︺�佽壊搴︺�佸姣斿害銆侀ケ鍜屽害锛�
	CMD_SET_VIDEO_SIGNAL_CONFIG,	//璁剧疆瑙嗛淇″彿鍙傛暟锛堜寒搴︺�佽壊搴︺�佸姣斿害銆侀ケ鍜屽害锛�
	CMD_SET_PAN_CTRL,				//浜戝彴鎺у埗
	CMD_SET_COMM_SENDDATA,			//閫忔槑鏁版嵁浼犺緭
	CMD_SET_COMM_START_GETDATA,	//寮�濮嬮噰闆嗛�忔槑鏁版嵁
	CMD_SET_COMM_STOP_GETDATA,	//鍋滄閲囬泦閫忔槑鏁版嵁
	CMD_SET_OUTPUT_CTRL,			//缁х數鍣ㄦ帶鍒�
	CMD_SET_PRINT_DEBUG,			//璋冭瘯淇℃伅寮�鍏�
	CMD_SET_ALARM_CLEAR,			//娓呴櫎鎶ヨ
	CMD_GET_ALARM_INFO,			//鑾峰彇鎶ヨ鐘舵�佸拰缁х數鍣ㄧ姸鎬�
	CMD_SET_TW2824,				//璁剧疆澶氱敾闈㈣姱鐗囧弬鏁�(淇濈暀)
	CMD_SET_SAVE_PARAM,			//璁剧疆淇濆瓨鍙傛暟
	CMD_GET_USERINFO,				//鑾峰彇褰撳墠鐧婚檰鐨勭敤鎴蜂俊鎭�
	CMD_GET_DDNS,					//鑾峰彇DDNS
	CMD_SET_DDNS,					//璁剧疆DDNS
	CMD_GET_CAPTURE_PIC,			//鍓嶇鎶撴媿
	CMD_GET_SENSOR_CAP,			//鑾峰彇瑙﹀彂鎶撴媿璁剧疆
	CMD_SET_SENSOR_CAP,			//璁剧疆瑙﹀彂鎶撴媿璁剧疆
	CMD_GET_EXTINFO,				//鑾峰彇鎵╁睍閰嶇疆
	CMD_SET_EXTINFO,				//璁剧疆鎵╁睍閰嶇疆
	CMD_GET_USERDATA,				//鑾峰彇鐢ㄦ埛閰嶇疆
	CMD_SET_USERDATA,				//璁剧疆鐢ㄦ埛閰嶇疆
	CMD_GET_NTP,					//鑾峰彇NTP閰嶇疆
	CMD_SET_NTP,					//璁剧疆NTP閰嶇疆
	CMD_GET_UPNP,					//鑾峰彇UPNP閰嶇疆
	CMD_SET_UPNP,					//璁剧疆UPNP閰嶇疆
	CMD_GET_MAIL,					//鑾峰彇MAIL閰嶇疆
	CMD_SET_MAIL,					//璁剧疆MAIL閰嶇疆
	CMD_GET_ALARMNAME,				//鑾峰彇鎶ヨ鍚嶉厤缃�
	CMD_SET_ALARMNAME,				//璁剧疆鎶ヨ鍚嶉厤缃�
	CMD_GET_WFNET,					//鑾峰彇鏃犵嚎缃戠粶閰嶇疆
	CMD_SET_WFNET,					//璁剧疆鏃犵嚎缃戠粶閰嶇疆
	CMD_GET_SEND_DEST,				//璁剧疆瑙嗛瀹氬悜鍙戦�佺洰鏍囨満
	CMD_SET_SEND_DEST,				//璁剧疆瑙嗛瀹氬悜鍙戦�佺洰鏍囨満
	CMD_GET_AUTO_RESET,			//鍙栧緱瀹氭椂閲嶆柊娉ㄥ唽
	CMD_SET_AUTO_RESET,			//璁剧疆瀹氭椂閲嶆柊娉ㄥ唽
	CMD_GET_REC_SCHEDULE,			//鍙栧緱褰曞儚绛栫暐
	CMD_SET_REC_SCHEDULE,			//璁剧疆褰曞儚绛栫暐
	CMD_GET_DISK_INFO,				//鍙栧緱纾佺洏淇℃伅
	CMD_SET_MANAGE,				//璁剧疆鍛戒护鍜屾搷浣�
	CMD_GET_CMOS_REG,				//鍙栧緱CMOS鍙傛暟
	CMD_SET_CMOS_REG,				//璁剧疆CMOS鍙傛暟
	CMD_SET_SYSTEM_CMD,			//璁剧疆鎵ц鍛戒护
	CMD_SET_KEYFRAME_REQ,         //70.璁剧疆鍏抽敭甯ц姹�
    CMD_GET_CONFIGENCPAR,         //71.鍙栧緱瑙嗛鍙傛暟
    CMD_SET_CONFIGENCPAR,         //72.璁剧疆瑙嗛鍙傦拷
    CMD_GET_ALL_PARAMETER_NEW,    //73.鑾峰彇鎵�鏈夊弬鏁�
    CMD_FING_LOG,                  //74.鏌ユ壘鏃ュ織(鏌ヨ鏂瑰紡:0锛嶅叏閮紝1锛嶆寜绫诲瀷锛�2锛嶆寜鏃堕棿锛�3锛嶆寜鏃堕棿鍜岀被鍨� 0xFF-鍏抽棴鏈鎼滅储)
    CMD_GET_LOG,                   //75.璇诲彇鏌ユ壘鍒扮殑鏃ュ織 
    CMD_GET_SUPPORT_AV_FMT,       //76.鑾峰彇璁惧鏀寔鐨勭紪鐮佹牸寮忋�佸楂樺強闊抽鏍煎紡
    CMD_GET_VIDEO_CONFIG_NEW,     //77.鑾峰彇瑙嗛鍙傛暟锛峮ew
    CMD_SET_VIDEO_CONFIG_NEW,     //78.璁剧疆瑙嗛鍙傛暟锛峮ew
    CMD_GET_VMOTION_CONFIG_NEW,   //79.鑾峰彇绉诲姩鎶ヨ鍙傛暟锛峮ew
    CMD_SET_VMOTION_CONFIG_NEW,   //80.璁剧疆绉诲姩鎶ヨ鍙傛暟锛峮ew
    CMD_GET_VLOST_CONFIG_NEW,     //81.鑾峰彇瑙嗛涓㈠け鎶ヨ鍙傛暟锛峮ew
    CMD_SET_VLOST_CONFIG_NEW,     //82.璁剧疆瑙嗛涓㈠け鎶ヨ鍙傛暟锛峮ew
    CMD_GET_SENSOR_ALARM_NEW,     //83.鑾峰彇鎺㈠ご鎶ヨ鍙傛暟锛峮ew
    CMD_SET_SENSOR_ALARM_NEW,     //84.璁剧疆鎺㈠ご鎶ヨ鍙傛暟锛峮ew
    CMD_GET_NET_ALARM_CONFIG,     //85.鑾峰彇缃戠粶鏁呴殰鎶ヨ鍙傛暟
    CMD_SET_NET_ALARM_CONFIG,     //86.璁剧疆缃戠粶鏁呴殰鎶ヨ鍙傛暟
    CMD_GET_RECORD_CONFIG,        //87.鑾峰彇瀹氭椂褰曞儚鍙傛暟
    CMD_SET_RECORD_CONFIG,        //88.瀹氭椂褰曞儚鍙傛暟
    CMD_GET_SHOOT_CONFIG,         //89.鑾峰彇瀹氭椂鎶撴媿鍙傛暟
    CMD_SET_SHOOT_CONFIG,         //90.璁剧疆瀹氭椂鎶撴媿鍙傛暟
    CMD_GET_FTP_CONFIG,           //91.鑾峰彇FTP鍙傛暟
    CMD_SET_FTP_CONFIG,           //92.璁剧疆FTP鍙傛暟
    CMD_GET_RF_ALARM_CONFIG,      //93.鑾峰彇鏃犵嚎鎶ヨ鍙傛暟
    CMD_SET_RF_ALARM_CONFIG,      //94.璁剧疆鏃犵嚎鎶ヨ鍙傛暟
    CMD_GET_EXT_DATA_CONFIG,      //95.鑾峰彇鍏跺畠鎵╁睍鍙傛暟(濡傚钩鍙拌缃叾瀹冨弬鏁�)
    CMD_SET_EXT_DATA_CONFIG,      //96.璁剧疆鍏跺畠鎵╁睍鍙傛暟(濡傚钩鍙拌缃叾瀹冨弬鏁�)
    CMD_GET_FORMAT_PROCESS,       //97.鑾峰彇纭洏鏍煎紡鍖栬繘搴�
    CMD_GET_PING_CONFIG,          //98.PING 璁剧疆鑾峰彇
    CMD_SET_PING_CONFIG,          //99.PING 璁剧疆璁剧疆
	//瑙ｇ爜鍣ㄥ懡浠�
	DDCMD_GET_ALL_PARAMETER = 100,	//鑾峰彇瑙ｇ爜鍣ㄦ墍鏈夎缃�
	DDCMD_GET_TIME,				//鑾峰彇绯荤粺鏃堕棿
	DDCMD_SET_TIME,				//璁剧疆绯荤粺鏃堕棿
	DDCMD_GET_SYS_CONFIG,			//鑾峰彇绯荤粺閰嶇疆
	DDCMD_SET_SYS_CONFIG,			//璁剧疆绯荤粺閰嶇疆
	DDCMD_GET_NET_CONFIG,			//鑾峰彇缃戠粶閰嶇疆
	DDCMD_SET_NET_CONFIG,			//璁剧疆缃戠粶閰嶇疆
	DDCMD_GET_COM_CONFIG,			//鑾峰彇涓插彛閰嶇疆
	DDCMD_SET_COM_CONFIG,			//璁剧疆涓插彛閰嶇疆
	DDCMD_GET_VIDEO_CONFIG,		//鑾峰彇瑙嗛閰嶇疆
	DDCMD_SET_VIDEO_CONFIG,		//璁剧疆瑙嗛閰嶇疆
	DDCMD_GET_ALARM_OPT,			//鑾峰彇鎶ヨ閫夐」
	DDCMD_SET_ALARM_OPT,			//璁剧疆鎶ヨ閫夐」
	DDCMD_GET_USER_INFO,			//鑾峰彇鐢ㄦ埛璁剧疆淇℃伅
	DDCMD_SET_USER_INFO,			//璁剧疆鐢ㄦ埛璁剧疆淇℃伅
	DDCMD_GET_ALARM_RECORD,		//鑾峰彇鎶ヨ璁板綍淇℃伅
	DDCMD_GET_ADRRESS_BOOK,		//鑾峰彇鍦板潃钖勯厤缃�
	DDCMD_SET_ADRRESS_BOOK,		//璁剧疆鍦板潃钖勯厤缃�
	DDCMD_SET_COMM,				//璁剧疆鍙戦�佷覆鍙ｆ暟鎹�
	DDCMD_SET_CMD,					//璁剧疆閫忔槑鐨勫懡浠�
	DDCMD_GET_YUNTAI_INFO,			//鑾峰彇浜戝彴淇℃伅
	DDCMD_GET_YUNTAI_CONFIG,		//鑾峰彇浜戝彴閰嶇疆
	DDCMD_SET_YUNTAI_CONFIG,		//璁剧疆浜戝彴閰嶇疆
	DDCMD_GET_ONELINK_ADDR,		//鑾峰彇瑙ｇ爜鍣ㄥ崟璺繛鎺ョ殑淇℃伅
	DDCMD_SET_ONELINK_ADDR,		//璁剧疆瑙ｇ爜鍣ㄥ崟璺繛鎺ョ殑淇℃伅
	DDCMD_GET_CYCLELINK_ADDR,		//鑾峰彇瑙ｇ爜鍣ㄥ惊鐜繛鎺ョ殑淇℃伅
	DDCMD_SET_CYCLELINK_ADDR,		//璁剧疆瑙ｇ爜鍣ㄥ惊鐜繛鎺ョ殑淇℃伅
	DDCMD_GET_EXTINFO,				//鑾峰彇鎵╁睍閰嶇疆
	DDCMD_SET_EXTINFO,				//璁剧疆鎵╁睍閰嶇疆
	DDCMD_GET_NTP,					//鑾峰彇NTP閰嶇疆
	DDCMD_SET_NTP,					//璁剧疆NTP閰嶇疆
	DDCMD_GET_UPNP,				//鑾峰彇UPNP閰嶇疆
	DDCMD_SET_UPNP,				//璁剧疆UPNP閰嶇疆
	DDCMD_GET_MAIL,				//鑾峰彇MAIL閰嶇疆
	DDCMD_SET_MAIL,				//璁剧疆MAIL閰嶇疆
	DDCMD_GET_ALARMNAME,			//鑾峰彇鎶ヨ鍚嶉厤缃�
	DDCMD_SET_ALARMNAME,			//璁剧疆鎶ヨ鍚嶉厤缃�
	DDCMD_GET_WFNET,				//鑾峰彇鏃犵嚎缃戠粶閰嶇疆
	DDCMD_SET_WFNET,				//璁剧疆鏃犵嚎缃戠粶閰嶇疆
	DDCMD_GET_SEND_DEST,			//璁剧疆瑙嗛瀹氬悜鍙戦�佺洰鏍囨満
	DDCMD_SET_SEND_DEST,			//璁剧疆瑙嗛瀹氬悜鍙戦�佺洰鏍囨満

	CMD_GET_VPN_CONFIG = 200,		//200.鑾峰彇VPN璁剧疆鍙傛暟
	CMD_SET_VPN_CONFIG = 201,		//201.璁剧疆VPN鍙傛暟
	CMD_GET_3G_CONFIG  = 202,		//鑾峰彇3G鍙傛暟
	CMD_SET_3G_CONFIG  = 203,      //璁剧疆3G鍙傛暟
	CMD_GET_GPS_CONFIG = 204,
	CMD_SET_GPS_CONFIG = 205,
	CMD_GET_3G_DIALCTRL= 206,
	CMD_SET_3G_DIALCTRL= 207,	
	
	//鍙傛暟鎵╁睍===================
	CMD_GET_IR_CONFIG = 400,		//鑾峰彇绾㈠淇℃伅閰嶇疆
	CMD_SET_IR_CONFIG,				//璁剧疆绾㈠淇℃伅閰嶇疆
	CMD_GET_ALL_CONFIGPARAM,		//鑾峰彇鎵�鏈夊弬鏁�
	CMD_SET_FORMATTING, 		//鏍煎紡鍖�

	CMD_GET_VI_SENSOR=1000,
	CMD_SET_VI_SENSOR,
	CMD_GET_VI_SCENE,
	CMD_SET_VI_SCENE,
	CMD_GET_VI_CFG,
	CMD_SET_VI_CFG,
}CMD_NET;

typedef enum _RELAY_CHECK_RET
{
	RCRET_SUCCESS = 0,
	RCRET_FAILURE = -1,
	RCRET_AUTO_LINK = 0x0101,	
}RELAY_CHECK_RET;



/*********************************  浜屻�佸洖璋冨嚱鏁�  ******************************/

//1锛屽疄鏃堕煶瑙嗛鏁版嵁娴佸洖璋�
typedef int  ( *ChannelStreamCallback)(HANDLE hOpenChannel,void *pStreamData,UINT dwClientID,void *pContext,ENCODE_VIDEO_TYPE encodeVideoType,ULONG frameno);

//2锛屽疄鏃跺璁查煶棰戞暟鎹祦鍥炶皟
typedef int  ( *TalkStreamCallback)(void *pTalkData,UINT nTalkDataLen,void *pContext);

//3锛屾秷鎭�氱煡锛岄�氱煡璋冪敤
typedef int  ( *MsgNotifyCallback)(UINT dwMsgID,UINT ip,UINT port,HANDLE hNotify,void *pPar);

//4锛屾鏌ョ敤鎴凤紝瀹㈡埛绔櫥闄嗘椂妫�鏌�
typedef int  (*CheckUserPsw)(const CHAR *pUserName,const CHAR *pPsw);

//5锛屽鎴风浼犻�掔殑娑堟伅
typedef int  (*ServerMsgReceive)(ULONG ip,ULONG port,CHAR *pMsgHead);

//6锛屽崌绾�
typedef int	 (*ServerUpdateFile)(int nOperation,int hsock,ULONG ip,ULONG port,int nUpdateType,CHAR *pFileName,CHAR *pFileData,int nFileLen);


//7锛岃浆鍙戞湇鍔＄殑鐢ㄦ埛妫�娴嬪洖璋�
typedef int	 (*RelayCheckUserCallback)(RELAY_TYPE relayType,UINT dwClientIP,USHORT wClientPort,CHAR *pszUserName,CHAR *pszPassword,CHAR *pszDeviceName,UINT dwRequstChannel,INT bOnline,CHAR *pDeviceIP,UINT *pdwDevicePort,CHAR *pContext);

//8锛屼腑蹇冩湇鍔″櫒妯″紡鐢ㄦ埛妫�娴嬪洖璋�
typedef int	 (*CenterCheckUserCallback)(INT bOnLine,DVS_MACHINE_TYPE machineType,UINT dwDeviceID,UINT dwChannelNum,UINT ip,USHORT port,CHAR *pszDeviceName,CHAR *pszUserName,CHAR *pszPassword,LPVOID pNetPar);

//9锛屾悳绱㈠綍鍍廚VS鍥炶皟
typedef void (*SearchRecNVSCallback)(CHAR *szNvsBuffer,int nBufferLen);

//10锛屾悳绱㈠綍鍍忔枃浠�
//typedef void (WINAPI *SearchRecFileCallback)(void *pRecFile);
typedef void  (*SearchRecFileCallback)(UINT dwClientID,void *pRecFile);

//11锛屽簱娑堟伅鍥炴帀鍑芥暟
typedef int	 (*MessageNotifyCallback)(UINT wParam, UINT lParam);



/******************************  涓夈�佹帴鍙ｇ粨鏋勭被鍨嬪畾涔�  ***************************/
#ifndef AV_INFO_DEFINE
#define AV_INFO_DEFINE

//1锛岃棰戦煶棰戝弬鏁�
typedef struct _AV_INFO
{
    //瑙嗛鍙傛暟
    UINT			nVideoEncodeType;		//瑙嗛缂栫爜鏍煎紡
    UINT			nVideoHeight;			//瑙嗛鍥惧儚楂�
    UINT			nVideoWidth;			//瑙嗛鍥惧儚瀹�
    //闊抽鍙傛暟
    UINT			nAudioEncodeType;		//闊抽缂栫爜鏍煎紡
    UINT			nAudioChannels;			//閫氶亾鏁�
    UINT			nAudioBits;				//浣嶆暟
    UINT			nAudioSamples;			//閲囨牱鐜�
}AV_INFO,*PAV_INFO;

#endif //AV_INFO_DEFINE


//2锛岄煶瑙嗛鏁版嵁甯уご
typedef struct _FRAME_HEAD
{
	USHORT	zeroFlag;				// 0
	UCHAR   oneFlag;				// 1
	UCHAR	streamFlag;				// 鏁版嵁甯ф爣蹇� FRAME_FLAG_VP锛孎RAME_FLAG_VI锛孎RAME_FLAG_A
	
	ULONG	nByteNum;				//鏁版嵁甯уぇ灏�
	ULONG	nTimestamp;				//鏃堕棿鎴�
}FRAME_HEAD;

//3,鎶ヨ杈撳嚭

typedef struct _ALARM_STATUS_OUTPUT_NEW
{
	unsigned char year;

	unsigned char month;

	unsigned char day;

	unsigned char week;

	unsigned char hour;

	unsigned char minute;

	unsigned char second;

	unsigned char millsecond; 

	unsigned int SensorAlarm;
	unsigned int MotionAlarm;
	unsigned int ViLoseAlarm;
	unsigned int RFSensorAlarm;
	unsigned int NetAlarm;

	unsigned int SensorAlarmRec[MAX_SENSOR_NUM];
	unsigned int MotionAlarmRec[MAX_VIDEO_NUM];
	unsigned int ViLoseAlarmRec[MAX_VIDEO_NUM];
	unsigned int RFSensorAlarmRec[MAX_RF_SENSOR_NUM];
	unsigned int NetAlarmRec;

	unsigned int OutputStatus;

	unsigned int reserved[19];
}ALARM_STATUS_OUTPUT_NEW;


//4锛屾姤璀﹂�氱煡淇℃伅缁撴瀯

typedef struct _ALARM_MSG_NOTIFY_NEW
{
	HANDLE hLogonServer;

	UINT dwClientID;

	UINT dwServerIP;

	UINT dwServerPort;

	ALARM_STATUS_OUTPUT_NEW alarmStatus;
}ALARM_MSG_NOTIFY_NEW;



//5锛屾墦寮�瑙嗛閫氶亾鍙傛暟

typedef struct _OPEN_CHANNEL_INFO_EX
{
    ULONG                         dwClientID;                       //鍥炶皟鍙傛暟	(瀵瑰簲鍥炶皟鍑芥暟閲岀殑dwClientID)
    UINT                          nOpenChannel:8;                   //閫氶亾鍙凤紙0 锝� 3锛�
    UINT                          nSubChannel:8;                    //0: 鎵撳紑涓荤爜娴�      1: 鎵撳紑浠庣爜娴�
    UINT                          res:16;                            //澶囩敤
    NET_PROTOCOL_TYPE             protocolType;                     //杩炴帴鏂瑰紡锛圱CP銆乁DP銆佸鎾級  
    ChannelStreamCallback         funcStreamCallback;              //闊宠棰戞祦鏁版嵁鍥炶皟鍑芥暟 
    void                          *pCallbackContext;               //鍥炶皟鍙傛暟2(瀵瑰簲鍥炶皟鍑芥暟閲岀殑pContext) 
}OPEN_CHANNEL_INFO_EX;

//6锛屾墦寮�瑙嗛閫氶亾鍙傛暟
typedef struct _OPEN_VIEWPUSH_INFO
{
	UINT					dwClientID;
	UINT					nViewPushChannel;
	NET_PROTOCOL_TYPE		protocolType;
	AV_INFO				avInformation;
	UINT					nMulticastAddr;
	UINT					nMulticastPort;
	UINT					nScreenCount;
	UINT					nScreenIndex;
}OPEN_VIEWPUSH_INFO;

//7锛屾墦寮�鐨勬湇鍔″櫒淇℃伅
typedef struct _SERVER_INFO
{
	HANDLE					hServer;
	CHAR					szServerIP[MAX_IP_NAME_LEN+1];
	UINT					nServerPort;
	CHAR					szDeviceName[DVS_NAME_LEN+1];
	UINT					nDeviceID;
	CHAR					szUserName[USER_NAME_LEN+1];
	CHAR					szUserPassword[USER_PASSWD_LEN+1];
	UINT					dwClientID;
	CONNECT_STATUS			logonStatus;
	UINT					nVersion;
	UINT					nLogonID;
	UINT					nPriority;
	UINT					nServerChannelNum;
	UINT					nLanguageNo;
	DVS_MACHINE_TYPE		nMachineType;
	INT						bPalStandard;
	UINT					nMulticastAddr;
	UINT					nMulticastPort;
	AV_INFO					avInformation[MAX_VIDEO_NUM];
}SERVER_INFO;

//8锛屾墦寮�鐨勯�氶亾淇℃伅
typedef struct _CHANNEL_INFO
{
	HANDLE					hOpenChannel;
	CHAR					szServerIP[MAX_IP_NAME_LEN+1];
	UINT					nServerPort;
	CHAR					szDeviceName[DVS_NAME_LEN+1];
	CHAR					szUserName[USER_NAME_LEN+1];
	CHAR					szUserPassword[USER_PASSWD_LEN+1];
	UINT					dwClientID;
	CONNECT_STATUS			openStatus;
	UINT					nVersion;
	UINT					nOpenID;
	UINT					nPriority;
	UINT					nOpenChannelNo;
	UINT					nMulticastAddr;
	UINT					nMulticastPort;
	AV_INFO				avInformation;
	ENCODE_VIDEO_TYPE		encodeVideoType;
	NET_PROTOCOL_TYPE		protocolType;
	ChannelStreamCallback	funcStreamCallback;
	void					*pCallbackContext;
	UINT					dwDeviceID;	//V4.0 add
}CHANNEL_INFO;

//9锛屾墦寮�鐨勮В鐮佸櫒杈撳叆閫氶亾淇℃伅
typedef struct _VIEWPUSH_INFO
{
	HANDLE				hOpenChannel;
	CHAR				szServerIP[MAX_IP_NAME_LEN+1];
	UINT				nServerPort;
	CHAR				szDeviceName[DVS_NAME_LEN+1];
	CHAR				szUserName[USER_NAME_LEN+1];
	CHAR				szUserPassword[USER_PASSWD_LEN+1];
	UINT				dwClientID;
	CONNECT_STATUS		openStatus;
	UINT				nVersion;
	UINT				nOpenID;
	UINT				nPriority;
	UINT				nOpenChannelNo;
	UINT				nMulticastAddr;
	UINT				nMulticastPort;
	AV_INFO			avInformation;
	ENCODE_VIDEO_TYPE	encodeVideoType;
	NET_PROTOCOL_TYPE	protocolType;
	DVS_MACHINE_TYPE	nMachineType;
	UINT				dwChannelNum;	//瑙ｇ爜鍣ㄦ渶澶ц矾鏁�
}VIEWPUSH_INFO;

//10锛屽璁茬殑淇℃伅
typedef struct _SS_TALK_INFO 
{
	HANDLE				hServer;
	CHAR				szServerIP[MAX_IP_NAME_LEN+1];
	UINT				nServerPort;
	CHAR				szDeviceName[DVS_NAME_LEN+1];
	CHAR				szUserName[USER_NAME_LEN+1];
	CHAR				szUserPassword[USER_PASSWD_LEN+1];
	ULONG		version;	
	ULONG		nMachineType;
	CONNECT_STATUS		logonStatus;
	//audio parameter
	UINT				nAudioEncodeType;
    UINT				nAudioChannels;
    UINT				nAudioBits;
    UINT				nAudioSamples;
}TALKING_INFO;

//11锛岃闊冲箍鎾殑鐢ㄦ埛淇℃伅
typedef struct _BROADCAST_USER
{
	HANDLE	hBroadcastUser;
	CHAR	szServerIP[MAX_IP_NAME_LEN+1];
	UINT	port;
	CHAR	szDeviceName[DVS_NAME_LEN+1];	
	CHAR	szUserName[USER_NAME_LEN+1];
	CHAR	szUserPassword[USER_PASSWD_LEN+1];
	INT	bConnect;
	//SOCKET	hSock;
	//PVOID	pPar;
	//add v4.2
	UINT				machineType;
	ENCODE_VIDEO_TYPE	audioType;
}BROADCAST_USER;

//12锛岃浆鍙戣棰戞湇鍔″櫒
typedef struct _RELAY_NVS
{
	RELAY_TYPE			relayType;
	CHAR				szServerIP[MAX_IP_NAME_LEN+1];
	USHORT				wServerPort;
	CHAR				szDeviceName[DVS_NAME_LEN+1];
	UINT				dwChannelNo;
	UINT				dwCount;
}RELAY_NVS;	

//13锛岀櫨鍒嗘瘮娑堟伅閫氱煡
typedef struct _PERCENT_NOTIFY
{
	HANDLE				hLogon;
	UINT				dwClientID;
	UINT				dwPercent;
}PERCENT_NOTIFY;

//14锛岃棰戞枃浠跺弬鏁�
typedef struct _FILE_INFO
{
	CHAR                szFileName[MAX_PATH];
	CHAR				szServerIP[MAX_IP_NAME_LEN+1];
	USHORT				wServerPort;
	CHAR				szUserName[USER_NAME_LEN+1];
	CHAR				szUserPassword[USER_PASSWD_LEN+1];
	INT                bRelay     ; // 姝ゆ枃浠舵槸鍚﹂�氳繃杞彂
}FILE_INFO;

//15锛屾墦寮�瑙嗛鏂囦欢鍙傛暟
typedef struct _OPEN_FILE_INFO
{
	UINT				dwClientID  ;
	UINT				nOpenChannel;
	NET_PROTOCOL_TYPE	protocolType;
	CHAR				szDeviceName[DVS_NAME_LEN+1];
	FILE_INFO        hhFile      ;       
	CHAR                szOpenMode[5];
	UINT				dwSocketTimeOut;
	
    // 杈撳嚭
	UINT				dwFileSize;         // 鏂囦欢澶у皬
	UINT				dwStartTime;        // 鎾斁寮�濮嬫椂闂�(姣)
	UINT				dwEndTime;          // 鎾斁缁撴潫鏃堕棿(姣)
	
	UINT				nVideoEncodeType;	//瑙嗛缂栫爜鏍煎紡
	UINT				nAudioEncodeType;	//闊抽缂栫爜鏍煎紡	
}OPEN_FILE_INFO;

//16锛屽惎鍔ㄦ悳绱㈢鍚堟潯浠剁殑NVS
typedef struct _SEARCH_REC_NVS
{
	UINT				dwClientID  ;
	// 瀛樺偍褰曞儚鏂囦欢鐨勬潯浠�
	CHAR                Date[11];			// 鏉′欢1 褰曞儚鏃ユ湡 yyyy-mm-dd
    UCHAR                recType ;			// 鏉′欢2 褰曞儚绫诲瀷: 0-鎵�鏈夛紝1-鎶ヨ锛�2-鎵嬪姩锛�3-瀹氭椂
	
	//SearchRecNVSCallback	funcSearchRecNvsCallback;	
}SEARCH_REC_NVS;

// 17锛屽惎鍔ㄦ悳绱㈢鍚堟潯浠剁殑褰曞儚鏂囦欢
typedef struct _SEARCH_REC_FILE
{
	UINT				dwClientID  ;

	// 瀛樺偍褰曞儚鏂囦欢鐨勬潯浠�
    CHAR                szDir[MAX_PATH]; 	// 鏉′欢1: " Datae\\Ip-NVS\\Camera\\"
	CHAR				szTime1[6];      	// 鏉′欢2 鏃堕棿娈�1 hh:mm
	CHAR				szTime2[6];      	// 鏉′欢2 鏃堕棿娈�2 hh:mm
    UCHAR                recType ;	    	 // 鏉′欢3 褰曞儚绫诲瀷: 0-鎵�鏈夛紝1-鎶ヨ锛�2-鎵嬪姩锛�3-瀹氭椂
	
	SearchRecFileCallback	funcSearchRecFileCallback;
}SEARCH_REC_FILE;

//18锛岃浆鍙戞煡璇�
typedef struct _RELAY_QUERY_INFO
{
	UINT		dwSize;
	UINT		dwServerLogonNum;
	UINT		dwServerPreviewNum;
	UINT		dwServerTalkNum;
	UINT		dwServerBrdNum;
	UINT		dwClientLogonNum;
	UINT		dwClientPreviewNum;
	UINT		dwClientTalkNum;
	UINT		dwClientBrdNum;
	CHAR		reserve[16];
}RELAY_QUERY_INFO;



typedef struct _SEARCH_SER_INFO
{ 
	char				userName[USER_NAME_LEN+1];
	char				userPassword[USER_PASSWD_LEN+1];
	DVS_MACHINE_TYPE	nDeviceType;
	char				szDeviceName[64];
	unsigned long		ipLocal;
	unsigned char		macAddr[6];
	unsigned short		wPortWeb;
	unsigned short		wPortListen;
	unsigned long		ipSubMask;
	unsigned long		ipGateway;
	unsigned long		ipMultiAddr;
	unsigned long		ipDnsAddr;
	unsigned short		wMultiPort;
	int					nChannelNum;
}SEARCH_SER_INFO;

//========================================================================
//				鍥涖�� 闊宠棰戠綉缁滃抚澶�
//========================================================================

//1锛屾暟鎹抚鏍囧織
#define FRAME_FLAG_VP		0x0b	//瑙嗛鐨凱甯�
#define FRAME_FLAG_VI		0x0e	//瑙嗛鐨処甯�
#define FRAME_FLAG_A		0x0d	//闊抽甯�

//鎵╁睍甯уご
#define	EXT_HEAD_FLAG	0x06070809
#define	EXT_TAIL_FLAG	0x0a0b0c0d

//闊抽缂栫爜绠楁硶
typedef enum  _PT_AENC_TYPE_E
{
	PT_AENC_NONE   = 0x0,
	PT_AENC_G726   = 0x01,
	PT_AENC_G722   = 0x02,
	PT_AENC_G711A  = 0x03,
	PT_AENC_ADPCM  = 0x04,
	PT_AENC_MP3    = 0x05,
	PT_AENC_PCM    = 0x06,
	PT_AENC_G711U  = 0x07,
	PT_AENC_AACLC  = 0x08,
	PT_AENC_AMRNB  = 0x09,
}PT_AENC_TYPE_E;


//瑙嗛缂栫爜绠楁硶
typedef enum  _PT_VENC_TYPE_E
{
	PT_VENC_NONE   = 0x0,
	PT_VENC_H264   = 0x01,
	PT_VENC_MPEG4  = 0x02,
	PT_VENC_MJPEG  = 0x03,
	PT_VENC_JPEG   = 0x04,
}PT_VENC_TYPE_E;



//瑙嗛鍙傛暟
typedef struct _EXT_FRAME_VIDEO
{
	unsigned short	nVideoEncodeType;	//瑙嗛缂栫爜绠楁硶
	unsigned short	nVideoWidth;		//瑙嗛鍥惧儚瀹�
	unsigned short	nVideoHeight;		//瑙嗛鍥惧儚楂�
	unsigned char   nPal;               //鍒跺紡
	unsigned char   bTwoFeild;			//鏄惁鏄袱鍦虹紪鐮�(濡傛灉鏄袱鍦虹紪鐮侊紝PC绔渶瑕佸仛deinterlace)
	unsigned char   nFrameRate;			//甯х巼
	unsigned char   szReserve[7];		//

} EXT_FRAME_VIDEO;

//闊抽鍙傛暟
typedef struct _EXT_FRAME_AUDIO
{
	unsigned short	nAudioEncodeType;	//闊抽缂栫爜绠楁硶
	unsigned short	nAudioChannels;		//閫氶亾鏁�
	unsigned short	nAudioBits;			//浣嶆暟
	unsigned char   szReserve[2];
	unsigned long	nAudioSamples;		//閲囨牱鐜� 	
	unsigned long	nAudioBitrate;		//闊抽缂栫爜鐮佺巼
} EXT_FRAME_AUDIO;

typedef union _EXT_FRAME_TYPE
{
	EXT_FRAME_VIDEO	szFrameVideo;
	EXT_FRAME_AUDIO	szFrameAudio;
} EXT_FRAME_TYPE;

typedef struct _EXT_FRAME_HEAD
{
    unsigned long	nStartFlag;			//鎵╁睍甯уご璧峰鏍囪瘑
    unsigned short	nVer;				//鐗堟湰
    unsigned short	nLength;			//鎵╁睍甯уご闀垮害
	EXT_FRAME_TYPE	szFrameInfo;		
	unsigned long   nTimestamp;			//浠ユ绉掍负鍗曚綅鐨勬椂闂存埑
	unsigned long	nEndFlag;			//鎵╁睍甯уご璧峰鏍囪瘑
}EXT_FRAME_HEAD;




typedef INT  (*StreamWriteCheck)(int nOperation,const CHAR *pUserName,const CHAR *pPsw,ULONG ip,ULONG port,OPEN_VIEWPUSH_INFO viewPushInfo,HANDLE hOpen);
typedef INT (*CallbackServerFind)(SEARCH_SER_INFO *pSearchInfo);

/********************************  浜斻�佸嚱鏁版帴鍙ｅ畾涔�  *****************************/

//鍚姩鏈嶅姟
ERR_CODE		NET_Startup(USHORT nBasePort,MsgNotifyCallback msgCallback,CheckUserPsw checkUserPswCallback,ServerUpdateFile updateCallback,ServerMsgReceive msgCmdCallback,StreamWriteCheck streamWriteCheckCallback,ChannelStreamCallback funcChannelCallback);
//鍏抽棴鏈嶅姟
ERR_CODE		NET_Cleanup();


//鐧诲綍鏈嶅姟鍣�
ERR_CODE		NET_LogonServer(IN  CHAR *pServerIP,IN  UINT nServerPort,IN  CHAR *pDeviceName,IN	 CHAR *pUserName,IN	 CHAR *pUserPassword,IN UINT dwClientID,OUT  HANDLE *hLogonServer);
//娉ㄩ攢鏈嶅姟鍣�
ERR_CODE		NET_LogoffServer(IN  HANDLE hServer);
//璇诲彇鐧诲綍鏈嶅姟鍣ㄤ俊鎭�
ERR_CODE		NET_ReadServerInfo(IN  HANDLE hServer,OUT  SERVER_INFO *serverInfo);

//绋嬪簭鍗囩骇
ERR_CODE		NET_Update(IN HANDLE hServer,IN UPDATE_TYPE nUpdateType,IN CHAR *pFilePathName);


//璇诲彇鐧诲綍鏈嶅姟鍣ㄩ厤缃俊鎭�
ERR_CODE		NET_GetServerConfig(IN  HANDLE hServer,IN  CMD_NET nConfigCommand,OUT  CHAR *pConfigBuf,IN OUT  UINT *nConfigBufSize,IN  OUT  UINT *dwAppend);
//璁剧疆鐧诲綍鏈嶅姟鍣ㄩ厤缃俊鎭�
ERR_CODE		NET_SetServerConfig(IN  HANDLE hServer,IN  CMD_NET nConfigCommand,IN   CHAR *pConfigBuf,IN  UINT nConfigBufSize,IN  UINT dwAppend);


//鎵撳紑瑙嗛閫氶亾
ERR_CODE		NET_OpenChannel(IN  CHAR *pServerIP,IN  UINT nServerPort,IN  CHAR *pDeviceName,IN  CHAR *pUserName,IN  CHAR *pUserPassword,IN  OPEN_CHANNEL_INFO_EX *pOpenInfo,OUT  HANDLE *hOpenChannel);
//鍏抽棴瑙嗛閫氶亾
ERR_CODE		NET_CloseChannel(IN  HANDLE hOpenChannel);

//璇诲彇鎵撳紑瑙嗛閫氶亾淇℃伅
ERR_CODE		NET_ReadChannelInfo(IN  HANDLE hOpenChannel,OUT  CHANNEL_INFO *channelInfo);


//璇锋眰鎵撳紑瀵硅
//ERR_CODE		NET_TalkRequsest(IN  CHAR *pServerIP,IN  UINT nServerPort,IN  CHAR *pDeviceName,IN	 CHAR *pUserName,IN	 CHAR *pUserPassword,IN TalkStreamCallback funcTalkCallback,IN void *pContext);
ERR_CODE		NET_TalkRequsest(IN  char *pServerIP, IN  UINT nServerPort,IN  char *pDeviceName,IN	 char *pUserName,IN	 char *pUserPassword,IN TalkStreamCallback funcTalkCallback,IN void *pContext, OUT  TALKHANDLE *hTalkback);

//缁撴潫瀵硅
//ERR_CODE		NET_TalkStop();
ERR_CODE		NET_TalkStop(IN  TALKHANDLE hTalkback);

//璇诲彇鎵撳紑瀵硅淇℃伅
//ERR_CODE		NET_TalkReadInfo(TALKING_INFO *talkInfo);
//鍙戦�佸璁叉暟鎹�
//ERR_CODE		NET_TalkSend(IN CHAR *pTalkData,IN UINT nDataLen);
ERR_CODE		NET_TalkSend(IN  TALKHANDLE hTalkback, IN char *pTalkData,IN UINT nDataLen);


//鎼滅储缃戜笂璁惧
ERR_CODE		NET_SearchAllServer(UINT nTimeWait,	 CallbackServerFind  funcServerFind);
//璁剧疆瑙嗛鏈嶅姟鍣ㄧ綉缁滈厤缃�
ERR_CODE		NET_ConfigServer(UCHAR macAddr[6],CHAR *pUserName,CHAR *pUserPassword,
													   CHAR *pIP,CHAR *pDeviceName,CHAR *pSubMask,CHAR *pGateway,CHAR *pMultiAddr,
													   CHAR *pDnsAddr,USHORT wPortWeb,USHORT wPortListen,USHORT wPortMulti);

void NET_UpdatePercentNotify(int hsock,ULONG nPercent);

ERR_CODE        NET_SetNetApiSupportVersion(int nVersion);


#endif


