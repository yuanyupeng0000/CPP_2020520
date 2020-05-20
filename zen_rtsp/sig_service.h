/*
 * sig_service.h
 *
 *  Created on: 2016??6??22??
 *      Author: Administrator
 */

#ifndef INCLUDE_SIG_SERVICE_H_
#define INCLUDE_SIG_SERVICE_H_
#include "g_define.h"
#include "cam_alg.h"

#include <pthread.h>
#pragma pack(push)
#pragma pack(1)

enum LINKID {
    LINKID_ETH1 = 1,
    LINKID_ETH2,
    LINKID_UART1,
    LINKID_UART2,
    LINKID_UART3,
    LINKID_UART4,
    LINKID_UART5,
    LINKID_CAN,
    LINKID_UART_BETWEEN_COMBOARD_AND_MAINCONTLBOARD,
    LINKID_BROARDCAST = 0xFF
};
//��??????
enum PROTOCOLTYPE {
    PROTOCOTYPE_VIDEO_DETECTOR = 0x01,
    PROTOCOTYPE_MAX
};
//?��??
enum DEVICECLASS {
    DEVICECLASS_VIDEO_SEPERATED_MACHINE = 0x01,
    DEVICECLASS_IPCAMERAL,
    DEVICECLASS_VIDEO_INTEGRATED_MACHINE,
    DEVICECLASS_MAIN_CONTROL_BOARD,
    DEVICECLASS_COMMUNICATION_BOARD,
    DEVICECLASS_BROADCAST = 0x0F,
};

typedef struct {
    unsigned char mHeaderTag;
    unsigned char mSenderLinkId;
    unsigned char mRecieverLinkId;
    unsigned char mProtocolType;
    unsigned char mProtocolVersion;
    struct {
        unsigned char mDeviceIndex: 4;
        unsigned char mDeviceClass: 4;
    } mSenderDeviceID, mRecieverDeviceId;
    unsigned char mSessionID;
    unsigned char mDataType;
    //unsigned char pContent[1];
} FrameHeader;

typedef struct {
    unsigned char mXor;
    unsigned char mTailTag;
} FrameTail;
///////////////////////////////////////////////// //?????????/////////////
typedef struct EachChannelPack {
    unsigned char mDetectChannelIndex;
    unsigned char mQueueLength;
    unsigned char mRealTimeSingleSpeed;
    struct {
        unsigned char bOccupyStatus0: 1;
        unsigned char bOccupyStatus1: 1;
        unsigned char bFix0: 2;
        unsigned char flow: 2;
        unsigned char bVehicleType: 2;
    } mDetectDataOfHeaderVirtualCoil;
    unsigned char mHeadwayTimeOfHeaderVirtualCoil;
    unsigned char mOccupancyOfHeaderVirtualCoil;
    struct {
        unsigned char bOccupyStatus0: 1;
        unsigned char bOccupyStatus1: 1;
        unsigned char bFix0: 6;
    } mDetectDataOfTailVirtualCoil;
    unsigned char mHeadwayTimeOfTailVirtualCoil;
    unsigned char mOccupancyOfTailVirtualCoil;
    struct {
        unsigned char bFix0: 7;
        unsigned char bDataIsValid: 1;
    } mWorkStatusOfDetectChannle;
} EachChannelPackm;

typedef struct {
    unsigned char mDetectChannelCount;
    EachChannelPackm EachChannelPack[1];
} RealTimeTrafficData;

/////// //?��????????????? ////////////??????????? ///////////////////////////////////////////
//?��???????????? /////////////////////////////////////////

typedef struct {
    struct {
        unsigned char mDeviceIndex: 4;
        unsigned char mDeviceClass: 4;
    } mCameralDeviceId;
    struct {
        unsigned char bNorth: 1;
        unsigned char bEastNorth: 1;
        unsigned char bEast: 1;
        unsigned char bEastSouth: 1;
        unsigned char bSouth: 1;
        unsigned char bWestSouth: 1;
        unsigned char bWest: 1;
        unsigned char bWestNorth: 1;
    } mCameralPosition;
    struct {
        unsigned char bWorkMode: 2;
        unsigned char bBackgroundRefreshed: 1;
        unsigned char bH264DecodeStatus: 1;
        unsigned char bCameralOnLine: 1;
        unsigned char bPictureStable: 1;
        unsigned char bFix0: 2;
    } mCameralStatus;
} EachCameralStatus;

typedef struct {
    struct {
        unsigned char mDeviceIndex: 4;
        unsigned char mDeviceClass: 4;
    } mDetectMainMachineDeviceId;
    union {
        struct {
            unsigned char bRegisted: 1;
            unsigned char bTimeCorrected: 1;
            unsigned char bFix0: 6;
        } mStatusWhenSeperatedMachine;
        unsigned char mFix0WhenIntegratedMachine;
    } uDetectMainMachineStatus;
    unsigned char mCameralCount;
    EachCameralStatus mEachCameralStatus[1];
} DeviceWorkStatusQueryResponse;

//////////////////////////////////////// //????????? ////////
typedef struct {
    unsigned int mUTCTime;
} TimeBroadcastCommand;

///////////////////// //????????? /////////
typedef struct {
    unsigned char mIsRushHour;
} RushHourBroadcastCommand;

////////////////////// //??????????????? /////////////////// //???????????
/////////// //?????????????? ///
typedef struct {
    struct {
        unsigned short mListenPort; unsigned char mIp[4];
        unsigned char mNetmask[4]; unsigned char mGateway[4];
    } mEthernet[2];
    struct {
        unsigned char mWorkElectricLevel;
        unsigned char mBaudRate;
        unsigned char mDataBits;
        unsigned char mStopBits;
        unsigned char mParityBit;
    } mUart[5];
    unsigned char mCanBaudRate;
} CommunicationBoardArgumentQueryResponse;

///////////////////// //???????????????? ////////////
typedef struct {
    struct {
        unsigned short mListenPort;
        unsigned char mIp[4];
        unsigned char mNetmask[4];
        unsigned char mGateway[4];
    } mEthernet[2];
    struct {
        unsigned char mWorkElectricLevel;
        unsigned char mBaudRate;
        unsigned char mDataBits;
        unsigned char mStopBits;
        unsigned char mParityBit;
    } mUart[5];
    unsigned char mCanBaudRate;
} CommunicationBoardArgumentSetUpCommand;

///////////////////////////////// ////??????????????? ///////////////////////////
typedef struct {
    unsigned char mResult;
} CommunicationBoardArgumentSetUpResponse;
//////////////////////////////////////////////////////////////
//?��??????????????
typedef struct {
    struct {
        unsigned char mDeviceIndex: 4;
        unsigned char mDeviceClass: 4;
    } mEventDeviceId;
    union {
        struct {
            unsigned char bNorth: 1;
            unsigned char bEastNorth: 1;
            unsigned char bEast: 1;
            unsigned char bEastSouth: 1;
            unsigned char bSouth: 1;
            unsigned char bWestSouth: 1;
            unsigned char bWest: 1;
            unsigned char bWestNorth: 1;
        } mEventOccourPositionWhenCameralOrIntegratedCameral;
        unsigned char mFix0WhenSeperatedMachine;
    } uEventOccourPosition;
    unsigned char mEvent;
    unsigned char mReserverFix0;
    unsigned int mEventTime;
} DeviceEventsAutoReporte;

#define CHANNELMAXNUM 4
typedef struct car_info {
    int g_flow[4] = {0};
    int g_50frame1[4] = {0};
    int g_50frame2[4] = {0};
    int g_50frametail1[4] = {0};
    int g_50frametail2[4] = {0};
    int g_occupancyframe[4] = {0};
    int g_occupancyframetail[4] = {0};
    int g_staytm[8] = {0};
    int g_staytm_tail[8] = {0};
} m_car_info;
typedef struct staticchannel {
    //unsigned char index;
    unsigned char status;   //??????????��??, 0 ?????, ????????, 1???????, 2??????????????.
    //unsigned char mode;
    //unsigned char black;
    //unsigned char algswitch;
    EachCameralStatus EachStatus;//Camera status
    EachChannelPackm Eachchannel[CHANNELMAXNUM];//This is what signal machine wants
    int camera_state_change = 0;
    int lane_num;
    m_car_info car_info;
    //unsigned int  frameNum;
} m_sig_data;
extern m_sig_data * get_sig_data(int index);

void reset_sig_machine();
void init_sig_service_yihualu();
void init_sig_service_nanjing();
void init_sig_service_haixiing();

extern m_sig_data * get_locked_sig_data(int index);
long get_holder_last_time(int index);
int get_holder_status(int index);
void set_holder_status(int index, unsigned char status);


extern void submit_unlock_sig_data(int index);
void reboot_cmd();
void reset_sig_machine();

//#define NANJING_CAM_NUM 4

typedef struct queue_info {
    unsigned char table_head[5];
    unsigned char table_no;
    unsigned char table_length;
    unsigned char detect_time[6];
    unsigned char detect_status;//whether device good or not
    unsigned char areano;
    unsigned char juncno;
    unsigned char dir_no;//camera direction
    unsigned char lane_dir_type;//income or out come
    unsigned char queue_len[8];//---------
    unsigned char queue_start_pos[8];
    unsigned char queue_veh_num[8];//---------
    unsigned char veh_speed[8];//
    unsigned char crc;
} queue_info_t;


typedef struct radar_realtime {
    //unsigned char addr[2];
    unsigned char addr;
    unsigned char ver;
    unsigned char type;
    unsigned char obj;
} radar_head_t;

typedef struct radar_head {
    unsigned char time[4];
    unsigned int  dev_no;
    unsigned int  camera_no;
    unsigned char lane_count;
} radar_realtime_t;


typedef struct radar_realtime_no_protocol {
    unsigned int sup_large_veh;//超大型车
    unsigned int large_veh;//大型车
    unsigned int mid_veh;//中型车
    unsigned int small_veh;//小型车
    unsigned int min_veh;//微型车

    unsigned int stay_time; //车辆停留时间
    //
    unsigned int head_time;//车头时距
    unsigned int head_space;//车头间距
    //
    unsigned int bus_num;
    unsigned int car_num;
    unsigned int truck_num;
} radar_realtime_no_protocol_t;

typedef struct radar_realtime_lane {
    unsigned char lane_no;
    unsigned char queue_len;
    unsigned char head_len;
    unsigned char tail_len;
    unsigned char queue_no;
    unsigned char lane_vihicle_count;
    unsigned char ocuppy;
    unsigned char average_speed;
    unsigned char location;
    unsigned char head_pos;
    unsigned char head_speed;
    unsigned char tail_pos;
    unsigned char tail_speed;
    radar_realtime_no_protocol_t no_pro_data;
    unsigned int out_car_num;
    int head_tm;
} radar_realtime_lane_t;


typedef struct sig_realtime_lane {
    unsigned char lane_no;
    unsigned char queue_len;
    unsigned char head_len;
    unsigned char tail_len;
    unsigned char queue_no;
    unsigned char lane_vihicle_count;
    unsigned char ocuppy;
    unsigned char average_speed;
    unsigned char location;
    unsigned char head_pos;
    unsigned char head_speed;
    unsigned char tail_pos;
    unsigned char tail_speed;
} sig_realtime_lane_t;

typedef struct radar_cam_realtime {
    pthread_mutex_t mutex_lock;
    radar_realtime_lane_t rt_lane[DETECTLANENUMMAX];
} radar_cam_realtime_t;

typedef struct radar_result {
    unsigned char period[2];
    unsigned char time[4];
    unsigned int  dev_no;
    unsigned int  camera_no;
    unsigned char lane_count;

} radar_result_t;
#if 0
typedef struct radar_result_lane {
    unsigned char lane_no;
    unsigned char queue_len;
    unsigned char condition;
    unsigned char stop_count[4];
    unsigned char average_delay[4];
    unsigned char oil_use[4];
    unsigned char smoge[4];

} radar_result_lane_t;
#else
typedef struct radar_result_lane {
    unsigned char lane_no;
    unsigned char flowA;
    unsigned char flowB;
    unsigned char flowC;
    unsigned char flowSum;
    unsigned char Occupy_rate;
    unsigned char average_speed;
    unsigned char average_len;
    unsigned char average_head_time;
} radar_result_lane_t;


typedef struct radar_result_came_lane {
    radar_result_lane_t lanes[MAX_LANE_NUM][NANJING_LANE_COIL_MAX];
} radar_result_came_lane_t;


#endif

typedef struct radar_cycle_result_lane { //车道号划分的统计周期数据
    unsigned char lane_no;
    unsigned char queue_len_max;
    unsigned char lane_status;
    float car_stop_sum;
    float average_delay_ms;
    float oil_consume;
    float gas_emission; //尾气排放
} radar_cycle_result_lane_t;

typedef struct radar_cycle_result_came_lane { //车道号划分的统计周期数据
    radar_cycle_result_lane_t lanes[MAX_LANE_NUM];
} radar_cycle_result_came_lane_t;

//
typedef struct radar_car_in_out_result {  //进车和出车需要给指定的udp ip
    unsigned char lane_no: 5;
    unsigned char over_len_flag: 1;
    unsigned char in_out_flag: 2;

} radar_car_in_out_result_t;


typedef struct radar_rt_lane_car_in_out_item {
    unsigned char lane_no;
    unsigned char in_flag;
    unsigned char out_flag;
} radar_rt_lane_car_in_out_item_t;

typedef struct radar_rt_lane_car_in_out_info {
    unsigned char index;//by anger
    radar_rt_lane_car_in_out_item_t lanes[MAX_LANE_NUM];
} radar_rt_lane_car_in_out_info_t;

//
typedef struct radar_car_in_out_status {  //上一次进车和出车状态
    unsigned char flag;
} radar_car_in_out_status_t;

typedef struct flow_info {
    unsigned char table_head[5];
    unsigned char table_no;
    unsigned char table_length;
    unsigned char detect_time[6];
    unsigned char areano;
    unsigned char juncno;
    unsigned char dir_no;
    unsigned char section_no;
    unsigned char flow[8];
    unsigned char average_speed[8];
    unsigned char ocuppy_percent[8];
    unsigned char crc;
} flow_info_t;

typedef struct outcar_info {
    unsigned char table_head[5];
    unsigned char table_no;
    unsigned char table_length;
    unsigned char pass_time[6];
    unsigned char areano;
    unsigned char juncno;
    unsigned char dir_no;
    unsigned char lane_dir_type;
    unsigned char section_number;
    unsigned char lane_number;
    unsigned char veh_type;
    unsigned char veh_speed;
    unsigned int occupy_time;
    unsigned char crc;
} ourcar_info_t;
static int out_car_flag;
extern pthread_mutex_t out_car_lock;


typedef struct data_1s {
} data_1s_t;


typedef struct gat920_car_in_out_result {  //进车和出车需要给指定的udp ip
    unsigned char start_flag;
    unsigned char adress;
    unsigned char version;
    unsigned char operation;
    unsigned char obj_flag;
    unsigned char lane_no;
    unsigned char affair;//
    unsigned char data_crc[2]; //
    unsigned char end_flag;
} gat920_car_in_out_result_t;

typedef struct radar_queue_frame_count {  //每5frame统计排队次数
    unsigned char frame_number; //总帧数
    unsigned char queue_number; //排队次数
    unsigned char no_queue_number; //没排队次数
    unsigned char queue_status; //上次排队的状态
} radar_queue_frame_count_t;

typedef struct person_area_time_result { //实时行人检测数据
    unsigned char frame_start;
    unsigned char s_link;
    unsigned char r_link;
    unsigned char prot_type;
    unsigned char prot_version;
    unsigned char s_deviceno;
    unsigned char r_deviceno;
    unsigned char session_id;
    unsigned char data_type;
    unsigned char cam_direct;
    unsigned char area_num;
    mPersonCheckData person_data[PERSON_AREAS_MAX];
    unsigned char crc;
    unsigned char frame_end;
} person_area_time_result_t;

/////////////////////////////////////////////
//公车车牌
typedef struct bus_number_info { //车牌
    unsigned int  rfid;
    unsigned char bus_no;
    char bus_number[20];
    char reserve[10];
} bus_number_info_t;

typedef struct lane_bus_info {
    int bus_cnt;
    bus_number_info bus_num[MAX_LANE_BUS_NUM];
} lane_bus_info_t;

typedef struct camera_bus {
    lane_bus_info_t lane_bus[MAX_LANE_NUM];
} camera_bus_t;

typedef struct recv_bus {
    char header[2];
    char cmd;
    char squence;
    char len_low;
    char len_hight;
    char data[MAX_BUS_DATA];
    char bcc;
} recv_bus_t;

typedef struct third_data_head {
    unsigned char start;
    unsigned char version;
    short         type;
    int           data_len;

} third_data_head_t;

typedef struct third_stat_data {
    char date_time[20];
    unsigned short stat_time;
    char address[50];
    int dector_id;
    int cam_id;
    unsigned char lane_num;
} third_stat_data_t;

typedef struct third_stat_data_loop {
    unsigned char lane_id;
    unsigned short total_num;
    unsigned char aver_speed;
    unsigned char aver_occ_per;
    unsigned short car_head;
    unsigned short car_head_tm;
    unsigned short bus_num;
    unsigned short car_num;
    unsigned short truck_num;
    unsigned short motor_num;
    unsigned short bicycle_num;
    unsigned short queue_len;
    unsigned char  road_stat;
} third_stat_data_loop_t;


typedef struct third_event_data {
    char date_time[20];
    char address[50];
    int dector_id;
    int cam_id;
    unsigned char event_type;
    char pic_name[100];
    char video_name[100];
} third_event_data_t;

typedef struct third_link_data {
    int dector_id;
    char run_status;
    char nomal_status;
} third_link_data_t;

/////////////////////////////////////////////////////
//非机动车辆
typedef struct rev_img { //接受图片id
    char   img_id[20];
}m_rev_img;

typedef struct pos{ //座标
    uint16 x;
	uint16 y;
	uint16 width;
	uint16 height;
} m_pos_t;

typedef struct no_motor{ //人和帽信息
    char   plate_no[20]; //车牌
    uint16 hat_num;
    uint16 person_num;
    uint16 vc_color; //颜色
    m_pos_t  vc_pst;  //车座标
    m_pos_t  hat_pos[10]; //帽子座标
}m_motor_t;

typedef struct img_det_info //非机动车辆信息
{
    char      img_id[20];
    short     non_motor_num;
    m_motor_t motors[20];
}m_img_det_info_t;

#pragma pack(pop)

//杰瑞信号机 
typedef struct jierui_rt_lane_car_data_item {
    unsigned char lane_no;
    unsigned char in_flag;
    unsigned char out_flag;
    unsigned char speed;
    unsigned char queue_len;
} jierui_rt_lane_car_data_item_t;

typedef struct jierui_rt_lane_car_data_info {
    unsigned char index;//by anger
    jierui_rt_lane_car_data_item_t lanes[MAX_LANE_NUM];
 } jierui_rt_lane_car_data_info_t;

//////////////////////
typedef struct sig_holder {
    m_sig_data traffic_rst;
    long last_time;
    unsigned char status; // 0x01--0.25  0x02 --0.5
    mAlgParam algparam;
    pthread_mutex_t sig_data_lock;
} m_holder;

/***********************************************************************/
void init_server_lock();
void init_sig_client();
void statis_handle();
void init_ntp();
void init_server();
void statis_data_insert_mysql();
void protocol_select(unsigned char type);
void protocol_change(unsigned char type, unsigned char old_type);

void init_sig_service_haixing();
void deinit_sig_service_haixing();
void init_sig_service_nanjing();
void deinit_sig_service_nanjing();
void init_sig_service_yihualu();
void deinit_sig_service_yihualu();
void calculate_60s_radar();
bool add_car_in_out_item(int index, radar_rt_lane_car_in_out_info_t *io_item);
//
int SendDataInLock(int sock, char * buffer, int len);
void *sig_client_handle_thread(void *data);
//
int sig_get_state();
int get_holder_status(int index);
int ReportedCammerStatus(int sock, int mSessionID);
//
void send_message(int type, char *buf);
void *message_handle_thread(void *data);
//
void person_area_data_hanle(int index, OUTBUF *p_outbuf);
//
void *init_udp_server(void *data);
//
void *tis_data_insert_mysql(int cam_id, void *data);
//
void netowrk_send_third_data(char *pack, int len);
void send_stat_to_third_server(int cam_id , radar_realtime_lane_t *p_lane_data);
void send_event_to_third_server(int cam_id, void *data);
//
void *gat920_callback_car_in_out_result(void *data);
//
void pic_data_insert_mysql(int cam_id, void *data, char *pic_path);


//void *http_server(void *data);
//void *websocket_server(void *data);
bool get_jierui_car_item(int index, jierui_rt_lane_car_data_info_t *item);
void add_jierui_car_item(int index, jierui_rt_lane_car_data_info_t *item);
int  gat920_get_jierui_car_info(int index, int type, jierui_rt_lane_car_data_info_t &list_data_item);
void add_jierui_60s_item(int index, jierui_60s_t *item);
bool get_jierui_60s_item(int index, jierui_60s_t *item);
void *init_g920_udp_server(void *data);
int  gat920_calculate_60s_jierui(int index);
void *gat920_callback_jierui_60s_result(void *data);
//////////////////////////
void *ping_server(void *data);
#endif /* INCLUDE_SIG_SERVICE_H_ */
