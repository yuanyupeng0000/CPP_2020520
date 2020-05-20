#include <unistd.h>
#include <sys/time.h>
#include <sys/socket.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include "client_net.h"
#include "client_obj.h"
#include "common.h"
#include "g_define.h"
#include "csocket.h"
#include <arpa/inet.h>
#include "file_op.h"
#include "camera_service.h"
#include "client_file.h"
#include "sig_service.h"
#include "ini/fvdconfig.h"
#include "tcp_server.h"
#include "g_fun.h"


////////////////////////////////////////////////////
extern g_all_lock_t g_all_lock;
extern sock_info_t g_sock_info;
extern unsigned char prt_log_leve;
extern unsigned int g_cycle_statis_time;
extern m_timed_func_data radar_realtime_result_info;
extern m_camera_info cam_info[CAM_MAX];
extern unsigned int real_test_run_flag[CAM_MAX];
real_test_data_pack_t real_test_data[CAM_MAX] = {0};
extern mPersonPlanInfo g_personplaninfo[CAM_MAX][PERSON_AREAS_MAX];
extern mEventInfo events_info[CAM_MAX];

#define CHECK_TIMES 15 * 60 //log???????

enum
{
    CLIENT_IP_SAME = 0,
    CLIENT_CAM_INDEX_SAME
};
typedef struct cam_cfg
{
    mCamParam cam_param;
    mCamDetectParam det_param;
} m_cam_cfg;
m_cam_cfg g_cam_cfg[CAM_MAX];
mDetectDeviceConfig g_dev_cfg;
//mRealStaticInfo static_info[CAM_MAX];
EX_mRealStaticInfo ex_static_info;

enum
{
    NOOP_FLAG,
    SET_FLAG,
    GET_FLAG
};

IVDDevInfo g_sysInfo;
IVDDevSets g_ivddevsets;
IVDNetInfo g_netInfo;
IVDTimeStatu g_timestatus;
TimeSetInterface g_timeset;
IVDStatisSets g_statisset;
RS485CONFIG g_rs485;
//mThirProtocol    g_protocol;
IVDNTP g_ivdntp;
IVDCAMERANUM g_cam_num;
mCamDetectParam g_camdetect[CAM_MAX];

m_list_info *ip_list[CAM_MAX];

int pack_cmd_status(int index, unsigned char cmd_no, unsigned char status, unsigned char *buff)
{
    ret_cmd_status_t ret_cmd = {0};
    
    ret_cmd.pack_head.version = 0x01;
    ret_cmd.pack_head.prottype = 0x10;
    ret_cmd.pack_head.objnumber = htons(index);
    ret_cmd.pack_head.objtype = htons(COMMAND_STATUS);
    ret_cmd.pack_head.objlen = htonl(sizeof(ret_cmd_status_t) - sizeof(mCommand));

    ret_cmd.index = index;
    ret_cmd.type = htons(cmd_no);
    ret_cmd.state = status;

    memcpy(buff, &ret_cmd, sizeof(ret_cmd_status_t));

    return sizeof(ret_cmd_status_t);
}

int send_cmd_status(int index, unsigned short cmd_no)
{
    unsigned char buff[50] = {0};
    int len = 0;
    prt(info, "send_cmd_status1[index]:%d", index, cmd_no);
    if (cam_info[index].cmd_fd < 1)
        return -1;

    len = pack_cmd_status(index, cmd_no, 1, buff);
    int s_len = SendDataToClient(cam_info[index].cmd_fd , (char *)buff, len);
    prt(info, "send_cmd_status1[index]:%d cmd: %d len: %d", index, cmd_no, s_len);
    return s_len;
}

void *client_send_udp(void *node_data, void *data)
{
    int index = *((int *)data);
    m_client_ip_data *p_data = (m_client_ip_data *)node_data; //获取IP信息
   
    if (p_data->camera_index != index)
        return NULL;
   

    unsigned int data_len = sizeof(mRealStaticInfo);
    real_data_pack_t read_data_pack;

    read_data_pack.pack_head.version = 0x01;
    read_data_pack.pack_head.prottype = 0x10;
    read_data_pack.pack_head.objnumber = htons(index);
    read_data_pack.pack_head.objtype = htons(REALDATA);
    read_data_pack.pack_head.objlen = htonl(data_len);

    if (p_data->times++ == CHECK_TIMES)
    {
        prt(info, "cam %d output to client %s", index, p_data->ip);
        p_data->times = 0;
    }

    //if (index == 0)
    //  prt(info,"person cam %d persondow: %d personup: %d",index, ex_static_info.static_info[index].upperson, ex_static_info.static_info[index].downperson);

    ex_static_info.static_info[index].camId = g_camdetect[index].other.camerId;
    memcpy(&read_data_pack.static_info, &ex_static_info.static_info[index], sizeof(mRealStaticInfo));
    //if (0 == ntohl(read_data_pack.static_info.frame_no) || ntohl(read_data_pack.static_info.frame_no) > 50000)
    //   prt(info, "ip: %s cam[%d] frame_no: %d lane[0]: %d lane[1]:%d lane[2]:%d...............", p_data->ip, index, ntohl(read_data_pack.static_info.frame_no) , ntohl(read_data_pack.static_info.lane[0].vehnum1), ntohl(read_data_pack.static_info.lane[1].vehnum1), ntohl(read_data_pack.static_info.lane[2].vehnum1) );
    //prt(info, "cam[%d] 1--SendDataToClient start", index);
    //prt(info, "cam[%d] frameno2: %d" ,index, ntohl(read_data_pack.static_info.frame_no) );
    int s_len = SendDataToClient(p_data->fd, (char *)&read_data_pack, sizeof(real_data_pack_t));
    //if (s_len != 25543)
    //   prt(info, "frame_no cam[%d] abc--realtime len: %d ", index, s_len);
    //////////////////////////////////////////////////////////////////////////////////////

    event_data_pack_t event_pack; //事件发送
    data_len = sizeof(mRealEventInfo);
    event_pack.pack_head.version = 0x01;
    event_pack.pack_head.prottype = 0x10;
    event_pack.pack_head.objnumber = htons(index);
    event_pack.pack_head.objtype = htons(EVENTUPDATE);
    event_pack.pack_head.objlen = htonl(data_len);

    memcpy(&event_pack.real_event_info, &ex_static_info.real_event_info[index], sizeof(mRealEventInfo));
    s_len = SendDataToClient(p_data->fd, (char *)&event_pack, sizeof(event_data_pack_t));
    //prt(info, "cam[%d] abc--event len: %d ",index, s_len);

    //////////////////////////////////////////////////////////////////////////////////////

    unsigned char one_flag = 0;
    unsigned char five_flag = 0;
    int average_speed = 0;
    int head_tm = 0;
    int ocuppy = 0;

    real_test_data[index].pack_head.version = 0x01;
    real_test_data[index].pack_head.prottype = 0x10;
    real_test_data[index].pack_head.objnumber = htons(index);
    real_test_data[index].pack_head.objtype = htons(REALTESTDATA);
    real_test_data[index].pack_head.objlen = htonl(sizeof(mRealTestInfo));

    pthread_cleanup_push(my_mutex_clean, &ex_static_info.real_test_lock[index]);
    pthread_mutex_lock(&ex_static_info.real_test_lock[index]);
    if (ex_static_info.real_test_updated[index] > 0)
    {
        if ((ex_static_info.real_test_updated[index] & (unsigned char)EM_ONE_MINUTE) > 0)
        {
            one_flag = 1;
        }

        if ((ex_static_info.real_test_updated[index] & (unsigned char)EM_FIVE_MINUTE) > 0)
        {
            five_flag = 1;
        }
    }

    for (int i = 0; i < DETECTLANENUMMAX; i++)
    {
        real_test_data[index].real_test_info.lane[i].Vehnum = htonl(ex_static_info.real_test_info[index].lane[i].Vehnum);
        real_test_data[index].real_test_info.lane[i].lagerVehnum = htonl(ex_static_info.real_test_info[index].lane[i].lagerVehnum);
        real_test_data[index].real_test_info.lane[i].smallVehnum = htonl(ex_static_info.real_test_info[index].lane[i].smallVehnum);
        real_test_data[index].real_test_info.lane[i].speed = htonl(ex_static_info.real_test_info[index].lane[i].speed);

        if (1 == one_flag)
        {
            real_test_data[index].real_test_info.lane[i].queueLength = htons(ex_static_info.real_test_one[index].lane[i].queueLength);
        }

        if (1 == five_flag)
        {
            //prt(info, "stay time: %d head tm: %d", ex_static_info.real_test_five[index].lane[i].share, ex_static_info.real_test_five[index].lane[i].timedist);
            ocuppy = ex_static_info.real_test_five[index].lane[i].share / (300 * 10);
            if (ocuppy > 100)
            {   //大于100%
                srand((unsigned)time(NULL));
                ocuppy = rand() % 11 + 85; //85~~95
            }

            if (ex_static_info.real_test_five[index].lane[i].Vehnum > 1)
            {
                average_speed = ex_static_info.real_test_five[index].lane[i].aveSpeed / ex_static_info.real_test_five[index].lane[i].Vehnum;
                head_tm = ex_static_info.real_test_five[index].lane[i].timedist / (ex_static_info.real_test_five[index].lane[i].Vehnum - 1);
            }

            if (ex_static_info.real_test_five[index].lane[i].Vehnum < 2)
            {   //0或1
                if (0 == head_tm)
                {
                    head_tm = 300; //5分钟
                }
            }

            if (ex_static_info.real_test_five[index].lane[i].Vehnum > 0)
            {
                if (0 == ocuppy)
                {
                    srand((unsigned)time(NULL));
                    ocuppy = rand() % 4 + 2; //2~5
                }

                if (0 == average_speed)
                {
                    srand((unsigned)time(NULL));
                    average_speed = rand() % 6 + 3; //3~8
                }

                if (ex_static_info.real_test_five[index].lane[i].Vehnum > 2)
                {
                    if (0 == head_tm)
                    {
                        srand((unsigned)time(NULL));
                        head_tm = 300 * (rand() % 3 + 1) / 100; //取(统计周期*0.01和统计周期*0.03)的随机值
                        if (0 == head_tm)
                        {
                            head_tm = 1; //获取1
                        }
                    }
                }
            }

            //prt(info, "ocuppy: %d head_tm: %d", ocuppy, head_tm);
            real_test_data[index].real_test_info.lane[i].aveSpeed = htonl(average_speed);
            real_test_data[index].real_test_info.lane[i].timedist = htonl(head_tm);
            real_test_data[index].real_test_info.lane[i].share = htonl(ocuppy);
        }
    }
    real_test_data[index].real_test_info.laneNum = get_lane_num(index);

    for (int mm = 0; mm < TEST_PERSON_AREA_SIZE; mm++)
    {
        real_test_data[index].real_test_info.person[mm].id = ex_static_info.real_test_info[index].person[mm].id;
        real_test_data[index].real_test_info.person[mm].time = htonl(ex_static_info.real_test_info[index].person[mm].time);
    }

    //prt(info, "cam[%d] 2--SendDataToClient start", index);
    SendDataToClient(p_data->fd, (char *)&real_test_data[index], sizeof(real_test_data_pack_t));
    //prt(info, "cam[%d] 2--SendDataToClient end", index);

    if (1 == one_flag)
    {
        memset(&ex_static_info.real_test_one[index], 0, sizeof(mRealTestInfo));
        ex_static_info.real_test_updated[index] &= ~((unsigned char)EM_ONE_MINUTE);
    }

    if (1 == five_flag)
    {
        memset(&ex_static_info.real_test_five[index], 0, sizeof(mRealTestInfo));
        ex_static_info.real_test_updated[index] &= ~((unsigned char)EM_FIVE_MINUTE);
    }

    pthread_mutex_unlock(&ex_static_info.real_test_lock[index]);
    pthread_cleanup_pop(0);

    //prt(info, "cam[%d] end...............", index);
}

void client_output(int index)
{
    list_operate_node_all(ip_list[index], (p_func)client_send_udp, (void *)&index);
}

void cmd_output(int index, unsigned short cmd_no)
{
    if ( cam_info[index].cmd_fd > 0 ) {
        // list_operate_node_cmd(ip_list[index], (p_func)send_cmd_status, (void *)&index);
        send_cmd_status(index, cmd_no);
        cam_info[index].cmd_fd = 0;
    }
}

#include "csocket.h"
void add_client(char *ip, int index, int sfd)
{
    m_client_ip_data *p_tmp;
    list_node_alloc_tail(ip_list[index]);
    p_tmp = (m_client_ip_data *)ip_list[index]->tail->data;
    p_tmp->camera_index = index;
    memcpy(p_tmp->ip, ip, 16);
    p_tmp->fd = sfd;

    prt(info, "add client :%s, index %d  ", p_tmp->ip, index);
    //prt(info,"total num of clients is %d",ip_list[index]->number);
}

int del_client(char *ip, int index, int sfd)
{
    int ret = -1;
    m_client_ip_data tmp_data;
    tmp_data.camera_index = index;
    tmp_data.fd = sfd;
    memcpy(tmp_data.ip, ip, 16);
    if (index > 0)
    {
        list_node_seek(ip_list[index], (void *)&tmp_data);
        ret = list_node_del_cur(ip_list[index]);
    }
    else
    {
        for (int i = 0; i < CAM_MAX; i++)
        {
            if (list_node_seek(ip_list[i], (void *)&tmp_data) >= 0)
            {
                ret = list_node_del_cur(ip_list[i]);
                break;
            }
        }
    }

    return ret;
}

int client_info_match(void *ori_info, void *new_info)
{
    m_client_ip_data *p1 = (m_client_ip_data *)ori_info;
    m_client_ip_data *p2 = (m_client_ip_data *)new_info;

    //if(!memcmp(p1->ip,p2->ip,16)){
    if (p1->fd == p2->fd)
    {
        if (p1->camera_index == p2->camera_index)
            return CLIENT_CAM_INDEX_SAME;
        else
            return CLIENT_IP_SAME;
    }
    return -1;
}

enum
{
    CHANGE_NOTHING,
    CHANGE_ALG,
    CHANGE_SIG_IP,
    CHANGE_CAM_RESET,
    CHANGE_ADJUST_IP,
    // CHANGE_PROTOCOL,
    CHANGE_STATIS,
    CHANGE_BASE,
    CHANGE_NTP,
    CHANGE_CAM_DEL,
    CHANG_CAM_OPEN,
    CHANG_DELAY_CAM_RESET,
    NOTIHNG_SAVE = 999
};
int client_cmd_flag = CHANGE_NOTHING;
void sync_obj(unsigned char *p_obj, int class_type, int index)
{
    int pos = 0;
    char filename[FILE_NAME_LENGTH];
    int len = get_obj_len(class_type);
    /*
        if(class_type==CLASS_mDetectDeviceConfig){
                pos=0;
        }else{
                pos=index*len;
        }
        */
    char *p_dst = NULL;
    switch (class_type)
    {
    case CLASS_mBaseInfo:
        p_dst = (char *)&g_ivddevsets;
        break;
    case CLASS_mNetworkInfo:
        p_dst = (char *)&g_netInfo;
        break;
    case CLASS_mAlgInfo:
    {
        p_dst = (char *)&g_camdetect[index];

        if (!g_cam_num.exist[index])
            g_cam_num.cam_num++;
        g_cam_num.exist[index] = 1;
    }
    break;
    case CLASS_mStatiscInfo:
        p_dst = (char *)&g_statisset;
        break;
    case CLASS_mChangeTIME:
        p_dst = (char *)&g_timestatus;
        break;
    case CLASS_mSerialInfo:
        p_dst = (char *)&g_rs485;
        break;
    /*
        case CLASS_mProtocolInfo:
            {
                p_dst=(char *)&g_protocol;

                pthread_mutex_lock(&g_all_lock.proto_lock);
                memcpy(p_dst,p_obj,len);
                pthread_mutex_unlock(&g_all_lock.proto_lock);
                return;
            }
            break;
         */
    case CLASS_mNTP:
    {
        p_dst = (char *)&g_ivdntp;
    }
    break;
    case CLASS_mCameraDelete:
    {
        if (g_cam_num.exist[index])
            g_cam_num.cam_num--;
        g_cam_num.exist[index] = 0;
    }
    break;
    case CLASS_mPersonAreaTimes:
    {
        p_dst = (char *)g_personplaninfo[index];
    }
    break;
    case CLASS_mEventInfo:
    {
        p_dst = (char *)&events_info[index];
    }
    break;
    default:
        prt(info, "unsupported save class %d", class_type);
        break;
    }

    if (p_dst != NULL)
        memcpy(p_dst, p_obj, len);
}

void handle_change(unsigned char *p_obj, int class_type, int index)
{
    char *p_old_obj = NULL;
    switch (class_type)
    {
    case CLASS_mBaseInfo:
    {
        prt_log_leve = ((IVDDevSets *)p_obj)->loglevel;
        client_cmd_flag = CHANGE_BASE;
    }
    break;
    case CLASS_mNetworkInfo:
    {
        IVDNetInfo *info = (IVDNetInfo *)p_obj;
        if (change_sig_ip(info->strIpaddrIO, info->strPortIO, info->strIpaddr2, info->strPort))
        {
            client_cmd_flag = CHANGE_SIG_IP;
        }

        if (set_ip_dns(info->strIpaddr, info->strNetmask, info->strDNS1, info->strDNS2, info->strGateway) > 0)
        {
            client_cmd_flag = CHANGE_ADJUST_IP;
        }
    }
    break;
    case CLASS_mDate:
    {
        struct tm _tm;
        struct timeval tv;
        time_t timep;

        _tm.tm_sec = ((TimeSetInterface *)p_obj)->second;
        _tm.tm_min = ((TimeSetInterface *)p_obj)->minute;
        _tm.tm_hour = ((TimeSetInterface *)p_obj)->hour;
        _tm.tm_mday = ((TimeSetInterface *)p_obj)->date;
        _tm.tm_mon = ((TimeSetInterface *)p_obj)->month - 1;
        _tm.tm_year = ((TimeSetInterface *)p_obj)->year - 1900;

        timep = mktime(&_tm);
        tv.tv_sec = timep;
        tv.tv_usec = 0;
        if (settimeofday(&tv, (struct timezone *)0) < 0)
        {
            prt(info, "Set system datatime error!/n");
        }

        char cmd[50] = {0};
        sprintf(cmd, "/sbin/hwclock -w");
        system(cmd);
    }
    break;
    case CLASS_mStatiscInfo:
    {
        //radar_realtime_result_info.time = ((IVDStatisSets*)p_obj)->period * 1000 * 1000;
        client_cmd_flag = CHANGE_STATIS;
    }
    break;
    case CLASS_mSerialInfo:
    {
        set_serial(((RS485CONFIG *)p_obj)->uartNo, ((RS485CONFIG *)p_obj)->buadrate, ((RS485CONFIG *)p_obj)->databit, ((RS485CONFIG *)p_obj)->stopbit, ((RS485CONFIG *)p_obj)->checkbit);
    }
    break;
    case CLASS_mAlgInfo:
    {
        if (cam_info[index].curr_stat == CAM_COMMAND || cam_info[index].curr_stat == CAM_OPENING){
            client_cmd_flag = NOTIHNG_SAVE;
            break;
        }

     
        if (0 == g_cam_num.exist[index] && 0 == cam_info[index].open_alg)
        {
            client_cmd_flag = CHANG_CAM_OPEN;
            if (g_cam_num.cam_num >= CAMERA_RUN_MAX)
            {   //限制相机启动个数
                client_cmd_flag = NOTIHNG_SAVE;
            }
        }
        else
        {  

            client_cmd_flag = CHANGE_CAM_RESET;

            //相机已经启动
            /*
                      bool ret = set_camera_network(index, ((CamDetectParam*)p_obj)->other.camIp, \
                        ((CamDetectParam*)p_obj)->other.username, (char *)((CamDetectParam*)p_obj)->other.passwd, \
                        ((CamDetectParam*)p_obj)->other.camPort);
                       */
            //client_cmd_flag = CHANG_DELAY_CAM_RESET; //wait receive event comment to reset
            // if (0 == strcmp(g_camdetect[index].other.rtsppath, ((CamDetectParam *)p_obj)->other.rtsppath))
            // {
            //     client_cmd_flag = CHANGE_ALG;
            // }
            // else
            // {
            //     client_cmd_flag = CHANGE_CAM_RESET;
            // }

            /*
                      if(ret){
                        //camera_ctrl(CAMERA_CONTROL_RESET, index, 0, NULL);
                        client_cmd_flag = CHANGE_CAM_IP;
                      }else {
                         client_cmd_flag=CHANGE_ALG;
                      }
                      */
        }
    }
    break;
    /*
            case CLASS_mProtocolInfo:
                {
                   int type = ((mThirProtocol*)p_obj)->type;

                   if (type == g_protocol.type)
                    return;


                   client_cmd_flag = CHANGE_PROTOCOL;

                }
                break;
                */
    case CLASS_mNTP:
    {
        client_cmd_flag = CHANGE_NTP;
    }
    break;
    case CLASS_mCameraDelete:
    {
        if (cam_info[index].curr_stat == CAM_COMMAND || cam_info[index].curr_stat == CAM_OPENING){
            client_cmd_flag = NOTIHNG_SAVE;
            break;
        }
        
        client_cmd_flag = CHANGE_CAM_DEL;
    }
    break;
    case CLASS_mEventInfo:
    {
        if (cam_info[index].curr_stat == CAM_COMMAND || cam_info[index].curr_stat == CAM_OPENING){
            client_cmd_flag = NOTIHNG_SAVE;
            break;
        }
        
        client_cmd_flag = CHANGE_CAM_RESET;
    }
    break;

    default:
        break;
    }
}
int handle_buffer(int s_index, unsigned char *buf, int len)
{
    int type;
    int length;
    int ret = -1;
    int match_ret = -1;
    unsigned char old_pro = 0;
    ///////
    int reply_type = 0;
    int class_type = 0;
    int class_len = 0;
    int flg = NOOP_FLAG;
    unsigned char *p_obj = NULL;
    ///////
    mCommand *cmd_p;
    cmd_p = (mCommand *)buf;
    net_decode_obj((unsigned char *)cmd_p, CLASS_mCommand, 0);
    type = cmd_p->objtype;
    int cmd_len = sizeof(mCommand);
    int index = cmd_p->objnumber;

    if (index >= CAM_MAX || cmd_p->objlen < 0)
    {
        printf("get cmd type %x, length %d,index %d data len: %d \n", type, len, cmd_p->objnumber, cmd_p->objlen);
        return -1;
    }

    static m_client_ip_data tmp_data;
    m_client_ip_data *current_data;
    tmp_data.camera_index = index;
    memcpy(tmp_data.ip, g_sock_info.ip_addr[s_index], 16);
    tmp_data.fd = g_sock_info.client_sockfd[s_index];

    if (type == STARTREALDATA)
    {   //||type==HEART||type==SHUTDOWN) {
        if ((match_ret = list_node_seek(ip_list[index], (void *)&tmp_data)) >= CLIENT_IP_SAME)
        {
            if (match_ret != CLIENT_CAM_INDEX_SAME)
            {
                current_data = (m_client_ip_data *)list_get_current_data(ip_list[index]);
                current_data->camera_index = index;
                printf("ip %s instead index %d \n", current_data->ip, index);
            } //else{
            //prt(info,":%s index change to %d  ",current_data->ip,tmp_data.camera_index);
            //prt(info,"same index " );
            // }
        }
        else
        {
            print("get index %d \n",tmp_data.camera_index);
            add_client(tmp_data.ip, index, g_sock_info.client_sockfd[s_index]);
        }

        pthread_mutex_lock(&ex_static_info.lock[index]);
        memset(&ex_static_info.pre_car_num[index], 0, sizeof(unsigned int) * DETECTLANENUMMAX * 2);
        memset(&ex_static_info.car_num[index], 0, sizeof(unsigned int) * DETECTLANENUMMAX * 2);
        ex_static_info.reset_flag[index] = 1;
        pthread_mutex_unlock(&ex_static_info.lock[index]);

        pthread_mutex_lock(&ex_static_info.real_test_lock[index]);
        ex_static_info.real_test_updated[index] = 0;
        real_test_run_flag[index] = (unsigned char)EM_RESET;
        memset(&ex_static_info.real_test_info[index], 0, sizeof(mRealTestInfo));
        memset(&ex_static_info.real_test_one[index], 0, sizeof(mRealTestInfo));
        memset(&ex_static_info.real_test_five[index], 0, sizeof(mRealTestInfo));
        memset(&real_test_data[index], 0, sizeof(real_test_data_pack_t));
        pthread_mutex_unlock(&ex_static_info.real_test_lock[index]);
        // if (cam_info[index].vtype == FILE_TYPE)
        //     cam_info[index].file_start = 1;
        //prt(info, "realtime data reset");
        return 0;
    }

    switch (type)
    {
    case GETBASEPARAM:
    {
        class_type = CLASS_mBaseInfo;
        p_obj = (unsigned char *)&g_ivddevsets;
        reply_type = REPBASEPARAM;
        flg = GET_FLAG;
    }
    break;
    case SETBASEPARAM:
    {
        old_pro = g_ivddevsets.pro_type;
        class_type = CLASS_mBaseInfo;
        flg = SET_FLAG;
    }
    break;
    case GETSYSPARAM:
    {
        class_type = CLASS_mSysInfo;
        p_obj = (unsigned char *)&g_sysInfo;
        reply_type = REPSYSPARAM;
        flg = GET_FLAG;
    }
    break;
    case GETNETWORKPARAM:
    {
        class_type = CLASS_mNetworkInfo;
        p_obj = (unsigned char *)&g_netInfo;
        reply_type = REPNETWORKPARAM;
        flg = GET_FLAG;
    }
    break;
    case SETNETWORKPARAM:
    {
        class_type = CLASS_mNetworkInfo;
        flg = SET_FLAG;
    }
    break;
    case GETALGSPARAM:
    {
        class_type = CLASS_mAlgInfo;
        p_obj = (unsigned char *)&g_camdetect[index];
        reply_type = REPALGPARAM;
        flg = GET_FLAG;
    }
    break;
    case SETALGPARAM:
    {
        class_type = CLASS_mAlgInfo;
        flg = SET_FLAG;
        prt(info, "CLASS_mAlgInfo");
    }
    break;
    case SHUTDOWN:
    {
        ret = del_client(tmp_data.ip, index, g_sock_info.client_sockfd[s_index]);

        pthread_mutex_lock(&ex_static_info.real_test_lock[index]);
        ex_static_info.real_test_updated[index] = 0;
        real_test_run_flag[index] = (unsigned char)EM_NOT;
        memset(&ex_static_info.real_test_info[index], 0, sizeof(mRealTestInfo));
        memset(&ex_static_info.real_test_one[index], 0, sizeof(mRealTestInfo));
        memset(&ex_static_info.real_test_five[index], 0, sizeof(mRealTestInfo));
        pthread_mutex_unlock(&ex_static_info.real_test_lock[index]);
        if (cam_info[index].vtype == FILE_TYPE)
            cam_info[index].file_start = 0;

#if 0 //关闭client socket
        if (!ret) {
            clear_socket(s_index);
            prt(info, "command shutdown");
        }
#endif
        cmd_len = 0;
        flg = NOOP_FLAG;
        ret = -1;
    }
    break;
    case GETDATEPARAM:
    {
        class_type = CLASS_mDate;
        struct tm *ptr;
        time_t lt;
        lt = time(NULL);
        ptr = localtime(&lt);
        g_timeset.year = ptr->tm_year + 1900;
        g_timeset.month = ptr->tm_mon + 1;
        g_timeset.date = ptr->tm_mday;
        g_timeset.hour = ptr->tm_hour;
        g_timeset.minute = ptr->tm_min;
        g_timeset.second = ptr->tm_sec;
        p_obj = (unsigned char *)&g_timeset;
        reply_type = REPDATEPARAM;
        flg = GET_FLAG;
    }
    break;
    case SETDATEPARAM:
    {
        class_type = CLASS_mDate;
        flg = SET_FLAG;
    }
    break;
    case GETSTATISPARAM:
    {
        p_obj = (unsigned char *)&g_statisset;
        reply_type = REPSTATISPARAM;
        flg = GET_FLAG;
        class_type = CLASS_mStatiscInfo;
    }
    break;
    case SETSTATISPARAM:
    {
        class_type = CLASS_mStatiscInfo;
        flg = SET_FLAG;
    }
    break;
    case GETCHTIMEPARAM:
    {
        class_type = CLASS_mChangeTIME;
        p_obj = (unsigned char *)&g_timestatus;
        reply_type = REPCHTIMEPARAM;
        flg = GET_FLAG;
    }
    break;
    case SETCHTIMEPARAM:
    {
        class_type = CLASS_mChangeTIME;
        flg = SET_FLAG;
    }
    break;
    case GETSERIALPARAM:
    {
        class_type = CLASS_mSerialInfo;
        p_obj = (unsigned char *)&g_rs485;
        reply_type = REPSERIALPARAM;
        flg = GET_FLAG;
    }
    break;
    case SETSERIALPARAM:
    {
        class_type = CLASS_mSerialInfo;
        flg = SET_FLAG;
    }
    break;
    /*
            case GETPROTOCOLPARAM:
            {
                class_type=CLASS_mProtocolInfo;
                p_obj=(unsigned char *)&g_protocol;
                reply_type = REPPROTOCOLPARAM;
                flg=GET_FLAG;
            }
                break;
            case SETPROTOCOLPARAM:
             {
                old_pro = g_protocol.type;
                class_type=CLASS_mProtocolInfo;
                flg=SET_FLAG;
             }
                break;
             */
    case GETNTPPARAM:
    {
        class_type = CLASS_mNTP;
        p_obj = (unsigned char *)&g_ivdntp;
        reply_type = REPNTPPARAM;
        flg = GET_FLAG;
    }
    break;
    case SETNTPPARAM:
    {
        class_type = CLASS_mNTP;
        flg = SET_FLAG;
    }
    break;
    case GETCAMERANUMPARAM:
    {
        class_type = CLASS_mCameraStatus;
        p_obj = (unsigned char *)&g_cam_num;
        reply_type = REPCAMERANUMPARAM;
        flg = GET_FLAG;
    }
    break;
    case DELCAMERANUMPARAM:
    {
        class_type = CLASS_mCameraDelete;
        flg = SET_FLAG;
    }
    break;
    case GETPERSONARETIM:
    {
        class_type = CLASS_mPersonAreaTimes;
        p_obj = (unsigned char *)g_personplaninfo[index];
        reply_type = REPPERSONARETIM;
        flg = GET_FLAG;
    }
    break;
    case SETPERSONARETIM:
    {
        class_type = CLASS_mPersonAreaTimes;
        flg = SET_FLAG;
    }
    break;
    case SETEVENTPARAM:
    {
        class_type = CLASS_mEventInfo;
        flg = SET_FLAG;
    }
    break;
    case GETEVENTPARAM:
    {
        class_type = CLASS_mEventInfo;
        p_obj = (unsigned char *)&events_info[index];
        reply_type = REPEVENTPARAM;
        flg = GET_FLAG;
    }
    break;
    case SETDEVREBOOT:
    {
        reboot_cmd();
        return 0;
    }
    break;
    case FILE_PLAY_START:
    {
        if (cam_info[index].vtype == FILE_TYPE)
        {
            cam_info[index].file_start = 1;
            pthread_mutex_lock(&cam_info[index].file_lock);
            cam_info[index].file_replay = 1;
            pthread_mutex_unlock(&cam_info[index].file_lock);
            clear_yuv_item(index);
        }
    }
    break;
    default:
        break;
    }

    class_len = get_obj_len(class_type);
    //prt(info, " get_obj_len: %d", class_len);
    switch (flg)
    {
    case GET_FLAG:
    {
        ret = prepare_pkt(buf, cmd_len, reply_type, class_type, class_len, p_obj);
        net_decode_obj((unsigned char *)cmd_p, CLASS_mCommand, 1);
    }
    break;
    case SET_FLAG:
    {
        if (cmd_p->objlen != class_len)
        {
            prt(info, "error-------->receive data len: %d  class len: %d", cmd_p->objlen, class_len);
            return -1;
        }
        net_decode_obj((unsigned char *)(buf + cmd_len), class_type, 0);

        client_cmd_flag = CHANGE_NOTHING;
        handle_change(buf + cmd_len, class_type, index);

        if (client_cmd_flag != int(NOTIHNG_SAVE))
        {
            sync_obj((unsigned char *)(buf + cmd_len), class_type, index);
            save_obj((unsigned char *)(buf + cmd_len), class_type, index);
        }

        switch (client_cmd_flag)
        {
            case CHANG_DELAY_CAM_RESET:
            {
                ret = pack_cmd_status(index, SETALGPARAM, 1, buf);
            }
                break;
            case NOTIHNG_SAVE:
            {
                switch (class_type)
                {
                    case CLASS_mAlgInfo:
                        ret = pack_cmd_status(index, SETALGPARAM, 0, buf);
                        break;
                    case CLASS_mCameraDelete:
                        ret = pack_cmd_status(index, DELCAMERANUMPARAM, 0, buf);
                        break;
                    case CLASS_mEventInfo:
                        ret = pack_cmd_status(index, SETEVENTPARAM, 0, buf);
                        break;
                    default:
                        break;
                }
            }
                break;
            case CHANGE_BASE:
            {   // PROTO_PRIVATE 和PROTO_HAIXIN 一样不用做切换
                if (old_pro != g_ivddevsets.pro_type && (old_pro + g_ivddevsets.pro_type) != PROTO_HAIXIN)
                {
                    protocol_change(g_ivddevsets.pro_type, old_pro);
                    reset_sig_machine();
                }
            }
            break;
            case CHANGE_SIG_IP:
                reset_sig_machine();
                break;
            case CHANGE_CAM_RESET:
            {
                prt(info, "CHANGE_CAM_RESET1");
                cam_info[index].cmd_fd = g_sock_info.client_sockfd[s_index];
                camera_set_curr_status(index, CAM_COMMAND);
                camera_ctrl(CAMERA_CONTROL_RESET, index, 0, NULL);
            }
            break;
            case CHANGE_ALG:
            {
                prt(info, "CHANGE_CAM_RESET2---alg");
                cam_info[index].cmd_fd = g_sock_info.client_sockfd[s_index];
                //camera_ctrl(CAMERA_CONTROL_RESET_ALG,index,0,NULL);
                camera_set_curr_status(index, CAM_COMMAND);
                camera_ctrl(CAMERA_CONTROL_RESET, index, 0, NULL);
            }
            break;
            case CHANGE_ADJUST_IP:
                kill_process();
                break;
            /*
                        case CHANGE_PROTOCOL:
                        {
                            if (old_pro != g_protocol.type) {
                                protocol_change(g_protocol.type, old_pro);
                                reset_sig_machine();
                            }
                        }
                            break;
                        */
            case CHANGE_STATIS:
                g_cycle_statis_time = g_statisset.period;
                break;
            case CHANGE_NTP:
            {
                pthread_t pid;
                if (strlen(g_ivdntp.ipaddr) > 0 && g_ivdntp.cycle > 0)
                    pthread_create(&pid, NULL, start_ntpclient, NULL);
                // start_ntpclient(g_ivdntp.ipaddr, g_ivdntp.cycle);//g_ivdntp.cycle*60*60);
            }
            break;
            case CHANGE_CAM_DEL:
            {
                prt(info, "CHANGE_CAM_RESET2---del");
                camera_set_curr_status(index, CAM_COMMAND);
                cam_info[index].cmd_fd = g_sock_info.client_sockfd[s_index];
                camera_ctrl(CAMERA_CONTROL_CLOSE, index, 0, NULL);
            }
            break;
            case CHANG_CAM_OPEN:
            {
                prt(info, "CHANGE_CAM_RESET2---open");
                cam_info[index].cmd_fd = g_sock_info.client_sockfd[s_index];
                camera_set_curr_status(index, CAM_COMMAND);
                camera_ctrl(CAMERA_CONTROL_OPEN, index, 0, NULL);
            }
            break;
            default:
                break;
            }
        }
            break;
        default:
            break;
    }

    //prt(info,"ret length %d",ret);
    return ret;
}

int get_mac(char *mac, int len_limit) //返回值是实际写入char * mac的字符个数（不包括'\0'）
{
    struct ifreq st_ifr;
    int sock;

    if ((sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        perror("socket");
        return -1;
    }
    strcpy(st_ifr.ifr_name, "enp1s0");

    if (ioctl(sock, SIOCGIFHWADDR, &st_ifr) < 0)
    {
        perror("ioctl");
        return -1;
    }

    close(sock);

    return snprintf(mac, len_limit, "%02X:%02X:%02X:%02X:%02X:%02X", (unsigned char)st_ifr.ifr_hwaddr.sa_data[0], (unsigned char)st_ifr.ifr_hwaddr.sa_data[1], (unsigned char)st_ifr.ifr_hwaddr.sa_data[2], (unsigned char)st_ifr.ifr_hwaddr.sa_data[3], (unsigned char)st_ifr.ifr_hwaddr.sa_data[4], (unsigned char)st_ifr.ifr_hwaddr.sa_data[5]);
}

void init_config()
{
    for (int i = 0; i < CAM_MAX; i++)
    {
        ip_list[i] = new_list(sizeof(m_client_ip_data), (void *)client_info_match);
    }

    //load cfg
    ReadBaseParam(&g_ivddevsets);
    prt_log_leve = g_ivddevsets.loglevel;
    //
    ReadStatisParam(&g_statisset);
    ReadSysParam(&g_sysInfo);
    ReadNetWorkParam(&g_netInfo);
    ReadChTimeParam(&g_timestatus);
    ReadSerialParam(&g_rs485);
    ReadNtpParam(&g_ivdntp);
    ReadCamStatusParam(&g_cam_num);

    for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
    {
        if (g_cam_num.exist[i])
        {
            ReadCamDetectParam(&g_camdetect[i], i);
            ReadPersonAreas((mPersonPlanInfo *)g_personplaninfo[i], i);
            j++;
        }
    }

    //g_netInfo.maxConn = 5; //初始值

    if (strlen(g_netInfo.strMac) < 1)
    {
        get_mac(g_netInfo.strMac, 20);
    }

    prt(info, "checkaddr: %s loglevel: %d", g_ivddevsets.checkaddr, g_ivddevsets.loglevel);
    g_cycle_statis_time = g_statisset.period;

    for (int i = 0; i < CAM_MAX; i++)
    {
        ReadEventParam(&events_info[i], i);
    }
}

void init_camera()
{
    for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num && j < CAMERA_RUN_MAX; i++)
    {
        if (1 == g_cam_num.exist[i])
        {
            // if (!camera_set_open_one(i))
            //     break;

            camera_open(i, (char *)g_camdetect[i].other.camIp,
                        (int)g_camdetect[i].other.camPort,
                        (char *)g_camdetect[i].other.username,
                        (char *)g_camdetect[i].other.passwd, g_camdetect[i].other.videotype); //打开相机
            j++;
        }
    }
}

IVDTimeStatu get_mDetectTime()
{
    IVDTimeStatu tmp = g_timestatus;
    return tmp;
}

mCamDetectParam *get_mCamDetectParam(int index)
{
    return &g_camdetect[index];
    //return &g_cam_cfg[index].det_param;
}
mCamParam *get_mCamParam(int index)
{
    return &g_cam_cfg[index].cam_param;
}

int get_cam_status(int index)
{
    return cam_info[index].open_flg;
    //return g_dev_cfg.cam_info[index].camstatus;
}
int get_cam_id(int index)
{
    //return  g_cam_cfg[index].cam_param.camattr.camID;
    return g_camdetect[index].other.camerId;
}

int get_cam_direction(int index)
{
    return g_camdetect[index].other.directio;
    //return  g_dev_cfg.cam_info[index].camdirect;
}

int get_cam_location(int index)
{
    return g_camdetect[index].other.camdirection;
}
int get_dev_id()
{
    //return  g_dev_cfg.deviceID;
    return atoi(g_ivddevsets.devUserNo);
}
void get_sig_ip(char *des)
{
    strcpy(des, (char *)g_netInfo.strIpaddrIO);
    //strcpy( des,(char *) g_cam_cfg[0].cam_param.camattr.signalIp);
}
int get_sig_port()
{
    return g_netInfo.strPortIO;
    //return g_cam_cfg[0].cam_param.camattr.signalport;
    //  return 5000;
}

int get_person_area_id(int cam_index, int area_index)
{
    //    prt(info,"cam %d area  %d id: %d",cam_index,area_index, g_camdetect[cam_index].personarea.area[area_index].id);
    return g_camdetect[cam_index].personarea.area[area_index].id;
}

int get_area_count(int cam_index)
{
    return g_camdetect[cam_index].personarea.num;
}

int get_lane_num(int index)
{
    //return g_cam_cfg[index].det_param.detectlane.lanenum;
    return g_camdetect[index].detectlane.lanenum;
}

int get_lane_index(int index, int lane_i)
{
    //return g_cam_cfg[index].cam_param.channelcoil[lane_i].number;
    return g_camdetect[index].detectlane.virtuallane[lane_i].landID;
}

void get_udp_info(char *ip, unsigned short *port)
{
    memcpy(ip, g_netInfo.strIpaddrIO, IPADDRMAX);
    *port = g_netInfo.strPortIO;
}

// 7- xi   --3
// 1 - bei  --0
// 5 - nan  --2
// 3 - dong  -- 1
int get_direction(int index)
{

    int ret = 0x00;
    switch (get_cam_location(index))
    {
    case Z_NORTH:
        ret = 0x0;
        break;

    case Z_EAST:
        ret = 0x1;
        break;

    case Z_SOUTH:
        ret = 0x2;
        break;

    case Z_WEST:
        ret = 0x3;
        break;
    default:
        ret = 0x0;
        break;
    }

    return ret;
}
