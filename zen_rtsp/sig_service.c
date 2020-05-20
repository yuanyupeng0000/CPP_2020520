
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/wait.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include <netinet/in.h>
#include "client_net.h"
#include "sig_service.h"
#include "csocket.h"
#include "common.h"
#include "g_define.h"
#include "udp_network.h"
#include "fvdmysql/fvdmysql.h"
#include "g_fun.h"
#include "camera_service.h"
#include "tcp_server.h"
#include "fvdconfig.h"
#include "ws_service.h"
#include "cam_event.h"
#include "ptserver/non_motor_server.h"

#define BUF_MAX 500
#define LIST_IN_OUT_MAX 50
#define MAX_MESSAGE_TEXT 512
#define CONTINUE_TIME 2          //閹镐胶鐢婚弮鍫曟？/s
#define NO_PERSON_THROUGH_TIME 2 //濞屸剝婀佹禍铏光敍鏉╁洨娈戦弮鍫曟？/s
#define LANE_QUEUE_LEN 75

//////////////////////////////////////
extern g_all_lock_t g_all_lock;
extern mThirProtocol g_protocol;
//////////////////////////////////////in out start
pthread_mutex_t yhl_mutex_lock;
pthread_mutex_t mutex_lock;
pthread_mutex_t mutex_in_out[CAM_MAX];
pthread_mutex_t jierui_mutex[CAM_MAX];
extern IVDStatisSets g_statisset;
extern IVDDevSets g_ivddevsets;
extern IVDNetInfo g_netInfo;
extern IVDNTP g_ivdntp;
extern IVDCAMERANUM g_cam_num;
extern mCamDetectParam g_camdetect[CAM_MAX];
extern EX_mRealStaticInfo ex_static_info;
//
extern m_camera_info cam_info[CAM_MAX];
//
extern mGlobalVAR g_member;

//////////////////////////////////////
unsigned int real_test_seconds[CAM_MAX] = {0};
unsigned int real_test_run_flag[CAM_MAX] = {0};
/////////////////////////////////////
int list_count[CAM_MAX] = {0};
int s_next_index[CAM_MAX] = {0};
int g_next_index[CAM_MAX] = {0};
unsigned char protocol_sel = 0;
unsigned int g_cycle_statis_time = 60;
unsigned char buf_gat920_in_out_result[BUF_MAX] = {0};
unsigned char buf_gat920_queue_result[BUF_MAX] = {0};
radar_rt_lane_car_in_out_info_t list_in_out[CAM_MAX][LIST_IN_OUT_MAX];
radar_cam_realtime_t radar_cam_realtime_data[CAM_MAX];
radar_car_in_out_status_t radar_in_out_status_920[CAM_MAX][MAX_LANE_NUM] = {0}; //by anger
radar_queue_frame_count_t radar_queue_count[MAX_LANE_NUM] = {0};
mPersonPlanInfo g_personplaninfo[CAM_MAX][PERSON_AREAS_MAX];
mRealTimePerson realtime_person[CAM_MAX];
mEventInfo events_info[CAM_MAX];
mRealEventInfo event_data[CAM_MAX];

//mPersonCheckData person_check_data[CAM_MAX][PERSON_AREAS_MAX];
camera_bus_t cam_bus[CAM_MAX];
///////////////////////////////////////////////////////////////////

m_timed_func_data gat920_car_in_out_result_info; //
m_timed_func_data gat920_car_queue_result_info;  // ---> 鏉烇箒绶犻幒鎺楁Е
m_timed_func_data gat920_jierui_realtime_result_info;//杰瑞
m_timed_func_data gat920_jierui_60s_result_info;//杰瑞

//////////////////////////////////////////////////////////////////
m_holder holder[CAM_MAX];
int sig_state = 0;
pthread_mutex_t sig_state_lock;
pthread_mutex_t sig_client_lock;
//////////////////////////////////////////////////////////////////
jierui_rt_lane_car_data_info jierui_lanes_data[CAM_MAX];
unsigned char gat920_jierui_data_buf[BUF_MAX] = {0};
jierui_60s_t gat920_jierui_60s_data_buf[BUF_MAX] = {0};
unsigned char gat920_jierui_60s_buf[BUF_MAX] = {0};
//////////////////////////////////////////////////////////////////

enum
{
    SIG_PRE_CONNECT = 1,
    SIG_CONNECTED,
    SIG_NULL
};
int sig_fd;
int sig_port;
char sig_ip[IP_LEN];
//鐞涘奔姹夊Λ鈧ù瀣╃瑢娣団冲娇
int sig_fd_pn;
int sig_port_pn;
//messate
//int msgid = -1;
struct msg_st
{
    long int msg_type;
    char text[BUFSIZ];
};
//////////////////////////////////////////////////////////////////

void init_server_lock()
{
    for (int i = 0; i < CAM_MAX; i++)
    {
        // pthread_mutex_init(&holder[i].sig_data_lock,NULL);
        pthread_mutex_init(&radar_cam_realtime_data[i].mutex_lock, NULL);
        pthread_mutex_init(&mutex_in_out[i], NULL);
        pthread_mutex_init(&jierui_mutex[i], NULL);
    }

    pthread_mutex_init(&mutex_lock, NULL);
    pthread_mutex_init(&yhl_mutex_lock, NULL);
    pthread_mutex_init(&sig_state_lock, NULL);
    pthread_mutex_init(&sig_client_lock, NULL);
}

//閹绘帒鍙嗛崗銉ㄦ簠閸戦缚婧呴惃鍕杽閺冭埖鏆熼幑锟
bool add_car_in_out_item(int index, radar_rt_lane_car_in_out_info_t *io_item)
{
    //lock
    pthread_cleanup_push(my_mutex_clean, &mutex_in_out[index]);
    pthread_mutex_lock(&mutex_in_out[index]);

    if (s_next_index[index] == g_next_index[index] && list_count[index] > 0)
    {   //閻滎垰鑸伴幓鎺戝弳
        pthread_mutex_unlock(&mutex_in_out[index]);
        return false;
    }

    memcpy(&list_in_out[index][s_next_index[index]], io_item, sizeof(radar_rt_lane_car_in_out_info_t));
    list_count[index]++;
    s_next_index[index]++;
    s_next_index[index] = s_next_index[index] % LIST_IN_OUT_MAX;

    pthread_mutex_unlock(&mutex_in_out[index]);
    pthread_cleanup_pop(0);
    return true;
    //unlock
}

//閼惧嘲褰囬崗銉ㄦ簠閸戦缚婧呴惃鍕杽閺冭埖鏆熼幑锟
bool get_car_in_out_item(int index, radar_rt_lane_car_in_out_info_t *io_item)
{

    pthread_mutex_lock(&mutex_in_out[index]);

    if (list_count[index] == 0)
    {
        pthread_mutex_unlock(&mutex_in_out[index]);
        return false;
    }

    memcpy(io_item, &list_in_out[index][g_next_index[index]], sizeof(radar_rt_lane_car_in_out_info_t));
    list_count[index]--;
    g_next_index[index]++;
    g_next_index[index] = g_next_index[index] % LIST_IN_OUT_MAX;

    pthread_mutex_unlock(&mutex_in_out[index]);

    return true;
}

///////////////////////////////////////////end in out

//jierui----------------------------------------------
unsigned char new_data_flag[CAM_MAX] = {0};
unsigned char new_data_60s_flag[CAM_MAX] = {0};
void add_jierui_car_item(int index, jierui_rt_lane_car_data_info_t *item)
{
    pthread_cleanup_push(my_mutex_clean, &jierui_mutex[index]);
    pthread_mutex_lock(&jierui_mutex[index]);
    memcpy(&jierui_lanes_data[index], item, sizeof(jierui_rt_lane_car_data_info_t));
    new_data_flag[index] = 1;
    pthread_mutex_unlock(&jierui_mutex[index]);
    pthread_cleanup_pop(0);
}

bool get_jierui_car_item(int index, jierui_rt_lane_car_data_info_t *item)
{
    bool ret = false;

    pthread_mutex_lock(&jierui_mutex[index]);
    if (new_data_flag[index] > 0) {
        memcpy(item, &jierui_lanes_data[index], sizeof(jierui_rt_lane_car_data_info_t));
        ret = true;
        new_data_flag[index] = 0;
    }
    pthread_mutex_unlock(&jierui_mutex[index]);

    return ret;
}

void add_jierui_60s_item(int index, jierui_60s_t *item)
{
    pthread_cleanup_push(my_mutex_clean, &jierui_mutex[index]);
    pthread_mutex_lock(&jierui_mutex[index]);

    for (int i = 0; i < MAX_LANE_NUM; i++){
        gat920_jierui_60s_data_buf[index].data.lane_data[i][0].exist_duration += item->data.lane_data[i][0].exist_duration;
        gat920_jierui_60s_data_buf[index].data.lane_data[i][0].pass_number += item->data.lane_data[i][0].pass_number;
        gat920_jierui_60s_data_buf[index].data.lane_data[i][0].speed_sum += item->data.lane_data[i][0].speed_sum;
        gat920_jierui_60s_data_buf[index].data.lane_data[i][0].veh_len_sum += item->data.lane_data[i][0].veh_len_sum;
        gat920_jierui_60s_data_buf[index].data.lane_data[i][0].head_time_sum += item->data.lane_data[i][0].head_time_sum;
        gat920_jierui_60s_data_buf[index].data.lane_data[i][0].car_a_sum += item->data.lane_data[i][0].car_a_sum;
        gat920_jierui_60s_data_buf[index].data.lane_data[i][0].car_b_sum += item->data.lane_data[i][0].car_b_sum;
        gat920_jierui_60s_data_buf[index].data.lane_data[i][0].car_c_sum += item->data.lane_data[i][0].car_c_sum;

        if (item->Lane_queue_len[i] > gat920_jierui_60s_data_buf[index].Lane_queue_len[i]) { //排队长度最大值
            gat920_jierui_60s_data_buf[index].Lane_queue_len[i] = item->Lane_queue_len[i];
        }

    }

    memset(item, 0, sizeof(jierui_60s_t));
    new_data_60s_flag[index] = 1;

    pthread_mutex_unlock(&jierui_mutex[index]);
    pthread_cleanup_pop(0);
}


bool get_jierui_60s_item(int index, jierui_60s_t *item)
{
    bool ret = false;

    pthread_mutex_lock(&jierui_mutex[index]);
    if (new_data_60s_flag[index] > 0) {
        memcpy(item, &gat920_jierui_60s_data_buf[index], sizeof(jierui_60s_t));
        memset(&gat920_jierui_60s_data_buf[index], 0, sizeof(jierui_60s_t));
        ret = true;
        new_data_60s_flag[index] = 0;
    }
    pthread_mutex_unlock(&jierui_mutex[index]);

    return ret;
}

//jierui--------------------------------------------end

unsigned char get_crc(unsigned char *buf, int sz)
{
    unsigned char crc = 0;
    for (int i = 0; i < sz; i++)
    {
        crc ^= *(buf + i);
    }
    return crc;
}
void get_outcar_info()
{
}
void get_time_string(char *buf, int len)
{
    time_t tt;
    tt = time(NULL);
    struct tm *t = localtime(&tt);

    sprintf(buf, "%d-%02d-%02d %02d:%02d:%02d", t->tm_year + 1900, t->tm_mon + 1, t->tm_mday, t->tm_hour, t->tm_min, t->tm_sec);
}
int get_year()
{
    time_t tt;
    tt = time(NULL);
    struct tm *t = localtime(&tt);
    return t->tm_year + 1900;
}
int get_year_tail()
{
    time_t tt;
    tt = time(NULL);
    struct tm *t = localtime(&tt);
    return (t->tm_year + 1900) % 2000;
}
int get_month()
{
    time_t tt;
    tt = time(NULL);
    struct tm *t = localtime(&tt);
    return t->tm_mon + 1;
}
int get_day()
{
    time_t tt;
    tt = time(NULL);
    struct tm *t = localtime(&tt);
    return t->tm_mday;
}
int get_hour()
{
    time_t tt;
    tt = time(NULL);
    struct tm *t = localtime(&tt);
    return t->tm_hour;
}
int get_min()
{
    time_t tt;
    tt = time(NULL);
    struct tm *t = localtime(&tt);
    return t->tm_min;
}
int get_sec()
{
    time_t tt;
    tt = time(NULL);
    struct tm *t = localtime(&tt);
    return t->tm_sec;
}

//pthread_mutex_t out_car_lock;
//pthread_mutex_t queue_lock;
//pthread_mutex_t flow_lock;

//m_sig_data channel_rst[CAM_NUM];

int sig_get_state()
{
    return sig_state;
}

void sig_set_state(int state)
{
    pthread_mutex_lock(&sig_state_lock);
    sig_state = state;
    pthread_mutex_unlock(&sig_state_lock);
}
extern void submit_unlock_sig_data(int index)
{
    //  int ori_state=0;
    //  pthread_mutex_lock(&holder[index].sig_data_lock);
    ////    ori_state=holder[index].traffic_rst.camera_state_change;
    //  memcpy(&holder[index].traffic_rst,p_channel_rst,sizeof(m_sig_data));
    //  if(p_channel_rst->camera_state_change==1){
    //      prt(info,"state change ");
    //      p_channel_rst->camera_state_change=0;
    //  }

    //  pthread_mutex_unlock(&holder[index].sig_data_lock);

    //  prt(info,"index %d,src num %d,dst num %d",index,p_channel_rst->lane_num,holder[index].traffic_rst.lane_num);
}
extern m_sig_data *get_locked_sig_data(int index)
{

    //pthread_mutex_lock(&holder[index].sig_data_lock);
    return &holder[index].traffic_rst;
}

long get_holder_last_time(int index)
{
    return holder[index].last_time;
}

int get_holder_status(int index)
{
    return holder[index].status;
}

void set_holder_status(int index, unsigned char status)
{
    holder[index].status = status;
}

int externalProtocolAddHeader(unsigned char *buff, int *size)
{
    buff[0] = 0xC0;
    buff[*size + 1] = 0xC0;
    *size = *size + 2;
    return 0;
}

void externalProtocolEncode(unsigned char *buff, int *size)
{
#if 0
    int i = 0, j = 0;
    for (i = 0; i < *size; i++)
    {
        if (buff[i] == 0XC0)
        {
            buff[i] = 0XDB;
            *size = *size + 1;
            for (j = *size - 1; j > i + 1; j--)
            {
                buff[j] = buff[j - 1];
            }
            buff[i + 1] = 0XDC;

        }
        else if (buff[i] == 0XDB)
        {
            buff[i] = 0XDB;
            *size = *size + 1;
            for (j = *size - 1; j > i + 1; j--)
            {
                buff[j] = buff[j - 1];
            }
            buff[i + 1] = 0XDD;
        }
    }
#else
    int i = 0, j = 0;
    for (i = 0; i < *size; i++)
    {
        if (buff[i] == 0xC0)
        {
            buff[i] = 0xDB;
            *size = *size + 1;
            for (j = *size - 1; j > i + 1; j--)
            {
                buff[j] = buff[j - 1];
            }
            buff[i + 1] = 0xDC;
        }
        else if (buff[i] == 0xDB)
        {
            buff[i] = 0xDB;
            *size = *size + 1;
            for (j = *size - 1; j > i + 1; j--)
            {
                buff[j] = buff[j - 1];
            }
            buff[i + 1] = 0xDD;
        }
    }
#endif
}

void externalProtocolDecode(unsigned char *buff, int *size)
{
#if 0
    int i = 0, j = 0;
    for (i = 0; i < *size; i++)
    {
        if (buff[i] == 0x7D && buff[i + 1] == 0x5E) {
            buff[i] = 0x7E;
            //闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊為悷婊冾樀瀵悂寮介鐔哄幐闂佹悶鍎崕閬嶆倶閳哄啰纾奸柟閭﹀弾濞堟粓鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ラ梻鍌欑閹诧繝宕圭憴鍕弿闁圭虎鍠栭悡姗鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠樼紒杈ㄦ尰閹峰懘宕崟纰夋嫹瀹鍕厱闁靛ǹ鍎抽崺锝団偓娈垮枛椤兘寮幇顓炵窞濠电姴瀚烽崥鍛存⒒娴ｇ懓顕滅紒璇插块獮澶娾槈閵忕姷顔掔紓鍌欑劍椤洭宕㈤幖浣光拺闁告稑锕ｇ欢閬嶆煕閻樻剚娈滅规洘鍔欓獮鍥偋閸垹骞愰柣搴ｆ閹风兘鎮楅敐搴″闁伙箑鐗撻幃妤冩喆閸曨剛顦ㄩ柣銏╁灡鐢喖鎮橀幒妤佲拺缂備焦枪婢规﹢鏌ｆ幊閸斿酣鏁撻懞銉ㄥ缂傚秴锕濠氬灳閹颁礁鎮戦柟鑲╄ˉ閿熻姤褰冮ˉ姘舵⒒娴ｇ儤鍤闁搞垺鐓″畷鏌ュ蓟閵夘垳绋忓┑鐘绘涧椤戞劙寮崱娑欑厱閻忕偛澧介弫鏍煏婵炵偓娅嗛柍閿嬪灴閺屾稑鈹戦崱妤婁紓濡炪値鍋撶粻鎾诲蓟濞戙垹绫嶉柨鐔剁矙瀹曞綊宕稿Δ鍐ㄧウ闂佸憡鍔﹂崰鏍綖閸涘瓨鐓忛柛顐ｇ箖閸ｅ湱绱掗幇顓犫姇缂佺粯绻堥幃浠嬫濞磋翰鍨介幃妤顫濋梻瀵告晼濠碘冲级閸旀瑥顕ｉ幘顔碱潊闁虫拝鎷烽柟椋庣帛缁绘稒娼忛崜褎鍋ч梺绋跨昂閸婃繂鐣烽悽鍛婃櫢濞寸姴顑嗛埛鎴︽煙閼测晛浠滈柨鐔诲Г閻楃娀骞冭閹晠鎳￠妶鍛偊闂傚⿴鍋勫ú锔剧矙閹烘鐤鹃柟闂寸劍閻撱儵鏌ｉ弴鐐测偓鍦偓姘炬嫹?闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊為悷婊冾樀瀵悂寮介鐔哄幐闂佹悶鍎崕閬嶆倶閳哄啰纾奸柟閭﹀弾濞堟粓鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ラ梻鍌欑閹诧繝宕圭憴鍕弿闁圭虎鍠栭悡姗鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠樼紒杈ㄦ尰閹峰懘宕崟纰夋嫹瀹鍕厱闁靛ǹ鍎抽崺锝団偓娈垮枛椤攱淇婇幖浣哥厸闁稿本鐭花浠嬫⒒娴ｄ警鐒鹃柡鍫墰缁瑩骞嬮敂缁樻櫆闂佽法鍣﹂幏锟???闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊為悷婊冾樀瀵悂寮介鐔哄幐闂佹悶鍎崕閬嶆倶閳哄啰纾奸柟閭﹀弾濞堟粓鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ラ梻鍌欑閹诧繝宕圭憴鍕弿闁圭虎鍠栭悡姗鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠樼紒杈ㄦ尰閹峰懘宕崟纰夋嫹瀹鍕厱闁靛ǹ鍎抽崺锝団偓娈垮枛椤攱淇婇幖浣哥厸闁稿本鐭花浠嬫⒒娴ｄ警鐒鹃柡鍫墰缁瑩骞嬮敂缁樻櫆闂佽法鍣﹂幏锟??闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊為悷婊冾樀瀵悂寮介鐔哄幐闂佹悶鍎崕閬嶆倶閳哄啰纾奸柟閭﹀弾濞堟粓鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ラ梻鍌欑閹诧繝宕圭憴鍕弿闁圭虎鍠栭悡姗鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠樼紒杈ㄦ尰閹峰懘宕崟纰夋嫹瀹鍕厱闁靛ǹ鍎抽崺锝団偓娈垮枛椤攱淇婇幖浣哥厸闁稿本鐭花浠嬫⒒娴ｄ警鐒鹃柡鍫墰缁瑩骞嬮敂缁樻櫆闂佽法鍣﹂幏锟
            for (j = i + 1; j < *size; j++)
            {
                buff[j] = buff[j + 1];
            }
            *size = *size - 1;
        } else if (buff[i] == 0x7D && buff[i + 1] == 0x5D) {
            buff[i] = 0x7D;
            //闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊為悷婊冾樀瀵悂寮介鐔哄幐闂佹悶鍎崕閬嶆倶閳哄啰纾奸柟閭﹀弾濞堟粓鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ラ梻鍌欑閹诧繝宕圭憴鍕弿闁圭虎鍠栭悡姗鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠樼紒杈ㄦ尰閹峰懘宕崟纰夋嫹瀹鍕厱闁靛ǹ鍎抽崺锝団偓娈垮枛椤兘寮幇顓炵窞濠电姴瀚烽崥鍛存⒒娴ｇ懓顕滅紒璇插块獮澶娾槈閵忕姷顔掔紓鍌欑劍椤洭宕㈤幖浣光拺闁告稑锕ｇ欢閬嶆煕閻樻剚娈滅规洘鍔欓獮鍥偋閸垹骞愰柣搴ｆ閹风兘鎮楅敐搴″闁伙箑鐗撻幃妤冩喆閸曨剛顦ㄩ柣銏╁灡鐢喖鎮橀幒妤佲拺缂備焦枪婢规﹢鏌ｆ幊閸斿酣鏁撻懞銉ㄥ缂傚秴锕濠氬灳閹颁礁鎮戦柟鑲╄ˉ閿熻姤褰冮ˉ姘舵⒒娴ｇ儤鍤闁搞垺鐓″畷鏌ュ蓟閵夘垳绋忓┑鐘绘涧椤戞劙寮崱娑欑厱閻忕偛澧介弫鏍煏婵炵偓娅嗛柍閿嬪灴閺屾稑鈹戦崱妤婁紓濡炪値鍋撶粻鎾诲蓟濞戙垹绫嶉柨鐔剁矙瀹曞綊宕稿Δ鍐ㄧウ闂佸憡鍔﹂崰鏍綖閸涘瓨鐓忛柛顐ｇ箖閸ｅ湱绱掗幇顓犫姇缂佺粯绻堥幃浠嬫濞磋翰鍨介幃妤顫濋梻瀵告晼濠碘冲级閸旀瑥顕ｉ幘顔碱潊闁虫拝鎷烽柟椋庣帛缁绘稒娼忛崜褎鍋ч梺绋跨昂閸婃繂鐣烽悽鍛婃櫢濞寸姴顑嗛埛鎴︽煙閼测晛浠滈柨鐔诲Г閻楃娀骞冭閹晠鎳￠妶鍛偊闂傚⿴鍋勫ú锔剧矙閹烘鐤鹃柟闂寸劍閻撱儵鏌ｉ弴鐐测偓鍦偓姘炬嫹?闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊為悷婊冾樀瀵悂寮介鐔哄幐闂佹悶鍎崕閬嶆倶閳哄啰纾奸柟閭﹀弾濞堟粓鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ラ梻鍌欑閹诧繝宕圭憴鍕弿闁圭虎鍠栭悡姗鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠樼紒杈ㄦ尰閹峰懘宕崟纰夋嫹瀹鍕厱闁靛ǹ鍎抽崺锝団偓娈垮枛椤攱淇婇幖浣哥厸闁稿本鐭花浠嬫⒒娴ｄ警鐒鹃柡鍫墰缁瑩骞嬮敂缁樻櫆闂佽法鍣﹂幏锟???闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊為悷婊冾樀瀵悂寮介鐔哄幐闂佹悶鍎崕閬嶆倶閳哄啰纾奸柟閭﹀弾濞堟粓鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ラ梻鍌欑閹诧繝宕圭憴鍕弿闁圭虎鍠栭悡姗鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠樼紒杈ㄦ尰閹峰懘宕崟纰夋嫹瀹鍕厱闁靛ǹ鍎抽崺锝団偓娈垮枛椤攱淇婇幖浣哥厸闁稿本鐭花浠嬫⒒娴ｄ警鐒鹃柡鍫墰缁瑩骞嬮敂缁樻櫆闂佽法鍣﹂幏锟??闂傚倸鍊搁崐鎼佸磹閹间礁纾归柟闂寸绾惧湱鈧懓瀚崳纾嬨亹閹烘垹鍊為悷婊冾樀瀵悂寮介鐔哄幐闂佹悶鍎崕閬嶆倶閳哄啰纾奸柟閭﹀弾濞堟粓鏌″畝鈧崰鏍箖瑜斿畷濂告偄閸濆嫬娈ラ梻鍌欑閹诧繝宕圭憴鍕弿闁圭虎鍠栭悡姗鏌熸潏楣冩闁稿鍔欓幃妤呭捶椤撶倫銏°亜閵夛妇绠樼紒杈ㄦ尰閹峰懘宕崟纰夋嫹瀹鍕厱闁靛ǹ鍎抽崺锝団偓娈垮枛椤攱淇婇幖浣哥厸闁稿本鐭花浠嬫⒒娴ｄ警鐒鹃柡鍫墰缁瑩骞嬮敂缁樻櫆闂佽法鍣﹂幏锟
            for (j = i + 1; j < *size; j++)
            {
                buff[j] = buff[j + 1];
            }
            *size = *size - 1;
        }
    }
    return;
#else
    int i = 0, j = 0;
    for (i = 0; i < *size; i++)
    {
        if (buff[i] == 0xDB && buff[i + 1] == 0xDC)
        {
            buff[i] = 0xC0;
            for (j = i + 1; j < *size; j++)
            {
                buff[j] = buff[j + 1];
            }
            *size = *size - 1;
        }
        else if (buff[i] == 0xDB && buff[i + 1] == 0xDD)
        {
            buff[i] = 0xDB;
            for (j = i + 1; j < *size; j++)
            {
                buff[j] = buff[j + 1];
            }
            *size = *size - 1;
        }
    }
    //return TRUE;
#endif
}

int externalProtocolAddCrcCode(unsigned char *buff, int *size)
{
#if 0
    unsigned char xor_crc = 0;
    int i = 0;
    for (i = 0; i < *size; i++)
        xor_crc ^= buff[i];

    *size = *size + 1;
    buff[*size - 1] = xor_crc;

    if (buff[*size - 1] == 0xC0) {
        buff[*size - 1] = 0xDB;
        *size = *size + 1;
        buff[*size - 1] = 0xDC;
    } else if (buff[*size - 1] == 0xDB) {
        buff[*size - 1] = 0xDB;
        *size = *size + 1;
        buff[*size - 1] = 0xDD;
    }
#else
    unsigned char xor_crc = 0;
    int i = 0;
    for (i = 0; i < *size; i++)
        xor_crc ^= buff[i];
    *size = *size + 1;
    buff[*size - 1] = xor_crc;
#endif
    return 0;
}

int externalProtocolCheckCrc(unsigned char *buff, int size)
{
#if 0
    unsigned char xor_crc = 0;
    unsigned char my_xor = 0;
    int i = 0;
    int len = 0;

    if (buff[size - 1] == 0xDB && buff[size - 2] == 0xDC) {
        my_xor = 0xC0;
        len = size - 2;
    } else if (buff[size - 1] == 0xDB && buff[size - 2] == 0xDD) {
        my_xor = 0xDB;
        len = size - 2;
    } else {
        len = size - 1;
        my_xor = buff[size - 1];
    }

    for (i = 0; i < len; i++)
        xor_crc ^= buff[i];

    if (xor_crc == my_xor) {
        return 0;
    } else {
        return -1;
    }
#else
    char xor_crc = 0;
    int i = 0;
    for (i = 0; i < size; i++)
    {
        xor_crc ^= buff[i];
    }
    if (xor_crc == buff[size])
    {
        return 1;
    }
    else
    {
        return 0;
    }
#endif
}

//鐠佹儳顦潻鎰攽閻樿埖鈧焦鐓＄拠銏犳惙鎼达拷
int ReportedCammerStatus(int sock, int mSessionID)
{
    int size = 0;
    unsigned char devid = 0;
    unsigned char sendbuff[100];
    m_sig_data *p_fvdstaticchannel; //=&p_detect_ctx->cam_ctx[i].thread_param.channel_rst;
    //  unsigned char *sendbuff=NULL;
    //  mSystemConfig * fsys = g_pSyscfg;

    /// sendbuff = (unsigned char *)malloc(100);
    if (sendbuff == NULL)
        return 0;
    size = 0;
    memset(sendbuff, 0, sizeof(sendbuff));
    FrameHeader *fnewhead = (FrameHeader *)sendbuff;
    fnewhead->mSenderLinkId = 0x0;
    fnewhead->mRecieverLinkId = LINKID_UART_BETWEEN_COMBOARD_AND_MAINCONTLBOARD;
    fnewhead->mProtocolType = PROTOCOTYPE_VIDEO_DETECTOR;
    fnewhead->mProtocolVersion = 0x12;
    fnewhead->mSenderDeviceID.mDeviceIndex = 0x1;
    fnewhead->mSenderDeviceID.mDeviceClass = 0x1;
    fnewhead->mRecieverDeviceId.mDeviceIndex = 0x0;
    fnewhead->mRecieverDeviceId.mDeviceClass = 0x4;
    fnewhead->mSessionID = mSessionID;
    fnewhead->mDataType = 0x04;

    size = sizeof(FrameHeader) - 1;
    DeviceWorkStatusQueryResponse *res = (DeviceWorkStatusQueryResponse *)(sendbuff + sizeof(FrameHeader));
    devid = get_dev_id();
    res->mDetectMainMachineDeviceId.mDeviceClass = 0x1;
    res->mDetectMainMachineDeviceId.mDeviceIndex = devid & 0x0F;
    res->uDetectMainMachineStatus.mStatusWhenSeperatedMachine.bRegisted = 0x1;
    res->uDetectMainMachineStatus.mStatusWhenSeperatedMachine.bTimeCorrected = 0x1;
    res->uDetectMainMachineStatus.mStatusWhenSeperatedMachine.bFix0 = 0x0;
    res->mCameralCount = 0x0;
    size += 3;

    int i = 0;
    int j;
    for (i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
    {

        if (!g_cam_num.exist[i])
            continue;
        else
            j++;

        p_fvdstaticchannel = &holder[i].traffic_rst;
        // if(get_cam_status(i)){
        EachCameralStatus *cstatus = (EachCameralStatus *)(sendbuff + sizeof(FrameHeader) + 3 + res->mCameralCount * sizeof(EachCameralStatus));

        //  cstatus->mCameralDeviceId.mDeviceIndex = p_detect_ctx->cam_ctx[i].thread_param.cam_cfg.camattr.camID;

        cstatus->mCameralDeviceId.mDeviceIndex = get_cam_id(i);

        cstatus->mCameralDeviceId.mDeviceClass = DEVICECLASS_IPCAMERAL;
        //          prt(info,"get direction %d",get_cam_direction(i));
        switch (get_cam_location(i))
        {
        case Z_NORTH:
            cstatus->mCameralPosition.bNorth = 0x1;
            break;

        case Z_NORTHEAST:
            cstatus->mCameralPosition.bEastNorth = 0x1;
            break;

        case Z_EAST:
            cstatus->mCameralPosition.bEast = 0x1;
            break;

        case Z_SOUTHEAST:
            cstatus->mCameralPosition.bEastSouth = 0x1;
            break;

        case Z_SOUTH:
            cstatus->mCameralPosition.bSouth = 0x1;
            break;

        case Z_SOUTHWEST:
            cstatus->mCameralPosition.bWestSouth = 0x1;
            break;

        case Z_WEST:
            cstatus->mCameralPosition.bWest = 0x1;
            break;

        case Z_NORTHWEST:
            cstatus->mCameralPosition.bWestNorth = 0x1;
            break;

        default:
            cstatus->mCameralPosition.bNorth = 0x1;
            break;
        }

#if 0
        switch (p_fvdstaticchannel->EachStatus.mCameralStatus.bWorkMode) {
        case MORNING:
        case DUSK:
        {
            cstatus->mCameralStatus.bWorkMode = 0x01;
        }
        break;
        case DAYTIME:
        {
            cstatus->mCameralStatus.bWorkMode = 0x00;
        }
        break;
        case NIGHT:
        {
            cstatus->mCameralStatus.bWorkMode = 0x02;
        }
        break;
        default:
        {
            cstatus->mCameralStatus.bWorkMode = 0x03;
        }
        break;
        }

#endif
        cstatus->mCameralStatus.bWorkMode = p_fvdstaticchannel->EachStatus.mCameralStatus.bWorkMode;
        //cstatus->mCameralStatus.bBackgroundRefreshed
        //       = p_fvdstaticchannel->EachStatus.mCameralStatus.bBackgroundRefreshed;
        cstatus->mCameralStatus.bBackgroundRefreshed = 0x01;
        cstatus->mCameralStatus.bH264DecodeStatus = p_fvdstaticchannel->EachStatus.mCameralStatus.bH264DecodeStatus;
        /*
            if(0x1 == p_fvdstaticchannel->status || 0x2 == p_fvdstaticchannel->status)
                cstatus->mCameralStatus.bCameralOnLine=0x1;
            else
                cstatus->mCameralStatus.bCameralOnLine=0x0;
            */
        //if (get_cam_status(i) > 0)
        if (get_cam_running_state(i) == CAM_RUNNING)
            cstatus->mCameralStatus.bCameralOnLine = 0x1;
        else
            cstatus->mCameralStatus.bCameralOnLine = 0x0;
        cstatus->mCameralStatus.bPictureStable = p_fvdstaticchannel->EachStatus.mCameralStatus.bPictureStable;
        cstatus->mCameralStatus.bFix0 = p_fvdstaticchannel->EachStatus.mCameralStatus.bFix0;
        res->mCameralCount++;
        size += sizeof(EachCameralStatus);
        // }
    }

#if 0
    externalProtocolEncode(sendbuff + 1, &size);
    externalProtocolAddCrcCode(sendbuff + 1, &size);
    externalProtocolAddHeader(sendbuff, &size);
#else
    externalProtocolAddCrcCode(sendbuff + 1, &size);
    externalProtocolEncode(sendbuff + 1, &size);
    externalProtocolAddHeader(sendbuff, &size);
#endif
    int tmp = 0;
    //          prt(stack,"=========report cam status======");
    //          for (tmp = 0; tmp < size; ++tmp) {
    //              prt(info,"%x",sendbuff[tmp]);
    //          }
    return SendDataInLock(sock, (char *)sendbuff, size);
}

int SendDataInLock(int sock, char *buffer, int len)
{
    prt(info, "SendDataInLock is work");
    int ret = -1;
    pthread_mutex_lock(&sig_client_lock);
    ret = SendDataByTcp(sock, (char *)buffer, len);
    pthread_mutex_unlock(&sig_client_lock);

    return ret;
}

int RequestPro(unsigned char *buffer, int len, int sock)
{
    printf("get request\n");
    unsigned char *sendbuff = NULL;
    int size = 0;
    int i = 0;
    while (i < len)
    {
        printf("%x\n", buffer[i]);
        i++;
    }
    if (buffer[0] != 0xC0 || buffer[len - 1] != 0xC0)
    {
        return 0;
    }

#if 0
    size = len - 2;
    if (externalProtocolCheckCrc(buffer + 1, size)) {
        return 0;
    }
    size -= 1;
    externalProtocolDecode(buffer + 1, &size);
    if ((size + 1) != sizeof(FrameHeader) && (size + 1) != (sizeof(FrameHeader) + 4)) {
        return 0;
    }
#else
    size = len - 2;
    printf("check crc \n\n");
    externalProtocolDecode(buffer + 1, &size);
    if (externalProtocolCheckCrc(buffer + 1, size))
    {
        printf("check crc err \n\n");
        return 0;
    }
#endif

    FrameHeader *fhead = (FrameHeader *)buffer;
    if (fhead->mSenderLinkId != LINKID_UART_BETWEEN_COMBOARD_AND_MAINCONTLBOARD || fhead->mRecieverLinkId != LINKID_BROARDCAST || fhead->mProtocolType != PROTOCOTYPE_VIDEO_DETECTOR || fhead->mProtocolVersion != 0x12)
    {
        return 0;
    }

    if (fhead->mDataType == 0x05)
    {
        char strTime[24];
        //time_t UTCTime = ntohl(*((unsigned int*)(buffer+sizeof(FrameHeader))))+28800;
        time_t UTCTime = *((unsigned int *)(buffer + sizeof(FrameHeader))) + 28800;
        struct tm *ppltime = gmtime(&UTCTime);
        strftime(strTime, 24, "%F %T", ppltime);
        char cmd[50] = {0};
        sprintf(cmd, "date -s \"%s\";hwclock -w", strTime);
        system(cmd);
        printf("reply tm\n");
    }
    else if (fhead->mDataType == 0x03)
    {
        printf("rply  \n");
        return ReportedCammerStatus(sock, fhead->mSessionID);
    }
    return 0;
}
#include "camera_service.h"
int ReportedEvent(int sock, char event)
{
    int size;
    unsigned char dev_id = 0;
    unsigned char mysbuf[100] = {0};

    memset(mysbuf, 0, sizeof(mysbuf));
    FrameHeader *fhead = (FrameHeader *)mysbuf;
    fhead->mSenderLinkId = 0x00;
    fhead->mRecieverLinkId = LINKID_UART_BETWEEN_COMBOARD_AND_MAINCONTLBOARD;
    fhead->mProtocolType = PROTOCOTYPE_VIDEO_DETECTOR;
    fhead->mProtocolVersion = 0x12;
    //fhead->mSenderDeviceID.mDeviceIndex = g_pSyscfg->devparam.deviceID;
    fhead->mSenderDeviceID.mDeviceClass = 0x1;
    fhead->mRecieverDeviceId.mDeviceIndex = 0x0;
    fhead->mRecieverDeviceId.mDeviceClass = 0x4;
    fhead->mSessionID = 0xFF;

    fhead->mDataType = 0x10;
    size = sizeof(FrameHeader) - 1;
    DeviceEventsAutoReporte *report = (DeviceEventsAutoReporte *)(mysbuf + sizeof(FrameHeader));
    dev_id = get_dev_id();
    report->mEventDeviceId.mDeviceIndex = dev_id & 0x0F;
    report->mEventDeviceId.mDeviceClass = 0x1;
    report->uEventOccourPosition.mEventOccourPositionWhenCameralOrIntegratedCameral.bNorth = 0x0;
    report->uEventOccourPosition.mEventOccourPositionWhenCameralOrIntegratedCameral.bEastNorth = 0x0;
    report->uEventOccourPosition.mEventOccourPositionWhenCameralOrIntegratedCameral.bEast = 0x0;
    report->uEventOccourPosition.mEventOccourPositionWhenCameralOrIntegratedCameral.bEastSouth = 0x0;
    report->uEventOccourPosition.mEventOccourPositionWhenCameralOrIntegratedCameral.bSouth = 0x0;
    report->uEventOccourPosition.mEventOccourPositionWhenCameralOrIntegratedCameral.bWestSouth = 0x0;
    report->uEventOccourPosition.mEventOccourPositionWhenCameralOrIntegratedCameral.bWest = 0x0;
    report->uEventOccourPosition.mEventOccourPositionWhenCameralOrIntegratedCameral.bWestNorth = 0x0;
    size += sizeof(DeviceEventsAutoReporte);
    report->mEvent = event;
    report->mEventTime = (int)time(NULL);
#if 0
    externalProtocolEncode(mysbuf + 1, &size);
    externalProtocolAddCrcCode(mysbuf + 1, &size);
    externalProtocolAddHeader(mysbuf, &size);
#else
    externalProtocolAddCrcCode(mysbuf + 1, &size);
    externalProtocolEncode(mysbuf + 1, &size);
    externalProtocolAddHeader(mysbuf, &size);
#endif
    int tmp = 0;
    //              prt(info,"=========report cam event======");
    //              for (tmp = 0; tmp < size; ++tmp) {
    //                  prt(info,"%x",mysbuf[tmp]);
    //              }

    return SendDataInLock(sock, (char *)mysbuf, size);
}

void reset_sig_machine()
{
    sig_set_state(SIG_NULL);
    if (sig_fd > 0)
    {
        close_socket(&sig_fd);
        sig_fd = -1;
    }
    //get_sig_ip(sig_ip);
    //prt(info,"reset sig ip to %s",sig_ip);
    //sig_port=get_sig_port();
    init_sig_client();
    sig_set_state(SIG_PRE_CONNECT);
}

/*
void *report_data_callback_fun(void *data)
{
    if (sig_state == SIG_CONNECTED) {
        if (ReportedRealStatic(sig_fd) <= 0) {
            close_socket(&sig_fd);
            sig_set_state(SIG_PRE_CONNECT);
        }
    }

}
void *check_event_callback_fun(void *data)
{
    int i;
    for (i = 0; i < ACTIVE_CAM_NUM; i++) {
        if (holder[i].traffic_rst.camera_state_change&&sig_state==SIG_CONNECTED) {
            if(ReportedCammerStatus(sig_fd, 0xFF)>0)
            {
                holder[i].traffic_rst.camera_state_change = 0;
            }else{
                prt(info,"fail to send");
            }
        }
    }
}
*/

void reboot_cmd()
{
    if (PROTO_HUAITONG == g_ivddevsets.pro_type)
    {
        int retlen = ReportedEvent(sig_fd, 0x1);
        if (retlen < 0)
        {
            close_socket(&sig_fd);
            // close_socket(&sig_fd);
            //      return ;
        }
        retlen = ReportedEvent(sig_fd, 0x4);
        if (retlen < 0)
        {
            close_socket(&sig_fd);
            //close_socket(&sig_fd);
            //      continue;
        }
    }

    usleep(200000);
    prt(info, "rebooting ");
    system("reboot");
}
int connect_sig(int &fd, char *ip, unsigned short port)
{
    int retlen;
    if (fd > 0)
    {
        usleep(1000000);
        prt(debug_sig, "try to connect signal machine,%s", sig_ip);
        retlen = ConnectTcpClient(fd, ip, port, false);
        prt(debug_sig, "ret %d", retlen);
        if (-1 == retlen)
        {
            close_socket(&fd);
        }
        else
        {
            prt(net, "ok to connect signal machine");
        }
    }
    return retlen;
}
void *sig_service_thread(void *data)
{
    int retlen = 0;
    while (1)
    {
        usleep(100000);
        switch (sig_state)
        {
        case SIG_PRE_CONNECT:
        {
            if (sig_fd <= 0)
            {
                //sig_fd=CreateTcpClientSock(SIGNALPORTL, 0);
                //sig_fd=CreateTcpClientSock(0, 0);// random port
                //  sig_fd = creat_sig_sock(0x01, 500, 0);
                sig_fd = creat_sig_sock(0x03, 500, 2000);
            }

            if (sig_fd < 0)
                break;

            if ((sig_port > 0) && connect_sig(sig_fd, (char *)sig_ip, sig_port) <= 0)
                break;

            if (PROTO_HUAITONG == g_ivddevsets.pro_type)
            {
                retlen = ReportedEvent(sig_fd, 0x2);
                if (retlen < 0)
                {
                    close_socket(&sig_fd);
                    break;
                }
                retlen = ReportedEvent(sig_fd, 0x3);
                if (retlen < 0)
                {
                    close_socket(&sig_fd);
                    break;
                }
                retlen = ReportedCammerStatus(sig_fd, 0xFF);
                if (retlen < 0)
                {
                    close_socket(&sig_fd);
                    break;
                }
            }
            sig_set_state(SIG_CONNECTED);
        }
        break;
        case SIG_CONNECTED:
            break;
        case SIG_NULL:
            break;
        default:
            break;
        }
    }

    prt(debug_sig, "sig_service_thread exit!");
}

void init_sig_client()
{
    int index = 0;
    int n = 0;
    sig_set_state(SIG_NULL);
    sig_fd = -1;

    for (int j = 0; j < CAM_MAX && n < g_cam_num.cam_num; j++)
    {
        if (!g_cam_num.exist[j])
        {
            continue;
        }
        else
        {
            n++;
            index = j;
        }
    }

    if (n < 1)
        return;

    // if (g_camdetect[index].other.detecttype == 2) {
    enum PROTOCOL_TYPE pro_type = (enum PROTOCOL_TYPE)g_ivddevsets.pro_type;
    switch (pro_type)
    {
    case PROTO_PRIVATE:
    case PROTO_HAIXIN:
    case PROTO_YIHUALU:
    {
        strcpy(sig_ip, (char *)g_netInfo.strIpaddrIO);
        sig_port = g_netInfo.strPortIO;
        //1-濞磋渹淇婇崡蹇氼唴 //2-閺勬挸宕曡ぐ锟
        sig_set_state(SIG_PRE_CONNECT);
    }
    break;
    case PROTO_NANJING:
    {
        strcpy(sig_ip, (char *)g_netInfo.strIpaddrIO);
        sig_port = g_netInfo.strPortIO;
        //3-閸楁ぞ鍚懢杈ㄦ焿
        sig_set_state(SIG_PRE_CONNECT);
    }
    break;
    case PROTO_HUAITONG:
    case PROTO_HUAITONG_PERSON:
    {
        strcpy(sig_ip, (char *)g_netInfo.strIpaddrIO);
        sig_port = g_netInfo.strPortIO;
        sig_set_state(SIG_PRE_CONNECT);
    }
    break;
    case PROTO_PRIVATE_PERSON:
    {
        strcpy(sig_ip, (char *)g_netInfo.strIpaddrIO);
        sig_port = 18767;
    }
    break;
    default:
        break;
    }

    // }else if (g_camdetect[index].other.detecttype == 1) {
    //         strcpy(sig_ip, (char *) g_netInfo.strIpaddrIO);
    //        sig_port = 18767;
    // }
}

#define MAXDATASIZE 100
void *sig_client_handle_thread(void *data)
{
    int ret = 0;
    int num = 0;
    //int last_num = 0;
    fd_set rd_set;
    unsigned char buf[MAXDATASIZE];

    while (1)
    {
        if (sig_state != SIG_CONNECTED)
        {
            sleep(1);
            continue;
        }

        struct timeval timeout = {2, 0}; //2 seconds

        FD_ZERO(&rd_set);
        FD_SET(sig_fd, &rd_set);

        ret = select(sig_fd + 1, &rd_set, NULL, NULL, &timeout);

        if (0 > ret)
        {
            printf("select error\n");
            sig_set_state(SIG_PRE_CONNECT);
            continue;
        }

        if (0 == ret)
        {
            printf("time out\n");
            continue;
        }

        memset(buf, 0, MAXDATASIZE);
        pthread_mutex_lock(&sig_client_lock);
        if ((num = recv(sig_fd, buf, MAXDATASIZE, 0)) == -1)
        {
            pthread_mutex_unlock(&sig_client_lock);
            printf("recv() error\n");
            sig_set_state(SIG_PRE_CONNECT);
        }
        else
        {
            pthread_mutex_unlock(&sig_client_lock);
            int sz_tail = sizeof(FrameTail);
            int sz_head = sizeof(FrameHeader);
            if (num >= (sz_head + sz_tail) && num < MAXDATASIZE)
            {
                if (buf[0] != 0xC0 || buf[num - 1] != 0xC0)
                    continue;

                FrameHeader *fhead = (FrameHeader *)buf;
                if (fhead->mDataType == 0x05)
                {
                    char strTime[24];

                    time_t UTCTime = *((unsigned int *)(buf + sizeof(FrameHeader))) + 28800;
                    struct tm *ppltime = gmtime(&UTCTime);
                    strftime(strTime, 24, "%F %T", ppltime);
                    char cmd[50] = {0};
                    sprintf(cmd, "date -s \"%s\";hwclock -w", strTime);
                    system(cmd);
                    printf("reply tm\n");
                }
                else if (fhead->mDataType == 0x03)
                {
                    printf("rply  \n");
                    if (ReportedCammerStatus(sig_fd, fhead->mSessionID) < 0)
                        sig_set_state(SIG_PRE_CONNECT);
                }
            }
        }
    }
}

//m_timed_func_data callback_data;
//m_timed_func_data callback_data1;

//####################nan jing #####################
data_60s_t d60[CAM_MAX] = {0};
data_300s_t d300[CAM_MAX] = {0};

//---------queue start----------------
flow_info_t flow_data[CAM_MAX];
queue_info_t queue_data[CAM_MAX];

radar_result_t radar_result_tmp;
radar_result_came_lane_t radar_came_lane[CAM_MAX];

//radar_result_lane_t radar_came_lane[m].lanes[MAX_LANE_NUM][NANJING_LANE_COIL_MAX];

radar_realtime_t radar_realtime_tmp;
//radar_realtime_lane_t radar_realtime_lane_tmp[NANJING_CAM_NUM][MAX_LANE_NUM];

radar_cycle_result_came_lane_t radar_300_came_lane[CAM_MAX];
//radar_cycle_result_lane_t radar_300_came_lane[m].lanes[MAX_LANE_NUM];

#if 0
void get_queue_info()
{
    int i, j;
    for (i = 0; i < NANJING_CAM_NUM; i++) {
        for (j = 0; j < NANJING_LANE_MAX; j++) {
            queue_data[i].queue_len[j] = info.cams[i].lanes[j].queue_len;
            queue_data[i].crc =  get_crc((unsigned char *)&queue_data[i], sizeof(queue_info_t) - 1);




            queue_data[i].detect_time[0] = get_year_tail();
            queue_data[i].detect_time[1] = get_month();
            queue_data[i].detect_time[2] = get_day();
            queue_data[i].detect_time[3] = get_hour();
            queue_data[i].detect_time[4] = get_min();
            queue_data[i].detect_time[5] = get_sec();

            queue_data[i].dir_no = get_direction(i);
            queue_data[i].lane_dir_type = 0;
            queue_data[i].queue_start_pos[j] = 0;
            queue_data[i].queue_veh_num[j] = info.cams[i].lanes[j].veh_no;

            for (int t = 0; t < 5; t++) {
                queue_data[i].table_head[t] = 0xfe;
            }
            queue_data[i].table_no = 0x0e;
            queue_data[i].detect_status = (unsigned char)info.cams[i].lanes[j].det_status;
            queue_data[i].table_length = 43;
            queue_data[i].veh_speed[j] = info.cams[i].lanes[j].speed;
        }
    }

}
#else
void get_queue_info()
{
    int i, j, n;
    for (i = 0, n = 0; i < CAM_MAX && n < g_cam_num.cam_num; i++)
    {
        if (!g_cam_num.exist[i])
            continue;
        else
        {
            n++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }

        for (int t = 0; t < 5; t++)
        {
            queue_data[i].table_head[t] = 0xfe;
        }
        queue_data[i].table_no = 0x1b;//0x0e;
        queue_data[i].table_length = 45;
        queue_data[i].detect_time[0] = get_year_tail();
        queue_data[i].detect_time[1] = get_month();
        queue_data[i].detect_time[2] = get_day();
        queue_data[i].detect_time[3] = get_hour();
        queue_data[i].detect_time[4] = get_min();
        queue_data[i].detect_time[5] = get_sec();
        queue_data[i].detect_status = (unsigned char)info.cams[i].lanes[0].det_status;
        queue_data[i].areano = atoi(g_ivddevsets.devUserNo);
        queue_data[i].juncno = g_camdetect[i].other.personlimit & 0xFF;
        queue_data[i].dir_no = get_direction(i);
        queue_data[i].lane_dir_type = 0;
        for (j = 0; j < MAX_LANE_NUM; j++)
        {
            queue_data[i].queue_len[j] = info.cams[i].lanes[j].queue_len;
            queue_data[i].queue_start_pos[j] = 0;
            queue_data[i].queue_veh_num[j] = info.cams[i].lanes[j].veh_no;
            queue_data[i].veh_speed[j] = info.cams[i].lanes[j].speed;
        }

        queue_data[i].crc = get_crc((unsigned char *)&queue_data[i] + 5, sizeof(queue_info_t) - 6);
    }
}
#endif
int send_queue_info(int fd)
{
    printf("--->");
    fflush(stdout);
    int i, n;
    int len = 0;
    for (i = 0, n = 0; i < CAM_MAX && n < g_cam_num.cam_num; i++)
    {
        if (!g_cam_num.exist[i])
            continue;
        else
        {
            n++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }
        //for(i=0;i<NANJING_CAM_NUM;i++){
        //   pthread_mutex_lock(&holder[i].sig_data_lock);
        len = SendDataByTcp(fd, (char *)&queue_data[i], sizeof(queue_info_t));
        //  printf("---> %d",  info.cams[1].lanes[1].in_car);
        // pthread_mutex_unlock(&holder[i].sig_data_lock);
    }
    return len;
}

int send_radar(int fd, char *buf, int sz)
{
    printf("--->");
    fflush(stdout);
    int i = 0;
    ////for(i=0;i<1;i++){
    //for(i=0;i<NANJING_CAM_NUM;i++){
    //   pthread_mutex_lock(&holder[i].sig_data_lock);
    ////int len = SendDataByTcp(fd, (char *) buf, sz);
    //  printf("---> %d",  info.cams[1].lanes[1].in_car);
    // pthread_mutex_unlock(&holder[i].sig_data_lock);
    ////}
    return SendDataByTcp(fd, (char *)buf, sz);
}

m_timed_func_data camera_queue_info; // 1 s ---> focus on all camera
void *callback_camera_queue_info(void *data)
{
    pthread_mutex_lock(&mutex_lock);
    if (sig_state == SIG_CONNECTED)
    {
        get_queue_info();
        if (send_queue_info(sig_fd) <= 0)
        {
            close_socket(&sig_fd);
            sig_set_state(SIG_PRE_CONNECT);
        }
    }
    pthread_mutex_unlock(&mutex_lock);
}
//---------queue end----------------

//----------60s flow start----------------
#include <sys/time.h>
#include <time.h>
int send_flow_info(int fd)
{
    int i = 0;
    int j = 0;
    int len = 0;
    int ret = 0;
    int tmp_len = 0;
    for (i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
    {

        if (!g_cam_num.exist[i])
            continue;
        else
        {
            j++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }
        if ((ret = SendDataByTcp(fd, (char *)&flow_data[i], sizeof(flow_info_t)) > 0))
            len += ret;
        else
        {
            break;
        }
    }
    //prt(info,"sent %x\n",flow_data[0].table_length);
    //prt(info,"sent1 %x\n",flow_data[1].table_length);
    return len;
}

void calculate_60s()
{
    //lock
    int i = 0;
    int j = 0;
    int n;
    for (i = 0, n = 0; i < CAM_MAX && n < g_cam_num.cam_num; i++)
    {

        if (!g_cam_num.exist[i])
            continue;
        else
        {
            n++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }

        for (int t = 0; t < 5; t++)
        {
            flow_data[i].table_head[t] = 0xfe;
        }

        flow_data[i].table_no = 0x1f;//0x0c;
        flow_data[i].table_length = 36;
        flow_data[i].detect_time[0] = get_year_tail();
        flow_data[i].detect_time[1] = get_month();
        flow_data[i].detect_time[2] = get_day();
        flow_data[i].detect_time[3] = get_hour();
        flow_data[i].detect_time[4] = get_min();
        flow_data[i].detect_time[5] = get_sec();
        //get_time_string((char *)flow_data[i].detect_time, sizeof(flow_data[i].detect_time));
        flow_data[i].areano = atoi(g_ivddevsets.devUserNo);
        flow_data[i].juncno = g_camdetect[i].other.personlimit & 0xFF;
        flow_data[i].dir_no = get_direction(i);
        flow_data[i].section_no = 1;

        //prt(info,"%x\n",flow_data[i].table_length);
        //if(d60[i].data_valid){//time up, stop acu,start cal and send

        for (j = 0; j < MAX_LANE_NUM; j++)
        {
            flow_data[i].ocuppy_percent[j] = (unsigned char)(d60[i].lane_data[j][0].exist_duration / 600);
            // prt(info, "...............exist_duration:%d ocuppy: %d", d60[i].lane_data[j][0].exist_duration, flow_data[i].ocuppy_percent[j]);
            if (flow_data[i].flow[j])
                flow_data[i].average_speed[j] = (unsigned char)d60[i].lane_data[j][0].speed_sum / flow_data[i].flow[j];
            d60[i].lane_data[j][0].speed_sum = 0;
            flow_data[i].flow[j] = (unsigned char)d60[i].lane_data[j][0].pass_number;
            d60[i].lane_data[j][0].pass_number = 0;
            d60[i].lane_data[j][0].exist_duration = 0;
            //  prt(info,"(%d %d )-> %x\n\n",i,j,flow_data[0].table_length);
        }

        flow_data[i].crc = get_crc((unsigned char *)&flow_data[i] + 5, sizeof(flow_info_t) - 6);
        // prt(info,"%x\n\n",flow_data[i].table_length);
        //  prt(info,"(%d %d )--> %x\n\n",i,j,flow_data[0].table_length);
    }
    // prt(info,"---> %x\n\n",flow_data[0].table_length);

    //  prt(info,"done\n");
}

void calculate_60s_radar()
{
    for (int m = 0, n = 0; m < CAM_MAX && n < g_cam_num.cam_num; m++)
    {
        if (!g_cam_num.exist[m])
            continue;
        else
            n++;

        for (int i = 0; i < MAX_LANE_NUM; i++)
        {
            for (int j = 0; j < NANJING_LANE_COIL_MAX; j++)
            {

                radar_came_lane[m].lanes[i][j].flowA = (unsigned char)d60[m].lane_data[i][j].car_a_sum;
                radar_came_lane[m].lanes[i][j].flowB = (unsigned char)d60[m].lane_data[i][j].car_b_sum;
                radar_came_lane[m].lanes[i][j].flowC = (unsigned char)d60[m].lane_data[i][j].car_c_sum;
                radar_came_lane[m].lanes[i][j].flowSum = (unsigned char)d60[m].lane_data[i][j].pass_number;
                radar_came_lane[m].lanes[i][j].Occupy_rate = (unsigned char)(d60[m].lane_data[i][j].exist_duration * 200 / g_cycle_statis_time / 1000);

                if (radar_came_lane[m].lanes[i][j].flowSum > 0)
                {   //楠炲啿娼庢潪锕傛毐 楠炲啿娼庢潪锕傗偓锟 楠炲啿娼庢潪锕銇旈弮鎯扮獩
                    radar_came_lane[m].lanes[i][j].average_len = (unsigned char)(d60[m].lane_data[i][j].veh_len_sum / radar_came_lane[m].lanes[i][j].flowSum) * 10; //閸楁洑缍:0.1m
                    radar_came_lane[m].lanes[i][j].average_speed = (unsigned char)d60[m].lane_data[i][j].speed_sum / radar_came_lane[m].lanes[i][j].flowSum;
                    radar_came_lane[m].lanes[i][j].average_head_time = (unsigned char)d60[m].lane_data[i][j].head_time_sum / radar_came_lane[m].lanes[i][j].flowSum;
                }
                else
                {
                    radar_came_lane[m].lanes[i][j].average_len = 0;
                    radar_came_lane[m].lanes[i][j].average_speed = 0;
                    radar_came_lane[m].lanes[i][j].average_head_time = 0;
                }

                d60[m].lane_data[i][j].exist_duration = 0;
                d60[m].lane_data[i][j].pass_number = 0;
                d60[m].lane_data[i][j].head_len_sum = 0;
                d60[m].lane_data[i][j].car_a_sum = 0;
                d60[m].lane_data[i][j].car_b_sum = 0;
                d60[m].lane_data[i][j].car_c_sum = 0;
                d60[m].lane_data[i][j].veh_len_sum = 0;
                d60[m].lane_data[i][j].speed_sum = 0;
                d60[m].lane_data[i][j].head_time_sum = 0;
            }
        }
    }
}

void calculate_300s_radar() //鏉烇箓浜鹃崣宄板灊閸掑棛娈戠紒鐔活吀閸涖劍婀￠弫鐗堝祦
{
    for (int m = 0, j = 0; m < CAM_MAX && j < g_cam_num.cam_num; m++)
    {
        if (!g_cam_num.exist[m])
            continue;
        else
            j++;

        for (int i = 0; i < MAX_LANE_NUM; i++)
        {

            int occupy_rate = d300[m].lane_data[i].exist_duration * 100 / g_cycle_statis_time / 1000;
            if (occupy_rate > 70)
            {   //閹枫儱鐗
                radar_300_came_lane[m].lanes[i].lane_status = 2;
            }
            else if (occupy_rate > 40)
            {   //缂傛捁顢
                radar_300_came_lane[m].lanes[i].lane_status = 1;
            }
            else
            {   //閻ｅ懘鈧拷
                radar_300_came_lane[m].lanes[i].lane_status = 0;
            }
            radar_300_came_lane[m].lanes[i].lane_no = d300[m].lane_data[i].lane_no;
            radar_300_came_lane[m].lanes[i].queue_len_max = d300[m].lane_data[i].queue_len_max;
            radar_300_came_lane[m].lanes[i].car_stop_sum = d300[m].lane_data[i].car_stop_sum;
        }
        memset(&d300[m], 0, sizeof(data_300s_t));
    }
}

m_timed_func_data camera_flow_info; // 60 s---> focus on all camera
void *callback_camera_flow_info(void *data)
{
    struct timeval tv;
    volatile uint16 period = g_cycle_statis_time;

    if (period > 0)
    {
        gettimeofday(&tv, NULL);
        int delay_sec = tv.tv_sec % period;

        if (delay_sec > 0)
        {
            sleep(period - delay_sec);
        }
        else
        {
            sleep(period);
        }

        pthread_mutex_lock(&mutex_lock);
        //if(tm%60==0){
        calculate_60s();
        // prt(info, "calculate_60s............");
        //send_flow_info(sig_fd);
        if (sig_state == SIG_CONNECTED)
        {
            if (send_flow_info(sig_fd) <= 0)
            {
                close_socket(&sig_fd);
                sig_set_state(SIG_PRE_CONNECT);
            }
        }
        //}
        pthread_mutex_unlock(&mutex_lock);
    }
}
//----------60s flow end------------------
//--------- outcar start------------------
ourcar_info_t ourcar;
int send_outcar_info(int fd, int cam_index, int lane_index)
{

    ourcar.veh_speed = info.cams[cam_index].lanes[lane_index].speed;
    ourcar.lane_number = get_lane_index(cam_index, lane_index);

    int i = 0;
    int j = 0;
    for (i = 0; i < 5; i++)
    {
        ourcar.table_head[i] = 0xfe;
    }
    ourcar.table_no = 0x1c;//0x0f;
    ourcar.table_length = 20;
    ourcar.pass_time[0] = get_year_tail();
    ourcar.pass_time[1] = get_month();
    ourcar.pass_time[2] = get_day();
    ourcar.pass_time[3] = get_hour();
    ourcar.pass_time[4] = get_min();
    ourcar.pass_time[5] = get_sec();
    ourcar.areano = atoi(g_ivddevsets.devUserNo);
    ourcar.juncno = g_camdetect[cam_index].other.personlimit & 0xFF;
    ourcar.dir_no = get_direction(cam_index);
    ourcar.lane_dir_type = 0;
    ourcar.section_number = 1;
    ourcar.veh_type = 1;
    ourcar.occupy_time = info.cams[cam_index].lanes[lane_index].coils[0].at_out_car_time - info.cams[cam_index].lanes[lane_index].coils[0].at_in_car_time;
    ourcar.crc = get_crc((unsigned char *)&ourcar + 5, sizeof(ourcar_info_t) - 6);
    // for(i=0;i<NANJING_CAM_NUM;i++){
    //   int len =

    //    }

    return SendDataByTcp(fd, (char *)&ourcar, sizeof(ourcar_info_t));
}

m_timed_func_data outcar_check_info; // 1ms   --> focus on 1 lane
void *callback_outcar_check_info(void *data)
{
    int i = 0;
    int j = 0;
    int n = 0;
    pthread_mutex_lock(&mutex_lock);
    for (i = 0, n = 0; i < CAM_MAX && n < g_cam_num.cam_num; i++)
    {

        if (!g_cam_num.exist[i])
            continue;
        else
        {
            n++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }

        for (j = 0; j < MAX_LANE_NUM; j++)
        {
            if (info.cams[i].lanes[j].coils[0].out_car)
            {
                if (sig_state == SIG_CONNECTED)
                {
                    if (send_outcar_info(sig_fd, i, j) <= 0)
                    {
                        close_socket(&sig_fd);
                        sig_set_state(SIG_PRE_CONNECT);
                    }
                }
                info.cams[i].lanes[j].coils[0].out_car = 0;
            }
        }
    }
    pthread_mutex_unlock(&mutex_lock);
}
//--------- outcar end------------------

typedef void *(func)(void *);
void regist_callback(m_timed_func_data *data, func fc, int ms)
{
    data->func = fc;
    data->time = ms * 1000;
    regist_timed_callback(data);
}

void encode_radar(unsigned char *buff, int *size)
{
    int i = 0, j = 0;
    for (i = 0; i < *size; i++)
    {
        if (buff[i] == 0X7D)
        {
            buff[i] = 0X7D;
            *size = *size + 1;
            for (j = *size - 1; j > i + 1; j--)
            {
                buff[j] = buff[j - 1];
            }
            buff[i + 1] = 0X5D;
        }
        else if (buff[i] == 0X7E)
        {
            buff[i] = 0X7D;
            *size = *size + 1;
            for (j = *size - 1; j > i + 1; j--)
            {
                buff[j] = buff[j - 1];
            }
            buff[i + 1] = 0X5E;
        }
    }
}

int time_seconds()
{
    int seconds = time(NULL);
    return seconds;
}

char *int2str(int t, char *buf)
{
    snprintf(buf, 100, "%d", t);
    return buf;
}

unsigned char buf_radar_result[10000];
unsigned char buf_radar_realtime_result[10000];
unsigned char buf_radar_cycle_result[10000];
#define BUF_IN_OUT_MAX 9
unsigned char buf_radar_in_out_result[BUF_IN_OUT_MAX] = {0};
//radar_car_in_out_status_t radar_in_out_status[MAX_LANE_NUM] = {0};

int get_result_info(int index) //zhouqi  tong ji
{
    radar_head_t head;
    int sz_head = sizeof(radar_head_t);
    unsigned char dev_id = 0;
    dev_id = (unsigned char)get_dev_id();
    dev_id = (dev_id << 2) | 0x01;
    // head data
    head.addr = dev_id;
    head.ver = 0x20;
    head.type = 0x82;
    head.obj = 0x54;
    //
    int lanesize = get_lane_num(0);
    if (lanesize < 1)
        return 0;

    radar_result_t rst;
    int sz_rst = sizeof(radar_result_t);
    memset(&rst, 0, sz_rst);
    //
    if (strlen(g_ivddevsets.devUserNo) > 0)
        rst.dev_no = atoi(g_ivddevsets.devUserNo);
    rst.camera_no = g_camdetect[index].other.camerId;
    rst.period[1] = (g_cycle_statis_time >> 8) & 0xFF;
    rst.period[0] = g_cycle_statis_time & 0xFF; //5閸掑棝鎸
    int tm_run = time_seconds();
    memcpy(rst.time, (unsigned char *)&tm_run, 4);
    rst.lane_count = (unsigned char)lanesize * 2;
    //
    memset(buf_radar_result, 0, 10000);
    memcpy(buf_radar_result + 1, &head, sz_head);
    memcpy(buf_radar_result + 1 + sz_head, &rst, sz_rst);
    //
    radar_result_lane_t result_lane;
    int sz_lane = sizeof(radar_result_lane_t);

    for (int i = 0; i < lanesize; i++)
    {
        for (int j = 0; j < NANJING_LANE_COIL_MAX; j++)
        {
            memset(&result_lane, 0, sizeof(radar_result_lane_t)); //濞屸剝婀侀惃鍕偓鐓庢皑婵夛拷255     by roger
            result_lane.lane_no = (i + 1) + j * lanesize;         //濡偓濞村娅掔痪鍨箑缂傛牕褰
            result_lane.flowA = radar_came_lane[index].lanes[i][j].flowA;
            result_lane.flowB = radar_came_lane[index].lanes[i][j].flowB;
            result_lane.flowC = radar_came_lane[index].lanes[i][j].flowC;
            result_lane.Occupy_rate = radar_came_lane[index].lanes[i][j].Occupy_rate;
            result_lane.flowSum = radar_came_lane[index].lanes[i][j].flowSum;
            result_lane.average_speed = radar_came_lane[index].lanes[i][j].average_speed;
            result_lane.average_len = radar_came_lane[index].lanes[i][j].average_len;
            result_lane.average_head_time = radar_came_lane[index].lanes[i][j].average_head_time;

            memcpy(buf_radar_result + 1 + sz_head + sz_rst + sz_lane * (i * 2 + j), &result_lane, sz_lane);
            //memset(&result_lane,0xFF,sizeof(radar_result_lane_t));
            //result_lane.lane_no = i*2 + 2;
            //memcpy(buf_radar_result+1+sz_head+sz_rst+sz_lane*(i*2 + 1), &result_lane, sz_lane);
            //memcpy(buf_radar_result+1+sz_head+sz_rst+i*sz_lane, &result_lane, sz_lane);
        }
    }

    return sz_head + sz_rst + lanesize * sz_lane * 2; //閺堝寮荤痪鍨箑 by roger
}

int get_cycle_statis_result_info(int index) //add by roger 20190109
{
    radar_head_t head;
    int sz_head = sizeof(radar_head_t);
    unsigned char dev_id = 0;
    dev_id = (unsigned char)get_dev_id();
    dev_id = (dev_id << 2) | 0x01;
    // head data
    head.addr = dev_id;
    head.ver = 0x20;
    head.type = 0x82;
    head.obj = 0x55;
    //
    int lanesize = get_lane_num(index);
    if (lanesize < 1)
        return 0;

    radar_result_t rst;
    int sz_rst = sizeof(radar_result_t);
    memset(&rst, 0, sz_rst);
    //
    if (strlen(g_ivddevsets.devUserNo) > 0)
        rst.dev_no = atoi(g_ivddevsets.devUserNo);
    rst.camera_no = g_camdetect[index].other.camerId;
    rst.period[1] = (g_cycle_statis_time >> 8) & 0xFF;
    rst.period[0] = g_cycle_statis_time & 0xFF; //5閸掑棝鎸
    int tm_run = time_seconds();
    memcpy(rst.time, (unsigned char *)&tm_run, 4);
    rst.lane_count = (unsigned char)lanesize;
    //
    memset(buf_radar_cycle_result, 0, 10000);
    memcpy(buf_radar_cycle_result + 1, &head, sz_head);
    memcpy(buf_radar_cycle_result + 1 + sz_head, &rst, sz_rst);
    //
    radar_cycle_result_lane_t result_lane;
    int sz_lane = sizeof(radar_cycle_result_lane_t);

    for (int i = 0; i < lanesize; i++)
    {
        memset(&result_lane, 0, sizeof(radar_cycle_result_lane_t)); //
        result_lane.lane_no = get_lane_index(index, i);             //
        result_lane.queue_len_max = radar_300_came_lane[index].lanes[i].queue_len_max;
        if (radar_300_came_lane[index].lanes[i].lane_status < 3)
            result_lane.lane_status = radar_300_came_lane[index].lanes[i].lane_status;
        else
            result_lane.lane_status = 0xFE;
        result_lane.car_stop_sum = radar_300_came_lane[index].lanes[i].car_stop_sum;
        result_lane.average_delay_ms = 0;
        result_lane.oil_consume = 0;
        result_lane.gas_emission = 0;

        memcpy(buf_radar_cycle_result + 1 + sz_head + sz_rst + sz_lane * i, &result_lane, sz_lane);
    }

    return sz_head + sz_rst + lanesize * sz_lane;
}

int get_realtime_result_info(int index) //tong ji
{
    radar_head_t head;
    int sz_head = sizeof(radar_head_t);
    int dev_id = 0;

    dev_id = get_dev_id();
    dev_id = (dev_id << 2) | 0x03;
    // head data
    head.ver = 0x20;
    head.type = 0x82;
    head.obj = 0x56;
    head.addr = dev_id;
    //

    int lanesize = get_lane_num(index);
    if (lanesize < 1)
        return 0;
    radar_realtime_t rst;
    int sz_rst = sizeof(radar_realtime_t);
    memset(&rst, 0, sz_rst);
    if (strlen(g_ivddevsets.devUserNo) > 0)
        rst.dev_no = atoi(g_ivddevsets.devUserNo);
    rst.camera_no = g_camdetect[index].other.camerId;
    int tm_run = time_seconds();
    memcpy(rst.time, &tm_run, 4);
    rst.lane_count = lanesize;
    sig_realtime_lane_t lane[lanesize] = {0};
    int i;
    for (i = 0; i < lanesize; i++)
    {
        lane[i].lane_no = get_lane_index(index, i);
        lane[i].queue_len = info.cams[0].lanes[i].queue_len;
        lane[i].head_len = info.cams[0].lanes[i].queue_head_len;
        lane[i].tail_len = info.cams[0].lanes[i].queue_tail_len;
        lane[i].queue_no = info.cams[0].lanes[i].queue_no;
        lane[i].lane_vihicle_count = info.cams[0].lanes[i].veh_no;
        lane[i].ocuppy = info.cams[0].lanes[i].ocupation_ratio;
        lane[i].average_speed = info.cams[0].lanes[i].average_speed;
        lane[i].location = info.cams[0].lanes[i].locate;
        lane[i].head_pos = info.cams[0].lanes[i].head_veh_pos;
        lane[i].head_speed = info.cams[0].lanes[i].head_veh_speed;
        lane[i].tail_pos = info.cams[0].lanes[i].tail_veh_pos;
        lane[i].tail_speed = info.cams[0].lanes[i].tail_veh_speed;
    }

    int sz_bus_num = 0;
    int sz_bus_cnt = 0; //鏉烇妇澧濋弫浼村櫤
    int sz_lane = sizeof(sig_realtime_lane_t);

    memset(buf_radar_realtime_result, 0, 10000);
    memcpy(buf_radar_realtime_result + 1, &head, sz_head);
    memcpy(buf_radar_realtime_result + 1 + sz_head, &rst, sz_rst);

    for (int i = 0; i < lanesize; i++)
    {

        memcpy(buf_radar_realtime_result + 1 + sz_head + sz_rst + i * sz_lane + sz_bus_cnt + sz_bus_num, &lane[i], sz_lane);
#if 1 == DEF_BUS_NUMBER

        *(buf_radar_realtime_result + 1 + sz_head + sz_rst + (i + 1) * sz_lane + sz_bus_cnt) = cam_bus[index].lane_bus[i].bus_cnt; //閸忣兛姘︽潪锔芥殶闁诧拷
        sz_bus_cnt++;

        if (0 == i && cam_bus[index].lane_bus[i].bus_cnt > 0)
        {   //閸旂姾婧呴悧锟
            memcpy(buf_radar_realtime_result + 1 + sz_head + sz_rst + (i + 1) * sz_lane + sz_bus_cnt, &cam_bus[index].lane_bus[i].bus_num, sizeof(bus_number_info) * cam_bus[index].lane_bus[i].bus_cnt);
            sz_bus_num += sizeof(bus_number_info) * cam_bus[index].lane_bus[i].bus_cnt;
        }

#endif
    }

    memset(&cam_bus[index], 0, sizeof(camera_bus_t));
    return sz_head + sz_rst + lanesize * sz_lane + sz_bus_cnt + sz_bus_num;
}

int get_car_in_out_info(int index) //閼惧嘲褰囬崗銉ㄦ簠閸滃苯鍤潪锔炬畱閻樿埖鈧礁鎷伴弫鐗堝祦
{
    bool send_flag = false;
    int pack_len = 0;
    int lane_size = get_lane_num(index);
    if (lane_size < 1)
        return 0;
    radar_car_in_out_result_t car_in_out;
    radar_rt_lane_car_in_out_info_t list_in_out_item;

    if (!get_car_in_out_item(index, &list_in_out_item))
        return 0;

    memset(buf_radar_in_out_result, 0, BUF_IN_OUT_MAX);
    buf_radar_in_out_result[0] = 0xEE; //head flag

    for (int i = 0; i < lane_size; i++)
    {
        memset(&car_in_out, 0, sizeof(radar_car_in_out_result_t));
        if (list_in_out_item.lanes[i].in_flag > 0)
            car_in_out.in_out_flag = 1; //閸忋儴婧
        else if (list_in_out_item.lanes[i].out_flag > 0)
            car_in_out.in_out_flag = 2; //閸戦缚婧
        else
            car_in_out.in_out_flag = 0;

        if (car_in_out.in_out_flag > 0) //鏉烇箓浜鹃崐鍏兼箒娑撳秴鎮撻幍宥呭絺闁菇dp閺佺増宓
            send_flag = true;

        //radar_in_out_status[i].flag = car_in_out.in_out_flag;
        car_in_out.lane_no = get_lane_index(index, i);
        memcpy(buf_radar_in_out_result + 1 + pack_len, &car_in_out, sizeof(radar_car_in_out_result_t));
        pack_len += sizeof(radar_car_in_out_result_t);
    }

    if (!send_flag) //閺佺増宓佸▽鈥冲綁閸栵拷
        return 0;

    return BUF_IN_OUT_MAX; //閸欘亝婀9娑擃亜鐡ч懞锟
}

m_timed_func_data radar_result_info = {0};              // 1 s ---> focus on all camera
m_timed_func_data radar_realtime_result_info = {0};     // 1 s ---> focus on all camera
m_timed_func_data radar_cycle_statis_result_info = {0}; // 1 s ---> focus on all camera
m_timed_func_data radar_car_in_out_result_info = {0};   // 1 s ---> 鏉烇箒绶犻崗銉ュ毉
m_timed_func_data statis_func_data = {0};               //缂佺喕顓搁崗銉ョ氨
m_timed_func_data cycle_minute_func_data = {0};         //閹革拷1,5閸掑棝鎸撶紒鐔活吀
m_timed_func_data cycle_person_func_data = {0};         //閹革拷1,5閸掑棝鎸撶紒鐔活吀
m_timed_func_data cycle_500ms_detector_func_data = {0}; //闁俺顔嗛弶鑳殾娑撶粯甯堕弶锟 500ms閸欐垳绔村▎锟

void *callback_radar_realtime_result(void *data)
{

    // pthread_mutex_lock(&g_all_lock.proto_lock);
    //  int type = g_protocol.type;
    // pthread_mutex_unlock(&g_all_lock.proto_lock);

    //  if (type != PROTO_HAIXIN){
    //   }

    if (g_netInfo.UpServer < 1)
    {
        sleep(1);
        return NULL;
    }

    pthread_mutex_lock(&mutex_lock);
    for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
    {

        if (!g_cam_num.exist[i])
            continue;
        else
        {
            j++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }

        int len = get_realtime_result_info(i);

        if (len > 0 && sig_state == SIG_CONNECTED)
        {
            buf_radar_realtime_result[0] = 0x7e;
            *(buf_radar_realtime_result + 1 + len) = get_crc(buf_radar_realtime_result + 1, len);
            len++; //閺嶏繝鐛欓惍锟
            encode_radar(buf_radar_realtime_result + 1, &len);
            *(buf_radar_realtime_result + 1 + len) = 0x7e;
            pthread_mutex_lock(&yhl_mutex_lock);
            if (send_radar(sig_fd, (char *)buf_radar_realtime_result, len + 2) <= 0)
            {
                close_socket(&sig_fd);
                sig_set_state(SIG_PRE_CONNECT);
            }
            pthread_mutex_unlock(&yhl_mutex_lock);
        }
    }

    pthread_mutex_unlock(&mutex_lock);
}

void *callback_radar_result(void *data)
{
    struct timeval tv;
    volatile uint16 period = g_cycle_statis_time;

    if (period > 0)
    {
        gettimeofday(&tv, NULL);
        int delay_sec = tv.tv_sec % period;

        if (delay_sec > 0)
        {
            sleep(period - delay_sec);
        }
        else
        {
            sleep(period);
        }

        if (g_netInfo.UpServer < 1)
        {
            return NULL;
        }

        pthread_mutex_lock(&mutex_lock);
        //if(tm%g_cycle_statis_tim==0){
        calculate_60s_radar();
        if (sig_state == SIG_CONNECTED)
        {

            for (int i = 0, j = 0; i < CAM_MAX; i++)
            {

                if (!g_cam_num.exist[i])
                    continue;
                else
                {
                    j++;
                    if ((g_camdetect[i].other.detecttype != 2))
                        continue;
                }

                int len = get_result_info(i);
                if (len < 1)
                    continue;

                buf_radar_result[0] = 0x7e;
                *(buf_radar_result + 1 + len) = get_crc(buf_radar_result + 1, len);
                len++; //閺嶏繝鐛欓惍锟
                encode_radar(buf_radar_result + 1, &len);
                *(buf_radar_result + 1 + len) = 0x7e;
                pthread_mutex_lock(&yhl_mutex_lock);
                if (send_radar(sig_fd, (char *)buf_radar_result, len + 2) <= 0)
                {
                    close_socket(&sig_fd);
                    sig_set_state(SIG_PRE_CONNECT);
                }
                pthread_mutex_unlock(&yhl_mutex_lock);
            }
        }
        //}
        pthread_mutex_unlock(&mutex_lock);
    }
}

void *callback_cycle_statis_result(void *data)
{
    struct timeval tv;
    volatile uint16 period = g_cycle_statis_time;

    if (period > 0)
    {
        gettimeofday(&tv, NULL);
        int delay_sec = tv.tv_sec % period;

        if (delay_sec > 0)
        {
            sleep(period - delay_sec);
        }
        else
        {
            sleep(period);
        }

        if (g_netInfo.UpServer < 1)
        {
            return NULL;
        }

        pthread_mutex_lock(&mutex_lock);
        //if(tm%g_cycle_statis_tim==0){
        calculate_300s_radar();
        if (sig_state == SIG_CONNECTED)
        {

            for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
            {

                if (!g_cam_num.exist[i])
                    continue;
                else
                {
                    j++;
                    if ((g_camdetect[i].other.detecttype != 2))
                        continue;
                }

                int len = get_cycle_statis_result_info(i);
                if (len < 1)
                    continue;

                buf_radar_cycle_result[0] = 0x7e;
                *(buf_radar_cycle_result + 1 + len) = get_crc(buf_radar_cycle_result + 1, len);
                len++; //閺嶏繝鐛欓惍锟
                encode_radar(buf_radar_cycle_result + 1, &len);
                *(buf_radar_cycle_result + 1 + len) = 0x7e;
                pthread_mutex_lock(&yhl_mutex_lock);
                if (send_radar(sig_fd, (char *)buf_radar_cycle_result, len + 2) <= 0)
                {
                    close_socket(&sig_fd);
                    sig_set_state(SIG_PRE_CONNECT);
                }
                pthread_mutex_unlock(&yhl_mutex_lock);
            }
        }
        // }
        pthread_mutex_unlock(&mutex_lock);
    }
}

extern int fd_udp_car_in_out;
void *callback_car_in_out_result(void *data)
{
    // pthread_mutex_lock(&mutex_lock);
    for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
    {

        if (!g_cam_num.exist[i])
            continue;
        else
        {
            j++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }

        int len = get_car_in_out_info(i);
        char ip[IPADDRMAX];
        unsigned short port = 0;
        get_udp_info(ip, &port);

        if (len > 0 && fd_udp_car_in_out > 0)
        {
            UdpSendData(fd_udp_car_in_out, ip, port, (char *)buf_radar_in_out_result, len);
        }
    }
    // pthread_mutex_unlock(&mutex_lock);
}

void init_sig_service_haixing()
{
    regist_callback(&radar_realtime_result_info, callback_radar_realtime_result, 1000);
    regist_callback(&radar_result_info, callback_radar_result, 1000);
    regist_callback(&radar_cycle_statis_result_info, callback_cycle_statis_result, 1000);
    //regist_callback(&radar_car_in_out_result_info,callback_car_in_out_result,0);
    regist_callback(&gat920_car_in_out_result_info, gat920_callback_car_in_out_result, 30);

    if (sig_ip[0] != '0')
    {
        sig_set_state(SIG_PRE_CONNECT);
    }
}

void deinit_sig_service_haixing()
{
    if (radar_realtime_result_info.handle != NULL)
    {
        unregist_timed_callback(radar_realtime_result_info.handle);
        radar_realtime_result_info.handle = NULL;
    }
    if (radar_result_info.handle != NULL)
    {
        unregist_timed_callback(radar_result_info.handle);
        radar_result_info.handle = NULL;
    }
    if (radar_cycle_statis_result_info.handle != NULL)
    {
        unregist_timed_callback(radar_cycle_statis_result_info.handle);
        radar_cycle_statis_result_info.handle = NULL;
    }
    if (radar_car_in_out_result_info.handle != NULL)
    {
        unregist_timed_callback(radar_car_in_out_result_info.handle);
        radar_car_in_out_result_info.handle = NULL;
    }
}

void init_sig_service_nanjing()
{
    regist_callback(&camera_queue_info, callback_camera_queue_info, 1000); //1.3.1閹烘帡妲︾悰銊ょ瑐娴肩姵绉烽幁锟
    regist_callback(&outcar_check_info, callback_outcar_check_info, 10);   //鏉╁洩婧呮稉濠佺炊
    regist_callback(&camera_flow_info, callback_camera_flow_info, 1000);   //鏉烇箓浜剧紒鐔活吀娑撳﹣绱舵穱鈩冧紖

    if (sig_ip[0] != '0')
    {
        sig_set_state(SIG_PRE_CONNECT);
    }
}

void deinit_sig_service_nanjing()
{
    if (camera_queue_info.handle != NULL)
    {
        unregist_timed_callback(camera_queue_info.handle);
        camera_queue_info.handle = NULL;
    }
    if (outcar_check_info.handle != NULL)
    {
        unregist_timed_callback(outcar_check_info.handle);
        outcar_check_info.handle = NULL;
    }
    if (camera_flow_info.handle != NULL)
    {
        unregist_timed_callback(camera_flow_info.handle);
        camera_flow_info.handle = NULL;
    }
}

int gat920_get_car_in_out_info(int index) //閼惧嘲褰囬崗銉ㄦ簠閸滃苯鍤潪锔炬畱閻樿埖鈧礁鎷伴弫鐗堝祦
{
    bool send_flag = false;
    int pack_len = 0;
    int subt_len = 0;
    int lane_size; //=get_lane_num(list_in_out_item.index);
    gat920_car_in_out_result_t car_in_out;
    radar_rt_lane_car_in_out_info_t list_in_out_item;

    if (!get_car_in_out_item(index, &list_in_out_item))
        return 0;

    lane_size = get_lane_num(index);
    if (lane_size < 1)
        return 0;
    memset(buf_gat920_in_out_result, 0, BUF_MAX);
    //unsigned char address = (unsigned char)get_dev_id();
    unsigned char address = atoi(g_ivddevsets.devUserNo);
    car_in_out.start_flag = 0x7E;
    car_in_out.adress = (address << 2) | 0x03;
    car_in_out.version = 0x10;
    car_in_out.operation = 0x82;
    car_in_out.obj_flag = 0x08;
    car_in_out.end_flag = 0x7E;

    for (int i = 0; i < lane_size; i++)
    {

        subt_len = 0;
        if ((list_in_out_item.lanes[i].out_flag == 0 && list_in_out_item.lanes[i].in_flag == 0) ||
                (list_in_out_item.lanes[i].in_flag == 1 && radar_in_out_status_920[list_in_out_item.index][i].flag == 1))
        {   //閸欘亜褰傞崗銉ㄦ簠閸滃苯鍤潪锔芥殶閹癸拷
            radar_in_out_status_920[list_in_out_item.index][i].flag = list_in_out_item.lanes[i].in_flag;
            continue;
        }

        if (list_in_out_item.lanes[i].in_flag > 0)
        {
            car_in_out.affair = 1; //閸忋儴婧
            radar_in_out_status_920[list_in_out_item.index][i].flag = 1;
        }
        else if (list_in_out_item.lanes[i].out_flag > 0)
        {
            car_in_out.affair = 0; //閸戦缚婧
            radar_in_out_status_920[list_in_out_item.index][i].flag = 2;
        }

        car_in_out.lane_no = get_lane_index(index, i);

        car_in_out.data_crc[0] = get_crc((unsigned char *)&car_in_out + 1, sizeof(car_in_out) - 4);
        if (car_in_out.data_crc[0] == 0x7D || car_in_out.data_crc[0] == 0x7E)
        {
            car_in_out.data_crc[0] = 0x7D;
            car_in_out.data_crc[1] = 0x5D + car_in_out.data_crc[0] - 0x7D;
        }
        else
        {
            car_in_out.data_crc[1] = car_in_out.end_flag;
            subt_len = 1;
        }

        memcpy(buf_gat920_in_out_result + pack_len, &car_in_out, sizeof(gat920_car_in_out_result_t) - subt_len);
        pack_len += sizeof(gat920_car_in_out_result_t) - subt_len;
        send_flag = true;
        //prt(info,"anger gat920_get_car_in_out_info=%d,in_flag=%d,out_flag=%d,pack_len=%\r\n",list_in_out_item.index, list_in_out_item.lanes[i].in_flag,list_in_out_item.lanes[i].out_flag,pack_len);
    }

    if (!send_flag)
        return 0;
    //prt(info,"angerlen=%d \r\n",pack_len);
    return pack_len;
}

void *gat920_callback_car_in_out_result(void *data)
{
    char ip[IPADDRMAX];
    unsigned short port = 0;
    get_udp_info(ip, &port);

    enum PROTOCOL_TYPE pro_type = (enum PROTOCOL_TYPE)g_ivddevsets.pro_type;

    for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
    {

        if (!g_cam_num.exist[i])
            continue;
        else
        {
            j++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }

        int len = gat920_get_car_in_out_info(i);

        if (pro_type == PROTO_YIHUALU)
        {
            pthread_mutex_lock(&mutex_lock);
            if (len > 0 && fd_udp_car_in_out > 0)
            {
                UdpSendData(fd_udp_car_in_out, ip, port, (char *)buf_gat920_in_out_result, len);
            }
            pthread_mutex_unlock(&mutex_lock);
        }
        else
        {
            if (len > 0 && sig_state == SIG_CONNECTED)
            {
                pthread_mutex_lock(&yhl_mutex_lock);
                if (send_radar(sig_fd, (char *)buf_gat920_in_out_result, len) <= 0)
                {
                    close_socket(&sig_fd);
                    sig_set_state(SIG_PRE_CONNECT);
                }
                pthread_mutex_unlock(&yhl_mutex_lock);
            }
        }
    }
}

#define PER_FRAME_NUMBER 5        //濮ｅ繐顦跨亸鎴濇姎閸掋倖鏌囨稉鈧▎锟
#define QUEUE_FRAME_NUMBER 3      //婢舵艾鐨敮褏鐣婚弰顖涘笓闂冿拷
int get_car_queue_info(int index) //鏉烇箒绶犻幒鎺楁Е娴滃娆
{
    bool send_flag = false;
    int pack_len = 0;
    int subt_len = 0;
    int lane_size = get_lane_num(index);
    if (lane_size)
        return 0;

    gat920_car_in_out_result_t car_in_out;

    memset(buf_gat920_queue_result, 0, BUF_MAX);

    unsigned char address = atoi(g_ivddevsets.devUserNo); //(unsigned char)get_dev_id();
    car_in_out.start_flag = 0x7E;
    car_in_out.adress = (address << 2) | 0x3;
    car_in_out.version = 0x10;
    car_in_out.operation = 0x82;
    car_in_out.obj_flag = 0x08;
    car_in_out.end_flag = 0x7E;

    for (int i = 0; i < lane_size; i++)
    {

        subt_len = 0;
        radar_queue_count[i].frame_number++;
        if (info.cams[index].lanes[i].queue_len > 0)
        {
            radar_queue_count[i].queue_number++;
        }
        else
        {
            radar_queue_count[i].no_queue_number++;
        }

        if (radar_queue_count[i].queue_number > 0 && radar_queue_count[i].no_queue_number > 0)
        {   //娑撳秷绻涚紒锟
            radar_queue_count[i].queue_number = 0;
            radar_queue_count[i].no_queue_number = 0;
            radar_queue_count[i].frame_number = 0;

            continue;
        }

        if (radar_queue_count[i].frame_number != PER_FRAME_NUMBER)
        {   //濞屸冲煂鐢勬殶
            continue;
        }
        else
        {

            if (radar_queue_count[i].queue_number >= QUEUE_FRAME_NUMBER)
            {
                car_in_out.affair = 2; //閹烘帡妲﹂幋鎰彌
            }
            else
            {
                car_in_out.affair = 3; //閹烘帡妲︽稉宥嗗灇缁旓拷
            }

            radar_queue_count[i].queue_number = 0;
            radar_queue_count[i].no_queue_number = 0;
            radar_queue_count[i].frame_number = 0;
        }

        if (radar_queue_count[i].queue_status == car_in_out.affair) //閻╃ǹ鎮撻惃鍕笓闂冪喓濮搁幀锟
            continue;

        radar_queue_count[i].queue_status = car_in_out.affair;

        car_in_out.lane_no = get_lane_index(index, i);
        ;
        car_in_out.data_crc[0] = get_crc((unsigned char *)&car_in_out + 1, sizeof(car_in_out) - 4);

        if (car_in_out.data_crc[0] == 0x7D || car_in_out.data_crc[0] == 0x7E)
        {
            car_in_out.data_crc[0] = 0x7D;
            car_in_out.data_crc[1] = 0x5D + car_in_out.data_crc[0] - 0x7D;
        }
        else
        {
            car_in_out.data_crc[1] = car_in_out.end_flag;
            subt_len = 1;
        }

        memcpy(buf_gat920_queue_result + pack_len, &car_in_out, sizeof(gat920_car_in_out_result_t) - subt_len);
        pack_len += sizeof(gat920_car_in_out_result_t) - subt_len;
        send_flag = true;
    }

    if (!send_flag)
        return 0;

    return pack_len;
}

void *gat920_callback_car_queue_result(void *data)
{
    char ip[IPADDRMAX];
    unsigned short port = 0;
    get_udp_info(ip, &port);

    for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
    {
        if (!g_cam_num.exist[i])
            continue;
        else
        {
            j++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }

        int len = get_car_queue_info(i);

        pthread_mutex_lock(&mutex_lock);
        if (len > 0 && fd_udp_car_in_out > 0)
        {
            UdpSendData(fd_udp_car_in_out, ip, port, (char *)buf_gat920_queue_result, len);
        }
        pthread_mutex_unlock(&mutex_lock);
    }
}

void init_sig_service_yihualu()
{
    regist_callback(&gat920_car_in_out_result_info, gat920_callback_car_in_out_result, 20);
    regist_callback(&gat920_car_queue_result_info, gat920_callback_car_queue_result, 20);
}

void deinit_sig_service_yihualu()
{

    if (gat920_car_in_out_result_info.handle != NULL)
    {
        unregist_timed_callback(gat920_car_in_out_result_info.handle);
        gat920_car_in_out_result_info.handle = NULL;
    }
    if (gat920_car_queue_result_info.handle != NULL)
    {
        unregist_timed_callback(gat920_car_queue_result_info.handle);
        gat920_car_queue_result_info.handle = NULL;
    }
}

/*
void *ps_data_insert_mysql(void *data)
{
    char dt_buf[100] = {0};
    char buf[100] = {0};
    int flow1 = 0;
    int flow2 = 0;
    time_t rawtime;
    struct tm *ptminfo;
    struct timeval tv;

   while(true) {
        volatile uint16 period = g_cycle_statis_time;
        if (period == 0) {
            sleep(3600);
            continue;
        } else {
            gettimeofday(&tv, NULL);
            int delay_sec = tv.tv_sec % period;

            if (delay_sec > 0) {
                sleep(period-delay_sec);
             }else {
                sleep(period);
             }
            get_person_flow(flow1, flow2);
            time(&rawtime);
            ptminfo = localtime(&rawtime);
            sprintf(dt_buf, "%4d-%02d-%02d %02d-%02d-%02d", ptminfo->tm_year + 1900, ptminfo->tm_mon + 1, ptminfo->tm_mday, ptminfo->tm_hour, ptminfo->tm_min, ptminfo->tm_sec);

            sprintf(buf, "insert into PS(PPSum, PNSum, PSum,SST,SP ) values(%d, %d, %d,'%s', %d)", flow1, flow2, flow1 + flow2, dt_buf, g_cycle_statis_time);
            my_mysql_exec(buf);
        }
    }
}

*/

void *tis_data_insert_mysql(int cam_id, void *data)
{
    char dt_buf[100] = {0};
    char buf[300] = {0};
    time_t rawtime;
    struct tm *ptminfo;

    time(&rawtime);
    ptminfo = localtime(&rawtime);
    sprintf(dt_buf, "%4d-%02d-%02d %02d:%02d:%02d", ptminfo->tm_year + 1900, ptminfo->tm_mon + 1, ptminfo->tm_mday, ptminfo->tm_hour, ptminfo->tm_min, ptminfo->tm_sec);

    frame_event_node_t *p_tis = (frame_event_node_t *)data;

    int analyceID = atoi(g_ivddevsets.devUserNo);

    for (int i = 0; i < p_tis->event_num; i++)
    {
        snprintf(buf, 300, "insert into TIS(RecordID, SST, SP,AnalyceID,SAvenue,CameraID,RegionID,TEType,TEPAddr,TEVAddr ) values(%d, '%s', '%d','%d', '%s','%d','%d',%d,'%s','%s')", 0, dt_buf, 60, analyceID, g_ivddevsets.checkaddr, g_camdetect[cam_id].other.camerId, p_tis->event_data[i].region_id, p_tis->event_data[i].type, p_tis->event_data[i].pic_name, p_tis->event_data[i].video_name);
        my_mysql_exec(buf);
        memset(buf, 0, 300);
    }
}

void *callback_cycle_minute_result(void *data)
{
    for (int i = 0; i < CAM_MAX; i++)
    {
        pthread_mutex_lock(&ex_static_info.real_test_lock[i]);
        if ((real_test_run_flag[i] & (unsigned char)EM_RESET) > 0)
        {
            real_test_seconds[i] = 0;
            real_test_run_flag[i] = (unsigned char)EM_RUNNING;
        }

        if (real_test_run_flag[i] == (unsigned char)EM_RUNNING)
        {
            real_test_seconds[i]++;

            if ((real_test_seconds[i] % 60) == 0)
            {
                ex_static_info.real_test_updated[i] |= (unsigned char)EM_ONE_MINUTE;
                prt(info, "time out one: %d abc: %d", i, ex_static_info.real_test_updated[i]);
            }

            if ((real_test_seconds[i] % 300) == 0)
            {
                ex_static_info.real_test_updated[i] |= (unsigned char)EM_FIVE_MINUTE;
                real_test_seconds[i] = 0; //濞擄拷0
                prt(info, "time out five: %d abc: %d", i, ex_static_info.real_test_updated[i]);
            }
        }
        pthread_mutex_unlock(&ex_static_info.real_test_lock[i]);
    }
}

unsigned int pack_crc(unsigned char *data, unsigned int len, unsigned char &crc)
{
    unsigned char buf[500] = {0};
    unsigned int exlen = 1;
    crc = 0;
    buf[0] = data[0];

    for (unsigned int i = 1; i < len; i++, exlen++)
    {
        if (data[i] == 0xC0 || data[i] == 0xDB)
        {
            buf[exlen++] = 0xDB;
            if (data[i] == 0xC0)
                buf[exlen] = 0xDC;
            else
                buf[exlen] = 0xDD;
        }
        else
        {
            buf[exlen] = buf[i];
        }
    }

    if (exlen != len)
    {
        memcpy(data, buf, exlen);
    }

    for (unsigned int i = 1; i < exlen; i++)
    {
        crc ^= data[i];
    }

    return exlen;
}

void person_area_data_hanle(int index, OUTBUF *p_outbuf)
{
    int area_num = 0;
    mAreaPlanInfo planInfo;
    long long ms = get_ms();
    time_t t;
    struct tm *lt;
    time(&t);           //閼惧嘲褰嘦nix閺冨爼妫块幋鐐解偓锟
    lt = localtime(&t); //鏉烆兛璐熼弮鍫曟？缂佹挻鐎妴锟

    pthread_mutex_lock(&realtime_person[index].lock);

    for (int p = 0; p < g_camdetect[index].personarea.num && p < AREAMAX; p++)
    {
        memset(&planInfo, 0, sizeof(mAreaPlanInfo));
        unsigned char prev_hour = 0;
        unsigned char prev_min = 0;
        int z = 0;
        for (z = 0; z < g_personplaninfo[index][p].planTotal; z++)
        {
            if (lt->tm_hour > g_personplaninfo[index][p].plan[z].start_hour ||
                    (lt->tm_hour == g_personplaninfo[index][p].plan[z].start_hour && lt->tm_min > g_personplaninfo[index][p].plan[z].start_minute))
            {
                if (g_personplaninfo[index][p].plan[z].start_hour > prev_hour ||
                        (g_personplaninfo[index][p].plan[z].start_hour == prev_hour && g_personplaninfo[index][p].plan[z].start_minute >= prev_min))
                {
                    planInfo = g_personplaninfo[index][p].plan[z]; //闁瀚ㄩ弮鍫曟？閸栧搫鐓
                    prev_hour = g_personplaninfo[index][p].plan[z].start_hour;
                    prev_min = g_personplaninfo[index][p].plan[z].start_minute;
                }
            }
        }

        realtime_person[index].perso_num[p] = p_outbuf->uPersonRegionNum[p]; //3.1.鐎圭偞妞傜悰灞兼眽濡偓濞村鏆熼幑锟
        realtime_person[index].up_perso_num[p] = p_outbuf->uPersonDirNum[p][0];
        realtime_person[index].down_perso_num[p] = p_outbuf->uPersonDirNum[p][1];

        if (p_outbuf->uPersonRegionNum[p] > 0)
        {

            if (!realtime_person[index].prev_status[p])
            {   //婵″倹妫ゆ禍鐚寸礉闁絼绠炴禒搴ｎ儑娑撯偓娑擃亙姹夋潻娑樺弳閸栧搫鐓欏鈧慨瀣吀閺冩儼顓哥粻锟
                if (!realtime_person[index].start_wait_ms[p])
                {
                    realtime_person[index].start_wait_ms[p] = get_ms();
                    //prt(info, "prev_status: %d", realtime_person[index].start_wait_ms[p]);
                }
            }

            //婵″倸灏崺鐔剁閻╁瓨婀佹禍鐚寸礉闁俺绻冪悰灞兼眽閺傜懓鎮滅痪璺ㄧ埠鐠佲剝娼电涵顔肩暰閿涘苯婀稉鈧规碍妞傞梻鏉戝敶濞屸剝婀佹禍铏圭病鏉╁洩顢戞禍鐑樻煙閸氭垹鍤庨弮锟
            if (p_outbuf->uPersonDirNum[p][0] > 0 || p_outbuf->uPersonDirNum[p][1] > 0)
            {
                realtime_person[index].line_no_person_time[p] = 0;
                realtime_person[index].start_wait_ms[p] = 0;
                //realtime_person[index].start_over_person_limit_ms[p] = 0; //閺堝姹夋潻鍥敎
            }
            else
            {
                if (0 == realtime_person[index].line_no_person_time[p])
                    realtime_person[index].line_no_person_time[p] = get_ms();
                else if ((ms - realtime_person[index].line_no_person_time[p]) >= planInfo.overTime * 1000)
                {
                    if (!realtime_person[index].start_wait_ms[p])
                    {
                        realtime_person[index].start_wait_ms[p] = get_ms();
                        //prt(info, "line_no_person_time: %d", realtime_person[index].start_wait_ms[p]);
                    }
                }
            }

            if (p_outbuf->uPersonRegionNum[p] >= planInfo.personlimit)
            {   //閸栧搫鐓欐禍鐑樻殶婢堆傜艾闂冨牆鈧拷
                if (0 == realtime_person[index].start_over_person_limit_ms[p])
                    realtime_person[index].start_over_person_limit_ms[p] = get_ms();
                realtime_person[index].start_unover_person_limit_ms[p] = 0;
            }
            else
            {
                realtime_person[index].start_over_person_limit_ms[p] = 0;
                if (realtime_person[index].start_unover_person_limit_ms[p] == 0)
                {
                    realtime_person[index].start_unover_person_limit_ms[p] = get_ms();
                }
            }

            if (0 == realtime_person[index].gj_have_person_time[p])
            {   //閸栧搫鐓欓張澶夋眽
                realtime_person[index].gj_have_person_time[p] = get_ms();
            }

            realtime_person[index].prev_status[p] = 1;
            realtime_person[index].no_person_time[p] = 0;
            realtime_person[index].gj_no_person_time[p] = 0;
        }
        else
        {
            realtime_person[index].prev_status[p] = 0;
            realtime_person[index].start_over_person_limit_ms[p] = 0;
            if (realtime_person[index].start_unover_person_limit_ms[p] == 0)
            {
                realtime_person[index].start_unover_person_limit_ms[p] = get_ms();
            }
            realtime_person[index].gj_have_person_time[p] = 0;

            if (0 == realtime_person[index].no_person_time[p])
            {
                realtime_person[index].no_person_time[p] = get_ms();
            }

            if (0 == realtime_person[index].gj_no_person_time[p])
            {   //鏉╁洩顢
                realtime_person[index].gj_no_person_time[p] = get_ms();
            }
            realtime_person[index].line_no_person_time[p] = 0;

            realtime_person[index].start_wait_ms[p] = 0; //閸栧搫鐓欏▽鈩冩箒娴滐拷
        }
        //prt(info, "person_limit_ms[%d]: %d person: %d", p, realtime_person[index].start_over_person_limit_ms[p], p_outbuf->uPersonRegionNum[p]);
        //prt(info, "wait[%d]: %d up person: %d  down: %d", p, realtime_person[index].start_wait_ms[p],p_outbuf->uPersonDirNum[p][0], p_outbuf->uPersonDirNum[p][1] );
        area_num++;
    }

    add_person_flow(index, p_outbuf->uPersonDirNum, realtime_person[index].start_wait_ms);
    pthread_mutex_unlock(&realtime_person[index].lock);
}

void *callback_cycle_person_area_time(void *data)
{
    static unsigned char rev_afer_cnt = 0;
    int area_num = 0;
    mAreaPlanInfo planInfo;

    person_area_time_result_t pact;
    pact.frame_start = 0xC0;
    pact.s_link = 0;
    pact.r_link = 9;
    pact.prot_type = 0x02;
    pact.prot_version = 0x12;
    pact.s_deviceno = 0x30 | (0x0F & get_dev_id());
    pact.r_deviceno = 0x40;
    pact.session_id = 0xFF;
    pact.data_type = 0x20;

    time_t t;
    struct tm *lt;
    time(&t);           //閼惧嘲褰嘦nix閺冨爼妫块幋鐐解偓锟
    lt = localtime(&t); //鏉烆兛璐熼弮鍫曟？缂佹挻鐎妴锟

    long long ms = get_ms();

    for (int j = 0, n = 0; j < CAM_MAX && n < g_cam_num.cam_num; j++)
    {
        if (!g_cam_num.exist[j])
            continue;
        else
            n++;
        memset(pact.person_data, 0, sizeof(mPersonCheckData) * PERSON_AREAS_MAX);
        pthread_mutex_lock(&realtime_person[j].lock);

        area_num = 0;
        pact.area_num = g_camdetect[j].personarea.num;
        pact.cam_direct = g_camdetect[j].other.camdirection;

        for (int p = 0; p < g_camdetect[j].personarea.num && p < AREAMAX; p++)
        {
            unsigned int wait_tm = 0;
            memset(&planInfo, 0, sizeof(mAreaPlanInfo));
            int z = 0;
            unsigned char prev_hour = 0;
            unsigned char prev_min = 0;

            for (z = 0; z < g_personplaninfo[j][p].planTotal; z++)
            {
                if (lt->tm_hour > g_personplaninfo[j][p].plan[z].start_hour ||
                        (lt->tm_hour == g_personplaninfo[j][p].plan[z].start_hour && lt->tm_min > g_personplaninfo[j][p].plan[z].start_minute))
                {
                    if (g_personplaninfo[j][p].plan[z].start_hour > prev_hour ||
                            (g_personplaninfo[j][p].plan[z].start_hour == prev_hour && g_personplaninfo[j][p].plan[z].start_minute >= prev_min))
                    {
                        planInfo = g_personplaninfo[j][p].plan[z]; //闁瀚ㄩ弮鍫曟？閸栧搫鐓
                        prev_hour = g_personplaninfo[j][p].plan[z].start_hour;
                        prev_min = g_personplaninfo[j][p].plan[z].start_minute;
                        //prt(info,"get plan time ok.");
                    }
                }
            }

            pact.person_data[p].area_no = g_camdetect[j].personarea.area[p].id;    //閸栧搫鐓欑紓鏍у娇
            pact.person_data[p].area_person_num = realtime_person[j].perso_num[p]; //閸栧搫鐓欐禍鐑樻殶
            //prt(info, "area_person_num:%d", realtime_person[j].perso_num[p]);

            if (pact.person_data[p].area_person_num > 0)
            {
                pact.person_data[p].occupt = pact.person_data[p].area_person_num * 2; // *100/50; //50娴滆桨璐100%
                if (pact.person_data[p].occupt > 100)
                    pact.person_data[p].occupt = 100;
            }
            else
            {
                pact.person_data[p].occupt = 0;
            }

            //閺堝姹夌涙ê婀崣鎴︹偓锟1閿涘本妫ゆ禍锟0閿涘本妫ゆ禍鐑樻闂傜绉存潻鍥啎缂冾喚娈戦弮鐘辨眽閺冨爼妫块弮璁圭礉閸欐垿鈧拷2
            if (realtime_person[j].no_person_time[p] > 0 && (ms - realtime_person[j].no_person_time[p]) >= planInfo.noPersonTime * 1000)
            {
                pact.person_data[p].person_status = 0x02;
                realtime_person[j].no_person_time[p] = 0;
            }
            else if (realtime_person[j].perso_num[p] > 0)
            {
                pact.person_data[p].person_status = 0x01;
            }
            else
            {
                pact.person_data[p].person_status = 0x0;
            }

            if (1 == g_camdetect[j].other.pensondetecttype)
            {   //缁涘绶熼崠锟

                if (realtime_person[j].start_wait_ms[p] > 0 && (ms - realtime_person[j].start_wait_ms[p]) > planInfo.maxWaitTime * 1000)
                {
                    pact.person_data[p].area_status = 0x01;
                    pact.person_data[p].wait_sec = 0;
                    realtime_person[j].prev_send_area_status[p] = 0x01;
                    realtime_person[j].start_over_person_limit_ms[p] = 0;
                    realtime_person[j].start_wait_ms[p] = 0;
                    realtime_person[j].prev_type[p] = 0x01;
                }
                else if (0x01 == realtime_person[j].prev_type[p])
                {
                    pact.person_data[p].area_status = 0;
                    realtime_person[j].prev_type[p] = 0;
                }

                if (realtime_person[j].start_over_person_limit_ms[p] > 0 &&
                        (ms - realtime_person[j].start_over_person_limit_ms[p]) >= planInfo.overTime * 1000)
                {
                    pact.person_data[p].area_status = 0x01;
                    realtime_person[j].prev_type[p] = 0x02;
                    pact.person_data[p].wait_sec = 0;
                    realtime_person[j].prev_send_area_status[p] = 0x01;
                    realtime_person[j].start_over_person_limit_ms[p] = 0;
                    realtime_person[j].start_wait_ms[p] = 0;
                }
                else if (0x02 == realtime_person[j].prev_type[p])
                {
                    //閻樿埖鈧椒绮1閸欐ɑ鍨0闂団偓鐟曚椒绔村▓鍨闂傝揪绱濋梼鍙夘剾閸欐ê瀵叉径顏勬彥
                    if (realtime_person[j].start_unover_person_limit_ms[p] > 0 && (ms - realtime_person[j].start_unover_person_limit_ms[p]) >= planInfo.overTime * 1000)
                    {   //2second
                        pact.person_data[p].area_status = 0;
                        realtime_person[j].prev_send_area_status[p] = 0;
                        realtime_person[j].prev_type[p] = 0;
                    }
                    else
                    {
                        pact.person_data[p].area_status = realtime_person[j].prev_send_area_status[p];
                    }
                }

                //prt(info, "wait ms: %d second: %d", realtime_person[j].start_wait_ms[p], (ms - realtime_person[j].start_wait_ms[p])/1000);
                if (realtime_person[j].start_wait_ms[p] > 0)
                {   //缁涘绶熼弮鍫曟？缁夛拷
                    pact.person_data[p].wait_sec = (ms - realtime_person[j].start_wait_ms[p]) / 1000; //second
                }
            }
            else if (2 == g_camdetect[j].other.pensondetecttype)
            {   //鐞涘奔姹夋潻鍥敎
                if (realtime_person[j].gj_have_person_time[p] > 0 && (ms - realtime_person[j].gj_have_person_time[p]) >= planInfo.overTime * 1000)
                {
                    pact.person_data[p].area_status = 0x01;
                    realtime_person[j].prev_send_area_status[p] = 0x01;
                    realtime_person[j].gj_have_person_time[p] = 0;
                }
                else
                {
                    if (realtime_person[j].gj_no_person_time[p] > 0 && (ms - realtime_person[j].gj_no_person_time[p]) >= planInfo.overTime * 1000)
                    {
                        pact.person_data[p].area_status = 0;
                        realtime_person[j].prev_send_area_status[p] = 0;
                        realtime_person[j].gj_no_person_time[p] = 0;
                    }
                    else
                    {
                        pact.person_data[p].area_status = realtime_person[j].prev_send_area_status[p];
                    }
                }

                pact.person_data[p].wait_sec = 0;
            }

            if (!cam_info[j].open_alg || realtime_person[j].work_staus > 0)
            {   //閸欘亝婀侀幏澶夌瑝閸掓媽顫嬫０鎴炵ウ閵嗕浇顫嬫０鎴濈磽鐢悶鈧浇鍏樼憴浣稿娴ｅ孩澧犻崣鎴︹偓浣规￥閺佸牊鏆熼幑锟
                pact.person_data[p].work_status = 0x01;
            }
            else
            {
                pact.person_data[p].work_status = 0;
            }

            if (p < TEST_PERSON_AREA_SIZE)
            {   //濞村鐦弫鐗堝祦鐞涘奔姹夌粵澶婄窡閺冨爼妫
                ex_static_info.real_test_info[j].person[p].id = g_camdetect[j].personarea.area[p].id;
                ex_static_info.real_test_info[j].person[p].time = pact.person_data[p].wait_sec;
            }

            area_num++;
        }

        pthread_mutex_unlock(&realtime_person[j].lock);

        if (g_camdetect[j].personarea.num > 0 && g_camdetect[j].personarea.num < AREAMAX)
        {
            unsigned char crc = 0;
            int len = pack_crc((unsigned char *)&pact, 11 + area_num * sizeof(mPersonCheckData), crc);
            unsigned char *p_pact = (unsigned char *)&pact;
            *(p_pact + len) = crc;
            *(p_pact + len + 1) = 0xC0;

            if (sig_fd > 0 && send_radar(sig_fd, (char *)&pact, len + 2) <= 0)
            {
                if (SIG_CONNECTED == sig_get_state())
                {
                    close_socket(&sig_fd);
                    sig_set_state(SIG_PRE_CONNECT);
                }
            }

            if ( rev_afer_cnt > 5) {
                if (sig_fd > 0 && SIG_CONNECTED == sig_get_state()) {
                    char buf[1024] = {0};
                    int ret_val = recv(sig_fd, buf, 1024, 0);

                    if (ret_val < 0) {
                        close_socket(&sig_fd);
                        sig_set_state(SIG_PRE_CONNECT); 
                    }
                }
                rev_afer_cnt = 0;
            }

            rev_afer_cnt++;
        }
    }
}

void *callback_statis_result(void *data)
{
    struct timeval tv;
    volatile uint16 period = g_cycle_statis_time;

    if (period > 0)
    {
        gettimeofday(&tv, NULL);
        int delay_sec = tv.tv_sec % period;

        if (delay_sec > 0)
        {
            sleep(period - delay_sec);
        }
        else
        {
            sleep(period);
        }

        statis_data_insert_mysql();
    }
}

void statis_data_insert_mysql()
{
    //
    time_t rawtime;
    struct tm *ptminfo;
    time(&rawtime);

    int lanesize = 0;
    char buf[1000] = {0};
    char dt_buf[50] = {0};
    Uint16 area_person[MAX_REGION_NUM][MAX_DIRECTION_NUM] = {0};
    long long wait_second[MAX_REGION_NUM] = {0};
    ptminfo = localtime(&rawtime);

    sprintf(dt_buf, "%4d-%02d-%02d %02d:%02d:%02d", ptminfo->tm_year + 1900, ptminfo->tm_mon + 1, ptminfo->tm_mday, ptminfo->tm_hour, ptminfo->tm_min, ptminfo->tm_sec);

    for (int j = 0, n = 0; j < CAM_MAX && n < g_cam_num.cam_num; j++)
    {

        if (!g_cam_num.exist[j])
            continue;
        else
            n++;

        lanesize = get_lane_num(j);
        radar_realtime_lane_t rt_lane[DETECTLANENUMMAX];
        pthread_mutex_lock(&radar_cam_realtime_data[j].mutex_lock);
        memcpy(rt_lane, radar_cam_realtime_data[j].rt_lane, sizeof(radar_realtime_lane_t) * DETECTLANENUMMAX);
        memset((void *)radar_cam_realtime_data[j].rt_lane, 0, sizeof(radar_realtime_lane_t) * DETECTLANENUMMAX);
        pthread_mutex_unlock(&radar_cam_realtime_data[j].mutex_lock);

        for (int i = 0; i < lanesize; i++)
        {
            unsigned road_stat = 0;
            int ocuppy = 0;
            int lane_no = get_lane_index(j, i);
            int average_speed = 0;
            int head_len = 0;
            int head_tm = 0;
            int queue_len = 0;

            //濞夈劍鍓版径鍝ョ埠鐠佲剝妞傞梻鎾６妫帮拷
            //if (radar_cam_realtime_data[j].rt_lane[i].out_car_num > 0 && radar_cam_realtime_data[j].rt_lane[i].lane_vihicle_count > 0) {
            // if (radar_cam_realtime_data[j].rt_lane[i].out_car_num == radar_cam_realtime_data[j].rt_lane[i].lane_vihicle_count)
            //     average_speed = radar_cam_realtime_data[j].rt_lane[i].average_speed /radar_cam_realtime_data[j].rt_lane[i].lane_vihicle_count;
            //  else if( (radar_cam_realtime_data[j].rt_lane[i].lane_vihicle_count - 1) > 0 )
            //     average_speed = radar_cam_realtime_data[j].rt_lane[i].average_speed /(radar_cam_realtime_data[j].rt_lane[i].lane_vihicle_count - 1);
            //   average_speed = radar_cam_realtime_data[j].rt_lane[i].average_speed / radar_cam_realtime_data[j].rt_lane[i].out_car_num;
            // }
            average_speed = rt_lane[i].average_speed;

            if (rt_lane[i].out_car_num > 1)
            {
                head_tm = (rt_lane[i].no_pro_data.head_time % g_statisset.period) / (rt_lane[i].out_car_num - 1);
                head_len = average_speed * head_tm * 5 / 18; //(1000/3600)
            }
            // if (radar_cam_realtime_data[j].rt_lane[i].lane_vihicle_count > 0)
            //     average_speed = radar_cam_realtime_data[j].rt_lane[i].average_speed /radar_cam_realtime_data[j].rt_lane[i].lane_vihicle_count;
            if (g_cycle_statis_time > 0)
            {
                ocuppy = rt_lane[i].no_pro_data.stay_time / (g_statisset.period * 10); //(stay_time*100)/(g_statisset.period *1000 )
                if (ocuppy > 100)
                {   //婢堆傜艾100%
                    //prt(info, "ocuppy[%d]..................%u..........%u.....ms", i, radar_cam_realtime_data[0].rt_lane[i].no_pro_data.stay_time, ocuppy);
                    srand((unsigned)time(NULL));
                    ocuppy = rand() % 11 + 85; //85~~95
                }
            }

            queue_len = rt_lane[i].queue_len;
            if (queue_len > 0)
            {
                queue_len += g_camdetect[j].camdem.recent2stop/100;
            }

            // if (queue_len > 500 || queue_len < 0)//disable by roger 2020-04-23
            if(queue_len < 0)
            {
                queue_len = 0;
            }

            if (rt_lane[i].out_car_num < 2)
            {   //0閹达拷1
                if (0 == head_tm)
                {
                    head_tm = g_statisset.period;
                }
                if (0 == head_len)
                {
                    head_len = (g_camdetect[j].camdem.farth2stop - g_camdetect[j].camdem.recent2stop)/100;
                }
            }

            if (rt_lane[i].out_car_num > 0)
            {
                if (0 == average_speed)
                {
                    srand((unsigned)time(NULL));
                    average_speed = rand() % 6 + 3; //3~8
                }

                if (0 == ocuppy)
                {
                    srand((unsigned)time(NULL));
                    ocuppy = rand() % 4 + 2; //2~5
                }

                if (rt_lane[i].out_car_num > 2)
                {
                    if (0 == head_tm)
                    {
                        srand((unsigned)time(NULL));
                        head_tm = g_statisset.period * (rand() % 3 + 1) / 100; //閸欙拷(缂佺喕顓搁崨銊︽埂*0.01閸滃瞼绮虹拋鈥虫噯閺堬拷*0.03)閻ㄥ嫰娈㈤張鍝勨偓锟
                        if (0 == head_tm)
                        {
                            head_tm = 1; //閼惧嘲褰1
                        }
                    }

                    if (0 == head_len)
                    {
                        srand((unsigned)time(NULL));
                        head_len = rand() % 7 + 4; //6~~10
                    }
                }
            }

            road_stat = queue_len > LANE_QUEUE_LEN ? 1 : 0;
            //prt(info, "*****************average_speed1: %d /lane_vihicle_count: %d = %d", radar_cam_realtime_data[j].rt_lane[i].average_speed, radar_cam_realtime_data[j].rt_lane[i].lane_vihicle_count,average_speed);
            //prt(info, "*****************stay_time: %d /period: %d = %d", radar_cam_realtime_data[j].rt_lane[i].no_pro_data.stay_time, g_statisset.period*10,ocuppy);
            sprintf(buf, "insert into VS(SAvenue, DetectorID, LaneID,Sum, ASpeed, AOccupy, ASpace, ATime,QueuingLength,RoadCondition,SP, SST, CameraId) \
                values('%s', '%s', %d, '%d', '%d' ,'%d', '%d', '%d' ,'%d', '%d', %d, '%s', %d)",
                    g_ivddevsets.checkaddr, g_ivddevsets.devUserNo, lane_no, rt_lane[i].out_car_num, average_speed, ocuppy, head_len, head_tm, queue_len, road_stat, g_statisset.period, dt_buf, g_camdetect[j].other.camerId);
            if (0 == my_mysql_exec(buf))
            {
                int id = my_mysql_last_id();

                if (id > 0)
                {
                    sprintf(buf, "insert into Carlength(id, SuperCar,LargeCar,MidsizeCar,SmallCar,MiniCar) values('%d','%d','%d' ,'%d', '%d', '%d')", id, rt_lane[i].no_pro_data.sup_large_veh,
                            rt_lane[i].no_pro_data.large_veh, rt_lane[i].no_pro_data.mid_veh, rt_lane[i].no_pro_data.small_veh, rt_lane[i].no_pro_data.min_veh);
                    my_mysql_exec(buf);

                    sprintf(buf, "insert into CarModel(id, bus,car,truck) values('%d', '%d', '%d', '%d')", id, rt_lane[i].no_pro_data.bus_num,
                            rt_lane[i].no_pro_data.car_num, rt_lane[i].no_pro_data.truck_num);
                    my_mysql_exec(buf);
                }
            }

            rt_lane[i].average_speed = average_speed;
            rt_lane[i].ocuppy = ocuppy;
            rt_lane[i].head_len = head_len;
            rt_lane[i].head_tm = head_tm;
            rt_lane[i].queue_len = queue_len;
        }

        if (g_netInfo.UpServer)
            send_stat_to_third_server(j, rt_lane);
        //memset((void*)radar_cam_realtime_data[j].rt_lane, 0, sizeof(radar_realtime_lane_t)*DETECTLANENUMMAX);
        // pthread_mutex_unlock(&radar_cam_realtime_data[j].mutex_lock);
        memset(&area_person, 0, sizeof(Uint16) * MAX_REGION_NUM * MAX_DIRECTION_NUM);
        get_person_flow(j, area_person, wait_second);
        for (int m = 0; m < g_camdetect[j].personarea.num; m++)
        {
            unsigned int ts = wait_second[m];
            if (wait_second[m] > g_statisset.period)
                ts = g_statisset.period;
            sprintf(buf, "insert into PS(AnalyceID, SAvenue, RegionID, RecordID, PPSum, PNSum, PSum,SST,SP, CameraId, PDensity) values('%s','%s',%d,3,%d, %d, %d,'%s', %d, %d, %d)", g_ivddevsets.devUserNo, g_ivddevsets.checkaddr, g_camdetect[j].personarea.area[m].id, area_person[m][0], area_person[m][1], area_person[m][0] + area_person[m][1], dt_buf, g_statisset.period, g_camdetect[j].other.camerId, ts);
            my_mysql_exec(buf);
        }
    }
}

void send_event_to_third_server(int cam_id, void *data)
{
    unsigned char crc;
    int data_len = 0;
    char dt_buf[20] = {0};
    char buf[500] = {0};
    time_t rawtime;
    struct tm *ptminfo;
    time(&rawtime);
    ptminfo = localtime(&rawtime);
    sprintf(dt_buf, "%4d-%02d-%02d %02d:%02d:%02d", ptminfo->tm_year + 1900, ptminfo->tm_mon + 1, ptminfo->tm_mday, ptminfo->tm_hour, ptminfo->tm_min, ptminfo->tm_sec);

    third_data_head_t *p_head = (third_data_head_t *)buf;
    p_head->start = 0x7E;
    p_head->version = 0x10;
    p_head->type = htons(0x7003);
    int head_len = sizeof(third_data_head_t);
    third_event_data_t *p_data = (third_event_data_t *)(buf + head_len);
    strncpy(p_data->date_time, dt_buf, 20);
    strncpy(p_data->address, g_ivddevsets.checkaddr, strlen(g_ivddevsets.checkaddr));
    //snprintf(p_data->dector_id, 4, "%s", "66");
    p_data->dector_id = htonl(atoi(g_ivddevsets.devUserNo));
    p_data->cam_id = htonl(g_camdetect[cam_id].other.camerId);

    frame_event_node_t *p_tis = (frame_event_node_t *)data;
    for (int i = 0; i < p_tis->event_num; i++)
    {
        memset(p_data->pic_name, 0, FILENAME_SIZE);
        memset(p_data->video_name, 0, FILENAME_SIZE);
        p_data->event_type = (int)p_tis->event_data[i].type;
        strncpy(p_data->pic_name, p_tis->event_data[i].pic_name, strlen(p_tis->event_data[i].pic_name));
        strncpy(p_data->video_name, p_tis->event_data[i].video_name, strlen(p_tis->event_data[i].video_name));
        data_len = pack_crc((unsigned char *)p_data, sizeof(third_event_data_t), crc);
        p_head->data_len = htonl(data_len);
        buf[head_len + data_len] = crc;
        buf[head_len + data_len + 1] = 0x7E;
        netowrk_send_third_data(buf, head_len + data_len + 2);
    }
}

//
void send_link_to_third_server(int flag)
{
    unsigned char crc;
    char buf[500] = {0};

    third_data_head_t *p_head = (third_data_head_t *)buf;
    p_head->start = 0x7E;
    p_head->version = 0x10;
    p_head->type = htons(0x7005);

    int head_len = sizeof(third_data_head_t);
    third_link_data_t *p_data = (third_link_data_t *)(buf + head_len);
    p_data->dector_id = htonl(atoi(g_ivddevsets.devUserNo));
    p_data->run_status = 1;
    p_data->nomal_status = 0;
    int data_len = pack_crc((unsigned char *)p_data, sizeof(third_link_data_t), crc);
    netowrk_send_third_data(buf, head_len + data_len + 2);
}

void send_stat_to_third_server(int cam_id, radar_realtime_lane_t *p_lane_data)
{
    //
    unsigned char crc;
    char dt_buf[20] = {0};
    char buf[1000] = {0};
    time_t rawtime;
    struct tm *ptminfo;
    time(&rawtime);
    third_data_head_t *p_head = (third_data_head_t *)buf;
    p_head->start = 0x7E;
    p_head->version = 0x10;
    p_head->type = htons(0x7001);

    int head_len = sizeof(third_data_head_t);
    third_stat_data_t *p_data = (third_stat_data_t *)(buf + head_len);
    int lanesize = get_lane_num(cam_id);

    ptminfo = localtime(&rawtime);

    sprintf(dt_buf, "%4d-%02d-%02d %02d:%02d:%02d", ptminfo->tm_year + 1900, ptminfo->tm_mon + 1, ptminfo->tm_mday, ptminfo->tm_hour, ptminfo->tm_min, ptminfo->tm_sec);

    strncpy(p_data->date_time, dt_buf, 20);
    p_data->stat_time = htons(g_statisset.period);
    strncpy(p_data->address, g_ivddevsets.checkaddr, strlen(g_ivddevsets.checkaddr));
    //snprintf((p_data+i)->dector_id, 4, "%s", "66");
    p_data->dector_id = htonl(atoi(g_ivddevsets.devUserNo));
    p_data->cam_id = htonl(g_camdetect[cam_id].other.camerId);
    p_data->lane_num = lanesize;

    third_stat_data_loop_t *p_loop_data = (third_stat_data_loop_t *)(buf + head_len + sizeof(third_stat_data_t));

    int i = 0;
    for (i = 0; i < lanesize; i++)
    {
        int lane_no = get_lane_index(cam_id, i);
        (p_loop_data + i)->lane_id = lane_no;
        (p_loop_data + i)->total_num = htons(p_lane_data[i].out_car_num);
        (p_loop_data + i)->aver_speed = p_lane_data[i].average_speed;
        (p_loop_data + i)->aver_occ_per = p_lane_data[i].ocuppy;
        (p_loop_data + i)->car_head = htons(p_lane_data[i].head_len);
        (p_loop_data + i)->car_head_tm = htons(p_lane_data[i].head_tm);
        (p_loop_data + i)->bus_num = htons(p_lane_data[i].no_pro_data.bus_num);
        (p_loop_data + i)->car_num = htons(p_lane_data[i].no_pro_data.car_num);
        (p_loop_data + i)->truck_num = htons(p_lane_data[i].no_pro_data.truck_num);
        (p_loop_data + i)->motor_num = 0;
        (p_loop_data + i)->bicycle_num = 0;
        (p_loop_data + i)->queue_len = htons(p_lane_data[i].queue_len);
        (p_loop_data + i)->road_stat = 0;
    }

    int data_len = pack_crc((unsigned char *)p_data, sizeof(third_stat_data_t) + i * sizeof(third_stat_data_loop_t), crc);
    p_head->data_len = htonl(data_len);
    buf[head_len + data_len] = crc;
    buf[head_len + data_len + 1] = 0x7E;

    netowrk_send_third_data(buf, head_len + data_len + 2);
}

int ReportedRealStatic(int sock)
{
    unsigned char dev_id = 0;
    int len = 1;
    int size, i, j, ret, n;
    unsigned char mysbuf[100] = {0};
    memset(mysbuf, 0, sizeof(mysbuf));
    FrameHeader *fhead = (FrameHeader *)mysbuf;
    size = sizeof(FrameHeader) - 1;
    fhead->mSenderLinkId = 0x00;
    fhead->mRecieverLinkId = LINKID_UART_BETWEEN_COMBOARD_AND_MAINCONTLBOARD;
    fhead->mProtocolType = PROTOCOTYPE_VIDEO_DETECTOR;
    fhead->mProtocolVersion = 0x12;
    dev_id = get_dev_id();
    fhead->mSenderDeviceID.mDeviceIndex = dev_id & 0x0F;
    fhead->mSenderDeviceID.mDeviceClass = 0x1;
    fhead->mRecieverDeviceId.mDeviceIndex = 0x0;
    fhead->mRecieverDeviceId.mDeviceClass = 0x4;
    fhead->mSessionID = 0xFF;
    fhead->mDataType = 0x01;

    struct timeval tv;
    gettimeofday(&tv, NULL);
    long now_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    holder[i].last_time = now_time;

    for (i = 0, n = 0; i < CAM_MAX && n < g_cam_num.cam_num; i++)
    {
        if (!g_cam_num.exist[i])
            continue;
        else
            n++;

        if (get_holder_status(i) == 0)
        {
            set_holder_status(i, 0x01);
            detector_static_data(i);
            continue;
        }
        else if (get_holder_status(i) == 0x01)
        {
            set_holder_status(i, 0x02);
            detector_static_data(i);
        }
        else
        {
            prt(info, "detector holder status error!");
        }

        //if(get_cam_status(i)){
        size = sizeof(FrameHeader) - 1;

        RealTimeTrafficData *tmpReal = (RealTimeTrafficData *)(mysbuf + sizeof(FrameHeader));
        holder[i].traffic_rst.lane_num = get_lane_num(i);
        tmpReal->mDetectChannelCount = holder[i].traffic_rst.lane_num;
        //  prt(info,"%d",  tmpReal->mDetectChannelCount);
        size += 1;
        //  pthread_mutex_lock(&holder[i].sig_data_lock);

        for (j = 0; j < tmpReal->mDetectChannelCount; j++)
        {

            struct EachChannelPack *ptrChannel = (struct EachChannelPack *)(mysbuf + sizeof(FrameHeader) + 1 + j * sizeof(EachChannelPackm));
            holder[i].traffic_rst.Eachchannel[j].mWorkStatusOfDetectChannle.bDataIsValid = get_cam_running_state(i);
            memcpy((void *)ptrChannel, (void *)&holder[i].traffic_rst.Eachchannel[j], sizeof(EachChannelPackm));
            size += sizeof(EachChannelPackm);
        }

        memset(holder[i].traffic_rst.Eachchannel, 0, sizeof(EachChannelPackm) * CHANNELMAXNUM);
        memset(&holder[i].traffic_rst.car_info, 0, sizeof(m_car_info));

        //holder[i].last_time = now_time;
        set_holder_status(i, 0);
        //  pthread_mutex_unlock(&holder[i].sig_data_lock);
        externalProtocolAddCrcCode(mysbuf + 1, &size);
        externalProtocolEncode(mysbuf + 1, &size);
        externalProtocolAddHeader(mysbuf, &size);

        len = SendDataInLock(sock, (char *)mysbuf, size);

        return len;
        // }
    }
    return len;
}

void *callback_cycle_500ms_detector(void *data)
{
    if (sig_fd > 0 && ReportedRealStatic(sig_fd) < 0)
    {
        if (SIG_CONNECTED == sig_get_state())
        {
            close_socket(&sig_fd);
            sig_set_state(SIG_PRE_CONNECT);
        }
    }
}

void statis_handle()
{
    regist_callback(&statis_func_data, callback_statis_result, 1000);
    regist_callback(&cycle_minute_func_data, callback_cycle_minute_result, 1000);

    if (PROTO_HUAITONG == g_ivddevsets.pro_type)
    {
        regist_callback(&cycle_500ms_detector_func_data, callback_cycle_500ms_detector, 250);
    }

    if (PROTO_HUAITONG_PERSON == g_ivddevsets.pro_type)
    {
        regist_callback(&cycle_person_func_data, callback_cycle_person_area_time, 1000);
    }
}

void init_server()
{
    pthread_t pid;
    init_udp();
    init_sig_client();
    init_watchdog();
#if 1 == DEF_BUS_NUMBER
    create_detach_thread(init_udp_server, 1, NULL);
#endif

    if (PROTO_JIERUI == g_ivddevsets.pro_type)
        create_detach_thread(init_g920_udp_server, 1, NULL);
 
    if (PROTO_HUAITONG == g_ivddevsets.pro_type)
    {
        create_detach_thread(sig_client_handle_thread, 1, NULL);
        create_detach_thread(message_handle_thread, 1, NULL);
    }
    create_detach_thread(sig_service_thread, 1, NULL); //濡偓濞村娅--闁矮淇婇崡鈩冨瘹娴犮倕顦╅悶锟
    create_detach_thread(http_server, 1, NULL);        //http閺堝秴濮熼崳锟
    create_detach_thread(websocket_server, 1, NULL);   //websocket閺堝秴濮熼崳锟
    create_detach_thread(radar_client, 1, NULL);       //闂嗙柉鎻广垺鍩涚粩锟
    create_detach_thread(pic_video_handle, 1, NULL);   //閸ュ墽澧栨稉搴ゎ潒妫版垵顦╅悶锟
    // create_detach_thread(ping_server, 1, NULL);   //閸ュ墽澧栨稉搴ゎ潒妫版垵顦╅悶锟
    // create_detach_thread(ws_hanle_realtime_data,1,NULL); //鐎圭偞妞傞弫鐗堝祦婢跺嫮鎮婃稉搴″絺闁拷
    //non motor server
#if 1 == RECEIVE_VS
    pthread_create(&pid, NULL, non_motor_udp_server, NULL);
#endif

}

void init_ntp()
{
    pthread_t pid;

    if (strlen(g_ivdntp.ipaddr) > 0 && g_ivdntp.cycle > 0)
    {
        pthread_create(&pid, NULL, start_ntpclient, NULL);
        //start_ntpclient(g_ivdntp.ipaddr,g_ivdntp.cycle);// g_ivdntp.cycle*60*60);
    }
}

void cancel_proto(unsigned char type)
{

    switch ((int)type)
    {
    case PROTO_PRIVATE:
    case PROTO_HAIXIN:
    {
        deinit_sig_service_haixing(); //1-濞磋渹淇婇崡蹇氼唴
    }
    break;
    case PROTO_YIHUALU:
    {
        deinit_sig_service_yihualu(); //2-閺勬挸宕曡ぐ锟
    }
    break;
    case PROTO_NANJING:
    {
        deinit_sig_service_nanjing(); //3-閸楁ぞ鍚懢杈ㄦ焿
    }
    break;
    }
}

void *gat920_callback_jierui_realtime_result(void *data)
{
    char ip[IPADDRMAX];
    unsigned short port = 0;
    get_udp_info(ip, &port);

    enum PROTOCOL_TYPE pro_type = (enum PROTOCOL_TYPE)g_ivddevsets.pro_type;

    for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++)
    {
        if (!g_cam_num.exist[i])
            continue;
        else
        {
            j++;
            if ((g_camdetect[i].other.detecttype != 2))
                continue;
        }

        jierui_rt_lane_car_data_info_t list_data_item;
        if ( !get_jierui_car_item(i, &list_data_item) )
            continue;

        pthread_mutex_lock(&mutex_lock);

        for (int m = 4; m < 5; m++) {
            int len = gat920_get_jierui_car_info(i, m, list_data_item);
            if (len > 0 && fd_udp_car_in_out > 0)
            {
                UdpSendData(fd_udp_car_in_out, ip, port, (char *)gat920_jierui_data_buf, len);
            }
        }

        pthread_mutex_unlock(&mutex_lock);
    }
}

void *gat920_callback_jierui_60s_result(void *data)
{
    char ip[IPADDRMAX];
    unsigned short port = 0;

    while (1) {
        if (g_statisset.period < 1) {
            sleep(1);
            continue;
        }
   
        sleep(g_statisset.period - 5);
        get_udp_info(ip, &port);    

        for (int i = 0, j = 0; i < CAM_MAX && j < g_cam_num.cam_num; i++) {
            if (!g_cam_num.exist[i])
                continue;
            else {
                j++;
                if ((g_camdetect[i].other.detecttype != 2))
                    continue;
            }

            pthread_mutex_lock(&mutex_lock);

            int len = gat920_calculate_60s_jierui(i);
            if (len > 0 && fd_udp_car_in_out > 0) {
                UdpSendData(fd_udp_car_in_out, ip, port, (char *)gat920_jierui_60s_buf, len);
            }

            pthread_mutex_unlock(&mutex_lock);
        }
    }
}

void init_sig_service_jierui()
{
    regist_callback(&gat920_jierui_realtime_result_info, gat920_callback_jierui_realtime_result, 1000);
    regist_callback(&gat920_jierui_60s_result_info, gat920_callback_jierui_60s_result, 0);
}

void protocol_select(unsigned char type)
{
    protocol_sel = type;
    enum PROTOCOL_TYPE pro_type = (enum PROTOCOL_TYPE)type;

    switch (pro_type)
    {
    case PROTO_PRIVATE:
    case PROTO_HAIXIN:
    {
        init_sig_service_haixing(); //1-濞磋渹淇婇崡蹇氼唴
    }
    break;
    case PROTO_YIHUALU:
    {
        init_sig_service_yihualu(); //2-閺勬挸宕曡ぐ锟
    }
    break;
    case PROTO_NANJING:
    {
        init_sig_service_nanjing(); //3-閸楁ぞ鍚懢杈ㄦ焿
    }
    break;
    case PROTO_HUAITONG:
    case PROTO_HUAITONG_PERSON:
    {
    }
    break;
    case PROTO_PRIVATE_PERSON:
    {
    }
    break;
    case PROTO_WS_RADAR:
    case PROTO_WS_VIDEO:
    case PROTO_WS_RADAR_VIDEO:
    {
        create_queue();
    }
    break;
    case PROTO_JIERUI:
    {
        init_sig_service_jierui();
    }
    break;
    default:
        protocol_sel = PROTO_NONE;
        break;
    }
}

void protocol_change(unsigned char type, unsigned char old_type)
{
    cancel_proto(old_type);
    protocol_select(type);
}

void send_message(int type, char *buf)
{
    prt(info, "send message");

    struct msg_st msg_data;
    msg_data.msg_type = type;
    if (buf != NULL && strlen(buf) > 0)
    {
        strncpy(msg_data.text, buf, strlen(buf));
    }

    int msgid = msgget((key_t)12333, 0777 | IPC_CREAT);
    if (msgid == -1)
    {
        prt(info, "msgget failed with error: %d\n", errno);
    }

    if (msgid > -1 && msgsnd(msgid, (void *)&msg_data, MAX_MESSAGE_TEXT, 0) == -1)
    {
        prt(info, "msg send failed\n");
    }
}

void *message_handle_thread(void *data)
{
    long int msgtype = 0; //濞夈劍鍓1
    struct msg_st msg_data;

    int msgid = msgget((key_t)12333, 0777 | IPC_CREAT);
    if (msgid == -1)
    {
        prt(info, "msgget failed with error: %d\n", errno);
        return NULL;
    }

    while (1)
    {
        if (msgrcv(msgid, (void *)&msg_data, BUFSIZ, msgtype, 0) == -1)
        {
            prt(info, "msgrcv failed with errno: %d\n", errno);
        }

        switch (msg_data.msg_type)
        {
        case 0x01:
        {
            prt(info, "handle message.");
            ReportedCammerStatus(sig_fd, 0xFF);
        }
        break;
        default:
            prt(info, "no this message.");
            break;
        }
    }
}

char bcc(char *data, unsigned int size)
{
    char crc = 0;

    for (int i = 0; i < size; i++)
    {
        crc ^= data[i];
    }

    return crc;
}

#define N 128
void *init_udp_server(void *data)
{
    struct sockaddr_in client_addr;
    socklen_t addrlen = sizeof(struct sockaddr_in);
    char buf[N] = {0};

    //bzero  memset
    bzero(&client_addr, sizeof(client_addr));

    int server_fd = UdpCreateSocket(11160);

    if (server_fd < 0)
    {
        prt(err, "create upd server failed.");
        return NULL;
    }

    while (1)
    {
        unsigned short bus_type_size = 0;
        int rec_num = recvfrom(server_fd, buf, N, 0, (struct sockaddr *)&client_addr, &addrlen);

        if (rec_num < 0)
        {
            prt(err, "fail to recvfrom");
            usleep(300);
            continue;
        }

        bus_type_size = sizeof(recv_bus_t);
        if (rec_num != bus_type_size || buf[0] != 0x43 || buf[1] != 0x4D || buf[2] != 0x2A) //閺佺増宓侀梹鍨娑撳秴顕
            continue;

        recv_bus_t *p_pack = (recv_bus_t *)buf;
        char crc = bcc(p_pack->data, MAX_BUS_DATA); //閺嶏繝鐛欐径杈Е
        if (crc != buf[bus_type_size - 1])
            continue;

        int cam_id = 0;
        cam_id |= buf[3];

        for (int i = 0; i < CAM_MAX; i++)
        {

            if (cam_id == g_camdetect[i].other.camerId)
            {
                int index = cam_bus[i].lane_bus[0].bus_cnt % MAX_LANE_BUS_NUM; //缁楊兛绔存稉顏囨簠闁拷,鏉烇妇澧濇稉宥堢Т鏉╋拷50
                cam_bus[i].lane_bus[0].bus_num[index].rfid = cam_id;
                cam_bus[i].lane_bus[0].bus_num[index].bus_no = index + 1;
                strncpy(cam_bus[i].lane_bus[0].bus_num[index].bus_number, p_pack->data, 12); //鏉烇妇澧濋梹鍨12鐎涙濡
                cam_bus[i].lane_bus[0].bus_cnt++;
                break;
            }
        }

        //printf("%s ---> %d\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
    }

    close(server_fd);

    return NULL;
}

void *init_g920_udp_server(void *data)
{
    char cmd_buf[20] = {0};
    struct sockaddr_in client_addr;
    socklen_t addrlen = sizeof(struct sockaddr_in);
    char buf[N] = {0};

    //bzero  memset
    bzero(&client_addr, sizeof(client_addr));

    int server_fd = UdpCreateSocket(7166);

    if (server_fd < 0)
    {
        prt(err, "create upd server failed.");
        return NULL;
    }

    while (1)
    {
        unsigned short bus_type_size = 0;
        int rec_num = recvfrom(server_fd, buf, N, 0, (struct sockaddr *)&client_addr, &addrlen);

        if (rec_num < 0)
        {
            prt(err, "fail to recvfrom");
            usleep(300);
            continue;
        }

        if (rec_num < 4 || buf[1] != 0x10 || buf[2] != 0x81 ) //閺佺増宓侀梹鍨娑撳秴顕
            continue;

        if (buf[3] == 0x02) {
            int crc = get_crc((unsigned char *)&buf[1], 8);
            if (crc != buf[9])
                continue;

            sprintf(cmd_buf, "date -d @%d", *((int*)&buf[5])  );
            system(buf);
        }
        //printf("%s ---> %d\n", inet_ntoa(client_addr.sin_addr), ntohs(client_addr.sin_port));
    }

    close(server_fd);

    return NULL;
}

//
void netowrk_send_third_data(char *pack, int len)
{
    pthread_mutex_lock(&g_member.third_lock);
    if (g_member.sock_fd > 0)
    {

        if (send(g_member.sock_fd, pack, len, MSG_NOSIGNAL) < 0)
        {
            close(g_member.sock_fd);
            g_member.sock_fd = creat_client_sock(g_netInfo.strIpaddr2, g_netInfo.strPort);
        }
    }
    else
    {
        g_member.sock_fd = creat_client_sock(g_netInfo.strIpaddr2, g_netInfo.strPort);
    }

    pthread_mutex_unlock(&g_member.third_lock);
}

void pic_data_sub_insert_mysql(char *parent_id, void *data)
{

    char buf[300] = {0};
    int ihat = 0;
    int ioverload = 0;

    NonMotorInfo *p_data = (NonMotorInfo*)data;

    int analyceID = atoi(g_ivddevsets.devUserNo);
    if (p_data->hasHelmet)
        ihat = 1;

    if (p_data->overLoad)
        ioverload = 1;

    snprintf(buf, 300, "insert into MPS_Natt(RecordID, PSum,CPSum,CPState,PSOut,NPlate,YPlate ) values('%s', '%d','%d','%d','%d','%s','%d')", parent_id, p_data->riderNum, p_data->helmetNum, ihat, ioverload, "0", 0);
    my_mysql_exec(buf);

}

#if 0
void pic_data_insert_mysql(int cam_id, void *data, char *pic_path)
{
    char dt_buf[100] = {0};
    char buf[300] = {0};
    char uid[50] = {0};
    time_t rawtime;
    struct tm *ptminfo;

    NPOUTBUF *p_data = (NPOUTBUF *)data;

    if (!p_data)
        return;

    get_uid(uid);

    time(&rawtime);
    ptminfo = localtime(&rawtime);
    sprintf(dt_buf, "%4d-%02d-%02d %02d:%02d:%02d", ptminfo->tm_year + 1900, ptminfo->tm_mon + 1, ptminfo->tm_mday, ptminfo->tm_hour, ptminfo->tm_min, ptminfo->tm_sec);

    int analyceID = atoi(g_ivddevsets.devUserNo);

    if (p_data->uNonMotorNum > 0)
    {
        snprintf(buf, 300, "insert into MPS(SST, AnalyceID,SAvenue,CameraID,NSum,RecordID,TEPAddr ) values('%s', '%d','%s', '%d','%d','%s','%s')", dt_buf, analyceID, g_ivddevsets.checkaddr, g_camdetect[cam_id].other.camerId, p_data->uNonMotorNum, uid, pic_path);

        my_mysql_exec(buf);

        for (int i = 0; i < p_data->uNonMotorNum; i++) {
            pic_data_sub_insert_mysql(uid, (void *)&p_data->nonMotorInfo[i]);
        }

    } else {
        snprintf(buf, 300, "insert into MPS(SST, AnalyceID,SAvenue,CameraID,NSum,RecordID,TEPAddr ) values('%s', '%d','%s', '%d','%d','%s','%s')", dt_buf, analyceID, g_ivddevsets.checkaddr, g_camdetect[cam_id].other.camerId, 0, uid, pic_path);
        my_mysql_exec(buf);
    }
}
#endif

void pic_data_insert_mysql(int cam_id, void *data, char *pic_path)
{
    char dt_buf[100] = {0};
    char buf[300] = {0};
    char tmp_buf[2][20] = {0};
    char dest[2][200] = {0};
    char uid[50] = {0};
    Uint16  rideMaxNum[2] = {0}; /*ÏÂ±ê0±íÊ¾µ±Ç°Í¼Æ¬ÖÐ×î´óÆïÐÐÈËÊý£¬1±íÊ¾×î´óÆïÐÐÖÐ´÷Ã±ÈËÊý,2±íÊ¾ÊÇ·ñ´÷Ã±£¬3±íÊ¾ÊÇ·ñ³¬ÔØ*/
    time_t rawtime;
    struct tm *ptminfo;

    NPOUTBUF *p_data = (NPOUTBUF *)data;

    if (!p_data)
        return;

    get_uid(uid);

    time(&rawtime);
    ptminfo = localtime(&rawtime);
    sprintf(dt_buf, "%4d-%02d-%02d %02d:%02d:%02d", ptminfo->tm_year + 1900, ptminfo->tm_mon + 1, ptminfo->tm_mday, ptminfo->tm_hour, ptminfo->tm_min, ptminfo->tm_sec);

    int analyceID = atoi(g_ivddevsets.devUserNo);

    if (p_data->uNonMotorNum > 0)
    {
        snprintf(buf, 300, "insert into MPS(SST, AnalyceID,SAvenue,CameraID,NSum,RecordID,TEPAddr ) values('%s', '%d','%s', '%d','%d','%s','%s')", dt_buf, analyceID, g_ivddevsets.checkaddr, g_camdetect[cam_id].other.camerId, p_data->uNonMotorNum, uid, pic_path);

        my_mysql_exec(buf);

        //  pic_data_sub_insert_count(uid, (void *)p_data,rideMaxNum);

#if 0
        rideMaxNum[0] = p_data->nonMotorInfo[0].riderNum;
        rideMaxNum[1] = p_data->nonMotorInfo[0].helmetNum;
        rideMaxNum[2] = 1; //Ä¬ÈÏ´÷Ã±
        rideMaxNum[3] = 0;//Ä¬ÈÏÃ»ÓÐ³¬ÔØ

        for (int i = 0; i < p_data->uNonMotorNum; i++)
        {
            if (p_data->nonMotorInfo[i].riderNum > rideMaxNum[0])
            {
                rideMaxNum[0] = p_data->nonMotorInfo[i].riderNum;
                rideMaxNum[1] = p_data->nonMotorInfo[i].helmetNum;
            }
            if (!p_data->nonMotorInfo[i].hasHelmet)
                rideMaxNum[2] = 0;
            if (p_data->nonMotorInfo[i].overLoad)
                rideMaxNum[3] = 1;
        }
#endif
        for (int i = 0; i < p_data->uNonMotorNum; i++)
        {
            rideMaxNum[0] += p_data->nonMotorInfo[i].riderNum;
            rideMaxNum[1] += p_data->nonMotorInfo[i].helmetNum;

            if (p_data->nonMotorInfo[i].hasHelmet)
                snprintf(tmp_buf[0],20, " %d:是", i+1);
            else
               snprintf(tmp_buf[0],20, " %d:否", i+1); 
         
            if (p_data->nonMotorInfo[i].overLoad) 
              snprintf(tmp_buf[1],20, " %d:是", i+1);
            else
               snprintf(tmp_buf[1],20," %d:否", i+1); 

            strcat(dest[0], tmp_buf[0]);
            strcat(dest[1], tmp_buf[1]);
            memset(tmp_buf, 0, sizeof(tmp_buf));
        }

        snprintf(buf, 300, "insert into MPS_Natt(RecordID, PSum,CPSum,CPState,PSOut,NPlate,YPlate ) values('%s', '%d','%d','%s','%s','%s','%d')", uid, rideMaxNum[0], rideMaxNum[1], dest[0], dest[1], "0", 0);
        my_mysql_exec(buf);


    } else {
        rideMaxNum[0] = p_data->nonMotorInfo[0].riderNum;
        rideMaxNum[1] = p_data->nonMotorInfo[0].helmetNum;

        if (p_data->nonMotorInfo[0].hasHelmet)
            snprintf(tmp_buf[0],20, " No1: Y");
        else
           snprintf(tmp_buf[0],20, "No1: N"); 
     
        if (p_data->nonMotorInfo[0].overLoad) 
          snprintf(tmp_buf[1],20, " No1: Y");
        else
           snprintf(tmp_buf[1],20," No1: N"); 

        strcat(dest[0], tmp_buf[0]);
        strcat(dest[1], tmp_buf[1]);
     
        snprintf(buf, 300, "insert into MPS(SST, AnalyceID,SAvenue,CameraID,NSum,RecordID,TEPAddr ) values('%s', '%d','%s', '%d','%d','%s','%s')", dt_buf, analyceID, g_ivddevsets.checkaddr, g_camdetect[cam_id].other.camerId, 0, uid, pic_path);
        my_mysql_exec(buf);
        snprintf(buf, 300, "insert into MPS_Natt(RecordID, PSum,CPSum,CPState,PSOut,NPlate,YPlate ) values('%s', '%d','%d','%s','%s','%s','%d')", uid, rideMaxNum[0], rideMaxNum[1], dest[0], dest[1], "0", 0);
        my_mysql_exec(buf);
    }
}

//
int gat920_get_jierui_car_info(int index, int type, jierui_rt_lane_car_data_info_t &list_data_item) //閼惧嘲褰囬崗銉ㄦ簠閸滃苯鍤潪锔炬畱閻樿埖鈧礁鎷伴弫鐗堝祦
{
    int pack_len = 0;
    int lane_size; //=get_lane_num(list_in_out_item.index);

    // jierui_rt_lane_car_data_info_t list_data_item;

    // if ( !get_jierui_car_item(index, &list_data_item) )
    //     return 0;

    lane_size = get_lane_num(index);
    if (lane_size < 1)
        return 0;

    memset(gat920_jierui_data_buf, 0, BUF_MAX);
    //unsigned char address = (unsigned char)get_dev_id();
    unsigned char address = atoi(g_ivddevsets.devUserNo);
    gat920_jierui_data_buf[0] = 0x7E;
    gat920_jierui_data_buf[1] = (address << 2) | 0x03; //address
    gat920_jierui_data_buf[2] = 0x10; //version
    gat920_jierui_data_buf[3] = 0x82; //operation

    switch (type)
    {
        case 1:
                gat920_jierui_data_buf[4] = 0x46; //car in out
            break;
        case 2:
            gat920_jierui_data_buf[4] = 0x41; //realtime speed
            break;
        case 3:
            gat920_jierui_data_buf[4] = 0x42; //queue len
            break;
        case 4:
            gat920_jierui_data_buf[4] = 0x40; //road status
            break;
        default:
            break;
    }
    
    gat920_jierui_data_buf[5] = lane_size; //lane number

    pack_len = 6;

    for (int i = 0; i < lane_size; i++)
    {
        gat920_jierui_data_buf[pack_len] = get_lane_index(index, i);

        switch (type)
        {
        case 1: //car in out
            {
                if (list_data_item.lanes[i].out_flag || list_data_item.lanes[i].in_flag){
                    if (list_data_item.lanes[i].speed > 40)
                        gat920_jierui_data_buf[pack_len + 1] = 1;
                    else if (list_data_item.lanes[i].speed > 25)
                        gat920_jierui_data_buf[pack_len + 1] = 2;
                    else 
                        gat920_jierui_data_buf[pack_len + 1] = 3;
                }else{
                    gat920_jierui_data_buf[pack_len + 1] = 0;
                }
            }
            break;
        case 2: //speed 
            {
                 gat920_jierui_data_buf[pack_len + 1] = list_data_item.lanes[i].speed;
                 if (gat920_jierui_data_buf[pack_len + 1] == 0x7D || gat920_jierui_data_buf[pack_len + 1] == 0x7E) {
                    pack_len++;
                    gat920_jierui_data_buf[pack_len] = 0x5D + gat920_jierui_data_buf[pack_len] - 0x7D;
                 }
            }
            break;
        case 3: //queue len
            {
                 gat920_jierui_data_buf[pack_len + 1] = list_data_item.lanes[i].queue_len;
                 if (gat920_jierui_data_buf[pack_len + 1] == 0x7D || gat920_jierui_data_buf[pack_len + 1] == 0x7E) {
                    pack_len++;
                    gat920_jierui_data_buf[pack_len] = 0x5D + gat920_jierui_data_buf[pack_len] - 0x7D;
                 }
            }
            break;
        case 4: //road status
            {
                if (list_data_item.lanes[i].queue_len > 60) //拥堵
                    gat920_jierui_data_buf[pack_len + 1] = 3;
                else if (list_data_item.lanes[i].queue_len > 20) //密集
                    gat920_jierui_data_buf[pack_len + 1] = 2;
                else if (list_data_item.lanes[i].queue_len > 5) //畅通
                    gat920_jierui_data_buf[pack_len + 1] = 1;
                else
                    gat920_jierui_data_buf[pack_len + 1] = 0; //非常畅通
            }
            break;
        
        default:
            break;
        }
        pack_len += 2;
    }

    if (lane_size > 0) {
       gat920_jierui_data_buf[pack_len] =  get_crc((unsigned char *)(gat920_jierui_data_buf + 1), pack_len - 1);
       pack_len++;
       if(gat920_jierui_data_buf[pack_len] == 0x7D || gat920_jierui_data_buf[pack_len] == 0x7E)
        {
            gat920_jierui_data_buf[pack_len] = 0x5D + gat920_jierui_data_buf[pack_len] - 0x7D;
            pack_len++;
        }
    }else
    {
        return 0;
    }

    gat920_jierui_data_buf[pack_len] = 0x7E;
    pack_len++;

    return pack_len;
}

int gat920_calculate_60s_jierui(int index)
{
    int pack_len = 0;
    int lane_size; //=get_lane_num(list_in_out_item.index);

    jierui_60s_t data60;

    if ( !get_jierui_60s_item(index, &data60))
        return 0;

    lane_size = get_lane_num(index);
    if (lane_size < 1)
        return 0;

    memset(gat920_jierui_60s_buf, 0, BUF_MAX);
    unsigned char address = atoi(g_ivddevsets.devUserNo);
    gat920_jierui_60s_buf[0] = 0x7E;
    gat920_jierui_60s_buf[1] = (address << 2) | 0x03; //address
    gat920_jierui_60s_buf[2] = 0x10; //version
    gat920_jierui_60s_buf[3] = 0x82; //operation
    gat920_jierui_60s_buf[4] = 0x05;
    //
     struct timeval tv;
    gettimeofday(&tv,NULL);
    gat920_jierui_60s_buf[5] = (unsigned char)tv.tv_sec & 0xFF;
    gat920_jierui_60s_buf[6] = (unsigned char)(tv.tv_sec>>8) & 0xFF;
    gat920_jierui_60s_buf[7] = (unsigned char)(tv.tv_sec>>16) & 0xFF;
    gat920_jierui_60s_buf[8] = (unsigned char)(tv.tv_sec>>24) & 0xFF;
    //
    *((unsigned short *)&gat920_jierui_60s_buf[9]) = g_statisset.period;
    pack_len = 11;
    gat920_jierui_60s_buf[pack_len] = g_statisset.large*10;
    if (gat920_jierui_60s_buf[pack_len] == 0x7D || gat920_jierui_60s_buf[pack_len] == 0x7E) {
        pack_len++;
        gat920_jierui_60s_buf[pack_len] = 0x5D + gat920_jierui_60s_buf[pack_len-1] - 0x7D;
    }
    pack_len++;
    gat920_jierui_60s_buf[pack_len] = g_statisset.mediu*10;
    if (gat920_jierui_60s_buf[pack_len] == 0x7D || gat920_jierui_60s_buf[pack_len] == 0x7E) {
        pack_len++;
        gat920_jierui_60s_buf[pack_len] = 0x5D + gat920_jierui_60s_buf[pack_len-1] - 0x7D;
    }
    pack_len++;
    gat920_jierui_60s_buf[pack_len] = g_statisset.small*10;
    pack_len+=4;
    //
    pack_len++;
    gat920_jierui_60s_buf[pack_len] = lane_size;
  
    for (int i = 0; i < lane_size; i++)
    {
        pack_len++;
        gat920_jierui_60s_buf[pack_len] = get_lane_index(index, i);
        pack_len++;
        gat920_jierui_60s_buf[pack_len] = (unsigned char)data60.data.lane_data[i][0].car_a_sum;
        pack_len++;
        gat920_jierui_60s_buf[pack_len] = (unsigned char)data60.data.lane_data[i][0].car_b_sum;
        pack_len++;
        gat920_jierui_60s_buf[pack_len] = (unsigned char)data60.data.lane_data[i][0].car_c_sum;
        pack_len++;
        gat920_jierui_60s_buf[pack_len] = (unsigned char)(data60.data.lane_data[i][0].exist_duration * 200 / g_cycle_statis_time / 1000);

        if (gat920_jierui_60s_buf[pack_len] == 0x7D || gat920_jierui_60s_buf[pack_len] == 0x7E) {
            pack_len++;
            gat920_jierui_60s_buf[pack_len] = 0x5D + gat920_jierui_60s_buf[pack_len-1] - 0x7D;
        }
        pack_len++;
        if (data60.data.lane_data[i][0].pass_number > 0)
            gat920_jierui_60s_buf[pack_len] = (unsigned char)data60.data.lane_data[i][0].speed_sum / data60.data.lane_data[i][0].pass_number;
        else  
            gat920_jierui_60s_buf[pack_len] = 0;

        if (gat920_jierui_60s_buf[pack_len] == 0x7D || gat920_jierui_60s_buf[pack_len] == 0x7E) {
            pack_len++;
            gat920_jierui_60s_buf[pack_len] = 0x5D + gat920_jierui_60s_buf[pack_len-1] - 0x7D;
        }
        pack_len++;
        if (data60.data.lane_data[i][0].pass_number > 0)
            gat920_jierui_60s_buf[pack_len] = (unsigned char)(data60.data.lane_data[i][0].veh_len_sum / data60.data.lane_data[i][0].pass_number) * 10;
        else
            gat920_jierui_60s_buf[pack_len] = 0;

        if (gat920_jierui_60s_buf[pack_len] == 0x7D || gat920_jierui_60s_buf[pack_len] == 0x7E) {
            pack_len++;
            gat920_jierui_60s_buf[pack_len] = 0x5D + gat920_jierui_60s_buf[pack_len-1] - 0x7D;
        }
        pack_len++;
        if (data60.data.lane_data[i][0].pass_number > 0)
            gat920_jierui_60s_buf[pack_len] = (unsigned char)(data60.data.lane_data[i][0].head_time_sum / data60.data.lane_data[i][0].pass_number) * 10;
        else
            gat920_jierui_60s_buf[pack_len] = 0;
        
        if (gat920_jierui_60s_buf[pack_len] == 0x7D || gat920_jierui_60s_buf[pack_len] == 0x7E) {
            pack_len++;
            gat920_jierui_60s_buf[pack_len] = 0x5D + gat920_jierui_60s_buf[pack_len-1] - 0x7D;
        }
        pack_len++;
        gat920_jierui_60s_buf[pack_len] = data60.Lane_queue_len[i];
        pack_len +=4;//stay

    }

    if (lane_size > 0) {
        pack_len++;
        gat920_jierui_60s_buf[pack_len] =  get_crc((unsigned char *)(gat920_jierui_60s_buf + 1), pack_len - 1);
        pack_len++;
        if(gat920_jierui_60s_buf[pack_len-1] == 0x7D || gat920_jierui_60s_buf[pack_len-1] == 0x7E)
        {
            gat920_jierui_60s_buf[pack_len] = 0x5D + gat920_jierui_60s_buf[pack_len-1] - 0x7D;
            pack_len++;
        }
    }else
    {
        return 0;
    }

    gat920_jierui_60s_buf[pack_len] = 0x7E;
    pack_len++;

    return pack_len;
 }


// int do_ping(char *ip)
// {
//     pid_t pid;
//     int status;
// 	int i = 0;

//     for(i = 0; i < 5; i++) {

//         if(pid = vfork() < 0) {
//             print("vfork error.\n");
//             continue;
//         }
        
//         if (pid == 0){
//             if (execlp("ping", "ping", "-c", "1", ip, (char*)0) < 0) {
//                 printf("ip: %s not exist.\n", ip);
//                 exit(1);
//             }
//         }

//         waitpid(pid, &status, 0);

//         if (status == 0)
//             return 0;

//         sleep(1);
//     }

//     return -1;
// }


int go_ping(char *svrip)
{
        int i = 0;
        while(i < 2)
        {
                pid_t pid;
                if ((pid = vfork()) < 0)
                {
                        printf("vfork error");
                        exit(1);
                }
                else if (pid == 0)
                {
                        if ( execlp("ping", "ping","-c", "1",svrip, (char*)0) < 0)
                        {
                                printf("execlp error\n");
                                exit(1);
                        }
                }

                int stat;
                waitpid(pid, &stat, 0);

                if (stat == 0)
                {
                        return 0;
                }
                sleep(1);
                i++;
        }
        return -1;
}


 void *ping_server(void *data)
{
	while(true) {

        // int sockfd;
        // if((sockfd = socket(AF_INET, SOCK_STREAM, 0)) < 0){
        //     perror("create_sockfd error");
        //     exit(EXIT_FAILURE);
        // }

        // struct sockaddr_in servaddr;
        // memset(&servaddr, 0, sizeof(servaddr));
        // servaddr.sin_family = AF_INET;
        // if(inet_pton(AF_INET, "127.0.0.1", &servaddr.sin_addr) <= 0){
        //     perror("convert_servaddr error");
        //     exit(EXIT_FAILURE);
        // }
        // servaddr.sin_port = htons(SERV_PORT);

        // if(connect(sockfd, (struct sockaddr *)&servaddr, sizeof(servaddr)) < 0){
        //     perror("connect_sockfd error");
        //     exit(EXIT_FAILURE);
        // }

        // char buf[BUF_SIZE];
        // int nbuf = strlen(argv[1]);
        // send(sockfd, argv[1], nbuf, 0);
        // nbuf = recv(sockfd, buf, BUF_SIZE, 0);
        // buf[nbuf] = 0;
        // printf("echo msg: %s\n", buf);


        if (strlen(g_netInfo.strIpaddrIO) > 0) {
            // int ret = do_ping(g_netInfo.strIpaddrIO);
            // int ret = go_ping(g_netInfo.strIpaddrIO);
            // if (ret == -1) {
            //     sig_set_state(SIG_PRE_CONNECT);
            // }

            int i_socket = socket(AF_INET, SOCK_STREAM, 0);
            if (i_socket < 0)
            {
                printf("Create Socket Failed!\n");
                sleep(2);
                continue;
            }




        }
	}
}





