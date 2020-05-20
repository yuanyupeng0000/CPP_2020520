#include <pthread.h>
#include <stdio.h>
#include <unistd.h>
#include<stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <sys/ioctl.h>
#include <fcntl.h>
#include <stropts.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <errno.h>
#include <time.h>
#include <sys/time.h>
#include <libssh2.h>
#include "common.h"
#include "cam_alg.h"
#include "client_net.h"
#include "sig_service.h"
#include "camera_service.h"
#include "ini/fvdconfig.h"
#include "g_fun.h"
#include "tcp_server.h"
#include "url_downfile.h"
#include "ptserver/non_motor_server.h"


#define LOCAL_NTPCLIENT_PORT 10009
#define BORDER_SIZE  1000 //m
#define FILE_NUM     1000
#define FILE_LEFT    200

/////////////////////////////////////////////////////
pthread_mutex_t client_sock_lock;
unsigned char prt_log_leve = 3; //"0">关闭日志 "1">全部信息"2">调试信息"3">一般信息"4">警告信息
g_all_lock_t g_all_lock;
////////////////////////////////////////////////////
mGlobalVAR g_member; //全局变量

extern IVDDevSets g_ivddevsets;
extern mThirProtocol g_protocol;
extern IVDNetInfo g_netInfo;
extern mAlgParam  algparam[CAM_MAX];
extern IVDDevInfo       g_sysInfo;
extern IVDTimeStatu     g_timestatus;
extern TimeSetInterface        g_timeset;
extern IVDStatisSets    g_statisset;
extern RS485CONFIG      g_rs485;
extern mCamDetectParam g_camdetect[CAM_MAX];
extern m_camera_info cam_info[CAM_MAX];
extern IVDNTP g_ivdntp;
extern IVDCAMERANUM     g_cam_num;
extern EX_mRealStaticInfo ex_static_info;
extern mPersonPlanInfo  g_personplaninfo[CAM_MAX][PERSON_AREAS_MAX];
extern mEventInfo events_info[CAM_MAX];
extern mRealTimePerson realtime_person[CAM_MAX];
///////////////////////////////
extern m_holder holder[CAM_MAX];
extern camera_bus_t cam_bus[CAM_MAX];
//////////////////////////////////////////////
extern pthread_mutex_t gpu_lock;
#if (DECODE_TYPE == 2)
extern ffmpeg_object_type_t g_ffmpeg_obj[CAM_MAX];
#endif


//初始化全局变量
void  init_variable()
{
    memset(&g_ivddevsets, 0, sizeof(IVDDevSets));
    memset(&g_statisset, 0, sizeof(IVDStatisSets));
    memset(&g_sysInfo, 0, sizeof(IVDDevInfo));
    memset(&g_timestatus, 0, sizeof(IVDTimeStatu));
    memset(&g_rs485, 0, sizeof(RS485CONFIG));
    memset(&g_timeset, 0, sizeof(TimeSetInterface));
    memset(&g_camdetect, 0, sizeof(mCamDetectParam)*CAM_MAX);

//  memset(&g_protocol, 0, sizeof(mThirProtocol));
    memset(&g_netInfo, 0, sizeof(IVDNetInfo));
    memset(algparam, 0, sizeof(mAlgParam)*CAM_MAX);
    memset(cam_info, 0, sizeof(m_camera_info)*CAM_MAX);
    memset(&g_ivdntp, 0, sizeof(IVDNTP));
    memset(&g_cam_num, 0, sizeof(IVDCAMERANUM));
    memset(&ex_static_info, 0, sizeof(EX_mRealStaticInfo));
    memset(g_personplaninfo, 0, sizeof(mPersonPlanInfo)*CAM_MAX * PERSON_AREAS_MAX);
    memset(events_info, 0, sizeof(mEventInfo)*CAM_MAX);
    memset(cam_bus, 0, sizeof(camera_bus_t)*CAM_MAX);
    //
    g_ivddevsets.overWrite = 0;
    g_ivddevsets.autoreset = 8;
    //
    memset(realtime_person, 0, sizeof(mRealTimePerson)*CAM_MAX);
    //
#if (DECODE_TYPE == 2)
    memset(g_ffmpeg_obj, 0, sizeof(ffmpeg_object_type_t)*CAM_MAX);
#endif

    g_member.cmd_play = false;
    g_member.radar_sock = 0;

    g_member.sock_fd = creat_client_sock(g_netInfo.strIpaddr2, g_netInfo.strPort);
    pthread_mutex_init(&g_member.third_lock, NULL);

    struct timeval tv;
    gettimeofday(&tv, NULL);
    long now_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;

    for (int i = 0; i < CAM_MAX; i++) {
        pthread_mutex_init(&cam_info[i].cmd_lock, NULL);
        pthread_mutex_init(&cam_info[i].run_lock, NULL);
        pthread_mutex_init(&cam_info[i].frame_lock, NULL);
        pthread_mutex_init(&cam_info[i].file_lock, NULL);
        pthread_mutex_init(&ex_static_info.lock[i], NULL);
        pthread_mutex_init(&ex_static_info.real_test_lock[i], NULL);
        //pthread_mutex_init(&cam_info[i].frame_lock_ex, NULL);
        pthread_mutex_init(&cam_info[i].open_one_lock, NULL);

        holder[i].last_time = now_time;
        holder[i].status = 0;
        pthread_mutex_init(&holder[i].sig_data_lock, NULL);
        pthread_mutex_init(&realtime_person[i].lock, NULL);
#if 0
        if (g_ivddevsets.pro_type == PROTO_WS_RADAR || g_ivddevsets.pro_type == PROTO_WS_RADAR_VIDEO) {
            cam_info[i].p_radar = InitQueue();// camera radar data queue
        }
#endif
        cam_info[i].p_5s_frame_queue = InitQueue();  //前5秒帧数
        cam_info[i].p_f_e_queue = InitQueue();  //帧与对应的事件列表
        cam_info[i].p_file_queue = InitQueue(); //写入文件列表
        read_passid_from_file(cam_info[i].pass_id, i);
        int len = strlen(cam_info[i].pass_id);
        if (len > 0 && cam_info[i].pass_id[len-1] == '\n' ) {
            cam_info[i].pass_id[len-1] = 0;
        }

        cam_info[i].curr_stat = CAM_RESET_NONE;
        cam_info[i].cmd_fd = 0;
        cam_info[i].rtsp_fag = 0;
    }
    pthread_mutex_init(&client_sock_lock, NULL);
    pthread_mutex_init(&gpu_lock, NULL);

    init_yuv_list();
    init_non_motor_list();
}


//删除日志文件
void delete_log_files(char *dirname, int left)
{
    char buf[100]; memset(buf, 0, 100);
    sprintf(buf, "cd %s ; ls -tp  | grep -v '/$' | tail -n +%d | xargs -d '\n' -r rm --;cd -", dirname, left );
    system(buf);
}

void handle_log_file()
{
    char cmd[50] = {0};
    char line[50] = {0};
    FILE *fp;
    int del_num = 0;

    strcpy(cmd, "du -sm log");
    fp = popen(cmd, "r");
    if (fp && fgets(line, 20, fp) != NULL) {
        pclose(fp);

        char *pstr = strchr(line , 'l');
        if (pstr != NULL)
            *pstr = '\0';

        int total_size = atoi(line);

        if (total_size > BORDER_SIZE) {

            strcpy(cmd, "find log -type f -print | wc -l");
            fp = popen(cmd, "r");
            if (fp && fgets(line, 30, fp) != NULL) {
                pclose(fp);
                del_num = atoi(line) / 2; //删掉一半

                if (del_num > 0)
                    delete_log_files("log", del_num);
            }
        }

    }

    strcpy(cmd, "find /ftphome/pic -type f -print | wc -l");
    fp = popen(cmd, "r");
    if (fp && fgets(line, 30, fp) != NULL) {
        pclose(fp);
        del_num = atoi(line);

        if (del_num > FILE_NUM)
            delete_log_files("/ftphome/pic", FILE_LEFT);
    }

    strcpy(cmd, "find /ftphome/video -type f -print | wc -l ");
    fp = popen(cmd, "r");
    if (fp && fgets(line, 30, fp) != NULL) {
        pclose(fp);
        del_num = atoi(line);

        if (del_num > FILE_NUM)
            delete_log_files("/ftphome/video", FILE_LEFT);
    }
}

void set_serial(int sno, unsigned int buadrate, int databit, int stopbit, int checkbit)
{
    char buf[100] = {0};
    char checkbuf[20] = {0};

    if (databit < 5 || databit > 8)
        databit = 8;

    switch (checkbit) {
    case 0:
        break;
    case 1:
        sprintf(checkbuf, " parodd");
        break;
    case 2:
        sprintf(checkbuf, " -parodd");
        break;
    }

    if (1 == stopbit )
        stopbit = 45;
    else
        stopbit = 32;

    sprintf(buf, "stty -F //dev//ttyS%d %u cs%d %s %ccstopb  -echo", sno, buadrate, databit, checkbuf, stopbit);

    system(buf);
}


int set_ip_dns(char *ip, char *mask, char *dns1, char *dns2, char *gateway)
{
    int iret = 0;

    char buf[100] = {0};
    char ret[20] = {0};

#if 0
    if (!strcmp(ip, g_netInfo.strIpaddr) && !strcmp(mask, g_netInfo.strNetmask) && !strcmp(gateway, g_netInfo.strGateway)) {
        iret = 0;
    } else { //改变路由IP

        char bufip[100]; memset(bufip, 0, 100); sprintf(bufip, "uci set network.wan.ipaddr=%s", ip);
        char bufNetmask[100]; memset(bufNetmask, 0, 100); sprintf(bufNetmask, "uci set network.wan.netmask=%s", mask);
        char bufGateway[100]; memset(bufGateway, 0, 100); sprintf(bufGateway, "uci set network.wan.gateway=%s", gateway);
        if ( exec_ssh_cmd("10.10.10.1", "root", "admin", bufip) != 0)
            iret = -1;
        if ( exec_ssh_cmd("10.10.10.1", "root", "admin", bufNetmask) != 0 )
            iret = -1;
        if ( exec_ssh_cmd("10.10.10.1", "root", "admin", bufGateway) != 0)
            iret = -1;
        if ( exec_ssh_cmd("10.10.10.1", "root", "admin", "uci commit") != 0)
            iret = -1;
        if ( exec_ssh_cmd("10.10.10.1", "root", "admin", "/etc/init.d/network restart") != 0)
            iret = -1;
    }
#endif

#if 1
    char *file     =  "/etc/network/interfaces";
    char *file_dns =  "/etc/resolvconf/resolv.conf.d/base";

    if (ip[0] != 0) {
        sprintf(buf, "sed -i \"s/^address .*$/address %s/g\"  %s", ip, file);
        system(buf);
        memset(buf, 0, 100);
    }

    if (mask[0] != 0) {
        sprintf(buf, "sed -i \"s/^netmask .*$/netmask %s/g\"  %s", mask, file);
        system(buf);
        memset(buf, 0, 100);
    }

    if (gateway[0] != 0) {
        sprintf(buf, "sed -i \"s/^gateway .*$/gateway %s/g\"  %s", gateway, file);
        system(buf);
        memset(buf, 0, 100);
    }


    //sprintf(buf, "wc -l /etc/resolvconf/resolv.conf.d/base");
    // FILE *fp=popen(buf,"r");
    //if(fgets(ret, 20, fp) == NULL)
    // return;
    // pclose(fp);

    memset(buf, 0, 100);
    sprintf(buf, "echo '' >> /etc/resolvconf/resolv.conf.d/base");
    for (int i = 0; i < 2; i ++) {
        system(buf);
    }

    if (dns1[0] != 0) {
        memset(buf, 0, 100);
        sprintf(buf, "sed -i '1c %s' %s", dns1, file_dns);
        system(buf);

    }

    if (dns2[0] != 0) {
        memset(buf, 0, 100);
        sprintf(buf, "sed -i '2c %s' %s", dns2, file_dns);
        system(buf);

    }

    memset(buf, 0, 100);
    sprintf(buf, "/etc/init.d/networking restart");
    system(buf);

    if (ip[0] != 0 && mask[0] != 0) {
        sprintf(buf, "ifconfig enp1s0 %s netmask %s ", ip, mask);
        system(buf);
    }

    if (gateway[0] != 0) {
        sprintf(buf, "route add default gw %s", gateway);
        system(buf);
    }
#endif

    if (-1 == iret) { //改为本地IP
        sprintf(buf, "/etc/init.d/networking restart");
        system(buf);
    }

    return 0; //不重启
}

bool change_sig_ip(char *ip, unsigned int port, char *ip2, unsigned int port2)
{
    bool ret = false;

    if (0 != strcmp(ip, g_netInfo.strIpaddrIO) || (port != g_netInfo.strPortIO) ) {
        ret = true;
    }

    if (0 != strcmp(ip2, g_netInfo.strIpaddr2) || (port2 != g_netInfo.strPort) ) {
        ret = true;
    }

    return ret;
}

void kill_process()
{
    system("killall -9 zenith");
}

void reboot_system()
{
    system("reboot");
}

int  get_random_port()//7001-8000
{
    struct timeval tpstart;

    gettimeofday(&tpstart, NULL);

    srand(tpstart.tv_usec);

    return (7000 + 1 + (int) (1000.0 * rand() / (RAND_MAX + 1.0)));
//
//  srand(time(0));
//  printf("%d\n",rand()%100+1);
}

int create_joinable_thread(THREAD_ENTITY callback, int level, void *data)
{
    pthread_t tid;
    pthread_attr_t      attr;
    struct sched_param  schedParam;

#if 1
    /* Initialize the thread attributes */
    if (pthread_attr_init(&attr)) {
        //__D("Failed to initialize thread attrs\n");
        return -1;
    }
    if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE))
    {
        //__D("Failed to set PTHREAD_CREATE_DETACH\n");
        return -1;
    }
    /* Force the thread to use custom scheduling attributes */
    if (pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED)) {
        //__D("Failed to set schedule inheritance attribute\n");
        return -1;
    }

//    /* Set the thread to be fifo real time scheduled */
    if (pthread_attr_setschedpolicy(&attr, SCHED_FIFO)) {
        //__D("Failed to set FIFO scheduling policy\n");
        return -1;
    }
    schedParam.sched_priority = sched_get_priority_max(SCHED_FIFO) - level;
    if (pthread_attr_setschedparam(&attr, &schedParam)) {
        //__D("Failed to set scheduler parameters\n");
        return -1;
    }
#endif
    //if(pthread_create(&tid,&attr,callback, data))
    if (pthread_create(&tid, NULL, callback, data))
    {
        return -1;
    }
    return tid;
}

int create_detach_thread(THREAD_ENTITY callback, int level, void *data)
{
    pthread_t tid;
    pthread_attr_t      attr;
    struct sched_param  schedParam;

#if 0
    /* Initialize the thread attributes */
    if (pthread_attr_init(&attr)) {
        //__D("Failed to initialize thread attrs\n");
        return -1;
    }
    if (pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED))
    {
        //__D("Failed to set PTHREAD_CREATE_DETACH\n");
        return -1;
    }
    /* Force the thread to use custom scheduling attributes */
    if (pthread_attr_setinheritsched(&attr, PTHREAD_EXPLICIT_SCHED)) {
        //__D("Failed to set schedule inheritance attribute\n");
        return -1;
    }

//    /* Set the thread to be fifo real time scheduled */
    if (pthread_attr_setschedpolicy(&attr, SCHED_FIFO)) {
        //__D("Failed to set FIFO scheduling policy\n");
        return -1;
    }
    schedParam.sched_priority = sched_get_priority_max(SCHED_FIFO) - level;
    if (pthread_attr_setschedparam(&attr, &schedParam)) {
        //__D("Failed to set scheduler parameters\n");
        return -1;
    }
#endif
    //if(pthread_create(&tid,&attr,callback, data))
    if (pthread_create(&tid, NULL, callback, data))
    {
        return -1;
    }
    if (data != NULL)
        ((m_timed_func_data*)data)->handle = tid;

    return 0;
}

void sig_handle(int signo,   siginfo_t *info, void* p)
{
    prt(err, "get signal %d", signo);
    if (signo == SIGSEGV) {
        exit(1);
    }
    if (signo == SIGPIPE) {
        prt(err, "ignore sig pipe");
    }
}
void watch_sig(int signo, void (*SignalAction)(int, siginfo_t*, void*))
{
#if 1
    struct sigaction act;
    sigemptyset(&act.sa_mask);

    act.sa_flags = SA_SIGINFO;
    act.sa_sigaction = SignalAction;
    sigaction(signo, &act, NULL);
#endif
}
void init_sig()
{
    watch_sig(SIGSEGV, sig_handle);//11
    watch_sig(SIGALRM, sig_handle);//14
    watch_sig(SIGTERM, sig_handle);//15
    watch_sig(SIGPWR , sig_handle);//30
    watch_sig(SIGKILL , sig_handle);
    watch_sig(SIGPIPE , sig_handle);//30
}

void *timed_fun(void *data)
{
    m_timed_func_data *p_data = (m_timed_func_data *) data;
    while (1) {
        usleep(p_data->time);
        p_data->func(p_data->data);
        pthread_testcancel();
    }
}

int regist_timed_callback(m_timed_func_data *p_ctx)
{
    return create_detach_thread(timed_fun, 1, p_ctx);
}

void unregist_timed_callback(pthread_t p)
{
    pthread_cancel(p);
    pthread_join(p, NULL);
}

m_timed_func_data *regist_timed_func(int time_us, void *ptr, void *data)
{
    m_timed_func_data *p_d = (m_timed_func_data *)malloc(sizeof(m_timed_func_data));
    p_d->data = data;
    p_d->func = (THREAD_ENTITY )ptr;
    p_d->time = time_us;
    return p_d;
}

pthread_t start_timed_func(m_timed_func_data *p_ctx)
{
    return  create_detach_thread(timed_fun, 1, p_ctx);
}
pthread_t start_detached_func(void *func, void* data)
{
    return (pthread_t)create_detach_thread((THREAD_ENTITY)func, 1, data);
}
pthread_t start_delayed_func(void *func, void* data, int delay_ms)
{
    usleep(delay_ms);
    return (pthread_t)create_detach_thread((THREAD_ENTITY)func, 1, data);
}
void stop_timed_func(m_timed_func_data *p_ctx)
{

    pthread_cancel(p_ctx->handle);
    free(p_ctx);
}

m_list_info *new_list(int data_size, void *func)
{
    m_list_info *info_p = (m_list_info *)malloc(sizeof(m_list_info));
    info_p->head = NULL;
    info_p->cur = NULL;
    info_p->tail = NULL;
    info_p->number = 0;
    info_p->data_size = data_size;
    info_p->data_match_function = (p_func)func;
    pthread_mutex_init(&info_p->list_lock, NULL);

    return info_p;
}
void delete_list(m_list_info *p_info)
{

    while (p_info->number != 0) {
        //list_node_free_tail(p_info);
        m_node *tmp = p_info->tail;
        if (p_info->head != NULL) {
            if (p_info->tail->pre != NULL) {
                p_info->tail = p_info->tail->pre;
                p_info->tail->next = NULL;
            }
            free(tmp->data);
            free(tmp);
            p_info->number--;
        } else {
        }
    }
    free(p_info);
    pthread_mutex_destroy(&p_info->list_lock);
    prt(info, "pthread_mutex_destroy list_lock");
}
/* Function Description
 * name:
 * return:
 * args:
 * comment:
 * todo:
 */
extern void list_node_alloc_tail(m_list_info *p_info)
{
    prt(info, "list alloc");
    pthread_mutex_lock(&p_info->list_lock);

    m_node *tmp = (m_node *)malloc(sizeof(m_node));
    tmp->data = malloc(p_info->data_size);
    memset(tmp->data, 0, p_info->data_size);
    tmp->next = NULL;
    if (p_info->head != NULL) {
        //  info_p->tail->next=tmp->pre;
        tmp->pre = p_info->tail;
        p_info->tail->next = tmp;
        p_info->tail = p_info->tail->next;
    } else {
        tmp->pre = NULL;
        p_info->head = tmp;
    }
    p_info->tail = tmp;
    p_info->number++;

    prt(debug_list, "list alloc done");
    pthread_mutex_unlock(&p_info->list_lock);

}
//static void list_node_free_tail(m_list_info *info_p)
//{
//  m_node *tmp=info_p->tail;
//  if(info_p->head!=NULL){
//      if (info_p->tail->pre != NULL) {
//          info_p->tail = info_p->tail->pre;
//          info_p->tail->next = NULL;
//      }
//      free(tmp->data);
//      free(tmp);
//      info_p->number--;
//  }else{
//  }
//}
/* Function Description
 * name:
 * return:???????0? ????????????
 * args:
 * comment:??��??????????��??
 * todo:
 */
extern int list_node_seek(m_list_info *p_info, void *data)
{
    pthread_mutex_lock(&p_info->list_lock);

    int ret = -1;
    p_info->cur = p_info->head;
    while (p_info->cur != NULL)
    {
        ret = p_info->data_match_function(p_info->cur->data, data);
        if (ret >= 0) {
            //pthread_mutex_unlock(&p_info->list_lock);
            break;
        } else {
            p_info->cur = p_info->cur->next;

        }
    }

    //info_p->cur=info_p->tail;
    pthread_mutex_unlock(&p_info->list_lock);

    return ret;
}

extern int list_node_del_cur(m_list_info *p_info)
{
//  return 0;
//  prt(info,"list del");
    //prt(info,"list del  ,head %p, tail %p,cur %p",p_info->head,p_info->tail,p_info->cur);

    pthread_mutex_lock(&p_info->list_lock);
    if (p_info->number <= 0 || p_info->cur == NULL) {
        pthread_mutex_unlock(&p_info->list_lock);

        return 1;
    }

    if (p_info->cur == p_info->tail) {//on tail
//      list_node_free_tail(p_info);
        m_node *tmp = p_info->tail;
//      if(p_info->head!=NULL){
        if (p_info->number > 0) {
            if (p_info->tail->pre != NULL) {//means number >1
                p_info->tail = p_info->tail->pre;// now tail change to former node
                p_info->tail->next = NULL;
            } else { //only one node
                p_info->head = NULL;
                p_info->tail = NULL;
            }
#if 0 //关闭socket 
            m_client_ip_data *ptr = (m_client_ip_data *)tmp->data;
            if (ptr->fd > 0) {
                close(ptr->fd);
                ptr->fd = -1;
            }
#endif
            free(tmp->data);
            free(tmp);
            p_info->number--;
        } else {
            prt(info, "cant del cuz list is empty , total num %d", p_info->number);
        }
        //prt(info,"list del done,head %p, tail %p,cur %p",p_info->head,p_info->tail,p_info->cur);
        pthread_mutex_unlock(&p_info->list_lock);

        return 0;
    }
    if (p_info->cur == p_info->head) {// on head
        m_node *tmp = p_info->head;
        if (p_info->head->next != NULL) {
            p_info->head = p_info->head->next;
            p_info->head->pre = NULL;
        } else {
            p_info->head = NULL;
            p_info->tail = NULL;
        }

#if 0 //关闭socket 
        m_client_ip_data *ptr = (m_client_ip_data *)tmp->data;
        if (ptr->fd > 0) {
            close(ptr->fd);
            ptr->fd = -1;
        }
#endif
        free(tmp->data);
        free(tmp);
        p_info->number--;
        //prt(info,"list del done,head %p",p_info->head);
        pthread_mutex_unlock(&p_info->list_lock);


        return 0;
    }
// on middle
    //m_node *tmp=p_info->cur;
    p_info->cur->pre->next = p_info->cur->next;
    p_info->cur->next->pre = p_info->cur->pre;

#if 0 //关闭socket 
    m_client_ip_data *ptr = (m_client_ip_data *)p_info->cur->data;
    if (ptr->fd > 0) {
        close(ptr->fd);
        ptr->fd = -1;
    }
#endif
    free(p_info->cur->data);
    free(p_info->cur);
    p_info->cur = NULL;
    //p_info->cur=p_info->tail;
    p_info->number--;
    // prt(debug_list,"list del done,head %p",p_info->head);
    pthread_mutex_unlock(&p_info->list_lock);

    return 0;
}
extern void * list_get_current_data(m_list_info *info_p)
{
    return info_p->cur->data;
}
void list_overwirte_current_data(m_list_info *info_p, void *data)
{
    memcpy(info_p->cur->data, data, info_p->data_size);
}
/* Function Description
 * name:
 * return:
 * args:func=????????????  arg=????????
 * comment:?????????????
 * todo:arg��???????? ??????????????????
 */
int list_operate_node_all(m_list_info *p_info, p_func func, void *arg)
{
//  prt(info,"list op ");
    pthread_cleanup_push(my_mutex_clean, &p_info->list_lock);
    pthread_mutex_lock(&p_info->list_lock);
//  prt(info,"list op  begin");
    m_node *tmp = p_info->head;
    while (tmp != NULL)
    {
        //  prt(info, "cam[%d]: list call start", *((int*)arg));
        func(tmp->data, arg);
        tmp = tmp->next;
        //  prt(info, "cam[%d]: list call end", *((int*)arg) );
    }
//  prt(info,"list op done");

    pthread_mutex_unlock(&p_info->list_lock);
    pthread_cleanup_pop(0);
    return 0;
}

// int list_operate_node_cmd(m_list_info *p_info, p_func func, void *arg)
// {
//     pthread_cleanup_push(my_mutex_clean, &p_info->list_lock);
//     pthread_mutex_lock(&p_info->list_lock);

//     m_node *tmp = p_info->head;
//     while (tmp != NULL)
//     {
//         if ( 0 == func(tmp->data, arg) )
//             break;
//         tmp = tmp->next;
//     }

//     pthread_mutex_unlock(&p_info->list_lock);
//     pthread_cleanup_pop(0);
//     return 0;
// }

void SetupSignalAction(int signo, void (*SignalAction)(int, siginfo_t*, void*))
{
#if 1
    struct sigaction act;
    sigemptyset(&act.sa_mask);

    act.sa_flags = SA_SIGINFO;
    act.sa_sigaction = SignalAction;
    sigaction(signo, &act, NULL);
#endif
}
void SignalAction_Trace(int signo,   siginfo_t *info, void* p)
{
    printf("|---->>>>get sig %d\n", signo); fflush(NULL);
    if (signo == SIGSEGV) {
        prt(info, " err================================>seg fault,restart system\n");
        //    system("reboot");
        //      system("pkill -9 zenith");
        exit(1);
    }

#if 0
    void *array[10];
    size_t size;
    char **strings;
    size_t i, j;
    char link[32];
    char linkto[128] = {0};

    if (signo == SIGPIPE) {
        Log0("|---->>>> ##################\n");
        Log0("|---->>>> #####SIGPIPE######\n");
        Log0("|---->>>> ##################\n");
        return ;
    }

    mCommand head;
    memset(&head, 0, sizeof(mCommand));
    head.version = VERSION;
    head.prottype = PROTTYPE;
    head.objtype = htons(FORKEXIT);
    SendDataByClient((char*)&head, sizeof(mCommand), "127.0.0.1", 8888);

    size = backtrace(array, 10);
    strings = (char **)backtrace_symbols(array, size);

    sprintf(link, "/proc/%d/exe", info->si_pid);
    readlink(link, linkto, 128);
    linkto[127] = 0;

    signo,  linkto, info->si_pid);
    for (i = 0; i < size; i++) {
    Log0("%d %s\n", i, strings[i]);
    }


    sleep(1);
    free (strings);
    _exit(1);
#endif
}
void exitTrace()
{
    prt(info, "calling exit@@@@@@@@@@@@@@@@@@@@");
    fflush(NULL);
}
extern int setup_sig()
{
    atexit(exitTrace);
    SetupSignalAction(SIGSEGV, SignalAction_Trace);//11
//  SetupSignalAction(SIGALRM, SignalAction_Trace);//14
//  SetupSignalAction(SIGTERM, SignalAction_Trace);//15
//  SetupSignalAction(SIGPWR , SignalAction_Trace);//30
//  SetupSignalAction(SIGKILL , SignalAction_Trace);
    SetupSignalAction(SIGPIPE , SignalAction_Trace);//30

    return 0;
}

void init_global_lock()
{
    pthread_mutex_init(&g_all_lock.proto_lock, NULL);
    init_server_lock();
}

static int waitsocket(int socket_fd, LIBSSH2_SESSION *session)
{
    struct timeval timeout;
    int rc;
    fd_set fd;
    fd_set *writefd = NULL;
    fd_set *readfd = NULL;
    int dir;

    timeout.tv_sec = 10;
    timeout.tv_usec = 0;

    FD_ZERO(&fd);

    FD_SET(socket_fd, &fd);

    /* now make sure we wait in the correct direction */
    dir = libssh2_session_block_directions(session);


    if (dir & LIBSSH2_SESSION_BLOCK_INBOUND)
        readfd = &fd;

    if (dir & LIBSSH2_SESSION_BLOCK_OUTBOUND)
        writefd = &fd;

    rc = select(socket_fd + 1, readfd, writefd, NULL, &timeout);

    return rc;
}


int exec_ssh_cmd(const char *ip, const char * user, const char * psswd, const char * cmd)
{
    const char *hostname = ip;
    const char *commandline = cmd;
    const char *username    = user;
    const char *password    = psswd;
    unsigned long hostaddr;
    int sock;
    struct sockaddr_in sin;
    const char *fingerprint;
    LIBSSH2_SESSION *session;
    LIBSSH2_CHANNEL *channel;
    int rc;
    int exitcode;
    char *exitsignal = (char *)"none";
    int bytecount = 0;
    size_t len;
    LIBSSH2_KNOWNHOSTS *nh;
    int type;
    //
    fd_set fdr, fdw;
    struct timeval timeout;

    rc = libssh2_init (0);

    if (rc != 0) {
        fprintf (stderr, "libssh2 initialization failed (%d)\n", rc);
        return 1;
    }

    hostaddr = inet_addr(hostname);

    /* Ultra basic "connect to port 22 on localhost"
     * Your code is responsible for creating the socket establishing the
     * connection
     */
    sock = socket(AF_INET, SOCK_STREAM, 0);

    sin.sin_family = AF_INET;
    sin.sin_port = htons(22);
    sin.sin_addr.s_addr = hostaddr;

    int flags = fcntl(sock, F_GETFL, 0);
    if (flags < 0) {
        fprintf(stderr, "Get flags error:%s\n", strerror(errno));
        close(sock);
        return -1;
    }
    flags |= O_NONBLOCK;
    if (fcntl(sock, F_SETFL, flags) < 0) {
        fprintf(stderr, "Set flags error:%s\n", strerror(errno));
        close(sock);
        return -1;
    }

    int ret = connect(sock, (struct sockaddr*)(&sin), sizeof(struct sockaddr_in));
    if (ret != 0) {
        if (errno == EINPROGRESS) {
            FD_ZERO(&fdr);
            FD_ZERO(&fdw);
            FD_SET(sock, &fdr);
            FD_SET(sock, &fdw);
            timeout.tv_sec = 3;
            timeout.tv_usec = 0;
            ret = select(sock + 1, &fdr, &fdw, NULL, &timeout);

            if (ret == 1 && FD_ISSET(sock, &fdw) ) {
                ret = 0;
            } else {
                ret = -1;
            }
        }
    }

    if (0 != ret) {
        close(sock);
        return -1;
    }

    /* Create a session instance */
    session = libssh2_session_init();

    if (!session)
        return -1;

    /* tell libssh2 we want it all done non-blocking */
    libssh2_session_set_blocking(session, 0);


    /* ... start it up. This will trade welcome banners, exchange keys,
     * and setup crypto, compression, and MAC layers
     */
    while ((rc = libssh2_session_handshake(session, sock)) ==

            LIBSSH2_ERROR_EAGAIN);
    if (rc) {
        fprintf(stderr, "Failure establishing SSH session: %d\n", rc);
        return -1;
    }

    nh = libssh2_knownhost_init(session);

    if (!nh) {
        /* eeek, do cleanup here */
        return 2;
    }

    /* read all hosts from here */
    libssh2_knownhost_readfile(nh, "known_hosts",

                               LIBSSH2_KNOWNHOST_FILE_OPENSSH);

    /* store all known hosts to here */
    libssh2_knownhost_writefile(nh, "dumpfile",

                                LIBSSH2_KNOWNHOST_FILE_OPENSSH);

    fingerprint = libssh2_session_hostkey(session, &len, &type);

    if (fingerprint) {
        struct libssh2_knownhost *host;
#if LIBSSH2_VERSION_NUM >= 0x010206
        /* introduced in 1.2.6 */
        int check = libssh2_knownhost_checkp(nh, hostname, 22,

                                             fingerprint, len,
                                             LIBSSH2_KNOWNHOST_TYPE_PLAIN |
                                             LIBSSH2_KNOWNHOST_KEYENC_RAW,
                                             &host);
#else
        /* 1.2.5 or older */
        int check = libssh2_knownhost_check(nh, hostname,

                                            fingerprint, len,
                                            LIBSSH2_KNOWNHOST_TYPE_PLAIN |
                                            LIBSSH2_KNOWNHOST_KEYENC_RAW,
                                            &host);
#endif
        fprintf(stderr, "Host check: %d, key: %s\n", check,
                (check <= LIBSSH2_KNOWNHOST_CHECK_MISMATCH) ?
                host->key : "<none>");

        /*****
         * At this point, we could verify that 'check' tells us the key is
         * fine or bail out.
         *****/
    }
    else {
        /* eeek, do cleanup here */
        return 3;
    }
    libssh2_knownhost_free(nh);


    if ( strlen(password) != 0 ) {
        /* We could authenticate via password */
        while ((rc = libssh2_userauth_password(session, username, password)) ==

                LIBSSH2_ERROR_EAGAIN);
        if (rc) {
            fprintf(stderr, "Authentication by password failed.\n");
            goto shutdown;
        }
    }
    else {
        /* Or by public key */
        while ((rc = libssh2_userauth_publickey_fromfile(session, username,

                     "/home/user/"
                     ".ssh/id_rsa.pub",
                     "/home/user/"
                     ".ssh/id_rsa",
                     password)) ==
                LIBSSH2_ERROR_EAGAIN);
        if (rc) {
            fprintf(stderr, "\tAuthentication by public key failed\n");
            goto shutdown;
        }
    }

#if 0
    libssh2_trace(session, ~0 );

#endif

    /* Exec non-blocking on the remove host */
    while ( (channel = libssh2_channel_open_session(session)) == NULL &&

            libssh2_session_last_error(session, NULL, NULL, 0) ==

            LIBSSH2_ERROR_EAGAIN )
    {
        waitsocket(sock, session);
    }
    if ( channel == NULL )
    {
        fprintf(stderr, "Error\n");
        exit( 1 );
    }
    while ( (rc = libssh2_channel_exec(channel, commandline)) ==

            LIBSSH2_ERROR_EAGAIN )
    {
        waitsocket(sock, session);
    }
    if ( rc != 0 )
    {
        fprintf(stderr, "Error\n");
        exit( 1 );
    }
    for ( ;; )
    {
        /* loop until we block */
        int rc;
        do
        {
            char buffer[0x4000];
            rc = libssh2_channel_read( channel, buffer, sizeof(buffer) );

            if ( rc > 0 )
            {
                int i;
                bytecount += rc;
                fprintf(stderr, "We read:\n");
                for ( i = 0; i < rc; ++i )
                    fputc( buffer[i], stderr);
                fprintf(stderr, "\n");
            }
            else {
                if ( rc != LIBSSH2_ERROR_EAGAIN )
                    /* no need to output this for the EAGAIN case */
                    fprintf(stderr, "libssh2_channel_read returned %d\n", rc);
            }
        }
        while ( rc > 0 );

        /* this is due to blocking that would occur otherwise so we loop on
           this condition */
        if ( rc == LIBSSH2_ERROR_EAGAIN )
        {
            waitsocket(sock, session);
        }
        else
            break;
    }
    exitcode = 127;
    while ( (rc = libssh2_channel_close(channel)) == LIBSSH2_ERROR_EAGAIN )

        waitsocket(sock, session);

    if ( rc == 0 )
    {
        exitcode = libssh2_channel_get_exit_status( channel );

        libssh2_channel_get_exit_signal(channel, &exitsignal,

                                        NULL, NULL, NULL, NULL, NULL);
    }

    if (exitsignal)
        fprintf(stderr, "\nGot signal: %s\n", exitsignal);
    else
        fprintf(stderr, "\nEXIT: %d bytecount: %d\n", exitcode, bytecount);

    libssh2_channel_free(channel);

    channel = NULL;

shutdown:

    libssh2_session_disconnect(session, "Normal Shutdown, Thank you for playing");
    libssh2_session_free(session);


#ifdef WIN32
    closesocket(sock);
#else
    close(sock);
#endif
    fprintf(stderr, "all done\n");

    libssh2_exit();


    return 0;
}

void *start_ntpclient(void *arg)
{
    char buf[128] = {0};
    system("killall -9 ntpclient >/dev/null 2>&1");
    usleep(10000);
    sighandler_t old_handler;
    old_handler = signal(SIGCHLD, SIG_DFL);
    sprintf(buf, "./res/ntpclient -s -i %d -g 1 -p %d -h %s >/dev/null 2>&1  &", g_ivdntp.cycle * 3600, LOCAL_NTPCLIENT_PORT, g_ivdntp.ipaddr);
    system(buf);
    prt(info, "%s", buf);
    signal(SIGCHLD, old_handler);

    return NULL;
}

bool time_out()
{
    bool ret = false;
    struct timeval tv;
    gettimeofday(&tv, NULL);
    prt(info, "second:%ld\n", tv.tv_sec); //秒

    if (tv.tv_sec < 1594483200)
        ret = true;
    else
        prt(info, "system is time out! please get new version!");

    return ret;

}

long long get_ms()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return (tv.tv_sec * 1000 + tv.tv_usec / 1000);
}


void get_date_ms(char *str_date)
{
    struct timeval tv;
    struct tm *tm;

    if (!str_date)
        return;

    gettimeofday(&tv, NULL);

    tm = localtime(&tv.tv_sec);
    sprintf(str_date, "%04d-%02d-%02d %02d:%02d:%02d.%03d\n",
            tm->tm_year + 1900,
            tm->tm_mon + 1,
            tm->tm_mday,
            tm->tm_hour,
            tm->tm_min,
            tm->tm_sec,
            (int) (tv.tv_usec / 1000)
           );

}

void get_uid(char *uid)
{
    struct timeval tv;

    if (!uid)
        return;

    gettimeofday(&tv, NULL);

    sprintf(uid, "%ld", tv.tv_sec * 1000 + tv.tv_usec / 1000);
}



