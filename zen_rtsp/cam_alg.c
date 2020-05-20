
#ifdef __cplusplus
extern "C" {
#endif
//#include "DSPARMProto.h"
#ifdef __cplusplus
}
#endif
#include <unistd.h>
#include<dirent.h>
#include<sys/types.h>
#include "common.h"
#include "client_net.h"
#include "g_fun.h"
#include "csocket.h"
#include "queue.h"
#include "ws_service.h"
#include "cam_event.h"
#include "camera_service.h"
#include "sig_service.h"
#include "cam_alg.h"
#include "ptserver/non_motor_server.h"


////////////////////////////////////
extern int sig_fd;
extern int fd_person;
extern m_holder holder[CAM_MAX];
extern unsigned char protocol_sel;
extern IVDStatisSets    g_statisset;
extern EX_mRealStaticInfo ex_static_info;
extern IVDDevSets       g_ivddevsets;
extern mCamDetectParam g_camdetect[CAM_MAX];
extern radar_cam_realtime_t radar_cam_realtime_data[CAM_MAX];
extern mRealTimePerson realtime_person[CAM_MAX];
extern mGlobalVAR g_member;
extern m_camera_info cam_info[CAM_MAX];
extern mEventInfo events_info[CAM_MAX];
///////////////////////////////////
static int person_flow1[CAM_MAX][MAX_REGION_NUM] = {0};
static int person_flow2[CAM_MAX][MAX_REGION_NUM] = {0};
static unsigned int person_density[CAM_MAX][MAX_REGION_NUM] = {0};
//////////////////////////////////
mAlgParam algparam[CAM_MAX];
pthread_mutex_t person_flow_lock[CAM_MAX];
///////////////////////////////////
jierui_60s_t jierui_d60[CAM_MAX];
/////////////////////////////////
//int time_section_changed(mCamDetectParam *p_camdetectparam, mAlgParam *algparam)
int time_section_changed(IVDTimeStatu p_timeparam, mAlgParam *algparam)
{   // qiang

    unsigned char day_night = 1;

    int newstatus = 0;
    unsigned int value;
    struct tm *pltime;
    int ret = 0;

    time_t now;
    time(&now);
    unsigned int pTime[4];

    // if (p_timeparam) {
    pTime[0] = p_timeparam.timep1;
    pTime[1] = p_timeparam.timep2;
    pTime[2] = p_timeparam.timep3;
    pTime[3] = p_timeparam.timep4;
    // }
    pltime = localtime(&now);
    value = pltime->tm_hour * 60 + pltime->tm_min;
    prt(debug_time_shift, "look at time %d , ling chen(%d->%d),baitian(%d->%d), huanghun(%d->%d),", value,
        pTime[0], pTime[1],
        pTime[1], pTime[2],
        pTime[2], pTime[3]);
    if (value >= pTime[0] && value < pTime[1]) {
        newstatus = MORNING;//ling cheng
        day_night = 2;
    }
    else if (value >= pTime[1] && value < pTime[2]) {
        newstatus = DAYTIME;// bai tian
        day_night = 1;
    }
    else if (value >= pTime[2] && value < pTime[3]) {
        newstatus = DUSK;//huang hun
        day_night = 1;
    }
    else {
        newstatus = NIGHT;//wan shang
        day_night = 2;
    }
    //0,1,2,3,4
    if (newstatus != algparam->time_section) {

        algparam->time_section = newstatus ;
        algparam->LaneIn.uEnvironmentStatus = algparam->time_section;

        ret = 1;
    }

    algparam->alg_arg.p_outbuf->uEnvironmentStatus = day_night;

    return ret;
}

int lane_param_init(int curstatus, mDetectParam *tmpDetect, mCamDemarcateParam * tmpCamdem , mAlgParam *algparam)
{
    print_alg("alg ctrl switch");

    algparam->LaneIn.uTransFactor = tmpDetect[curstatus].uTransFactor;
    algparam->LaneIn.uGraySubThreshold = tmpDetect[curstatus].uGraySubThreshold;
    algparam->LaneIn.uSpeedCounterChangedThreshold = tmpDetect[curstatus].uSpeedCounterChangedThreshold;
    algparam->LaneIn.uSpeedCounterChangedThreshold1 = tmpDetect[curstatus].uSpeedCounterChangedThreshold1;
    algparam->LaneIn.uSpeedCounterChangedThreshold2 = tmpDetect[curstatus].uSpeedCounterChangedThreshold2;
    algparam->LaneIn.uDayNightJudgeMinContiuFrame = tmpDetect[curstatus].uDayNightJudgeMinContiuFrame;
    algparam->LaneIn.uComprehensiveSens = tmpDetect[curstatus].uComprehensiveSens;
    algparam->LaneIn.uDetectSens1 = tmpDetect[curstatus].uDetectSens1;
    algparam->LaneIn.uDetectSens2 = tmpDetect[curstatus].uDetectSens2;
    algparam->LaneIn.uStatisticsSens1 = tmpDetect[curstatus].uStatisticsSens1;
    algparam->LaneIn.uStatisticsSens2 = tmpDetect[curstatus].uStatisticsSens2;
    algparam->LaneIn.uSobelThreshold = tmpDetect[curstatus].uSobelThreshold;


    algparam->LaneIn.uEnvironment = algparam->alg_index;
    algparam->LaneIn.uEnvironmentStatus = algparam->time_section;
    algparam->LaneIn.base_length = (float)tmpCamdem->baselinelen;
	algparam->LaneIn.cam2stop = (float)tmpCamdem->cam2stop;
    algparam->LaneIn.near_point_length = (float)tmpCamdem->recent2stop;

    algparam->LaneIn.horizon_base_length = (float)tmpCamdem->horizontallinelen;
    return 0;
}

#define POINTSIZE 16
int lane_pos_init(mChannelCoil *tmpcoil, mLine *tmpline, CPoint *m_ptEnd, int lanenum)
{
    int i = 0;
    for (i = 0; i < lanenum; i++)
    {
        m_ptEnd[0 + i * POINTSIZE].x = tmpline[i * 2].startx;
        m_ptEnd[0 + i * POINTSIZE].y = tmpline[i * 2].starty;
        m_ptEnd[1 + i * POINTSIZE].x = tmpline[i * 2].endx;
        m_ptEnd[1 + i * POINTSIZE].y = tmpline[i * 2].endy;
        m_ptEnd[2 + i * POINTSIZE].x = tmpline[i * 2 + 1].startx;
        m_ptEnd[2 + i * POINTSIZE].y = tmpline[i * 2 + 1].starty;
        m_ptEnd[3 + i * POINTSIZE].x = tmpline[i * 2 + 1].endx;
        m_ptEnd[3 + i * POINTSIZE].y = tmpline[i * 2 + 1].endy;


        m_ptEnd[4 + i * POINTSIZE].x = tmpcoil[i].FrontCoil[1].x;
        m_ptEnd[4 + i * POINTSIZE].y = tmpcoil[i].FrontCoil[1].y;

        m_ptEnd[5 + i * POINTSIZE].x = tmpcoil[i].FrontCoil[2].x;
        m_ptEnd[5 + i * POINTSIZE].y = tmpcoil[i].FrontCoil[2].y;

        m_ptEnd[6 + i * POINTSIZE].x = tmpcoil[i].FrontCoil[0].x;
        m_ptEnd[6 + i * POINTSIZE].y = tmpcoil[i].FrontCoil[0].y;
        m_ptEnd[7 + i * POINTSIZE].x = tmpcoil[i].FrontCoil[3].x;
        m_ptEnd[7 + i * POINTSIZE].y = tmpcoil[i].FrontCoil[3].y;
        m_ptEnd[8 + i * POINTSIZE].x = tmpcoil[i].RearCoil[1].x;
        m_ptEnd[8 + i * POINTSIZE].y = tmpcoil[i].RearCoil[1].y;
        m_ptEnd[9 + i * POINTSIZE].x = tmpcoil[i].RearCoil[2].x;
        m_ptEnd[9 + i * POINTSIZE].y = tmpcoil[i].RearCoil[2].y;
        m_ptEnd[10 + i * POINTSIZE].x = tmpcoil[i].RearCoil[0].x;
        m_ptEnd[10 + i * POINTSIZE].y = tmpcoil[i].RearCoil[0].y;
        m_ptEnd[11 + i * POINTSIZE].x = tmpcoil[i].RearCoil[3].x;
        m_ptEnd[11 + i * POINTSIZE].y = tmpcoil[i].RearCoil[3].y;
        //add by roger 2019.04.08
        m_ptEnd[12 + i * POINTSIZE].x = tmpcoil[i].MiddleCoil[1].x;
        m_ptEnd[12 + i * POINTSIZE].y = tmpcoil[i].MiddleCoil[1].y;
        m_ptEnd[13 + i * POINTSIZE].x = tmpcoil[i].MiddleCoil[2].x;
        m_ptEnd[13 + i * POINTSIZE].y = tmpcoil[i].MiddleCoil[2].y;
        m_ptEnd[14 + i * POINTSIZE].x = tmpcoil[i].MiddleCoil[0].x;
        m_ptEnd[14 + i * POINTSIZE].y = tmpcoil[i].MiddleCoil[0].y;
        m_ptEnd[15 + i * POINTSIZE].x = tmpcoil[i].MiddleCoil[3].x;
        m_ptEnd[15 + i * POINTSIZE].y = tmpcoil[i].MiddleCoil[3].y;
        //


    }
    return 0;
}
int get_alg_index(mAlgParam *algparam)
{
    if (DAYTIME == algparam->time_section
            || DUSK == algparam->time_section)
        return ALG_DAYTIME;
    else
        return ALG_NIGHT;
}
#define USE_INI_CONFIG 1
#define TICK_CHECK_POINT 50//250
#define TICK_ORI_POINT 0

void init_alg_dsp(mAlgParam *algparam, mCamDetectParam *p_camdetectparam, unsigned short gpu_index, unsigned short cam_index)
{
    int i;

    if (algparam->alg_index == ALG_NULL)
        algparam->alg_index = ALG_DAYTIME;

    ///algparam->alg_index;  //æš‚æ—¶æ²¡ç»™ï¿?

    if (TIME_SECTION_NULL == algparam->time_section)
        algparam->time_section = DUSK;
    algparam->tick = TICK_ORI_POINT;
    memset(algparam->alg_arg.m_ptEnd, 0, sizeof(algparam->alg_arg.m_ptEnd));
    memset(algparam->alg_arg.ptimage, 0, sizeof(algparam->alg_arg.ptimage));
    memset(&algparam->LaneIn, 0, sizeof(algparam->LaneIn));

    lane_param_init(algparam->alg_index - 1, p_camdetectparam->detectparam, &p_camdetectparam->camdem, algparam);
    lane_pos_init(p_camdetectparam->detectlane.virtuallane,
                  p_camdetectparam->laneline.laneline, algparam->alg_arg.m_ptEnd, p_camdetectparam->detectlane.lanenum);
    mDemDetectArea *tmpArea = &p_camdetectparam->area;
    for (i = 0; i < DETECT_AREA_MAX; i++) {
        algparam->alg_arg.ptimage[i].x = tmpArea->vircoordinate[i].x;
        algparam->alg_arg.ptimage[i].y = tmpArea->vircoordinate[i].y;
    }

    //ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
    memcpy(&algparam->alg_arg.pDetectCfgSeg.uSegData[0].PersonDetectArea, &p_camdetectparam->personarea, sizeof(mPersonDetectArea));
    //ï¿½Â¼ï¿½ï¿½ï¿½ï¿½ï¿½
    memcpy(&algparam->alg_arg.pDetectCfgSeg.uSegData[0].EventDetectArea, &events_info[cam_index], sizeof(mEventInfo));

    algparam->algNum = p_camdetectparam->detectlane.lanenum;
    transform_init_DSP_VC(USE_INI_CONFIG, algparam->algNum, algparam->LaneIn, &algparam->outbuf, &algparam->alg_arg, gpu_index);
    usleep(100000);
}


frame_info info;
void process_nanjing(int index , OUTBUF *p_outbuf)
{
    radar_rt_lane_car_in_out_info_t car_in_out_info;
    jierui_rt_lane_car_data_info_t jierui_car_info;
    memset(&car_in_out_info, 0, sizeof(radar_rt_lane_car_in_out_info_t));
    memset(&jierui_car_info, 0, sizeof(jierui_rt_lane_car_data_info_t));

    for (int j = 0; j < MAX_LANE_NUM; j++) {
        int exist[MAX_LANE_NUM][NANJING_LANE_COIL_MAX] = {0};
        for (int m = 0; m < NANJING_LANE_COIL_MAX; m++) {
            
            info.cams[index].lanes[j].coils[m].in_car = 0;
            info.cams[index].lanes[j].coils[m].out_car = 0;

            info.cams[index].lanes[j].coils[m].exist_flag = p_outbuf->CoilAttribute[j][m].calarflag; //1 means exist,0 means empty
            info.cams[index].lanes[j].coils[m].obj_type = p_outbuf->CoilAttribute[j][m].uVehicleType;
            // if (j== 0 && m == 0)
            //    prt(info, "##################cams[%d]lanes[%d].coils[%d]: %d#######################",index, j, m, info.cams[index].lanes[j].coils[m].exist_flag);

            if ( (2 == info.cams[index].lanes[j].coils[m].exist_flag) || ( (info.cams[index].lanes[j].coils[m].exist_flag == 0) && (info.cams[index].lanes[j].coils[m].last_exist_flag == 1)) ) {
                //ï¿½ï¿½ï¿½ï¿½
                info.cams[index].lanes[j].coils[m].in_car = 0;
                info.cams[index].lanes[j].coils[m].out_car = 1;

                info.cams[index].lanes[j].coils[m].stay_ms = (get_ms() - info.cams[index].lanes[j].coils[m].in_car_time);
                info.cams[index].lanes[j].coils[m].in_car_time = 0;
                info.cams[index].lanes[j].coils[m].at_out_car_time = get_ms();
                //prt(info, "out car[%d][%d] ************************** ms: %u",j, m, info.cams[index].lanes[j].coils[m].stay_ms);
            } else if ((info.cams[index].lanes[j].coils[m].exist_flag == 1) &&  (info.cams[index].lanes[j].coils[m].last_exist_flag == 0 || 2 == info.cams[index].lanes[j].coils[m].last_exist_flag) ) {
                //ï¿½ë³µ
                info.cams[index].lanes[j].coils[m].in_car = 1;
                info.cams[index].lanes[j].coils[m].out_car = 0;
                info.cams[index].lanes[j].coils[m].in_car_time = get_ms();
                info.cams[index].lanes[j].coils[m].stay_ms = 0;
                info.cams[index].lanes[j].coils[m].at_in_car_time = info.cams[index].lanes[j].coils[m].in_car_time;
                // prt(info, "in car[%d][%d] ************************** ms: %u", j, m, info.cams[index].lanes[j].coils[m].stay_ms);
            } else if (info.cams[index].lanes[j].coils[m].exist_flag == 0) {
                //Ã»ï¿½ï¿½
                info.cams[index].lanes[j].coils[m].in_car = 0;
                info.cams[index].lanes[j].coils[m].out_car = 0;
                info.cams[index].lanes[j].coils[m].in_car_time = 0;
                info.cams[index].lanes[j].coils[m].stay_ms = 0;
                //prt(info, "exist car[%d][%d] ************************** ms", j, m);
            }//else if ((1 == info.cams[index].lanes[j].coils[m].in_car) &&  (1 == info.cams[index].lanes[j].coils[m].exist_flag) ){ //
            else if ( (1 == info.cams[index].lanes[j].coils[m].last_exist_flag) &&  (1 == info.cams[index].lanes[j].coils[m].exist_flag) ) { //
                //ï¿½ï¿½ï¿½ï¿½Í£ï¿½ï¿½
                info.cams[index].lanes[j].coils[m].stay_ms = (get_ms() - info.cams[index].lanes[j].coils[m].in_car_time);
                info.cams[index].lanes[j].coils[m].in_car_time = get_ms();
                exist[j][m] = 1;
                //prt(info, "stay car[%d][%d] *****in car: %u********************* ms: %u", j, m,info.cams[index].lanes[j].coils[m].in_car, info.cams[index].lanes[j].coils[m].stay_ms);
            }

            info.cams[index].lanes[j].coils[m].last_exist_flag = info.cams[index].lanes[j].coils[m].exist_flag;
            //
            info.cams[index].lanes[j].coils[m].head_time =  p_outbuf->CoilAttribute[j][m].uVehicleHeadtime; //ï¿½ï¿½Í·Ê±ï¿½ï¿½
            info.cams[index].lanes[j].coils[m].veh_len = p_outbuf->CoilAttribute[j][m].uVehicleLength; //ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ by roger


        }

        //info.cams[index].lanes[j].veh_type=1;
        //
        info.cams[index].lanes[j].queue_len = p_outbuf->uVehicleQueueLength[j]; //ï¿½ï¿½ï¿½ï¿½ï¿½Å¶Ó³ï¿½ï¿½ï¿½
        info.cams[index].lanes[j].queue_head_len = p_outbuf->uQueueHeadDis[j];
        info.cams[index].lanes[j].queue_tail_len = p_outbuf->uQueueTailDis[j];
        info.cams[index].lanes[j].speed = p_outbuf->CoilAttribute[j][0].uVehicleSpeed;
        info.cams[index].lanes[j].queue_no = p_outbuf->uQueueVehiSum[j];
        info.cams[index].lanes[j].veh_no = p_outbuf->DetectRegionVehiSum[j];
        info.cams[index].lanes[j].ocupation_ratio = p_outbuf->uVehicleDensity[j];
        info.cams[index].lanes[j].average_speed = p_outbuf->uAverVehicleSpeed[j];
        info.cams[index].lanes[j].locate = p_outbuf->uVehicleDistribution[j];
        info.cams[index].lanes[j].head_veh_pos = p_outbuf->uHeadVehiclePos[j];
        info.cams[index].lanes[j].head_veh_speed = p_outbuf->uHeadVehicleSpeed[j];
        info.cams[index].lanes[j].tail_veh_pos = p_outbuf->uLastVehiclePos[j];
        info.cams[index].lanes[j].tail_veh_speed = p_outbuf->uLastVehicleSpeed[j];


        info.cams[index].lanes[j].start_pos = 0;
        info.cams[index].lanes[j].det_status = (p_outbuf->visibility || p_outbuf->fuzzyflag); //ï¿½Ü¼ï¿½ï¿½È£ï¿½ï¿½ì³£

        car_in_out_info.lanes[j].in_flag = info.cams[index].lanes[j].coils[0].in_car;
        car_in_out_info.lanes[j].out_flag = info.cams[index].lanes[j].coils[0].out_car;
        car_in_out_info.index = index;//by anger
        //jierui
        jierui_car_info.index = index;
        jierui_car_info.lanes[j].in_flag = info.cams[index].lanes[j].coils[0].in_car;
        jierui_car_info.lanes[j].out_flag = info.cams[index].lanes[j].coils[0].out_car;
        if (info.cams[index].lanes[j].speed >= 0x00FF)
            jierui_car_info.lanes[j].speed = 0xFF;
        else
            jierui_car_info.lanes[j].speed = info.cams[index].lanes[j].speed;
        jierui_car_info.lanes[j].queue_len = info.cams[index].lanes[j].queue_len;
        if (exist[j][0] > 0) {
            jierui_car_info.lanes[j].in_flag = 1;
        }
            
        // if ( jierui_car_info.lanes[j].in_flag | jierui_car_info.lanes[j].out_flag)
        //     prt(info, "jierui[%d]--in: %d out: %d speed: %d ", index,jierui_car_info.lanes[j].in_flag,jierui_car_info.lanes[j].out_flag, jierui_car_info.lanes[j].speed );

        //my_mutex_lock(&radar_cam_realtime_data[index].mutex_lock);
        if (PROTO_HUAITONG == g_ivddevsets.pro_type ) {
            pthread_cleanup_push(my_mutex_clean, &holder[index].sig_data_lock);
            pthread_mutex_lock(&holder[index].sig_data_lock);

            m_sig_data *p_channel_rst = get_locked_sig_data(index);
            //if (1 == info.cams[index].lanes[j].coils[0].out_car) {

            p_channel_rst->car_info.g_staytm[j] += info.cams[index].lanes[j].coils[0].stay_ms;
            p_channel_rst->car_info.g_staytm_tail[j] += info.cams[index].lanes[j].coils[1].stay_ms;
            //if (j == 1)
            //  prt(info, "status: %d %d coilds: %d stay_ms[%d]: %d",info.cams[index].lanes[j].coils[0].exist_flag, info.cams[index].lanes[j].coils[0].last_exist_flag,
            //  info.cams[index].lanes[j].coils[0].stay_ms, j, p_channel_rst->car_info.g_staytm[j]);

            //}

            if (1 == info.cams[index].lanes[j].coils[0].in_car) {
                p_channel_rst->Eachchannel[j].mDetectDataOfHeaderVirtualCoil.flow++;
                //prt(info, "flow: %d", p_channel_rst->Eachchannel[j].mDetectDataOfHeaderVirtualCoil.flow);
            }

            pthread_mutex_unlock(&holder[index].sig_data_lock);
            pthread_cleanup_pop(0);
        }


#if 0
        if ( 1 == info.cams[index].lanes[j].coils[0].in_car) {

            radar_cam_realtime_data[index].rt_lane[j].lane_vihicle_count++;
            // prt(info, "*******************radar_cam_realtime_data[%d]lane[%d]: %d************************", index, j, radar_cam_realtime_data[index].rt_lane[j].lane_vihicle_count);

            if (info.cams[index].lanes[j].coils[0].veh_len > g_statisset.large)
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.sup_large_veh++;
            else if (info.cams[index].lanes[j].coils[0].veh_len > g_statisset.mediu)
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.large_veh++;
            else if (info.cams[index].lanes[j].coils[0].veh_len > g_statisset.small)
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.mid_veh++;
            else if (info.cams[index].lanes[j].coils[0].veh_len > g_statisset.tiny)
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.small_veh++;
            else
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.min_veh++;

            switch (info.cams[index].lanes[j].coils[0].obj_type) {
            case 1:
            {
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.bus_num++;
            }
            break;
            case 2:
            {
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.car_num++;
            }
            break;
            case 3:
            {
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.truck_num++;
            }
            break;
            }

        }
#endif

        pthread_cleanup_push(my_mutex_clean, &radar_cam_realtime_data[index].mutex_lock);
        pthread_mutex_lock(&radar_cam_realtime_data[index].mutex_lock);
        if (1 == info.cams[index].lanes[j].coils[0].out_car) {

            radar_cam_realtime_data[index].rt_lane[j].lane_vihicle_count++;
            // prt(info, "*******************radar_cam_realtime_data[%d]lane[%d]: %d************************", index, j, radar_cam_realtime_data[index].rt_lane[j].lane_vihicle_count);

            if (info.cams[index].lanes[j].coils[0].veh_len > g_statisset.large)
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.sup_large_veh++;
            else if (info.cams[index].lanes[j].coils[0].veh_len > g_statisset.mediu)
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.large_veh++;
            else if (info.cams[index].lanes[j].coils[0].veh_len > g_statisset.small)
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.mid_veh++;
            else if (info.cams[index].lanes[j].coils[0].veh_len > g_statisset.tiny)
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.small_veh++;
            else
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.min_veh++;

            switch (info.cams[index].lanes[j].coils[0].obj_type) {
            case 1:
            {
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.bus_num++;
            }
            break;
            case 2:
            {
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.car_num++;
            }
            break;
            case 3:
            {
                radar_cam_realtime_data[index].rt_lane[j].no_pro_data.truck_num++;
            }
            break;
            }

            radar_cam_realtime_data[index].rt_lane[j].average_speed = \
                    ( (radar_cam_realtime_data[index].rt_lane[j].out_car_num * radar_cam_realtime_data[index].rt_lane[j].average_speed) \
                      + info.cams[index].lanes[j].speed) / (radar_cam_realtime_data[index].rt_lane[j].out_car_num + 1);
            radar_cam_realtime_data[index].rt_lane[j].out_car_num++;
            radar_cam_realtime_data[index].rt_lane[j].no_pro_data.head_time += p_outbuf->CoilAttribute[j][0].uVehicleHeadtime;
        }

        radar_cam_realtime_data[index].rt_lane[j].no_pro_data.stay_time += info.cams[index].lanes[j].coils[0].stay_ms;
        if (info.cams[index].lanes[j].queue_len > radar_cam_realtime_data[index].rt_lane[j].queue_len)
            radar_cam_realtime_data[index].rt_lane[j].queue_len = info.cams[index].lanes[j].queue_len;
        radar_cam_realtime_data[index].rt_lane[j].head_len = info.cams[index].lanes[j].queue_head_len;
        radar_cam_realtime_data[index].rt_lane[j].tail_len = info.cams[index].lanes[j].queue_tail_len;
        radar_cam_realtime_data[index].rt_lane[j].queue_no = info.cams[index].lanes[j].queue_no;
        radar_cam_realtime_data[index].rt_lane[j].ocuppy  = info.cams[index].lanes[j].ocupation_ratio;
        //radar_cam_realtime_data[index].rt_lane[j].average_speed += info.cams[index].lanes[j].average_speed;
        radar_cam_realtime_data[index].rt_lane[j].location = info.cams[index].lanes[j].locate;
        radar_cam_realtime_data[index].rt_lane[j].head_pos = info.cams[index].lanes[j].head_veh_pos;
        radar_cam_realtime_data[index].rt_lane[j].head_speed = info.cams[index].lanes[j].head_veh_speed;
        radar_cam_realtime_data[index].rt_lane[j].tail_pos = info.cams[index].lanes[j].tail_veh_pos;
        radar_cam_realtime_data[index].rt_lane[j].tail_speed = info.cams[index].lanes[j].tail_veh_speed;
        pthread_mutex_unlock(&radar_cam_realtime_data[index].mutex_lock);
        pthread_cleanup_pop(0);
        ///////////////////////////////////////////////////////////////////////////////////////////////

        ex_static_info.real_test_five[index].lane[j].share += info.cams[index].lanes[j].coils[0].stay_ms;
        if (info.cams[index].lanes[j].queue_len > ex_static_info.real_test_one[index].lane[j].queueLength)
            ex_static_info.real_test_one[index].lane[j].queueLength = info.cams[index].lanes[j].queue_len;

        if (1 == info.cams[index].lanes[j].coils[0].out_car) {
            pthread_cleanup_push(my_mutex_clean, &ex_static_info.real_test_lock[index]);
            pthread_mutex_lock(&ex_static_info.real_test_lock[index]);
            switch (info.cams[index].lanes[j].coils[0].obj_type) {
            case 1:
            case 3:
            {
                ex_static_info.real_test_info[index].lane[j].lagerVehnum++;
                ex_static_info.real_test_info[index].lane[j].speed = info.cams[index].lanes[j].speed;
                //ex_static_info.real_test_updated[index] |= EM_CAR_OUT;
                ex_static_info.real_test_info[index].lane[j].Vehnum++;
                ex_static_info.real_test_five[index].lane[j].Vehnum++;
                ex_static_info.real_test_five[index].lane[j].aveSpeed += info.cams[index].lanes[j].speed;
                ex_static_info.real_test_five[index].lane[j].timedist += p_outbuf->CoilAttribute[j][0].uVehicleHeadtime;
            }
            break;
            case 2:
            {
                ex_static_info.real_test_info[index].lane[j].smallVehnum++;
                ex_static_info.real_test_info[index].lane[j].speed = info.cams[index].lanes[j].speed;
                //ex_static_info.real_test_updated[index] |= EM_CAR_OUT;
                ex_static_info.real_test_info[index].lane[j].Vehnum++;
                ex_static_info.real_test_five[index].lane[j].Vehnum++;
                ex_static_info.real_test_five[index].lane[j].aveSpeed += info.cams[index].lanes[j].speed;
                ex_static_info.real_test_five[index].lane[j].timedist += p_outbuf->CoilAttribute[j][0].uVehicleHeadtime;
            }
            break;
            }
            pthread_mutex_unlock(&ex_static_info.real_test_lock[index]);
            pthread_cleanup_pop(0);
        }
        //ÊµÊ±ï¿½ï¿½ï¿½Ý£ï¿½ï¿½ï¿½Í·ï¿½ï¿½ï¿?
        ex_static_info.static_info[index].lane[j].Headway = info.cams[index].lanes[j].speed * p_outbuf->CoilAttribute[j][0].uVehicleHeadtime * 5 / 18;
        //my_mutex_unlock(&radar_cam_realtime_data[index].mutex_lock);

        // if (j == 0 || j == 1)
        //  prt(info, "radar_cam_realtime_data[%d]lane[%d]: incar: %d outcar: %d speed: %d totalspeed: %d", index, j, radar_cam_realtime_data[index].rt_lane[j].lane_vihicle_count, radar_cam_realtime_data[index].rt_lane[j].out_car_num, info.cams[index].lanes[j].average_speed, radar_cam_realtime_data[index].rt_lane[j].average_speed );
    }

    data_60s_t *p_data_60s = &d60[index];
    if (protocol_sel == PROTO_JIERUI)
        p_data_60s = &jierui_d60[index].data;

    if (protocol_sel == PROTO_HAIXIN || protocol_sel == PROTO_NANJING || protocol_sel == PROTO_PRIVATE || protocol_sel == PROTO_JIERUI) {
        //lock
        //if(!d60[index].data_valid){

        for (int i = 0; i < MAX_LANE_NUM; i++)  {
            for (int j = 0; j < NANJING_LANE_COIL_MAX; j++) {
                p_data_60s->lane_data[i][j].exist_duration += info.cams[index].lanes[i].coils[j].stay_ms;
                //if (j == 0 && d60[i].lane_data[i][0].exist_duration > 0)
                // prt(info, "1exist_duration:%d", d60[i].lane_data[i][0].exist_duration);
                if (  info.cams[index].lanes[i].coils[j].out_car) {

                    //int duration=info.cams[index].lanes[i].coils[j].out_car_time-info.cams[index].lanes[i].coils[j].in_car_time;
                    // if(duration > 0)
                    //    d60[index].lane_data[i][j].exist_duration+=duration;

                    p_data_60s->lane_data[i][j].pass_number++;
                    p_data_60s->lane_data[i][j].speed_sum += info.cams[index].lanes[i].average_speed; //info.cams[index].lanes[i].speed;
                    p_data_60s->lane_data[i][j].veh_len_sum += info.cams[index].lanes[i].coils[j].veh_len;
                    p_data_60s->lane_data[i][j].head_time_sum += info.cams[index].lanes[i].coils[j].head_time;

                    if (info.cams[index].lanes[i].coils[j].veh_len > g_statisset.large) {//13 //ï¿½ï¿½
                        p_data_60s->lane_data[i][j].car_a_sum++;
                    }
                    else if (info.cams[index].lanes[i].coils[j].veh_len > g_statisset.mediu) { //6 //ï¿½Ð³ï¿½
                        p_data_60s->lane_data[i][j].car_b_sum++;
                    }
                    else {
                        p_data_60s->lane_data[i][j].car_c_sum++;  //Ð¡ï¿½ï¿½
                    }

                }
                
                if (protocol_sel == PROTO_JIERUI) {
                    jierui_d60[index].Lane_queue_len[i] = info.cams[index].lanes[i].queue_len;
                }
            }

        }
        //}
    }
    //unlock
    if ( protocol_sel == PROTO_HAIXIN  || protocol_sel == PROTO_NANJING || protocol_sel == PROTO_PRIVATE) {

        //ï¿½ï¿½ï¿½ï¿½ï¿½Å»ï¿½ï¿½Öµï¿½Í³ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½ï¿½
        for (int i = 0; i < MAX_LANE_NUM; i++)  {
            d300[index].lane_data[i].exist_duration += info.cams[index].lanes[i].coils[0].stay_ms;
            if (  info.cams[index].lanes[i].coils[0].out_car) {
                // int duration=info.cams[index].lanes[i].coils[0].out_car_time-info.cams[index].lanes[i].coils[0].in_car_time;
                // if(duration > 0)
                //   d300[index].lane_data[i].exist_duration+=duration;
            }

            if (info.cams[index].lanes[i].queue_len > 0) {
                if (info.cams[index].lanes[i].queue_len > d300[index].lane_data[i].queue_len_max)
                    d300[index].lane_data[i].queue_len_max = info.cams[index].lanes[i].queue_len;

                if (d300[index].lane_data[i].queue_status == 0) { //Ò»ï¿½ï¿½ï¿½Å¶Ó¼ï¿½ï¿½ï¿½Ò»ï¿½ï¿½
                    d300[index].lane_data[i].car_stop_sum++;
                }

                d300[index].lane_data[i].queue_status = 1;

            } else {
                d300[index].lane_data[i].queue_status = 0; //Ã»ï¿½ï¿½ï¿½Å¶ï¿½

            }
            d300[index].lane_data[i].lane_no = info.cams[index].lanes[i].no;

        }

    }

    add_jierui_60s_item(index ,&jierui_d60[index]);
    //ï¿½ï¿½ï¿½ï¿½ï¿½ë³µï¿½ï¿½ï¿½ï¿½
    add_car_in_out_item(index, &car_in_out_info);
    //if (protocol_sel == PROTO_HUAITONG_PERSON)
    person_area_data_hanle(index, p_outbuf);
    //
    add_jierui_car_item(index, &jierui_car_info);

}

int AlgProcessFrameData(int index, mAlgParam *algparam, unsigned char *inbuf, unsigned char *inubuf, unsigned char *invbuf, unsigned short iwidth, unsigned short iheight)
{
    int ret = 0;

    //algparam->framecount++;
    PNode node = NULL;
    if (g_ivddevsets.pro_type == PROTO_WS_RADAR || g_ivddevsets.pro_type == PROTO_WS_RADAR_VIDEO) {
        node = DeQueue(cam_info[index].p_radar);
    }

    if (node) {
        mRadarRTObj *objs = (mRadarRTObj*)node->p_value;
        if (objs)
            transform_Proc_DSP_VC(index, inbuf, inubuf, invbuf, iwidth, iheight, 0, objs, node->val_len / sizeof(mRadarRTObj), &algparam->outbuf, &algparam->alg_arg);
    }
    else
        transform_Proc_DSP_VC(index, inbuf, inubuf, invbuf, iwidth, iheight, 0, NULL, 0, &algparam->outbuf, &algparam->alg_arg);

    if (node) {
        if (node->p_value)
            free(node->p_value);
        free(node);
    }

    return ret;
}

static int alloc_alg(mAlgParam *algparam, mCamDetectParam *p_camdetectparam, unsigned short gpu_index, unsigned short cam_index)
{
    init_alg_dsp(algparam, p_camdetectparam, gpu_index, cam_index);
    return 0;
}

void pack_sig_data(int index)
{
    mAlgParam * p_algparam = &algparam[index];
    //p_channel_rst->lane_num=p_algparam->algNum;
    //pthread_cleanup_push(my_mutex_clean, &holder[index].sig_data_lock);
    //pthread_mutex_lock(&holder[index].sig_data_lock);

#if 0
    m_sig_data *p_channel_rst = get_locked_sig_data(index);
    m_sig_data *FVDChannel = p_channel_rst;
    int i;
    //  pthread_mutex_lock(&p->channel_rst_lock);
    for (i = 0; i < p_algparam->algNum; i++) {
        // printf("p_algparam->alg_arg.p_outbuf->calarflag[%d]:[%d]\n", i, p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag);
        if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag == 0x2) {
            if (0 == p_channel_rst->car_info.g_50frame1[i])
                p_channel_rst->car_info.g_50frame1[i] = p_algparam->framecount;
            else {
                p_channel_rst->car_info.g_50frame2[i] = p_channel_rst->car_info.g_50frame1[i];
                p_channel_rst->car_info.g_50frame1[i] = p_algparam->framecount;
            }
        } else if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag == 0x1)
            p_channel_rst->car_info.g_occupancyframe[i]++;

        if (p_algparam->alg_arg.p_outbuf->IsCarInTail[i]) {
            if (0 == p_channel_rst->car_info.g_50frame1[i])
                p_channel_rst->car_info.g_50frametail1[i] = p_algparam->framecount;
            else {
                p_channel_rst->car_info.g_50frametail2[i] = p_channel_rst->car_info.g_50frametail1[i];
                p_channel_rst->car_info.g_50frametail1[i] = p_algparam->framecount;
            }
            p_channel_rst->car_info.g_occupancyframetail[i]++;
        }

        //prt(info, "qlenght[%d]: %02X speed: %02X", i, p_algparam->alg_arg.p_outbuf->uVehicleQueueLength[i], p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleSpeed);
    }

#endif
    pthread_cleanup_push(my_mutex_clean, &holder[index].sig_data_lock);
    pthread_mutex_lock(&holder[index].sig_data_lock);
    memcpy(&holder[index].algparam, p_algparam, sizeof(mAlgParam));
    pthread_mutex_unlock(&holder[index].sig_data_lock);
    pthread_cleanup_pop(0);

#if 0 //(PROTOCOL_SEL==DETECTOR)
    char bBackgroundRefreshed = 0x1;
    mCamParam *p = get_mCamParam(index);
    struct timeval tv;
    gettimeofday(&tv, NULL);
    long now_time = tv.tv_sec * 1000 + tv.tv_usec / 1000;
    long last_time = get_holder_last_time(index);
    unsigned char status = get_holder_status(index);

    prt(info, "nowtime: %ld lasttime: %ld s: %ld", now_time, last_time, (now_time - last_time));

    for (i = 0; i < p_algparam->algNum; i++) {

        if ( (now_time - last_time) >= 500 &&  (0 == (status & 0x02)) ) {
            if ( (i + 1) == p_algparam->algNum)
                set_holder_status(index, status | 0x02) ;
            //FVDChannel->Eachchannel[i].mDetectChannelIndex =p->channelcoil[i].number;
            //printf("sub->cammer.channelcoil[%d].number: [%d]\n",i, FVDChannel->Eachchannel[i].mDetectChannelIndex);
            FVDChannel->Eachchannel[i].mDetectChannelIndex = get_lane_index(index, i);
            prt(info, "laneid[%d]: %d", i,  get_lane_index(index, i))
            if (FVDChannel->Eachchannel[i].mDetectChannelIndex < 41
                    && FVDChannel->Eachchannel[i].mDetectChannelIndex > 96)
                FVDChannel->Eachchannel[i].mDetectChannelIndex = 255;  //é—î‚£å²¸éæ’»å¼¬éŠˆå——î¶é—è·¨å–é‹å©šå¹é”Ÿï¿?
            FVDChannel->Eachchannel[i].mQueueLength =
                p_algparam->alg_arg.p_outbuf->uLastVehicleLength[i];  //é—è·¨å–•é“å¥¸æ¢¼é”ä¾¯åŠœé–¹å‡¤æ‹·
            FVDChannel->Eachchannel[i].mRealTimeSingleSpeed =
                p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleSpeed; //é—è·¨å–Žæ¿®î… æ‹‹ç‘™å‹«ï¿?
            if (0x1 == p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag)
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bOccupyStatus1 =
                    0x1; // 0.5sé—è·¨å–é‹å©šå¹å®„æ¿çª—é—è·¨å–é‹å©šå¹æ¤‹åº¡Ð¦é–¹î„Šæ‹·
            else
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bOccupyStatus1 =
                    0x0; // 0.5sé—è·¨å–é‹å©šå¹å®„æ¿çª—é—è·¨å–é‹å©šå¹æ¤‹åº¡Ð¦é–¹î„Šæ‹·

            FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bFix0 =
                0x0; //é—è·¨å–é†ï¿½é ä½½î‰ç€šå½’å´é”Ÿï¿?

            if (!p_channel_rst->car_info.g_flow[i])
                p_channel_rst->car_info.g_flow[i] = p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectOutSum;
            else {
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow =
                    p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectOutSum
                    - p_channel_rst->car_info.g_flow[i]; //é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šï¿½
                p_channel_rst->car_info.g_flow[i] = p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectOutSum;

                if (FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow
                        > 3)
                    FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow =
                        0x3;
            }
            if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag) {
                if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleLength > 7)
                    FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bVehicleType =
                        0x3;
                else if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleLength > 5)
                    FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bVehicleType =
                        0x2;
                else
                    FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bVehicleType =
                        0x1;
            } else
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bVehicleType =
                    0x0; //é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é—æ‰®æ•¸çã„©å¹é”Ÿï¿?

            if (p_channel_rst->car_info.g_50frame1[i] && p_channel_rst->car_info.g_50frame2[i]) {
                FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil =
                    (p_channel_rst->car_info.g_50frame1[i] - p_channel_rst->car_info.g_50frame2[i])
                    * (0.4);
                if (FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil
                        > 254)
                    FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil =
                        255;
            } else
                FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil =
                    0x0; //å©¢èˆµæŒ³éæ’»å¼¬éŠˆå——î¶é–ºå†¨çˆ¼éæ’»å¼¬éŠˆå——î¶

            if (p_channel_rst->car_info.g_occupancyframe[i]) {
                FVDChannel->Eachchannel[i].mOccupancyOfHeaderVirtualCoil =
                    ((float) (((float) p_channel_rst->car_info.g_occupancyframe[i]) / 12))
                    * 100;
                p_channel_rst->car_info.g_occupancyframe[i] = 0;
            } else
                FVDChannel->Eachchannel[i].mOccupancyOfHeaderVirtualCoil = 0x0; //å©¢èˆµæŒ³éæ’»å¼¬éŠˆå——î¶é–¸æ¥ƒå¨€éæ’»å¼¬éŠˆå——î¶é—è·¨å–é‹å©šå¹é”Ÿï¿½

            /*****éå¿›ç®–éæ’»å¼¬éŠˆå——ï¿?*/
            FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bOccupyStatus1 =
                p_algparam->alg_arg.p_outbuf->IsCarInTail[i]; // 0.5sé—è·¨å–é‹å©šå¹å®„æ¿çª—é—è·¨å–é‹å©šå¹æ¤‹åº¡Ð¦é–¹î„Šæ‹·
            FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bFix0 = 0x0; //é—è·¨å–é†ï¿½é ä½½î‰ç€šå½’å´é”Ÿï¿?

            //printf("g_50frametail1[%d]:[%d] g_50frametail2[%d]:[%d]\n", i, g_50frametail1[i], i, g_50frametail2[i]);
            if (p_channel_rst->car_info.g_50frametail1[i]
                    && p_channel_rst->car_info.g_50frametail2[i]) {
                FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil =
                    (p_channel_rst->car_info.g_50frametail1[i]
                     - p_channel_rst->car_info.g_50frametail2[i]) * (0.4);
                if (FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil
                        > 254)
                    FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil =
                        255;
            } else
                FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil = 0x0; //å©¢èˆµæŒ³éæ’»å¼¬éŠˆå——î¶é–ºå†¨çˆ¼éæ’»å¼¬éŠˆå——î¶ éå¿›ç®–éæ’»å¼¬éŠˆå——î¶é—è·¨å–“å¨¼é¹ƒå´™éŠˆå——î¶ç¼‚ä½ºå––éæ’»å¼¬éŠˆå——ï¿?

            //printf("g_occupancyframetail[%d]:[%d]\n", i, g_occupancyframetail[i]);
            if (p_channel_rst->car_info.g_occupancyframetail[i]) {
                FVDChannel->Eachchannel[i].mOccupancyOfTailVirtualCoil =
                    ((float) (((float) p_channel_rst->car_info.g_occupancyframetail[i])
                              / 12)) * 100;
                //printf("g_occupancyframetail[%d]:[%f]\n", i, (float)(((float)g_occupancyframetail[i])/12));
                p_channel_rst->car_info.g_occupancyframetail[i] = 0;
            } else
                FVDChannel->Eachchannel[i].mOccupancyOfTailVirtualCoil = 0x0; //å©¢èˆµæŒ³éæ’»å¼¬éŠˆå——î¶é–¸æ¥ƒå¨€éæ’»å¼¬éŠˆå——î¶é—è·¨å–é‹å©šå¹é”Ÿï¿½

            if (p_algparam->alg_arg.p_outbuf->visibility) {
                print_alg("==error p_algparam visibility: 0x1");
                FVDChannel->Eachchannel[i].mWorkStatusOfDetectChannle.bDataIsValid =
                    0x1; //é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šå½’æŸ¨é”å³°å»ºé–¿ç‡‚ï¿?
            } else {
                FVDChannel->Eachchannel[i].mWorkStatusOfDetectChannle.bDataIsValid =
                    0x0; //é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šå½’æŸ¨é”å³°å»ºé–¿ç‡‚ï¿?
            }
            /*ç¼‚ä½ºå––éæ’»å¼¬éŠˆå——î¶é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šç‘°Ãºå¦¤å‘®æ™¸é–¿ç‡‚æ‹· End*/

            bBackgroundRefreshed &= p_algparam->alg_arg.p_outbuf->getQueback_flag[i];
            if (FVDChannel->EachStatus.mCameralStatus.bBackgroundRefreshed
                    != bBackgroundRefreshed) {
                p_channel_rst->camera_state_change = 1;
            }
            FVDChannel->EachStatus.mCameralStatus.bBackgroundRefreshed =
                bBackgroundRefreshed;
            //Log0("getQueback_flag[%d]:[%d] bBackgroundRefreshed:[%d]\n",i,p_algparam->alg_arg.p_outbuf->getQueback_flag[i], bBackgroundRefreshed);

            int ori = FVDChannel->EachStatus.mCameralStatus.bWorkMode ;
            if (p_algparam->alg_index == 0x1) //ç™½å¤©
            {
                //  prt(info,"index %d,day time",index);
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x0;
            }
            else if (p_algparam->alg_index == 0x2) //æ™šä¸Š
            {
                //      prt(info,"index %d,night time",index);
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x2;
            }

            //0x1é—è·¨å–“ç€šæ¶™å¯¼å¨†æ„¬î¶ é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸éŸæ¬å¸’æ´æ»ˆå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šå½’æŸ¨é”å‘Šç®é–¹é£Žå…˜éæ’»æ•“é”Ÿï¿½   0x3é—è·¨å–Žé”Ÿç•Œå–å¨…ï¿½ é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šå½’æŸ¨é”å‘Šç®é–¹é£Žå…˜éæ’´æ½éî…§åš‹é–¹é£Žå…˜éæ’»å¼¬éŠˆå——î¶
            if ((p_algparam->LaneIn.uEnvironmentStatus == 0x1
                    && p_algparam->alg_index == 0x1)
                    || (p_algparam->LaneIn.uEnvironmentStatus == 0x3
                        && p_algparam->alg_index == 0x2)) {
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x1;
                //prt(info,"index %d,shift time==> %d",index,p_algparam->LaneIn.uEnvironmentStatus);
                //      FVDChannel->Eachchannel[i].mWorkStatusOfDetectChannle.bDataIsValid=0x1; //é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šå½’æŸ¨é”å³°å»ºé–¿ç‡‚ï¿?
            }

            if (ori != FVDChannel->EachStatus.mCameralStatus.bWorkMode ) {
                p_channel_rst->camera_state_change = 1;
            }

            FVDChannel->EachStatus.mCameralStatus.bH264DecodeStatus = 0x1;
            FVDChannel->EachStatus.mCameralStatus.bCameralOnLine = 0x1;
            if (FVDChannel->EachStatus.mCameralStatus.bPictureStable == (p_algparam->alg_arg.p_outbuf->fuzzyflag ? 1 : 0)  ) {
                p_channel_rst->camera_state_change = 1;
            }

            if (p_algparam->alg_arg.p_outbuf->fuzzyflag) {
                FVDChannel->EachStatus.mCameralStatus.bPictureStable = 0x0; //é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸å¦¤æ¥…ç¼šé¡”æ„°å¹æ¤‹åº¡Ð¦é–¹î„Šæ‹·
                FVDChannel->Eachchannel[i].mWorkStatusOfDetectChannle.bDataIsValid =
                    0x1; //é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šå½’æŸ¨é”å³°å»ºé–¿ç‡‚ï¿?
            } else {
                FVDChannel->EachStatus.mCameralStatus.bPictureStable = 0x1; //é—è·¨å––éŠˆè™¹æ‹‹ç‘™å‹«î¶é–»æ¨¿åŸ–é”Ÿæ–¤ï¿?
            }

            FVDChannel->lane_num = p_algparam->algNum;
            FVDChannel->status = 0x1;
            FVDChannel->EachStatus.mCameralStatus.bBackgroundRefreshed = bBackgroundRefreshed;
        }

        if ((now_time - last_time) >= 250 &&  (0 == (status & 0x01))) {
            if ( (i + 1) == p_algparam->algNum)
                set_holder_status(index, status | 0x01);
            if (0x1 == p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag)
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bOccupyStatus0 =
                    0x1;  // 0.25sé—è·¨å–é‹å©šå¹å®„æ¿çª—é—è·¨å–é‹å©šå¹æ¤‹åº¡Ð¦é–¹î„Šæ‹·
            else
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bOccupyStatus0 =
                    0x0;  // 0.25sé—è·¨å–é‹å©šå¹å®„æ¿çª—é—è·¨å–é‹å©šå¹æ¤‹åº¡Ð¦é–¹î„Šæ‹·
            FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bOccupyStatus0 =
                p_algparam->alg_arg.p_outbuf->IsCarInTail[i];  // 0.25sé—è·¨å–é‹å©šå¹å®„æ¿çª—é—è·¨å–é‹å©šå¹æ¤‹åº¡Ð¦é–¹î„Šæ‹·
        }
    }
    pthread_mutex_unlock(&holder[index].sig_data_lock);
    //submit_unlock_sig_data(index);
    //pthread_cleanup_pop(0);
#endif

    process_nanjing(index, p_algparam->alg_arg.p_outbuf);
    //pthread_mutex_unlock(&p->channel_rst_lock);
}
#include <arpa/inet.h>


void pack_realtime_data(EX_mRealStaticInfo *p_real_info,    mAlgParam *p_algparam, int index)
{
    int i;
    static int total_upperson[CAM_MAX][MAX_REGION_NUM]         = {0};
    static int total_downperson[CAM_MAX][MAX_REGION_NUM]       = {0};
    static long long lane_pre_in_time[CAM_MAX][DETECTLANENUMMAX][2]  = {0};
    static long long  lane_pre_out_time[CAM_MAX][DETECTLANENUMMAX][2] = {0};
    static int lane_pre_staus[CAM_MAX][DETECTLANENUMMAX][2] = {0};

    mRealStaticInfo *p_real_time_data = &(p_real_info->static_info[index]);

    mAlgParam *algparam = p_algparam;
    p_real_time_data->flag = 0xFF;
    p_real_time_data->laneNum = algparam->algNum;
    p_real_time_data->curstatus = algparam->alg_arg.p_outbuf->uEnvironmentStatus;//algparam->alg_index;
    p_real_time_data->fuzzyflag = algparam->alg_arg.p_outbuf->fuzzyflag;
    p_real_time_data->visibility = algparam->alg_arg.p_outbuf->visibility;

    for (i = 0; i < 20; i++) {
        p_real_time_data->uDegreePoint[i][0] = htons(algparam->alg_arg.p_outbuf->uDegreePoint[i][0]);
        p_real_time_data->uDegreePoint[i][1] = htons(algparam->alg_arg.p_outbuf->uDegreePoint[i][1]);
    }

    for (i = 0; i < 10; i++) {
        p_real_time_data->uHorizontalDegreePoint[i][0] = htons(algparam->alg_arg.p_outbuf->uHorizontalDegreePoint[i][0]);
        p_real_time_data->uHorizontalDegreePoint[i][1] = htons(algparam->alg_arg.p_outbuf->uHorizontalDegreePoint[i][1]);
    }

    for (i = 0; i < algparam->algNum; i++) {
        p_real_time_data->lane[i].state   = algparam->alg_arg.p_outbuf->IsCarInTail[i]; //algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag;
        p_real_time_data->lane[i].isCarMid = algparam->alg_arg.p_outbuf->CoilAttribute[i][1].calarflag;
        p_real_time_data->lane[i].isCarInTail = algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag;
        p_real_time_data->lane[i].queueLength = htons(algparam->alg_arg.p_outbuf->uLastVehicleLength[i]);
        //p_real_time_data->lane[i].vehnum1 = htonl(algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectOutSum);
        //p_real_time_data->lane[i].vehnum2 = htonl(algparam->alg_arg.p_outbuf->CoilAttribute[i][1].DetectOutSum);
        pthread_mutex_lock(&p_real_info->lock[index]);
        if (!p_real_info->reset_flag[index]) {
            p_real_info->car_num[index][i][0] += algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectInSum - p_real_info->pre_car_num[index][i][0];
            p_real_time_data->lane[i].vehnum1 = htonl(p_real_info->car_num[index][i][0]);


            p_real_info->car_num[index][i][1] += algparam->alg_arg.p_outbuf->CoilAttribute[i][1].DetectInSum - p_real_info->pre_car_num[index][i][1];
            p_real_time_data->lane[i].vehnum2 = htonl(p_real_info->car_num[index][i][1]);
        } else {
            p_real_time_data->lane[i].vehnum1 = 0;
            p_real_time_data->lane[i].vehnum2 = 0;

        }
        p_real_info->pre_car_num[index][i][0] = algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectInSum;
        p_real_info->pre_car_num[index][i][1] = algparam->alg_arg.p_outbuf->CoilAttribute[i][1].DetectInSum;

        pthread_mutex_unlock(&p_real_info->lock[index]);
        p_real_time_data->lane[i].vehlength1 = htonl((int)algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleLength);
        p_real_time_data->lane[i].vehlength2 = htonl((int)algparam->alg_arg.p_outbuf->CoilAttribute[i][1].uVehicleLength);
        p_real_time_data->lane[i].speed1 = htonl((int)algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleSpeed);
        p_real_time_data->lane[i].speed2 = htonl((int)algparam->alg_arg.p_outbuf->CoilAttribute[i][1].uVehicleSpeed);
        // p_real_time_data->lane[i].existtime1 = algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleSpeed;
        // p_real_time_data->lane[i].existtime2 = algparam->alg_arg.p_outbuf->CoilAttribute[i][1].uVehicleSpeed;

        //ï¿½ï¿½Ò»ï¿½ï¿½È¦ï¿½ï¿½ï¿½ï¿½Ê±ï¿½ï¿½ ï¿½ï¿½Î»ms
        if (2 == p_real_time_data->lane[i].isCarInTail) { //ï¿½ï¿½ï¿½ï¿½
            lane_pre_staus[index][i][0] = 2;
            lane_pre_out_time[index][i][0] = get_ms();
            p_real_time_data->lane[i].existtime1 = htonl(lane_pre_out_time[index][i][0] - lane_pre_in_time[index][i][0]);

        } else if (1 == p_real_time_data->lane[i].isCarInTail) { //ï¿½ï¿½ï¿½ï¿½

            if (0 == lane_pre_staus[index][i][0]) //ï¿½ï¿½Ò»ï¿½ÎµÄ±ï¿½Ö¾
                lane_pre_in_time[index][i][0] = get_ms();
            else if (2 == lane_pre_staus[index][i][0])  //ï¿½ï¿½ï¿½ï¿½ï¿½Ä±È½Ï½ï¿½ï¿½ï¿½Ê±ï¿½ï¿½Ã»ï¿½Ð¼ï¿½ï¿?
                lane_pre_in_time[index][i][0] = lane_pre_out_time[index][i][0];

            lane_pre_staus[index][i][0] = 1;
        } else {

            lane_pre_staus[index][i][0] = 0;
        }

        //ï¿½Ú¶ï¿½ï¿½ï¿½È¦ï¿½ï¿½ï¿½ï¿½Ê±ï¿½ï¿½
        if (2 == p_real_time_data->lane[i].isCarMid) { //ï¿½ï¿½ï¿½ï¿½
            lane_pre_staus[index][i][1] = 2;
            lane_pre_out_time[index][i][1] = get_ms();
            p_real_time_data->lane[i].existtime2 = htonl(lane_pre_out_time[index][i][1] - lane_pre_in_time[index][i][1]);

        } else if (1 == p_real_time_data->lane[i].isCarMid) { //ï¿½ï¿½ï¿½ï¿½

            if (0 == lane_pre_staus[index][i][1]) //ï¿½ï¿½Ò»ï¿½ÎµÄ±ï¿½Ö¾
                lane_pre_in_time[index][i][1] = get_ms();
            else if (2 == lane_pre_staus[index][i][1])  //ï¿½ï¿½ï¿½ï¿½ï¿½Ä±È½Ï½ï¿½ï¿½ï¿½Ê±ï¿½ï¿½Ã»ï¿½Ð¼ï¿½ï¿?
                lane_pre_in_time[index][i][1] = lane_pre_out_time[index][i][1];

            lane_pre_staus[index][i][1] = 1;
        } else {

            lane_pre_staus[index][i][1] = 0;
        }


        p_real_time_data->lane[i].uActualDetectLength =
            htons(algparam->alg_arg.p_outbuf->uActualDetectLength[i]);
        p_real_time_data->lane[i].uActualTailLength =
            htons(algparam->alg_arg.p_outbuf->uActualTailLength[i]);
        //p_real_time_data->lane[i].isCarInTail = algparam->alg_arg.p_outbuf->IsCarInTail[i];

        p_real_time_data->lane[i].LineUp[0].x =
            htons(algparam->alg_arg.p_outbuf->LineUp[i][0].x);
        p_real_time_data->lane[i].LineUp[0].y =
            htons(algparam->alg_arg.p_outbuf->LineUp[i][0].y);

        p_real_time_data->lane[i].LineUp[1].x =
            htons(algparam->alg_arg.p_outbuf->LineUp[i][1].x);
        p_real_time_data->lane[i].LineUp[1].y =
            htons(algparam->alg_arg.p_outbuf->LineUp[i][1].y);

    }

    if (p_real_info->reset_flag[index] > 0) {
        pthread_mutex_lock(&p_real_info->lock[index]);
        memset(total_upperson[index], 0, sizeof(int)*MAX_REGION_NUM);
        memset(total_downperson[index], 0, sizeof(int)*MAX_REGION_NUM);
        p_real_info->reset_flag[index] = 0;
        pthread_mutex_unlock(&p_real_info->lock[index]);
    }

    for (i = 0; i < algparam->algNum; i++) {
        p_real_time_data->queue_len[i]    = algparam->alg_arg.p_outbuf->uVehicleQueueLength[i];
        p_real_time_data->area_car_num[i] = algparam->alg_arg.p_outbuf->DetectRegionVehiSum[i];
        //prt(info,"---------------area car no----->%d\n",p_real_time_data->area_car_num[i]);
        p_real_time_data->queue_line[i][0].x = htons(algparam->alg_arg.p_outbuf->QueLine[i][0].x);
        p_real_time_data->queue_line[i][0].y = htons(algparam->alg_arg.p_outbuf->QueLine[i][0].y);
        p_real_time_data->queue_line[i][1].x = htons(algparam->alg_arg.p_outbuf->QueLine[i][1].x);
        p_real_time_data->queue_line[i][1].y = htons(algparam->alg_arg.p_outbuf->QueLine[i][1].y);
//    prt(info,"lane %d , count: %d ---->(%d,%d)(%d,%d)",i, p_real_time_data->area_car_num[i]
//        , p_real_time_data->queue_line[i][0].x
//            , p_real_time_data->queue_line[i][0].y
//            , p_real_time_data->queue_line[i][1].x
//            , p_real_time_data->queue_line[i][1].y);
    }

    //   usleep(1000000);

    p_real_time_data->rcs_num = htonl((int)algparam->alg_arg.p_outbuf->udetNum);
    //   p_real_time_data->rcs_num= algparam->alg_arg.p_outbuf->udetNum;
    //          printf("\n --> %d\n", p_real_time_data->rcs_num);

    //for(tmp;tmp<100;tmp++){
    for (int tmp = 0; tmp < algparam->alg_arg.p_outbuf->udetNum; tmp++) {

        p_real_time_data->rcs[tmp].x = htonl( algparam->alg_arg.p_outbuf->udetBox[tmp].x);
        //  if(tmp<algparam->alg_arg.p_outbuf->udetNum)
        //printf("\n 9((((((((((((((((((( -> %d\n", algparam->alg_arg.p_outbuf->udetBox[tmp].x);
        p_real_time_data->rcs[tmp].y = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].y);
        p_real_time_data->rcs[tmp].w = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].w);
        p_real_time_data->rcs[tmp].h = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].h);
        p_real_time_data->rcs[tmp].label = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].label);
        p_real_time_data->rcs[tmp].confidense = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].confidence);
        p_real_time_data->rcs[tmp].id = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].id);
        p_real_time_data->rcs[tmp].distance[0] = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].distance[0]);
        p_real_time_data->rcs[tmp].distance[1] = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].distance[1]);
        p_real_time_data->rcs[tmp].landid = htonl((int)algparam->alg_arg.p_outbuf->udetBox[tmp].laneid);
        p_real_time_data->rcs[tmp].speed = htons(algparam->alg_arg.p_outbuf->udetBox[tmp].speed);
        p_real_time_data->rcs[tmp].speed_Vx = htons(algparam->alg_arg.p_outbuf->udetBox[tmp].speed_Vx);
        p_real_time_data->rcs[tmp].width = htons(algparam->alg_arg.p_outbuf->udetBox[tmp].width);
        p_real_time_data->rcs[tmp].lenght = htons(algparam->alg_arg.p_outbuf->udetBox[tmp].length);

        // prt(info, "*****box[%d]: {X:%d Y:%d W:%d H:%d}", tmp, algparam->alg_arg.p_outbuf->udetBox[tmp].x,algparam->alg_arg.p_outbuf->udetBox[tmp].y
        // ,algparam->alg_arg.p_outbuf->udetBox[tmp].w, algparam->alg_arg.p_outbuf->udetBox[tmp].h)
    }

    /*
    for(tmp = 0; tmp < 2; tmp++) {
        p_real_time_data->detectline[tmp].x = g_camdetect[index].personarea.detectline[tmp].x;
        p_real_time_data->detectline[tmp].y = g_camdetect[index].personarea.detectline[tmp].y;
    }
    */

#if 0
    total_upperson[index] += algparam->alg_arg.p_outbuf->uPersonDirNum[0];
    p_real_time_data->upperson = htonl(total_upperson[index]);
    total_downperson[index] += algparam->alg_arg.p_outbuf->uPersonDirNum[1];
    p_real_time_data->downperson = htonl(total_downperson[index]);
#endif

    memset(p_real_time_data->udetPersonBox, 0, sizeof(p_real_time_data->udetPersonBox) );
    for (i = 0; i < 100; i++) {
        p_real_time_data->udetPersonBox[i].height = htons(algparam->alg_arg.p_outbuf->udetPersonBox[i].h);
        p_real_time_data->udetPersonBox[i].width = htons(algparam->alg_arg.p_outbuf->udetPersonBox[i].w);
        p_real_time_data->udetPersonBox[i].x = htons(algparam->alg_arg.p_outbuf->udetPersonBox[i].x);
        p_real_time_data->udetPersonBox[i].y = htons(algparam->alg_arg.p_outbuf->udetPersonBox[i].y);

        p_real_time_data->udetPersonBox[i].label = htonl(algparam->alg_arg.p_outbuf->udetPersonBox[i].label);
        p_real_time_data->udetPersonBox[i].confidence = htonl(algparam->alg_arg.p_outbuf->udetPersonBox[i].confidence);
        p_real_time_data->udetPersonBox[i].id = htonl(algparam->alg_arg.p_outbuf->udetPersonBox[i].id);
        p_real_time_data->udetPersonBox[i].distance[0] = htonl(algparam->alg_arg.p_outbuf->udetPersonBox[i].distance[0]);
        p_real_time_data->udetPersonBox[i].distance[1] = htonl(algparam->alg_arg.p_outbuf->udetPersonBox[i].distance[1]);
        p_real_time_data->udetPersonBox[i].landid = htonl(algparam->alg_arg.p_outbuf->udetPersonBox[i].laneid);
        p_real_time_data->udetPersonBox[i].speed = htons(algparam->alg_arg.p_outbuf->udetPersonBox[i].speed);
         p_real_time_data->udetPersonBox[i].speed_Vx = htons(algparam->alg_arg.p_outbuf->udetPersonBox[i].speed_Vx);
        p_real_time_data->udetPersonBox[i].rl_width = algparam->alg_arg.p_outbuf->udetPersonBox[i].width;
        p_real_time_data->udetPersonBox[i].rl_lenght = algparam->alg_arg.p_outbuf->udetPersonBox[i].length;

    }

    p_real_time_data->udetPersonNum = htons(algparam->alg_arg.p_outbuf->udetPersonNum);

    for (i = 0; i < MAX_REGION_NUM; i++) {
        p_real_time_data->personRegion[i].id = get_person_area_id(index, i);
        p_real_time_data->personRegion[i].personNum = htons(algparam->alg_arg.p_outbuf->uPersonRegionNum[i]);
#if 0
        realtime_person[index].perso_num[i] = algparam->alg_arg.p_outbuf->uPersonRegionNum[i]; //3.1.ÊµÊ±ï¿½ï¿½ï¿½Ë¼ï¿½ï¿½ï¿½ï¿½ï¿½ï¿?
        realtime_person[index].up_perso_num[i] =   algparam->alg_arg.p_outbuf->uPersonDirNum[i][0];
        realtime_person[index].down_perso_num[i] = algparam->alg_arg.p_outbuf->uPersonDirNum[i][1];
#endif
        total_upperson[index][i] += algparam->alg_arg.p_outbuf->uPersonDirNum[i][0];
        p_real_time_data->personRegion[i].upperson = htonl(total_upperson[index][i]);
        total_downperson[index][i] += algparam->alg_arg.p_outbuf->uPersonDirNum[i][1];
        p_real_time_data->personRegion[i].downperson = htonl(total_downperson[index][i]);

        //prt(info, "person area[%d]: up: %d down: %d", i, algparam->alg_arg.p_outbuf->uPersonDirNum[i][0], algparam->alg_arg.p_outbuf->uPersonDirNum[i][1]);

    }
    realtime_person[index].work_staus = algparam->alg_arg.p_outbuf->visibility || algparam->alg_arg.p_outbuf->fuzzyflag;
    //if (index == 0) {
    //  prt(info,"total_upperson[%d]: %d : %d ############uPersonDirNum[0]: %d ",index, total_upperson[index], p_real_time_data->upperson , algparam->alg_arg.p_outbuf->uPersonDirNum[0]);
    //   prt(info,"total_downperson[%d]: %d %d ############uPersonDirNum[1]: %d", index, total_downperson[index], p_real_time_data->downperson, algparam->alg_arg.p_outbuf->uPersonDirNum[1]);
    //}

    p_real_time_data->plateNumber = htons(algparam->alg_arg.p_outbuf->udetPlateNum);
    for(i = 0; i < algparam->alg_arg.p_outbuf->udetPlateNum; i++) {
        p_real_time_data->plate_objs[i].id = htonl(algparam->alg_arg.p_outbuf->car_number[i].id);
        p_real_time_data->plate_objs[i].x = htons(algparam->alg_arg.p_outbuf->car_number[i].x);
        p_real_time_data->plate_objs[i].y = htons(algparam->alg_arg.p_outbuf->car_number[i].y);
        p_real_time_data->plate_objs[i].h = htons(algparam->alg_arg.p_outbuf->car_number[i].h);
        p_real_time_data->plate_objs[i].w = htons(algparam->alg_arg.p_outbuf->car_number[i].w);
        p_real_time_data->plate_objs[i].landid = htonl(algparam->alg_arg.p_outbuf->car_number[i].landid);     
        p_real_time_data->plate_objs[i].confidence = algparam->alg_arg.p_outbuf->car_number[i].confidence;
        p_real_time_data->plate_objs[i].type = algparam->alg_arg.p_outbuf->car_number[i].type; 
        p_real_time_data->plate_objs[i].colour = algparam->alg_arg.p_outbuf->car_number[i].colour; 
        memcpy(p_real_time_data->plate_objs[i].car_number, algparam->alg_arg.p_outbuf->car_number[i].car_number, 50);
    }
}

int extern open_alg(int index, unsigned short gpu_index)
{
    //alloc_alg(&algparam[index],get_mCamDetectParam(index),get_mCamParam(index));
    alloc_alg(&algparam[index], get_mCamDetectParam(index), gpu_index, index);
    memset(&ex_static_info.static_info[index], 0, sizeof(mRealStaticInfo));
    memset(&ex_static_info.pre_car_num[index], 0, sizeof(unsigned int)*DETECTLANENUMMAX * 2);
    memset(&ex_static_info.car_num[index], 0, sizeof(unsigned int)*DETECTLANENUMMAX * 2);
    ex_static_info.reset_flag[index] = 1;

    return 0;
}

int extern run_alg(int index, unsigned char *y, unsigned char *u, unsigned char *v, unsigned short w, unsigned short h, unsigned int frame_no)
{
    mAlgParam *p_algparam = &algparam[index];

    if (p_algparam->tick++ == TICK_CHECK_POINT) {
        if (time_section_changed(get_mDetectTime(), p_algparam) > 0) {

            transform_arg_ctrl_DSP_VC(&p_algparam->alg_arg);
        }
        p_algparam->tick = TICK_ORI_POINT;
    }
    //prt(info, "cam[%d] frame_no: %u----%u", index, frame_no, htonl(frame_no));
    ex_static_info.static_info[index].frame_no = htonl(frame_no);
    //prt(info, "frameno: %d", frame_no);

    AlgProcessFrameData(index, p_algparam, y, u, v, w, h);
    pack_realtime_data(&ex_static_info, p_algparam, index);
    pack_sig_data(index);

    //add_person_flow(index, algparam[index].alg_arg.p_outbuf->uPersonDirNum);
    if ( g_camdetect[index].other.detecttype == 1 && PROTO_PRIVATE_PERSON == g_ivddevsets.pro_type) {
        //if (g_camdetect[index].other.detecttype == 1) { //ï¿½ï¿½ï¿½Ë¼ï¿½ï¿?
        if (g_camdetect[index].other.pensondetecttype == 1) {
            send_person_sig(index, p_algparam->alg_arg.p_outbuf->udetPersonNum
                            , g_camdetect[index].other.personlimit); //deng dai qu
        } else {
            send_person_sig(index, p_algparam->alg_arg.p_outbuf->udetPersonNum , 0); //guo jie
        }
        // }
    }

    if (g_member.cmd_play && (g_ivddevsets.pro_type == PROTO_WS_VIDEO || g_ivddevsets.pro_type == PROTO_WS_RADAR_VIDEO)  ) {
        if (p_algparam->alg_arg.p_outbuf->udetNum > 0 || p_algparam->alg_arg.p_outbuf->udetPersonNum > 0) {
            add_video_data_2_queue(index, (char *)p_algparam->alg_arg.p_outbuf, sizeof(OUTBUF));
        }
    }

    if (events_info[index].eventAreaNum > 0)
        add_event_list(index, y, u, v, &p_algparam->alg_arg.p_outbuf->eventData, h, w);

    return 0;
}

int extern reset_alg(int index, unsigned short gpu_index)
{
    release_alg(index);
    open_alg(index, gpu_index);
    return 0;
}

void extern release_alg(int index)
{
    transform_release_DSP_VC(&algparam[index].alg_arg);
}
void extern init_alg(int index)
{
    pthread_mutex_init(&person_flow_lock[index], NULL);

    algparam[index].time_section = TIME_SECTION_NULL;
    algparam[index].alg_index = ALG_NULL;
}

void add_person_flow(int index, Uint16 (*p_area_person)[MAX_DIRECTION_NUM], long long *p_density)
{
    long long ms = get_ms();
    pthread_cleanup_push(my_mutex_clean, &person_flow_lock[index]);
    pthread_mutex_lock(&person_flow_lock[index]);
    for (int i = 0; i < MAX_REGION_NUM; i++) {
        person_flow1[index][i] += p_area_person[i][0];
        person_flow2[index][i] += p_area_person[i][1];
        if (p_density[i] > 0 && (  (ms - p_density[i]) / 1000 > person_density[index][i] ) ) //second
            person_density[index][i] = (ms - p_density[i]) / 1000;
    }

    pthread_mutex_unlock(&person_flow_lock[index]);
    pthread_cleanup_pop(0);
}

void get_person_flow(int index, Uint16 (*p_area_person)[MAX_DIRECTION_NUM], long long  *p_density)
{
    pthread_cleanup_push(my_mutex_clean, &person_flow_lock[index]);
    pthread_mutex_lock(&person_flow_lock[index]);
    //flow1 = person_flow1[index];
    //flow2 = person_flow2[index];
    for (int i = 0; i < MAX_REGION_NUM ; i++) {
        p_area_person[i][0] = person_flow1[index][i];
        p_area_person[i][1] = person_flow2[index][i];
        p_density[i] = person_density[index][i];
    }
    memset(&person_flow1[index], 0, sizeof(Uint16)*MAX_REGION_NUM);
    memset(&person_flow2[index], 0, sizeof(Uint16)*MAX_REGION_NUM);
    memset(&person_density[index], 0, sizeof(int)*MAX_REGION_NUM);
    pthread_mutex_unlock(&person_flow_lock[index]);
    pthread_cleanup_pop(0);
}

///////////////////////////////////////////////////////////////////////
//static int beijing_fd=-1;
static char sig_ip[20];
static int sig_port;
static int flg = 0;
static int old_valid[8][8];
static int valid[8][8];
static int repeat_times[8][8];

#define DATA_START 0x7e//
#define DATA_END 0x7e//8
#define VER 0x01//3
#define OP 0x61//4
#define CLASS 0x1f//5

typedef struct cdata_type {
    uint8_t no;
    uint8_t exist;
    uint8_t percent;//zhan you lv
    uint8_t busy_state;// 1-5 , judge by percent
    uint8_t valid;
} cdata_t;

typedef struct pro_data_type {
    uint8_t direction;
    uint8_t channel_count;
    cdata_t channels[8];

} data_t;

void process_protocal(int index, data_t data , unsigned char dst[], int sz )
{
    dst[0] = DATA_START; //    bs.push_back(DATA_START);
    dst[1] = data.channel_count; //    bs.push_back(ID);
    dst[2] = VER; //    bs.push_back(VER);
    dst[3] = OP; //    bs.push_back(OP);
    dst[4] = CLASS; //    bs.push_back(this->cam_cfg.camera_id+0x10);

    dst[5] = data.direction; //    bs.push_back(0x01);//direction
    dst[6] = get_area_count(index);

    uint8_t check = 0;

    if (1) {
        check = dst[0];
        for (int i = 1; i < sz; i++) {
            check ^= dst[i];
        }
    }
    dst[sz - 2] = check; // bs.push_back(check);
    dst[sz - 1] = DATA_END; // bs.push_back(DATA_END);
}


int send_value1(int person_count, int thre, int cam_index, int area_index, int &need_send, int last_frames = 10)
{
    //  int send_flag=0;
    static int send_val[8][8];

#if 0
#else
    if (person_count > thre )
        valid[cam_index][area_index] = 1;
    else
        valid[cam_index][area_index] = 0;
    if (valid[cam_index][area_index] == old_valid[cam_index][area_index])
        repeat_times[cam_index][area_index]++;
    old_valid[cam_index][area_index] = valid[cam_index][area_index];

    if (repeat_times[cam_index][area_index] > last_frames && valid[cam_index][area_index] == 1)
    {
        if (send_val[cam_index][area_index] == 0)
            need_send = 1;
        repeat_times[cam_index][area_index] = 0;
        send_val[cam_index][area_index] = 1;
    }
    if (repeat_times[cam_index][area_index] > last_frames && valid[cam_index][area_index] == 0)
    {
        if (send_val[cam_index][area_index] == 1)
            need_send = 1;
        repeat_times[cam_index][area_index] = 0;
        send_val[cam_index][area_index] = 0;
    }
    if ( send_val[cam_index][area_index])
    {
        prt(info, "get exist %d %d,count %d", cam_index, area_index, person_count);
    }
    return send_val[cam_index][area_index];
#endif
}

void send_person_sig(int index, int person_count, int thre)
{
    int send_flag = 0;
    get_sig_ip(sig_ip);
    sig_port = get_sig_port();
    data_t t;
    unsigned  char buf[14 + 35];
    unsigned  char *cur = buf + 7;
    for (int i = 0; i < 8; i++) {
        int need_send = 0;
        int id = get_person_area_id(index, i);

        t.channel_count = atoi(g_ivddevsets.devUserNo);
        int di = get_cam_direction(index);
        t.direction = di;
        if (thre)
            t.channels[i].exist = send_value1(person_count, thre, index, i, need_send);
        else {
            t.channels[i].exist = send_value1(person_count, thre, index, i, need_send, 40);
        }
        if (need_send)
            send_flag = 1;
        t.channels[i].no = id;
        //  prt(info,"(%d)",id);
        cur[0 + i * 5] = id;

        if (id) {
            cur[1 + i * 5] = t.channels[i].exist;
            cur[2 + i * 5] = 50; //TODO:ocupy percent
            cur[3 + i * 5] = 5;
            cur[4 + i * 5] = 1;
        } else {
            cur[1 + i * 5] = 0;
            cur[2 + i * 5] = 0; //TODO:ocupy percent
            cur[3 + i * 5] = 0;
            cur[4 + i * 5] = 0;
        }
    }

    process_protocal(index, t, buf, 14 + 35);

    if (fd_person > 0) {
        if (send_flag) {
            int sended = UdpSendData(fd_person, sig_ip, sig_port,
                                     (char *) buf, 14 + 35);
            //   prt(info,"send %d bytes  ,person count %d (thre %d).  in_out: %d",sended,person_count,thre,send_val);
        }
    } else {
        fd_person = UdpCreateSocket(get_random_port());

    }
    // prt(info,"send 4");
}


void detector_static_data(int index)
{
    mAlgParam algparam;
    mAlgParam *p_algparam = (mAlgParam *)&algparam;

    pthread_mutex_lock(&holder[index].sig_data_lock);
    memcpy(&algparam, &holder[index].algparam, sizeof(mAlgParam));
    //pthread_mutex_unlock(&holder[index].sig_data_lock);

    char bBackgroundRefreshed = 0x1;
    unsigned char status = get_holder_status(index);
    m_sig_data *p_channel_rst = get_locked_sig_data(index);
    m_sig_data *FVDChannel = p_channel_rst;

    for (int i = 0; i < p_algparam->algNum; i++) {

        if ( status & 0x02) {

            FVDChannel->Eachchannel[i].mDetectChannelIndex = get_lane_index(index, i);

            if (FVDChannel->Eachchannel[i].mDetectChannelIndex < 41
                    && FVDChannel->Eachchannel[i].mDetectChannelIndex > 96)
                FVDChannel->Eachchannel[i].mDetectChannelIndex = 255;
            FVDChannel->Eachchannel[i].mQueueLength = p_algparam->alg_arg.p_outbuf->uVehicleQueueLength[i];
            //        p_algparam->alg_arg.p_outbuf->uLastVehicleLength[i];
            //prt(info, "uLastVehicleLength[%d]: %2X", i, p_algparam->alg_arg.p_outbuf->uLastVehicleLength[i]);
            if (p_channel_rst->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow > 0) {
                FVDChannel->Eachchannel[i].mRealTimeSingleSpeed =
                    p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleSpeed;
            }
            if (0x1 == p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag)
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bOccupyStatus1 = 0x1; // 0.5s
            else
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bOccupyStatus1 = 0x0; // 0.5s

            FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bFix0 = 0x0;
#if 0
            if (!p_channel_rst->car_info.g_flow[i])
                p_channel_rst->car_info.g_flow[i] = p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectOutSum;
            else {
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow =
                    p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectOutSum
                    - p_channel_rst->car_info.g_flow[i];
                prt(info, "flowtotal[%d]: %02X  %02X", i, p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectOutSum, p_channel_rst->car_info.g_flow[i]);
                p_channel_rst->car_info.g_flow[i] = p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].DetectOutSum;

                if (FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow > 3)
                    FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow = 0x3;
            }
#endif
            if (FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow > 3)
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow = 0x3;

            if (p_channel_rst->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.flow > 0 &&
                    0x01 == p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag) {
                if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleLength > 7)
                    FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bVehicleType = 0x3;
                else if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleLength > 5)
                    FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bVehicleType = 0x2;
                else
                    FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bVehicleType = 0x1;
            } else
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bVehicleType = 0x0;
#if 0

            if (p_channel_rst->car_info.g_50frame1[i] && p_channel_rst->car_info.g_50frame2[i]) {
                FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil =
                    (p_channel_rst->car_info.g_50frame1[i] - p_channel_rst->car_info.g_50frame2[i]) * (0.4);
                if (FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil > 254)
                    FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil = 255;
            } else
                FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil = 0x0;

            if (p_channel_rst->car_info.g_occupancyframe[i]) {
                FVDChannel->Eachchannel[i].mOccupancyOfHeaderVirtualCoil =
                    ((float) (((float) p_channel_rst->car_info.g_occupancyframe[i]) / 12))
                    * 100;
                p_channel_rst->car_info.g_occupancyframe[i] = 0;
            } else
                FVDChannel->Eachchannel[i].mOccupancyOfHeaderVirtualCoil = 0x0;
#endif
            if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleHeadtime > 25400) {//25.4 s
                FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil = 255;
            } else {
                FVDChannel->Eachchannel[i].mHeadwayTimeOfHeaderVirtualCoil = p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].uVehicleHeadtime / 100; //0.1s
            }

            //prt(info, "g_staytm[%d]: %d", i, p_channel_rst->car_info.g_staytm[i]);
            if (p_channel_rst->car_info.g_staytm[i] >= 500) {
                FVDChannel->Eachchannel[i].mOccupancyOfHeaderVirtualCoil = 100;
            } else {
                FVDChannel->Eachchannel[i].mOccupancyOfHeaderVirtualCoil = p_channel_rst->car_info.g_staytm[i] / 5; //persen 100
            }

#if 0

            if (p_channel_rst->car_info.g_50frametail1[i]
                    && p_channel_rst->car_info.g_50frametail2[i]) {
                FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil =
                    (p_channel_rst->car_info.g_50frametail1[i]
                     - p_channel_rst->car_info.g_50frametail2[i]) * (0.4);
                if (FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil
                        > 254)
                    FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil =
                        255;
            } else
                FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil = 0x0;

            //printf("g_occupancyframetail[%d]:[%d]\n", i, g_occupancyframetail[i]);
            if (p_channel_rst->car_info.g_occupancyframetail[i]) {
                FVDChannel->Eachchannel[i].mOccupancyOfTailVirtualCoil =
                    ((float) (((float) p_channel_rst->car_info.g_occupancyframetail[i])
                              / 12)) * 100;
                //printf("g_occupancyframetail[%d]:[%f]\n", i, (float)(((float)g_occupancyframetail[i])/12));
                p_channel_rst->car_info.g_occupancyframetail[i] = 0;
            } else
                FVDChannel->Eachchannel[i].mOccupancyOfTailVirtualCoil = 0x0;
#endif

            if (p_algparam->alg_arg.p_outbuf->CoilAttribute[i][1].uVehicleHeadtime > 25400) {//25.4 s
                FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil = 255;
            } else {
                FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil = p_algparam->alg_arg.p_outbuf->CoilAttribute[i][1].uVehicleHeadtime / 100; //0.1s
            }


            if (p_channel_rst->car_info.g_staytm_tail[i] >= 500) {
                FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil = 100;
            } else {
                FVDChannel->Eachchannel[i].mHeadwayTimeOfTailVirtualCoil = p_channel_rst->car_info.g_staytm_tail[i] / 5; //persen 100
            }

            if (0x1 == p_algparam->alg_arg.p_outbuf->CoilAttribute[i][1].calarflag)
                FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bOccupyStatus1 = 0x01;

            // FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bOccupyStatus1 =
            //         p_algparam->alg_arg.p_outbuf->IsCarInTail[i]; // 0.5s
            FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bFix0 = 0x0;

            if (p_algparam->alg_arg.p_outbuf->visibility) {
                print_alg("==error p_algparam visibility: 0x1");
                FVDChannel->Eachchannel[i].mWorkStatusOfDetectChannle.bDataIsValid = 0x1;
            } else {
                FVDChannel->Eachchannel[i].mWorkStatusOfDetectChannle.bDataIsValid = 0x0;
            }

            bBackgroundRefreshed &= p_algparam->alg_arg.p_outbuf->getQueback_flag[i];
            if (FVDChannel->EachStatus.mCameralStatus.bBackgroundRefreshed
                    != bBackgroundRefreshed) {
                p_channel_rst->camera_state_change = 1;
            }
            FVDChannel->EachStatus.mCameralStatus.bBackgroundRefreshed =
                bBackgroundRefreshed;
            //Log0("getQueback_flag[%d]:[%d] bBackgroundRefreshed:[%d]\n",i,p_algparam->alg_arg.p_outbuf->getQueback_flag[i], bBackgroundRefreshed);

            int ori = FVDChannel->EachStatus.mCameralStatus.bWorkMode ;
#if 0
            if (p_algparam->alg_index == 0x1)
            {
                //  prt(info,"index %d,day time",index);
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x0;
            }
            else if (p_algparam->alg_index == 0x2)
            {
                //      prt(info,"index %d,night time",index);
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x2;
            }


            if ((p_algparam->LaneIn.uEnvironmentStatus == 0x1
                    && p_algparam->alg_index == 0x1)
                    || (p_algparam->LaneIn.uEnvironmentStatus == 0x3
                        && p_algparam->alg_index == 0x2)) {
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x1;
                //prt(info,"index %d,shift time==> %d",index,p_algparam->LaneIn.uEnvironmentStatus);
                //      FVDChannel->Eachchannel[i].mWorkStatusOfDetectChannle.bDataIsValid=0x1; //é—è·¨å–é‹å©šå¹é‘èŠ¥æ™¸é–ºå‚˜å€–ç€šå½’æŸ¨é”å³°å»ºé–¿ç‡‚ï¿?
            }
#endif

            switch (p_algparam->time_section) {
            case MORNING:
            case DUSK:
            {
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x01;
            }
            break;
            case DAYTIME:
            {
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x00;
            }
            break;
            case NIGHT:
            {
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x02;
            }
            break;
            default:
            {
                FVDChannel->EachStatus.mCameralStatus.bWorkMode = 0x03;
            }
            break;
            }

            if (ori != FVDChannel->EachStatus.mCameralStatus.bWorkMode ) {
                p_channel_rst->camera_state_change = 1;

            }

            FVDChannel->EachStatus.mCameralStatus.bH264DecodeStatus = 0x1;
            FVDChannel->EachStatus.mCameralStatus.bCameralOnLine = 0x1;
            if (FVDChannel->EachStatus.mCameralStatus.bPictureStable == (p_algparam->alg_arg.p_outbuf->fuzzyflag ? 1 : 0)  ) {
                p_channel_rst->camera_state_change = 1;
            }

            if (p_algparam->alg_arg.p_outbuf->fuzzyflag) {
                FVDChannel->EachStatus.mCameralStatus.bPictureStable = 0x0;
                FVDChannel->Eachchannel[i].mWorkStatusOfDetectChannle.bDataIsValid =
                    0x1;
            } else {
                FVDChannel->EachStatus.mCameralStatus.bPictureStable = 0x1;
            }

            FVDChannel->lane_num = p_algparam->algNum;
            FVDChannel->status = 0x1;
            FVDChannel->EachStatus.mCameralStatus.bBackgroundRefreshed = bBackgroundRefreshed;
        }

        if (status & 0x01) {
            if (0x1 == p_algparam->alg_arg.p_outbuf->CoilAttribute[i][0].calarflag)
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bOccupyStatus0 = 0x1;  // 0.25s
            else
                FVDChannel->Eachchannel[i].mDetectDataOfHeaderVirtualCoil.bOccupyStatus0 = 0x0;  // 0.25s
            // FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bOccupyStatus0 = p_algparam->alg_arg.p_outbuf->IsCarInTail[i];  // 0.25s
            if (0x1 == p_algparam->alg_arg.p_outbuf->CoilAttribute[i][1].calarflag)
                FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bOccupyStatus0 = 0x01;
            else
                FVDChannel->Eachchannel[i].mDetectDataOfTailVirtualCoil.bOccupyStatus0 = 0x0;
        }
    }

    pthread_mutex_unlock(&holder[index].sig_data_lock);

    //prt(info, "camera_state_change is: %d  %d status: %d", p_algparam->time_section, FVDChannel->EachStatus.mCameralStatus.bWorkMode, p_channel_rst->camera_state_change);
    if (p_channel_rst->camera_state_change) {
        //prt(info, "send message");
        send_message(0x01, NULL);
        p_channel_rst->camera_state_change = 0;
    }
}


void add_info_in_pic(int index, Mat img, void *data, char *save_pic_path)
{
    int x, y;
    char title[50] = {0};
    char pic_name[100] = {0};
    NPOUTBUF *p_data = (NPOUTBUF *)data;
    for (int i = 0; i < p_data->uNonMotorNum; i++) {
        x = p_data->nonMotorInfo[i].nonMotorBox.x;
        y =  p_data->nonMotorInfo[i].nonMotorBox.y;
        Rect rect(x, y, p_data->nonMotorInfo[i].nonMotorBox.width, p_data->nonMotorInfo[i].nonMotorBox.height);
        cv::rectangle(img, rect, Scalar(255, 0, 0), 5, 8, 0);

        for (int j = 0; j < p_data->nonMotorInfo[i].helmetNum; j++) {

            Rect gdirec(p_data->nonMotorInfo[i].helmetBox[j].x, p_data->nonMotorInfo[i].helmetBox[j].y, p_data->nonMotorInfo[i].helmetBox[j].width, p_data->nonMotorInfo[i].helmetBox[j].height);
            cv::rectangle(img, gdirec, Scalar(0, 0, 255), 5, 8, 0);
        }

        if (p_data->nonMotorInfo[i].helmetNum > 0 && p_data->nonMotorInfo[i].riderNum <= p_data->nonMotorInfo[i].helmetNum)
            snprintf(title, 50, "%d:%d-Y", i+1, p_data->nonMotorInfo[i].riderNum); //带冒
        else
            snprintf(title, 50, "%d:%d-N", i+1, p_data->nonMotorInfo[i].riderNum);

        y -= 10;
        cv::putText(img, title, cv::Point(x, y), cv::FONT_HERSHEY_SIMPLEX, 1.8, Scalar(0, 255, 255), 3);
    }

    struct timeval tv;
    gettimeofday(&tv, NULL);
    sprintf(pic_name, "%snm_%d_%lu.jpg", "/ftphome/pic/nonmotor/", index, tv.tv_sec * 1000000 + tv.tv_usec);
    imwrite(pic_name, img);

    strcpy(save_pic_path, pic_name);
}


Mat prv_img, pprv_img;
char pprv_pic_path[200] = {0};
char prv_pic_path[200] = {0};

void handle_pic(int index, char *file_url, char *mv_dir)
{

    DIR *dir;
    struct dirent *ptr;
    unsigned short cnt = 0;
    char cmd_buf[200] = {0};
    char pic_path[100] = {0};
    char read_pic_path[200] = {0};

    Mat img;
    mAlgParam *p_algparam = &algparam[index];
   
    if ((dir = opendir(file_url)) == NULL)
    {
        prt(info, "Open dir error...");
        return;
    }

    while ((ptr = readdir(dir)) != NULL ) //|| pprv_img.data
    {

        if (strcmp(ptr->d_name, ".") == 0 || strcmp(ptr->d_name, "..") == 0) ///current dir OR parrent dir
            continue;
        else if (ptr->d_type == 8)   ///file
        {
            char *p_last_name = strrchr(ptr->d_name,'.'); 
            if (p_last_name && (!strcmp(p_last_name, ".jpg") || !strcmp(p_last_name, ".jpeg") || !strcmp(p_last_name, ".png") || !strcmp(p_last_name, ".bmp") ) )
                p_last_name = p_last_name;
            else
                continue;

            sprintf(read_pic_path, "%s/%s\0", file_url, ptr->d_name);
            prt(info, ".............load picture file: %s", read_pic_path);
            img.release();
            img = imread(read_pic_path);
            if (!img.data) {
                prt(info, "load picture file: %s failed", read_pic_path);
                continue;
            }

            //Mat bgr(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));
            NP_Proc_DSP_VC(index, img, &algparam->alg_arg);

            if (pprv_img.data) {
                //handle output data
                add_info_in_pic(index, pprv_img, &algparam->alg_arg.p_outbuf->NPData, pic_path);
                //insert into database
                pic_data_insert_mysql(index, &algparam->alg_arg.p_outbuf->NPData, pic_path);

                snprintf(cmd_buf, 200, "mv %s %s", pprv_pic_path, mv_dir);//picture is handled,move to mv_dir director
                system(cmd_buf);
            }

            pprv_img.release();
            pprv_img = prv_img.clone();
            prv_img.release();
            prv_img = img.clone();

            strcpy(pprv_pic_path, prv_pic_path);
            strcpy(prv_pic_path, read_pic_path);
        }
        else if (ptr->d_type == 10)   ///link file
        {
            printf("link name:%s\n", ptr->d_name);
        }
        else if (ptr->d_type == 4)   ///dir
        {
            printf("dir  name:%s\n", ptr->d_name);
        }
    }
}

void handle_last_pic(int index, char *mv_dir) //last two picture
{
    char cmd_buf[200] = {0};
    char pic_path[100] = {0};

    mAlgParam *p_algparam = &algparam[index];

    while ( pprv_img.data || prv_img.data)
    {
        if (prv_img.data)
            NP_Proc_DSP_VC(index, prv_img, &algparam->alg_arg);
        else
            NP_Proc_DSP_VC(index, pprv_img, &algparam->alg_arg); 
        
        if (pprv_img.data) {
            
            //handle output data
            add_info_in_pic(index, pprv_img, &algparam->alg_arg.p_outbuf->NPData, pic_path);
            //insert into database
            pic_data_insert_mysql(index, &algparam->alg_arg.p_outbuf->NPData, pic_path);

            snprintf(cmd_buf, 200, "mv %s %s", pprv_pic_path, mv_dir);//picture is handled,move to mv_dir director
            system(cmd_buf);
        }
        pprv_img.release();
        pprv_img = prv_img.clone();
        prv_img.release();

        strcpy(pprv_pic_path, prv_pic_path);
 
    }
}
//发送结果到客户端
void send_result_to_cli(int index, void *data, char *img_id, struct sockaddr_in *sock_in)
{
    m_img_det_info_t img_info = {0};
    NPOUTBUF *p_data = (NPOUTBUF *)data;
    
    for (int i = 0; i < p_data->uNonMotorNum; i++) {
        img_info.motors[i].vc_pst.x = htons(p_data->nonMotorInfo[i].nonMotorBox.x);
        img_info.motors[i].vc_pst.y =  htons(p_data->nonMotorInfo[i].nonMotorBox.y);
        img_info.motors[i].vc_pst.width = htons(p_data->nonMotorInfo[i].nonMotorBox.width);
        img_info.motors[i].vc_pst.height = htons(p_data->nonMotorInfo[i].nonMotorBox.height);

        img_info.motors[i].hat_num = htons(p_data->nonMotorInfo[i].helmetNum);
        for (int j = 0; j < p_data->nonMotorInfo[i].helmetNum; j++) {
            img_info.motors[i].hat_pos[j].x = htons(p_data->nonMotorInfo[i].helmetBox[j].x);
            img_info.motors[i].hat_pos[j].y = htons(p_data->nonMotorInfo[i].helmetBox[j].y);
            img_info.motors[i].hat_pos[j].width = htons(p_data->nonMotorInfo[i].helmetBox[j].width);
            img_info.motors[i].hat_pos[j].height = htons(p_data->nonMotorInfo[i].helmetBox[j].height);
        }

        img_info.motors[i].person_num = htons(p_data->nonMotorInfo[i].riderNum);
    }

     strncpy(img_info.img_id,img_id, MAX_IMG_ID_LEN);
    if (p_data->uNonMotorNum > 0){
        img_info.non_motor_num = htons(p_data->uNonMotorNum);
    }

    int sock_fd = get_non_motor_fd();
    if ( sock_fd > 0 && sock_in->sin_port > 0) {
        sendto(sock_fd, (char*)&img_info, sizeof(m_img_det_info_t), 0, (struct sockaddr*) sock_in, sizeof(struct sockaddr));
    }
}

Mat prv_img_cli, pprv_img_cli;
char pprv_pic_path_cli[200] = {0};
char prv_pic_path_cli[200] = {0};
char pprv_img_id[MAX_IMG_ID_LEN+1] = {0};
char prv_img_id[MAX_IMG_ID_LEN+1] = {0};
struct sockaddr_in pprv_sock_in = {0};
struct sockaddr_in prv_sock_in = {0};

void handle_non_motor_pic_cli(int index, char *file_path, char *mv_dir, char *img_id, struct sockaddr_in *p_cli_fd)
{
    struct dirent *ptr;
    unsigned short cnt = 0;
    char cmd_buf[200] = {0};
    char pic_path[100] = {0};
    char read_pic_path[200] = {0};

    Mat img;
    mAlgParam *p_algparam = &algparam[index];

   if (file_path && 0 == access(file_path, F_OK))   ///file
    {
        img.release();
        img = imread(file_path);
        if (!img.data) {
            prt(info, "load picture file: %s failed", file_path);
            return;
        }

        //Mat bgr(img.rows, img.cols, CV_8UC3, Scalar(0, 0, 0));
        NP_Proc_DSP_VC(index, img, &algparam->alg_arg);

        if (pprv_img_cli.data) {
            //handle output data
            add_info_in_pic(index, pprv_img_cli, &algparam->alg_arg.p_outbuf->NPData, pic_path);
            //insert into database
            pic_data_insert_mysql(index, &algparam->alg_arg.p_outbuf->NPData, pic_path);
            if (mv_dir) {
                snprintf(cmd_buf, 200, "mv %s %s", pprv_pic_path_cli, mv_dir);//picture is handled,move to mv_dir director
                system(cmd_buf);
            }
            send_result_to_cli(index, &algparam->alg_arg.p_outbuf->NPData, pprv_img_id, &pprv_sock_in);
        }

        pprv_img_cli.release();
        pprv_img_cli = prv_img_cli.clone();
        prv_img_cli.release();
        prv_img_cli = img.clone();

        strcpy(pprv_pic_path_cli, prv_pic_path_cli);
        strcpy(prv_pic_path_cli, file_path);
        strncpy(pprv_img_id, prv_img_id, MAX_IMG_ID_LEN);
        strncpy(prv_img_id, img_id, MAX_IMG_ID_LEN);
        memcpy(&pprv_sock_in, &prv_sock_in, sizeof(struct sockaddr_in));
        memcpy(&prv_sock_in, p_cli_fd, sizeof(struct sockaddr_in));
    }
    
}


void handle_last_pic_cli(int index, char *mv_dir) //last two picture
{
    char cmd_buf[200] = {0};
    char pic_path[100] = {0};

    mAlgParam *p_algparam = &algparam[index];

    while ( pprv_img_cli.data || prv_img_cli.data)
    {
        if (prv_img_cli.data)
            NP_Proc_DSP_VC(index, prv_img_cli, &algparam->alg_arg);
        else
            NP_Proc_DSP_VC(index, pprv_img_cli, &algparam->alg_arg); 
        
        if (pprv_img_cli.data) {
            //handle output data
            add_info_in_pic(index, pprv_img_cli, &algparam->alg_arg.p_outbuf->NPData, pic_path);
            //insert into database
            pic_data_insert_mysql(index, &algparam->alg_arg.p_outbuf->NPData, pic_path);
            if (mv_dir) {
                snprintf(cmd_buf, 200, "mv %s %s", pprv_pic_path_cli, mv_dir);//picture is handled,move to mv_dir director
                system(cmd_buf);
            }

            send_result_to_cli(index, &algparam->alg_arg.p_outbuf->NPData, pprv_img_id, &pprv_sock_in);
        }
        pprv_img_cli.release();
        pprv_img_cli = prv_img_cli.clone();
        prv_img_cli.release();

        strcpy(pprv_pic_path_cli, prv_pic_path_cli);
        strncpy(pprv_img_id, prv_img_id, MAX_IMG_ID_LEN);
        memcpy(&pprv_sock_in, &prv_sock_in, sizeof(struct sockaddr_in));
    }
}


