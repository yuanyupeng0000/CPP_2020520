/*
 * client_net.h
 *
 *  Created on: 2016��10��19��
 *      Author: root
 */

#ifndef CLIENT_NET_H_
#define CLIENT_NET_H_
#include "client_obj.h"


typedef struct client_ip_data {
        char ip[16];
        int camera_index;
        int time;
        int fd;
        int times;
} m_client_ip_data;



int handle_buffer(int index, unsigned char *buf, int len);
void init_camera();
void init_config();

mCamDetectParam *get_mCamDetectParam(int index);
mCamParam *get_mCamParam(int index);

//
IVDTimeStatu get_mDetectTime();
//

int get_cam_status(int index);
int get_cam_id(int index);
int get_cam_direction(int index);
int get_cam_location(int index);

int get_dev_id();
void get_sig_ip(char *des);
int  get_sig_port();
int get_person_area_id(int cam_index,int area_index);
int get_area_count(int cam_index);

int get_lane_num(int index);
//void save_obj(unsigned char * p_obj,int class_type,int index);
void client_output(int index);
//mRealStaticInfo *client_get_info(int index);

int get_lane_index(int index,int lane_i);
int get_direction(int index);
void get_udp_info(char *ip, unsigned short *port); //��ȡudp ip��port
//
int del_client(char *ip,int index, int sfd);
void add_client(char *ip,int index, int sfd);
//
void cmd_output(int index, unsigned short cmd_no);
int send_cmd_status(int index, unsigned short cmd_no);
int pack_cmd_status(int index, unsigned char cmd_no, unsigned char status, unsigned char *buff);

//#define CAM_MAX 4
#define BUFFER_MAX  8192//1000
#define TCP_TIMEOUT 100
#endif /* CLIENT_NET_H_ */
