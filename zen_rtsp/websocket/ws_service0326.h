/*
 * websocket_service.h
 *
 *  Created on: 2016??6??22??
 *      Author: Administrator
 */

#ifndef INCLUDE_WEBSOCKET_SERVICE_H_
#define INCLUDE_WEBSOCKET_SERVICE_H_
#include "queue.h"
#include "m_arith.h"

typedef struct  {
	int  cam_index;
	char data_ms[50];
	OUTBUF out_buf;
} WS_Node_Data_t;

typedef struct ws_list {
	Queue *radar_queue;
	Queue *video_queue;
} WS_LIST_t;

void *http_server(void *data);
void *websocket_server(void *data);
//
void create_queue();
void *radar_client(void *data);
void add_radar_data_2_queue(char *buf, int len);
void add_video_data_2_queue(int index, char *buf, int len);

void *ws_hanle_realtime_data(void *data);
void ws_send_msg(void **p_vhd, char *p_msg, unsigned short msg_len);

#endif /* INCLUDE_WEBSOCKET_SERVICE_H_ */
