
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>
#include <signal.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <time.h>
#include <sys/time.h>
#include <errno.h>
#include <sys/msg.h>
#include <sys/ipc.h>
#include "common.h"
#include "client_obj.h"
#include "g_define.h"
#include "fvdconfig.h"
#include "csocket.h"
#include "libwebsockets.h"
#include "ws_service.h"
#include "m_arith.h"
#include "camera_service.h"

//#define RADAR_IP  "192.168.1.166"
#define RADAR_SERVER_PORT       4088
#define BUFFER_SIZE             1500
//#define READ_TIMEOUT            20 //ms
//#define WRITE_TIMEOUT           20 //ms
//
extern mGlobalVAR       g_member; //全局变量
extern IVDDevSets	    g_ivddevsets;
extern m_camera_info    cam_info[CAM_MAX];
extern IVDCAMERANUM     g_cam_num;
extern mCamDetectParam g_camdetect[CAM_MAX];
//
WS_LIST_t g_ws_queue;

/***************************************************/
/**** http server ***/
/**** get post ***/
/***************************************************/

struct pss {
	struct lws_spa *spa;
};

static const char * const param_names[] = {
	"Longitude",
	"Latitude",
	"Altitude",
	"RelativeHeight",
	"YawAngle",
};

static int
callback_http_get(struct lws *wsi, enum lws_callback_reasons reason,
                  void *user, void *in, size_t len)
{
	uint8_t buf[LWS_PRE + 2048], *start = &buf[LWS_PRE], *p = start,
	                              *end = &buf[sizeof(buf) - LWS_PRE - 1];
	int n;

	switch (reason) {
	case LWS_CALLBACK_HTTP:
		/* in contains the url part after our mountpoint /dyn, if any */
		//lws_snprintf(pss->path, sizeof(pss->path), "%s", (const char *)in);

		if (lws_add_http_common_headers(wsi, HTTP_STATUS_OK,
		                                "application/json",
		                                LWS_ILLEGAL_HTTP_CONTENT_LEN, /* no content len */
		                                &p, end))
			return 1;
		if (lws_finalize_write_http_header(wsi, start, &p, end))
			return 1;

		/* write the body separately */
		lws_callback_on_writable(wsi);

		return 0;

	case LWS_CALLBACK_HTTP_WRITEABLE:
	{
		//n = LWS_WRITE_HTTP_FINAL;
		mPositionData info = {0};
		ReadPosition(&info, 0);
		p += lws_snprintf((char *)p, end - p, "{\"Longitude\":%f,\"Latitude\":%f,\"Altitude\":%f,\"RelativeHeight\":%f,\"YawAngle\":%f}"
		                  , info.Longitude, info.Latitude, info.Altitude, info.RelativeHeight, info.YawAngle);

		if (lws_write(wsi, (uint8_t *)start, lws_ptr_diff(p, start), (lws_write_protocol)n) !=
		        lws_ptr_diff(p, start))
			return 1;

		//if (n == (int)LWS_WRITE_HTTP_FINAL) {
		if (lws_http_transaction_completed(wsi))
			return -1;
		//} else
		//	lws_callback_on_writable(wsi);
	}
	return 0;

	default:
		break;
	}

	return lws_callback_http_dummy(wsi, reason, user, in, len);
}


static int
callback_http_post(struct lws *wsi, enum lws_callback_reasons reason, void *user,
                   void *in, size_t len)
{
	struct pss *pss = (struct pss *)user;
	uint8_t buf[LWS_PRE + LWS_RECOMMENDED_MIN_HEADER_SPACE], *start = &buf[LWS_PRE],
	                                                          *p = start, *end = &buf[sizeof(buf) - 1];
	int n = 0;

	switch (reason) {
	case LWS_CALLBACK_HTTP:

		lwsl_user("LWS_CALLBACK_HTTP\n");

		if (lws_add_http_common_headers(wsi, HTTP_STATUS_OK,
		                                "application/json",	LWS_ILLEGAL_HTTP_CONTENT_LEN, &p, end))
			return 1;
		if (lws_finalize_write_http_header(wsi, start, &p, end))
			return 1;

		return 0;
		break;
	case LWS_CALLBACK_HTTP_BODY:
		;
		/* create the POST argument parser if not already existing */

		if (!pss->spa) {
			pss->spa = lws_spa_create(wsi, param_names,
			                          LWS_ARRAY_SIZE(param_names), 1024,
			                          NULL, NULL); /* no file upload */
			if (!pss->spa)
				return -1;
		}

		/* let it parse the POST data */

		if (lws_spa_process(pss->spa, (const char *)in, (int)len))
			return -1;
		break;

	case LWS_CALLBACK_HTTP_BODY_COMPLETION:
	{
		/* inform the spa no more payload data coming */
		mPositionData info;
		memset(&info, 0, sizeof(mPositionData));

		//lwsl_user("LWS_CALLBACK_HTTP_BODY_COMPLETION\n");
		lws_spa_finalize(pss->spa);

		/* we just dump the decoded things to the log */
		int  num_param_names = (int)LWS_ARRAY_SIZE(param_names);
#if 0
		for (n = 0; n < num_param_names; n++) {

			if (!lws_spa_get_string(pss->spa, n))
				lwsl_user("%s: undefined\n", param_names[n]);
			else {
				lwsl_user("%s: (len %d) '%s'\n",
				          param_names[n],
				          lws_spa_get_length(pss->spa, n),
				          lws_spa_get_string(pss->spa, n));

			}
		}

		n = 0;
#endif
		if (lws_spa_get_string(pss->spa, 0)) {
			info.Longitude = atof(lws_spa_get_string(pss->spa, 0));
			n++;
		}

		if (lws_spa_get_string(pss->spa, 1)) {
			info.Latitude = atof(lws_spa_get_string(pss->spa, 1));
			n++;
		}

		if (lws_spa_get_string(pss->spa, 2)) {
			info.Altitude = atof(lws_spa_get_string(pss->spa, 2));
			n++;
		}

		if (lws_spa_get_string(pss->spa, 3)) {
			info.RelativeHeight = atof(lws_spa_get_string(pss->spa, 3));
			n++;
		}

		if (lws_spa_get_string(pss->spa, 4)) {
			info.YawAngle = atof(lws_spa_get_string(pss->spa, 4));
			n++;
		}

		if (n == num_param_names) {
			WritePosition(&info, 0);
		}
	}
	break;

	case LWS_CALLBACK_HTTP_DROP_PROTOCOL:
		/* called when our wsi user_space is going to be destroyed */
		if (pss->spa) {
			lws_spa_destroy(pss->spa);
			pss->spa = NULL;
		}
		break;

	default:
		break;
	}

	return lws_callback_http_dummy(wsi, reason, user, in, len);
}

#define LWS_PLUGIN_PROTOCOL_GET_DEMO \
{ \
  "protocol-get-demo", \
  callback_http_get, \
  sizeof(struct pss), \
  1024, \
  0, NULL, 0 \
}

#define LWS_PLUGIN_PROTOCOL_POST_DEMO \
{ \
   "protocol-post-demo", \
   callback_http_post, \
   sizeof(struct pss), \
   1024, \
   0, NULL, 0 \
}


static struct lws_protocols protocols[] = {
	{ "http-only", lws_callback_http_dummy, 0, 0, },
	LWS_PLUGIN_PROTOCOL_POST_DEMO,
	LWS_PLUGIN_PROTOCOL_GET_DEMO,
	{ NULL, NULL, 0, 0 } /* terminator */
};


static const struct lws_http_mount mount_get = {
	NULL, /* linked-list pointer to next*/
	"/radarConfigure/getV2XCfg",		/* mountpoint in URL namespace on this vhost */
	NULL,	/* handler */
	NULL,	/* default filename if none given */
	"protocol-get-demo",
	NULL,
	NULL,
	NULL,
	0,
	0,
	0,
	0,
	0,
	0,
	LWSMPRO_CALLBACK,	/* origin points to a callback */
	25,			/* strlen("/formtest"), ie length of the mountpoint */
	NULL,
	{ NULL, NULL } // sentinel
};


static const struct lws_http_mount mount_post = {
	(struct lws_http_mount *)&mount_get, /* linked-list pointer to next*/
	"/radarConfigure/setV2XCfg",		/* mountpoint in URL namespace on this vhost */
	"protocol-post-demo",	/* handler */
	NULL,	/* default filename if none given */
	NULL,
	NULL,
	NULL,
	NULL,
	0,
	0,
	0,
	0,
	0,
	0,
	LWSMPRO_CALLBACK,	/* origin points to a callback */
	25,			/* strlen("/formtest"), ie length of the mountpoint */
	NULL,

	{ NULL, NULL } // sentinel
};



/* default mount serves the URL space from ./mount-origin */

static const struct lws_http_mount mount = {
	/* .mount_next */	       (struct lws_http_mount *)&mount_post,		/* linked-list "next" */
	/* .mountpoint */		"/",		/* mountpoint URL */
	/* .origin */		"./default",	/* serve from dir */
	/* .def */			"index.html",	/* default filename */
	/* .protocol */			NULL,
	/* .cgienv */			NULL,
	/* .extra_mimetypes */		NULL,
	/* .interpret */		NULL,
	/* .cgi_timeout */		0,
	/* .cache_max_age */		0,
	/* .auth_mask */		0,
	/* .cache_reusable */		0,
	/* .cache_revalidate */		0,
	/* .cache_intermediaries */	0,
	/* .origin_protocol */		LWSMPRO_FILE,	/* files in a dir */
	/* .mountpoint_len */		1,		/* char count */
	/* .basic_auth_login_file */	NULL,
};

void *http_server(void *data)
{
	struct lws_context_creation_info info;
	struct lws_context *context;
	const char *p;
	int n = 0, logs = LLL_USER | LLL_ERR | LLL_WARN | LLL_NOTICE;

	lws_set_log_level(logs, NULL);
	//lwsl_user("LWS minimal http server POST | visit http://localhost:7681\n");

	memset(&info, 0, sizeof info); /* otherwise uninitialized garbage */
	info.port = 4004;
	info.protocols = protocols;
	info.mounts = &mount;
	info.options = 	LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE;

	context = lws_create_context(&info);
	if (!context) {
		lwsl_err("lws init failed\n");
		return NULL;
	}

	while (n >= 0)
		n = lws_service(context, 0);

	lws_context_destroy(context);

	//return 0;
}


/***************************************************/
/**** websocket server ***/
/***************************************************/
struct msg {
	void *payload; /* is malloc'd */
	size_t len;
};

/*
 * One of these is created for each client connecting to us.
 *
 * It is ONLY read or written from the lws service thread context.
 */

struct per_session_data__minimal {
	struct per_session_data__minimal *pss_list;
	struct lws *wsi;
	uint32_t tail;
};

/* one of these is created for each vhost our protocol is used with */

struct per_vhost_data__minimal {
	struct lws_context *context;
	struct lws_vhost *vhost;
	const struct lws_protocols *protocol;

	struct per_session_data__minimal *pss_list; /* linked-list of live pss*/
	pthread_t pthread_spam[1];

	pthread_mutex_t lock_ring; /* serialize access to the ring buffer */
	struct lws_ring *ring; /* {lock_ring} ringbuffer holding unsent content */

	const char *config;
	char finished;
};

/*
 * This runs under both lws service and "spam threads" contexts.
 * Access is serialized by vhd->lock_ring.
 */

static void
__minimal_destroy_message(void *_msg)
{
	struct msg *msg = (struct msg *)_msg;

	free(msg->payload);
	msg->payload = NULL;
	msg->len = 0;
}
/* this runs under the lws service thread context only */

static int
callback_minimal(struct lws *wsi, enum lws_callback_reasons reason,
                 void *user, void *in, size_t len)
{
	struct per_session_data__minimal *pss =
	    (struct per_session_data__minimal *)user;
	struct per_vhost_data__minimal *vhd =
	    (struct per_vhost_data__minimal *)
	    lws_protocol_vh_priv_get(lws_get_vhost(wsi),
	                             lws_get_protocol(wsi));
	const struct lws_protocol_vhost_options *pvo;
	const struct msg *pmsg;
	void *retval;
	int n, m, r = 0;
	struct msg amsg;

	switch (reason) {
	case LWS_CALLBACK_PROTOCOL_INIT:
		/* create our per-vhost struct */
		vhd = (struct per_vhost_data__minimal *)lws_protocol_vh_priv_zalloc(lws_get_vhost(wsi),
		        lws_get_protocol(wsi),
		        sizeof(struct per_vhost_data__minimal));
		if (!vhd)
			return 1;

		pthread_mutex_init(&vhd->lock_ring, NULL);

		/* recover the pointer to the globals struct */
		pvo = lws_pvo_search(
		          (const struct lws_protocol_vhost_options *)in,
		          "config");
		if (!pvo || !pvo->value) {
			lwsl_err("%s: Can't find \"config\" pvo\n", __func__);
			return 1;
		}
		vhd->config = pvo->value;

		vhd->context = lws_get_context(wsi);
		vhd->protocol = lws_get_protocol(wsi);
		vhd->vhost = lws_get_vhost(wsi);

		vhd->ring = lws_ring_create(sizeof(struct msg), 8,
		                            __minimal_destroy_message);
		if (!vhd->ring) {
			lwsl_err("%s: failed to create ring\n", __func__);
			return 1;
		}

		/* start the content-creating threads */

		for (n = 0; n < (int)LWS_ARRAY_SIZE(vhd->pthread_spam); n++)
			if (pthread_create(&vhd->pthread_spam[n], NULL,
			                   ws_hanle_realtime_data, vhd)) {
				lwsl_err("thread creation failed\n");
				r = 1;
				goto init_fail;
			}
		break;

	case LWS_CALLBACK_PROTOCOL_DESTROY:
init_fail:
		vhd->finished = 1;
		for (n = 0; n < (int)LWS_ARRAY_SIZE(vhd->pthread_spam); n++)
			if (vhd->pthread_spam[n])
				pthread_join(vhd->pthread_spam[n], &retval);

		if (vhd->ring)
			lws_ring_destroy(vhd->ring);

		pthread_mutex_destroy(&vhd->lock_ring);
		break;

	case LWS_CALLBACK_ESTABLISHED:
		/* add ourselves to the list of live pss held in the vhd */
		lws_ll_fwd_insert(pss, pss_list, vhd->pss_list);
		pss->tail = lws_ring_get_oldest_tail(vhd->ring);
		pss->wsi = wsi;
		break;

	case LWS_CALLBACK_CLOSED:
		/* remove our closing pss from the list of live pss */
		lws_ll_fwd_remove(struct per_session_data__minimal, pss_list,
		                  pss, vhd->pss_list);
		break;

	case LWS_CALLBACK_SERVER_WRITEABLE:
		pthread_mutex_lock(&vhd->lock_ring); /* --------- ring lock { */

		pmsg = (const struct msg *)lws_ring_get_element(vhd->ring, &pss->tail);
		if (!pmsg) {
			pthread_mutex_unlock(&vhd->lock_ring); /* } ring lock ------- */
			break;
		}

		/* notice we allowed for LWS_PRE in the payload already */
		m = lws_write(wsi, ((unsigned char *)pmsg->payload) + LWS_PRE,
		              pmsg->len, LWS_WRITE_TEXT);
		if (m < (int)pmsg->len) {
			pthread_mutex_unlock(&vhd->lock_ring); /* } ring lock ------- */
			lwsl_err("ERROR %d writing to ws socket\n", m);
			return -1;
		}

		lws_ring_consume_and_update_oldest_tail(
		    vhd->ring,	/* lws_ring object */
		    struct per_session_data__minimal, /* type of objects with tails */
		    &pss->tail,	/* tail of guy doing the consuming */
		    1,		/* number of payload objects being consumed */
		    vhd->pss_list,	/* head of list of objects with tails */
		    tail,		/* member name of tail in objects with tails */
		    pss_list	/* member name of next object in objects with tails */
		);

		/* more to do? */
		if (lws_ring_get_element(vhd->ring, &pss->tail))
			/* come back as soon as we can write more */
			lws_callback_on_writable(pss->wsi);

		pthread_mutex_unlock(&vhd->lock_ring); /* } ring lock ------- */
		break;

	case LWS_CALLBACK_RECEIVE:
	{
		//int msg_len = 0;
		bool status = false;
		char send_data[100] = {0};

		if (len > 0 && in != NULL) {
			if (strstr((char *)in, "Command") ) {
				if (strstr((char *)in, "play")) {
					status = true;
					g_member.cmd_play = true;
				}

				if (strstr((char *)in, "stop")) {
					status = true;
					g_member.cmd_play = false;
				}
			}
		}

		if (status) {
			snprintf(send_data, 100, "{\"WSType\":\"control\", \"Status\":\"success\"}");

		} else {
			snprintf(send_data, 100, "{\"WSType\":\"control\", \"Status\":\"fail\"}" );
		}

		ws_send_msg((void **)&vhd, send_data, strlen(send_data) + 1);
	}
	break;

	case LWS_CALLBACK_EVENT_WAIT_CANCELLED:
		if (!vhd)
			break;

		lws_start_foreach_llp(struct per_session_data__minimal **,
		                      ppss, vhd->pss_list) {
			lws_callback_on_writable((*ppss)->wsi);
		} lws_end_foreach_llp(ppss, pss_list);
		break;

	default:
		break;
	}

	return r;
}

#define LWS_PLUGIN_PROTOCOL_MINIMAL \
	{ \
		"realtime", \
		callback_minimal, \
		sizeof(struct per_session_data__minimal), \
		128, \
		0, NULL, 0 \
	}



static struct lws_protocols protocols_ws[] = {
	{ "http", lws_callback_http_dummy, 0, 0 },
	LWS_PLUGIN_PROTOCOL_MINIMAL,
	{ NULL, NULL, 0, 0 } /* terminator */
};

static int interrupted;

static const struct lws_http_mount mount_ws = {
	/* .mount_next */		NULL,		/* linked-list "next" */
	/* .mountpoint */		"/hurysradar/v1/realtimev2x",		/* mountpoint URL */
	/* .origin */			"./mount-origin", /* serve from dir */
	/* .def */			"index.html",	/* default filename */
	/* .protocol */			NULL,
	/* .cgienv */			NULL,
	/* .extra_mimetypes */		NULL,
	/* .interpret */		NULL,
	/* .cgi_timeout */		0,
	/* .cache_max_age */		0,
	/* .auth_mask */		0,
	/* .cache_reusable */		0,
	/* .cache_revalidate */		0,
	/* .cache_intermediaries */	0,
	/* .origin_protocol */		LWSMPRO_FILE,	/* files in a dir */
	/* .mountpoint_len */		26,		/* char count */
	/* .basic_auth_login_file */	NULL,
};

/*
 * This demonstrates how to pass a pointer into a specific protocol handler
 * running on a specific vhost.  In this case, it's our default vhost and
 * we pass the pvo named "config" with the value a const char * "myconfig".
 *
 * This is the preferred way to pass configuration into a specific vhost +
 * protocol instance.
 */

static const struct lws_protocol_vhost_options pvo_ops = {
	NULL,
	NULL,
	"config",		/* pvo name */
	(char *)"myconfig"	/* pvo value */
};

static const struct lws_protocol_vhost_options pvo = {
	NULL,		/* "next" pvo linked-list */
	&pvo_ops,	/* "child" pvo linked-list */
	"realtime",	/* protocol name we belong to on this vhost */
	""		/* ignored */
};

void *websocket_server(void *data)
{
	struct lws_context_creation_info info;
	struct lws_context *context;
	const char *p;
	int logs = LLL_USER | LLL_ERR | LLL_WARN | LLL_NOTICE;

	lws_set_log_level(logs, NULL);
	//lwsl_user("LWS minimal ws server + threads | visit http://localhost:7681\n");

	memset(&info, 0, sizeof info); /* otherwise uninitialized garbage */
	info.port = 4005;
	info.mounts = &mount_ws;
	info.protocols = protocols_ws;
	info.pvo = &pvo; /* per-vhost options */
	info.options =
	    LWS_SERVER_OPTION_HTTP_HEADERS_SECURITY_BEST_PRACTICES_ENFORCE;

	context = lws_create_context(&info);
	if (!context) {
		lwsl_err("lws init failed\n");
		return NULL;
	}

	/* start the threads that create content */
	while (1) {
		if (lws_service(context, 0) )
			break;
	}

	lws_context_destroy(context);
}

//高低地址转换
void buffer_data_reverse(char *buf, int len)
{
	unsigned char val = 0;
	for (int i = 0; i < len / 2; i++) {
		val = buf[i];
		buf[i] = buf[len - i - 1];
		buf[len - i - 1] = val;
	}
}

void create_queue()
{
	if (g_ivddevsets.pro_type == PROTO_WS_VIDEO || g_ivddevsets.pro_type == PROTO_WS_RADAR_VIDEO) {
		g_ws_queue.video_queue = InitQueue(); //realtime data queue
	}

	if (g_ivddevsets.pro_type == PROTO_WS_RADAR || g_ivddevsets.pro_type == PROTO_WS_RADAR_VIDEO) {
		g_ws_queue.radar_queue = InitQueue();// radar data queue
	}
}

#if 0

#define RADAR_TAIL_LEN 10
void *radar_client(void *data)
{
	//char send_data[1] = {0xFF};
	int n_data_len = 0;
	int n_left = 0;
	int n_read = 0;
	char *p_buf = NULL;
	bool con_flag = false;
	char buffer[BUFFER_SIZE];


	// 设置一个socket地址结构server_addr,代表服务器的internet地址和端口
	struct sockaddr_in  server_addr;
	bzero(&server_addr, sizeof(server_addr));
	server_addr.sin_family = AF_INET;

	//int client_socket = creat_tcp_socket(30, 30);
	g_member.radar_sock = creat_tcp_socket(READ_TIMEOUT, WRITE_TIMEOUT);

	// 服务器的IP地址来自程序的参数
	if (inet_aton(RADAR_IP, &server_addr.sin_addr) == 0)
	{
		printf("Server IP Address Error!\n");
		exit(1);
	}

	server_addr.sin_port = htons(RADAR_SERVER_PORT);
	socklen_t server_addr_length = sizeof(server_addr);

	do {
		if (!con_flag )
		{
			if (g_member.radar_sock < 1) {
				g_member.radar_sock = creat_tcp_socket(READ_TIMEOUT, WRITE_TIMEOUT);
			}
			if (connect(g_member.radar_sock, (struct sockaddr*)&server_addr, server_addr_length) < 0) {
				printf("Can Not Connect To %d!\n", g_member.radar_sock);
				sleep(2);
				//close(client_socket);
				continue;
			} else {
				con_flag = true;
			}
		}

		if (!g_member.cmd_play) {
			sleep(2);
			continue;
		}

#if 0
		if ( send(g_member.radar_sock, send_data, 1, 0) < 0 ) { //发送0xFF获取雷达数据
			close(g_member.radar_sock);
			g_member.radar_sock = 0;
			con_flag = false;
		}
#endif

		n_left = BUFFER_SIZE;

		bzero(buffer, sizeof(buffer));
		p_buf = buffer;

		while ( n_left > 0 )   {
			n_read = read(g_member.radar_sock, p_buf, n_left); //接收雷达数据

			if ( n_read < 0 ) {
				if (errno == EINTR)
					continue;
				if (errno == EAGAIN || errno == EWOULDBLOCK) //超时
					break;

				close(g_member.radar_sock);
				g_member.radar_sock = 0;
				con_flag = false;  //socket is fail
				break;
			}
			else if ( n_read == 0) { //connection close
				close(g_member.radar_sock);
				g_member.radar_sock = 0;
				con_flag = false;  //socket is fail
				break;
			}

			p_buf += n_read;
			n_left = BUFFER_SIZE - n_read;
		}


		//handle data
		n_data_len = BUFFER_SIZE - n_left;
		//printf(".........receive data len: %d \n", n_data_len);
		if (g_member.cmd_play && n_data_len > 0 && n_data_len > 47 && (n_data_len % 11) == 1 ) {

			if ((unsigned char)buffer[0] != 0x00CA ||  (unsigned char)buffer[1] != 0x00CB || (unsigned char)buffer[2] != 0x00CC || (unsigned char)buffer[3] != 0x00CD ) //head
				continue;

			if ((unsigned char)buffer[n_data_len - RADAR_TAIL_LEN] != 0xEA ||  (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 1] != 0xEB || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 2] != 0xEC || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 3] != 0xED || \
			        (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 4] != 0x61 || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 5] != 0x67 || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 6] != 0x5C || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 7] != 0x14 || \
			        (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 8] != 0x89 || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 9] != 0xC6 ) //tail
				continue;

			add_radar_data_2_queue(&buffer[4], n_data_len - RADAR_TAIL_LEN - 4);
		}


	} while (1);

	close(g_member.radar_sock);
	return 0;


}

#endif


#define RADAR_TAIL_LEN 10
void *radar_client(void *data)
{
	//char send_data[1] = {0xFF};
	int n_data_len = 0;
	int n_left = 0;
	int n_read = 0;
	char *p_buf = NULL;

	char buffer[BUFFER_SIZE];

	// 设置一个socket地址结构server_addr,代表服务器的internet地址和端口
	struct sockaddr_in  client_addr;
	socklen_t addr_len;
	addr_len = sizeof(sockaddr_in);

	g_member.radar_sock = UdpCreateSocket(RADAR_SERVER_PORT);

	do {

		bzero(buffer, sizeof(buffer));
		p_buf = buffer;

		//while ( n_left > 0 )   {
		n_read = recvfrom(g_member.radar_sock, p_buf, BUFFER_SIZE, 0, (struct sockaddr*)&client_addr, &addr_len); //接收雷达数据
		//num =recvfrom(sockfd,buf,MAXDATASIZE, 0, (struct sockaddr*)&client_addr, &addr_len);

		if ( n_read < 0 ) {
			continue;
		}
		//}

		if (g_ivddevsets.pro_type != PROTO_WS_RADAR && g_ivddevsets.pro_type != PROTO_WS_RADAR_VIDEO)
			continue;

		//handle data
		n_data_len = n_read;
		//printf(".........receive data len: %d \n", n_data_len);
		if (n_data_len > 0 && n_data_len > 47 && (n_data_len % 11) == 1 ) {

			if ((unsigned char)buffer[0] != 0x00CA ||  (unsigned char)buffer[1] != 0x00CB || (unsigned char)buffer[2] != 0x00CC || (unsigned char)buffer[3] != 0x00CD ) //head
				continue;

			if ((unsigned char)buffer[n_data_len - RADAR_TAIL_LEN] != 0xEA ||  (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 1] != 0xEB || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 2] != 0xEC || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 3] != 0xED || \
			        (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 4] != 0x61 || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 5] != 0x67 || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 6] != 0x5C || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 7] != 0x14 || \
			        (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 8] != 0x89 || (unsigned char)buffer[n_data_len - RADAR_TAIL_LEN + 9] != 0xC6 ) //tail
				continue;

			add_radar_data_2_queue(&buffer[4], n_data_len - RADAR_TAIL_LEN - 4);
		}


	} while (1);

	close(g_member.radar_sock);
	return 0;


}


void add_video_data_2_queue(int index, char *buf, int len)
{
	if ( len < 1)
		return;

	//char *p_buf = (char *)malloc(sizeof(int) + len);
	//*((int *)p_buf) = index;
	//memcpy(p_buf + sizeof(int), buf, len);
	WS_Node_Data_t *p_buf = (WS_Node_Data_t *)malloc(sizeof(WS_Node_Data_t));
	p_buf->cam_index = index;
	get_date_ms(p_buf->data_ms);
	memcpy((char *)&p_buf->out_buf, buf, len);

	PNode p_node = FixEnQueue(g_ws_queue.video_queue, (char*)p_buf, sizeof(WS_Node_Data_t), NULL, 50);
	if (p_node) {
		if (p_node->p_value)
			free(p_node->p_value);
		free(p_node);
	}
}

//雷达数据插链表
void add_radar_data_2_queue(char *buf, int len)
{
	if ( len < 1)
		return;

	//char *p_buf = (char *)malloc(len);
	//if (!p_buf) {
	//	return;
	//}

	//memcpy(p_buf, buf, len);

	//EnQueue(g_ws_queue.radar_queue, p_buf, len, NULL);

	short array_num = (len - 53) / 11;

	if (array_num < 1)
		return;

	short array_sz = array_num * sizeof(mRadarRTObj);
	mRadarRTObj *p_rt_objs = (mRadarRTObj *)malloc(array_sz);
	mRadarRTObj *p_start = p_rt_objs;

	for (int m = 0; m < array_num; m++) {
		buffer_data_reverse(buf + m * 11 + 3, 8); //头3个字节不转
		mRadarObj *obj = (mRadarObj *)(buf + m * 11);
		p_rt_objs->x_Point = (obj->x_Point - 4096) * 0.128;
		p_rt_objs->y_Point = (obj->y_Point - 4096) * 0.128;
		p_rt_objs->Speed_x = (obj->Speed_x - 1024) * 0.36;
		p_rt_objs->Speed_y = (obj->Speed_y - 1024) * 0.36;
		p_rt_objs->Obj_Len =  obj->Object_Length * 0.2;
		p_rt_objs++;
	}

	char num[2] = {0};
	unsigned int radar_id = 0;  //雷达编码
	for (int j = 0; j < 20; j++) {
		num[0] = buf[len - 20 + j];
		if (!num[0])
			break;

		radar_id *= 10;
		radar_id += atoi(num);
	}

	//for(int i = 0, n = 0; i < CAM_MAX && n < g_cam_num.cam_num; i++) {

	if (radar_id < 1 || radar_id > CAM_MAX || !g_cam_num.exist[radar_id - 1])
		return;


	mRadarRTObj *p_tmp_objs = (mRadarRTObj *)malloc(array_sz);
	memcpy(p_tmp_objs, p_start, array_sz);
	PNode tmp_node = FixEnQueue(cam_info[radar_id - 1].p_radar, p_tmp_objs, array_sz, NULL, 3); //入算法队列

	if (tmp_node) {
		if (tmp_node->p_value)
			free(tmp_node->p_value);
		free(tmp_node);
	}
	//}
	if ( g_member.cmd_play && g_ivddevsets.pro_type == PROTO_WS_RADAR) {
		tmp_node = FixEnQueue(g_ws_queue.radar_queue, (char *)p_start, array_sz, NULL, 50);//入雷达数据发送队列
		if (tmp_node) {
			if (tmp_node->p_value)
				free(tmp_node->p_value);
			free(tmp_node);
		}
	}
}


float ab_angle(float a, float b, float c) //b夹角
{
	float ret = 0.0;
	float cb = 0.0;
	if (a > 0.0000001 && c > 0.0000001)
		cb == (a * a - b * b + c * c) / (2 * c * a);
	cb = cb > 1.0 ? 1.0 : cb < -1.0 ? -1.0 : cb;
	ret = acos(cb) * 180 / 3.1415926;

	return ret;
}


float radar_obj_angle(float x, float y)
{	float ret = 0.0;
	float z_t = 0.0;
	if (x < 0.0000001 || y < 0.0000001) {//right is positive  left is negative
		float x_t = fabs(x);
		float y_t = fabs(y);
		z_t = sqrt(x_t * x_t + y_t * y_t);
		ret = ab_angle(x_t, y_t, z_t);
		ret = 90.0 - ret;
	} else {
		z_t = sqrt(x * x + y * y);
		ret = ab_angle(x, y, z_t);
		ret += 270.0;
	}

	return ret;
}

#define PI 3.14159265359
#define CONVERT(x) (x*180/PI)
const double long_radius = 6378137;
const double short_radius = 6356752.3142;
const double flat_rate = 1 / 298.2572236;

///返回度
double radar_obj_angle_ex(double x, double y) //deg 
{
	double ret = 0.00;

//    ret = atan2(fabs(y),fabs(x) )*(180/PI); //2020-02-24 by roger modify
#if 1
	if (y > 0.0000001) {

		if (x < 0.0000001) {
			ret = atan(fabs(x) / y)*(180/PI);
		} else {
			ret = atan(x / y);
			ret = 360.0 - ret*(180/PI);
		}
	}
#endif
	return ret;
}

double deg(double x)
{
	return x * 180.0 / PI;
}

double rad(double d)
{
	return d * PI / 180.0;
}

void computerThatLonLat(double lon, double lat, double brng, double dist, double *lon_lat)
{

	double alpha1 = rad(brng);
	double sinAlpha1 = sin(alpha1);
	double cosAlpha1 = cos(alpha1);

	double tanU1 = (1 - flat_rate) * tan(rad(lat));
	double cosU1 = 1 / sqrt((1 + tanU1 * tanU1));
	double sinU1 = tanU1 * cosU1;
	double sigma1 = atan2(tanU1, cosAlpha1);
	double sinAlpha = cosU1 * sinAlpha1;
	double cosSqAlpha = 1 - sinAlpha * sinAlpha;
	double uSq = cosSqAlpha * (long_radius * long_radius - short_radius * short_radius) / (short_radius * short_radius);
	double A = 1 + uSq / 16384 * (4096 + uSq * (-768 + uSq * (320 - 175 * uSq)));
	double B = uSq / 1024 * (256 + uSq * (-128 + uSq * (74 - 47 * uSq)));

	double cos2SigmaM = 0;
	double sinSigma = 0;
	double cosSigma = 0;
	double sigma = dist / (short_radius * A), sigmaP = 2 * PI;
	while (abs(sigma - sigmaP) > 1e-12) {
		cos2SigmaM = cos(2 * sigma1 + sigma);
		sinSigma = sin(sigma);
		cosSigma = cos(sigma);
		double deltaSigma = B * sinSigma * (cos2SigmaM + B / 4 * (cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)
		                                    - B / 6 * cos2SigmaM * (-3 + 4 * sinSigma * sinSigma) * (-3 + 4 * cos2SigmaM * cos2SigmaM)));
		sigmaP = sigma;
		sigma = dist / (short_radius * A) + deltaSigma;
	}

	double tmp = sinU1 * sinSigma - cosU1 * cosSigma * cosAlpha1;
	double lat2 = atan2(sinU1 * cosSigma + cosU1 * sinSigma * cosAlpha1,
	                    (1 - flat_rate) * sqrt(sinAlpha * sinAlpha + tmp * tmp));
	double lambda = atan2(sinSigma * sinAlpha1, cosU1 * cosSigma - sinU1 * sinSigma * cosAlpha1);
	double C = flat_rate / 16 * cosSqAlpha * (4 + flat_rate * (4 - 3 * cosSqAlpha));
	double L = lambda - (1 - C) * flat_rate * sinAlpha
	           * (sigma + C * sinSigma * (cos2SigmaM + C * cosSigma * (-1 + 2 * cos2SigmaM * cos2SigmaM)));

	double revAz = atan2(sinAlpha, -tmp); // final bearing

	//System.out.println(revAz);
	//System.out.println(lon + deg(L) + "," + deg(lat2));
	lon_lat[0] = lon + deg(L);
	lon_lat[1] = deg(lat2);
}

//a,b 经纬度求航向角
double get_angle1(double lat_a, double lng_a, double lat_b, double lng_b)
{
	double y = sin(lng_b - lng_a) * cos(lat_b);
	double x = cos(lat_a) * sin(lat_b) - sin(lat_a) * cos(lat_b) * cos(lng_b - lng_a);
	double bearing = atan2(y, x);
	//bearing = toDegrees(bearing);
	bearing = deg(bearing);

	if (bearing < 0) {
		bearing = bearing + 360;
	}

	return bearing;
}

//a,b 坐标求夹角
double get_2pos_angle(double x1, double y1, double x2, double y2)
{
    double angle = atan2(fabs(y2-y1), fabs(x2-x1) );
    //double angle = atan2(y2-y1, x2-x1);
	angle = deg(angle);

	return angle;
}

#define SEND_BUFF_MAX 8192
#define TMP_BUF_MAX 250
//实时数据处理及发送
void *ws_hanle_realtime_data(void *d)
{
	double lon_lat[2] = {0.0};

	short total_len = 0;
	short buf_len = 0;
	struct per_vhost_data__minimal *vhd = (struct per_vhost_data__minimal *)d;

	//time_t t = NULL;
	char buf[SEND_BUFF_MAX] = {0};
	char tmpBuf[TMP_BUF_MAX] = {0};
	mPositionData info = {0};

	do {

		if (!g_member.cmd_play) {
			sleep(1);
			continue;
		}
		
		memset(buf, 0, SEND_BUFF_MAX);
		//t = time(0);
		//strftime(tmpBuf, 50, "%Y-%m-%d %H:%M:%S.000", localtime(&t));


		if (g_ivddevsets.pro_type == PROTO_WS_VIDEO || g_ivddevsets.pro_type == PROTO_WS_RADAR_VIDEO) { //视频实时数据
			PNode node = DeQueue(g_ws_queue.video_queue);
			if (node) {
				total_len = 0;
				buf_len = 0;

				//int cam_index = *((int *)node->p_value);
				//OUTBUF *p_outbuf = (OUTBUF *)(node->p_value + sizeof(int));
				WS_Node_Data_t *p_node_data = (WS_Node_Data_t*)node->p_value;
				int cam_index = p_node_data->cam_index;
				OUTBUF *p_outbuf = (OUTBUF*)&p_node_data->out_buf;
				ReadPosition(&info, cam_index);

				if (p_outbuf->udetNum > 0) {
					memset(buf, 0, SEND_BUFF_MAX);
					snprintf(buf, 128, "{\"WSType\":\"data\",\"DeviceNo\":\"%s\",\"Timestamp\":\"%s\",\"DetectorType\":1,\"TargetNum\":%d,\"TargetArray\":[ ",   \
					         g_ivddevsets.devUserNo, p_node_data->data_ms, p_outbuf->udetNum);

					total_len = strlen(buf);

					for (int i = 0; i < p_outbuf->udetNum; i++) {
                        int sel_mm = -1;
                        bool is_new = true;
						double yaw_angle = 0.0;
						double obj_distan = 0.0;
						memset(tmpBuf, 0, TMP_BUF_MAX);

                        if (cam_info[cam_index].tg_num > 0) {
                            int cnt = 0; //满200之后作搜索
                            if (cam_info[cam_index].is_full)
                                cnt = 200;
                            else 
                                cnt = cam_info[cam_index].tg_num;
                            
                            for(int mm = 0; mm < cnt && mm < 200; mm++) {
                                if ( cam_info[cam_index].vec_x_y[mm].id == p_outbuf->udetBox[i].id) {
                                    sel_mm = mm;
                                    is_new = false;
                                    break;
                                }
                            }
                        }

                        if (-1 == sel_mm) {
                           int tg_num = cam_info[cam_index].tg_num;
                           if (tg_num < 200) {
                                cam_info[cam_index].vec_x_y[tg_num].id = p_outbuf->udetBox[i].id;
                                cam_info[cam_index].vec_x_y[tg_num].x = p_outbuf->udetBox[i].distance[0];
                                cam_info[cam_index].vec_x_y[tg_num].y = p_outbuf->udetBox[i].distance[1];
                                sel_mm = tg_num;
                                cam_info[cam_index].tg_num++;
                                
                           }else {
                                cam_info[cam_index].is_full = true;
                                cam_info[cam_index].vec_x_y[0].id = p_outbuf->udetBox[i].id;
                                cam_info[cam_index].vec_x_y[0].x = p_outbuf->udetBox[i].distance[0];
                                cam_info[cam_index].vec_x_y[0].y = p_outbuf->udetBox[i].distance[1];
                                cam_info[cam_index].tg_num = 1;
                                sel_mm = 0;
                          }
                  
                        }

						yaw_angle = radar_obj_angle_ex(p_outbuf->udetBox[i].distance[0], p_outbuf->udetBox[i].distance[1]);
						obj_distan = sqrt(p_outbuf->udetBox[i].distance[0] * p_outbuf->udetBox[i].distance[0] + (p_outbuf->udetBox[i].distance[1]+1.5) * (p_outbuf->udetBox[i].distance[1]+1.5) );
 #if 0                      
					   //2020-02-24 by roger modify
                        switch(g_camdetect[cam_index].other.camdirection) {
                            case 1: //北
                             {   
                                if (p_outbuf->udetBox[i].distance[0] > 0) {
                                    if (p_outbuf->udetBox[i].distance[1] > 0)
                                        yaw_angle += 180.0; 
                                    else if (p_outbuf->udetBox[i].distance[1] < 0)
                                        yaw_angle = 180.0 - yaw_angle;
                                    else 
                                        yaw_angle = 180.0;
                                }else if (p_outbuf->udetBox[i].distance[0] < 0) {
                                    if (p_outbuf->udetBox[i].distance[1] > 0)
                                        yaw_angle = 360.0 - yaw_angle;
                                    else if (p_outbuf->udetBox[i].distance[1] == 0)
                                        yaw_angle = 0.0;
                              
                                }else {
                                    if (p_outbuf->udetBox[i].distance[1] > 0) 
                                        yaw_angle = 270.0;
                                    else 
                                        yaw_angle = 90.0;
                                }   
                            }
                                break;
                            case 16://南
                            {   if (p_outbuf->udetBox[i].distance[0] > 0) {
                                    if (p_outbuf->udetBox[i].distance[1] < 0)
                                        yaw_angle = 360.0 - yaw_angle; 
                                    else if (p_outbuf->udetBox[i].distance[1] == 0)
                                        yaw_angle = 0.0;
                                }else if (p_outbuf->udetBox[i].distance[0] < 0) {
                                    if (p_outbuf->udetBox[i].distance[1] > 0)
                                        yaw_angle = 180.0 - yaw_angle;
                                    else if (p_outbuf->udetBox[i].distance[1] < 0)
                                        yaw_angle += 180.0;
                                    else
                                        yaw_angle = 180.0;
                                }else {
                                    if (p_outbuf->udetBox[i].distance[1] > 0) 
                                        yaw_angle = 90.0;
                                    else 
                                        yaw_angle = 270.0;
                                }   
                            }
                                break;
                            case 4://东
                                if (p_outbuf->udetBox[i].distance[0] > 0) {
                                    if (p_outbuf->udetBox[i].distance[1] > 0)
                                        yaw_angle += 90.0; 
                                    else if (p_outbuf->udetBox[i].distance[1] < 0)
                                        yaw_angle = 90.0 - yaw_angle;
                                    else 
                                        yaw_angle = 90.0;
                                }else if (p_outbuf->udetBox[i].distance[0] < 0) {
                                    if (p_outbuf->udetBox[i].distance[1] > 0)
                                        yaw_angle = 270.0 - yaw_angle;
                                    else if (p_outbuf->udetBox[i].distance[1]< 0)
                                        yaw_angle += 270.0;
                                    else
                                        yaw_angle = 270.0;
                              
                                }else {
                                    if (p_outbuf->udetBox[i].distance[1] > 0) 
                                        yaw_angle = 180.0;
                                    else 
                                        yaw_angle = 0.0;
                                }   
                                break;
                            case 64://西
                                if (p_outbuf->udetBox[i].distance[0] > 0) {
                                    if (p_outbuf->udetBox[i].distance[1] > 0)
                                        yaw_angle += 270.0; 
                                    else if (p_outbuf->udetBox[i].distance[1] < 0)
                                        yaw_angle = 270.0 - yaw_angle;
                                    else 
                                        yaw_angle = 270.0;
                                }else if (p_outbuf->udetBox[i].distance[0] < 0) {
                                    if (p_outbuf->udetBox[i].distance[1] > 0)
                                        yaw_angle = 90.0 - yaw_angle;
                                    else if (p_outbuf->udetBox[i].distance[1]< 0)
                                        yaw_angle += 90.0;
                                    else
                                        yaw_angle = 90.0;
                              
                                }else {
                                    if (p_outbuf->udetBox[i].distance[1] > 0) 
                                        yaw_angle = 0.0;
                                    else 
                                        yaw_angle = 180.0;
                                }  
                               break;
                  
				       }
#endif
#if 1
                       switch(g_camdetect[cam_index].other.camdirection) {
                            case 1: //北
                                yaw_angle += 180.0;
                                break;
                            case 16://南
                                break;
                            case 4://东
                                yaw_angle += 90.0;
                                break;
                            case 64://西
                                yaw_angle += 270.0;
                               break;
                  
				       }
#endif
						computerThatLonLat(info.Longitude, info.Latitude, yaw_angle, obj_distan, lon_lat);

						//yaw_angle = get_angle1(info.Latitude, info.Longitude, lon_lat[1], lon_lat[0]);
						yaw_angle =0.000000;
                        
						if (!is_new) {
                            double pt_x_sq = (p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x) *(p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x);
                            double pt_y_sq = (p_outbuf->udetBox[i].distance[1] - cam_info[cam_index].vec_x_y[sel_mm].y) *(p_outbuf->udetBox[i].distance[1] - cam_info[cam_index].vec_x_y[sel_mm].y);
                            double pt_dist = sqrt(pt_x_sq + pt_y_sq);
                            if (pt_dist > 1.00000) {
                                yaw_angle = get_2pos_angle(cam_info[cam_index].vec_x_y[sel_mm].x, cam_info[cam_index].vec_x_y[sel_mm].y,p_outbuf->udetBox[i].distance[0],p_outbuf->udetBox[i].distance[1]);
                         
                                switch(g_camdetect[cam_index].other.camdirection) {
                                    case 1: //北
                                    {
                                        if ( p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x < 0 )
                                           yaw_angle = 180 - yaw_angle;
                                        else if(p_outbuf->udetBox[i].distance[0] == cam_info[cam_index].vec_x_y[sel_mm].x) 
                                            yaw_angle = 90;
                                    }
                                 
                                    break;
                                    case 16://南
                                    {
                                        #if 0
                                        if ( p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x > 0 )
                                           yaw_angle = 360 - yaw_angle;
                                        else if(p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x < 0)
                                            yaw_angle = 180 + yaw_angle;
                                        else
                                            yaw_angle = 270;
                                        #endif

                                        if (p_outbuf->udetBox[i].distance[0] > cam_info[cam_index].vec_x_y[sel_mm].x ) {
                                            if (p_outbuf->udetBox[i].distance[1] < cam_info[cam_index].vec_x_y[sel_mm].y) //东南
                                                yaw_angle = 270.0 + (90.0 - yaw_angle);
                                            else if (p_outbuf->udetBox[i].distance[1] == cam_info[cam_index].vec_x_y[sel_mm].y)
                                                yaw_angle = 0.0;
                                        }else if (p_outbuf->udetBox[i].distance[0] < cam_info[cam_index].vec_x_y[sel_mm].x) {
                                            if (p_outbuf->udetBox[i].distance[1] > cam_info[cam_index].vec_x_y[sel_mm].y)
                                                yaw_angle = 90.0 + 90.0 - yaw_angle;
                                            else if (p_outbuf->udetBox[i].distance[1] == cam_info[cam_index].vec_x_y[sel_mm].y)
                                                yaw_angle = 180.0;
                                            else
                                                yaw_angle += 180.0;
                                        }else {
                                        
                                           if (p_outbuf->udetBox[i].distance[1] > cam_info[cam_index].vec_x_y[sel_mm].y) 
                                                yaw_angle = 90.0;
                                           else
                                             yaw_angle = 270.0;
                                        }

                                    }
                                        break;
                                    
                                    case 4://东
                                    {
                                        if ( p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x > 0 )
                                           yaw_angle = 270 + yaw_angle;
                                        else if(p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x < 0)
                                            yaw_angle = 90 - yaw_angle;
                                        else
                                            yaw_angle = 0;
                                    }
                                        break;
                                    case 64://西
                                    {
                                        if ( p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x > 0 )
                                           yaw_angle = 90 + yaw_angle;
                                        else if ( p_outbuf->udetBox[i].distance[0] - cam_info[cam_index].vec_x_y[sel_mm].x < 0 )
                                            yaw_angle = 270 - yaw_angle;
                                        else
                                            yaw_angle = 180;
                                    }
                                       break;
            				    }
                                prt(info, "*******id:%d--%d[box--X: %d Y: %d org--X: %d Y: %d] angle: %f ",p_outbuf->udetBox[i].id, cam_info[cam_index].vec_x_y[sel_mm].id, p_outbuf->udetBox[i].distance[0], p_outbuf->udetBox[i].distance[1],cam_info[cam_index].vec_x_y[sel_mm].x,cam_info[cam_index].vec_x_y[sel_mm].y, yaw_angle);
                                cam_info[cam_index].vec_x_y[sel_mm].x = p_outbuf->udetBox[i].distance[0];
                                cam_info[cam_index].vec_x_y[sel_mm].y = p_outbuf->udetBox[i].distance[1];
                                cam_info[cam_index].vec_x_y[sel_mm].yaw_angle = yaw_angle;
                            }else{
                                 yaw_angle = cam_info[cam_index].vec_x_y[sel_mm].yaw_angle;
                            }
                      
						}else {
                            yaw_angle = 0.00000000;
                  		}

#if 0
						snprintf(tmpBuf, TMP_BUF_MAX, "{\"ID\":%d,\"ParticipantType\":%d,\" XPos\":%.7f,\"YPos\":%.7f,\"Speed\":%.6f,\"Length\":%.6f,\"Width\":%.6f,\"Longitude\":%.7f,\"Latitude\":%.7f,\"Altitude\":%.7f,\"YawAngle\":%.7f},", \
						         i + 1, 3, 1.00 * p_outbuf->udetBox[i].distance[0], 1.00 * p_outbuf->udetBox[i].distance[1], 1.00 * p_outbuf->udetBox[i].speed, 1.00 * p_outbuf->udetBox[i].length \
					         , 1.00 * p_outbuf->udetBox[i].width, lon_lat[0], lon_lat[1], info.Altitude, yaw_angle);
#endif
                        snprintf(tmpBuf, TMP_BUF_MAX, "{\"ID\":%d,\"ParticipantType\":%d,\"XPos\":%.7f,\"YPos\":%.7f,\"Speed\":%.6f,\"Speed_VX\":%.6f,\"Length\":%.6f,\"Width\":%.6f,\"Longitude\":%.7f,\"Latitude\":%.7f,\"Altitude\":%.7f,\"YawAngle\":%.7f},", \
						         i + 1, 1, 1.00 * p_outbuf->udetBox[i].distance[0], 1.00 * p_outbuf->udetBox[i].distance[1], 1.00 * p_outbuf->udetBox[i].speed, 1.00 * p_outbuf->udetBox[i].speed_Vx,0.1 * p_outbuf->udetBox[i].length \
					         , 1.00 * p_outbuf->udetBox[i].width, lon_lat[0], lon_lat[1], info.Altitude, yaw_angle);
						buf_len = strlen(tmpBuf);
						strncpy(buf + total_len, tmpBuf, buf_len);
						//snprintf(buf+total_len, buf_len,"%s", tmpBuf);
						total_len += buf_len;
					}

					if (buf[total_len - 1] == ',')
						buf[total_len - 1] = ' ';

					strncpy(buf + total_len, "]} ", 3);
					total_len += 3;

					ws_send_msg((void **)&vhd, buf, total_len);
				}

				if (p_outbuf->udetPersonNum > 0) {
					total_len = 0;
					buf_len = 0;
					//strftime(tmpBuf, 50, "%Y-%m-%d %H:%M:%S.000", localtime(&t));
					memset(buf, 0, SEND_BUFF_MAX);
					snprintf(buf, 128, "{\"WSType\":\"data\",\"DeviceNo\":\"%s\",\"Timestamp\":\"%s\",\"DetectorType\":1,\"TargetNum\":%d,\"TargetArray\":[ ",   \
					         g_ivddevsets.devUserNo, p_node_data->data_ms, p_outbuf->udetPersonNum);

					total_len = strlen(buf);

					for (int i = 0; i < p_outbuf->udetPersonNum; i++) {
                //        int sel_mm = -1;
                //        bool is_new = false;
						double yaw_angle = 0.0;
						double obj_distan = 0.0;
						memset(tmpBuf, 0, TMP_BUF_MAX);
               #if 0
                        if (cam_info[cam_index].ps_tg_num > 0) {
                            int cnt = 0; //满200之后作搜索
                            if (cam_info[cam_index].ps_is_full)
                                cnt = 200;
                            else 
                                cnt = cam_info[cam_index].ps_tg_num;
                            
                            for(int mm = 0; mm < cnt && mm < 200; mm++) {
                                if ( cam_info[cam_index].ps_x_y[mm].id == p_outbuf->udetPersonBox[i].id) {
                                    sel_mm = mm;
                                    break;
                                }
                            }
                        }

                        if (-1 == sel_mm) {
                           int tg_num = cam_info[cam_index].tg_num;
                           if (tg_num < 200) {
                             
                                cam_info[cam_index].ps_x_y[tg_num].id = p_outbuf->udetPersonBox[i].id;
                                cam_info[cam_index].ps_x_y[tg_num].x = p_outbuf->udetPersonBox[i].distance[0];
                                cam_info[cam_index].ps_x_y[tg_num].y = p_outbuf->udetPersonBox[i].distance[1];
                                sel_mm = cam_info[cam_index].ps_tg_num;
                                cam_info[cam_index].ps_tg_num++;
                                
                           }else {
                                cam_info[cam_index].ps_is_full = true;
                                cam_info[cam_index].ps_x_y[0].id = p_outbuf->udetPersonBox[i].id;
                                cam_info[cam_index].ps_x_y[0].x = p_outbuf->udetPersonBox[i].distance[0];
                                cam_info[cam_index].ps_x_y[0].y = p_outbuf->udetPersonBox[i].distance[1];
                                cam_info[cam_index].ps_tg_num = 1;
                                sel_mm = 0;
                           }
                           is_new = true;
                        }
                #endif
						yaw_angle = radar_obj_angle_ex(p_outbuf->udetPersonBox[i].distance[0], p_outbuf->udetPersonBox[i].distance[1]);
						obj_distan = sqrt(p_outbuf->udetPersonBox[i].distance[0] * p_outbuf->udetPersonBox[i].distance[0] + p_outbuf->udetPersonBox[i].distance[1] * p_outbuf->udetPersonBox[i].distance[1] );
				        //yaw_angle += 270;
				        switch(g_camdetect[cam_index].other.camdirection) {
                            case 1: //北
                                yaw_angle += 180.0;
                                break;
                            case 16://南
                                break;
                            case 4://东
                                yaw_angle += 90.0;
                                break;
                            case 64://西
                                yaw_angle += 270.0;
                               break;
                  
				       }
						computerThatLonLat(info.Longitude, info.Latitude, yaw_angle, obj_distan, lon_lat);
                        #if 0
						if (!is_new) {
                            yaw_angle = get_2pos_angle(cam_info[cam_index].ps_x_y[sel_mm].x, cam_info[cam_index].ps_x_y[sel_mm].y,p_outbuf->udetPersonBox[i].distance[0],p_outbuf->udetPersonBox[i].distance[1]);
                            yaw_angle = 360 - yaw_angle;
						}else {
                            yaw_angle = 0.00000000;
						}
                        #endif;
                        yaw_angle = 0.00000000;
 #if 0                       
						snprintf(tmpBuf, TMP_BUF_MAX, "{\"ID\":%d,\"ParticipantType\":%d,\" XPos\":%.7f,\"YPos\":%.7f,\"Speed\":%.6f,\"Length\":%.6f,\"Width\":%.6f,\"Longitude\":%.7f,\"Latitude\":%.7f,\"Altitude\":%.7f,\"YawAngle\":%.7f},", \
						         i + 1, 3, 1.00 * p_outbuf->udetPersonBox[i].distance[0], 1.00 * p_outbuf->udetPersonBox[i].distance[1], 1.00 * p_outbuf->udetPersonBox[i].speed, 1.00 * p_outbuf->udetPersonBox[i].length \
						         , 1.00 * p_outbuf->udetPersonBox[i].width, lon_lat[0], lon_lat[1], info.Altitude, yaw_angle);
#endif
                        snprintf(tmpBuf, TMP_BUF_MAX, "{\"ID\":%d,\"ParticipantType\":%d,\"XPos\":%.7f,\"YPos\":%.7f,\"Speed\":%.6f,\"Speed_Vx\":%.6f,\"Length\":%.6f,\"Width\":%.6f,\"Longitude\":%.7f,\"Latitude\":%.7f,\"Altitude\":%.7f,\"YawAngle\":%.7f},", \
						         i + 1, 2, 1.00 * p_outbuf->udetPersonBox[i].distance[0], 1.00 * p_outbuf->udetPersonBox[i].distance[1], 1.00 * p_outbuf->udetPersonBox[i].speed, 1.00 * p_outbuf->udetPersonBox[i].speed_Vx,0.1 * p_outbuf->udetPersonBox[i].length \
						         , 1.00 * p_outbuf->udetPersonBox[i].width, lon_lat[0], lon_lat[1], info.Altitude, yaw_angle);
                        buf_len = strlen(tmpBuf);
						strncpy(buf + total_len, tmpBuf, buf_len);
						//snprintf(buf+total_len, buf_len,"%s", tmpBuf);
						total_len += buf_len;
					}

					if (buf[total_len - 1] == ',')
						buf[total_len - 1] = ' ';

					strncpy(buf + total_len, "]} ", 3);
					total_len += 3;

					ws_send_msg((void **)&vhd, buf, total_len);
				}

				if (node->p_del_fun)
					node->p_del_fun(node->p_value);
				if (node->p_value)
					free(node->p_value);
				free(node);
			}
			//prt(info, "video_queue size: %d", g_ws_queue.video_queue->size);

		}

		if (g_ivddevsets.pro_type == PROTO_WS_RADAR) { //雷达数据

			PNode node = DeQueue(g_ws_queue.radar_queue);
			if (node && node->val_len > 33) {
				total_len = 0;
				buf_len = 0;

				short array_num = node->val_len / sizeof(mRadarRTObj);

				if (array_num > 0) {

					memset(buf, 0, SEND_BUFF_MAX);
					snprintf(buf, 128, "{\"WSType\":\"data\",\"DeviceNo\":\"%s\",\"Timestamp\":\"%s\",\"DetectorType\":1,\"TargetNum\":%d,\"TargetArray\":[ ",   \
					         g_ivddevsets.devUserNo, tmpBuf, array_num);

					total_len = strlen(buf);
					mRadarRTObj *obj = (mRadarRTObj *)node->p_value;

					for (int i = 0; i < array_num; i++) {
						memset(tmpBuf, 0, TMP_BUF_MAX);
						//buffer_data_reverse(node->p_value + i*11 + 3, 8);
						//mRadarObj *obj = (mRadarObj *)(node->p_value + i*11);
						//snprintf(tmpBuf, TMP_BUF_MAX, "{\"ID\":%d,\"ParticipantType\":%d,\" XPos\":%f,\"YPos\":%f,\"Speed\":%f,\"Length\":%f,\"Width\":%f,\"Longitude\":%f,\"Latitude\":%f,\"Altitude\":%f,\"YawAngle\":%f},", \ 
						//	i+1, 3, (obj->x_Point-4096)*0.128, (obj->y_Point-4096)*0.128, (obj->Speed_x-1024)*0.36, obj->Object_Length*0.2 \
						//	 ,0.00,info.Longitude, info.Latitude, info.Altitude,info.YawAngle);


						snprintf(tmpBuf, TMP_BUF_MAX, "{\"ID\":%d,\"ParticipantType\":%d,\"XPos\":%f,\"YPos\":%f,\"Speed\":%f,\"Length\":%f,\"Width\":%f,\"Longitude\":%f,\"Latitude\":%f,\"Altitude\":%f,\"YawAngle\":%f},", \
						         i + 1, 1, obj->x_Point, obj->y_Point, obj->Speed_x, obj->Obj_Len \
						         , 0.00, info.Longitude, info.Latitude, info.Altitude, info.YawAngle);
						buf_len = strlen(tmpBuf);
						strncpy(buf + total_len, tmpBuf, buf_len);
						//snprintf(buf+total_len, buf_len,"%s", tmpBuf);
						total_len += buf_len;

						obj++;
					}

					if (buf[total_len - 1] == ',')
						buf[total_len - 1] = ' ';

					strncpy(buf + total_len, "]} ", 3);
					total_len += 3;
				}

				//释放node对象
				if (node->p_del_fun)
					node->p_del_fun(node->p_value);
				if (node->p_value)
					free(node->p_value);
				free(node);

				//printf("ws_hanle_realtime_data: %d %s \n", total_len, buf);
				ws_send_msg((void **)&vhd, buf, total_len);
			}

		}

		usleep(30000); //30ms

	} while (1);
}


void ws_send_msg(void **p_vhd, char *p_msg, unsigned short msg_len)
{
	struct per_vhost_data__minimal **vhd =
	    (struct per_vhost_data__minimal **)p_vhd;
	struct msg amsg;
	int n;

	//do {
	/* don't generate output if nobody connected */
	if (!(*vhd)->pss_list)
		goto wait;

	pthread_mutex_lock(&(*vhd)->lock_ring); /* --------- ring lock { */

	/* only create if space in ringbuffer */
	n = (int)lws_ring_get_count_free_elements( (*vhd)->ring);
	if (!n) {
		lwsl_user("dropping!\n");
		goto wait_unlock;
	}


	amsg.payload = malloc(LWS_PRE + msg_len);
	if (!amsg.payload) {
		lwsl_user("OOM: dropping\n");
		goto wait_unlock;
	}
	lws_strncpy((char *)amsg.payload + LWS_PRE, p_msg, msg_len);
	amsg.len = msg_len;
	n = lws_ring_insert( (*vhd)->ring, &amsg, 1);
	if (n != 1) {
		__minimal_destroy_message(&amsg);
		lwsl_user("dropping!\n");
	} else
		lws_cancel_service( (*vhd)->context);

wait_unlock:
	pthread_mutex_unlock(&(*vhd)->lock_ring); /* } ring lock ------- */

wait:
	usleep(100000);

	//} while (!(*vhd)->finished);
}



