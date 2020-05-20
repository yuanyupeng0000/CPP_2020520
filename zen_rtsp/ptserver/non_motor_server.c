#include <unistd.h>
#include <sys/select.h>
#include <sys/types.h>
#include <sys/time.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <stdio.h>
#include "non_motor_server.h"
#include "../g_define.h"
#include "../queue/queue.h"
#include "../camera_service.h"


extern m_camera_info cam_info[CAM_MAX];

int non_motor_srv_fd = 0;

Queue *g_non_motor_queue = NULL; 

void init_non_motor_list()
{
    g_non_motor_queue = InitQueue();
}

void *get_non_motor_queue()
{
    return (void *)g_non_motor_queue;
}

int get_non_motor_fd()
{
    return non_motor_srv_fd;
}

void *non_motor_udp_server(void *data)
{
    int srv_fd;
    
    fd_set r_fds;
    char data_buf[MAX_IMG_ID_LEN + 1] = {0};
    struct sockaddr_in srv, clt_addr;
    socklen_t clt_len = sizeof(clt_addr);
    
    srv_fd = socket(PF_INET, SOCK_DGRAM, IPPROTO_UDP);

    if (srv_fd < 0) {
        prt(info,"non motor opening socket failed.");
        return NULL;
    }

    srv.sin_family = AF_INET;
    srv.sin_port = htons(NON_MOTOR_SERVER_PORT);
    srv.sin_addr.s_addr = htonl(INADDR_ANY);

    if (bind(srv_fd, (struct sockaddr*)&srv, sizeof(srv)) < 0) {
        prt(info, "non motor bind socket failed.");
        return NULL;
    }

    non_motor_srv_fd = srv_fd;

    while (1) {
        FD_ZERO(&r_fds);
        FD_SET(srv_fd, &r_fds);
        memset(data_buf, 0, MAX_IMG_ID_LEN+1);
        int ret = select(srv_fd+1, &r_fds, NULL, 0, 0);

        if (ret < 0)
            break;

        if (FD_ISSET(srv_fd, &r_fds)) {
            int n_bytes = recvfrom(srv_fd, data_buf, MAX_IMG_ID_LEN, 0, (struct sockaddr*)&clt_addr, &clt_len);
            if (n_bytes < 0){
                close(srv_fd);
                break;
            }
            if (n_bytes == 0 || n_bytes > MAX_IMG_ID_LEN){
                prt(info, "non motor receive len is error.");
                continue;
            }
            FD_CLR(srv_fd, &r_fds);

            img_list_node_t *p_node = (img_list_node_t*)malloc(sizeof(img_list_node_t)); 
            if(!p_node)
                continue;

            memset((void*)p_node, 0, sizeof(img_list_node_t)); 
            memcpy(&p_node->img_id, data_buf, MAX_IMG_ID_LEN);
            memcpy(&p_node->cli_fd, &clt_addr, sizeof(struct sockaddr_in));

             //add to list
            if (g_non_motor_queue) {
                EnQueue(g_non_motor_queue, (char*)p_node, sizeof(img_list_node_t), NULL);
            }else {
                free(p_node);
            }
        }
    }
}