/*
 * tcp_server.h
 *
 *  Created on: 2016��9��10��
 *      Author: root
 */

#ifndef TCP_SERVER_H_
#define TCP_SERVER_H_
#include <sys/socket.h>


#define SOCK_FD_MAX 100

typedef struct {

    char ip_addr[SOCK_FD_MAX][16];
    int conn_amount;    /*用来记录描述符数量*/
    fd_set client_fdset_all;
    int client_sockfd[SOCK_FD_MAX];   /*存放活动的sockfd*/

}sock_info_t;

void clear_socket(int index);
int  init_tcp_server();
int creat_sig_sock(int flag, int send_ms, int recv_ms);
int creat_client_sock(char *ip, int port);

#endif /* TCP_SERVER_H_ */
