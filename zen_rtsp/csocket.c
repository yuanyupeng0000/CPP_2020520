#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <string.h>
#include <sys/ioctl.h>
#include <net/if.h>
#include <fcntl.h>
#include <stdlib.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>
#include <linux/netlink.h>
#include <linux/rtnetlink.h>
#include <errno.h>
//#include <tcp.h>
#include <sys/ioctl.h>
#include "csocket.h"
#include "common.h"
//#include "zenlog.h"

extern pthread_mutex_t client_sock_lock;

#define TCP_NODELAY 1
int StartTcpServerSock(unsigned short port, int timeoutsec, int maxqueue)
{
    int fd;
    int value;
    struct timeval timeo;
    socklen_t len = sizeof(timeo);

    memset(&timeo, 0, sizeof(timeo));
    if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    {
        prt(err, "socket err");
        return -1;
    }
    value = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &value, sizeof(value)) < 0)
    {
        prt(err, "socket err");
        close(fd);
        return -1;
    }

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(struct sockaddr_in));
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = htonl(INADDR_ANY);
    sa.sin_port = htons(port);
    if (bind(fd, (struct sockaddr *)(&sa), sizeof(struct sockaddr)) < 0)
    {
        prt(err, "socket err,bind err");
        close(fd);
        return -1;
    }

    if (listen(fd, maxqueue) < 0)
    {
        prt(err, "socket err");
        close(fd);
        return -1;
    }
    return fd;
}

int CreatBroadcast(int port)
{
    int socket_fd;
    struct sockaddr_in user_addr;

    user_addr.sin_family = AF_INET;
    user_addr.sin_port = htons(port);
    user_addr.sin_addr.s_addr = htonl(INADDR_ANY);
    if ((socket_fd = (socket(AF_INET, SOCK_DGRAM, 0))) == -1)
    {
        perror("socket");
        return -1;
    }

    int so_broadcast = 1;
    setsockopt(socket_fd, SOL_SOCKET, SO_BROADCAST, &so_broadcast, sizeof(so_broadcast));
    if ((bind(socket_fd, (struct sockaddr *)&user_addr, sizeof(struct sockaddr))) == -1)
    {
        perror("bind");
        return -1;
    }
    return socket_fd;
}

int RecvFromBroadcast(unsigned char *buffer, int len, int sock)
{
    return recvfrom(sock, buffer, len, 0, NULL, NULL);
}

int WaitTcpConnect(int sock, unsigned long sec, char *ip, unsigned short *port)
{
    int cl_fd = 0, one;
    fd_set rd_set;
    struct timeval timeout;
    struct sockaddr_in client_addr;
    int client_len = sizeof(struct sockaddr_in);

    timeout.tv_sec = sec;
    timeout.tv_usec = 0;

    FD_ZERO(&rd_set);
    FD_SET(sock, &rd_set);
    while ((cl_fd = select(sock + 1, &rd_set, NULL, NULL, &timeout)) < 0)
    {

        if (EINTR == errno)
        {
            prt(err, "socket err EINTR");
        }
        if (EINVAL == errno)
        {
            prt(err, "socket err EINVAL");
        }
        if (ENOMEM == errno)
        {
            prt(err, "socket err ENOMEM");
        }
        if (EBADF == errno)
        {
            prt(err, "socket err EBADF");
        }
        prt(err, "socket err,need rebind socket");
        cl_fd = -1;
        return cl_fd;
    }
    //      prt(net,"select rst %d",cl_fd);
    //  if(0== cl_fd){
    //      prt(err,"select time out,need select again  fd %d",cl_fd);
    //  return 0;
    //  }

    if (FD_ISSET(sock, &rd_set))
    {
        //  prt(net,"get msg from client");
#if 0
        while ((cl_fd = accept(sock, (struct sockaddr *)&client_addr, (socklen_t *)&client_len)) < 0) {
            if (EINTR == errno) {
                //Log0("accept continue: cl_fd[%d] errno:[%s]", cl_fd, strerror(errno));
                continue;
            }
            else {
                //Log0("accept error: cl_fd[%d] errno:[%s]", cl_fd, strerror(errno));
                return 0;
            }
        }
#else
        if ((cl_fd = accept(sock, (struct sockaddr *)&client_addr, (socklen_t *)&client_len)) <= 0)
        {

            prt(err, "accept  err  %d", cl_fd);
            close_socket(&sock);

            cl_fd = -1;
        }
        else
        {
            //          prt(net,"accept rst %d",cl_fd);
        }
#endif
#if 0
#if 0
        one = 1;
        if (setsockopt(cl_fd, IPPROTO_TCP, TCP_NODELAY,
                       (char *)&one, sizeof(one)) < 0) {
            close(cl_fd);
            //Log0("setsockopt error: cl_fd[%d]", cl_fd);
            return 1;
        }
#else
        //设置发送超时
        //设置接收超时
        struct timeval timeout1 = {0, 450000};
        if ((setsockopt(cl_fd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeout1, sizeof(struct timeval)) < 0)
                || (setsockopt(cl_fd, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeout1, sizeof(struct timeval)) < 0)) {
            close(cl_fd);
            //Log0("setsockopt error: cl_fd[%d] errno:[%s]", cl_fd, strerror(errno));
            return 1;
        }
#endif
#endif

        if (ip != NULL)
            strcpy(ip, inet_ntoa(client_addr.sin_addr));
        if (port != NULL)
            *port = ntohs(client_addr.sin_port);
        // return cl_fd;
    }
    else
    {
        prt(net, "client time out ,returning %d ", cl_fd);
    }
    return cl_fd;
    //   return 1;
}

int ConnectTcpClient(int sock, char *ip, unsigned short port, bool block)
{
    struct sockaddr_in server_addr;
    memset(&server_addr, 0, sizeof(struct sockaddr_in));
    server_addr.sin_addr.s_addr = inet_addr(ip);
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(port);

#if 0
    int error = -1, len, one;
    len = sizeof(int);
    timeval tm;
    fd_set set;
    unsigned long ul = 1;
    ioctl(sock, FIONBIO, &ul); //设置为非阻塞模式
#endif

    fd_set set;
    unsigned long ul = 1;
    ioctl(sock, FIONBIO, &ul); //设置为非阻塞模式

    ul = 0;
    ioctl(sock, FIONBIO, &ul); //设置为阻塞模式

    /*
    if (!block) {
        int save_mode = fcntl(sock, F_GETFL, 0);
        fcntl(sock, F_SETFL, save_mode | O_NONBLOCK);//设置为阻塞模式
    }
    */
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(struct sockaddr)) == -1)
    {
        /*
                {
                    ul = 0;
                        ioctl(sock, FIONBIO, &ul); //设置为 阻塞模式

                        return -1;
                }
                */
#if 0
        tm.tv_sec = 1;
        tm.tv_usec = 0;
        FD_ZERO(&set);
        FD_SET(sock, &set);
        if (select(sock + 1, NULL, &set, NULL, &tm) > 0) {
            getsockopt(sock, SOL_SOCKET, SO_ERROR, &error, (socklen_t *)&len);
            if (error != 0)
                return -1;
        } else
            return -1;
#if 0
        one = 1;
        if (setsockopt(sock, IPPROTO_TCP, TCP_NODELAY,
                       (char *)&one, sizeof(one)) < 0) {
            close(sock);
            return 1;
        }
#endif
#endif
        return -1;
    }

#if 0
    ul = 0;
    ioctl(sock, FIONBIO, &ul); //设置为阻塞模式
#endif
    return 1;
}

int RecvDataByTcp(int sock, char *buffer, int len)
{
    prt(info, "get buffer len : %d", len);
    int offset = 0;
    while (len > 0)
    {
        int re = recv(sock, buffer + offset, len, 0);
        if (re <= 0)
        {
            if (re < 0 && EINTR == errno)
            {
                //Log0("recv continue: re[%d] errno:[%s]", re, strerror(errno));
                continue;
            }
            else
            {
                //Log0("recv error: re[%d] errno:[%s]", re, strerror(errno));
                return offset;
            }
        }
        len -= re;
        offset += re;
    }
    return offset;
}

//int RecvDataByTcp(int sock,char* buffer,int len){
//int re = recv(sock,buffer+offset,len,0);

//}
int SendDataToClient(int sock, char *buffer, int len)
{
    int ret = 0;
    int send_total = 0;
    pthread_mutex_lock(&client_sock_lock);
    //prt(info, "SendDataToClient fd: %d start1--1--", sock);
    while (send_total < len) {

        ret = send(sock, buffer + send_total, len - send_total, MSG_NOSIGNAL);

        if (ret <= 0 ) {

            if (errno == EWOULDBLOCK || errno == EAGAIN) {
                continue;
            }

            break;
        }

        send_total += ret;
    }
    //prt(info, "SendDataToClient end--1--1--");
    pthread_mutex_unlock(&client_sock_lock);

    //if(ret<=0){
    //  return -1;
    // }

    return send_total;
}

int SendDataByTcp(int sock, char *buffer, int len)
{
    int ret = send(sock, buffer, len, MSG_NOSIGNAL);
    if (ret <= 0)
    {
        return -1;
    }
    return ret;
}

#if 0
int SendDataByTcp(int sock, char * buffer, int len)
{
    int ret, lens = 0;
    while (1) {
        ret = send(sock, buffer + lens, len - lens, 0);
        if (ret <= 0) {
            if (ret < 0 && EINTR == errno) {
                //Log0("send continue: re[%d] errno:[%s]", ret, strerror(errno));
                continue;
            }
            else {
                //Log0("send error: ret[%d] errno:[%s]", ret, strerror(errno));
                return ret;
            }
        }
        else {
            lens += ret;
            if (lens == len)
                break;
        }
    }
    return lens;
}
#endif

int SendDataByClient(char *buffer, int len, char *ip, int port)
{
    int sc = CreateTcpClientSock(0, 1);
    if (sc == -1)
        return sc;

    int ret = ConnectTcpClient(sc, ip, port, true);
    if (ret == -1)
    {
        close(sc);
        return -1;
    }

    ret = SendDataByTcp(sc, buffer, len);
    close(sc);
    return ret;
}

int CreateTcpClientSock(unsigned short port, int timeoutsec)
{
    int fd;
    int value;
    struct timeval timeo;
    socklen_t len = sizeof(timeo);

    memset(&timeo, 0, sizeof(timeo));
    if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0)
        return -1;

    value = 1;
    if (setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &value, sizeof(value)) < 0)
    {
        close(fd);
        return -1;
    }

    struct sockaddr_in sa;
    memset(&sa, 0, sizeof(struct sockaddr_in));
    sa.sin_family = AF_INET;
    sa.sin_addr.s_addr = INADDR_ANY;
    sa.sin_port = htons(port);
    if (bind(fd, (struct sockaddr *)(&sa), sizeof(struct sockaddr)) < 0)
    {

        close(fd);
        return -1;
    }
    return fd;
}

int UdpCreateSocket(unsigned short port)
{
    int sockfd;
    int optval = 1;
    struct sockaddr_in my_addr;
    if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
    {
        return -1;
    }

    my_addr.sin_family = AF_INET;
    my_addr.sin_port = htons(port);
    my_addr.sin_addr.s_addr = INADDR_ANY;
    setsockopt(sockfd, SOL_SOCKET, SO_REUSEPORT, &optval, sizeof(optval));

    if (bind(sockfd, (struct sockaddr *)&my_addr, sizeof(struct sockaddr)) < 0)
    {
        print_net("bind port %d err", port);
        return -1;
    }
    else
    {
        print_net("bind port %d ok", port);
    }

    return sockfd;
}

int UdpSendData(int sock, char *ip, unsigned short port, char *buf, int slen)
{
    struct sockaddr_in their_addr;
    their_addr.sin_family = AF_INET;
    their_addr.sin_port = htons(port);
    inet_pton(AF_INET, ip, &their_addr.sin_addr);
    int ret, sended = 0;
    int len = slen;
    while (1)
    {
        ret = sendto(sock, buf + sended, len, 0, (struct sockaddr *)&their_addr, sizeof(struct sockaddr));
        if (ret == -1)
        {

            //            print_stacktrace();
            break;
        }
        else
        {
            sended += ret;
            len -= ret;
            if (sended == slen)
                break;
        }
    }
    //  print_net("send port %d ok,len %d,shoud be %d",port,sended,slen);
    //  dump_cam_ip(ip);
    return sended;
}

int close_socket(int *s)
{
    //  prt(info,"closing fd %d",*s);
    //print_stacktrace();
    shutdown(*s, SHUT_RDWR);
    close(*s);
    *s = -1;
}

int creat_tcp_socket(unsigned int r_timeout_ms, unsigned int s_timeout_ms)
{
    struct timeval tv_out;
    tv_out.tv_sec = r_timeout_ms / 1000;
    tv_out.tv_usec = (r_timeout_ms % 1000) * 1000;
    struct sockaddr_in client_addr;
    bzero(&client_addr, sizeof(client_addr));
    client_addr.sin_family = AF_INET;                // internet鍗忚鏃�
    client_addr.sin_addr.s_addr = htons(INADDR_ANY); // INADDR_ANY琛ㄧず鑷姩鑾峰彇鏈満鍦板潃
    client_addr.sin_port = htons(0);                 // auto allocated, 璁╃郴缁熻嚜鍔ㄥ垎閰嶄竴涓┖闂茬鍙�

    // 鍒涘缓鐢ㄤ簬internet鐨勬祦鍗忚(TCP)绫诲瀷socket锛岀敤client_socket浠ｈ〃瀹㈡埛绔痵ocket
    int i_socket = socket(AF_INET, SOCK_STREAM, 0);
    if (i_socket < 0)
    {
        printf("Create Socket Failed!\n");
        goto err;
    }

    setsockopt(i_socket, SOL_SOCKET, SO_RCVTIMEO, &tv_out, sizeof(tv_out));
    tv_out.tv_sec = s_timeout_ms / 1000;
    tv_out.tv_usec = (s_timeout_ms % 1000) * 1000;
    setsockopt(i_socket, SOL_SOCKET, SO_SNDTIMEO, &tv_out, sizeof(tv_out));

    // 鎶婂鎴风鐨剆ocket鍜屽鎴风鐨剆ocket鍦板潃缁撴瀯缁戝畾
    if (i_socket > 0 && bind(i_socket, (struct sockaddr *)&client_addr, sizeof(client_addr)) < 0)
    {
        printf("Client Bind Port Failed!\n");
        close(i_socket);
        i_socket = -1;
    }

err:
    return i_socket;
}
