/*
 * tcp_server.c
 *
 *  Created on: 2016年8月16日
 *      Author: root
 *      初始化本机8888端口作为tcp服务端，不断接收tcp客户端的访问，大小限制为1k字节。
 *      凡是来消息的客户端，在客户访问列表里面保持一定时间，并且保存来访相机索引号，
 *      用以确定该相机发送udp数据至客户ip
 */

#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <netinet/in.h>
#include <netinet/tcp.h>

//#include <linux/in.h>
#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <fcntl.h>
#include <fcntl.h>
#include <sys/ioctl.h>
//#include <linux/time.h>
#include "common.h"
#include "client_net.h"
#include <errno.h>
#include <sys/select.h>
#include "tcp_server.h"
#include "csocket.h"

sock_info_t g_sock_info;


//#define BUFFER_MAX 1000
//#define TCP_TIMEOUT 100

extern IVDNetInfo       g_netInfo;
extern IVDDevInfo       g_sysInfo;

int init_skt()
{
		int port=10001;
		int maxqueue=100;
		int fd;
		int value;
		struct timeval timeo;
		socklen_t len = sizeof(timeo);

		memset(&timeo, 0, sizeof(timeo));
        timeo.tv_usec = 200000;
        timeo.tv_sec = 2;

		if ((fd = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
			prt(err, "socket err");
			return -1;
		}

	    value = 1;
	    if(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &value, sizeof(value)) < 0){
	        close(fd);
	        return -1;
	    }

#if 0
		if(setsockopt(fd, SOL_SOCKET, SO_REUSEPORT, &value, sizeof(value)) < 0){
	        close(fd);
	        return -1;
	    }
#endif
		struct sockaddr_in sa;
		memset(&sa, 0, sizeof(struct sockaddr_in));
		sa.sin_family = AF_INET;
		sa.sin_addr.s_addr = htonl(INADDR_ANY);
		sa.sin_port = htons(port);
		if (bind(fd, (struct sockaddr *) (&sa), sizeof(struct sockaddr)) < 0) {
			prt(err, "socket err,bind err,fd %d",fd);
			close(fd);
			return -1;
		}

		if (listen(fd, maxqueue) < 0) {
			prt(err, "socket err");
			close(fd);
			return -1;
		}

		setsockopt(fd, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeo, sizeof(struct timeval));
        //setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,(char *)&timeo,sizeof(struct timeval));
		return fd;
}

#if 0
int WaitTcpConnect(int sock, unsigned long sec, char * ip, unsigned short * port)
{
    int cl_fd=0, one;
    fd_set rd_set;
    struct timeval timeout;
    struct sockaddr_in  client_addr;
    int client_len = sizeof(struct sockaddr_in);

    timeout.tv_sec = sec;
    timeout.tv_usec = 0;

    FD_ZERO(&rd_set);
    FD_SET(sock, &rd_set);
	while ((cl_fd = select(sock + 1, &rd_set, NULL, NULL, &timeout)) < 0) {

		if (EINTR == errno) {
			prt(err, "socket err EINTR");

		}
		if (EINVAL == errno) {
			prt(err, "socket err EINVAL");

		}
		if (ENOMEM == errno) {
			prt(err, "socket err ENOMEM");

		}
		if (EBADF == errno) {
			prt(err, "socket err EBADF");

		}
		prt(err, "socket err,need rebind socket");
		cl_fd = -1;
		return cl_fd;
	}
//  	prt(net,"select rst %d",cl_fd);
//	if(0== cl_fd){
//	   	prt(err,"select time out,need select again  fd %d",cl_fd);
	//	return 0;
//	}

    if(FD_ISSET(sock, &rd_set))
    {
  //  	prt(net,"get msg from client");

		if((cl_fd = accept(sock, (struct sockaddr *)&client_addr, (socklen_t *)&client_len))<=0){

		   	prt(err,"accept  err  %d",cl_fd);
			//close_socket(&sock);

		    cl_fd=-1;
		}else{
//		  	prt(net,"accept rst %d",cl_fd);
		}



		if(ip != NULL)
            strcpy(ip, inet_ntoa(client_addr.sin_addr));
		if(port != NULL)
            *port = ntohs(client_addr.sin_port);
       // return cl_fd;
	}else{
		prt(net,"no client in 10s ",cl_fd);
	}
    return cl_fd;
 //   return 1;
}

#endif
void handle_buf(char *bf)
{
	mCommand *p=(mCommand *)bf;
	prt(info,"%d ",p->version);
	prt(info,"number %d ",ntohs(p->objnumber));
	prt(info,"type %d ",ntohs(p->objtype));
	prt(info,"len %d ",ntohl(p->objlen));
	prt(info,"%d ",p->prottype);


//	mDetectDeviceConfig test;

}

#if 0
int recv_buf(int fd)
{
	unsigned short port;
	char ip[16];
	unsigned char buf[BUFFER_MAX];
	memset(buf,0,BUFFER_MAX);
	int ret;
	int client;
	prt(server,"waiting for clients ");
	client=WaitTcpConnect(fd,TCP_TIMEOUT,ip,&port);
	if(client>0){

		ret=recv(client,buf,BUFFER_MAX,0);
		int send_len=handle_buffer(buf,ret,ip);
		if(send_len>0)
		{
			send(client,&buf,send_len,0);
		}

		prt(server,"    clients  done ");

		//close(client); //why close 2019.03.26

		return 0;
	}else{
		return -1;
	}

}
#endif

int recv_and_send(int index, char *buff)
{
    int ret = 0;
    int send_len=handle_buffer(index, (unsigned char *)buff,ret);
	if(send_len>0){
		send(g_sock_info.client_sockfd[index],buff,send_len, MSG_NOSIGNAL);
         //prt(info, "socket already send len: %d ", send_len);
		 //prt(info, "send data: %02X %02X %02X %02X %02X %02X %02X %02X %02X %02X", 
		 //	buff[0], buff[1], buff[2], buff[3],(unsigned char)buff[4],buff[5],buff[6],buff[7],buff[8],(unsigned char)buff[9]);
		 
	}

}

int recv_param()
{
	return 0;
}

int handle_cmd()
{
	return 0;
}

int send_rst()
{
	return 0;
}

int  init_tcp_server()
{
    int aready_index = 0;
    int ret = 0;
    int maxsock = 0;
    int serverfd;
    char buffer[BUFFER_MAX];
    //char g_sock_info.ip_addr[SOCK_FD_MAX][16];
    //int g_sock_info.conn_amount = 0;    /*鐢ㄦ潵璁板綍鎻忚堪绗?/
    //int g_sock_info.client_sockfd[SOCK_FD_MAX];   /*瀛樻斁娲诲姩鐨剆ockfd*/
    int cmd_head_len = 0;
    int cmd_data_len = 0;
    int received_len = 0;
    int handled_cnt = 0;
    struct timeval tv;      /*瓒呮椂杩斿洖鏃堕棿*/
    fd_set client_fdset;    /*鐩戞帶鏂囦欢鎻忚堪绗﹂泦 */
    //fd_set g_sock_info.client_fdset_all;
    memset(&g_sock_info, 0, sizeof(sock_info_t));
    /*鐩戞帶鏂囦欢鎻忚堪绗︿腑鏈€澶х殑鏂囦欢 */
    bzero((void*)g_sock_info.client_sockfd,sizeof(g_sock_info.client_sockfd));

	serverfd=init_skt();

    if (serverfd < 1) {
        prt(err, "tcp server init failed, so quit!");
        return -1;
    }

    maxsock = serverfd;
    cmd_head_len = sizeof(mCommand);
    FD_ZERO(&client_fdset);
    FD_ZERO(&g_sock_info.client_fdset_all);
    FD_SET(serverfd,&g_sock_info.client_fdset_all);

while(1){
        memset(buffer,0, BUFFER_MAX);
        tv.tv_sec = 30;
        tv.tv_usec = 0;

        //prt(info, "sock while.....................>>>>>");
        client_fdset = g_sock_info.client_fdset_all;
        ret = select(maxsock+1, &client_fdset, NULL, NULL, &tv);

        if(ret < 0 ){
            prt(info, "select error: %d maxsock: %d fdset: %d g_sock_info.client_fdset_all: %d \n", ret, maxsock, client_fdset, g_sock_info.client_fdset_all);
            continue;
            //break;
        } else if(ret == 0){
            printf("socket timeout!\n");
            continue;
        }

        aready_index =0;
      //  handled_cnt = 0;
    //    prt(info, " connect number: %d ",  g_sock_info.conn_amount);
      //  while( (handled_cnt < g_sock_info.conn_amount) &&  (aready_index < SOCK_FD_MAX) ) {
        for(int i = 0; i < SOCK_FD_MAX; i++) {
            if (g_sock_info.client_sockfd[i] == 0){
                continue;
            }

            received_len = 0;
            cmd_data_len = 0;

           // int i = aready_index;
          //  handled_cnt++;
          //  aready_index++;

            if(FD_ISSET(g_sock_info.client_sockfd[i], &client_fdset)){

                prt(info,"start recv from client[%d]:\n",i);
                ret = recv(g_sock_info.client_sockfd[i], buffer, cmd_head_len, 0); //鍏堣鍗忚

                if (ret <=0 ) {

                     if (del_client(g_sock_info.ip_addr[i], -1, g_sock_info.client_sockfd[i]) != 0) {
                        close(g_sock_info.client_sockfd[i]);
                     }

                    clear_socket(i);

                    /*
                    FD_CLR(g_sock_info.client_sockfd[i], &g_sock_info.client_fdset_all);
                    g_sock_info.client_sockfd[i] = 0;
                    memset(g_sock_info.ip_addr[i], 0, sizeof(g_sock_info.ip_addr[i]));
                    g_sock_info.conn_amount--;
                    */

                    continue;
                }else if (ret != cmd_head_len) {
                    prt(info,"read command head error\n");
                    recv(g_sock_info.client_sockfd[i], buffer, BUFFER_MAX, 0);

                    continue;
                }


                if (0x01 != buffer[0] ||  0x10 != buffer[1])
                    continue;

                 cmd_data_len = ntohl( ((mCommand *)buffer)->objlen );

                 while ( (cmd_data_len - received_len) > 0) {
                    ret = recv(g_sock_info.client_sockfd[i], buffer + cmd_head_len + received_len, cmd_data_len - received_len, 0); //鍏堣鍗忚澶?

                     if (ret > 0)
                        received_len += ret;

                     if(ret <=0 &&  !(errno == EAGAIN||errno == EWOULDBLOCK||errno == EINTR))
                        break;

                 }

                 if (cmd_data_len != received_len)
                    continue;

                if(ret <= 0){
                    if (del_client(g_sock_info.ip_addr[i], 0, g_sock_info.client_sockfd[i]) != 0) {
                        close(g_sock_info.client_sockfd[i]);

                    }

                    clear_socket(i);
                    /*
                    FD_CLR(g_sock_info.client_sockfd[i], &g_sock_info.client_fdset_all);
                    g_sock_info.client_sockfd[i] = 0;
                    memset(g_sock_info.ip_addr[i], 0, sizeof(g_sock_info.ip_addr[i]));
                    g_sock_info.conn_amount--;
                    */

                }
                else{

					printf("......recv_and_send start.............. \n");

                    for (int m = 0; m < ret; m++){
                        printf("%02X ", (unsigned char)buffer[m]);
                    }
					printf("\n");
					printf("......recv_and_send finish..............\n");
                    recv_and_send(i, buffer);
                    
                }
            }

        }

        if(FD_ISSET(serverfd, &client_fdset))
        {
            struct sockaddr_in client_addr;
            size_t size = sizeof(struct sockaddr_in);
			//prt(info, "accept start!\n");
            int sock_client = accept(serverfd, (struct sockaddr*)(&client_addr), (unsigned int*)(&size));
            if(sock_client < 0){
                prt(info, "accept error!\n");
                continue;
            }

            //prt(info,"client accept ip: %s maxConn: %d", inet_ntoa(client_addr.sin_addr), g_netInfo.maxConn);

            if(g_sock_info.conn_amount < g_netInfo.maxConn  &&  g_sock_info.conn_amount < SOCK_FD_MAX )
            {
                int sock_index = -1;

                for(int i=0; i < SOCK_FD_MAX; i++) {
                    if (g_sock_info.client_sockfd[i] == 0) {
                        sock_index = i;
                       break;
                    }
                }

                if (sock_index == -1) {
                    close(sock_client);
                    prt(info, "sock index err");
                    continue;
                }

                fcntl(sock_client, F_SETFL, O_NONBLOCK);
                g_sock_info.client_sockfd[sock_index] = sock_client;
                FD_SET(sock_client,&g_sock_info.client_fdset_all);
                sprintf(g_sock_info.ip_addr[sock_index],"%s",inet_ntoa(client_addr.sin_addr));


                if(sock_client >= maxsock){
                    maxsock = sock_client;
                }

            	g_sock_info.conn_amount++;
				//prt(info, "conn number: %d", g_sock_info.conn_amount);
            }else {
				prt(info, "conn overt error.");
            }
        }
    }

    for(int i = 0; i < SOCK_FD_MAX; ++i){
        if(g_sock_info.client_sockfd[i] != 0){
            close(g_sock_info.client_sockfd[i]);
        }
    }
    close(serverfd);

}


void test_client()
{
	int fd;
    if((fd =socket(AF_INET, SOCK_STREAM, 0))<0)
    {

    }
    int value = 1;
//    if(setsockopt(fd, SOL_SOCKET, SO_REUSEADDR, &value , sizeof(value))<0){
//        close(fd);
//        return -1;
//    }

   // struct sockaddr addr;
  //  addr.sa_data
	struct sockaddr_in addr;
	int port=5678;
	memset(&addr, 0, sizeof(struct sockaddr_in));
    addr.sin_family= AF_INET;
	addr.sin_addr.s_addr = inet_addr("192.168.1.113");
//	addr.sin_addr.s_addr = htonl(INADDR_ANY);
	addr.sin_port = htons(port);
	if (connect(fd, (struct sockaddr *) &addr, sizeof(struct sockaddr)) == -1) {
		printf("connect fail\n");
		//return -1;
	}else{
		printf("connect ok\n");
 	}
	unsigned long ul = 1;
	ioctl(fd, FIONBIO, &ul); //设置为非阻塞模式
	char buf[10];
	memset(buf,0,10);
	printf("prepare to reve\n");
	int ret=0;
	ret=recv(fd,buf,10,0);
	ret=printf("rece ok, ret %d,get %d %d %d\n",ret,buf[0],buf[1],buf[2]);
	ret=recv(fd,buf,10,0);
	printf("rece ok,ret %d,get %d %d %d\n",ret,buf[0],buf[1],buf[2]);
	ret=recv(fd,buf,10,0);
	printf("rece ok,ret %d,get %d %d %d\n",ret,buf[0],buf[1],buf[2]);
	while(1) ;

}


void clear_socket(int index)
{
    prt(info, "clear socket ip: %s", g_sock_info.ip_addr[index]);
	if (g_sock_info.client_sockfd[index] > 0) {
	    FD_CLR(g_sock_info.client_sockfd[index], &g_sock_info.client_fdset_all);
	    g_sock_info.client_sockfd[index] = 0;
	    memset(g_sock_info.ip_addr[index], 0, sizeof(g_sock_info.ip_addr[index]));
	}
	if (g_sock_info.conn_amount > 0)
    	g_sock_info.conn_amount--;

}

int creat_sig_sock(int flag, int send_ms, int recv_ms) 
{
	int ret_sock = CreateTcpClientSock(0, 0);
	
	struct timeval timeo;
	
	if (ret_sock > 0) {
		if ( (flag & 0x01) > 0 ) { //send
			memset(&timeo, 0, sizeof(timeo));
		    timeo.tv_usec = (send_ms%1000)*1000;
		    timeo.tv_sec = send_ms/1000;
			setsockopt(ret_sock, SOL_SOCKET, SO_SNDTIMEO, (char *)&timeo, sizeof(struct timeval));
		}

		if ( (flag & 0x02) > 0 ) { //receive
			memset(&timeo, 0, sizeof(timeo));
		    timeo.tv_usec = (recv_ms%1000)*1000;
		    timeo.tv_sec = recv_ms/1000;
			setsockopt(ret_sock, SOL_SOCKET, SO_RCVTIMEO, (char *)&timeo, sizeof(struct timeval));
		}
	}

	return ret_sock;
}

int set_nonblock(int sock_fd, bool b_set)
{
	int flags = 0;

	flags = fcntl(sock_fd, F_GETFL, 0);
	if (-1 == flags)
	{
		printf("get sockfd flag -1, errno=%d\n", errno);
		return -1;
	}

	if (b_set)
	{
		if (fcntl(sock_fd, F_SETFL, flags | O_NONBLOCK) == -1)
		{
			printf("set sockfd nonblock -1, errno=%d\n", errno);
			return -1;
		}
	}
	else
	{
		if (fcntl(sock_fd, F_SETFL, flags & (~O_NONBLOCK)) == -1)
		{
			printf("set sockfd nonblock -1, errno=%d\n", errno);
			return -1;
		}
	}

	return 0;
}

int creat_client_sock(char *ip, int port)
{
	int ret = 0;
	int sock_fd = -1;
    if((sock_fd =socket(AF_INET, SOCK_STREAM, 0))<0)  {
		return -1;
    }
	
	struct sockaddr_in server_addr;
	bzero(&server_addr, sizeof(struct sockaddr_in));
	server_addr.sin_family= AF_INET;
	server_addr.sin_port = htons(port);
	inet_pton(AF_INET, ip, &server_addr.sin_addr);

	set_nonblock(sock_fd, true);
	
	if (connect(sock_fd, (struct sockaddr *) &server_addr, sizeof(struct sockaddr)) == 0) {
		printf("connect ok\n");
		return sock_fd;
	
	}else{
		if (errno == EINPROGRESS) {
			int times = 0;
			while (times++ < 3) { //5
				fd_set rfds, wfds;
				struct timeval tv;
	
				printf("errno = %d\n", errno);
				FD_ZERO(&rfds);
				FD_ZERO(&wfds);
				FD_SET(sock_fd, &rfds);
				FD_SET(sock_fd, &wfds);

				/* set select() time out */
				tv.tv_sec = 2;//10
				tv.tv_usec = 0;
				int selres = select(sock_fd + 1, &rfds, &wfds, NULL, &tv);
				switch (selres)
				{
					case -1:
						printf("select error\n");
						ret = -1;
						break;
					case 0:
						printf("select time out\n");
						ret = -1;
						break;
					default:
						if (FD_ISSET(sock_fd, &rfds) || FD_ISSET(sock_fd, &wfds)) {
			              connect(sock_fd, (struct sockaddr *)&server_addr, sizeof(struct sockaddr_in));
			              int err = errno;
			              if (err == EISCONN)
			              {
			                printf("connect finished .\n");
			                ret = 0;
			              }
			              else
			              {
			                printf("connect failed. errno = %d\n", errno);
			                printf("FD_ISSET(sock_fd, &rfds): %d\n FD_ISSET(sock_fd, &wfds): %d\n", FD_ISSET(sock_fd, &rfds) , FD_ISSET(sock_fd, &wfds));
			                ret = errno;
			              }
			          
			            }
	        	}
	       
		        if (-1 != selres && (ret != 0)) {
		          printf("check connect result again... %d\n", times);
		          continue;
		        } else 
		          break;
	    
			}
		 } else {
	 	 	 printf("connect to host %s:%d failed.\n", ip, port);
	 	 	 return -1;
	 	 }
 }

	if (0 == ret)
		return sock_fd;
	else {
		close(sock_fd);
		return -1;
	}
		
}

