#ifndef __TCP_SOCK_C_
#define __TCP_SOCK_C_

#define SWAP_32(x) ((((x)&0xFF000000)>>24)|(((x)&0x00FF0000)>>8)|(((x)&0x0000FF00)<<8)|(((x)&0x000000FF)<<24))
#define SWAP_16(x) ((((x)& 0xFF00)>>8)|(((x) & 0x00FF)<<8))

#if defined (__LITTLE_ENDIAN)
    #define HTONL(x) SWAP_32(x)
    #define HTONS(x) SWAP_16(x)
    #define NTOHL(x) SWAP_32(x)
    #define NTOHS(x) SWAP_16(x)
#elif defined (__BIG_ENDIAN)
    #define HTONL(x) (x)
    #define HTONS(x) (x)
    #define NTOHL(x) (x)
    #define NTOHS(x) (x) 
#endif

#ifdef __cplusplus
extern "C"
{
#endif

int StartTcpServerSock(unsigned short port, int timeoutsec,int maxqueue);
int WaitTcpConnect(int sock, unsigned long msec, char * ip, unsigned short * port);
int ConnectTcpClient(int sock, char * ip, unsigned short port, bool block);

int RecvDataByTcp(int sock,char* buf,int len);
int SendDataByTcp(int sock, char * buffer, int len);
//
int SendDataToClient(int sock, char * buffer, int len);


int CreateTcpClientSock(unsigned short port, int timeoutsec);
int SendDataByClient(char*buffer, int len, char* ip, int port);

int CreatBroadcast(int port);
int RecvFromBroadcast(unsigned char *buffer, int len, int sock);

int UdpCreateSocket(unsigned short port);
int UdpSendData(int sock, char * ip, unsigned short port, char * buf, int len);
int close_socket(int *s);
int creat_tcp_socket(unsigned int r_timeout_ms, unsigned int s_timeout_ms);


#ifdef __cplusplus
}
#endif

#endif //__TCP_SOCK_C_
