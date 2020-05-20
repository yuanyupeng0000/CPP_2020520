#include "zenarp.h"

extern int g_flag;
extern unsigned char sourcemac[6];
extern unsigned char sourceip[4];

extern char local_mac[20];
extern char local_ip[16];
extern char local_netmask[16];
extern char local_gateway[16];

char mymac[20];
unsigned char a2x(unsigned char c)
{
	switch(c) {
		case '0'...'9':
			return (c-'0');
		case 'a'...'f':
			return (0xa + (c-'a'));
		case 'A'...'F':
			return (0xa + (c-'A'));
		default:
			return 0;
	}
}
void mac_str2array(unsigned char *mac_array, char *mac_str)
{
	int i;
	for(i = 0; i <6; i++){
		mac_array[i] = (a2x(mac_str[i*3]) << 4) + a2x(mac_str[i*3+1]);
	}
}

void Sendarp(void)
{
	strcpy(mymac, local_mac);
	while(1)
	{
		if(!g_flag){
			sleep(2);
			continue;
		}
		
		int sfd, len, i;
		struct arp_packet ap;
		struct in_addr inaddr_sender, inaddr_receiver;
		struct sockaddr_ll sl;
		
		GetNetInfo();
		
		sfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
		if(-1 == sfd){
			perror("socket");
		}

		memset(&ap, 0, sizeof(ap));

		int ik;
		for(ik=0;ik<6; ik++)
			ap.mac_target[ik] = sourcemac[ik];
		for(ik=0;ik<4; ik++)
			ap.ip_receiver[ik] = sourceip[ik];

		printf("local_mac: [%s]\n", mymac);
		mac_str2array(ap.mac_source,mymac);
		
		ap.ethertype = htons(0x0806);
		ap.hw_type = htons(0x1);
		ap.proto_type = htons(ETH_P_IP);
		ap.mac_addr_len = ETH_ALEN;
		ap.ip_addr_len = 4;
		ap.operation_code = htons(0x2);
		
		for(i = 0; i < sizeof(ap.mac_source); i++)
			ap.mac_sender[i] =  ap.mac_source[i];
		
		inet_aton(local_ip, &inaddr_sender);
		memcpy(&ap.ip_sender, &inaddr_sender, sizeof(inaddr_sender));
		
		for(i = 0; i < sizeof(ap.mac_target); i++)
			ap.mac_receiver[i] =  ap.mac_target[i];
		
		memset(&sl, 0, sizeof(sl));
		sl.sll_family = AF_PACKET;
		sl.sll_ifindex = IFF_BROADCAST;//非常重要
		
		//设备类型：00:所有设备;01:抓拍机；02：识别机；03：车辆检测机；04：高清摄像机；05：抓拍-识别
		//数据包类型：01：在线；02：设置IP
		unsigned char netmask[5];
		inet_aton(local_netmask, &inaddr_receiver);
		memcpy(netmask, (char *)&inaddr_receiver, sizeof(inaddr_receiver));
					
		unsigned char gateway[5];
		inet_aton(local_gateway, &inaddr_receiver);
		memcpy(gateway, &inaddr_receiver, sizeof(inaddr_receiver));	

		#if 1
		unsigned char ipaddr[5];
		inet_aton(local_ip, &inaddr_receiver);
		memcpy(ipaddr, &inaddr_receiver, sizeof(inaddr_receiver));

		sprintf(ap.padding, "zenres,3,%.2x%.2x%.2x%.2x,%.2x%.2x%.2x%.2x,%.2x%.2x%.2x%.2x",
							ipaddr[0],
							ipaddr[1],
							ipaddr[2],
							ipaddr[3],
							netmask[0],
							netmask[1],
							netmask[2],
							netmask[3],
							gateway[0],
							gateway[1],
							gateway[2],
							gateway[3]);
		#else
		sprintf(ap.padding, "zenres,3,%.2x%.2x%.2x%.2x,%.2x%.2x%.2x%.2x",
							netmask[0],
							netmask[1],
							netmask[2],
							netmask[3],
							gateway[0],
							gateway[1],
							gateway[2],
							gateway[3]);
		#endif
			Log0("Send Arp struct: ========>>>>>>>>");
		    Log0("mac_target:     [%.2x:%.2x:%.2x:%.2x:%.2x:%.2x]\n", 
					ap.mac_target[0],
					ap.mac_target[1],
					ap.mac_target[2],
					ap.mac_target[3],
					ap.mac_target[4],
					ap.mac_target[5]);
			Log0("mac_source:     [%.2x:%.2x:%.2x:%.2x:%.2x:%.2x]\n", 
					ap.mac_source[0],
					ap.mac_source[1],
					ap.mac_source[2],
					ap.mac_source[3],
					ap.mac_source[4],
					ap.mac_source[5]);
						
			Log0("ethertype:      [0x%x]\n", ntohs(ap.ethertype));
			Log0("hw_type:        [0x%x]\n", ntohs(ap.hw_type));
			Log0("proto_type:     [0x%x]\n", ntohs(ap.proto_type));
			Log0("mac_addr_len:   [%d]\n", ap.mac_addr_len);
			Log0("ip_addr_len:    [%d]\n", ap.ip_addr_len);
			Log0("operation_code: [0x%x]\n", ntohs(ap.operation_code));
			Log0("mac_sender:     [%.2x:%.2x:%.2x:%.2x:%.2x:%.2x]\n", 
					ap.mac_sender[0],
					ap.mac_sender[1],
					ap.mac_sender[2],
					ap.mac_sender[3],
					ap.mac_sender[4],
					ap.mac_sender[5]);
			Log0("mac_receiver:   [%.2x:%.2x:%.2x:%.2x:%.2x:%.2x]\n", 
					ap.mac_receiver[0],
					ap.mac_receiver[1],
					ap.mac_receiver[2],
					ap.mac_receiver[3],
					ap.mac_receiver[4],
					ap.mac_receiver[5]);
			Log0("ip_sender:      [%d.%d.%d.%d]\n", 
					ap.ip_sender[0],
					ap.ip_sender[1],
					ap.ip_sender[2],
					ap.ip_sender[3]);			
			Log0("ip_receiver:    [%d.%d.%d.%d]\n", 
					ap.ip_receiver[0],
					ap.ip_receiver[1],
					ap.ip_receiver[2],
					ap.ip_receiver[3]);
			Log0("padding:        [%s]\n", ap.padding);		
			Log0("Send Arp struct: <<<<<<<<========");

		len = sendto(sfd, &ap, sizeof(ap), 0, (struct sockaddr*)&sl, sizeof(sl));
		if(-1 == len){
			Log0("sendto faulure");	
		};
		close(sfd);
		g_flag=0x0;
	}

	return ;
}
#define ON_FEDORAR_20 1
//#define ON_CENTOS_72 1
//#define ON_CENTOS_66 1
int sendnetinfo(char (*parray)[20])
{
	char netDetail[512*2] = {'\0'};
	char netBuffer[BUFSIZEMAX] = {'\0'};

	FILE * fp;
#ifdef ON_CENTOS_66
	system("ifconfig eth0 > /tmp/sendnettmp");
#elif ON_CENTOS_72
	system("ifconfig enp2s0 > /tmp/sendnettmp");
#elif ON_FEDORAR_20
	system("ifconfig p4p1 > /tmp/sendnettmp");
#endif
	fp = fopen("/tmp/sendnettmp","r");
	if(fp == NULL)
		return -1;

	while(!feof(fp))
	{
		memset(netBuffer, 0, sizeof(netBuffer));
		fgets(netBuffer,sizeof(netBuffer),fp);
		strcat(netDetail, netBuffer);
	}
	
	fclose(fp);
	system("rm /tmp/sendnettmp");
	
	char *p, *line1, *line2, *pmac, *pip, *pbroadcast, *pnetmask, *pgateway;
	Log0("netDetail is:%s\n",netDetail);

#ifdef ON_CENTOS_66
	line1 = strtok(netDetail,"\n");
	line2 = strtok(NULL,"\n");	
	line2 = strstr(line2,"addr");
	
	pmac = strstr(line1,"HWadd");
	pip = strtok(line2," ");
	pbroadcast = strtok(NULL," ");
	pnetmask = strtok(NULL," ");
	
	strtok(pmac," ");
	pmac = strtok(NULL," ");	//MAC地址
	strtok(pip,":");
	pip = strtok(NULL,":");		//IP地址
	strtok(pbroadcast,":");
	pbroadcast = strtok(NULL,":");		//广播地址
	strtok(pnetmask,":");
	pnetmask = strtok(NULL,":");		//子网掩码
	
#elif ON_CENTOS_72
	int line4;
	strtok(netDetail, "\n");
	line2 = strtok(NULL, "\n");
	strtok(NULL, "\n");
	line4 = strtok(NULL, "\n");
	pmac = strtok(line4," ");
	pmac = strtok(NULL," ");
	pip = strtok(line2," ");
	pip = strtok(NULL," ");
	pnetmask = strtok(NULL," ");
	pnetmask = strtok(NULL," ");
	pbroadcast = strtok(NULL," ");
	pbroadcast = strtok(NULL," ");
#elif ON_FEDORAR_20
	int line4;
	strtok(netDetail, "\n");
	line2 = strtok(NULL, "\n");
	strtok(NULL, "\n");
	line4 = strtok(NULL, "\n");
	pmac = strtok(line4," ");
	pmac = strtok(NULL," ");
	pip = strtok(line2," ");
	pip = strtok(NULL," ");
	pnetmask = strtok(NULL," ");
	pnetmask = strtok(NULL," ");
	pbroadcast = strtok(NULL," ");
	pbroadcast = strtok(NULL," ");
#endif

	strcat(parray[0], pmac);
	strcat(parray[1], pip);
	strcat(parray[2], pnetmask);

	//system("ip route show");
	system("ip route show > /tmp/sendwaynettmp");
	
	fp = fopen("/tmp/sendwaynettmp","r");
	if(fp == NULL)
	{
		return -1;	
	}

	memset(netDetail, '\0', sizeof(netDetail));
	while(!feof(fp))
	{
		memset(netBuffer, '\0', sizeof(netBuffer));
		fgets(netBuffer,sizeof(netBuffer),fp);
		strcat(netDetail, netBuffer);
	}
	
	fclose(fp);
	if(pgateway = strstr(netDetail,"default"))
	{
		pgateway = strstr(pgateway,"via");
		pgateway = strtok(pgateway," ");
		pgateway = strtok(NULL," ");	//默认网关
		strcat(parray[3], pgateway);
	}

	system("rm /tmp/sendwaynettmp");;
	return 1;
}





#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netdb.h>
#include </usr/include/linux/sockios.h>
#include <unistd.h>
#include <string.h>
#include <arpa/inet.h>  //for in_addr
#include <linux/rtnetlink.h>    //for rtnetlink
#include <net/if.h> //for IF_NAMESIZ, route_info
#include <stdlib.h> //for malloc(), free()
#include <string.h> //for strstr(), memset()

#include <string.h>

#define BUFSIZE 8192
#define MAXINTERFACES 16
struct route_info{
 u_int dstAddr;
 u_int srcAddr;
 u_int gateWay;
 char ifName[IF_NAMESIZE];
};
int readNlSock(int sockFd, char *bufPtr, int seqNum, int pId)
{
  struct nlmsghdr *nlHdr;
  int readLen = 0, msgLen = 0;
  do{
    //收到内核的应答
    if((readLen = recv(sockFd, bufPtr, BUFSIZE - msgLen, 0)) < 0)
    {
      perror("SOCK READ: ");
      return -1;
    }

    nlHdr = (struct nlmsghdr *)bufPtr;
    //检查header是否有效
    if((NLMSG_OK(nlHdr, readLen) == 0) || (nlHdr->nlmsg_type == NLMSG_ERROR))
    {
      perror("Error in recieved packet");
      return -1;
    }


    if(nlHdr->nlmsg_type == NLMSG_DONE)
    {
      break;
    }
    else
    {

      bufPtr += readLen;
      msgLen += readLen;
    }


    if((nlHdr->nlmsg_flags & NLM_F_MULTI) == 0)
    {

     break;
    }
  } while((nlHdr->nlmsg_seq != seqNum) || (nlHdr->nlmsg_pid != pId));
  return msgLen;
}
//分析返回的路由信息
void parseRoutes(struct nlmsghdr *nlHdr, struct route_info *rtInfo,char *gateway)
{
  struct rtmsg *rtMsg;
  struct rtattr *rtAttr;
  int rtLen;
  char *tempBuf = NULL;
  struct in_addr dst;
  struct in_addr gate;

  tempBuf = (char *)malloc(100);
  rtMsg = (struct rtmsg *)NLMSG_DATA(nlHdr);
  // If the route is not for AF_INET or does not belong to main routing table
  //then return.
  if((rtMsg->rtm_family != AF_INET) || (rtMsg->rtm_table != RT_TABLE_MAIN))
  return;

  rtAttr = (struct rtattr *)RTM_RTA(rtMsg);
  rtLen = RTM_PAYLOAD(nlHdr);
  for(;RTA_OK(rtAttr,rtLen);rtAttr = RTA_NEXT(rtAttr,rtLen)){
   switch(rtAttr->rta_type) {
   case RTA_OIF:
    if_indextoname(*(int *)RTA_DATA(rtAttr), rtInfo->ifName);
    break;
   case RTA_GATEWAY:
    rtInfo->gateWay = *(u_int *)RTA_DATA(rtAttr);
    break;
   case RTA_PREFSRC:
    rtInfo->srcAddr = *(u_int *)RTA_DATA(rtAttr);
    break;
   case RTA_DST:
    rtInfo->dstAddr = *(u_int *)RTA_DATA(rtAttr);
    break;
   }
  }
  dst.s_addr = rtInfo->dstAddr;
  if (strstr((char *)inet_ntoa(dst), "0.0.0.0"))
  {
    printf("oif:%s\n",rtInfo->ifName);
    gate.s_addr = rtInfo->gateWay;
    sprintf(gateway, (char *)inet_ntoa(gate));
    printf("%s\n",gateway);
    gate.s_addr = rtInfo->srcAddr;
    printf("src:%s\n",(char *)inet_ntoa(gate));
    gate.s_addr = rtInfo->dstAddr;
    printf("dst:%s\n",(char *)inet_ntoa(gate));
  }
  free(tempBuf);
  return;
}

int get_gateway(char *gateway)
{
 struct nlmsghdr *nlMsg;
 struct rtmsg *rtMsg;
 struct route_info *rtInfo;
 char msgBuf[BUFSIZE];

 int sock, len, msgSeq = 0;

 if((sock = socket(PF_NETLINK, SOCK_DGRAM, NETLINK_ROUTE)) < 0)
 {
  perror("Socket Creation: ");
  return -1;
 }


 memset(msgBuf, 0, BUFSIZE);


 nlMsg = (struct nlmsghdr *)msgBuf;
 rtMsg = (struct rtmsg *)NLMSG_DATA(nlMsg);


 nlMsg->nlmsg_len = NLMSG_LENGTH(sizeof(struct rtmsg)); // Length of message.
 nlMsg->nlmsg_type = RTM_GETROUTE; // Get the routes from kernel routing table .

 nlMsg->nlmsg_flags = NLM_F_DUMP | NLM_F_REQUEST; // The message is a request for dump.
 nlMsg->nlmsg_seq = msgSeq++; // Sequence of the message packet.
 nlMsg->nlmsg_pid = getpid(); // PID of process sending the request.


 if(send(sock, nlMsg, nlMsg->nlmsg_len, 0) < 0){
  printf("Write To Socket Failed…\n");
  return -1;
 }


 if((len = readNlSock(sock, msgBuf, msgSeq, getpid())) < 0) {
  printf("Read From Socket Failed…\n");
  return -1;
 }

 rtInfo = (struct route_info *)malloc(sizeof(struct route_info));
 for(;NLMSG_OK(nlMsg,len);nlMsg = NLMSG_NEXT(nlMsg,len)){
  memset(rtInfo, 0, sizeof(struct route_info));
  parseRoutes(nlMsg, rtInfo,gateway);
 }
 free(rtInfo);
 close(sock);
 return 0;
}


void hex2str(char  *pbDest, char *pbSrc, int nLen)
{
	char ddl, ddh;
	int i;

	for (i = 0; i < nLen; i++) {
		ddh = 48 + pbSrc[i] / 16;
		ddl = 48 + pbSrc[i] % 16;
		if (ddh > 57)
			ddh = ddh + 7;
		if (ddl > 57)
			ddl = ddl + 7;
		pbDest[i * 2] = ddh;
		pbDest[i * 2 + 1] = ddl;
	}

	pbDest[nLen * 2] = '\0';
}
void get_ipaddr(char *ipaddr,char *mac,char *mask)
{
	int tmp_fd;
	int ifs;
	char *ip_str;
	char *mask_str;
	char *mac_str;
	struct ifconf ifconfig;
	struct ifreq if_buf[16];
	ifconfig.ifc_len = sizeof(if_buf);
	ifconfig.ifc_buf = (caddr_t) if_buf;

	if ((tmp_fd = socket(AF_INET, SOCK_DGRAM, 0)) == -1) {
		perror("socket(AF_INET, SOCK_DGRAM, 0)");
	}
	if (ioctl(tmp_fd, SIOCGIFCONF, (char *) &ifconfig) == -1) {
		perror("SIOCGIFCONF ioctl");
	}
	ifs = ifconfig.ifc_len / sizeof(struct ifreq);
	while (ifs-- > 0) {
	//	printf("1");
		if (!(ioctl(tmp_fd, SIOCGIFFLAGS, (char *) &if_buf[ifs]))) {

		//	if (buf[if_len].ifr_flags & IFF_UP) {
				if (1) {
							// printf("status: UP\n");

				if (!(ioctl(tmp_fd, SIOCGIFADDR, (char *) &if_buf[ifs]))) {
					ip_str =(char*) inet_ntoa(((struct sockaddr_in*) (&if_buf[ifs].ifr_addr))->sin_addr);
//					printf("addr:%s\n",
//							ip_str =
//									(char*) inet_ntoa(
//											((struct sockaddr_in*) (&buf[ifs].ifr_addr))->sin_addr));
				} else {
					char str[256];
					sprintf(str, "SIOCGIFADDR ioctl %s", if_buf[ifs].ifr_name);
					perror(str);
				}
				if (!strcmp(ip_str, "127.0.0.1")) {
					continue;
				}


			        char ip1[20];
			        char ip2[20];
				strcpy(ip1,ip_str);
				ip1[10]=0;
		//		printf("get ip -----------> %s\n", ip1);
				if (strcmp(ip1, "192.168.1.")) {
					continue;
				}
				printf("get ip -----------> %s\n", ip_str);
				memcpy(ipaddr,ip_str,16);


				if (!(ioctl(tmp_fd, SIOCGIFNETMASK, (char *) &if_buf[ifs]))) {
					mask_str=	(char*) inet_ntoa(
							((struct sockaddr_in*) (&if_buf[ifs].ifr_addr))->sin_addr);
					printf("mask:%s\n",mask_str
						);

					memcpy(mask,mask_str,16);
				} else {
					char str[256];
					sprintf(str, "SIOCGIFADDR ioctl %s",
							if_buf[ifs].ifr_name);
					perror(str);
				}



		        if (!(ioctl(tmp_fd, SIOCGIFBRDADDR, (char *) &if_buf[ifs]))) {
					printf("broad:%s\n",
							(char*) inet_ntoa(
									((struct sockaddr_in*) (&if_buf[ifs].ifr_addr))->sin_addr));
				} else {
					char str[256];
					sprintf(str, "SIOCGIFADDR ioctl %s", if_buf[ifs].ifr_name);
					perror(str);
				}
				if (!(ioctl(tmp_fd, SIOCGIFHWADDR, (char *) &if_buf[ifs]))) {
					printf("mac:%x:%x:%x:%x:%x:%x\n",
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[0],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[1],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[2],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[3],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[4],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[5]);


					mac_str=(char *)if_buf[ifs].ifr_hwaddr.sa_data;
					hex2str(mac,mac_str,6);
					sprintf(mac,"%02x:%02x:%02x:%02x:%02x:%02x",
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[0],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[1],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[2],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[3],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[4],
							(unsigned char) if_buf[ifs].ifr_hwaddr.sa_data[5]);
					//memcpy(mac,mac_str,14);
				} else {
					char str[256];
					sprintf(str, "SIOCGIFHWADDR ioctl %s",
							if_buf[ifs].ifr_name);
					perror(str);
				}


				return 1;




			} else {

				// printf("status: DOWN\n");
			}
		}
	}
}
int GetNetInfo1()
{
//	static char gw[256];
//	static char ip[256];
//	static char mac[256];
//	static char mask[256];
	get_gateway(local_gateway);
	get_ipaddr(local_ip,local_mac,local_netmask);

//	printf("\n====> %s\n",ip);
//	printf("\n====> %s\n",mac);
//	printf("\n====> %s\n",mask);
//	printf("\n====> %s\n",gw);
}

int GetNetInfo(void)
{
	GetNetInfo1();
	return 0;
	char netInfo[4][20];
	memset(netInfo, 0, sizeof(netInfo));
	int re = sendnetinfo(netInfo);
	if(!re)
		return -1;
	strcpy(local_mac, netInfo[0]);
	strcpy(local_ip, netInfo[1]);
	strcpy(local_netmask, netInfo[2]);
	strcpy(local_gateway, netInfo[3]);

	Log0("local_mac:     %s", netInfo[0]);
	Log0("local_ip:      %s", netInfo[1]);
	Log0("local_netmask: %s", netInfo[2]);
	Log0("local_gateway: %s", netInfo[3]);
	
	return 1;	
}
