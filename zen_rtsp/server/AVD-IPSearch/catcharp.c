#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <net/ethernet.h>
#include <net/if_arp.h>
#include <netpacket/packet.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>
#include <ctype.h>
//#include "cgic.h"

#define DEBUG 0

struct arp_packet
{
	//DLC Header
	//接收方mac
	unsigned char mac_target[ETH_ALEN];
	//发送方mac
	unsigned char mac_source[ETH_ALEN];
	//Ethertype - 0x0806是ARP帧的类型值
	unsigned short ethertype;

	//ARP Frame
	//硬件类型 - 以太网类型值0x1
	unsigned short hw_type;
	//上层协议类型 - IP协议(0x0800)
	unsigned short proto_type;
	//MAC地址长度
	unsigned char mac_addr_len;
	//IP地址长度
	unsigned char ip_addr_len;
	//操作码 - 0x1表示ARP请求包,0x2表示应答包
	unsigned short operation_code;
	//发送方mac
	unsigned char mac_sender[ETH_ALEN];
	//发送方ip
	unsigned char ip_sender[4];
	//接收方mac
	unsigned char mac_receiver[ETH_ALEN];
	//接收方ip
	unsigned char ip_receiver[4];
	//填充数据
	unsigned char padding[42];
};
//device_type设备类型：00:所有设备;01:抓拍机；02：识别机；03：车辆检测机；04：高清摄像机；05：抓拍-识别
//operation数据包类型：01：在线；02：设置IP
//pack_type数据类型：ip, netmask, gateway
struct device_arp
{
	char device_type[3];
	char operation[18];
	char pack_type[16];
	char data[16];
};
struct device_node
{
	char device_type[3];
	char mac[18];
	char ip[16];
	char netmask[16];
	char gateway[16];
};

void die(const char*pre);
void print_arp_packet(struct arp_packet ap);
int netInfoMacaddr(char *pstr);
int netInfoIpaddr(char *pstr);
int netInfoNetmask(char *pstr);
int netInfoGateway(char *pstr);
int modify_netconfig(char *p2ip, char *p2netmask, char *p2gateway);
int modify_config_ini(char *p2ip, char *p2netmask, char *p2gateway);

int main(void)
{
	int sfd, i, arp_received_flag = 0;
	char mac_tmp[18];
	struct sockaddr_ll my_etheraddr;
	struct arp_packet rcvBuffer;
	struct device_arp zenith_arp;
	struct device_node zenith_device;
	
	char local_ip[20] = {'\0'};
	char local_netmask[20] = {'\0'};
	char local_gateway[20] = {'\0'};
	
	sleep(8);
	
	while(1)
	{
		if(netInfoIpaddr(local_ip))
			break;
		else
			sleep(3);
	}
	while(1)
	{
		if(netInfoNetmask(local_netmask))
			break;
		else
			sleep(3);
	}
	while(1)
	{
		if(netInfoGateway(local_gateway))
			break;
		else
			sleep(3);
	}
	
	while(1)
	{
		sfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ARP));
		if(-1 == sfd)
		{
			printf("fail to open listen socket, trying again......\n");
			sleep(3);
		}
		else
		{
			printf("success to open listen socket.\n");
			break;
		}
	}

	memset(&my_etheraddr, 0, sizeof(my_etheraddr));
	my_etheraddr.sll_family = AF_PACKET;
	my_etheraddr.sll_protocol = htons(ETH_P_ARP);
	my_etheraddr.sll_ifindex = IFF_BROADCAST;

	while(1)
	{
		if(-1 == bind(sfd, (struct sockaddr *)&my_etheraddr, sizeof(my_etheraddr)))
		{
			printf("fail to bind ip to socket, trying again......\n");
			sleep(3);
		}
		else
		{
			printf("success to bind ip to socket.\n");
			break;
		}
	}
	memset(&zenith_device, '\0', sizeof(zenith_device));
	
	while(1)
	{
		printf("#device_type = %s#\n", zenith_device.device_type);
		printf("#mac = %s#\n", zenith_device.mac);
		printf("#ip = %s#\n", zenith_device.ip);
		printf("#netmask = %s#\n", zenith_device.netmask);
		printf("#gateway = %s#\n", zenith_device.gateway);
		
		
		
		char *p = NULL;
		memset(&zenith_arp, '\0', sizeof(zenith_arp));	//初始化，用于临时存放当前收到的解析数据
		memset(mac_tmp, '\0', sizeof(mac_tmp));		//初始化，用于临时存放将MAC数组转换成的MAC字符串
		
		if(-1 == recv(sfd, &rcvBuffer, sizeof(rcvBuffer), 0))	//开始监听，接收ARP广播包
		{
			printf("didn't receive any arp packet, trying again......\n");
			continue;
		}
		printf("#rcvBuffer.mac_source = %.2x:%.2x:%.2x:%.2x:%.2x:%.2x#\n",rcvBuffer.mac_source[0],rcvBuffer.mac_source[1],rcvBuffer.mac_source[2],rcvBuffer.mac_source[3],rcvBuffer.mac_source[4],rcvBuffer.mac_source[5]);
		printf("#rcvBuffer.padding = %s#\n", rcvBuffer.padding);
		
		if(strstr(rcvBuffer.padding, "zenith"))	//判断是否是zenith发出的包
		{
			
			//解析zenith包中的填充数据
			if(p = strtok(rcvBuffer.padding, ","))
			{
				if(p = strtok(NULL, ","))
					strcpy(zenith_arp.device_type, p);	//解析设备类型
				if(p = strtok(NULL, ","))
					strcpy(zenith_arp.operation, p);	//解析数据包操作类型
				if(p = strtok(NULL, ","))
					strcpy(zenith_arp.pack_type, p);	//解析数据类型
				if(p = strtok(NULL, ","))
					strcpy(zenith_arp.data, p);			//解析数据
			}
			
			
			//printf("%s = %s\n", zenith_arp.pack_type, zenith_arp.data);
			if(strcmp(zenith_arp.operation, "02") == 0)
			{
				if(arp_received_flag == 0)			//判断是否是收到的第一个zenith arp包，是的如下
				{
					arp_received_flag = 1;
					
					sprintf(zenith_device.mac, "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x",rcvBuffer.mac_source[0],rcvBuffer.mac_source[1],rcvBuffer.mac_source[2],rcvBuffer.mac_source[3],rcvBuffer.mac_source[4],rcvBuffer.mac_source[5]);
					for(i = 0; i < sizeof(zenith_device.mac); i++)
					{
						zenith_device.mac[i] = toupper(zenith_device.mac[i]);//记录发送者的MAC地址
					}
					strcat(zenith_device.device_type, zenith_arp.device_type);//记录设备类型
					if(strcmp(zenith_arp.pack_type, "ip") == 0)
						strcpy(zenith_device.ip, zenith_arp.data);
					else if(strcmp(zenith_arp.pack_type, "netmask") == 0)
						strcpy(zenith_device.netmask, zenith_arp.data);
					else if(strcmp(zenith_arp.pack_type, "gateway") == 0)
						strcpy(zenith_device.gateway, zenith_arp.data);
				}
				else if(arp_received_flag)			//判断是否是收到的第一个zenith arp包
				{
					sprintf(mac_tmp, "%.2x:%.2x:%.2x:%.2x:%.2x:%.2x",rcvBuffer.mac_source[0],rcvBuffer.mac_source[1],rcvBuffer.mac_source[2],rcvBuffer.mac_source[3],rcvBuffer.mac_source[4],rcvBuffer.mac_source[5]);
					for(i = 0; i < sizeof(mac_tmp); i++)
					{
						mac_tmp[i] = toupper(mac_tmp[i]);//记录MAC地址，将MAC字符串中的字母转换成大写，以便比较与之前记录的MAC地址比较
					}
					if(strcmp(zenith_device.mac, mac_tmp) == 0)		//判断收到的ARP包是不是之前同一个设备发出的
					{
						if(strcmp(zenith_arp.pack_type, "ip") == 0)
							strcpy(zenith_device.ip, zenith_arp.data);
						else if(strcmp(zenith_arp.pack_type, "netmask") == 0)
							strcpy(zenith_device.netmask, zenith_arp.data);
						else if(strcmp(zenith_arp.pack_type, "gateway") == 0)
							strcpy(zenith_device.gateway, zenith_arp.data);
					}
					else
					{
						arp_received_flag = 0;	//接收到新一个设备发来的IP设置请求，清空当前缓存
						memset(&zenith_device, '\0', sizeof(zenith_device));
						
					}
				}
				
				if(strlen(zenith_device.device_type) && strlen(zenith_device.mac) && strlen(zenith_device.ip) && strlen(zenith_device.netmask) && strlen(zenith_device.gateway))			
				{
					//cgiHeaderContentType("application/json");
					//fprintf(cgiOut, "{\"type\":\"%s\",\"mac\":\"%s\",\"ip\":\"%s\",\"netmask\":\"%s\",\"gateway\":\"%s\"}",zenith_device.device_type,zenith_device.mac,zenith_device.ip,zenith_device.netmask,zenith_device.gateway);
					printf("get ip = %s\n", zenith_device.ip);
					printf("get netmask = %s\n", zenith_device.netmask);
					printf("get gateway = %s\n", zenith_device.gateway);
					
					printf("local ip = %s\n", local_ip);
					printf("local netmask = %s\n", local_netmask);
					printf("local gateway = %s\n", local_gateway);
					
					printf("re = %d\n", strcmp(zenith_device.ip, local_ip));
					printf("re = %d\n", strcmp(zenith_device.netmask, local_netmask));
					printf("re = %d\n", strcmp(zenith_device.gateway, local_gateway));
					if(strcmp(zenith_device.ip, local_ip) == 0 && strcmp(zenith_device.netmask, local_netmask) == 0 && strcmp(zenith_device.gateway, local_gateway) == 0)
					{	
						printf("same netconfig\n");
					}
					else
					{
						printf("set netconfig\n");
						modify_netconfig(zenith_device.ip, zenith_device.netmask,zenith_device.gateway);
						modify_config_ini(zenith_device.ip, zenith_device.netmask,zenith_device.gateway);
						system("reboot");
						printf("netconfig setted\n");
					}
				}
			}
			
		}
		
		int shell = -1;	
		shell = is_sendarp_exist();
		if(shell < 1)
		{
			printf("sendarp is dead, let me start it......\n");
			system("/www/zenith/sendarp > /dev/null &");
		}
		else
			printf("sendarp is alive\n");
		//sleep(3);
	}

	return 0;
}

void die(const char*pre)
{
	perror(pre);
	exit(1);
}
int modify_netconfig(char *p2ip, char *p2netmask, char *p2gateway)
{
	char tmp[512] = {'\0'};
	int fd;
	struct flock flock;	
	flock.l_type = F_WRLCK;
	flock.l_whence = SEEK_SET;
	flock.l_start = 0;
	flock.l_len = 0;
	flock.l_pid = -1;
	fd = open("/etc/init.d/netconfig",O_RDWR|O_TRUNC);
	fcntl(fd, F_SETLKW, &flock);
	if(fd < 0)
	{
		printf("line %d err:%s\n", __LINE__, strerror(errno));
		return -1;
	}
	sprintf(tmp, "ifconfig eth0 %s netmask %s\nroute add default gw %s", p2ip, p2netmask,p2gateway);
	if(write(fd, tmp, strlen(tmp)) != strlen(tmp))
	{
		printf("line %d err:%s\n", __LINE__, strerror(errno));
	}
	close(fd);
	
	return 0;
}
int modify_config_ini(char *p2ip, char *p2netmask, char *p2gateway)
{
	char page[1024] = {'\0'};
	char tmp[1024] = {'\0'};
	char buf[128] = {'\0'};
	char linetmp[128] = {'\0'};
	char *p = NULL;
	int fd;
	struct flock flock;	
	
	
	flock.l_type = F_WRLCK;
	flock.l_whence = SEEK_SET;
	flock.l_start = 0;
	flock.l_len = 0;
	flock.l_pid = -1;
	
	fd = open("/www/config/config.ini",O_RDWR);
	fcntl(fd, F_SETLKW, &flock);
	
	if(fd < 0)
	{
		printf("line %d err:%s\n", __LINE__, strerror(errno));
		return -1;
	}
	while(read(fd, buf, sizeof(buf)) > 0)
	{	
		strcat(page, buf);
		memset(buf, '\0', sizeof(buf));
	}
	
	
	p = strtok(page,"\n");
	strcat(tmp, p);
	strcat(tmp, "\n");
	while((p = strtok(NULL,"\n")))
	{
		
		memset(linetmp, '\0', sizeof(linetmp));
		if(strstr(p, "eth0ipaddr"))
		{
			sprintf(linetmp, "eth0ipaddr=%s\n", p2ip);
			strcat(tmp, linetmp);
		}
		else if(strstr(p, "netmask"))
		{
			sprintf(linetmp, "netmask=%s\n", p2netmask);
			strcat(tmp, linetmp);
		}
		else if(strstr(p, "gateway"))
		{
			sprintf(linetmp, "gateway=%s\n", p2gateway);
			strcat(tmp, linetmp);
		}
		else
		{
			strcat(tmp, p);
			strcat(tmp, "\n");
		}
	}
	close(fd);
	
	fd = open("/www/config/config.ini",O_RDWR|O_TRUNC);
	fcntl(fd, F_SETLKW, &flock);
	if(write(fd, tmp, strlen(tmp)) != strlen(tmp))
	{
		printf("line %d err:%s\n", __LINE__, strerror(errno));
	}
	close(fd);
	return 0;
}
int netinfo(char (*parray)[20])
{

	
	char netDetail[512] = {'\0'};
	char netBuffer[80] = {'\0'};
	
	FILE * fp;
	while(1)
	{
		system("ifconfig eth0 > /tmp/catchnettmp");
		fp = fopen("/tmp/catchnettmp","r");
		if(fp == NULL)
		{
			printf("[ifconfig eth0 > /tmp/catchnettmp]fail to open catchnettmp, trying again......\n");
			//fclose(fp);
			sleep(1);
		}
		else
		{
			break;
		}
	}

	while(!feof(fp))
	{
		memset(netBuffer, '\0', sizeof(netBuffer));
		fgets(netBuffer,sizeof(netBuffer),fp);
		strcat(netDetail, netBuffer);
	}
	fclose(fp);
	system("rm /tmp/catchnettmp");
	
	char *line1, *line2, *pmac, *pip, *pbroadcast, *pnetmask, *pgateway;
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
	
	strcat(parray[0], pmac);
	strcat(parray[1], pip);
	strcat(parray[2], pnetmask);
	
	while(1)
	{
		system("ip route show > /tmp/catchgatewaytmp");
		fp = fopen("/tmp/catchgatewaytmp","r");
		if(fp == NULL)
		{
			printf("[ip route show > catchgatewaytmp]fail to open catchgatewaytmp, trying again......\n");
			//fclose(fp);
			sleep(1);
		}
		else
		{
#if DEBUG
	printf("[ip route show > /tmp/catchgatewaytmp]success to open catchgatewaytmp\n");
#endif
			break;
		}
	}
	memset(netDetail, '\0', sizeof(netDetail));
	while(!feof(fp))
	{
#if DEBUG	
	printf("reading default tmp file......\n");
#endif	
		memset(netBuffer, '\0', sizeof(netBuffer));
		fgets(netBuffer,sizeof(netBuffer),fp);
		strcat(netDetail, netBuffer);
	}
	fclose(fp);
	system("rm /tmp/catchgatewaytmp");
	
#if DEBUG	
	printf("=====gateway===\n");
	printf("%s\n",netDetail);
	printf("===============\n");
#endif		
	
	pgateway = strstr(netDetail,"default");
	pgateway = strstr(pgateway,"via");
	pgateway = strtok(pgateway," ");
	pgateway = strtok(NULL," ");	//默认网关
	strcat(parray[3], pgateway);
	
#if DEBUG	
	printf("debug: netinfo process done\n");
#endif	
	return 1;
}

int netInfoMacaddr(char *pstr)
{
#if DEBUG
	printf("begin to get local mac address......\n");
#endif
	char netInfo[4][20];
	memset(netInfo, '\0', sizeof(netInfo));
	int re = netinfo(netInfo);
	if(!re)
		return -1;
	memset(pstr, '\0', 20);
	strcat(pstr, netInfo[0]);
	return 1;	
}
int netInfoIpaddr(char *pstr)
{
#if DEBUG
	printf("begin to get local ip address......\n");
#endif
	char netInfo[4][20];
	memset(netInfo, '\0', sizeof(netInfo));
	int re = netinfo(netInfo);
	if(!re)
	{
		printf("fail to get ip address!\n");
		return -1;
	}
	memset(pstr, '\0', 20);
	strcat(pstr, netInfo[1]);
	return 1;	
}
int netInfoNetmask(char *pstr)
{
#if DEBUG
	printf("begin to get local netmask.....\n");
#endif
	printf("begin to get netmask......\n");
	char netInfo[4][20];
	memset(netInfo, '\0', sizeof(netInfo));
	int re = netinfo(netInfo);
	if(!re)
	{
		printf("fail to get netmask!\n");
		return -1;
	}
	memset(pstr, '\0', 20);
	strcat(pstr, netInfo[2]);
	return 1;	
}
int netInfoGateway(char *pstr)
{
#if DEBUG
	printf("begin to get local default gateway......\n");
#endif
	printf("begin to get gateway......\n");
	char netInfo[4][20];
	memset(netInfo, '\0', sizeof(netInfo));
	int re = netinfo(netInfo);
	if(!re)
	{
		printf("fail to get gateway!\n");
		return -1;
	}
	memset(pstr, '\0', 20);
	strcat(pstr, netInfo[3]);
	return 1;	
}
unsigned char a2x(unsigned char c)
{
	switch(c) {
		case '0'...'9':
			return (unsigned char)atoi(&c);
		case 'a'...'f':
			return 0xa + (c-'a');
		case 'A'...'F':
			return 0xa + (c-'A');
		default:
			exit(0);
	}
}
void mac_str2array(unsigned char *mac_array, char *mac_str)
{
	int i;
	for(i = 0; i <6; i++)
	{
		mac_array[i] = (a2x(mac_str[i*3]) << 4) + a2x(mac_str[i*3+1]);
		//printf("%d\n", mac_array[i]);
	}
}
int is_sendarp_exist()
{
	FILE * fp;
	int count = -1;
	char buf[80] = {'\0'};
	char temp[80] = {'\0'};
	while(1)
	{
		system("ps | grep sendarp | grep -v grep | wc -l > /tmp/sendtmp");
		fp = fopen("/tmp/sendtmp","r");
		if(fp == NULL)
		{
			sleep(1);
		}
		else
			break;
	}
	while(!feof(fp))
	{
		memset(temp, '\0', sizeof(temp));
		fgets(temp,sizeof(temp),fp);
		strcat(buf, temp);
	}
	fclose(fp);
	system("rm /tmp/sendtmp");
	count =  atoi(buf);
	return count;
}

