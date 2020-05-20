#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <net/ethernet.h>
#include <netpacket/packet.h>
#include <net/if.h>
#include <unistd.h>
#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include <assert.h>

#include <ctype.h>

#include <fcntl.h>
#include <errno.h>





//如果只是想让对方断网，那就把mac源都设成MAC_TRICK，
//想截获数据包那就用MAC_SOURCE
#define MAC_TRICK {0x00, 0x00, 0x00, 0x00, 0x00, 0x01}
#define MAC_SOURCE {0x00, 0x0c, 0x29, 0xc7, 0x16, 0x33}
//冒充的IP
#define IP_TRICK "192.168.1.109"
//目标机器的MAC
#define MAC_TARGET {0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
//目标机器的IP
#define IP_TARGET "192.168.1.204"
#define IP_TARGET "192.168.1.204"

#define DEVCONFIG              "/www/config/config.ini"  //anger

#define udp_send_brodcast 0 //发送广播数据
#define udp_reciv_brodcast 1 //接收广播数据
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

void die(const char *pre);
int netinfo(char (*parray)[20]);	//获取本地网络信息，包括MAC地址、IP地址、子网掩码和默认网关
int netInfoMacaddr(char *pstr);		//只获取本地MAC
int netInfoIpaddr(char *pstr);		//只获取本地IP
int netInfoNetmask(char *pstr);		//只获取本地子网掩码
int netInfoGateway(char *pstr);		//只获取默认网关
unsigned char a2x(unsigned char c);	//将MAC地址字符串中的字符转成数值
void mac_str2array(unsigned char *mac_array, char *mac_str);	//将字符串MAC转换成数组（6组）
static int modify_netconfig(char *p2ip, char *p2netmask, char *p2gateway);
 int modify_config_ini(char *p2ip, char *p2netmask, char *p2gateway);
static int write_profile_string(const char *section, const char *key,
                         const char *value, const char *file);
int getSubnetMask();
int getLocalInfo(void);
//set ip
struct device_node
{
	char device_type[3];
	char mac[18];
	char ip[16];
	char netmask[16];
	char gateway[16];
};



int main(void)
{

	static int so_broadcast = 1;
	int iSocket_Send = 0;
	int iSocket_Send_brodcast = 0;
	int iRet = 0;
	uint16_t port = 54321;
	char buffer[256] = {0};
	struct sockaddr_in addr_server;
	struct sockaddr_in addr_local;
	struct sockaddr_in addr_brodcast;
	socklen_t addr_len = 0;
	int ret=0;
	 char pair[20];
	 //char equals[20];
	 char pp[5][10];
	 
	 	char ipaddr[16];
		unsigned short port_r;

		struct device_node zenith_device;

		//getSubnetMask();
		//getLocalInfo();

	iSocket_Send = socket(AF_INET, SOCK_DGRAM, 0);
	if(iSocket_Send == -1)
	{
		printf("%s\n","socket fault");
		return -1;
	}

	
	
	iRet = setsockopt(iSocket_Send, SOL_SOCKET, SO_BROADCAST, &so_broadcast, sizeof(so_broadcast));
	if(iRet == -1)
	{
		perror("setsockopt()");
		return -1;
	}
     
	#if udp_reciv_brodcast
	memset(&addr_local, 0, sizeof(struct sockaddr_in));
	addr_local.sin_family = AF_INET;
	addr_local.sin_port = htons(port);
	addr_local.sin_addr.s_addr = htonl(INADDR_ANY);
	 if(bind(iSocket_Send, (struct sockaddr *)&addr_local, sizeof(struct sockaddr)) == -1)
    {
        return -1;
    }
	#if 1
		memset(&addr_brodcast, 0, sizeof(struct sockaddr_in));
		addr_brodcast.sin_family = AF_INET;
		addr_brodcast.sin_port = htons(port);
		addr_brodcast.sin_addr.s_addr = inet_addr("255.255.255.255");// htonl(INADDR_ANY);//inet_add
		
		iSocket_Send_brodcast = socket(AF_INET, SOCK_DGRAM, 0);
		if(iSocket_Send_brodcast == -1)
		{
			printf("%s\n","socket iSocket_Send_brodcast fault");
			return -1;
		}
		iRet = setsockopt(iSocket_Send_brodcast, SOL_SOCKET, SO_BROADCAST, &so_broadcast, sizeof(so_broadcast));
		if(iRet == -1)
		{
			perror("setsockopt()");
			return -1;
		}
    
	#endif
	#endif
	while(1)
	{
		int sfd, len, i;
		struct arp_packet ap;
		struct in_addr inaddr_sender, inaddr_receiver;
		struct sockaddr_ll sl;
		char local_mac[20] = {'\0'};
		char local_ip[20] = {'\0'};
		char local_netmask[20] = {'\0'};
		char local_gateway[20] = {'\0'};
		
	
	
		#if udp_reciv_brodcast
		
		
		memset(buffer, 0, 256);
		memset(&addr_server, 0, sizeof(struct sockaddr_in));
		addr_len = sizeof(struct sockaddr_in);
		printf("start udp rec\n");
		iRet = recvfrom(iSocket_Send, buffer, 256, 0, (struct sockaddr *)&addr_server, &addr_len);
		if(iRet == -1)
		{
		printf("recvfrom()\n");
		continue;
		}
		else
		{
			printf("\nsucessfMessage:[%s]\n", buffer);
			if(strncmp(buffer,"zenith-request",14) == 0)
			{
				#if 1 //anger 20191219
				char netInfo[4][20];
				memset(netInfo, '\0', sizeof(netInfo));
				int re = netinfo(netInfo);
				if(!re)
					continue;
				memset(local_mac, '\0', 20);
				strcat(local_mac, netInfo[0]);
				memset(local_ip, '\0', 20);
				strcat(local_ip, netInfo[1]);
				memset(local_netmask, '\0', 20);
				strcat(local_netmask, netInfo[2]);
				memset(local_gateway, '\0', 20);
				strcat(local_gateway, netInfo[3]);
				#else
				netInfoMacaddr(local_mac);
				netInfoIpaddr(local_ip);
				netInfoNetmask(local_netmask);
				netInfoGateway(local_gateway);
				#endif
				char routernetwork[150] = "zenith,03,01,mac,"; 
				strcat(routernetwork, local_mac);
				strcat(routernetwork, ",ip,");
				strcat(routernetwork, local_ip);
				strcat(routernetwork, ",netmask,");
				strcat(routernetwork, local_netmask);
				strcat(routernetwork, ",gateway,");
				strcat(routernetwork, local_gateway);
				
				
				addr_server.sin_port =htons(port);
				ret=sendto(iSocket_Send,routernetwork,strlen(routernetwork),0, (const struct sockaddr *)&addr_server,sizeof(struct sockaddr_in));//向接收地址原路返回网络参数
				
				strcpy(ipaddr,(char *)inet_ntoa(addr_server.sin_addr));
				port_r =htons(addr_server.sin_port);
				printf("recvfrom:ipaddr=%s,port=%d\r\n",ipaddr,port_r);
			
				printf("I an is router,I send brodcast\n");
				ret=sendto(iSocket_Send_brodcast,routernetwork,strlen(routernetwork),0, (const struct sockaddr *)&addr_brodcast,sizeof(struct sockaddr_in));//向广播地址发布消息
		
				//ARP广播
				#if 0
				sfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ALL));
				if(-1 == sfd)
				{
					perror("socket");
				}
				
				memset(&ap, 0, sizeof(ap));
				ap.mac_target[0] = 0xff;
				ap.mac_target[1] = 0xff;
				ap.mac_target[2] = 0xff;
				ap.mac_target[3] = 0xff;
				ap.mac_target[4] = 0xff;
				ap.mac_target[5] = 0xff;
				mac_str2array(ap.mac_source, local_mac);
	
				ap.ethertype = htons(0x0806);
				ap.hw_type = htons(0x1);
				ap.proto_type = htons(0x0800);
				ap.mac_addr_len = ETH_ALEN;
				ap.ip_addr_len = 4;
				ap.operation_code = htons(0x1);
				
				for(i = 0; i < sizeof(ap.mac_source); i++)
					ap.mac_sender[i] =  ap.mac_source[i];
				inet_aton(local_ip, &inaddr_sender);
				memcpy(&ap.ip_sender, &inaddr_sender, sizeof(inaddr_sender));
				
				for(i = 0; i < sizeof(ap.mac_target); i++)
					ap.mac_receiver[i] =  ap.mac_target[i];
				
				inet_aton(local_gateway, &inaddr_receiver);
				memcpy(&ap.ip_receiver, &inaddr_receiver, sizeof(inaddr_receiver));
				
				memset(&sl, 0, sizeof(sl));
				sl.sll_family = AF_PACKET;
				//sl.sll_addr = MAC_SOURCE;
				//sl.sll_halen = ETH_ALEN;
				sl.sll_ifindex = IFF_BROADCAST;//非常重要
				
				//设备类型：00:所有设备;01:抓拍机；02：识别机；03：车辆检测机；04：高清摄像机；05：抓拍-识别
				//数据包类型：01：在线；02：设置IP
				char ip_str[42] = "zenith,03,01,ip,";
				strcat(ip_str, local_ip);
				char netmask_str[42] = "zenith,03,01,netmask,";
				strcat(netmask_str, local_netmask);
				char gateway_str[42] = "zenith,03,01,gateway,";
				strcat(gateway_str, local_gateway);
				i = 0; 
				{
					memset(ap.padding, '\0', sizeof(ap.padding));
					strcpy(ap.padding, ip_str);
					len = sendto(sfd, &ap, sizeof(ap), 0, (struct sockaddr*)&sl, sizeof(sl));
					printf("i = %d: %s\n", i, ap.padding);
					if(-1 == len)
					{
						die("sendto");
					}
					
					
					memset(ap.padding, '\0', sizeof(ap.padding));
					strcpy(ap.padding, netmask_str);
					len = sendto(sfd, &ap, sizeof(ap), 0, (struct sockaddr*)&sl, sizeof(sl));
					printf("i = %d: %s\n", i, ap.padding);
					
					if(-1 == len)
					{
						die("sendto");
					}
					
				
					
					memset(ap.padding, '\0', sizeof(ap.padding));
					strcpy(ap.padding, gateway_str);
					len = sendto(sfd, &ap, sizeof(ap), 0, (struct sockaddr*)&sl, sizeof(sl));
					printf("i = %d: %s\n", i, ap.padding);
					if(-1 == len)
					{
						die("sendto");
					}
					
					
					i++;
					close(sfd);
				}
				#endif
				fflush(stdout);
			}
			else if(strncmp(buffer,"zenith-setaddr",14) == 0)
			{
				printf("\nMessage:[%s]. length:[%d]\n", buffer, iRet);//zenith-setaddr,03,08,mac,00:95:19:09:21:48,ip,192.168.1.100,netmask,255.255.255.0,gateway,192.168.1.1
				char *token =NULL;
				//token = strtok(buffer, ",");
				//while( token != NULL ) {
					//printf( "%s\n", token );
					//token = strtok(NULL, ",");
					//}

				if(token = strtok(buffer, ","))//zenith-setaddr
				{
					if(token = strtok(NULL, ","));//03
					if(token = strtok(NULL, ","));//08
					if(token = strtok(NULL, ","));//mac
					if(token = strtok(NULL, ","))//mac地址
					{
						strcpy(zenith_device.mac, token);	//解析mac
						printf("mac:[%s]\r\n", zenith_device.mac);
					}
					if(token = strtok(NULL, ","));//ip
					if(token = strtok(NULL, ","))//ip地址
					{
						strcpy(zenith_device.ip, token);	//解析数据包ip
						printf("ip:[%s]\r\n", zenith_device.ip);
					}
					if(token = strtok(NULL, ","));//netmask
					if(token = strtok(NULL, ","))//netmask地址
					{
						strcpy(zenith_device.netmask, token);	//解析数据子网掩码
						printf("netmask:[%s]\r\n", zenith_device.netmask);
					}
					if(token = strtok(NULL, ","));//gateway
					if(token = strtok(NULL, ","))//gateway地址
					{
						strcpy(zenith_device.gateway, token);			//解析数据网关
						printf("gateway:[%s]\r\n", zenith_device.gateway);
					}
				}
				if( strlen(zenith_device.ip) && strlen(zenith_device.netmask) && strlen(zenith_device.gateway))			
				{
					
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
						//system("reboot");
						printf("netconfig setted\n");
					}
				}
				
			
			}
			//printf("\nReceive Message from : [%s:%d].\n", inet_ntoa(addr_server.sin_addr), ntohs(addr_server.sin_port));
			//printf("\nMessage:[%s]. length:[%d]\n", buffer, iRet);
		}
		#endif
		
		#if udp_send_brodcast
		
		netInfoMacaddr(local_mac);
		netInfoIpaddr(local_ip);
		netInfoNetmask(local_netmask);
		netInfoGateway(local_gateway);	
		
		char routernetwork[150] = "zenith,03,01,mac,"; 
		strcat(routernetwork, local_mac);
		strcat(routernetwork, ",ip,");
		strcat(routernetwork, local_ip);
		strcat(routernetwork, ",netmask,");
		strcat(routernetwork, local_netmask);
		strcat(routernetwork, ",gateway,");
		strcat(routernetwork, local_gateway);
		
		
		//指定设置的iP发送数据
		memset(&addr_local, 0, sizeof(struct sockaddr_in));
		addr_local.sin_family = AF_INET;
		addr_local.sin_port = htons(port);
		addr_local.sin_addr.s_addr = inet_addr("255.255.255.255");// htonl(INADDR_ANY);//inet_addr("192.168.1.35");

		printf("I an is router\n");
		ret=sendto(iSocket_Send,routernetwork,strlen(routernetwork),0, (const struct sockaddr *)&addr_local,sizeof(struct sockaddr_in));//向广播地址发布消息
		//printf("routernetwork = %d\n",ret);
		printf("I an is router\n");
			#endif
			
		sleep(1);
	}
	
	return 0;
}


static int modify_netconfig(char *p2ip, char *p2netmask, char *p2gateway)
{
	   int iret = 0;

    char buf[100] = {0};
    char ret[20] = {0};


    char *file     =  "/etc/network/interfaces";
 
    if (p2ip[0] != 0) {
        sprintf(buf, "sed -i \"s/^address .*$/address %s/g\"  %s", p2ip, file);
        system(buf);
        memset(buf, 0, 100);
    }

    if (p2netmask[0] != 0) {
        sprintf(buf, "sed -i \"s/^netmask .*$/netmask %s/g\"  %s", p2netmask, file);
        system(buf);
        memset(buf, 0, 100);
    }

    if (p2gateway[0] != 0) {
        sprintf(buf, "sed -i \"s/^gateway .*$/gateway %s/g\"  %s", p2gateway, file);
        system(buf);
        memset(buf, 0, 100);
    }


    memset(buf, 0, 100);
    sprintf(buf, "/etc/init.d/networking restart");
    system(buf);

  if (p2ip[0] != 0 && p2netmask[0] != 0) {
        sprintf(buf, "ifconfig enp1s0 %s netmask %s ", p2ip, p2netmask);
        system(buf);
    }

    if (p2gateway[0] != 0) {
        sprintf(buf, "route add default gw %s", p2gateway);
        system(buf);
    }
	
	return 0;
}


#define MAX_FILE_SIZE 1024*16
#define LEFT_BRACE '['
#define RIGHT_BRACE ']'


static int load_ini_file(const char *file, char *buf,int *file_size)
{
	FILE *in = NULL;
	int i=0;
	*file_size =0;

	 assert(file !=NULL);
	assert(buf !=NULL);

	in = fopen(file,"r");
	if( NULL == in) {
		return 0;
	}

	buf[i]=fgetc(in);
	
	//load initialization file
	while( buf[i]!= (char)EOF) {
		i++;
		//assert( i < MAX_FILE_SIZE ); //file too big, you can redefine MAX_FILE_SIZE to fit the big file 
		buf[i]=fgetc(in);
	}
	
	buf[i]='\0';
	*file_size = i;

	fclose(in);
	return 1;
}

static int newline(char c)
{
	return ('\n' == c ||  '\r' == c )? 1 : 0;
}
static int end_of_string(char c)
{
	return '\0'==c? 1 : 0;
}
static int left_barce(char c)
{
	return LEFT_BRACE == c? 1 : 0;
}
static int isright_brace(char c )
{
	return RIGHT_BRACE == c? 1 : 0;
}
static int parse_file(const char *section, const char *key, const char *buf,int *sec_s,int *sec_e,
					  int *key_s,int *key_e, int *value_s, int *value_e)
{
	const char *p = buf;
	int i=0;

	assert(buf!=NULL);
	assert(section != NULL && strlen(section));
	assert(key != NULL && strlen(key));

	*sec_e = *sec_s = *key_e = *key_s = *value_s = *value_e = -1;

	while( !end_of_string(p[i]) ) {
		//find the section
		if( ( 0==i ||  newline(p[i-1]) ) && left_barce(p[i]) )
		{
			int section_start=i+1;

			//find the ']'
			do {
				i++;
			} while( !isright_brace(p[i]) && !end_of_string(p[i]));

			if( 0 == strncmp(p+section_start,section, i-section_start)) {
				int newline_start=0;

				i++;

				//Skip over space char after ']'
				while(isspace(p[i])) {
					i++;
				}

				//find the section
				*sec_s = section_start;
				*sec_e = i;

				while( ! (newline(p[i-1]) && left_barce(p[i])) 
				&& !end_of_string(p[i]) ) {
					int j=0;
					//get a new line
					newline_start = i;

					while( !newline(p[i]) &&  !end_of_string(p[i]) ) {
						i++;
					}
					
					//now i  is equal to end of the line
					j = newline_start;

					if(';' != p[j]) //skip over comment
					{
						while(j < i && p[j]!='=') {
							j++;
							if('=' == p[j]) {
								if(strncmp(key,p+newline_start,j-newline_start)==0)
								{
									//find the key ok
									*key_s = newline_start;
									*key_e = j-1;

									*value_s = j+1;
									*value_e = i;

									return 1;
								}
							}
						}
					}

					i++;
				}
			}
		}
		else
		{
			i++;
		}
	}
	return 0;
}

/**
 * @brief write a profile string to a ini file
 * @param section [in] name of the section,can't be NULL and empty string
 * @param key [in] name of the key pairs to value, can't be NULL and empty string
 * @param value [in] profile string value
 * @param file [in] path of ini file
 * @return 1 : success\n 0 : failure
 */
static int write_profile_string(const char *section, const char *key,
                         const char *value, const char *file)
{
	char buf[MAX_FILE_SIZE] = {0};
	char w_buf[MAX_FILE_SIZE] = {0};
	int sec_s, sec_e, key_s, key_e, value_s, value_e;
	int value_len = (int)strlen(value);
	int file_size;
	FILE *out;

	//check parameters
	assert(section != NULL && strlen(section));
	assert(key != NULL && strlen(key));
	assert(value != NULL);
	assert(file != NULL && strlen(key));

	if (!load_ini_file(file, buf, &file_size))
	{
		sec_s = -1;
	}
	else
	{
		parse_file(section, key, buf, &sec_s, &sec_e, &key_s, &key_e, &value_s, &value_e);
	}

	if ( -1 == sec_s)
	{
		if (0 == file_size)
		{
			sprintf(w_buf + file_size, "[%s]\n%s=%s\n", section, key, value);
		}
		else
		{
			//not find the section, then add the new section at end of the file
			memcpy(w_buf, buf, file_size);
			sprintf(w_buf + file_size, "\n[%s]\n%s=%s\n", section, key, value);
		}
	}
	else if (-1 == key_s)
	{
		//not find the key, then add the new key=value at end of the section
		memcpy(w_buf, buf, sec_e);
		sprintf(w_buf + sec_e, "%s=%s\n", key, value);
		sprintf(w_buf + sec_e + strlen(key) + strlen(value) + 2, buf + sec_e, file_size - sec_e);
	}
	else
	{
		//update value with new value
		memcpy(w_buf, buf, value_s);
		memcpy(w_buf + value_s, value, value_len);
		memcpy(w_buf + value_s + value_len, buf + value_e, file_size - value_e);
	}

	out = fopen(file, "w+");
	if (NULL == out)
	{
		return 0;
	}

	if (-1 == fputs(w_buf, out) )
	{
		fclose(out);
		return 0;
	}

	fclose(out);
	return 1;
}

 int modify_config_ini(char *p2ip, char *p2netmask, char *p2gateway)
{
	write_profile_string( "NETWORK", "strIpaddr", (char*)p2ip, "/home/zen/config/network.ini");

	write_profile_string( "NETWORK", "strNetmask", (char*)p2netmask, "/home/zen/config/network.ini");
	write_profile_string( "NETWORK", "strGateway", (char*)p2gateway, "/home/zen/config/network.ini");
	
	
	return 0;
}

void die(const char *pre)
{
	perror(pre);
	exit(1);
}
int netinfo(char (*parray)[20])
{
	char netDetail[1024*4] = {'\0'};
	char netBuffer[1024] = {'\0'};
 	 char interinfo[20]={0};
	 #if 1//anger 20191219
	 char ifconfigInfo[512] = "ifconfig "; 
	getSubnetMask(interinfo);
	strcat(ifconfigInfo, interinfo);
	strcat(ifconfigInfo, " > /tmp/nettmp");
	//printf("ifconfigInfo:%s\r\n",ifconfigInfo);
	system(ifconfigInfo);
	printf("system(ifconfigInfo)\r\n");
	#else
	system("ifconfig enp1s0 > /tmp/nettmp");
	#endif
	FILE * fp;
	fp = fopen("/tmp/nettmp","r");
	if(fp == NULL)
		return -1;
	while(!feof(fp))
	{
		memset(netBuffer, '\0', sizeof(netBuffer));
		fgets(netBuffer,sizeof(netBuffer),fp);
		strcat(netDetail, netBuffer);
	}
	fclose(fp);
	printf("fclose(fp)\r\n");
	
	char *p, *line1, *line2, *pmac, *pip, *pbroadcast, *pnetmask, *pgateway;
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
	
	system("ip route show > /tmp/nettmp");
	fp = fopen("/tmp/nettmp","r");
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
	
	system("rm /tmp/nettmp");
	/**/
	return 1;
}

int netInfoMacaddr(char *pstr)
{
	printf("\netInfoMacaddr:[%d]\n",88);
	char netInfo[4][20];
	memset(netInfo, '\0', sizeof(netInfo));
	int re = netinfo(netInfo);
	printf("\netInfoMacaddr:[%d]\n", re);
	if(!re)
		return -1;
	memset(pstr, '\0', 20);
	strcat(pstr, netInfo[0]);
	
	return 1;	
}
int netInfoIpaddr(char *pstr)
{
	
	char netInfo[4][20];
	memset(netInfo, '\0', sizeof(netInfo));
	int re = netinfo(netInfo);
	if(!re)
		return -1;
	memset(pstr, '\0', 20);
	strcat(pstr, netInfo[1]);
	return 1;	
}
int netInfoNetmask(char *pstr)
{
	char netInfo[4][20];
	memset(netInfo, '\0', sizeof(netInfo));
	int re = netinfo(netInfo);
	if(!re)
		return -1;
	memset(pstr, '\0', 20);
	strcat(pstr, netInfo[2]);
	return 1;	
}
int netInfoGateway(char *pstr)
{
	char netInfo[4][20];
	memset(netInfo, '\0', sizeof(netInfo));
	int re = netinfo(netInfo);
	if(!re)
		return -1;
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

//遍历网卡，不获取物理地址
#include <ifaddrs.h>  
int getSubnetMask(char *info)
{
	char interfaceNameInfo[10][20];
	int i=0;
	struct sockaddr_in *sin = NULL;
	struct ifaddrs *ifa = NULL, *ifList;
	if (getifaddrs(&ifList) < 0)
	{
		return -1;
	}
	for (ifa = ifList; ifa != NULL; ifa = ifa->ifa_next)
	{
		if(ifa->ifa_addr->sa_family == AF_INET)
		{
			printf(">>> interfaceName: %s\r\n", ifa->ifa_name);
			if (strcmp( ifa->ifa_name, "lo") == 0)
				continue;
			strcpy(interfaceNameInfo[i],ifa->ifa_name);
			//printf(">>> interfaceNameprecess: %s\r\n", interfaceNameInfo[i]);
			i++;
			if(i>10)
				i=0;
			
			sin = (struct sockaddr_in *)ifa->ifa_addr;
			printf(">>> ipAddress: %s\r\n", inet_ntoa(sin->sin_addr));
			sin = (struct sockaddr_in *)ifa->ifa_dstaddr;
			printf(">>> broadcast: %s\r\n", inet_ntoa(sin->sin_addr));
			sin = (struct sockaddr_in *)ifa->ifa_netmask;
			printf(">>> subnetMask: %s\r\n", inet_ntoa(sin->sin_addr));
		}
	}
	freeifaddrs(ifList);
	strcpy(info,interfaceNameInfo[0]);
	return 0;
}

//遍历网卡，获取物理地址
//果不需要获取MAC地址，那么使用getifaddrs函数来获取更加方便与简洁。
#include <sys/ioctl.h>
int getLocalInfo(void)
  {
      int fd;
     int interfaceNum = 0;
     struct ifreq buf[16];
     struct ifconf ifc;
     struct ifreq ifrcopy;
     char mac[16] = {0};
    char ip[32] = {0};
     char broadAddr[32] = {0};
     char subnetMask[32] = {0};
 
     if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) < 0)
     {
        perror("socket");
 
        close(fd);
         return -1;
     }
 
    ifc.ifc_len = sizeof(buf);
     ifc.ifc_buf = (caddr_t)buf;
    if (!ioctl(fd, SIOCGIFCONF, (char *)&ifc))
     {
         interfaceNum = ifc.ifc_len / sizeof(struct ifreq);
        printf("interface num = %d\r\n", interfaceNum);
         while (interfaceNum-- > 0)
         {
             printf("ndevice name: %s\r\n", buf[interfaceNum].ifr_name);
 
             //ignore the interface that not up or not runing  
            ifrcopy = buf[interfaceNum];
            if (ioctl(fd, SIOCGIFFLAGS, &ifrcopy))
             {
                 printf("ioctl: %s [%s:%d]\r\n", strerror(errno), __FILE__, __LINE__);
                 close(fd);
                  return -1;
              }
 
              //get the mac of this interface  
              if (!ioctl(fd, SIOCGIFHWADDR, (char *)(&buf[interfaceNum])))
             {
                 memset(mac, 0, sizeof(mac));
                 snprintf(mac, sizeof(mac), "%02x%02x%02x%02x%02x%02x",
                     (unsigned char)buf[interfaceNum].ifr_hwaddr.sa_data[0],
                     (unsigned char)buf[interfaceNum].ifr_hwaddr.sa_data[1],
                     (unsigned char)buf[interfaceNum].ifr_hwaddr.sa_data[2],
 
                     (unsigned char)buf[interfaceNum].ifr_hwaddr.sa_data[3],
                     (unsigned char)buf[interfaceNum].ifr_hwaddr.sa_data[4],
                     (unsigned char)buf[interfaceNum].ifr_hwaddr.sa_data[5]);
                 printf("device mac: %s\r\n", mac);
             }
              else
             {
                 printf("ioctl: %s [%s:%d]\r\n", strerror(errno), __FILE__, __LINE__);
                 close(fd);
                 return -1;
             }
 
           //get the IP of this interface  

             if (!ioctl(fd, SIOCGIFADDR, (char *)&buf[interfaceNum]))
              {
                 snprintf(ip, sizeof(ip), "%s",
                     (char *)inet_ntoa(((struct sockaddr_in *)&(buf[interfaceNum].ifr_addr))->sin_addr));
                 printf("device ip: %s\r\n", ip);
            }
             else
             {
                 printf("ioctl: %s [%s:%d]\r\n", strerror(errno), __FILE__, __LINE__);
                 close(fd);
                 return -1;
             }
 
             //get the broad address of this interface  
 
             if (!ioctl(fd, SIOCGIFBRDADDR, &buf[interfaceNum]))
              {
                 snprintf(broadAddr, sizeof(broadAddr), "%s",
                     (char *)inet_ntoa(((struct sockaddr_in *)&(buf[interfaceNum].ifr_broadaddr))->sin_addr));
                 printf("device broadAddr: %s\r\n", broadAddr);
           }
             else
              {
                 printf("ioctl: %s [%s:%d]\r\n", strerror(errno), __FILE__, __LINE__);
                 close(fd);
                 return -1;
             }
 
             //get the subnet mask of this interface  
             if (!ioctl(fd, SIOCGIFNETMASK, &buf[interfaceNum]))
             {
                 snprintf(subnetMask, sizeof(subnetMask), "%s",
                     (char *)inet_ntoa(((struct sockaddr_in *)&(buf[interfaceNum].ifr_netmask))->sin_addr));
                 printf("device subnetMask: %s\r\n", subnetMask);
             }
             else
             {
                 printf("ioctl: %s [%s:%d]\r\n", strerror(errno), __FILE__, __LINE__);
                 close(fd);
                 return -1;
 
             }
         }
     }
     else
     {
         printf("ioctl: %s [%s:%d]\r\n", strerror(errno), __FILE__, __LINE__);
         close(fd);
         return -1;
     }
   
     close(fd);
 
     return 0;
 }

