#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <arpa/inet.h>
#include <net/ethernet.h>
#include <netpacket/packet.h>
#include <net/if.h>

#include <stddef.h>
#include <stdbool.h>
#include <stdint.h>

#include <assert.h>

#include <ctype.h>

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


//anger 

#define    SECTIONMAX    50
#define    KEYMAX         50
#define    VALUEMAX       20

int read_profile_string( const char *section, const char *key,char *value, int size,const char *default_value, const char *file);
int read_profile_int( const char *section, const char *key,int default_value, const char *file);
int write_profile_string( const char *section, const char *key,const char *value, const char *file);

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
		assert( i < MAX_FILE_SIZE ); //file too big, you can redefine MAX_FILE_SIZE to fit the big file 
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
*@brief read string in initialization file\n
* retrieves a string from the specified section in an initialization file
*@param section [in] name of the section containing the key name
*@param key [in] name of the key pairs to value 
*@param value [in] pointer to the buffer that receives the retrieved string
*@param size [in] size of result's buffer 
*@param default_value [in] default value of result
*@param file [in] path of the initialization file
*@return 1 : read success; \n 0 : read fail
*/
int read_profile_string( const char *section, const char *key,char *value, 
		 int size, const char *default_value, const char *file)
{
	char buf[MAX_FILE_SIZE]={0};
	int file_size;
	int sec_s,sec_e,key_s,key_e, value_s, value_e;

	//check parameters
	assert(section != NULL && strlen(section));
	assert(key != NULL && strlen(key));
	assert(value != NULL);
	assert(size > 0);
	assert(file !=NULL &&strlen(key));

	if(!load_ini_file(file,buf,&file_size))
	{
		if(default_value!=NULL)
		{
			strncpy(value,default_value, size);
		}
		return 0;
	}

	if(!parse_file(section,key,buf,&sec_s,&sec_e,&key_s,&key_e,&value_s,&value_e))
	{
		if(default_value!=NULL)
		{
			strncpy(value,default_value, size);
		}
		return 0; //not find the key
	}
	else
	{
		int cpcount = value_e -value_s;

		if( size-1 < cpcount)
		{
			cpcount =  size-1;
		}
	
		memset(value, 0, size);
		memcpy(value,buf+value_s, cpcount );
		value[cpcount] = '\0';

		return 1;
	}
}

/**
*@brief read int value in initialization file\n
* retrieves int value from the specified section in an initialization file
*@param section [in] name of the section containing the key name
*@param key [in] name of the key pairs to value 
*@param default_value [in] default value of result
*@param file [in] path of the initialization file
*@return profile int value,if read fail, return default value
*/
int read_profile_int( const char *section, const char *key,int default_value, 
				const char *file)
{
	char value[32] = {0};
	if(!read_profile_string(section,key,value, sizeof(value),NULL,file))
	{
		return default_value;
	}
	else
	{
		return atoi(value);
	}
}

/**
 * @brief write a profile string to a ini file
 * @param section [in] name of the section,can't be NULL and empty string
 * @param key [in] name of the key pairs to value, can't be NULL and empty string
 * @param value [in] profile string value
 * @param file [in] path of ini file
 * @return 1 : success\n 0 : failure
 */
int write_profile_string(const char *section, const char *key,
					const char *value, const char *file)
{
	char buf[MAX_FILE_SIZE]={0};
	char w_buf[MAX_FILE_SIZE]={0};
	int sec_s,sec_e,key_s,key_e, value_s, value_e;
	int value_len = (int)strlen(value);
	int file_size;
	FILE *out;

	//check parameters
	assert(section != NULL && strlen(section));
	assert(key != NULL && strlen(key));
	assert(value != NULL);
	assert(file !=NULL &&strlen(key));

	if(!load_ini_file(file,buf,&file_size))
	{
		sec_s = -1;
	}
	else
	{
		parse_file(section,key,buf,&sec_s,&sec_e,&key_s,&key_e,&value_s,&value_e);
	}

	if( -1 == sec_s)
	{
		if(0==file_size)
		{
			sprintf(w_buf+file_size,"[%s]\n%s=%s\n",section,key,value);
		}
		else
		{
			//not find the section, then add the new section at end of the file
			memcpy(w_buf,buf,file_size);
			sprintf(w_buf+file_size,"\n[%s]\n%s=%s\n",section,key,value);
		}
	}
	else if(-1 == key_s)
	{
		//not find the key, then add the new key=value at end of the section
		memcpy(w_buf,buf,sec_e);
		sprintf(w_buf+sec_e,"%s=%s\n",key,value);
		sprintf(w_buf+sec_e+strlen(key)+strlen(value)+2,buf+sec_e, file_size - sec_e);
	}
	else
	{
		//update value with new value
		memcpy(w_buf,buf,value_s);
		memcpy(w_buf+value_s,value, value_len);
		memcpy(w_buf+value_s+value_len, buf+value_e, file_size - value_e);
	}
	
	out = fopen(file,"w");
	if(NULL == out)
	{
		return 0;
	}
	
	if(-1 == fputs(w_buf,out) )
	{
		fclose(out);
		return 0;
	}

	fclose(out);
	return 1;
}

static int int2string(long lNum,char chWord[]) 
{
	int i=0,j;
	char chTemp;
	if(lNum == 0)
	{
		chWord[i] = '0' ;
		i++;
	}
	else
	{
		while(lNum!=0)/*依次取整数的末位，存入chWord */
		{
			chWord[i] = '0' +lNum%10;/*转换为数字的ASCII码*/
			i++;
			lNum = lNum/10;
		}
		//chWord[i] = '\0';	/*字符串最后一位加'\0'*/
		for(j=0;j<i/2;j++)/*将字符串转置*/
		{
			chTemp = chWord[j];
			chWord[j] = chWord[i-1-j];
			chWord[i-1-j] = chTemp;
		}
	}
	return i;
}

//anger

int main(void)
{

	
	int iSocket_Send = 0;
	int iRet = 0;
	uint16_t port = 12349;
	char buffer[256] = {0};
	struct sockaddr_in addr_server;
	struct sockaddr_in addr_local;
	socklen_t addr_len = 0;
	static int so_broadcast = 1;
	 char pair[20];
	 //char equals[20];
	 char pp[5][10];
	 
	
	iSocket_Send = socket(PF_INET, SOCK_DGRAM, 0);
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
		
		char sever_ip[20] = {'\0'};
		int cnt_ip=0;
		int loop=0;
		int cn=0;
		int ret;
		unsigned char routerip[20]; //anger router ip
	    unsigned char routernetmask[20]; //anger router netmask
	   unsigned char routergateway[20]; //anger router gateway
	    unsigned char canmrerip[20]; //
		 unsigned char udpreserverip[20]; //
	    
		
		#if 0 //udp接收
	   memset(buffer, 0, 256);
	   memset(&addr_server, 0, sizeof(struct sockaddr_in));
	   addr_len = sizeof(struct sockaddr_in);
	   iRet = recvfrom(iSocket_Send, buffer, 256, 0, (struct sockaddr *)&addr_server, &addr_len);
	
		if(iRet<0)
		{
			printf("%s\n","fault");

		}
		else
		{
			printf("\nReceive Message from : [%s:%d].\n", inet_ntoa(addr_server.sin_addr),
			ntohs(addr_server.sin_port));
			printf("\nMessage:[%s]. length:[%d]\n", buffer, iRet);

		}
		#endif
	
	
		read_profile_string("network", "eth0ipaddr", (char*)routerip, 20, "0.0.0.0", DEVCONFIG); //anger
        read_profile_string("network", "netmask", (char*)routernetmask, 20, "0.0.0.0", DEVCONFIG);//anger
	    read_profile_string("network", "gateway", (char*)routergateway, 20, "0.0.0.0", DEVCONFIG);//anger
		read_profile_string("network", "routeripaddr", (char*)canmrerip, 20, "0.0.0.0", DEVCONFIG);//anger
		read_profile_string("network", "eth1ipaddr", (char*)udpreserverip, 20, "0.0.0.0", DEVCONFIG);//anger
		
		
		memcpy(pair,routerip,20);
		printf("pair:%s\n",pair);
		for(cn=0;cn<3;cn++)
		{
			for(loop=0;loop<4;loop++)
			{
				if(pair[loop] == '.')
				{
					memcpy(pp[cn],pair,loop);
					memcpy(pair,pair+loop+1,20);
					pp[cn][loop] ='\0';
					//printf("pp%d:%s\n",cn,pp[cn]);
					//printf("pair:%s\n",pair);
					continue;
				}
				
			}
		}
		memcpy(pp[cn],pair,3);
		//printf("pp:%s:%s:%s:%s\n",pp[0],pp[1],pp[2],pp[3]);
		
		//strcat(sever_ip, pp[0]);
		
		
		netInfoMacaddr(local_mac);
		netInfoIpaddr(local_ip);
		netInfoNetmask(local_netmask);
		netInfoGateway(local_gateway);	
		
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
		//strcat(ip_str, routerip);//anger
		char netmask_str[42] = "zenith,03,01,netmask,";
		strcat(netmask_str, local_netmask);
		//strcat(netmask_str, routernetmask);//anger
		char gateway_str[42] = "zenith,03,01,gateway,";
		strcat(gateway_str, local_gateway);
		//strcat(gateway_str, routergateway);//anger
		
		char routernetwork[150] = "zenith,03,01,mac,"; 
		strcat(routernetwork, local_mac);
		strcat(routernetwork, ",ip,");
		strcat(routernetwork, routerip);
		strcat(routernetwork, ",netmask,");
		strcat(routernetwork, routernetmask);
		strcat(routernetwork, ",gateway,");
		strcat(routernetwork, routergateway);
		
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
			#if 0
			if(strncmp(canmrerip,"10.10.10.12",8) == 0) //anger udp 数据
			{
				
				cnt_ip=0;
				for(cnt_ip=2;cnt_ip<255;cnt_ip++)
				{
					char stringbuf[10]={0};
					int2string(cnt_ip,stringbuf);
					sprintf(sever_ip,"%s%c%s%c%s%c%s",pp[0],'.',pp[1],'.',pp[2],'.',stringbuf);
					printf("sever_ip:%s\n",sever_ip);
					
		
					memset(&addr_local, 0, sizeof(struct sockaddr_in));
					addr_local.sin_family = AF_INET;
					addr_local.sin_port = htons(port);
					addr_local.sin_addr.s_addr = inet_addr(sever_ip); //htonl(INADDR_ANY);//
					//printf("routerip:%s\n",routerip);
					
					ret=sendto(iSocket_Send,routernetwork,strlen(routernetwork),0, (const struct sockaddr *)&addr_local,sizeof(struct sockaddr_in));//向广播地址发布消息
						//printf("routernetwork = %d\n",ret);
					//close(iSocket_Send);
					//sleep(1);
				}
				//指定设置的iP发送数据
				memset(&addr_local, 0, sizeof(struct sockaddr_in));
					addr_local.sin_family = AF_INET;
					addr_local.sin_port = htons(port);
					addr_local.sin_addr.s_addr = inet_addr(udpreserverip); //htonl(INADDR_ANY);//
					printf("udpreserverip:%s\n",udpreserverip);
					
					ret=sendto(iSocket_Send,routernetwork,strlen(routernetwork),0, (const struct sockaddr *)&addr_local,sizeof(struct sockaddr_in));//向广播地址发布消息
						//printf("routernetwork = %d\n",ret);
						printf("I an is router\n");
				
			}
			#endif
			memset(&addr_local, 0, sizeof(struct sockaddr_in));
					addr_local.sin_family = AF_INET;
					addr_local.sin_port = htons(port);
					addr_local.sin_addr.s_addr = inet_addr("192.168.1.255"); //htonl(INADDR_ANY);//
					printf("udpreserverip:%s\n",udpreserverip);
					
					ret=sendto(iSocket_Send,routernetwork,strlen(routernetwork),0, (const struct sockaddr *)&addr_local,sizeof(struct sockaddr_in));//向广播地址发布消息
						//printf("routernetwork = %d\n",ret);
						printf("I an is udp BROADCAST\n");
			
			memset(ap.padding, '\0', sizeof(ap.padding));
			strcpy(ap.padding, netmask_str);
			len = sendto(sfd, &ap, sizeof(ap), 0, (struct sockaddr*)&sl, sizeof(sl));
			printf("i = %d: %s\n", i, ap.padding);
			
			if(-1 == len)
			{
				die("sendto");
			}
			
			//ret=sendto(iSocket_Send,netmask_str,strlen(netmask_str),0,(const struct sockaddr *)&addr_local,sizeof(struct sockaddr_in));//向广播地址发布消息
			//printf("netmask_str = %d\n",ret);
			
			memset(ap.padding, '\0', sizeof(ap.padding));
			strcpy(ap.padding, gateway_str);
			len = sendto(sfd, &ap, sizeof(ap), 0, (struct sockaddr*)&sl, sizeof(sl));
			printf("i = %d: %s\n", i, ap.padding);
			if(-1 == len)
			{
				die("sendto");
			}
			//ret=sendto(iSocket_Send,gateway_str,strlen(gateway_str),0,(const struct sockaddr *)&addr_local,sizeof(struct sockaddr_in));//向广播地址发布消息
			//printf("gateway_str = %d\n",ret);
			
			i++;
			close(sfd);
		}
		
		fflush(stdout);
		
		int shell = -1;	
		shell = is_catcharp_exist();
		if(shell < 1)
		{
			printf("catcharp is dead, let me start it......\n");
			system("/www/zenith/catcharp > /dev/null &");
		}
		else
			printf("catcharp is alive\n");
		
		
		sleep(3);
	}
	
	return 0;
}


void die(const char *pre)
{
	perror(pre);
	exit(1);
}
int netinfo(char (*parray)[20])
{
	char netDetail[512] = {'\0'};
	char netBuffer[80] = {'\0'};
	
	system("ifconfig eth0 > /tmp/nettmp");
	
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
int is_catcharp_exist()
{
	FILE * fp;
	int count = -1;
	char buf[80] = {'\0'};
	char temp[80] = {'\0'};
	
	sleep(8);
	
	while(1)
	{
		system("ps | grep catcharp | grep -v grep | wc -l > /tmp/catchtmp");
		fp = fopen("/tmp/catchtmp","r");
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
	system("rm /tmp/catchtmp");
	count =  atoi(buf);
	return count;
}
