#include "zenarp.h"

extern int g_flag;
extern unsigned char sourcemac[6];
extern unsigned char sourceip[4];

extern char local_ip[16];
extern char local_netmask[16];
extern char local_gateway[16];
int ptr2ip(char * inbuf, char *oubuf){
	unsigned int i, a, b;
	unsigned int tmp[4]={0};
	for(i=0; i<4; i++){
		a=inbuf[i*2];
		b=inbuf[i*2+1];
		if(a>=0x30 && a<=0x39)
			a -= 0x30;
		else if(a>=0x41&&a<=0x46)
			a -= 0x37;
		else if(a>=0x61&&a<=0x66)
			a -= 0x57;
			
		if(b>=0x30 && b<=0x39)
			b -= 0x30;
		else if(b>=0x41&&b<=0x46)
			b -= 0x37;
		else if(b>=0x61&&b<=0x66)
			b -= 0x57;
		tmp[i]=a*16+b;
	}
	sprintf(oubuf,"%d.%d.%d.%d", tmp[0], tmp[1], tmp[2], tmp[3]);
}

int ptr2ipint(char * inbuf, char *oubuf){
	unsigned int i, a, b;
	unsigned int tmp[4]={0};
	for(i=0; i<4; i++){
		a=inbuf[i*2];
		b=inbuf[i*2+1];
		if(a>=0x30 && a<=0x39)
			a -= 0x30;
		else if(a>=0x41&&a<=0x46)
			a -= 0x37;
		else if(a>=0x61&&a<=0x66)
			a -= 0x57;
			
		if(b>=0x30 && b<=0x39)
			b -= 0x30;
		else if(b>=0x41&&b<=0x46)
			b -= 0x37;
		else if(b>=0x61&&b<=0x66)
			b -= 0x57;
		tmp[i]=a*16+b;
		oubuf[i]=tmp[i];
	}
	//sprintf(oubuf,"%d.%d.%d.%d", tmp[0], tmp[1], tmp[2], tmp[3]);
}

int CreartSerSock()
{
	int sfd = -1;
	struct sockaddr_ll my_etheraddr;
	
	while(1){
		sfd = socket(AF_PACKET, SOCK_RAW, htons(ETH_P_ARP));
		if(-1 == sfd){
			Log0("fail to open listen socket, trying again......\n");
			sleep(3);
		}
		else{
			Log0("success to open listen socket.\n");
			break;
		}
	}
	
	memset(&my_etheraddr, 0, sizeof(my_etheraddr));
	my_etheraddr.sll_family = AF_PACKET;
	my_etheraddr.sll_protocol = htons(ETH_P_ARP);
	my_etheraddr.sll_ifindex = IFF_BROADCAST;

	while(1){
		if(-1 == bind(sfd, (struct sockaddr *)&my_etheraddr, sizeof(my_etheraddr))){
			printf("fail to bind ip to socket, trying again......\n");
			sleep(3);
		}
		else{
			printf("success to bind ip to socket.\n");
			break;
		}
	}

	return sfd;
}

void  Catcharp()
{
	int cl_fd, len;
	int sfd = 0;
	fd_set rd_set;
	struct arp_packet rcvBuffer;
	struct device_node zenith_device;
	
	if(-1 == (sfd = CreartSerSock()))
		return ;
	
	memset(&zenith_device, '\0', sizeof(zenith_device));
	while(1)
	{
		FD_ZERO(&rd_set);
	    FD_SET(sfd, &rd_set);
		if((cl_fd=select(sfd+1,&rd_set,NULL,NULL, NULL))<=0){
			if(errno == EINTR){
				Log0("select continue errno: [%s]", strerror(errno));
				continue;
			}else{
				Log0("select error errno: [%s] need start ser sock", strerror(errno));
				close(sfd);
				if(-1 == (sfd = CreartSerSock()))
					return;
				else
					continue;
			}
		}
		
	    if(FD_ISSET(sfd, &rd_set)){
			memset(&rcvBuffer, 0, sizeof(rcvBuffer));
	    	if((len=recv(sfd, &rcvBuffer, sizeof(rcvBuffer),0))<=0){
				if(errno == EINTR){
					Log0("recv continue errno: [%s]", strerror(errno));
                	continue;
	            }else{
	            	Log0("recv error errno: [%s] need start ser sock", strerror(errno));
					close(sfd);
					if(-1 == (sfd = CreartSerSock()))
						return;
					else
						continue;
	           	}	
			}
		}
		
		if(len!=sizeof(rcvBuffer))
			continue;
		
		if(strstr(rcvBuffer.padding, "zenreq") || strstr(rcvBuffer.padding, "zenset")){
			Log0("Send Arp struct: ========>>>>>>>>");
			Log0("mac_target:   [%.2x:%.2x:%.2x:%.2x:%.2x:%.2x]", 
					rcvBuffer.mac_target[0],
					rcvBuffer.mac_target[1],
					rcvBuffer.mac_target[2],
					rcvBuffer.mac_target[3],
					rcvBuffer.mac_target[4],
					rcvBuffer.mac_target[5]);
			Log0("mac_source:   [%.2x:%.2x:%.2x:%.2x:%.2x:%.2x]", 
					rcvBuffer.mac_source[0],
					rcvBuffer.mac_source[1],
					rcvBuffer.mac_source[2],
					rcvBuffer.mac_source[3],
					rcvBuffer.mac_source[4],
					rcvBuffer.mac_source[5]);
						
			Log0("ethertype:  [0x%x]", ntohs(rcvBuffer.ethertype));
			Log0("hw_type:    [0x%x]", ntohs(rcvBuffer.hw_type));
			Log0("proto_type: [0x%x]", ntohs(rcvBuffer.proto_type));
			Log0("mac_addr_len: [%d]", rcvBuffer.mac_addr_len);
			Log0("ip_addr_len:  [%d]", rcvBuffer.ip_addr_len);
			Log0("operation_code: [0x%x]", ntohs(rcvBuffer.operation_code));
			Log0("mac_sender:   [%.2x:%.2x:%.2x:%.2x:%.2x:%.2x]", 
					rcvBuffer.mac_sender[0],
					rcvBuffer.mac_sender[1],
					rcvBuffer.mac_sender[2],
					rcvBuffer.mac_sender[3],
					rcvBuffer.mac_sender[4],
					rcvBuffer.mac_sender[5]);
			Log0("mac_receiver: [%.2x:%.2x:%.2x:%.2x:%.2x:%.2x]", 
					rcvBuffer.mac_receiver[0],
					rcvBuffer.mac_receiver[1],
					rcvBuffer.mac_receiver[2],
					rcvBuffer.mac_receiver[3],
					rcvBuffer.mac_receiver[4],
					rcvBuffer.mac_receiver[5]);
			Log0("ip_sender:    [%d.%d.%d.%d]", 
					rcvBuffer.ip_sender[0],
					rcvBuffer.ip_sender[1],
					rcvBuffer.ip_sender[2],
					rcvBuffer.ip_sender[3]);			
			Log0("ip_receiver:    [%d.%d.%d.%d]", 
					rcvBuffer.ip_receiver[0],
					rcvBuffer.ip_receiver[1],
					rcvBuffer.ip_receiver[2],
					rcvBuffer.ip_receiver[3]);
			Log0("padding:     [%s]", rcvBuffer.padding);
			Log0("Send Arp struct: >>>>>>>>========");

			//�ɰ汾
			#if 0
			int ik;
			for(ik=0; ik<6; ik++){
				sourcemac[ik] = rcvBuffer.mac_source[ik];
			}
			for(ik=0; ik<4; ik++){
				sourceip[ik] = rcvBuffer.ip_sender[ik];
			}
			#endif
			
			if(strstr(rcvBuffer.padding, "zenset")){
				struct in_addr p;
				unsigned char * ptr;
				char * tmp;
				ptr = (unsigned char *)&p;
				if(tmp = strtok(rcvBuffer.padding, ","))
				{
					if(tmp = strtok(NULL, ","))
						ptr2ip(tmp, zenith_device.ip);
					if(tmp = strtok(NULL, ","))
						ptr2ip(tmp, zenith_device.netmask);
					if(tmp = strtok(NULL, ","))
						ptr2ip(tmp, zenith_device.gateway);			
				}
				Log0("ip:          [%s]", zenith_device.ip);
				Log0("netmask:     [%s]", zenith_device.netmask);
				Log0("gateway:     [%s]", zenith_device.gateway);
				if(strcmp(zenith_device.ip, local_ip)
					|| strcmp(zenith_device.netmask, local_netmask) 
					|| strcmp(zenith_device.gateway, local_gateway)){
					modify_config_ini(zenith_device.ip, zenith_device.netmask,zenith_device.gateway);
					system("sync");
					system("reboot");
				}else
					Log0("net param not changle");
			}else{
				//�°汾
				int ik;
				for(ik=0; ik<6; ik++){
					sourcemac[ik] = rcvBuffer.mac_source[ik];
				}
				char * tmp;
				if(tmp = strtok(rcvBuffer.padding, ",")){
					if(tmp = strtok(NULL, ","))
						ptr2ipint(tmp, sourceip);
				}
				g_flag = 1;
			}
		}
	}
	return ;
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
	
	fd = open("/etc/sysconfig/network-scripts/ifcfg-p4p1", O_RDWR);
	fcntl(fd, F_SETLKW, &flock);
	
	if(fd < 0)
	{
		Log0("line %d err:%s", __LINE__, strerror(errno));
		return -1;
	}
	while(read(fd, buf, sizeof(buf)) > 0)
	{	
		strcat(page, buf);
		memset(buf, '\0', sizeof(buf));
	}
	
	p = strtok(page,"\n");
	printf("p = %s\n",p);
	strcat(tmp, p);
	printf("temp = %s\n",tmp);
	strcat(tmp, "\n");
	while((p = strtok(NULL,"\n")))
	{
		memset(linetmp, '\0', sizeof(linetmp));
		if(strstr(p, "IPADDR"))
		{
			sprintf(linetmp, "IPADDR=%s\n", p2ip);
			strcat(tmp, linetmp);
		}
		else if(strstr(p, "NETMASK"))
		{
			sprintf(linetmp, "NETMASK=%s\n", p2netmask);
			strcat(tmp, linetmp);
		}
		else if(strstr(p, "GATEWAY"))
		{
			sprintf(linetmp, "GATEWAY=%s\n", p2gateway);
			strcat(tmp, linetmp);
		}
		else
		{
			strcat(tmp, p);
			strcat(tmp, "\n");
		}
	}
	close(fd);
	
	fd = open("/etc/sysconfig/network-scripts/ifcfg-p4p1",O_RDWR|O_TRUNC);
	fcntl(fd, F_SETLKW, &flock);
	if(write(fd, tmp, strlen(tmp)) != strlen(tmp)){
		Log0("line %d err:%s", __LINE__, strerror(errno));
	}
	close(fd);
	return 0;
}
