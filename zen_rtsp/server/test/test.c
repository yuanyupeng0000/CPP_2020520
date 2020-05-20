

#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <net/if.h>
#include <arpa/inet.h>
#include <netdb.h>
#include </usr/include/linux/sockios.h>
#include <unistd.h>
#include <string.h>
//main()
//{
//	int i = 0;
//	int sockfd;
//	struct ifconf ifconf;
//	unsigned char buf[512];
//	struct ifreq *ifreq;    //初始化ifconf
//	ifconf.ifc_len = 512;
//	ifconf.ifc_buf = buf;
//	if ((sockfd = socket(AF_INET, SOCK_DGRAM, 0)) < 0) {
//		perror("socket");
//		exit(1);
//	}
//	ioctl(sockfd, SIOCGIFCONF, &ifconf);    //获取所有接口信息
//
//	//接下来一个一个的获取IP地址
//	ifreq = (struct ifreq*) buf;
//	for (i = (ifconf.ifc_len / sizeof(struct ifreq)); i > 0; i--) {
//		if (ifreq->ifr_flags == AF_INET) {
//			//for ipv4
//			printf("name = [%s]\n", ifreq->ifr_name);
//			printf("local addr = [%s]\n",
//					inet_ntoa(
//							((struct sockaddr_in*) &(ifreq->ifr_addr))->sin_addr));
//			ifreq++;
//
//
//		}
//		//return 0;
//	}
//
//
//
//
//
//	return 0;
//}

#define MAXINTERFACES 16

int fd;
int if_len;
struct ifreq buf[MAXINTERFACES];
struct ifconf ifc;
void get_ipaddr(char *ipaddr)
{
	int tmp_fd;
	int ifs;
	char *ip_str;
	struct ifconf ifconfig;
	struct ifreq if_buf[16];
	ifconfig.ifc_len = sizeof(buf);
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
				printf("get ip %s\n", ip_str);

			} else {

				// printf("status: DOWN\n");
			}
		}
	}
}
//void get_mask(char *ipaddr)
//{
//
//}
//
//void get_mac(char *ipaddr)
//{
//
//}
int main(argc, argv)
{
	static char test[256];
	get_ipaddr(test);
	return 0;

	int flg=0;

    if ((fd = socket(AF_INET, SOCK_DGRAM, 0)) == -1)
    {
        perror("socket(AF_INET, SOCK_DGRAM, 0)");
        return -1;
    }


    ifc.ifc_len = sizeof(buf);
    ifc.ifc_buf = (caddr_t) buf;


    if (ioctl(fd, SIOCGIFCONF, (char *) &ifc) == -1)
    {
        perror("SIOCGIFCONF ioctl");
        return -1;
    }

    if_len = ifc.ifc_len / sizeof(struct ifreq);
    printf("num:%d\n", if_len);

    while (if_len-->0)
    {
      //  printf("接口：%s/n", buf[if_len].ifr_name);


        if (!(ioctl(fd, SIOCGIFFLAGS, (char *) &buf[if_len])))
        {

            if (buf[if_len].ifr_flags & IFF_UP)
            {
               // printf("status: UP\n");
                flg=1;
            }
            else
            {
            	flg=0;
               // printf("status: DOWN\n");
            }
        }
        else
        {
            char str[256];
            sprintf(str, "SIOCGIFFLAGS ioctl %s", buf[if_len].ifr_name);
            perror(str);
        }


        char *ip_str;
        if (!(ioctl(fd, SIOCGIFADDR, (char *) &buf[if_len])))
        {
            printf("addr:%s\n",
            		ip_str=(char*)inet_ntoa(((struct sockaddr_in*) (&buf[if_len].ifr_addr))->sin_addr));
        }
        else
        {
            char str[256];
            sprintf(str, "SIOCGIFADDR ioctl %s", buf[if_len].ifr_name);
            perror(str);
        }
        if(!strcmp(ip_str,"127.0.0.1")){
        	continue;
        }

        if (!(ioctl(fd, SIOCGIFNETMASK, (char *) &buf[if_len])))
        {
            printf("mask:%s\n",
                    (char*)inet_ntoa(((struct sockaddr_in*) (&buf[if_len].ifr_addr))->sin_addr));
        }
        else
        {
            char str[256];
            sprintf(str, "SIOCGIFADDR ioctl %s", buf[if_len].ifr_name);
            perror(str);
        }


        if (!(ioctl(fd, SIOCGIFBRDADDR, (char *) &buf[if_len])))
        {
            printf("broad:%s\n",
                    (char*)inet_ntoa(((struct sockaddr_in*) (&buf[if_len].ifr_addr))->sin_addr));
        }
        else
        {
            char str[256];
            sprintf(str, "SIOCGIFADDR ioctl %s", buf[if_len].ifr_name);
            perror(str);
        }


        if (!(ioctl(fd, SIOCGIFHWADDR, (char *) &buf[if_len])))
        {
            printf("mac:%x:%x:%x:%x:%x:%x\n",
                    (unsigned char) buf[if_len].ifr_hwaddr.sa_data[0],
                    (unsigned char) buf[if_len].ifr_hwaddr.sa_data[1],
                    (unsigned char) buf[if_len].ifr_hwaddr.sa_data[2],
                    (unsigned char) buf[if_len].ifr_hwaddr.sa_data[3],
                    (unsigned char) buf[if_len].ifr_hwaddr.sa_data[4],
                    (unsigned char) buf[if_len].ifr_hwaddr.sa_data[5]);
        }
        else
        {
            char str[256];
            sprintf(str, "SIOCGIFHWADDR ioctl %s", buf[if_len].ifr_name);
            perror(str);
        }
    }//–while end

    //关闭socket
    close(fd);
    return 0;
}



//#include <sys/ioctl.h>
//#include <sys/types.h>
//#include <sys/socket.h>
//#include <netinet/in.h>
//#include <arpa/inet.h>
//#include <net/if.h>
//#include <error.h>
//#include <net/route.h>
//
//int SetIfAddr(char *ifname, char *Ipaddr, char *mask,char *gateway)
//{
//    int fd;
//    int rc;
//    struct ifreq ifr;
//    struct sockaddr_in *sin;
//    struct rtentry  rt;
//
//     fd = socket(AF_INET, SOCK_DGRAM, 0);
////    if(fd < 0)
////    {
////            perror("socket   error");
////            return -1;
////    }
////    memset(&ifr,0,sizeof(ifr));
////    strcpy(ifr.ifr_name,ifname);
////    sin = (struct sockaddr_in*)&ifr.ifr_addr;
////    sin->sin_family = AF_INET;
////
////    //ipaddr
////    if(inet_aton(Ipaddr,&(sin->sin_addr)) < 0)
////    {
////        perror("inet_aton   error");
////        return -2;
////    }
////
////    if(ioctl(fd,SIOCSIFADDR,&ifr) < 0)
////    {
////        perror("ioctl   SIOCSIFADDR   error");
////        return -3;
////    }
////
////    //netmask
////    if(inet_aton(mask,&(sin->sin_addr)) < 0)
////    {
////        perror("inet_pton   error");
////        return -4;
////    }
////    if(ioctl(fd, SIOCSIFNETMASK, &ifr) < 0)
////    {
////        perror("ioctl");
////        return -5;
////    }
//
//    //gateway
////    memset(&rt, 0, sizeof(struct rtentry));
////    memset(sin, 0, sizeof(struct sockaddr_in));
////    sin->sin_family = AF_INET;
////    sin->sin_port = 0;
////    if(inet_aton(gateway, &sin->sin_addr)<0)
////    {
////       printf ( "inet_aton error\n" );
////    }
////    memcpy ( &rt.rt_gateway, sin, sizeof(struct sockaddr_in));
//
//
//
//
//
//
//
//
//
//     ((struct sockaddr_in *)&rt.rt_dst)->sin_family=AF_INET;
//    ((struct sockaddr_in *)&rt.rt_genmask)->sin_family=AF_INET;
//     rt.rt_flags = RTF_GATEWAY;
//    if (ioctl(fd, SIOCRTMSG, &rt)<0)
//    {
//     //   zError( "ioctl(SIOCADDRT) error in set_default_route\n");
//        close(fd);
//        return -1;
//    }
//    long *  gateway_addr=(long*)((rt.rt_gateway.sa_data));
//    printf("%x\n", *gateway_addr);
//    close(fd);
//    return rc;
//}
//void main()
//{
//static	char ipaddr[100];
//static	char gateway[100];
//static	char mask[100];
//
//		SetIfAddr("eth0",ipaddr,mask,gateway);
//}
