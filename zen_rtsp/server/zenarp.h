#ifndef __ZEN_IPSEARCH_H_
#define __ZEN_IPSEARCH_H_

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
#include <errno.h>
#include "zenlog.h"

//如果只是想让对方断网，那就把mac源都设成MAC_TRICK，
//想截获数据包那就用MAC_SOURCE
//#define MAC_TRICK {0x00, 0x00, 0x00, 0x00, 0x00, 0x01}
//#define MAC_SOURCE {0x00, 0x0c, 0x29, 0xc7, 0x16, 0x33}
//冒充的IP
//#define IP_TRICK "192.168.1.109"
//目标机器的MAC
//#define MAC_TARGET {0xff, 0xff, 0xff, 0xff, 0xff, 0xff}
//目标机器的IP
//#define IP_TARGET "192.168.1.204"
//#define IP_TARGET "192.168.1.204"

#define BUFSIZEMAX    500
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

void  Catcharp();
void Sendarp(void);
int GetNetInfo(void);

#endif
