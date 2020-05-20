#include <stdio.h>   //printf
#include <arpa/inet.h>   //inet_addr htons
#include <sys/types.h>
#include <sys/socket.h>  //socket 
#include <netinet/in.h>  //sockaddr_in
#include <stdlib.h>  //exit
#include <unistd.h> //close
#include <string.h> //strcat
#include <strings.h> 
 
#include "common.h"
#include "udp_network.h"
#include "csocket.h"

int fd_udp_car_in_out = 0;
int fd_person = 0;

void init_udp()
{
	fd_udp_car_in_out  = UdpCreateSocket(get_random_port());
	fd_person  = UdpCreateSocket(get_random_port());
}


