#include <stdio.h> 
#include <stdlib.h> 
#include <pthread.h> 
#include <unistd.h> 
#include <string.h>

#include "zenarp.h"

int g_flag=0;

unsigned char sourcemac[6];
unsigned char sourceip[4];

char local_mac[20];
char local_ip[20] = {'\0'};
char local_netmask[20] = {'\0'};
char local_gateway[20] = {'\0'};

int main()
{
	int err;
	pthread_t CatcharpID;
	pthread_t SendarpID;

	setLog_init();   
    setLog_display((LogLev)LevDispTRACE);
    setLog_local("log/ip_log",1000, LevDispTRACE);
	
	GetNetInfo();
	while(1){
		err=pthread_create(&SendarpID,NULL, (void *)Sendarp,NULL); 
		if(err != 0){
			Log0("sendarp fail");
			return 0;
		}
		err= pthread_create(&CatcharpID,NULL, (void *)Catcharp,NULL);
		if(err != 0){
			Log0("catcharp fail");
			return 0;
		}
		//printf("LINE: %d FUN: %s\n", __LINE__, __FUNCTION__);
		pthread_join(SendarpID,NULL);
		pthread_join(CatcharpID,NULL);
	}
	return 1;
}
