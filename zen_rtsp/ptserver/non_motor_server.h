#ifndef __NON_MOTOR_SERVER_H__ 
#define __NON_MOTOR_SERVER_H__

#include <arpa/inet.h>
#include <netinet/in.h>

// #pragma pack(push)
// #pragma pack(1)

typedef struct img_list_node
{
    char img_id[21];
    struct sockaddr_in cli_fd; //
}img_list_node_t;


// #pragma pack(pop)
////////function////////////
int  get_non_motor_fd();
void init_non_motor_list();
void *get_non_motor_queue();
void *non_motor_udp_server(void *data);

#endif
