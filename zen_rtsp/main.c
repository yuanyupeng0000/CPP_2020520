#include <unistd.h>
#include "camera_service.h"
#include "tcp_server.h"
#include "client_net.h"
#include "common.h"
#include "sig_service.h"
#include "udp_network.h"
#include "g_define.h"
#include "fvdmysql/fvdmysql.h"
#include "fvdmysql/fvdoracle.h"

#define CAMNUMSTR cameraLOG%d
/* Function Description
 * name:main
 * return:
 * args:
 * comment:entrance
 * todo:
 */
extern IVDDevSets  g_ivddevsets;

int main(int argc, char **argv)
{
    prt(info, ".............programe start.............");
    setup_sig();

#if (VERTION_TIMEOUT == 1)
    if ( !time_out() )
        return 0;
#endif

    init_global_lock();
    my_mysql_init();
//#if ORACLE_VS==1
//    my_oracle_init();
//#endif
    init_variable();
    init_config();
    init_ntp();
    camera_service_init();
    init_camera();

    init_server();
    statis_handle();
    protocol_select(g_ivddevsets.pro_type);

    if ( init_tcp_server() < 0 ) {
        prt(info, "init tcp server error. programe exit.");
        exit(-1);
    }

    while (1)
        usleep(1000);

    return 0;
}
