#include <unistd.h>
#include <stdlib.h>
#include <stdio.h>
#include "g_define.h"

bool url_download_file(char *url, char *dir)
{
    pid_t status;
    char cmd_buf[500] = {0};

    snprintf(cmd_buf, 500, "wget -P %s %s", dir, url);
    status = system(cmd_buf);

    //if (-1 != status && WIFEXITED(status) && 0 == WEXITSTATUS(status))
    if(-1 != status)
        return true;

    return false;
}

void write_passid_to_file(FILE *fp, char *pass_id, int len)
{
    if (fp) {
        fseek(fp, 0L, SEEK_SET);
        fwrite(pass_id, len, 1, fp );
        fflush(fp);
    }

}

void read_passid_from_file(char *p_pass_id, int cam_id)
{
    char file_url[50] = {0};

    snprintf(file_url, 50, ORA_PASSID_FILE, cam_id);
    FILE *fp = fopen( file_url , "r" );

    if (fp) {
        //fread(pass_id, 50, 1, fp);
        fgets(p_pass_id, 50, fp);
    }
}