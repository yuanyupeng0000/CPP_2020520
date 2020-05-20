#include<stdio.h>
#include <unistd.h>

int go_ping(char *svrip)
{
        int i = 0;
        while(i < 3)
        {
                pid_t pid;
                if ((pid = vfork()) < 0) 
                {
                        printf("vfork error");
                        exit(1);
                } 
                else if (pid == 0) 
                {
                        if ( execlp("ping", "ping","-c", "1",svrip, (char*)0) < 0)
                        {
                                printf("execlp error\n");
                                exit(1);
                        }
                }

                int stat;
                waitpid(pid, &stat, 0);

                if (stat == 0)
                {
                        return 0;
                }
                sleep(3);
                i++;
        }
        return -1;
}

int main(void)
{
    int ret = go_ping("192.168.1.20");
    printf("+++++++++++++ret:%d\n",ret);

}
