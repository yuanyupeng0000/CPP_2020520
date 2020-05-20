#include <stdio.h>
#include<stdlib.h>
#include<mysql/mysql.h>
#include<string.h>
#include <pthread.h>
#include "../common.h"
#include "fvdmysql.h"

MYSQL *g_conn;
const char *server   = "localhost";
const char *user     = "root";
const char *passwd   = "111111";
const char *dataBase = "AIPD";
unsigned char conn_status = 0;


pthread_mutex_t exec_lock;

void my_mysql_init()
{
    pthread_mutex_init(&exec_lock, NULL);
    g_conn = mysql_init(NULL);
    my_mysql_connect();
    return;
}

void my_mysql_deinit()
{
    mysql_close(g_conn);
}

int my_mysql_last_id()
{
    int id = 0;

    if (1 == conn_status) {
        pthread_mutex_lock(&exec_lock);
        id = mysql_insert_id(g_conn);
        pthread_mutex_unlock(&exec_lock);
    }

    return id;
}

int my_mysql_exec(char *p_sql)
{
    pthread_mutex_lock(&exec_lock);

    if ( 1 == conn_status ) {
        if (mysql_query(g_conn, p_sql)) {
            prt(info, "%s  exec error!", p_sql);
            pthread_mutex_unlock(&exec_lock);
            return -1;
        }
    } else {
        my_mysql_connect();
    }

    pthread_mutex_unlock(&exec_lock);
    return 0;
}

void my_mysql_connect()
{

    if (!mysql_real_connect(g_conn, server, user, passwd, dataBase, 0, NULL, 0)) {
        prt(info, "Error connecting to Mysql: %s\n", mysql_error(g_conn));
        conn_status = 0;

    } else {
        conn_status = 1;
        mysql_set_character_set(g_conn, "utf8");
    }


}



