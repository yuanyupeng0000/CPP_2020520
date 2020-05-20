#ifndef __MY_MYSQL_H__
#define __MY_MYSQL_H__

void my_mysql_init();
int my_mysql_last_id();
void my_mysql_deinit();
int  my_mysql_exec(char *p_sql);
void my_mysql_connect();



#endif /* __MY_MYSQL_H__ */
