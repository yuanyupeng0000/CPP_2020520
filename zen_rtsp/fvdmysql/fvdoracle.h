#ifndef __MY_ORACLE_H__
#define __MY_ORACLE_H__

#include "common.h"

void my_oracle_init();
int  get_recored_from_oracle(ora_record_t *rds, char *pass_id);

#endif /* __MY_MYSQL_H__ */
