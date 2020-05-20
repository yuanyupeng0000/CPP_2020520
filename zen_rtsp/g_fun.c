#include "g_fun.h"

void my_mutex_clean(void *lock)
{
    printf("mutex_clean........................！\n");
    pthread_mutex_unlock((pthread_mutex_t*)lock);
}

#if 0
void my_mutex_lock(void * lock)
{
   pthread_cleanup_push(my_mutex_clean,lock);
   pthread_mutex_lock((pthread_mutex_t *)lock);
}

void my_mutex_unlock(void * lock)
{
    pthread_mutex_unlock((pthread_mutex_t *)lock);
    pthread_cleanup_pop(0);
}

#endif


