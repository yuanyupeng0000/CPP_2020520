#ifndef __G_FUN__
#define __G_FUN__
#include <stdlib.h>
#include <stdio.h>
#include <pthread.h>

#if 0

#define my_mutex_lock(lock) \
 do { \
 pthread_cleanup_push(my_mutex_clean,lock); \
 pthread_mutex_lock((pthread_mutex_t *)lock); \
 }while(0)




#define my_mutex_unlock(lock)  \
do {  \
 pthread_mutex_unlock((pthread_mutex_t *)lock); \
 pthread_cleanup_pop(0); \
 }while(0)
#endif

void my_mutex_clean(void *lock);
//void my_mutex_lock(void * lock);
//void my_mutex_unlock(void * lock);


#endif
