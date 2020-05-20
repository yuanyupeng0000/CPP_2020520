#ifndef INCLUDE_QUEUE_H_
#define INCLUDE_QUEUE_H_
#include <stdlib.h>
#include <pthread.h>

typedef void (*PTRFUN)(void *);

typedef struct qnode
{
	char *p_value;
	unsigned short val_len; 
	struct qnode *next;
	PTRFUN p_del_fun;
}QNode, *PNode;

//typedef struct node *PNode;
 
typedef struct
{
	PNode front;
	PNode rear;
	int size;
	pthread_mutex_t q_lock;
	//pthread_cond_t cond;
}Queue;
 
/*构造一个空队列*/
Queue *InitQueue();
 
/*销毁一个队列*/
void DestroyQueue(Queue *pqueue);
 
/*清空一个队列*/
void ClearQueue(Queue *pqueue);
 
/*判断队列是否为空*/
int IsEmpty(Queue *pqueue);
 
/*返回队列大小*/
int GetSize(Queue *pqueue);
 
/*返回队头元素*/
PNode GetFront(Queue *pqueue);
 
/*返回队尾元素*/
PNode GetRear(Queue *pqueue);
 
/*将新元素入队*/
//PNode EnQueue(Queue *pqueue);
PNode EnQueue(Queue *pqueue, char *p_val, unsigned short len, PTRFUN p_fun);
 
/*队头元素出队*/
PNode DeQueue(Queue *pqueue);
 
/*遍历队列并对各数据项调用visit函数*/
//void QueueTraverse(Queue *pqueue,void (*visit)());
PNode FixEnQueue(Queue *pqueue, void *p_val, int len, PTRFUN p_fun, unsigned char fix_size);

//删除早于ts秒的node
void ConditionDeleteQueue(Queue *pqueue, unsigned long ts);

/*一次把所有node切出*/
PNode CutQueue(Queue *pqueue);

#endif


