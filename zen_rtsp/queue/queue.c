#include <iostream>
#include <stdlib.h>
#include <stdio.h>
#include <sys/time.h>
#include <unistd.h>
#include "queue.h"
#include "common.h"
#include "cam_event.h"

/*构造一个空队列*/
Queue *InitQueue()
{
	Queue *pqueue = (Queue *)malloc(sizeof(Queue));
	if(pqueue!=NULL)
	{
		pqueue->front = NULL;
		pqueue->rear = NULL;
		pqueue->size = 0;
		pthread_mutex_init(&pqueue->q_lock, NULL);		
		//pthread_cond_init(&pqueue->cond, NULL);
	}
	return pqueue;
}
 
/*销毁一个队列*/
void DestroyQueue(Queue *pqueue)
{
	if(!pqueue)
		return;
	ClearQueue(pqueue);
	pthread_mutex_destroy(&pqueue->q_lock);
	//pthread_cond_destroy(&pqueue->cond);
	free(pqueue);
	pqueue = NULL;
}
 
/*清空一个队列*/
void ClearQueue(Queue *pqueue)
{
	if(!IsEmpty(pqueue)) {
		pthread_mutex_lock(&pqueue->q_lock);
		PNode front = pqueue->front;
	
		while (front) {
			PNode node = front;
			front = front->next;
			if (node->p_del_fun)
				node->p_del_fun(node->p_value);
			if (node->p_value)
				free(node->p_value);
			free(node);
		}
		
		pqueue->size = 0;
		pqueue->front = NULL;
		pqueue->rear = NULL;
		pthread_mutex_unlock(&pqueue->q_lock);
	}
 
}

//删除前5s之前的node
void ConditionDeleteQueue(Queue *pqueue, unsigned long ts)
{
	PNode pnode = NULL;
	PNode tmp_node = NULL;
	
	pthread_mutex_lock(&pqueue->q_lock);
	
	if(!IsEmpty(pqueue)) {
		pnode = pqueue->front; //第一个node
		//pqueue->front->p_value)->ts > ts处理系统时间调整
		while ( pqueue->front && ( ((frame_node_t *)pqueue->front->p_value)->ts < ts-5 || ((frame_node_t *)pqueue->front->p_value)->ts > ts)  ) {
			tmp_node = pqueue->front;
			pqueue->front = pqueue->front->next;
			pqueue->size--;
		} //获取符合条件最后一个node
	
		if(pqueue->size==0)
			pqueue->rear = NULL;
	}
	
	pthread_mutex_unlock(&pqueue->q_lock);

	if (!tmp_node)
		return;

	tmp_node->next = NULL;

	 do{ //删除所有符合条件的node
		PNode del_node = pnode;
		pnode = pnode->next;
	
		if (del_node->p_del_fun)
			del_node->p_del_fun(del_node->p_value);
		
		free(del_node->p_value);
		free(del_node);
	}while(pnode); //&& pnode != tmp_node);
	
}
 
/*判断队列是否为空*/
int IsEmpty(Queue *pqueue)
{
	//if(pqueue->front==NULL&&pqueue->rear==NULL&&pqueue->size==0)
	if (pqueue->size==0)
		return 1;
	else
		return 0;
}
 
/*返回队列大小*/
int GetSize(Queue *pqueue)
{
	return pqueue->size;
}
 
/*返回队头元素*/
PNode GetFront(Queue *pqueue)
{
	#if 0
	PNode p_node = NULL;
	pthread_mutex_lock(&pqueue->q_lock);
	/*
	if(!IsEmpty(pqueue))
	{
		*frame = pqueue->front->frame;
	}else {
		pthread_cond_wait(&pqueue->cond, &pqueue->q_lock);
	}*/
	while(!IsEmpty(pqueue))
		p_node = pqueue->front;//---->此处有bug，队列为空时，在锁释放后，pqueue->front可能被入队操作赋值，出现frame等于NULL，而pqueue->front不等于NULL
	//*frame = pqueue->front->frame;
	pthread_mutex_unlock(&pqueue->q_lock);
	#endif
	
	return pqueue->front;
}
/*返回队尾元素*/
 
PNode GetRear(Queue *pqueue)
{
	//if(!IsEmpty(pqueue)) {
	//	*frame = pqueue->rear->frame;
	//}
	return pqueue->rear;
}
 
/*将新元素入队*/
PNode EnQueue(Queue *pqueue, char *p_val, unsigned short len, PTRFUN p_fun)
{
	if (!pqueue)
		return NULL;
	
	PNode pnode = (PNode)malloc(sizeof(qnode));
	if(pnode != NULL) {
		pnode->p_value = p_val;
		pnode->val_len = len;
		pnode->p_del_fun = p_fun;
		pnode->next = NULL;
		
		pthread_mutex_lock(&pqueue->q_lock);
		if(IsEmpty(pqueue)) {
			pqueue->front = pnode;
			
		} else {
			pqueue->rear->next = pnode;
			
		}
		pqueue->rear = pnode;
		pqueue->size++;
		
		//pthread_cond_signal(&pqueue->cond);
		pthread_mutex_unlock(&pqueue->q_lock);
	}
	return pnode;
}

//结点入队列，并保持队列的大小
PNode FixEnQueue(Queue *pqueue, void *p_val, int len, PTRFUN p_fun, unsigned char fix_size)
{
	if (!pqueue)
		return NULL;

	PNode prev_node = NULL;
	PNode pnode = (PNode)malloc(sizeof(qnode));
	if(pnode != NULL) {
		pnode->p_value = (char *)p_val;
		pnode->val_len = len;
		pnode->p_del_fun = p_fun;
		pnode->next = NULL;
		
		pthread_mutex_lock(&pqueue->q_lock);
		if(IsEmpty(pqueue)) {
			pqueue->front = pnode;
			
		} else {
			pqueue->rear->next = pnode;
			
		}
		pqueue->rear = pnode;
		
		if (fix_size == pqueue->size) {
			prev_node = pqueue->front;
			pqueue->front = pqueue->front->next;
		}else {
			pqueue->size++;
		}
		
		//pthread_cond_signal(&pqueue->cond);
		pthread_mutex_unlock(&pqueue->q_lock);
	}
	return prev_node;

}

 
/*队头元素出队*/
PNode DeQueue(Queue *pqueue)
{
	if (!pqueue)
		return NULL;
	PNode pnode = NULL;
	
	pthread_mutex_lock(&pqueue->q_lock);
	if(!IsEmpty(pqueue)) {
		pnode = pqueue->front;
		
		pqueue->size--;
		pqueue->front = pnode->next;
		/*
		if (pnode->p_del_fun)
			pnode->p_del_fun(pnode->p_value);
		free(pnode->p_value);
		free(pnode);
		*/
		if(pqueue->size==0)
			pqueue->rear = NULL;
	}
	pthread_mutex_unlock(&pqueue->q_lock);
	//return pqueue->front;
	return pnode;
}

/*一次把所有node切出*/
PNode CutQueue(Queue *pqueue)
{
	if (!pqueue)
		return NULL;
	PNode pnode = NULL;

	pthread_mutex_lock(&pqueue->q_lock);
	pnode = pqueue->front;
	pqueue->front = NULL;
	pqueue->rear = NULL;
	pqueue->size = 0;	
	pthread_mutex_unlock(&pqueue->q_lock);

	return pnode;
}

#if 0
/*遍历队列并对各数据项调用visit函数*/
void QueueTraverse(Queue *pqueue, void (*visit)())
{
	PNode pnode = pqueue->front;
	int i = pqueue->size;
	while(i--)
	{
		visit(pnode->p_value);
		pnode = pnode->next;
	}
		
}

#endif



