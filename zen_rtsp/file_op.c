/*
 * file_op.c
 *
 *  Created on: 2016��8��30��
 *      Author: root
 */
#include <stdio.h>
#include "file_op.h"
#include "common.h"
void save_buf(char *name,char *bf,int pos,int len)
{
	FILE *f=NULL;
	f=fopen(name,"rb+");
	if(f==NULL){
		prt(info,"err in opne %s",name);
	}else{
		fseek(f,pos,SEEK_SET);
		fwrite(bf,len,1,f);

	//	prt(info,"wirte file %s ok",name);
		fclose(f);
	}
}
void load_buf(char *name,char *bf,int pos,int len)
{
	FILE *f=NULL;
	f=fopen(name,"rb+");
	if(f==NULL){
		prt(info,"err in opne %s",name);
	}else{
		fseek(f,pos,SEEK_SET);
		fread(bf,len,1,f);
	//	prt(info,"read file %s ok",name);
		fclose(f);
	}
}
//int main(int argc, char **argv)
//{
//	char bf[]={1,1,1,1,1,1,1};
//	save_buf("cfg/test",bf,30,6);
//	return 0;
//}
