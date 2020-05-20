#ifndef __CLIENT_FILE__
#define __CLIENT_FILE__
void save_obj(unsigned char * p_obj,int class_type,int index);
void load_obj(unsigned char * p_obj,int class_type,int index);
char *cfg_file_name(char *filename,int class_type);
#endif
