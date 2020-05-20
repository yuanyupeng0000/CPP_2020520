/*
 * url_downfile.h
 *
 *  Created on: 2016Äê9ÔÂ9ÈÕ
 *      Author: root
 */

#ifndef URL_DOWNFILE_H_
#define URL_DOWNFILE_H_
#include <stdio.h>
#include <stdlib.h>

bool url_download_file(char *url, char *dir);
void read_passid_from_file(char *pass_id, int cam_id);
void write_passid_to_file(FILE *fp, char *pass_id, int len);

#endif /* CAMERA_SERVICE_H_ */
