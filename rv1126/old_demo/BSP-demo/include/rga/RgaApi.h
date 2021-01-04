/*
 * Copyright (C) 2016 Rockchip Electronics Co., Ltd.
 * Authors:
 *	Zhiqin Wei <wzq@rock-chips.com>
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include <stdint.h>
#include <sys/types.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>

#include <sys/mman.h>
#include <linux/stddef.h>

#include "drmrga.h"

#ifdef __cplusplus
extern "C"{
#endif


int c_RkRgaInit();
void c_RkRgaDeInit();
int c_RkRgaGetAllocBuffer(bo_t *bo_info, int width, int height, int bpp);
int c_RkRgaGetAllocBufferCache(bo_t *bo_info, int width, int height, int bpp);
int c_RkRgaGetMmap(bo_t *bo_info);
int c_RkRgaUnmap(bo_t *bo_info);
int c_RkRgaFree(bo_t *bo_info);
int c_RkRgaGetBufferFd(bo_t *bo_info, int *fd);
int c_RkRgaBlit(rga_info_t *src, rga_info_t *dst, rga_info_t *src1);
int c_RkRgaColorFill(rga_info_t *dst);
#ifdef __cplusplus
}
#endif
