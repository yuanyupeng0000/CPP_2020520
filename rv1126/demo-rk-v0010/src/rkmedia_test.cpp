#include "rkmedia_test.h"
#include <iostream>
#include <chrono>

#include <rga/RgaApi.h>
#include <rga/im2d.h>
#include "rkmedia_api.h"
#include "rkmedia_venc.h"
#include "librknn.hpp"

#define CROP_TARGET_WIDTH 640
#define CROP_TARGET_HEIGHT 640

MPP_CHN_S g_stViChn;
pthread_mutex_t gmuFrameNumReadWrite;
static int gMem_fd;
static char *gSh_mem;

static bool quit = false;



int Face_Sync_Shm_Init(void)
{
	int shm_size;

	shm_size = sizeof(FACE_SHM_PROFILES) + 64 + sizeof(FaceRec_Detect_Param) + sizeof(Face_Upgrade_Param) + sizeof(DEVICE_INFO_S);  

	printf("\n\n\nshm_size=%d   %d  %d   %d   %d\n\n", shm_size, sizeof(FACE_SHM_PROFILES), 
		      sizeof(FaceRec_Detect_Param), sizeof(Face_Upgrade_Param), sizeof(DEVICE_INFO_S));
	
	gMem_fd = open("/dev/vsmem", O_CREAT|O_RDWR, 00666);
	if(gMem_fd < 0)  
	{
		printf("shm_init open error!\n");
		return -1;
	}

	gSh_mem = (char *)mmap(NULL, shm_size, PROT_READ|PROT_WRITE, MAP_SHARED, gMem_fd, 0);  
	if(gSh_mem == MAP_FAILED)
	{
		printf("shm_init error!\n");
		close(gMem_fd);
		return -1;
	}
	
	close(gMem_fd);
	return 0;
}

int Face_Sync_Shm_Save_Result(RK_U64 u64pts, RW_FACE_INFO *pFRInfo, unsigned int face_num)
{
	int i = 0;
	
	FACE_SHM_PROFILES *pFSProfiles = (FACE_SHM_PROFILES *)gSh_mem;
	int index;
	FACE_SHM_UNIT *pFSUnit;

	if((pFSProfiles->write_index - pFSProfiles->read_index) > (FACE_SNM_MAX_NUM - 1))
	{
		printf("face sync shm buffer is full\n");	
	}
	else
	{		
		index = pFSProfiles->write_index % FACE_SNM_MAX_NUM;
		pFSUnit = (FACE_SHM_UNIT *)(&pFSProfiles->sf_unit[index]);

		memset(pFSUnit, 0, (sizeof(FACE_SHM_UNIT)));
		pFSUnit->s_time = 0;			
		pFSUnit->u64pts = u64pts;
			
		pFSUnit->face_num = face_num;
		if (pFSUnit->face_num > 40)
		{
			pFSUnit->face_num = 40;
		}
		
		for(i=0; i<face_num; i++)
		{
			memcpy(&pFSUnit->face_unit[i], &pFRInfo[i], sizeof(RW_FACE_INFO));
			pFSUnit->face_unit[i].release = 0x7E;
		}
		
		pFSUnit->flag = 1;
		pFSProfiles->write_index++;
	}
	
	return 0;
}

int ZRK_Face_Tracker(MEDIA_BUFFER src_mb, RW_FACE_INFO *pstFaceInfo, unsigned int *puFaceCout)
{
    //rag resize and convert

	static int j=0;
	int face_count = 5;

	*puFaceCout = face_count;
		
	pstFaceInfo[j].x = 200+100*j;
	pstFaceInfo[j].y = 200+100*j;
	pstFaceInfo[j].w = 600+100*j;
	pstFaceInfo[j].h= 500+50*j;
	pstFaceInfo[j].trackID = 115+j;
	pstFaceInfo[j].type= 0;
	pstFaceInfo[j].quality = 80;
	pstFaceInfo[j].confidence = 80;		

	j++;
	if (j > 5)
		j = 0;
	
	return 0;
}

int ZRK_Bbox_Tracker(std::vector<DetectionObject> &objs, RW_FACE_INFO *pstFaceInfo, unsigned int *puFaceCout)
{
    //rag resize and convert
    int face_count = objs.size();

    *puFaceCout = face_count;

    for(int i=0; i<face_count; i++){
        if(i >= FD_RW_FACE_COUNT_MAX){
            break;
        }
        //3840x2160
        float scale_w = 3840/416;
        float scale_h = 2160/416;
        pstFaceInfo[i].x = objs[i].xmin * scale_w;
        pstFaceInfo[i].y = objs[i].ymin * scale_h;
        pstFaceInfo[i].w = objs[i].xmax * scale_w - objs[i].xmin * scale_w + 1;
        pstFaceInfo[i].h= objs[i].ymax * scale_h - objs[i].ymin * scale_h + 1;
        pstFaceInfo[i].trackID = objs[i].class_id;
        pstFaceInfo[i].type= objs[i].class_id;
        pstFaceInfo[i].quality = (int)(objs[i].confidence*100);
        pstFaceInfo[i].confidence = (int)(objs[i].confidence*100);
        std::cout << pstFaceInfo[i].x << " "
                  << pstFaceInfo[i].y << " "
                  << pstFaceInfo[i].w << " "
                  << pstFaceInfo[i].h << " "
                  << pstFaceInfo[i].trackID << " "
                  << int(pstFaceInfo[i].confidence) << std::endl;
    }

    return 0;
}

void *ZRK_Task_Thread(void *arg)
{
	MEDIA_BUFFER pstFrame;		
	RK_U64 u64pts;
	RW_FACE_INFO stFaceInfo[FD_RW_FACE_COUNT_MAX];
	unsigned int uFaceCount = 0;		
	printf("\nRK_Task_Thread start \n");

	while(1)
	{
		pstFrame = RK_MPI_SYS_GetMediaBuffer(g_stViChn.enModId, g_stViChn.s32ChnId, -1);
		if (!pstFrame)
		{
			printf("RK_MPI_SYS_GetMediaBuffer error!\n");
			usleep(5000);
			continue;
		}

		uFaceCount = 0;	
		memset(&stFaceInfo, 0, sizeof(RW_FACE_INFO)*FD_RW_FACE_COUNT_MAX);
		ZRK_Face_Tracker(pstFrame, &stFaceInfo[0], &uFaceCount);

		u64pts = RK_MPI_MB_GetTimestamp(pstFrame);

		Face_Sync_Shm_Save_Result(u64pts, &stFaceInfo[0], uFaceCount);

		RK_MPI_MB_ReleaseBuffer(pstFrame);
		
		usleep(30000);
		
		continue;

	}

	RK_MPI_VI_DisableChn(g_stViChn.s32DevId, g_stViChn.s32ChnId);

	return NULL;
}

void *ZRK_Task_Thread_VI_RGA_RKNN(void *arg)
{
    MEDIA_BUFFER pstFrame;
    RK_U64 u64pts;
    RW_FACE_INFO stFaceInfo[FD_RW_FACE_COUNT_MAX];
    unsigned int uFaceCount = 0;
    printf("\nRK_Task_Thread start \n");

    while(1)
    {
        pstFrame = RK_MPI_SYS_GetMediaBuffer(g_stViChn.enModId, g_stViChn.s32ChnId, -1);
        if (!pstFrame)
        {
            printf("RK_MPI_SYS_GetMediaBuffer error!\n");
            usleep(5000);
            continue;
        }

        uFaceCount = 0;
        memset(&stFaceInfo, 0, sizeof(RW_FACE_INFO)*FD_RW_FACE_COUNT_MAX);
        ZRK_Face_Tracker(pstFrame, &stFaceInfo[0], &uFaceCount);

        u64pts = RK_MPI_MB_GetTimestamp(pstFrame);

        Face_Sync_Shm_Save_Result(u64pts, &stFaceInfo[0], uFaceCount);

        RK_MPI_MB_ReleaseBuffer(pstFrame);

        usleep(30000);

        continue;

    }

    RK_MPI_VI_DisableChn(g_stViChn.s32DevId, g_stViChn.s32ChnId);

    return NULL;
}

int ZRK_Vi_Param_Set(VI_CHN_ATTR_S *pstvi_chn_attr, int iwidth, int ihight)
{
	pstvi_chn_attr->pcVideoNode = "rkispp_scale0";
	pstvi_chn_attr->enPixFmt = IMAGE_TYPE_NV12;
	pstvi_chn_attr->u32BufCnt = 10;
	pstvi_chn_attr->u32Width = iwidth;
	pstvi_chn_attr->u32Height = ihight;
	pstvi_chn_attr->enWorkMode = VI_WORK_MODE_NORMAL;
	
	return 0;
}

int ZRK_Vi_Init(int ividevnum, int ivichn, int iwidth, int ihight)
{
	VI_CHN_ATTR_S vi_chn_attr;
	
	RK_MPI_SYS_Init();

    ZRK_Vi_Param_Set(&vi_chn_attr, iwidth, ihight);

	RK_MPI_VI_SetChnAttr(ividevnum, ivichn, &vi_chn_attr);
	RK_MPI_VI_EnableChn(ividevnum, ivichn);
	
	return 0;
}

static void *GetMediaBuffer(void *arg) {
  printf("#Start %s thread, arg:%p\n", __func__, arg);
  rga_info_t src;
  rga_info_t dst;
  MEDIA_BUFFER src_mb = NULL;
  MEDIA_BUFFER dst_mb = NULL;

  RK_U64 u64pts;
  RW_FACE_INFO stFaceInfo[FD_RW_FACE_COUNT_MAX];
  unsigned int uFaceCount = 0;

  int ret = c_RkRgaInit();
  if (ret) {
    printf("ERROR: c_RkRgaInit() failed! ret = %d\n", ret);
    return NULL;
  }

  typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
  while (!quit) {
    auto t0 = std::chrono::high_resolution_clock::now();
    src_mb =
        RK_MPI_SYS_GetMediaBuffer(g_stViChn.enModId, g_stViChn.s32ChnId, -1);
    if (!src_mb) {
      printf("ERROR: RK_MPI_SYS_GetMediaBuffer get null buffer!\n");
      break;
    }
    auto t00 = std::chrono::high_resolution_clock::now();
    ms detection0 = std::chrono::duration_cast<ms>(t00 - t0);
    std::cout << "vi ms:" << detection0.count() << std::endl;

/////////////////////////////////////////////////////////////////////////////////////

    auto t1 = std::chrono::high_resolution_clock::now();

    MB_IMAGE_INFO_S stImageInfo = {CROP_TARGET_WIDTH, CROP_TARGET_HEIGHT,
                                   CROP_TARGET_WIDTH, CROP_TARGET_HEIGHT,
                                   IMAGE_TYPE_NV12};
    dst_mb = RK_MPI_MB_CreateImageBuffer(&stImageInfo, RK_TRUE, 0);
    if (!dst_mb) {
      printf("ERROR: RK_MPI_MB_CreateImageBuffer get null buffer!\n");
      break;
    }

    memset(&src, 0, sizeof(rga_info_t));
    memset(&dst, 0, sizeof(rga_info_t));

    src.fd = RK_MPI_MB_GetFD(src_mb);
    src.mmuFlag = 1;
    dst.fd = RK_MPI_MB_GetFD(dst_mb);
    dst.mmuFlag = 1;
    RK_MPI_MB_SetTimestamp(dst_mb, RK_MPI_MB_GetTimestamp(src_mb));
    rga_set_rect(&src.rect, 0, 0, 1920, 1080, 1920, 1080,
                 RK_FORMAT_YCbCr_420_SP);
    int SIZE = 416;
    rga_set_rect(&dst.rect, 0, 0, SIZE, SIZE,
                 SIZE, SIZE, RK_FORMAT_RGB_888);
    ret = c_RkRgaBlit(&src, &dst, NULL);
    auto t11 = std::chrono::high_resolution_clock::now();
    ms detection1 = std::chrono::duration_cast<ms>(t11 - t1);
    std::cout << "rga ms:" << detection1.count() << std::endl;
    std::vector<DetectionObject> objs;
    rknn_ai_alg(RK_MPI_MB_GetPtr(dst_mb), objs);

    //save detection result to memery
    uFaceCount = 0;
    memset(&stFaceInfo, 0, sizeof(RW_FACE_INFO)*FD_RW_FACE_COUNT_MAX);
    //ZRK_Face_Tracker(src_mb, &stFaceInfo[0], &uFaceCount);
    ZRK_Bbox_Tracker(objs, &stFaceInfo[0], &uFaceCount);
    u64pts = RK_MPI_MB_GetTimestamp(src_mb);
    Face_Sync_Shm_Save_Result(u64pts, &stFaceInfo[0], uFaceCount);

    RK_MPI_MB_ReleaseBuffer(src_mb);
    RK_MPI_MB_ReleaseBuffer(dst_mb);
    src_mb = NULL;
    dst_mb = NULL;
    auto t_end = std::chrono::high_resolution_clock::now();
    ms detection2 = std::chrono::duration_cast<ms>(t_end - t0);
    std::cout << "loop ms:" << detection2.count() << std::endl;
  }

  if (src_mb)
    RK_MPI_MB_ReleaseBuffer(src_mb);
  if (dst_mb)
    RK_MPI_MB_ReleaseBuffer(dst_mb);

////////////////////////////////////////////////////////////////////////////////////

  return NULL;
}


int  ZRK_Task_Init(void)
{
	pthread_t p_thread;
	int ret = 0;

	pthread_mutex_init(&gmuFrameNumReadWrite, NULL);

	g_stViChn.enModId = RK_ID_VI;
	g_stViChn.s32DevId = 0;
	g_stViChn.s32ChnId = 1;

	ZRK_Vi_Init(g_stViChn.s32DevId, g_stViChn.s32ChnId, IVE_BT1120_VI_WIDTH, IVE_BT1120_VI_HEIGTH);

    ///ret = pthread_create(&p_thread, NULL, ZRK_Task_Thread, NULL);
    ret = pthread_create(&p_thread, NULL, GetMediaBuffer, NULL);
	if (ret != 0)
	{
		printf("Create pthread error\n");
		return 0;
	}
	
	ret = RK_MPI_VI_StartStream(g_stViChn.s32DevId, g_stViChn.s32ChnId);
	if (ret)
	{
		printf("ERROR: Start Vi[0] failed! ret=%d\n", ret);
		return -1;
	}

	return 0;
}

int main(void)
{
	Face_Sync_Shm_Init();

	ZRK_Task_Init();
	
	while(1)
	{
		sleep(1);	
	}
	
	return 0;
}


