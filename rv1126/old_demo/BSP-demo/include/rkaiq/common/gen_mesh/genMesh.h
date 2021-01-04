/*
 *  Copyright (c) 2020 Rockchip Corporation
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
 *
 */

#ifndef __GENMESH_H__
#define __GENMESH_H__

/* ������� */
struct CameraCoeff
{
	double cx, cy;							// ��ͷ�Ĺ���
	double a0, a2, a3, a4;					// ��ͷ�Ļ���ϵ��
	double c, d, e;							// �ڲ�[c d;e 1]
	double sf;								// sf�����ӽǣ�sfԽ���ӽ�Խ��

	/* level = 0ʱ��rho-tanTheta����ʽ��� */
	int invPolyTanNum0;						// ϵ������
	double invPolyTanCoeff0[21];			// ����ʽϵ������ߴ���20��
	/* level = 0ʱ��rho-cotTheta����ʽ��� */
	int invPolyCotNum0;						// ϵ������
	double invPolyCotCoeff0[21];			// ����ʽϵ������ߴ���20��

	/* level = 255ʱ��rho-tanTheta����ʽ��� */
	int invPolyTanNum255;					// ϵ������
	double invPolyTanCoeff255[21];			// ����ʽϵ������ߴ���20��
	/* level = 255ʱ��rho-cotTheta����ʽ��� */
	int invPolyCotNum255;					// ϵ������
	double invPolyCotCoeff255[21];			// ����ʽϵ������ߴ���20��
};

/* ����FECӳ�����صĲ��� */
struct FecParams
{
	int correctX;										/* ˮƽx����У����1����У����0����У�� */
	int correctY;										/* ��ֱy����У����1����У����0����У�� */
	int saveMesh4bin;									/* �Ƿ񱣴�meshxi,xf,yi,yf4��bin�ļ���1�����棬0�������� */
	char mesh4binPath[256];								/* ����meshxi,xf,yi,yf4��bin�ļ���·�� */
	int srcW, srcH, dstW, dstH;							/* �������ͼ��ķֱ��� */
	int srcW_ex, srcH_ex, dstW_ex, dstH_ex;				/* ��չ�����������ֱ��� */
	int meshSizeW, meshSizeH;
	double meshStepW, meshStepH;
	int meshSize1bin;
	int meshSize4bin;
	unsigned short	SpbNum;
	unsigned long	MeshPointNumW;
	unsigned short	SpbMeshPNumH;
	unsigned short	LastSpbMeshPNumH;

	double *mapx;
	double *mapy;
	unsigned short	*pMeshXY;

};

/* ����LDCHӳ�����صĲ��� */
struct LdchParams
{
	int saveMeshX;									/* �Ƿ񱣴�MeshX.bin�ļ���1�����棬0�������� */
	char meshPath[256];								/* ����MeshX.bin�ļ���·�� */
	int srcW, srcH, dstW, dstH;						/* �������ͼ��ķֱ��� */
	int meshSizeW, meshSizeH;
	double meshStepW, meshStepH;
	int mapxFixBit;									/* ���㻯����λ�� */
	int meshSize;
	int maxLevel;
	double *mapx;
	double *mapy;
};

/* FEC: ��ʼ��������ͼ������ֱ��ʣ�����FECӳ������ز�����������Ҫ��buffer */
void genFecMeshInit(int srcW, int srcH, int dstW, int dstH, FecParams &fecParams, CameraCoeff &camCoeff);

/* FEC: ����ʼ�� */
void genFecMeshDeInit(FecParams &fecParams);

/* FEC: Ԥ�ȼ���Ĳ���: ����δУ����С���level=0,level=255�Ķ���ʽ���� */
void genFecPreCalcPart(FecParams &fecParams, CameraCoeff &camCoeff);

/* FEC: 4��mesh �ڴ����� */
void mallocFecMesh(int meshSize, unsigned short **pMeshXI, unsigned char **pMeshXF, unsigned short **pMeshYI, unsigned char **pMeshYF);

/* FEC: 4��mesh �ڴ��ͷ� */
void freeFecMesh(unsigned short *pMeshXI, unsigned char *pMeshXF, unsigned short *pMeshYI, unsigned char *pMeshYF);

/*
��������: ���ɲ�ͬУ���̶ȵ�meshӳ�������ISP��FECģ��
	����:
	1��FECӳ������ز�����������Ҫ��buffer: FecParams &fecParams
	2������궨����: CameraCoeff &camCoeff
	3����ҪУ���ĳ̶�: level(0-255: 0��ʾУ���̶�Ϊ0%, 255��ʾУ���̶�Ϊ100%)
	���:
	1��bool �Ƿ�ɹ�����
	2��pMeshXI, pMeshXF, pMeshYI, pMeshYF
*/
bool genFECMeshNLevel(FecParams &fecParams, CameraCoeff &camCoeff, int level, unsigned short *pMeshXI, unsigned char *pMeshXF, unsigned short *pMeshYI, unsigned char *pMeshYF);


/* =============================================================================================================================================================================== */

/* LDCH: ��ʼ��������ͼ������ֱ��ʣ�����LDCHӳ������ز�����������Ҫ��buffer */
void genLdchMeshInit(int srcW, int srcH, int dstW, int dstH, LdchParams &ldchParams, CameraCoeff &camCoeff);

/* LDCH: ����ʼ�� */
void genLdchMeshDeInit(LdchParams &ldchParams);

/* LDCH: Ԥ�ȼ���Ĳ���: ����δУ����С���level=0,level=255�Ķ���ʽ���� */
void genLdchPreCalcPart(LdchParams &ldchParams, CameraCoeff &camCoeff);

/* LDCH: ����LDCH�ܹ�У�������̶� */
void calcLdchMaxLevel(LdchParams &ldchParams, CameraCoeff &camCoeff);

/*
��������: ���ɲ�ͬУ���̶ȵ�meshӳ�������ISP��LDCHģ��

	����:
	1��LDCHӳ������ز�����������Ҫ��buffer: LdchParams &ldchParams
	2������궨����: CameraCoeff &camCoeff
	3����ҪУ���ĳ̶�: level(0-255: 0��ʾУ���̶�Ϊ0%, 255��ʾУ���̶�Ϊ100%)
	���:
	1��bool �Ƿ�ɹ�����
	2��pMeshX
*/
bool genLDCMeshNLevel(LdchParams &ldchParams, CameraCoeff &camCoeff, int level, unsigned short *pMeshX);

#endif // !__GENMESH_H__
