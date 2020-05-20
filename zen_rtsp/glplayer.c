#include <stdio.h>
#include <unistd.h>
#include <string.h>
#include <X11/Xlib.h>
#include "glew.h"
#include "glut.h"
#include "common.h"
#include "glplayer.h"
inline unsigned char CONVERT_ADJUST(double tmp)
{
	return (unsigned char)((tmp >= 0 && tmp <= 255)?tmp:(tmp < 0 ? 0 : 255));
}
//YUV420P to RGB24
void CONVERT_YUV420PtoRGB24(unsigned char* yuv_src,unsigned char* rgb_dst,int nWidth,int nHeight)
{
	unsigned char *tmpbuf=(unsigned char *)malloc(nWidth*nHeight*3);
	unsigned char Y,U,V,R,G,B;
	unsigned char* y_planar,*u_planar,*v_planar;
	int rgb_width , u_width;
	rgb_width = nWidth * 3;
	u_width = (nWidth >> 1);
	int ypSize = nWidth * nHeight;
	int upSize = (ypSize>>2);
	int offSet = 0;

	y_planar = yuv_src;
	u_planar = yuv_src + ypSize;
	v_planar = u_planar + upSize;

	for(int i = 0; i < nHeight; i++)
	{
		for(int j = 0; j < nWidth; j ++)
		{
			// Get the Y value from the y planar
			Y = *(y_planar + nWidth * i + j);
			// Get the V value from the u planar
			offSet = (i>>1) * (u_width) + (j>>1);
			V = *(u_planar + offSet);
			// Get the U value from the v planar
			U = *(v_planar + offSet);

			// Cacular the R,G,B values
			// Method 1
			R = CONVERT_ADJUST((Y + (1.4075 * (V - 128))));
			G = CONVERT_ADJUST((Y - (0.3455 * (U - 128) - 0.7169 * (V - 128))));
			B = CONVERT_ADJUST((Y + (1.7790 * (U - 128))));
			/*
			// The following formulas are from MicroSoft' MSDN
			int C,D,E;
			// Method 2
			C = Y - 16;
			D = U - 128;
			E = V - 128;
			R = CONVERT_ADJUST(( 298 * C + 409 * E + 128) >> 8);
			G = CONVERT_ADJUST(( 298 * C - 100 * D - 208 * E + 128) >> 8);
			B = CONVERT_ADJUST(( 298 * C + 516 * D + 128) >> 8);
			R = ((R - 128) * .6 + 128 )>255?255:(R - 128) * .6 + 128;
			G = ((G - 128) * .6 + 128 )>255?255:(G - 128) * .6 + 128;
			B = ((B - 128) * .6 + 128 )>255?255:(B - 128) * .6 + 128;
			*/
			offSet = rgb_width * i + j * 3;

			rgb_dst[offSet] = B;
			rgb_dst[offSet + 1] = G;
			rgb_dst[offSet + 2] = R;
		}
	}
	free(tmpbuf);
}

void init_gl()
{
	XInitThreads();
	int testc = 0;
	char testv[][2] = { 1, 2 };

	glutInit(&testc, NULL);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
	glutInitWindowSize(640, 480);
}
//static int iii=0;
void start_gl_window(int *p_w)
{

	//	int testc = 0;
	//	char testv[][2] = { 1, 2 };

	//	glutInit(&testc, NULL);
//		glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
//		glutInitWindowSize(640, 480);
	prt(info,"start gl window %d",*p_w);


	*p_w = glutCreateWindow("a");

	//*p_w = glutCreateWindow("a");
	//	 glutShowWindow();
	prt(info,"start gl window %d",*p_w);
//	iii++;
}

void stop_gl_window(int *p_w)
{
	glutInitWindowPosition(100, 100);
    prt(info,"desrotying gl window %d ",*p_w)
	glutHideWindow();
 //   glutDestroyWindow( *p_w);
  //  glutHideWindow();
}

void diaplay_frames(unsigned char *bf_src,int wid ,int hei,unsigned char  *bf_dst,int w)
{
	glutSetWindow(w);
	glRasterPos3f(-1.0f,1.0f,0);
	glPixelZoom((float)600/(float)wid, -(float)600/hei);
	CONVERT_YUV420PtoRGB24(bf_src,bf_dst,wid,hei);
	glDrawPixels(wid, hei,GL_RGB, GL_UNSIGNED_BYTE, bf_dst);
	glutSwapBuffers();
}

void copy_frame(unsigned char *picy, unsigned char *picu, unsigned char *picv,int wid,int hei, unsigned char *bf)
{
	memcpy(bf,picy,wid*hei);
	memcpy(bf+wid*hei,picu,wid*hei/4);
	memcpy(bf+wid*hei*5/4,picv,wid*hei/4);
}
