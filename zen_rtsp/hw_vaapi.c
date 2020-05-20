#include <stdio.h>
#include <math.h>
#include <memory.h>
#include <stdio.h>
#include <pthread.h>
#include <fcntl.h>
#include <limits.h>

#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/frame.h>
#ifdef __cplusplus
}
#endif
#include <unistd.h>
#include <pthread.h>
#include <va/va.h>
#include <va/va_drm.h>
#include <va/va_compat.h>
#include "hw_vaapi.h"

#define MAX_IMAGE_PLANES 3
typedef struct _Image Image1;

struct _Image {
	uint32_t format;
	unsigned int width;
	unsigned int height;
	uint8_t *data;
	unsigned int data_size;
	unsigned int num_planes;
	uint8_t *pixels[MAX_IMAGE_PLANES];
	unsigned int offsets[MAX_IMAGE_PLANES];
	unsigned int pitches[MAX_IMAGE_PLANES];
};
#define IMAGE_FOURCC(ch0, ch1, ch2, ch3)        \
    ((uint32_t)(uint8_t) (ch0) |                \
     ((uint32_t)(uint8_t) (ch1) << 8) |         \
     ((uint32_t)(uint8_t) (ch2) << 16) |        \
     ((uint32_t)(uint8_t) (ch3) << 24 ))

#define IS_RGB_IMAGE_FORMAT(FORMAT)                      \
    ((FORMAT) == IMAGE_ARGB || (FORMAT) == IMAGE_BGRA || \
     (FORMAT) == IMAGE_RGBA || (FORMAT) == IMAGE_ABGR)

#ifdef WORDS_BIGENDIAN
# define IMAGE_FORMAT_NE(be, le) IMAGE_##be
#else
# define IMAGE_FORMAT_NE(be, le) IMAGE_##le
#endif

// Planar YUV 4:2:0, 12-bit, 1 plane for Y and 1 plane for UV
#define IMAGE_NV12   IMAGE_FOURCC('N','V','1','2')
// Planar YUV 4:2:0, 12-bit, 3 planes for Y V U
#define IMAGE_YV12   IMAGE_FOURCC('Y','V','1','2')
// Planar YUV 4:2:0, 12-bit, 3 planes for Y U V
#define IMAGE_IYUV   IMAGE_FOURCC('I','Y','U','V')
#define IMAGE_I420   IMAGE_FOURCC('I','4','2','0')
// Packed YUV 4:4:4, 32-bit, A Y U V
#define IMAGE_AYUV   IMAGE_FOURCC('A','Y','U','V')
// Packed YUV 4:2:2, 16-bit, Cb Y0 Cr Y1
#define IMAGE_UYVY   IMAGE_FOURCC('U','Y','V','Y')
// Packed YUV 4:2:2, 16-bit, Y0 Cb Y1 Cr
#define IMAGE_YUY2   IMAGE_FOURCC('Y','U','Y','2')
#define IMAGE_YUYV   IMAGE_FOURCC('Y','U','Y','V')
// Packed RGB 8:8:8, 32-bit, A R G B, native endian byte-order
#define IMAGE_RGB32  IMAGE_FORMAT_NE(RGBA, BGRA)
// Packed RGB 8:8:8, 32-bit, A R G B
#define IMAGE_ARGB   IMAGE_FOURCC('A','R','G','B')
// Packed RGB 8:8:8, 32-bit, B G R A
#define IMAGE_BGRA   IMAGE_FOURCC('B','G','R','A')
// Packed RGB 8:8:8, 32-bit, R G B A
  #define IMAGE_RGBA   IMAGE_FOURCC('R','G','B','A')
// Packed RGB 8:8:8, 32-bit, A B G R
#define IMAGE_ABGR   IMAGE_FOURCC('A','B','G','R')
struct avct_pair {
	AVCodecContext *p_ctx;
	struct vaapi_context_ffmpeg *p_vactx_ff;
	VAAPIContext *p_vactx;
};
static AVFrame *alloc_picture(enum AVPixelFormat pix_fmt, int width,
		int height) {
	AVFrame *picture;
	AVPicture *pp;
	uint8_t *picture_buf;
	int size;
	picture = av_frame_alloc();

	if (!picture)
		return NULL;
	size = avpicture_get_size(pix_fmt, width, height);
	picture_buf = (uint8_t *) av_malloc(size);
	if (!picture_buf) {
		av_free(picture);
		return NULL;
	}
	avpicture_fill((AVPicture *) picture, picture_buf, pix_fmt, width, height);
	return picture;
}
#define ARRAY_ELEMS(a) (sizeof(a) / sizeof((a)[0]))
#define ASSERT assert

void pkg_deliver(unsigned char *src[2],struct va_info *info)
{
	info->av_frame->data[0] = src[0];
	info->av_frame->data[1] = src[2];
	info->av_frame->data[2] = src[1];
	return;
}

int vaapi_check_status(VAStatus status, const char *msg) {
	if (status != VA_STATUS_SUCCESS) {
		return 0;
	}
	return 1;
}
static int release_image(VAImage *va_image, struct va_info *info) {
	VAAPIContext * const vaapi = info->vaapi_context;
	VAStatus status;

	status = vaUnmapBuffer(vaapi->display, va_image->buf);
	if (!vaapi_check_status(status, "vaUnmapBuffer()"))
		return -1;
	return 0;
}
static int set_display_attribute(VADisplayAttribType type, int value,
		struct va_info *info) {
	VAAPIContext * const vaapi = info->vaapi_context;
	VADisplayAttribute attr;
	VAStatus status;

	attr.type = type;
	attr.value = value;
	attr.flags = VA_DISPLAY_ATTRIB_SETTABLE;
	status = vaSetDisplayAttributes(vaapi->display, &attr, 1);
	if (!vaapi_check_status(status, "vaSetDisplayAttributes()"))
		return 1;
	return 0;
}
enum HWAccelType {
	HWACCEL_NONE, HWACCEL_VAAPI, HWACCEL_VDPAU, HWACCEL_XVBA
};
enum DisplayType {
	DISPLAY_X11 = 1, DISPLAY_GLX, DISPLAY_DRM
};
typedef struct _Size Size;

struct _Size {
	unsigned int width;
	unsigned int height;
};
typedef struct _Rectangle Rectangle;
struct _Rectangle {
	int x;
	int y;
	unsigned int width;
	unsigned int height;
};
enum TextureTarget {
	TEXTURE_TARGET_2D = 1, TEXTURE_TARGET_RECT
};
enum RotationMode {
	ROTATION_NONE = 0, ROTATION_90, ROTATION_180, ROTATION_270
};


enum PutImageMode {
	PUTIMAGE_NONE = 0, PUTIMAGE_OVERRIDE, PUTIMAGE_BLEND
};

static inline void ensure_bounds(Rectangle *r, unsigned int w, unsigned int h) {
	if (r->x < 0)
		r->x = 0;
	if (r->y < 0)
		r->y = 0;
	if (r->width > w - r->x)
		r->width = w - r->x;
	if (r->height > h - r->y)
		r->height = h - r->y;
}

static int has_display_attribute(VADisplayAttribType type,
		struct va_info *info)
{
	VAAPIContext * const vaapi = info->vaapi_context;
	int i;

	if (vaapi->display_attrs) {
		for (i = 0; i < vaapi->n_display_attrs; i++) {
			if (vaapi->display_attrs[i].type == type)
				return 0;
		}
	}
	return 1;
}

int vaapi_init(VADisplay display, struct va_info *info) {

	VAAPIContext *vaapi;
	int major_version, minor_version;
	int i, num_display_attrs, max_display_attrs;
	VADisplayAttribute *display_attrs = NULL;
	VAStatus status;
	if (info->vaapi_context)
		return 0;

	if (!display)
		goto error;

	status = vaInitialize(display, &major_version, &minor_version);

	if (!vaapi_check_status(status, "vaInitialize()"))
		goto error;
	// D(bug("VA API version %d.%d\n", major_version, minor_version));

	max_display_attrs = vaMaxNumDisplayAttributes(display);
	display_attrs = (VADisplayAttribute*) malloc(
			max_display_attrs * sizeof(display_attrs[0]));
	if (!display_attrs)
		goto error;
	num_display_attrs = 0; /* XXX: workaround old GMA500 bug */
	status = vaQueryDisplayAttributes(display, display_attrs,
			&num_display_attrs);
	if (!vaapi_check_status(status, "vaQueryDisplayAttributes()"))
		goto error;
	// D(bug("%d display attributes available\n", num_display_attrs));
	for (i = 0; i < num_display_attrs; i++) {
		VADisplayAttribute * const display_attr = &display_attrs[i];
	}


	if ((vaapi = (VAAPIContext *) calloc(1, sizeof(*vaapi))) == NULL)
		goto error;
	vaapi->display = display;
	vaapi->config_id = VA_INVALID_ID;
	vaapi->context_id = VA_INVALID_ID;
	vaapi->surface_id = VA_INVALID_ID;
	vaapi->subpic_image.image_id = VA_INVALID_ID;
	for (i = 0; i < ARRAY_ELEMS(vaapi->subpic_ids); i++)
		vaapi->subpic_ids[i] = VA_INVALID_ID;
	vaapi->pic_param_buf_id = VA_INVALID_ID;
	vaapi->iq_matrix_buf_id = VA_INVALID_ID;
	vaapi->bitplane_buf_id = VA_INVALID_ID;
	vaapi->huf_table_buf_id = VA_INVALID_ID;
	vaapi->display_attrs = display_attrs;
	vaapi->n_display_attrs = num_display_attrs;

	info->vaapi_context = vaapi;

	return 0;

	error: free(display_attrs);
	return -1;
}
static void destroy_buffers(VADisplay display, VABufferID *buffers,
		unsigned int n_buffers) {
	unsigned int i;
	for (i = 0; i < n_buffers; i++) {
		if (buffers[i] != VA_INVALID_ID) {
			vaDestroyBuffer(display, buffers[i]);
			buffers[i] = VA_INVALID_ID;
		}
	}
}
//int vaapi_exit(void)
//{
//    VAAPIContext * const vaapi = vaapi_context;
//    unsigned int i;

//    if (!vaapi)
//        return 0;

//#if USE_GLX
//    if (display_type() == DISPLAY_GLX)
//        vaapi_glx_destroy_surface();
//#endif

//    destroy_buffers(vaapi->display, &vaapi->pic_param_buf_id, 1);
//    destroy_buffers(vaapi->display, &vaapi->iq_matrix_buf_id, 1);
//    destroy_buffers(vaapi->display, &vaapi->bitplane_buf_id, 1);
//    destroy_buffers(vaapi->display, &vaapi->huf_table_buf_id, 1);
//    destroy_buffers(vaapi->display, vaapi->slice_buf_ids, vaapi->n_slice_buf_ids);

//    if (vaapi->subpic_flags) {
//        free(vaapi->subpic_flags);
//        vaapi->subpic_flags = NULL;
//    }

//    if (vaapi->subpic_formats) {
//        free(vaapi->subpic_formats);
//        vaapi->subpic_formats = NULL;
//        vaapi->n_subpic_formats = 0;
//    }

//    if (vaapi->image_formats) {
//        free(vaapi->image_formats);
//        vaapi->image_formats = NULL;
//        vaapi->n_image_formats = 0;
//    }

//    if (vaapi->entrypoints) {
//        free(vaapi->entrypoints);
//        vaapi->entrypoints = NULL;
//        vaapi->n_entrypoints = 0;
//    }

//    if (vaapi->profiles) {
//        free(vaapi->profiles);
//        vaapi->profiles = NULL;
//        vaapi->n_profiles = 0;
//    }

//    if (vaapi->slice_params) {
//        free(vaapi->slice_params);
//        vaapi->slice_params = NULL;
//        vaapi->slice_params_alloc = 0;
//        vaapi->n_slice_params = 0;
//    }

//    if (vaapi->slice_buf_ids) {
//        free(vaapi->slice_buf_ids);
//        vaapi->slice_buf_ids = NULL;
//        vaapi->n_slice_buf_ids = 0;
//    }

//    if (vaapi->subpic_image.image_id != VA_INVALID_ID) {
//        vaDestroyImage(vaapi->display, vaapi->subpic_image.image_id);
//        vaapi->subpic_image.image_id = VA_INVALID_ID;
//    }

//    for (i = 0; i < ARRAY_ELEMS(vaapi->subpic_ids); i++) {
//        if (vaapi->subpic_ids[i] != VA_INVALID_ID) {
//            vaDestroySubpicture(vaapi->display, vaapi->subpic_ids[i]);
//            vaapi->subpic_ids[i] = VA_INVALID_ID;
//        }
//    }

//    if (vaapi->context_id != VA_INVALID_ID) {
//        vaDestroyContext(vaapi->display, vaapi->context_id);
//        vaapi->context_id = VA_INVALID_ID;
//    }

//    if (vaapi->surface_id != VA_INVALID_ID) {
//        vaDestroySurfaces(vaapi->display, &vaapi->surface_id, 1);
//        vaapi->surface_id = VA_INVALID_ID;
//    }

//    if (vaapi->config_id != VA_INVALID_ID) {
//        vaDestroyConfig(vaapi->display, vaapi->config_id);
//        vaapi->config_id = VA_INVALID_ID;
//    }

//    if (vaapi->display_attrs) {
//        free(vaapi->display_attrs);
//        vaapi->display_attrs = NULL;
//        vaapi->n_display_attrs = 0;
//    }

//    if (vaapi->display) {
//        vaTerminate(vaapi->display);
//        vaapi->display = NULL;
//    }

//    free(vaapi_context);
//    return 0;
//}

void open_va(struct va_info *info) {
	VADisplay dpy;
	int fd = open("/dev/dri/card0", O_RDWR);
	dpy = vaGetDisplayDRM(fd);
	if (vaapi_init(dpy, info) < 0)
		;
	if ((info->vaapi_context_ffmpeg = (struct vaapi_context_ffmpeg *) calloc(1,
			sizeof(*info->vaapi_context_ffmpeg))) == NULL)
		;
	info->vaapi_context_ffmpeg->display = info->vaapi_context->display;
}

int vaapi_init_decoder(VAProfile profile, VAEntrypoint entrypoint,
		unsigned int picture_width, unsigned int picture_height,
		VAAPIContext * vaapi)
{
	VAConfigAttrib attrib;
	VAConfigID config_id = VA_INVALID_ID;
	VAContextID context_id = VA_INVALID_ID;
	VASurfaceID surface_id = VA_INVALID_ID;
	VAStatus status;

	if (!vaapi)
		return -1;
	if (vaapi->profile != profile || vaapi->entrypoint != entrypoint) {
		if (vaapi->config_id != VA_INVALID_ID)
			vaDestroyConfig(vaapi->display, vaapi->config_id);

		attrib.type = VAConfigAttribRTFormat;
		status = vaGetConfigAttributes(vaapi->display, profile, entrypoint,
				&attrib, 1);
		if (!vaapi_check_status(status, "vaGetConfigAttributes()"))
			return -1;
		if ((attrib.value & VA_RT_FORMAT_YUV420) == 0)
			return -1;

		status = vaCreateConfig(vaapi->display, profile, entrypoint, &attrib, 1,
				&config_id);
		if (!vaapi_check_status(status, "vaCreateConfig()"))
			return -1;
	} else
		config_id = vaapi->config_id;

	if (vaapi->picture_width != picture_width
			|| vaapi->picture_height != picture_height) {
		if (vaapi->surface_id != VA_INVALID_ID)
			vaDestroySurfaces(vaapi->display, &vaapi->surface_id, 1);

		status = vaCreateSurfaces(vaapi->display, picture_width, picture_height,
				VA_RT_FORMAT_YUV420, 1, &surface_id);
		if (!vaapi_check_status(status, "vaCreateSurfaces()"))
			return -1;

		if (vaapi->context_id != VA_INVALID_ID)
			vaDestroyContext(vaapi->display, vaapi->context_id);

		status = vaCreateContext(vaapi->display, config_id, picture_width,
				picture_height,
				VA_PROGRESSIVE, &surface_id, 1, &context_id);
		if (!vaapi_check_status(status, "vaCreateContext()"))
			return -1;
	} else {
		context_id = vaapi->context_id;
		surface_id = vaapi->surface_id;
	}

	vaapi->config_id = config_id;
	vaapi->context_id = context_id;
	vaapi->surface_id = surface_id;
	vaapi->profile = profile;
	vaapi->entrypoint = entrypoint;
	vaapi->picture_width = picture_width;
	vaapi->picture_height = picture_height;
	return 0;
}
static enum PixelFormat get_format(struct AVCodecContext *avctx,
		const enum PixelFormat *fmt)
{
	struct va_info *p_info = (struct va_info *) avctx->opaque;
	int i, profile;
	for (i = 0; fmt[i] != PIX_FMT_NONE; i++) {
		if (fmt[i] != PIX_FMT_VAAPI_VLD)
			continue;
		switch (avctx->codec_id) {
		case CODEC_ID_MPEG2VIDEO:
			profile = VAProfileMPEG2Main;
			break;
		case CODEC_ID_MPEG4:
		case CODEC_ID_H263:
			profile = VAProfileMPEG4AdvancedSimple;
			break;
		case CODEC_ID_H264:
			//	profile = VAProfileH264High;
			profile = VAProfileH264Main;
			break;
		case CODEC_ID_WMV3:
			profile = VAProfileVC1Main;
			break;
		case CODEC_ID_VC1:
			profile = VAProfileVC1Advanced;
			break;
		default:
			profile = -1;
			break;
		}
		if (profile >= 0) {
			VAAPIContext * vaapi = p_info->vaapi_context;
			struct vaapi_context_ffmpeg *vaapi_context_ffmpeg =
					p_info->vaapi_context_ffmpeg;
			if (vaapi_init_decoder((VAProfile) profile, VAEntrypointVLD,
					avctx->width, avctx->height, vaapi) == 0) {
				vaapi_context_ffmpeg->config_id = vaapi->config_id;
				vaapi_context_ffmpeg->context_id = vaapi->context_id;
				avctx->hwaccel_context = vaapi_context_ffmpeg;
				return fmt[i];
			}
		}
	}
	return PIX_FMT_NONE;
}
#include "common.h"
static int get_flg=0;
static int rls_flg=0;

static int get_buffer(struct AVCodecContext *avctx, AVFrame *pic)
{
	struct va_info *p_info = (struct va_info *) avctx->opaque;
	VAAPIContext * const vaapi = p_info->vaapi_context;
	VASurfaceID surface;
	int status = vaCreateSurfaces(vaapi->display, avctx->width, avctx->height,
			VA_RT_FORMAT_YUV420, 1, &surface);
	if (!vaapi_check_status(status, "vaCreateSurfaces()"))
		return -1;
	vaapi->surface_id_old=vaapi->surface_id;
	vaapi->surface_id = surface;

	pic->data[0] = (uint8_t *) surface;
//	pic->data[1] = NULL;
//	pic->data[2] = NULL;
	pic->data[3] = (uint8_t *) surface;


//	print("get buffer frm  %p",pic);
//	pic->width = SRC_PITCH;
	return 0;
}
static void release_buffer(struct AVCodecContext *avctx, AVFrame *pic)
{
	struct va_info *p_info = (struct va_info *) avctx->opaque;
	VAAPIContext * const vaapi = p_info->vaapi_context;
//	VASurfaceID surface=(VASurfaceID)(uintptr_t)pic->data[0];
	VASurfaceID surface=	vaapi->surface_id_old;

 	vaDestroySurfaces(vaapi->display, &surface, 1);
//	print("relse buffer frm  %p",pic);
	pic->data[0] = NULL;
	pic->data[1] = NULL;
	pic->data[2] = NULL;
	pic->data[3] = NULL;
}
int ffmpeg_init_context(AVCodecContext **avctx)
{
	(*avctx)->thread_count = 1;
	(*avctx)->get_format = get_format;
	(*avctx)->get_buffer = get_buffer;
	(*avctx)->reget_buffer = get_buffer;
	(*avctx)->release_buffer = release_buffer;
	return 0;
}
static const uint32_t image_formats[] = {
		VA_FOURCC('Y', 'V', '1', '2'),
		VA_FOURCC('N', 'V', '1', '2'),
		VA_FOURCC('U', 'Y', 'V', 'Y'),
		VA_FOURCC('Y', 'U', 'Y', 'V'),
		VA_FOURCC('A', 'R', 'G', 'B'),
		VA_FOURCC('A', 'B', 'G', 'R'),
		VA_FOURCC('B', 'G', 'R', 'A'),
		VA_FOURCC('R', 'G', 'B', 'A'),
		0
};
static int is_vaapi_rgb_format(const VAImageFormat *image_format)
{
	switch (image_format->fourcc) {
	case VA_FOURCC('A', 'R', 'G', 'B'):
	case VA_FOURCC('A', 'B', 'G', 'R'):
	case VA_FOURCC('B', 'G', 'R', 'A'):
	case VA_FOURCC('R', 'G', 'B', 'A'):
		return 1;
	}
	return 0;
}

static int bind_image(VAImage *va_image, Image1 *image, struct va_info *info)
{
	VAAPIContext * const vaapi = info->vaapi_context;
	VAImageFormat * const va_format = &va_image->format;
	VAStatus status;
	void *va_image_data;
	unsigned int i;

	if (va_image->num_planes > MAX_IMAGE_PLANES)
		return -1;

	status = vaMapBuffer(vaapi->display, va_image->buf, &va_image_data);
	if (!vaapi_check_status(status, "vaMapBuffer()"))
		return -1;

	memset(image, 0, sizeof(*image));
	image->width = va_image->width;
	image->height = va_image->height;
	image->num_planes = va_image->num_planes;
	for (i = 0; i < va_image->num_planes; i++) {
		image->pixels[i] = (uint8_t *) va_image_data + va_image->offsets[i];
		image->pitches[i] = va_image->pitches[i];
	}
	return 0;
}

static uint32_t get_vaapi_format(uint32_t format)
{
	uint32_t fourcc;

	/* Only translate image formats we support */
	switch (format) {
	case IMAGE_NV12:
	case IMAGE_YV12:
	case IMAGE_IYUV:
	case IMAGE_I420:
	case IMAGE_AYUV:
	case IMAGE_UYVY:
	case IMAGE_YUY2:
	case IMAGE_YUYV:
	case IMAGE_ARGB:
	case IMAGE_BGRA:
	case IMAGE_RGBA:
	case IMAGE_ABGR:
		fourcc = format;
		break;
	default:
		fourcc = 0;
		break;
	}
	return fourcc;
}

const char *string_of_FOURCC(uint32_t fourcc)
{
	static int buf;
	static char str[2][5];
	buf ^= 1;
	str[buf][0] = fourcc;
	str[buf][1] = fourcc >> 8;
	str[buf][2] = fourcc >> 16;
	str[buf][3] = fourcc >> 24;
	str[buf][4] = '\0';
	return str[buf];
}

static inline const char *string_of_VAImageFormat(VAImageFormat *imgfmt)
{
	return string_of_FOURCC(imgfmt->fourcc);
}

static int get_image_format(VAAPIContext *vaapi, uint32_t fourcc,
		VAImageFormat **image_format)
{
	VAStatus status;
	int i;

	if (image_format)
		*image_format = NULL;

	if (!vaapi->image_formats || vaapi->n_image_formats == 0) {
		vaapi->image_formats = (VAImageFormat*) calloc(
				vaMaxNumImageFormats(vaapi->display),
				sizeof(vaapi->image_formats[0]));
		if (!vaapi->image_formats)
			return 0;

		status = vaQueryImageFormats(vaapi->display, vaapi->image_formats,
				&vaapi->n_image_formats);
		if (!vaapi_check_status(status, "vaQueryImageFormats()"))
			return 0;
	}

	for (i = 0; i < vaapi->n_image_formats; i++) {
		if (vaapi->image_formats[i].fourcc == fourcc) {
			if (image_format)
				*image_format = &vaapi->image_formats[i];
			return 1;
		}
	}
	return 0;
}

int vaapi_decode_to_image(struct va_info *info)
{
	VAAPIContext * const vaapi = info->vaapi_context;
	VASurfaceID surface = vaapi->surface_id;
	VAImage image;
	VAImageFormat *image_format = NULL;
	VAStatus status;
	Image1 bound_image;
	int i, is_bound_image = 0, is_derived_image = 0, error = -1;

	image.image_id = VA_INVALID_ID;
	image.buf = VA_INVALID_ID;
	if (!image_format) {
		for (i = 0; image_formats[i] != 0; i++) {
			if (get_image_format(vaapi, image_formats[i], &image_format))
				break;
		}
	}

	if (!image_format)
		goto end;
	if (!is_derived_image) {
		status = vaCreateImage(vaapi->display, image_format,
				vaapi->picture_width, vaapi->picture_height, &image);
		if (!vaapi_check_status(status, "vaCreateImage()"))
			goto end;
		VARectangle src_rect;

		src_rect.x = 0;
		src_rect.y = 0;
		src_rect.width = vaapi->picture_width;
		src_rect.height = vaapi->picture_height;

		status = vaGetImage(vaapi->display, vaapi->surface_id, src_rect.x,
				src_rect.y, src_rect.width, src_rect.height, image.image_id);
		if (!vaapi_check_status(status, "vaGetImage()")) {
			vaDestroyImage(vaapi->display, image.image_id);
			goto end;
		}
	}

	if (bind_image(&image, &bound_image, info) < 0)
		goto end;
	is_bound_image = 1;
	error = 0;
	end: pkg_deliver(bound_image.pixels, info);
	if (is_bound_image) {
		if (release_image(&image, info) < 0)
			error = -1;
	}

	if (image.image_id != VA_INVALID_ID) {
		status = vaDestroyImage(vaapi->display, image.image_id);
		if (!vaapi_check_status(status, "vaDestroyImage()"))
			error = -1;
	}
	return error;
}












