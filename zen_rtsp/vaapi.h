#ifndef TYPES_H
#define TYPES_H
#ifdef __cplusplus
extern "C" {
#endif
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libswscale/swscale.h>
#include <libavutil/frame.h>
#include <unistd.h>
#include <pthread.h>
#include <va/va.h>
#include <va/va_drm.h>
#ifdef __cplusplus
}
#endif

typedef struct _VAAPIContext VAAPIContext;
struct _VAAPIContext {
    VADisplay           display;
    VADisplayAttribute *display_attrs;
    int                 n_display_attrs;
    VAConfigID          config_id;
    VAContextID         context_id;
    VASurfaceID         surface_id;
    VASurfaceID         surface_id_old;
      VASubpictureID      subpic_ids[5];
    VAImage             subpic_image;
    VAProfile           profile;
    VAProfile          *profiles;
    int                 n_profiles;
    VAEntrypoint        entrypoint;
    VAEntrypoint       *entrypoints;
    int                 n_entrypoints;
    VAImageFormat      *image_formats;
    int                 n_image_formats;
    VAImageFormat      *subpic_formats;
    unsigned int       *subpic_flags;
    unsigned int        n_subpic_formats;
    unsigned int        picture_width;
    unsigned int        picture_height;
    VABufferID          pic_param_buf_id;
    VABufferID          iq_matrix_buf_id;
    VABufferID          bitplane_buf_id;
    VABufferID          huf_table_buf_id;
    VABufferID         *slice_buf_ids;
    unsigned int        n_slice_buf_ids;
    unsigned int        slice_buf_ids_alloc;
    void               *slice_params;
    unsigned int        slice_param_size;
    unsigned int        n_slice_params;
    unsigned int        slice_params_alloc;
    const uint8_t      *slice_data;
    unsigned int        slice_data_size;
    int                 use_glx_copy;
    void               *glx_surface;
};
struct vaapi_context_ffmpeg {
    /**
     * Window system dependent data
     *
     * - encoding: unused
     * - decoding: Set by user
     */
    void *display;

    /**
     * Configuration ID
     *
     * - encoding: unused
     * - decoding: Set by user
     */
    uint32_t config_id;

    /**
     * Context ID (video decode pipeline)
     *
     * - encoding: unused
     * - decoding: Set by user
     */
    uint32_t context_id;

    /**
     * VAPictureParameterBuffer ID
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    uint32_t pic_param_buf_id;

    /**
     * VAIQMatrixBuffer ID
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    uint32_t iq_matrix_buf_id;

    /**
     * VABitPlaneBuffer ID (for VC-1 decoding)
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    uint32_t bitplane_buf_id;

    /**
     * Slice parameter/data buffer IDs
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    uint32_t *slice_buf_ids;

    /**
     * Number of effective slice buffer IDs to send to the HW
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    unsigned int n_slice_buf_ids;

    /**
     * Size of pre-allocated slice_buf_ids
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    unsigned int slice_buf_ids_alloc;

    /**
     * Pointer to VASliceParameterBuffers
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    void *slice_params;

    /**
     * Size of a VASliceParameterBuffer element
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    unsigned int slice_param_size;

    /**
     * Size of pre-allocated slice_params
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    unsigned int slice_params_alloc;

    /**
     * Number of slices currently filled in
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    unsigned int slice_count;

    /**
     * Pointer to slice data buffer base
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    const uint8_t *slice_data;

    /**
     * Current size of slice data
     *
     * - encoding: unused
     * - decoding: Set by libavcodec
     */
    uint32_t slice_data_size;
};
struct va_info{
	AVFrame *av_frame;
	//char *udp_addr[4];
    struct vaapi_context_ffmpeg *vaapi_context_ffmpeg;
    VAAPIContext *vaapi_context;
};
#endif
