TEMPLATE = app
CONFIG += console c++11
CONFIG -= app_bundle
CONFIG -= qt
INCLUDEPATH+=../alg-mutilthread
SOURCES +=  \
    ../alg-mutilthread/cascadedetect.cpp \
    ../alg-mutilthread/Detector2.cpp \
    ../alg-mutilthread/m_arith.cpp \
    ../alg-mutilthread/VC_DSP_TRANSFORM.cpp \
    ../server/test/get_gateway.c \
    ../server/test/test.c \
    ../server/test/test1.c \
    ../server/catcharp.c \
    ../server/main.c \
    ../server/sendarp.c \
    ../server/zenlog.c \
    ../cam_alg.c \
    ../cam_codec.c \
    ../cam_net.c \
    ../camera_service.c \
    ../client_config_main.c \
    ../client_file.c \
    ../client_net.c \
    ../client_obj.c \
    ../common.c \
    ../csocket.c \
    ../file_op.c \
    ../glplayer.c \
    ../h264_stream_file.c \
    ../hw_vaapi.c \
    ../main.c \
    ../sig_service.c \
    ../tcp_server.c

HEADERS += \
    ../alg-mutilthread/include/serial/tbb/parallel_for.h \
    ../alg-mutilthread/include/serial/tbb/tbb_annotate.h \
    ../alg-mutilthread/include/tbb/compat/ppl.h \
    ../alg-mutilthread/include/tbb/internal/_aggregator_impl.h \
    ../alg-mutilthread/include/tbb/internal/_concurrent_queue_impl.h \
    ../alg-mutilthread/include/tbb/internal/_concurrent_unordered_impl.h \
    ../alg-mutilthread/include/tbb/internal/_flow_graph_impl.h \
    ../alg-mutilthread/include/tbb/internal/_flow_graph_indexer_impl.h \
    ../alg-mutilthread/include/tbb/internal/_flow_graph_item_buffer_impl.h \
    ../alg-mutilthread/include/tbb/internal/_flow_graph_join_impl.h \
    ../alg-mutilthread/include/tbb/internal/_flow_graph_node_impl.h \
    ../alg-mutilthread/include/tbb/internal/_flow_graph_tagged_buffer_impl.h \
    ../alg-mutilthread/include/tbb/internal/_flow_graph_trace_impl.h \
    ../alg-mutilthread/include/tbb/internal/_flow_graph_types_impl.h \
    ../alg-mutilthread/include/tbb/internal/_mutex_padding.h \
    ../alg-mutilthread/include/tbb/internal/_range_iterator.h \
    ../alg-mutilthread/include/tbb/internal/_tbb_hash_compare_impl.h \
    ../alg-mutilthread/include/tbb/internal/_tbb_strings.h \
    ../alg-mutilthread/include/tbb/internal/_tbb_windef.h \
    ../alg-mutilthread/include/tbb/internal/_template_helpers.h \
    ../alg-mutilthread/include/tbb/internal/_x86_eliding_mutex_impl.h \
    ../alg-mutilthread/include/tbb/internal/_x86_rtm_rw_mutex_impl.h \
    ../alg-mutilthread/include/tbb/machine/gcc_armv7.h \
    ../alg-mutilthread/include/tbb/machine/gcc_generic.h \
    ../alg-mutilthread/include/tbb/machine/gcc_ia32_common.h \
    ../alg-mutilthread/include/tbb/machine/gcc_itsx.h \
    ../alg-mutilthread/include/tbb/machine/ibm_aix51.h \
    ../alg-mutilthread/include/tbb/machine/icc_generic.h \
    ../alg-mutilthread/include/tbb/machine/linux_common.h \
    ../alg-mutilthread/include/tbb/machine/linux_ia32.h \
    ../alg-mutilthread/include/tbb/machine/linux_ia64.h \
    ../alg-mutilthread/include/tbb/machine/linux_intel64.h \
    ../alg-mutilthread/include/tbb/machine/mac_ppc.h \
    ../alg-mutilthread/include/tbb/machine/macos_common.h \
    ../alg-mutilthread/include/tbb/machine/mic_common.h \
    ../alg-mutilthread/include/tbb/machine/msvc_armv7.h \
    ../alg-mutilthread/include/tbb/machine/msvc_ia32_common.h \
    ../alg-mutilthread/include/tbb/machine/sunos_sparc.h \
    ../alg-mutilthread/include/tbb/machine/windows_api.h \
    ../alg-mutilthread/include/tbb/machine/windows_ia32.h \
    ../alg-mutilthread/include/tbb/machine/windows_intel64.h \
    ../alg-mutilthread/include/tbb/machine/xbox360_ppc.h \
    ../alg-mutilthread/include/tbb/aggregator.h \
    ../alg-mutilthread/include/tbb/aligned_space.h \
    ../alg-mutilthread/include/tbb/atomic.h \
    ../alg-mutilthread/include/tbb/blocked_range.h \
    ../alg-mutilthread/include/tbb/blocked_range2d.h \
    ../alg-mutilthread/include/tbb/blocked_range3d.h \
    ../alg-mutilthread/include/tbb/cache_aligned_allocator.h \
    ../alg-mutilthread/include/tbb/combinable.h \
    ../alg-mutilthread/include/tbb/concurrent_hash_map.h \
    ../alg-mutilthread/include/tbb/concurrent_lru_cache.h \
    ../alg-mutilthread/include/tbb/concurrent_priority_queue.h \
    ../alg-mutilthread/include/tbb/concurrent_queue.h \
    ../alg-mutilthread/include/tbb/concurrent_unordered_map.h \
    ../alg-mutilthread/include/tbb/concurrent_unordered_set.h \
    ../alg-mutilthread/include/tbb/concurrent_vector.h \
    ../alg-mutilthread/include/tbb/critical_section.h \
    ../alg-mutilthread/include/tbb/enumerable_thread_specific.h \
    ../alg-mutilthread/include/tbb/flow_graph.h \
    ../alg-mutilthread/include/tbb/flow_graph_opencl_node.h \
    ../alg-mutilthread/include/tbb/global_control.h \
    ../alg-mutilthread/include/tbb/memory_pool.h \
    ../alg-mutilthread/include/tbb/mutex.h \
    ../alg-mutilthread/include/tbb/null_mutex.h \
    ../alg-mutilthread/include/tbb/null_rw_mutex.h \
    ../alg-mutilthread/include/tbb/parallel_do.h \
    ../alg-mutilthread/include/tbb/parallel_for.h \
    ../alg-mutilthread/include/tbb/parallel_for_each.h \
    ../alg-mutilthread/include/tbb/parallel_invoke.h \
    ../alg-mutilthread/include/tbb/parallel_reduce.h \
    ../alg-mutilthread/include/tbb/parallel_scan.h \
    ../alg-mutilthread/include/tbb/parallel_sort.h \
    ../alg-mutilthread/include/tbb/parallel_while.h \
    ../alg-mutilthread/include/tbb/partitioner.h \
    ../alg-mutilthread/include/tbb/pipeline.h \
    ../alg-mutilthread/include/tbb/queuing_mutex.h \
    ../alg-mutilthread/include/tbb/queuing_rw_mutex.h \
    ../alg-mutilthread/include/tbb/reader_writer_lock.h \
    ../alg-mutilthread/include/tbb/recursive_mutex.h \
    ../alg-mutilthread/include/tbb/runtime_loader.h \
    ../alg-mutilthread/include/tbb/scalable_allocator.h \
    ../alg-mutilthread/include/tbb/spin_mutex.h \
    ../alg-mutilthread/include/tbb/spin_rw_mutex.h \
    ../alg-mutilthread/include/tbb/task.h \
    ../alg-mutilthread/include/tbb/task_arena.h \
    ../alg-mutilthread/include/tbb/task_group.h \
    ../alg-mutilthread/include/tbb/task_scheduler_init.h \
    ../alg-mutilthread/include/tbb/task_scheduler_observer.h \
    ../alg-mutilthread/include/tbb/tbb.h \
    ../alg-mutilthread/include/tbb/tbb_allocator.h \
    ../alg-mutilthread/include/tbb/tbb_config.h \
    ../alg-mutilthread/include/tbb/tbb_exception.h \
    ../alg-mutilthread/include/tbb/tbb_machine.h \
    ../alg-mutilthread/include/tbb/tbb_profiling.h \
    ../alg-mutilthread/include/tbb/tbb_stddef.h \
    ../alg-mutilthread/include/tbb/tbb_thread.h \
    ../alg-mutilthread/include/tbb/tbbmalloc_proxy.h \
    ../alg-mutilthread/include/tbb/tick_count.h \
    ../alg-mutilthread/opencv2/calib3d/calib3d.hpp \
    ../alg-mutilthread/opencv2/contrib/contrib.hpp \
    ../alg-mutilthread/opencv2/contrib/detection_based_tracker.hpp \
    ../alg-mutilthread/opencv2/contrib/hybridtracker.hpp \
    ../alg-mutilthread/opencv2/contrib/openfabmap.hpp \
    ../alg-mutilthread/opencv2/contrib/retina.hpp \
    ../alg-mutilthread/opencv2/core/affine.hpp \
    ../alg-mutilthread/opencv2/core/core.hpp \
    ../alg-mutilthread/opencv2/core/core_c.h \
    ../alg-mutilthread/opencv2/core/cuda_devptrs.hpp \
    ../alg-mutilthread/opencv2/core/devmem2d.hpp \
    ../alg-mutilthread/opencv2/core/eigen.hpp \
    ../alg-mutilthread/opencv2/core/gpumat.hpp \
    ../alg-mutilthread/opencv2/core/internal.hpp \
    ../alg-mutilthread/opencv2/core/mat.hpp \
    ../alg-mutilthread/opencv2/core/opengl_interop.hpp \
    ../alg-mutilthread/opencv2/core/opengl_interop_deprecated.hpp \
    ../alg-mutilthread/opencv2/core/operations.hpp \
    ../alg-mutilthread/opencv2/core/types_c.h \
    ../alg-mutilthread/opencv2/core/version.hpp \
    ../alg-mutilthread/opencv2/core/wimage.hpp \
    ../alg-mutilthread/opencv2/features2d/features2d.hpp \
    ../alg-mutilthread/opencv2/flann/all_indices.h \
    ../alg-mutilthread/opencv2/flann/allocator.h \
    ../alg-mutilthread/opencv2/flann/any.h \
    ../alg-mutilthread/opencv2/flann/autotuned_index.h \
    ../alg-mutilthread/opencv2/flann/composite_index.h \
    ../alg-mutilthread/opencv2/flann/config.h \
    ../alg-mutilthread/opencv2/flann/defines.h \
    ../alg-mutilthread/opencv2/flann/dist.h \
    ../alg-mutilthread/opencv2/flann/dummy.h \
    ../alg-mutilthread/opencv2/flann/dynamic_bitset.h \
    ../alg-mutilthread/opencv2/flann/flann.hpp \
    ../alg-mutilthread/opencv2/flann/flann_base.hpp \
    ../alg-mutilthread/opencv2/flann/general.h \
    ../alg-mutilthread/opencv2/flann/ground_truth.h \
    ../alg-mutilthread/opencv2/flann/hdf5.h \
    ../alg-mutilthread/opencv2/flann/heap.h \
    ../alg-mutilthread/opencv2/flann/hierarchical_clustering_index.h \
    ../alg-mutilthread/opencv2/flann/index_testing.h \
    ../alg-mutilthread/opencv2/flann/kdtree_index.h \
    ../alg-mutilthread/opencv2/flann/kdtree_single_index.h \
    ../alg-mutilthread/opencv2/flann/kmeans_index.h \
    ../alg-mutilthread/opencv2/flann/linear_index.h \
    ../alg-mutilthread/opencv2/flann/logger.h \
    ../alg-mutilthread/opencv2/flann/lsh_index.h \
    ../alg-mutilthread/opencv2/flann/lsh_table.h \
    ../alg-mutilthread/opencv2/flann/matrix.h \
    ../alg-mutilthread/opencv2/flann/miniflann.hpp \
    ../alg-mutilthread/opencv2/flann/nn_index.h \
    ../alg-mutilthread/opencv2/flann/object_factory.h \
    ../alg-mutilthread/opencv2/flann/params.h \
    ../alg-mutilthread/opencv2/flann/random.h \
    ../alg-mutilthread/opencv2/flann/result_set.h \
    ../alg-mutilthread/opencv2/flann/sampling.h \
    ../alg-mutilthread/opencv2/flann/saving.h \
    ../alg-mutilthread/opencv2/flann/simplex_downhill.h \
    ../alg-mutilthread/opencv2/flann/timer.h \
    ../alg-mutilthread/opencv2/gpu/device/detail/color_detail.hpp \
    ../alg-mutilthread/opencv2/gpu/device/detail/reduce.hpp \
    ../alg-mutilthread/opencv2/gpu/device/detail/reduce_key_val.hpp \
    ../alg-mutilthread/opencv2/gpu/device/detail/transform_detail.hpp \
    ../alg-mutilthread/opencv2/gpu/device/detail/type_traits_detail.hpp \
    ../alg-mutilthread/opencv2/gpu/device/detail/vec_distance_detail.hpp \
    ../alg-mutilthread/opencv2/gpu/device/block.hpp \
    ../alg-mutilthread/opencv2/gpu/device/border_interpolate.hpp \
    ../alg-mutilthread/opencv2/gpu/device/color.hpp \
    ../alg-mutilthread/opencv2/gpu/device/common.hpp \
    ../alg-mutilthread/opencv2/gpu/device/datamov_utils.hpp \
    ../alg-mutilthread/opencv2/gpu/device/dynamic_smem.hpp \
    ../alg-mutilthread/opencv2/gpu/device/emulation.hpp \
    ../alg-mutilthread/opencv2/gpu/device/filters.hpp \
    ../alg-mutilthread/opencv2/gpu/device/funcattrib.hpp \
    ../alg-mutilthread/opencv2/gpu/device/functional.hpp \
    ../alg-mutilthread/opencv2/gpu/device/limits.hpp \
    ../alg-mutilthread/opencv2/gpu/device/reduce.hpp \
    ../alg-mutilthread/opencv2/gpu/device/saturate_cast.hpp \
    ../alg-mutilthread/opencv2/gpu/device/scan.hpp \
    ../alg-mutilthread/opencv2/gpu/device/simd_functions.hpp \
    ../alg-mutilthread/opencv2/gpu/device/static_check.hpp \
    ../alg-mutilthread/opencv2/gpu/device/transform.hpp \
    ../alg-mutilthread/opencv2/gpu/device/type_traits.hpp \
    ../alg-mutilthread/opencv2/gpu/device/utility.hpp \
    ../alg-mutilthread/opencv2/gpu/device/vec_distance.hpp \
    ../alg-mutilthread/opencv2/gpu/device/vec_math.hpp \
    ../alg-mutilthread/opencv2/gpu/device/vec_traits.hpp \
    ../alg-mutilthread/opencv2/gpu/device/warp.hpp \
    ../alg-mutilthread/opencv2/gpu/device/warp_reduce.hpp \
    ../alg-mutilthread/opencv2/gpu/device/warp_shuffle.hpp \
    ../alg-mutilthread/opencv2/gpu/devmem2d.hpp \
    ../alg-mutilthread/opencv2/gpu/gpu.hpp \
    ../alg-mutilthread/opencv2/gpu/gpumat.hpp \
    ../alg-mutilthread/opencv2/gpu/stream_accessor.hpp \
    ../alg-mutilthread/opencv2/highgui/cap_ios.h \
    ../alg-mutilthread/opencv2/highgui/highgui.hpp \
    ../alg-mutilthread/opencv2/highgui/highgui_c.h \
    ../alg-mutilthread/opencv2/highgui/ios.h \
    ../alg-mutilthread/opencv2/imgproc/imgproc.hpp \
    ../alg-mutilthread/opencv2/imgproc/imgproc_c.h \
    ../alg-mutilthread/opencv2/imgproc/types_c.h \
    ../alg-mutilthread/opencv2/legacy/blobtrack.hpp \
    ../alg-mutilthread/opencv2/legacy/compat.hpp \
    ../alg-mutilthread/opencv2/legacy/legacy.hpp \
    ../alg-mutilthread/opencv2/legacy/streams.hpp \
    ../alg-mutilthread/opencv2/ml/ml.hpp \
    ../alg-mutilthread/opencv2/nonfree/features2d.hpp \
    ../alg-mutilthread/opencv2/nonfree/gpu.hpp \
    ../alg-mutilthread/opencv2/nonfree/nonfree.hpp \
    ../alg-mutilthread/opencv2/nonfree/ocl.hpp \
    ../alg-mutilthread/opencv2/objdetect/objdetect.hpp \
    ../alg-mutilthread/opencv2/ocl/matrix_operations.hpp \
    ../alg-mutilthread/opencv2/ocl/ocl.hpp \
    ../alg-mutilthread/opencv2/photo/photo.hpp \
    ../alg-mutilthread/opencv2/photo/photo_c.h \
    ../alg-mutilthread/opencv2/stitching/detail/autocalib.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/blenders.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/camera.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/exposure_compensate.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/matchers.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/motion_estimators.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/seam_finders.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/util.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/util_inl.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/warpers.hpp \
    ../alg-mutilthread/opencv2/stitching/detail/warpers_inl.hpp \
    ../alg-mutilthread/opencv2/stitching/stitcher.hpp \
    ../alg-mutilthread/opencv2/stitching/warpers.hpp \
    ../alg-mutilthread/opencv2/superres/optical_flow.hpp \
    ../alg-mutilthread/opencv2/superres/superres.hpp \
    ../alg-mutilthread/opencv2/ts/gpu_perf.hpp \
    ../alg-mutilthread/opencv2/ts/gpu_test.hpp \
    ../alg-mutilthread/opencv2/ts/ts.hpp \
    ../alg-mutilthread/opencv2/ts/ts_gtest.h \
    ../alg-mutilthread/opencv2/ts/ts_perf.hpp \
    ../alg-mutilthread/opencv2/video/background_segm.hpp \
    ../alg-mutilthread/opencv2/video/tracking.hpp \
    ../alg-mutilthread/opencv2/video/video.hpp \
    ../alg-mutilthread/opencv2/videostab/deblurring.hpp \
    ../alg-mutilthread/opencv2/videostab/fast_marching.hpp \
    ../alg-mutilthread/opencv2/videostab/fast_marching_inl.hpp \
    ../alg-mutilthread/opencv2/videostab/frame_source.hpp \
    ../alg-mutilthread/opencv2/videostab/global_motion.hpp \
    ../alg-mutilthread/opencv2/videostab/inpainting.hpp \
    ../alg-mutilthread/opencv2/videostab/log.hpp \
    ../alg-mutilthread/opencv2/videostab/motion_stabilizing.hpp \
    ../alg-mutilthread/opencv2/videostab/optical_flow.hpp \
    ../alg-mutilthread/opencv2/videostab/stabilizer.hpp \
    ../alg-mutilthread/opencv2/videostab/videostab.hpp \
    ../alg-mutilthread/opencv2/opencv.hpp \
    ../alg-mutilthread/opencv2/opencv_modules.hpp \
    ../alg-mutilthread/algorithm.h \
    ../alg-mutilthread/cascade_day_xml.h \
    ../alg-mutilthread/cascade_dusk_xml.h \
    ../alg-mutilthread/cascade_night_xml.h \
    ../alg-mutilthread/cascade_xml.h \
    ../alg-mutilthread/cascadedetect.h \
    ../alg-mutilthread/core.hpp \
    ../alg-mutilthread/Detector2.h \
    ../alg-mutilthread/DSPARMProto.h \
    ../alg-mutilthread/imgproc.hpp \
    ../alg-mutilthread/m_arith.h \
    ../alg-mutilthread/opencv.hpp \
    ../include/fvd_extra_include/GL/internal/dri_interface.h \
    ../include/fvd_extra_include/GL/internal/glcore.h \
    ../include/fvd_extra_include/GL/freeglut.h \
    ../include/fvd_extra_include/GL/freeglut_ext.h \
    ../include/fvd_extra_include/GL/freeglut_std.h \
    ../include/fvd_extra_include/GL/gl.h \
    ../include/fvd_extra_include/GL/gl_mangle.h \
    ../include/fvd_extra_include/GL/glew.h \
    ../include/fvd_extra_include/GL/glext.h \
    ../include/fvd_extra_include/GL/glu.h \
    ../include/fvd_extra_include/GL/glu_mangle.h \
    ../include/fvd_extra_include/GL/glut.h \
    ../include/fvd_extra_include/GL/glx.h \
    ../include/fvd_extra_include/GL/glx_mangle.h \
    ../include/fvd_extra_include/GL/glxew.h \
    ../include/fvd_extra_include/GL/glxext.h \
    ../include/fvd_extra_include/GL/glxint.h \
    ../include/fvd_extra_include/GL/glxmd.h \
    ../include/fvd_extra_include/GL/glxproto.h \
    ../include/fvd_extra_include/GL/glxtokens.h \
    ../include/fvd_extra_include/GL/wglew.h \
    ../include/fvd_extra_include/libavcodec/avcodec.h \
    ../include/fvd_extra_include/libavcodec/avfft.h \
    ../include/fvd_extra_include/libavcodec/dv_profile.h \
    ../include/fvd_extra_include/libavcodec/dxva2.h \
    ../include/fvd_extra_include/libavcodec/old_codec_ids.h \
    ../include/fvd_extra_include/libavcodec/vaapi.h \
    ../include/fvd_extra_include/libavcodec/vda.h \
    ../include/fvd_extra_include/libavcodec/vdpau.h \
    ../include/fvd_extra_include/libavcodec/version.h \
    ../include/fvd_extra_include/libavcodec/vorbis_parser.h \
    ../include/fvd_extra_include/libavcodec/xvmc.h \
    ../include/fvd_extra_include/libavdevice/avdevice.h \
    ../include/fvd_extra_include/libavdevice/version.h \
    ../include/fvd_extra_include/libavfilter/asrc_abuffer.h \
    ../include/fvd_extra_include/libavfilter/avcodec.h \
    ../include/fvd_extra_include/libavfilter/avfilter.h \
    ../include/fvd_extra_include/libavfilter/avfiltergraph.h \
    ../include/fvd_extra_include/libavfilter/buffersink.h \
    ../include/fvd_extra_include/libavfilter/buffersrc.h \
    ../include/fvd_extra_include/libavfilter/version.h \
    ../include/fvd_extra_include/libavformat/avformat.h \
    ../include/fvd_extra_include/libavformat/avio.h \
    ../include/fvd_extra_include/libavformat/version.h \
    ../include/fvd_extra_include/libavutil/adler32.h \
    ../include/fvd_extra_include/libavutil/aes.h \
    ../include/fvd_extra_include/libavutil/attributes.h \
    ../include/fvd_extra_include/libavutil/audio_fifo.h \
    ../include/fvd_extra_include/libavutil/audioconvert.h \
    ../include/fvd_extra_include/libavutil/avassert.h \
    ../include/fvd_extra_include/libavutil/avconfig.h \
    ../include/fvd_extra_include/libavutil/avstring.h \
    ../include/fvd_extra_include/libavutil/avutil.h \
    ../include/fvd_extra_include/libavutil/base64.h \
    ../include/fvd_extra_include/libavutil/blowfish.h \
    ../include/fvd_extra_include/libavutil/bprint.h \
    ../include/fvd_extra_include/libavutil/bswap.h \
    ../include/fvd_extra_include/libavutil/buffer.h \
    ../include/fvd_extra_include/libavutil/cast5.h \
    ../include/fvd_extra_include/libavutil/channel_layout.h \
    ../include/fvd_extra_include/libavutil/common.h \
    ../include/fvd_extra_include/libavutil/cpu.h \
    ../include/fvd_extra_include/libavutil/crc.h \
    ../include/fvd_extra_include/libavutil/dict.h \
    ../include/fvd_extra_include/libavutil/display.h \
    ../include/fvd_extra_include/libavutil/downmix_info.h \
    ../include/fvd_extra_include/libavutil/error.h \
    ../include/fvd_extra_include/libavutil/eval.h \
    ../include/fvd_extra_include/libavutil/ffversion.h \
    ../include/fvd_extra_include/libavutil/fifo.h \
    ../include/fvd_extra_include/libavutil/file.h \
    ../include/fvd_extra_include/libavutil/frame.h \
    ../include/fvd_extra_include/libavutil/hash.h \
    ../include/fvd_extra_include/libavutil/hmac.h \
    ../include/fvd_extra_include/libavutil/imgutils.h \
    ../include/fvd_extra_include/libavutil/intfloat.h \
    ../include/fvd_extra_include/libavutil/intreadwrite.h \
    ../include/fvd_extra_include/libavutil/lfg.h \
    ../include/fvd_extra_include/libavutil/log.h \
    ../include/fvd_extra_include/libavutil/lzo.h \
    ../include/fvd_extra_include/libavutil/macros.h \
    ../include/fvd_extra_include/libavutil/mathematics.h \
    ../include/fvd_extra_include/libavutil/md5.h \
    ../include/fvd_extra_include/libavutil/mem.h \
    ../include/fvd_extra_include/libavutil/motion_vector.h \
    ../include/fvd_extra_include/libavutil/murmur3.h \
    ../include/fvd_extra_include/libavutil/old_pix_fmts.h \
    ../include/fvd_extra_include/libavutil/opt.h \
    ../include/fvd_extra_include/libavutil/parseutils.h \
    ../include/fvd_extra_include/libavutil/pixdesc.h \
    ../include/fvd_extra_include/libavutil/pixelutils.h \
    ../include/fvd_extra_include/libavutil/pixfmt.h \
    ../include/fvd_extra_include/libavutil/random_seed.h \
    ../include/fvd_extra_include/libavutil/rational.h \
    ../include/fvd_extra_include/libavutil/replaygain.h \
    ../include/fvd_extra_include/libavutil/ripemd.h \
    ../include/fvd_extra_include/libavutil/samplefmt.h \
    ../include/fvd_extra_include/libavutil/sha.h \
    ../include/fvd_extra_include/libavutil/sha512.h \
    ../include/fvd_extra_include/libavutil/stereo3d.h \
    ../include/fvd_extra_include/libavutil/threadmessage.h \
    ../include/fvd_extra_include/libavutil/time.h \
    ../include/fvd_extra_include/libavutil/timecode.h \
    ../include/fvd_extra_include/libavutil/timestamp.h \
    ../include/fvd_extra_include/libavutil/version.h \
    ../include/fvd_extra_include/libavutil/xtea.h \
    ../include/fvd_extra_include/libswresample/swresample.h \
    ../include/fvd_extra_include/libswresample/version.h \
    ../include/fvd_extra_include/libswscale/swscale.h \
    ../include/fvd_extra_include/libswscale/version.h \
    ../include/fvd_extra_include/opencv2/core/core.hpp \
    ../include/fvd_extra_include/opencv2/core/core_c.h \
    ../include/fvd_extra_include/opencv2/core/cuda_devptrs.hpp \
    ../include/fvd_extra_include/opencv2/core/devmem2d.hpp \
    ../include/fvd_extra_include/opencv2/core/eigen.hpp \
    ../include/fvd_extra_include/opencv2/core/gpumat.hpp \
    ../include/fvd_extra_include/opencv2/core/internal.hpp \
    ../include/fvd_extra_include/opencv2/core/mat.hpp \
    ../include/fvd_extra_include/opencv2/core/opengl_interop.hpp \
    ../include/fvd_extra_include/opencv2/core/opengl_interop_deprecated.hpp \
    ../include/fvd_extra_include/opencv2/core/operations.hpp \
    ../include/fvd_extra_include/opencv2/core/types_c.h \
    ../include/fvd_extra_include/opencv2/core/version.hpp \
    ../include/fvd_extra_include/opencv2/core/wimage.hpp \
    ../include/fvd_extra_include/va/va.h \
    ../include/fvd_extra_include/va/va_backend.h \
    ../include/fvd_extra_include/va/va_backend_egl.h \
    ../include/fvd_extra_include/va/va_backend_glx.h \
    ../include/fvd_extra_include/va/va_backend_tpi.h \
    ../include/fvd_extra_include/va/va_backend_vpp.h \
    ../include/fvd_extra_include/va/va_backend_wayland.h \
    ../include/fvd_extra_include/va/va_compat.h \
    ../include/fvd_extra_include/va/va_dec_hevc.h \
    ../include/fvd_extra_include/va/va_dec_jpeg.h \
    ../include/fvd_extra_include/va/va_dec_vp8.h \
    ../include/fvd_extra_include/va/va_dec_vp9.h \
    ../include/fvd_extra_include/va/va_dri2.h \
    ../include/fvd_extra_include/va/va_dricommon.h \
    ../include/fvd_extra_include/va/va_drm.h \
    ../include/fvd_extra_include/va/va_drmcommon.h \
    ../include/fvd_extra_include/va/va_egl.h \
    ../include/fvd_extra_include/va/va_enc_h264.h \
    ../include/fvd_extra_include/va/va_enc_hevc.h \
    ../include/fvd_extra_include/va/va_enc_jpeg.h \
    ../include/fvd_extra_include/va/va_enc_mpeg2.h \
    ../include/fvd_extra_include/va/va_enc_vp8.h \
    ../include/fvd_extra_include/va/va_glx.h \
    ../include/fvd_extra_include/va/va_tpi.h \
    ../include/fvd_extra_include/va/va_version.h \
    ../include/fvd_extra_include/va/va_vpp.h \
    ../include/fvd_extra_include/va/va_wayland.h \
    ../include/fvd_extra_include/va/va_x11.h \
    ../server/zenarp.h \
    ../server/zenlog.h \
    ../server/zenplat.h \
    ../cam_alg.h \
    ../cam_codec.h \
    ../cam_net.h \
    ../camera_service.h \
    ../cascadedetect.h \
    ../client_file.h \
    ../client_net.h \
    ../client_obj.h \
    ../common.h \
    ../csocket.h \
    ../file_op.h \
    ../g_define.h \
    ../glplayer.h \
    ../h264_stream_file.h \
    ../hw_vaapi.h \
    ../IPCNetSDK.h \
    ../Net.h \
    ../Net_param.h \
    ../sig_service.h \
    ../tcp_server.h \
    ../vaapi.h
