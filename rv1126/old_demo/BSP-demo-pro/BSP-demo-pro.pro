DISTFILES += \
    ../BSP-demo/example/librtsp/librtsp.a \
    ../BSP-demo/example/run_gcc.sh \
    ../BSP-demo/example/CMakeLists.txt.baK \
    ../BSP-demo/example/CMakeLists.txt \
    ../BSP-demo/example/librtsp/librtsp.a \
    ../BSP-demo/example/CMakeLists.txt

HEADERS += \
    ../BSP-demo/example/common/sample_common.h \
    ../BSP-demo/example/common/sample_double_cam_isp.h \
    ../BSP-demo/example/librtsp/rtsp_demo.h \
    ../BSP-demo/include/rga/drmrga.h \
    ../BSP-demo/include/rga/GrallocOps.h \
    ../BSP-demo/include/rga/im2d.h \
    ../BSP-demo/include/rga/im2d.hpp \
    ../BSP-demo/include/rga/rga.h \
    ../BSP-demo/include/rga/RgaApi.h \
    ../BSP-demo/include/rga/RgaMutex.h \
    ../BSP-demo/include/rga/RgaSingleton.h \
    ../BSP-demo/include/rga/RgaUtils.h \
    ../BSP-demo/include/rga/RockchipRga.h \
    ../BSP-demo/include/rkaiq/algos/a3dlut/rk_aiq_types_a3dlut_algo.h \
    ../BSP-demo/include/rkaiq/algos/a3dlut/rk_aiq_types_a3dlut_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/a3dlut/rk_aiq_types_a3dlut_hw.h \
    ../BSP-demo/include/rkaiq/algos/a3dlut/rk_aiq_uapi_a3dlut_int.h \
    ../BSP-demo/include/rkaiq/algos/ablc/rk_aiq_types_ablc_algo.h \
    ../BSP-demo/include/rkaiq/algos/ablc/rk_aiq_types_ablc_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/ablc/rk_aiq_types_ablc_hw.h \
    ../BSP-demo/include/rkaiq/algos/ablc/rk_aiq_uapi_ablc_int.h \
    ../BSP-demo/include/rkaiq/algos/accm/rk_aiq_types_accm_algo.h \
    ../BSP-demo/include/rkaiq/algos/accm/rk_aiq_types_accm_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/accm/rk_aiq_types_accm_hw.h \
    ../BSP-demo/include/rkaiq/algos/accm/rk_aiq_uapi_accm_int.h \
    ../BSP-demo/include/rkaiq/algos/acp/rk_aiq_types_acp_algo.h \
    ../BSP-demo/include/rkaiq/algos/acp/rk_aiq_types_acp_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/acp/rk_aiq_uapi_acp_int.h \
    ../BSP-demo/include/rkaiq/algos/adebayer/rk_aiq_types_algo_adebayer.h \
    ../BSP-demo/include/rkaiq/algos/adebayer/rk_aiq_types_algo_adebayer_int.h \
    ../BSP-demo/include/rkaiq/algos/adebayer/rk_aiq_uapi_adebayer_int.h \
    ../BSP-demo/include/rkaiq/algos/adehaze/rk_aiq_types_adehaze_algo.h \
    ../BSP-demo/include/rkaiq/algos/adehaze/rk_aiq_types_adehaze_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/adehaze/rk_aiq_types_adehaze_hw.h \
    ../BSP-demo/include/rkaiq/algos/adehaze/rk_aiq_uapi_adehaze_int.h \
    ../BSP-demo/include/rkaiq/algos/adpcc/rk_aiq_types_adpcc_algo.h \
    ../BSP-demo/include/rkaiq/algos/adpcc/rk_aiq_types_adpcc_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/adpcc/rk_aiq_types_adpcc_hw.h \
    ../BSP-demo/include/rkaiq/algos/adpcc/rk_aiq_uapi_adpcc_int.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_types_ae_algo.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_types_ae_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_types_ae_hw.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_uapi_ae_int.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_uapi_ae_int_types.h \
    ../BSP-demo/include/rkaiq/algos/af/rk_aiq_af_hw_v200.h \
    ../BSP-demo/include/rkaiq/algos/af/rk_aiq_types_af_algo.h \
    ../BSP-demo/include/rkaiq/algos/af/rk_aiq_types_af_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/af/rk_aiq_uapi_af_int.h \
    ../BSP-demo/include/rkaiq/algos/afec/fec_algo.h \
    ../BSP-demo/include/rkaiq/algos/afec/rk_aiq_types_afec_algo.h \
    ../BSP-demo/include/rkaiq/algos/afec/rk_aiq_types_afec_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/afec/rk_aiq_uapi_afec_int.h \
    ../BSP-demo/include/rkaiq/algos/agamma/rk_aiq_types_agamma_algo.h \
    ../BSP-demo/include/rkaiq/algos/agamma/rk_aiq_types_agamma_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/agamma/rk_aiq_uapi_agamma_int.h \
    ../BSP-demo/include/rkaiq/algos/agic/rk_aiq_types_algo_agic.h \
    ../BSP-demo/include/rkaiq/algos/agic/rk_aiq_types_algo_agic_int.h \
    ../BSP-demo/include/rkaiq/algos/agic/rk_aiq_uapi_agic_int.h \
    ../BSP-demo/include/rkaiq/algos/ahdr/rk_aiq_types_ahdr_algo.h \
    ../BSP-demo/include/rkaiq/algos/ahdr/rk_aiq_types_ahdr_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/ahdr/rk_aiq_types_ahdr_stat_v200.h \
    ../BSP-demo/include/rkaiq/algos/ahdr/rk_aiq_uapi_ahdr_int.h \
    ../BSP-demo/include/rkaiq/algos/aie/rk_aiq_types_aie_algo.h \
    ../BSP-demo/include/rkaiq/algos/aie/rk_aiq_types_aie_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/aldch/rk_aiq_types_aldch_algo.h \
    ../BSP-demo/include/rkaiq/algos/aldch/rk_aiq_types_aldch_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/aldch/rk_aiq_uapi_aldch_int.h \
    ../BSP-demo/include/rkaiq/algos/alsc/rk_aiq_types_alsc_algo.h \
    ../BSP-demo/include/rkaiq/algos/alsc/rk_aiq_types_alsc_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/alsc/rk_aiq_types_alsc_hw.h \
    ../BSP-demo/include/rkaiq/algos/alsc/rk_aiq_uapi_alsc_int.h \
    ../BSP-demo/include/rkaiq/algos/anr/rk_aiq_types_anr_algo.h \
    ../BSP-demo/include/rkaiq/algos/anr/rk_aiq_types_anr_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/anr/rk_aiq_types_anr_hw.h \
    ../BSP-demo/include/rkaiq/algos/anr/rk_aiq_uapi_anr_int.h \
    ../BSP-demo/include/rkaiq/algos/aorb/rk_aiq_orb_hw.h \
    ../BSP-demo/include/rkaiq/algos/aorb/rk_aiq_types_orb_algo.h \
    ../BSP-demo/include/rkaiq/algos/asd/rk_aiq_types_asd_algo.h \
    ../BSP-demo/include/rkaiq/algos/asd/rk_aiq_uapi_asd_int.h \
    ../BSP-demo/include/rkaiq/algos/asharp/rk_aiq_types_asharp_algo.h \
    ../BSP-demo/include/rkaiq/algos/asharp/rk_aiq_types_asharp_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/asharp/rk_aiq_types_asharp_hw.h \
    ../BSP-demo/include/rkaiq/algos/asharp/rk_aiq_uapi_asharp_int.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_algo.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_stat_v200.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_stat_v201.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_stat_v2xx.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_uapi_awb_int.h \
    ../BSP-demo/include/rkaiq/algos/rk_aiq_algo_des.h \
    ../BSP-demo/include/rkaiq/common/gen_mesh/genMesh.h \
    ../BSP-demo/include/rkaiq/common/gen_mesh/RkGenMeshVersion.h \
    ../BSP-demo/include/rkaiq/common/linux/compiler.h \
    ../BSP-demo/include/rkaiq/common/linux/rk-camera-module.h \
    ../BSP-demo/include/rkaiq/common/linux/rk-led-flash.h \
    ../BSP-demo/include/rkaiq/common/linux/v4l2-controls.h \
    ../BSP-demo/include/rkaiq/common/linux/videodev2.h \
    ../BSP-demo/include/rkaiq/common/mediactl/mediactl-priv.h \
    ../BSP-demo/include/rkaiq/common/mediactl/mediactl.h \
    ../BSP-demo/include/rkaiq/common/mediactl/tools.h \
    ../BSP-demo/include/rkaiq/common/mediactl/v4l2subdev.h \
    ../BSP-demo/include/rkaiq/common/opencv2/calib3d/calib3d_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/hal/interface.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/hal/msa_macros.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/core_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/cv_cpu_dispatch.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/cv_cpu_helper.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/cvdef.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/types_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/features2d/hal/interface.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/all_indices.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/allocator.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/any.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/autotuned_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/composite_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/config.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/defines.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/dist.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/dummy.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/dynamic_bitset.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/general.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/ground_truth.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/hdf5.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/heap.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/hierarchical_clustering_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/index_testing.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/kdtree_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/kdtree_single_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/kmeans_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/linear_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/logger.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/lsh_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/lsh_table.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/matrix.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/nn_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/object_factory.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/params.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/random.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/result_set.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/sampling.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/saving.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/simplex_downhill.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/timer.h \
    ../BSP-demo/include/rkaiq/common/opencv2/highgui/highgui_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgcodecs/legacy/constants_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgcodecs/imgcodecs_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgcodecs/ios.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgproc/hal/interface.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgproc/imgproc_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgproc/types_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/photo/legacy/constants_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/video/legacy/constants_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/videoio/legacy/constants_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/videoio/cap_ios.h \
    ../BSP-demo/include/rkaiq/common/opencv2/videoio/videoio_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/cvconfig.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq_comm.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq_pool.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq_types.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq_types_priv.h \
    ../BSP-demo/include/rkaiq/common/shared_item_pool.h \
    ../BSP-demo/include/rkaiq/iq_parser/RkAiqCalibDbTypes.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_a3dlut.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_ablc.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_accm.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_acp.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_adebayer.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_adehaze.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_adpcc.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_ae.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_af.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_afec.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_agamma.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_agic.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_ahdr.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_aldch.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_alsc.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_anr.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_asd.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_asharp.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_awb.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_debug.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_imgproc.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_sysctl.h \
    ../BSP-demo/include/rkaiq/xcore/base/xcam_common.h \
    ../BSP-demo/include/rkaiq/xcore/base/xcam_defs.h \
    ../BSP-demo/include/rkaiq/rkisp_api.h \
    ../BSP-demo/include/rkmedia/rkmedia_adec.h \
    ../BSP-demo/include/rkmedia/rkmedia_aenc.h \
    ../BSP-demo/include/rkmedia/rkmedia_ai.h \
    ../BSP-demo/include/rkmedia/rkmedia_ao.h \
    ../BSP-demo/include/rkmedia/rkmedia_api.h \
    ../BSP-demo/include/rkmedia/rkmedia_buffer.h \
    ../BSP-demo/include/rkmedia/rkmedia_common.h \
    ../BSP-demo/include/rkmedia/rkmedia_event.h \
    ../BSP-demo/include/rkmedia/rkmedia_move_detection.h \
    ../BSP-demo/include/rkmedia/rkmedia_occlusion_detection.h \
    ../BSP-demo/include/rkmedia/rkmedia_rga.h \
    ../BSP-demo/include/rkmedia/rkmedia_venc.h \
    ../BSP-demo/include/rkmedia/rkmedia_vi.h \
    ../BSP-demo/include/rkmedia/rkmedia_vo.h \
    ../BSP-demo/example/librknn.hpp \
    ../BSP-demo/example/common/sample_common.h \
    ../BSP-demo/example/common/sample_double_cam_isp.h \
    ../BSP-demo/example/librtsp/rtsp_demo.h \
    ../BSP-demo/example/librknn.hpp \
    ../BSP-demo/include/rga/drmrga.h \
    ../BSP-demo/include/rga/GrallocOps.h \
    ../BSP-demo/include/rga/im2d.h \
    ../BSP-demo/include/rga/im2d.hpp \
    ../BSP-demo/include/rga/rga.h \
    ../BSP-demo/include/rga/RgaApi.h \
    ../BSP-demo/include/rga/RgaMutex.h \
    ../BSP-demo/include/rga/RgaSingleton.h \
    ../BSP-demo/include/rga/RgaUtils.h \
    ../BSP-demo/include/rga/RockchipRga.h \
    ../BSP-demo/include/rkaiq/algos/a3dlut/rk_aiq_types_a3dlut_algo.h \
    ../BSP-demo/include/rkaiq/algos/a3dlut/rk_aiq_types_a3dlut_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/a3dlut/rk_aiq_types_a3dlut_hw.h \
    ../BSP-demo/include/rkaiq/algos/a3dlut/rk_aiq_uapi_a3dlut_int.h \
    ../BSP-demo/include/rkaiq/algos/ablc/rk_aiq_types_ablc_algo.h \
    ../BSP-demo/include/rkaiq/algos/ablc/rk_aiq_types_ablc_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/ablc/rk_aiq_types_ablc_hw.h \
    ../BSP-demo/include/rkaiq/algos/ablc/rk_aiq_uapi_ablc_int.h \
    ../BSP-demo/include/rkaiq/algos/accm/rk_aiq_types_accm_algo.h \
    ../BSP-demo/include/rkaiq/algos/accm/rk_aiq_types_accm_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/accm/rk_aiq_types_accm_hw.h \
    ../BSP-demo/include/rkaiq/algos/accm/rk_aiq_uapi_accm_int.h \
    ../BSP-demo/include/rkaiq/algos/acp/rk_aiq_types_acp_algo.h \
    ../BSP-demo/include/rkaiq/algos/acp/rk_aiq_types_acp_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/acp/rk_aiq_uapi_acp_int.h \
    ../BSP-demo/include/rkaiq/algos/adebayer/rk_aiq_types_algo_adebayer.h \
    ../BSP-demo/include/rkaiq/algos/adebayer/rk_aiq_types_algo_adebayer_int.h \
    ../BSP-demo/include/rkaiq/algos/adebayer/rk_aiq_uapi_adebayer_int.h \
    ../BSP-demo/include/rkaiq/algos/adehaze/rk_aiq_types_adehaze_algo.h \
    ../BSP-demo/include/rkaiq/algos/adehaze/rk_aiq_types_adehaze_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/adehaze/rk_aiq_types_adehaze_hw.h \
    ../BSP-demo/include/rkaiq/algos/adehaze/rk_aiq_uapi_adehaze_int.h \
    ../BSP-demo/include/rkaiq/algos/adpcc/rk_aiq_types_adpcc_algo.h \
    ../BSP-demo/include/rkaiq/algos/adpcc/rk_aiq_types_adpcc_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/adpcc/rk_aiq_types_adpcc_hw.h \
    ../BSP-demo/include/rkaiq/algos/adpcc/rk_aiq_uapi_adpcc_int.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_types_ae_algo.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_types_ae_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_types_ae_hw.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_uapi_ae_int.h \
    ../BSP-demo/include/rkaiq/algos/ae/rk_aiq_uapi_ae_int_types.h \
    ../BSP-demo/include/rkaiq/algos/af/rk_aiq_af_hw_v200.h \
    ../BSP-demo/include/rkaiq/algos/af/rk_aiq_types_af_algo.h \
    ../BSP-demo/include/rkaiq/algos/af/rk_aiq_types_af_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/af/rk_aiq_uapi_af_int.h \
    ../BSP-demo/include/rkaiq/algos/afec/fec_algo.h \
    ../BSP-demo/include/rkaiq/algos/afec/rk_aiq_types_afec_algo.h \
    ../BSP-demo/include/rkaiq/algos/afec/rk_aiq_types_afec_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/afec/rk_aiq_uapi_afec_int.h \
    ../BSP-demo/include/rkaiq/algos/agamma/rk_aiq_types_agamma_algo.h \
    ../BSP-demo/include/rkaiq/algos/agamma/rk_aiq_types_agamma_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/agamma/rk_aiq_uapi_agamma_int.h \
    ../BSP-demo/include/rkaiq/algos/agic/rk_aiq_types_algo_agic.h \
    ../BSP-demo/include/rkaiq/algos/agic/rk_aiq_types_algo_agic_int.h \
    ../BSP-demo/include/rkaiq/algos/agic/rk_aiq_uapi_agic_int.h \
    ../BSP-demo/include/rkaiq/algos/ahdr/rk_aiq_types_ahdr_algo.h \
    ../BSP-demo/include/rkaiq/algos/ahdr/rk_aiq_types_ahdr_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/ahdr/rk_aiq_types_ahdr_stat_v200.h \
    ../BSP-demo/include/rkaiq/algos/ahdr/rk_aiq_uapi_ahdr_int.h \
    ../BSP-demo/include/rkaiq/algos/aie/rk_aiq_types_aie_algo.h \
    ../BSP-demo/include/rkaiq/algos/aie/rk_aiq_types_aie_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/aldch/rk_aiq_types_aldch_algo.h \
    ../BSP-demo/include/rkaiq/algos/aldch/rk_aiq_types_aldch_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/aldch/rk_aiq_uapi_aldch_int.h \
    ../BSP-demo/include/rkaiq/algos/alsc/rk_aiq_types_alsc_algo.h \
    ../BSP-demo/include/rkaiq/algos/alsc/rk_aiq_types_alsc_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/alsc/rk_aiq_types_alsc_hw.h \
    ../BSP-demo/include/rkaiq/algos/alsc/rk_aiq_uapi_alsc_int.h \
    ../BSP-demo/include/rkaiq/algos/anr/rk_aiq_types_anr_algo.h \
    ../BSP-demo/include/rkaiq/algos/anr/rk_aiq_types_anr_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/anr/rk_aiq_types_anr_hw.h \
    ../BSP-demo/include/rkaiq/algos/anr/rk_aiq_uapi_anr_int.h \
    ../BSP-demo/include/rkaiq/algos/aorb/rk_aiq_orb_hw.h \
    ../BSP-demo/include/rkaiq/algos/aorb/rk_aiq_types_orb_algo.h \
    ../BSP-demo/include/rkaiq/algos/asd/rk_aiq_types_asd_algo.h \
    ../BSP-demo/include/rkaiq/algos/asd/rk_aiq_uapi_asd_int.h \
    ../BSP-demo/include/rkaiq/algos/asharp/rk_aiq_types_asharp_algo.h \
    ../BSP-demo/include/rkaiq/algos/asharp/rk_aiq_types_asharp_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/asharp/rk_aiq_types_asharp_hw.h \
    ../BSP-demo/include/rkaiq/algos/asharp/rk_aiq_uapi_asharp_int.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_algo.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_algo_int.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_stat_v200.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_stat_v201.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_types_awb_stat_v2xx.h \
    ../BSP-demo/include/rkaiq/algos/awb/rk_aiq_uapi_awb_int.h \
    ../BSP-demo/include/rkaiq/algos/rk_aiq_algo_des.h \
    ../BSP-demo/include/rkaiq/common/gen_mesh/genMesh.h \
    ../BSP-demo/include/rkaiq/common/gen_mesh/RkGenMeshVersion.h \
    ../BSP-demo/include/rkaiq/common/linux/compiler.h \
    ../BSP-demo/include/rkaiq/common/linux/rk-camera-module.h \
    ../BSP-demo/include/rkaiq/common/linux/rk-led-flash.h \
    ../BSP-demo/include/rkaiq/common/linux/v4l2-controls.h \
    ../BSP-demo/include/rkaiq/common/linux/videodev2.h \
    ../BSP-demo/include/rkaiq/common/mediactl/mediactl-priv.h \
    ../BSP-demo/include/rkaiq/common/mediactl/mediactl.h \
    ../BSP-demo/include/rkaiq/common/mediactl/tools.h \
    ../BSP-demo/include/rkaiq/common/mediactl/v4l2subdev.h \
    ../BSP-demo/include/rkaiq/common/opencv2/calib3d/calib3d_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/hal/interface.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/hal/msa_macros.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/core_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/cv_cpu_dispatch.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/cv_cpu_helper.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/cvdef.h \
    ../BSP-demo/include/rkaiq/common/opencv2/core/types_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/features2d/hal/interface.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/all_indices.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/allocator.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/any.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/autotuned_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/composite_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/config.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/defines.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/dist.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/dummy.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/dynamic_bitset.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/general.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/ground_truth.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/hdf5.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/heap.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/hierarchical_clustering_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/index_testing.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/kdtree_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/kdtree_single_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/kmeans_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/linear_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/logger.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/lsh_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/lsh_table.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/matrix.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/nn_index.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/object_factory.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/params.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/random.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/result_set.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/sampling.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/saving.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/simplex_downhill.h \
    ../BSP-demo/include/rkaiq/common/opencv2/flann/timer.h \
    ../BSP-demo/include/rkaiq/common/opencv2/highgui/highgui_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgcodecs/legacy/constants_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgcodecs/imgcodecs_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgcodecs/ios.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgproc/hal/interface.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgproc/imgproc_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/imgproc/types_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/photo/legacy/constants_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/video/legacy/constants_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/videoio/legacy/constants_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/videoio/cap_ios.h \
    ../BSP-demo/include/rkaiq/common/opencv2/videoio/videoio_c.h \
    ../BSP-demo/include/rkaiq/common/opencv2/cvconfig.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq_comm.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq_pool.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq_types.h \
    ../BSP-demo/include/rkaiq/common/rk_aiq_types_priv.h \
    ../BSP-demo/include/rkaiq/common/shared_item_pool.h \
    ../BSP-demo/include/rkaiq/iq_parser/RkAiqCalibDbTypes.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_a3dlut.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_ablc.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_accm.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_acp.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_adebayer.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_adehaze.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_adpcc.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_ae.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_af.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_afec.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_agamma.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_agic.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_ahdr.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_aldch.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_alsc.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_anr.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_asd.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_asharp.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_awb.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_debug.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_imgproc.h \
    ../BSP-demo/include/rkaiq/uAPI/rk_aiq_user_api_sysctl.h \
    ../BSP-demo/include/rkaiq/xcore/base/xcam_common.h \
    ../BSP-demo/include/rkaiq/xcore/base/xcam_defs.h \
    ../BSP-demo/include/rkaiq/rkisp_api.h \
    ../BSP-demo/include/rkmedia/rkmedia_adec.h \
    ../BSP-demo/include/rkmedia/rkmedia_aenc.h \
    ../BSP-demo/include/rkmedia/rkmedia_ai.h \
    ../BSP-demo/include/rkmedia/rkmedia_ao.h \
    ../BSP-demo/include/rkmedia/rkmedia_api.h \
    ../BSP-demo/include/rkmedia/rkmedia_buffer.h \
    ../BSP-demo/include/rkmedia/rkmedia_common.h \
    ../BSP-demo/include/rkmedia/rkmedia_event.h \
    ../BSP-demo/include/rkmedia/rkmedia_move_detection.h \
    ../BSP-demo/include/rkmedia/rkmedia_occlusion_detection.h \
    ../BSP-demo/include/rkmedia/rkmedia_rga.h \
    ../BSP-demo/include/rkmedia/rkmedia_venc.h \
    ../BSP-demo/include/rkmedia/rkmedia_vi.h \
    ../BSP-demo/include/rkmedia/rkmedia_vo.h

SOURCES += \
    ../BSP-demo/example/common/sample_common_isp.c \
    ../BSP-demo/example/common/sample_double_cam_isp.c \
    ../BSP-demo/example/rk_sample.c \
    ../BSP-demo/example/rk_sample.cpp \
    ../BSP-demo/example/common/sample_common_isp.c \
    ../BSP-demo/example/common/sample_double_cam_isp.c \
    ../BSP-demo/example/rk_sample.c \
    ../BSP-demo/example/common/sample_common_isp.cpp \
    ../BSP-demo/example/common/sample_double_cam_isp.cpp
INCLUDEPATH += ../BSP-demo/include \
    ../BSP-demo/include/rkaiq \
    ../BSP-demo/include/rkmedia \
    ./