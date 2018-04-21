#!/bin/bash
SUBSET="test"
PRED_HOI_DETS_HDF5="/home/tanmay/Data/weakly_supervised_hoi_exp/relation_classifier/factors_rcnn_feats_scores_imgs_per_batch_1_focal_loss_False_fp_to_tp_ratio_1000_box_aware_model_True/pred_hoi_dets_${SUBSET}_80000.hdf5"
OUT_DIR="/home/tanmay/Data/weakly_supervised_hoi_exp/relation_classifier/factors_rcnn_feats_scores_imgs_per_batch_1_focal_loss_False_fp_to_tp_ratio_1000_box_aware_model_True/mAP_eval/${SUBSET}_80000"
PROC_DIR="/home/ssd/hico_det_processed_20160224/"

python -m exp.hico_eval.compute_map \
    --pred_hoi_dets_hdf5 $PRED_HOI_DETS_HDF5 \
    --out_dir $OUT_DIR \
    --proc_dir $PROC_DIR \
    --subset $SUBSET
