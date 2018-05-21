#!/bin/bash
SUBSET="test"
HICO_EXP_DIR="${PWD}/data_symlinks/hico_exp"
EXP_NAME="hoi_classifier/factors_rcnn_det_prob_boxes_and_object_label"
MODEL_NUM="25000"
ORACLE="oracle_human_True_object_True_verb_False"
#PRED_HOI_DETS_HDF5="${HICO_EXP_DIR}/${EXP_NAME}/pred_hoi_dets_${SUBSET}_${MODEL_NUM}.hdf5"
#PRED_HOI_DETS_HDF5="${HICO_EXP_DIR}/${EXP_NAME}/pred_hoi_dets_${ORACLE}_${SUBSET}_${MODEL_NUM}.hdf5"
#OUT_DIR="${HICO_EXP_DIR}/${EXP_NAME}/mAP_eval_greedy_bug_fix/${SUBSET}_${MODEL_NUM}"
#OUT_DIR="${HICO_EXP_DIR}/${EXP_NAME}/mAP_eval_greedy_bug_fix/${ORACLE}_${SUBSET}_${MODEL_NUM}"
PRED_HOI_DETS_HDF5="${HICO_EXP_DIR}/ho_rcnn/ho_1_s/pred_hoi_dets.hdf5"
OUT_DIR="${HICO_EXP_DIR}/ho_rcnn/ho_1_s/mAP_eval_greedy_bug_fix"
PROC_DIR="${PWD}/data_symlinks/hico_processed"

echo $OUT_DIR
python -m exp.hico_eval.compute_map \
    --pred_hoi_dets_hdf5 $PRED_HOI_DETS_HDF5 \
    --out_dir $OUT_DIR \
    --proc_dir $PROC_DIR \
    --subset $SUBSET

python -m exp.hico_eval.sample_complexity_analysis \
    --out_dir $OUT_DIR \
    --ap_type "AP"

python -m exp.hico_eval.sample_complexity_analysis \
    --out_dir $OUT_DIR \
    --ap_type "NAP"