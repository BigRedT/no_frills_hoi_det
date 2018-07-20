#!/bin/bash
EXP_SUBNAME=$1
SUBSET="test"
HICO_EXP_DIR="${PWD}/data_symlinks/hico_exp"
EXP_NAME="embeddings_from_classifier/aes_and_concept_space/${EXP_SUBNAME}" #make_identity_True_adam_weight_decay_1e-6"
echo $EXP_NAME
MODEL_NUM="35000"
PRED_HOI_DETS_HDF5="${HICO_EXP_DIR}/${EXP_NAME}/pred_feats_pred_hoi_dets_${SUBSET}_${MODEL_NUM}.hdf5"
OUT_DIR="${HICO_EXP_DIR}/${EXP_NAME}/mAP_eval_pred_feats/${SUBSET}_${MODEL_NUM}"
PROC_DIR="${PWD}/data_symlinks/hico_processed"
python -m exp.hico_eval.compute_map \
    --pred_hoi_dets_hdf5 $PRED_HOI_DETS_HDF5 \
    --out_dir $OUT_DIR \
    --proc_dir $PROC_DIR \
    --subset $SUBSET

python -m exp.hico_eval.sample_complexity_analysis \
    --out_dir $OUT_DIR \
    --ap_type "AP"