#!/bin/bash
SUBSET="test"
PRED_HOI_DETS_HDF5="${PWD}/data_symlinks/hico_exp/dets_only/pred_hoi_dets/pred_hoi_dets_${SUBSET}.hdf5"
OUT_DIR="${PWD}/data_symlinks/hico_exp/dets_only/pred_hoi_dets/vis_topk_dets/$SUBSET"

python -m exp.dets_only.vis_top_pred_dets \
    --pred_hoi_dets_hdf5 $PRED_HOI_DETS_HDF5 \
    --out_dir $OUT_DIR \
    --subset $SUBSET
