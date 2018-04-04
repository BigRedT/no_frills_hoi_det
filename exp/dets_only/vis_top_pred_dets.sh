#!/bin/bash
SUBSET="test"
PRED_HOI_DETS_HDF5="/home/tanmay/Data/weakly_supervised_hoi_exp/dets_only/pred_hoi_dets/pred_hoi_dets_${SUBSET}.hdf5"
OUT_DIR="/home/tanmay/Data/weakly_supervised_hoi_exp/dets_only/vis_topk_dets/$SUBSET"

python -m exp.dets_only.vis_top_pred_dets \
    --pred_hoi_dets_hdf5 $PRED_HOI_DETS_HDF5 \
    --out_dir $OUT_DIR \
    --subset $SUBSET
