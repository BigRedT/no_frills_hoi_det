#!/bin/bash
HICO_DETS_DIR='/home/tanmay/Data/weakly_supervised_hoi_exp/dets_only/pred_hoi_dets'
OUT_DIR='/home/tanmay/Data/weakly_supervised_hoi_exp/dets_only/mAP_eval'
SUBSET='test'
PROC_DIR='/home/ssd/hico_det_processed_20160224/'
python -m exp.hico_eval.compute_map \
    --hico_dets_dir $HICO_DETS_DIR \
    --out_dir $OUT_DIR \
    --proc_dir $PROC_DIR \
    --subset $SUBSET
