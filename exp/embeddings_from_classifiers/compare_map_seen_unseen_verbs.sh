#!/bin/bash
EXP_SUBNAME=$1
HICO_EXP_DIR="${PWD}/data_symlinks/hico_exp"
EXP_NAME="embeddings_from_classifier/aes_and_concept_space/${EXP_SUBNAME}"
MAP_JSON="${HICO_EXP_DIR}/${EXP_NAME}/mAP_eval_pred_feats/test_35000/mAP.json"
python -m exp.embeddings_from_classifiers.compare_map_seen_unseen_verbs \
    --map_json $MAP_JSON