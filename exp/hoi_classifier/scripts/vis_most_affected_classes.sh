echo "All Factors"
EXP_DIR="${PWD}/data_symlinks/hico_exp/hoi_classifier/factors_rcnn_det_prob_appearance_boxes_and_object_label_human_pose"
MODEL_NUM="25000"
AP_JSON="${EXP_DIR}/mAP_eval/test_${MODEL_NUM}/mAP.json"
BASELINE_EXP_DIR="${PWD}/data_symlinks/hico_exp/hoi_classifier/factors_rcnn_det_prob"
BASELINE_MODEL_NUM="-1"
BASELINE_AP_JSON="${BASELINE_EXP_DIR}/mAP_eval/test_${BASELINE_MODEL_NUM}/mAP.json"
OUTDIR="${EXP_DIR}/vis"
echo $AP_JSON
echo $BASELINE_AP_JSON
echo $OUTDIR
python -m exp.hoi_classifier.vis.vis_most_affected_classes \
    --ap_json $AP_JSON \
    --ap_baseline_json $BASELINE_AP_JSON \
    --outdir $OUTDIR

echo "Det + App"
EXP_DIR="${PWD}/data_symlinks/hico_exp/hoi_classifier/factors_rcnn_det_prob_appearance"
MODEL_NUM="30000"
AP_JSON="${EXP_DIR}/mAP_eval/test_${MODEL_NUM}/mAP.json"
BASELINE_EXP_DIR="${PWD}/data_symlinks/hico_exp/hoi_classifier/factors_rcnn_det_prob"
BASELINE_MODEL_NUM="-1"
BASELINE_AP_JSON="${BASELINE_EXP_DIR}/mAP_eval/test_${BASELINE_MODEL_NUM}/mAP.json"
OUTDIR="${EXP_DIR}/vis"
echo $AP_JSON
echo $BASELINE_AP_JSON
echo $OUTDIR
python -m exp.hoi_classifier.vis.vis_most_affected_classes \
    --ap_json $AP_JSON \
    --ap_baseline_json $BASELINE_AP_JSON \
    --outdir $OUTDIR

echo "Det + Box"
EXP_DIR="${PWD}/data_symlinks/hico_exp/hoi_classifier/factors_rcnn_det_prob_boxes_and_object_label"
MODEL_NUM="25000"
AP_JSON="${EXP_DIR}/mAP_eval/test_${MODEL_NUM}/mAP.json"
BASELINE_EXP_DIR="${PWD}/data_symlinks/hico_exp/hoi_classifier/factors_rcnn_det_prob"
BASELINE_MODEL_NUM="-1"
BASELINE_AP_JSON="${BASELINE_EXP_DIR}/mAP_eval/test_${BASELINE_MODEL_NUM}/mAP.json"
OUTDIR="${EXP_DIR}/vis"
echo $AP_JSON
echo $BASELINE_AP_JSON
echo $OUTDIR
python -m exp.hoi_classifier.vis.vis_most_affected_classes \
    --ap_json $AP_JSON \
    --ap_baseline_json $BASELINE_AP_JSON \
    --outdir $OUTDIR

echo "Det + Pose"
EXP_DIR="${PWD}/data_symlinks/hico_exp/hoi_classifier/factors_rcnn_det_prob_human_pose"
MODEL_NUM="170000"
AP_JSON="${EXP_DIR}/mAP_eval/test_${MODEL_NUM}/mAP.json"
BASELINE_EXP_DIR="${PWD}/data_symlinks/hico_exp/hoi_classifier/factors_rcnn_det_prob"
BASELINE_MODEL_NUM="-1"
BASELINE_AP_JSON="${BASELINE_EXP_DIR}/mAP_eval/test_${BASELINE_MODEL_NUM}/mAP.json"
OUTDIR="${EXP_DIR}/vis"
echo $AP_JSON
echo $BASELINE_AP_JSON
echo $OUTDIR
python -m exp.hoi_classifier.vis.vis_most_affected_classes \
    --ap_json $AP_JSON \
    --ap_baseline_json $BASELINE_AP_JSON \
    --outdir $OUTDIR