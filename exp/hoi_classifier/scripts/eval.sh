GPU=$1
MODEL_NUM=$2
CUDA_VISIBLE_DEVICES=$GPU python -m exp.hoi_classifier.run \
    --exp exp_eval \
    --rcnn_det_prob \
    --verb_given_appearance \
    --verb_given_boxes_and_object_label \
    --verb_given_human_pose \
    --model_num $MODEL_NUM