GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python -m exp.hoi_classifier.run \
    --exp exp_train \
    --imgs_per_batch 1 \
    --fp_to_tp_ratio 1000 \
    --rcnn_det_prob \
    --verb_given_appearance \
    --verb_given_boxes_and_object_label \
    --verb_given_human_pose