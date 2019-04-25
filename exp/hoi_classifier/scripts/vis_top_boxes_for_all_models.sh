# # Det
# echo "Det"
# python -m exp.hoi_classifier.run --exp exp_top_boxes_per_hoi_wo_inference \
#     --rcnn_det_prob \
#     --model_num -1

# # Det + Box
# echo "Det + Box"
# python -m exp.hoi_classifier.run --exp exp_top_boxes_per_hoi_wo_inference \
#     --rcnn_det_prob \
#     --verb_given_boxes_and_object_label \
#     --model_num 25000

# # Det + App
# echo "Det + App"
# python -m exp.hoi_classifier.run --exp exp_top_boxes_per_hoi_wo_inference \
#     --rcnn_det_prob \
#     --verb_given_appearance \
#     --model_num 30000

# # Det + Pose
# echo "Det + Pose"
# python -m exp.hoi_classifier.run --exp exp_top_boxes_per_hoi_wo_inference \
#     --rcnn_det_prob \
#     --verb_given_human_pose \
#     --model_num 170000

# # Det + App + Box
# echo "Det + App + Box"
# python -m exp.hoi_classifier.run --exp exp_top_boxes_per_hoi_wo_inference \
#     --rcnn_det_prob \
#     --verb_given_appearance \
#     --verb_given_boxes_and_object_label \
#     --model_num 25000

# # Det + Box + Pose
# echo "Det + Box + Pose"
# python -m exp.hoi_classifier.run --exp exp_top_boxes_per_hoi_wo_inference \
#     --rcnn_det_prob \
#     --verb_given_boxes_and_object_label \
#     --verb_given_human_pose \
#     --model_num 150000

# # Det + App + Pose
# echo "Det + App + Pose"
# python -m exp.hoi_classifier.run --exp exp_top_boxes_per_hoi_wo_inference \
#     --rcnn_det_prob \
#     --verb_given_appearance \
#     --verb_given_human_pose \
#     --model_num 25000

# Det + App + Box + Pose
echo "Det + App + Box + Pose"
python -m exp.hoi_classifier.run --exp exp_top_boxes_per_hoi_wo_inference \
    --rcnn_det_prob \
    --verb_given_appearance \
    --verb_given_boxes_and_object_label \
    --verb_given_human_pose \
    --model_num 25000