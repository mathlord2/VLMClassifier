# python -m model_vqa \
#     --model-path /mmfs1/gscratch/raivn/exiao/VLMClassifier/training_analysis/llava/processed-llava-v1.5-7b-imagenet \
#     --question-file /mmfs1/gscratch/raivn/exiao/VLMClassifier/data/imagewikiqa.jsonl  \
#     --image-folder /mmfs1/gscratch/raivn/exiao/VLMClassifier \
#     --answers-file ./playground/data/imagewikiqa_predictions_llava-7b_imagenet-and-llava-trained.jsonl \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python -m model_vqa \
    --model-path liuhaotian/llava-v1.5-7b \
    --question-file /mmfs1/gscratch/raivn/exiao/VLMClassifier/data/imagewikiqa.jsonl  \
    --image-folder /mmfs1/gscratch/raivn/exiao/VLMClassifier \
    --answers-file ./playground/data/imagewikiqa_predictions_llava-7b_base.jsonl \
    --temperature 0 \
    --conv-mode vicuna_v1