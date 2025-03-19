## FINE-TUNED MODEL
# python -m inference_pipeline \
#     --model-path /mmfs1/gscratch/raivn/exiao/VLMClassifier/training_analysis/llava/processed-llava-v1.5-7b-imagenet \
#     --db-name lmms-lab/ai2d \
#     --db-type validation \
#     --answers-file ./playground/data/ai2d_predictions_llava-7b_imagenet-and-llava-trained.jsonl \
#     --errors-file ./errors/ai2d_imagenet-and-llava_errors.txt \
#     --question-key question \
#     --image-key image \
#     --answer-key answer \
#     --multiple-choice \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m inference_pipeline \
#     --model-path /mmfs1/gscratch/raivn/exiao/VLMClassifier/training_analysis/llava/processed-llava-v1.5-7b-imagenet \
#     --db-name facebook/textvqa \
#     --db-type validation \
#     --answers-file ./playground/data/textvqa_predictions_llava-7b_imagenet-and-llava-trained.jsonl \
#     --errors-file ./errors/textvqa_imagenet-and-llava_errors.txt \
#     --question-key question \
#     --image-key image \
#     --answer-key answers \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m inference_pipeline \
#     --model-path /mmfs1/gscratch/raivn/exiao/VLMClassifier/training_analysis/llava/processed-llava-v1.5-7b-imagenet \
#     --db-name lmms-lab/DocVQA \
#     --db-type validation \
#     --config-name DocVQA \
#     --answers-file ./playground/data/docvqa_predictions_llava-7b_imagenet-and-llava-trained.jsonl \
#     --errors-file ./errors/docvqa_imagenet-and-llava_errors.txt \
#     --question-key question \
#     --image-key image \
#     --answer-key answers \
#     --temperature 0 \
#     --conv-mode vicuna_v1

## BASE MODEL
# python -m inference_pipeline \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --db-name lmms-lab/ai2d \
#     --db-type test \
#     --answers-file ./playground/data/ai2d_predictions_llava-7b_base.jsonl \
#     --errors-file ./errors/ai2d_base_errors.txt \
#     --question-key question \
#     --image-key image \
#     --answer-key answer \
#     --multiple-choice True \
#     --temperature 0 \
#     --conv-mode vicuna_v1

# python -m inference_pipeline \
#     --model-path liuhaotian/llava-v1.5-7b \
#     --db-name facebook/textvqa \
#     --db-type validation \
#     --answers-file ./playground/data/textvqa_predictions_llava-7b_base.jsonl \
#     --errors-file ./errors/textvqa_base_errors.txt \
#     --question-key question \
#     --image-key image \
#     --answer-key answers \
#     --temperature 0 \
#     --conv-mode vicuna_v1

python -m inference_pipeline \
    --model-path liuhaotian/llava-v1.5-7b \
    --db-name lmms-lab/DocVQA \
    --db-type validation \
    --config-name DocVQA \
    --answers-file ./playground/data/docvqa_predictions_llava-7b_base.jsonl \
    --errors-file ./errors/docvqa_base_errors.txt \
    --question-key question \
    --image-key image \
    --answer-key answers \
    --temperature 0 \
    --conv-mode vicuna_v1