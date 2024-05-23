python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/prompt/flowers_llava7b_102classes_fixedorder.jsonl --including_label True --n_labels 102 --batch_size 8 --fixed_order True
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/flowers.jsonl --class_path ../data/flowers_classes.json --split test --output_path outputs/prompt/flowers_blip2_102classes_fixedorder.jsonl --including_label True --n_labels 102 --batch_size 8 --fixed_order True

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/prompt/caltech_llava7b_100classes_fixedorder.jsonl --including_label True --n_labels 100 --batch_size 8 --fixed_order True
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/caltech.jsonl --class_path ../data/caltech_classes.json --split test --output_path outputs/prompt/caltech_blip2_100classes_fixedorder.jsonl --including_label True --n_labels 100 --batch_size 8 --fixed_order True

python main.py --method vlm --model_id llava-hf/llava-1.5-7b-hf --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/prompt/cars_llava7b_196classes_fixedorder.jsonl --including_label True --n_labels 196 --batch_size 4 --fixed_order True
python main.py --method vlm --model_id Salesforce/blip2-opt-2.7b --data_path ../data/cars.jsonl --class_path ../data/cars_classes.json --split test --output_path outputs/prompt/cars_blip2_196classes_fixedorder.jsonl --including_label True --n_labels 196 --batch_size 4 --fixed_order True
