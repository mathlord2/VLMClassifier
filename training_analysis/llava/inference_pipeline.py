from datasets import load_dataset
import os
import tqdm
from PIL import Image
import torch
import shortuuid
import argparse
import json

from llava.constants import (
    DEFAULT_IM_END_TOKEN,
    DEFAULT_IM_START_TOKEN,
    DEFAULT_IMAGE_TOKEN,
    IMAGE_TOKEN_INDEX,
)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (
    get_model_name_from_path,
    process_images,
    tokenizer_image_token,
)
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

def get_data(db_name, db_type, config_name):
    """
    Get the specified dataset from HuggingFace, and print out the first sample.
    """

    if config_name is None:
        test = load_dataset(db_name, split=db_type, trust_remote_code=True)
    else:
        test = load_dataset(db_name, config_name, split=db_type, trust_remote_code=True)
    
    print(test[0])
    return test

def eval_model(args):
    """
    Evaluate a model on a dataset.
    """
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, args.model_base, model_name
    )

    data = get_data(args.db_name, args.db_type, args.config_name)
    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)
    ans_file = open(answers_file, "w")
    error_file = open(args.errors_file, "w")

    curr = 0
    for line in data:
        print(f"\nProcessing row {curr} out of {len(data)}")
        curr += 1
        print(line)

        try:
            idx = line[args.answer_key]
            image = line[args.image_key]
            qs = line[args.question_key]

            # Convert image to RGB if not already
            if isinstance(image, Image.Image) and image.mode != "RGB":
                image = image.convert("RGB")
            elif not isinstance(image, Image.Image):
                print(f"Image is not a PIL Image: {type(image)}")
                continue
            
            # Adding multiple-choice options in prompt
            if args.multiple_choice:
                if line["options"]:
                    for i in range(len(line["options"])):
                        qs += f"\nOption{i}: {line['options'][i]}"
            
                qs += "\nChoose the best answer out of Option0, Option1, Option2, or Option3, and output your answer in the format 'The answer is: Option[n]'."
            else:
                qs += "\nOutput your answer in the format 'The answer is: [answer]'."

            cur_prompt = qs
            if model.config.mm_use_im_start_end:
                qs = (
                    DEFAULT_IM_START_TOKEN
                    + DEFAULT_IMAGE_TOKEN
                    + DEFAULT_IM_END_TOKEN
                    + "\n"
                    + qs
                )
            else:
                qs = DEFAULT_IMAGE_TOKEN + "\n" + qs

            # Processing
            conv = conv_templates[args.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            conv.append_message(conv.roles[1], "Let's think step by step.")
            prompt = conv.get_prompt().replace("</s>", "")
            print(prompt)

            input_ids = (
                tokenizer_image_token(
                    prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
                )
                .unsqueeze(0)
                .cuda()
            )

            image_tensor = process_images([image], image_processor, model.config)[0]

            with torch.inference_mode():
                output_ids = model.generate(
                    input_ids,
                    images=image_tensor.unsqueeze(0).half().cuda(),
                    image_sizes=[image.size],
                    do_sample=True if args.temperature > 0 else False,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=args.num_beams,
                    # no_repeat_ngram_size=3,
                    max_new_tokens=1024,
                    use_cache=True,
                )

            outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[
                0
            ].strip()

            # Add to response output
            ans_id = shortuuid.uuid()
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "prompt": cur_prompt,
                        "text": outputs,
                        "answer_id": ans_id,
                        "model_id": model_name,
                        "metadata": {},
                    }
                )
                + "\n"
            )
            
            ans_file.flush()
        except Exception as e:
            print(f"Error parsing data {line}: {e}", file=error_file)

    ans_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, default="facebook/opt-350m")
    parser.add_argument("--model-base", type=str, default=None)
    parser.add_argument("--db-name", type=str, default=None)
    parser.add_argument("--db-type", type=str, default="test")
    parser.add_argument("--config-name", type=str, default=None)
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--errors-file", type=str, default="error.txt")
    parser.add_argument("--question-key", type=str, default="question")
    parser.add_argument("--image-key", type=str, default="image")
    parser.add_argument("--answer-key", type=str, default="answer")
    parser.add_argument("--multiple-choice", type=str, default=False)
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    args = parser.parse_args()

    eval_model(args)