import json
from vqaeval import VQAEvaluator
import re

# Initialize the evaluator
evaluator = VQAEvaluator()

# TODO: add other VQA data files here
response_files = [
    "playground/data/textvqa_predictions_llava-7b_imagenet-and-llava-trained.jsonl",
    "playground/data/textvqa_predictions_llava-7b_base.jsonl",
    "playground/data/docvqa_predictions_llava-7b_imagenet-and-llava-trained.jsonl",
    "playground/data/docvqa_predictions_llava-7b_base.jsonl",
]
out_files = [
    "results/eval_textvqa_combined.txt",
    "results/eval_textvqa_base.txt",
    "results/eval_docvqa_combined.txt",
    "results/eval_docvqa_base.txt",
]

for response, out in zip(response_files, out_files):
    # Load the .jsonl file
    model_responses = []
    with open(response, "r") as file:
        for line in file:
            data = json.loads(line)
            model_responses.append(data)

    # Extract model answers
    prompts = [response["prompt"] for response in model_responses]
    model_answers = [response["text"] for response in model_responses]
    human_answers = [response["question_id"] for response in model_responses]

    # Evaluate each model answer against human answers
    accuracies = []
    correct = []
    incorrect = []
    for prompt, model_answer, human_answer in zip(prompts, model_answers, human_answers):
        accuracy = evaluator.evaluate(model_answer, human_answer)
        accuracies.append(accuracy)

        if accuracy > 0:
            correct.append((prompt, model_answer, human_answer))
        else:
            incorrect.append((prompt, model_answer, human_answer))

    # Open output files
    file = open(out, "w")

    # Calculate average accuracy
    average_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.0
    print(f"Average Accuracy: {average_accuracy}\n", file=file)

    # Print correct and incorrect predictions
    print("Correct predictions:", file=file)
    for pred in correct:
        print(f"Prompt:", pred[0], file=file)
        print(f"Model output:", pred[1], file=file)
        print(f"Human annotations:", pred[2], file=file)
        print("", file=file)

    print("\n-------------------------------------------", file=file)
    print("Incorrect predictions:", file=file)
    for pred in incorrect:
        print(f"Prompt:", pred[0], file=file)
        print(f"Model output:", pred[1], file=file)
        print(f"Human annotations:", pred[2], file=file)
        print("", file=file)