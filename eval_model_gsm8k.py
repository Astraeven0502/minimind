import argparse
import random
import numpy as np
import torch
import warnings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import *
import re

warnings.filterwarnings('ignore')


def init_model(args):
    """Initialize the MiniMind model and tokenizer."""
    tokenizer = AutoTokenizer.from_pretrained('./model/minimind_tokenizer')

    if args.load == 0:
        moe_path = '_moe' if args.use_moe else ''
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason'}
        ckp = f'./{args.out_dir}/{modes[args.model_mode]}_{args.dim}{moe_path}.pth'

        model = MiniMindLM(LMConfig(
            dim=args.dim,
            n_layers=args.n_layers,
            max_seq_len=args.max_seq_len,
            use_moe=args.use_moe
        ))

        state_dict = torch.load(ckp, map_location=args.device)
        model.load_state_dict({k: v for k, v in state_dict.items() if 'mask' not in k}, strict=True)

        if args.lora_name != 'None':
            apply_lora(model)
            load_lora(model, f'./{args.out_dir}/lora/{args.lora_name}_{args.dim}.pth')
    else:
        transformers_model_path = './MiniMind2'
        tokenizer = AutoTokenizer.from_pretrained(transformers_model_path)
        model = AutoModelForCausalLM.from_pretrained(transformers_model_path, trust_remote_code=True)

    print(f'MiniMind Model Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M')
    return model.eval().to(args.device), tokenizer


def load_gsm8k():
    """Load GSM8K dataset from Hugging Face."""
    dataset = load_dataset("gsm8k", "main")  
    return dataset["test"]  # Use the test set for evaluation


def extract_answer(prediction):
    if not isinstance(prediction, str):
        print(f"Warning: extract_answer received non-string input: {type(prediction)}")
        return None
    numbers = re.findall(r"-?\d+\.?\d*", prediction)
    return float(numbers[-1]) if numbers else None  # Extract the last number if found



def compute_numeric_accuracy(prediction, ground_truth):
    """Compare the extracted numeric answers."""
    if prediction is None:
        print("Warning: Model generated None output!")
        answer = ""  # Assign an empty string to avoid errors
    pred_answer = extract_answer(prediction)
    true_answer = extract_answer(ground_truth)

    if pred_answer is None or true_answer is None:
        return 0.0

    return float(pred_answer) == float(true_answer)


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind on GSM8K Benchmark")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--temperature', default=0.7, type=float)
    parser.add_argument('--top_p', default=0.9, type=float)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--model_mode', default=1, type=int)
    parser.add_argument('--load', default=1, type=int)

    args = parser.parse_args()

    model, tokenizer = init_model(args)

    # Load GSM8K dataset
    test_data = load_gsm8k()

    correct_count = 0
    total_count = 0
    predictions = []

    print("\n🔍 Evaluating Model on GSM8K Benchmark...")

    for example in test_data:
        question = example["question"]
        ground_truth = example["answer"]

        # ✅ Format the question properly for reasoning-based response
        messages = [{"role": "user", "content": f"Solve the following math problem step by step:\n\n{question}"}]
        formatted_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )[-args.max_seq_len + 1:]

        # Generate model output
        with torch.no_grad():
            x = torch.tensor(tokenizer(formatted_input)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=150,
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id
            )

        # ✅ Extract numerical answer
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = str(generated_text.strip().split("\n")[0])  # Ensure it's a string

        # Compute evaluation metrics
        is_correct = compute_numeric_accuracy(answer, ground_truth)
        correct_count += is_correct
        total_count += 1

        predictions.append({
            "question": question,
            "ground_truth": ground_truth,
            "prediction": generated_text.strip(),
            "extracted_answer": answer,
            "correct": is_correct
        })

    # Compute final accuracy
    accuracy = (correct_count / total_count) * 100

    print(f"\n✅ Evaluation Complete!")
    print(f"📊 Accuracy: {accuracy:.2f}%")

    # Print sample outputs
    print("\n🔹 Sample Predictions:")
    for i, sample in enumerate(predictions[:5]):  # Show 5 samples
        print(f"\n📝 Question: {sample['question']}")
        print(f"✅ Ground Truth: {sample['ground_truth']}")
        print(f"🤖 Model Prediction: {sample['prediction']}")
        print(f"🔢 Extracted Answer: {sample['extracted_answer']} (Correct: {sample['correct']})")


if __name__ == "__main__":
    main()
