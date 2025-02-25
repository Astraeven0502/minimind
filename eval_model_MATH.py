import argparse
import torch
import warnings
import re
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from model.model import MiniMindLM
from model.LMConfig import LMConfig
from model.model_lora import *

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


def load_aimo_dataset(split="train"):
    """Load AI-MO/aimo-validation-amc dataset from Hugging Face."""
    dataset = load_dataset("AI-MO/aimo-validation-amc", split=split)
    return dataset


# def extract_mcq_answer(prediction):
#     """Extract the final multiple-choice letter answer from the model output."""
#     if not isinstance(prediction, str):
#         return None
    
#     # Look for a single letter A-E
#     match = re.search(r"\b([A-E])\b", prediction, re.IGNORECASE)
#     return match.group(1).upper() if match else None
def extract_numeric_answer(prediction):
    """Extract numeric answer from the model's output."""
    numbers = re.findall(r"-?\d+\.?\d*", prediction)  # Capture integers & decimals
    return numbers[-1] if numbers else "N/A"  # Get last number or return 'N/A'


# def compute_mcq_accuracy(prediction, ground_truth):
#     """Compute exact match accuracy for multiple-choice answers."""
#     pred_answer = extract_mcq_answer(prediction)
#     true_answer = ground_truth

#     return int(pred_answer == true_answer)
def compute_numeric_accuracy(prediction, ground_truth):
    """Check if the predicted answer matches the actual numerical answer."""
    try:
        pred_answer = float(extract_numeric_answer(prediction))  # Convert to float
        gt_answer = float(ground_truth)  # Convert to float
        return abs(pred_answer - gt_answer) < 1e-6  # Allow small precision errors
    except ValueError:
        return False  # Return False if conversion fails



def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind on AI-MO AMC Validation Dataset")
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
    parser.add_argument('--num_samples', default=50, type=int, help="Number of test samples")

    args = parser.parse_args()

    model, tokenizer = init_model(args)

    # Load AI-MO dataset
    test_data = load_aimo_dataset()

    # Store evaluation results
    accuracy_scores = []
    predictions = []

    print(f"\n🔍 Evaluating Model on AI-MO AMC Dataset (First {args.num_samples} Samples)...")
    num_samples = min(args.num_samples, len(test_data))  # Ensure we don't exceed dataset size

    for i, example in enumerate(test_data.shuffle(seed=42).select(range(num_samples))):
        question = example["problem"]
        # options = example["options"]
        ground_truth = example["answer"]

        # Format the multiple-choice question
        # formatted_question = f"{question}\n"
        # for idx, option in enumerate(options, start=1):
        #     formatted_question += f"{chr(64+idx)}. {option}\n"  # A, B, C, D, E

        # Structure the model prompt
        # messages = [{"role": "user", "content": f"{formatted_question}\nChoose the correct answer (A, B, C, D, or E):"}
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
                max_new_tokens=150,  # Shorter response
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id
            )

        # Decode response
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        # answer = extract_mcq_answer(generated_text)
        answer = extract_numeric_answer(generated_text)
        full_answer = str(generated_text.strip().split("\n")[0])  # Ensure it's a string

        # Compute accuracy
        is_correct = compute_numeric_accuracy(answer, ground_truth)
        accuracy_scores.append(is_correct)

        predictions.append({
            "question": question,
            # "options": options,
            "ground_truth": ground_truth,
            "prediction": full_answer,
            "correct": is_correct
        })

        if i % 10 == 0:
            print(f"[{i}/{args.num_samples}] Processed...")

    # Compute final accuracy
    final_accuracy = np.mean(accuracy_scores) * 100

    print(f"\n✅ Evaluation Complete!")
    print(f"📊 AI-MO AMC Dataset Accuracy: {final_accuracy:.2f}%")

    # Print sample outputs
    print("\n🔹 Sample Predictions:")
    for sample in predictions[:3]:  # Show 5 samples
        print(f"\n📝 Question: {sample['question']}")
        # for idx, option in enumerate(sample["options"], start=1):
        #     print(f"{chr(64+idx)}. {option}")
        print(f"✅ Ground Truth: {sample['ground_truth']}")
        print(f"🤖 Model Prediction: {sample['prediction']}")
        print(f"🎯 Correct: {'✔️' if sample['correct'] else '❌'}")


if __name__ == "__main__":
    main()