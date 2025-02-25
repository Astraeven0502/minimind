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
from collections import Counter

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


def load_clue_dataset(task_name):
    """Load CLUE dataset from Hugging Face datasets."""
    dataset = load_dataset("clue", task_name)
    return dataset["validation"]  # Using validation set


def compute_exact_match(prediction, ground_truth):
    """Compute Exact Match (EM) score."""
    return int(prediction.lower().strip() == ground_truth.lower().strip())


def compute_f1(prediction, ground_truth):
    """Compute F1-score between prediction and ground-truth."""
    pred_tokens = prediction.lower().split()
    gt_tokens = ground_truth.lower().split()

    common = Counter(pred_tokens) & Counter(gt_tokens)
    num_same = sum(common.values())

    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gt_tokens)
    f1 = 2 * (precision * recall) / (precision + recall)
    
    return f1


def main():
    parser = argparse.ArgumentParser(description="Evaluate MiniMind on CLUE Benchmark")
    parser.add_argument('--dataset', default='cmrc2018', type=str, help="CLUE task (e.g., cmrc2018, ocnli, tnews)")
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    parser.add_argument('--temperature', default=0.85, type=float)
    parser.add_argument('--top_p', default=0.85, type=float)
    parser.add_argument('--dim', default=512, type=int)
    parser.add_argument('--n_layers', default=8, type=int)
    parser.add_argument('--max_seq_len', default=512, type=int)
    parser.add_argument('--use_moe', default=False, type=bool)
    parser.add_argument('--model_mode', default=1, type=int, help="0: Pretrain, 1: SFT, 2: RLHF, 3: Reason")
    
    # ✅ Add `load` argument to prevent missing attribute error
    parser.add_argument('--load', default=1, type=int, help="0: Custom weights, 1: Transformers model")

    args = parser.parse_args()

    model, tokenizer = init_model(args)

    # Load CLUE benchmark dataset
    test_data = load_clue_dataset(args.dataset)

    # Store evaluation results
    exact_match_scores = []
    f1_scores = []
    predictions = []

    print(f"\n🔍 Evaluating Model on {args.dataset} Benchmark...")

    for example in test_data:
        context = example["context"]
        question = example["question"]
        ground_truth = example["answers"]["text"][0]  # Extract the first valid answer

        # ✅ Properly apply chat template
        messages = [{"role": "user", "content": f"{question}\n\n{context}"}]
        formatted_input = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )[-args.max_seq_len + 1:]

        # Generate model output
        with torch.no_grad():
            x = torch.tensor(tokenizer(formatted_input)['input_ids'], device=args.device).unsqueeze(0)
            outputs = model.generate(
                x,
                eos_token_id=tokenizer.eos_token_id,
                max_new_tokens=100,  # Increase space for generation
                temperature=args.temperature,
                top_p=args.top_p,
                pad_token_id=tokenizer.pad_token_id
            )

        # ✅ Properly decode response following the original logic
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answer = generated_text.strip().split("\n")[0]  # Take first sentence

        # Compute evaluation scores
        em_score = compute_exact_match(answer, ground_truth)
        f1_score = compute_f1(answer, ground_truth)

        exact_match_scores.append(em_score)
        f1_scores.append(f1_score)

        predictions.append({
            "question": question,
            "context": context,
            "ground_truth": ground_truth,
            "prediction": answer
        })

    # Compute final evaluation results
    final_em = np.mean(exact_match_scores) * 100
    final_f1 = np.mean(f1_scores) * 100

    print(f"\n✅ Evaluation Complete!")
    print(f"📊 Exact Match (EM): {final_em:.2f}%")
    print(f"📊 F1 Score: {final_f1:.2f}%")

    # Print sample outputs
    print("\n🔹 Sample Predictions:")
    for i, sample in enumerate(predictions[:5]):  # Show 5 samples
        print(f"\n📝 Question: {sample['question']}")
        print(f"📖 Context: {sample['context'][:300]}...")  # Truncate long context
        print(f"✅ Ground Truth: {sample['ground_truth']}")
        print(f"🤖 Model Prediction: {sample['prediction']}")


if __name__ == "__main__":
    main()