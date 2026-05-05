"""
Evaluate a LoRA-fine-tuned model. Loads the frozen pretrained base checkpoint,
re-applies the LoRA wrappers using the saved lora_config, and loads the saved
adapter weights. Then runs the standard chat evaluation tasks.

Example runs:
python -m scripts.lora_eval -g <lora_tag> -a ARC-Easy
torchrun --nproc_per_node=8 -m scripts.lora_eval -- -g <lora_tag>
"""

import os
import json
import glob
import argparse

import torch

from nanochat.common import compute_init, compute_cleanup, print0, autodetect_device_type, get_base_dir
from nanochat.checkpoint_manager import load_model
from nanochat.engine import Engine
from nanochat.lora import apply_lora, freeze_base, load_lora_state_dict, count_trainable
from scripts.chat_eval import run_chat_eval


def find_last_lora_step(lora_dir):
    files = glob.glob(os.path.join(lora_dir, "lora_*.pt"))
    if not files:
        raise FileNotFoundError(f"No LoRA checkpoints found in {lora_dir}")
    return int(max(os.path.basename(f).split("_")[-1].split(".")[0] for f in files))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-g', '--model-tag', type=str, required=True, help='LoRA output tag (directory under lora_checkpoints/)')
    parser.add_argument('-s', '--step', type=int, default=None, help='LoRA checkpoint step (default: last)')
    parser.add_argument('-a', '--task-name', type=str, default=None, help="Task name. Default = all tasks. Use | to split multiple tasks.")
    parser.add_argument('-t', '--temperature', type=float, default=0.0)
    parser.add_argument('-m', '--max-new-tokens', type=int, default=512)
    parser.add_argument('-n', '--num-samples', type=int, default=1)
    parser.add_argument('-k', '--top-k', type=int, default=50)
    parser.add_argument('-b', '--batch-size', type=int, default=8, help='Batch size for categorical evaluation')
    parser.add_argument('-x', '--max-problems', type=int, default=None, help='Max problems to evaluate')
    parser.add_argument('--device-type', type=str, default='', choices=['cuda', 'cpu', 'mps', ''], help='Device type for evaluation: cuda|cpu|mps. empty => autodetect')
    parser.add_argument('--base-model-tag', type=str, default=None, help='Override base model tag (default: read from LoRA meta)')
    parser.add_argument('--base-model-step', type=int, default=None, help='Override base model step (default: read from LoRA meta)')
    args = parser.parse_args()

    device_type = autodetect_device_type() if args.device_type == "" else args.device_type
    ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)

    # Locate the LoRA checkpoint
    base_dir = get_base_dir()
    lora_dir = os.path.join(base_dir, "lora_checkpoints", args.model_tag)
    step = args.step if args.step is not None else find_last_lora_step(lora_dir)
    lora_path = os.path.join(lora_dir, f"lora_{step:06d}.pt")
    meta_path = os.path.join(lora_dir, f"meta_{step:06d}.json")
    print0(f"Loading LoRA checkpoint: {lora_path}")
    print0(f"Loading LoRA meta: {meta_path}")
    with open(meta_path, "r", encoding="utf-8") as f:
        lora_meta = json.load(f)

    base_tag = args.base_model_tag if args.base_model_tag is not None else lora_meta.get("base_model_tag")
    base_step = args.base_model_step if args.base_model_step is not None else lora_meta.get("base_model_step")
    print0(f"Loading base model tag={base_tag}, step={base_step}")
    model, tokenizer, base_meta = load_model("base", device, phase="eval", model_tag=base_tag, step=base_step)

    # Re-apply LoRA wrappers using the saved config, then load the adapter weights
    lora_config = lora_meta["lora_config"]
    print0(f"Applying LoRA: {lora_config}")
    apply_lora(
        model,
        rank=lora_config["rank"],
        alpha=lora_config["alpha"],
        dropout=lora_config.get("dropout", 0.0),
        target_modules=lora_config["target_modules"],
    )
    freeze_base(model)
    trainable, total, pct = count_trainable(model)
    print0(f"Trainable (LoRA) params: {trainable:,} / {total:,} ({pct:.4f}%)")

    lora_sd = torch.load(lora_path, map_location=device)
    load_lora_state_dict(model, lora_sd)
    model.eval()
    print0("LoRA adapters loaded.")

    engine = Engine(model, tokenizer)

    all_tasks = ['ARC-Easy', 'ARC-Challenge', 'MMLU', 'GSM8K', 'HumanEval', 'SpellingBee']
    baseline_accuracies = {
        'ARC-Easy': 0.25,
        'ARC-Challenge': 0.25,
        'MMLU': 0.25,
        'GSM8K': 0.0,
        'HumanEval': 0.0,
        'SpellingBee': 0.0,
    }
    task_names = all_tasks if args.task_name is None else args.task_name.split('|')

    results = {}
    for task_name in task_names:
        acc = run_chat_eval(
            task_name,
            model, tokenizer, engine,
            batch_size=args.batch_size,
            num_samples=args.num_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            max_problems=args.max_problems,
        )
        results[task_name] = acc
        print0(f"{task_name} accuracy: {100 * acc:.2f}%")

    from nanochat.report import get_report
    all_tasks_were_evaluated = all(task_name in results for task_name in all_tasks)
    chatcore_metric_dict = {}
    if all_tasks_were_evaluated:
        centered_mean = 0
        for task_name, acc in results.items():
            baseline_acc = baseline_accuracies.get(task_name, 0.0)
            centered_acc = (acc - baseline_acc) / (1.0 - baseline_acc)
            centered_mean += centered_acc
        chatcore_metric = centered_mean / len(results)
        chatcore_metric_dict = {"ChatCORE metric": chatcore_metric}
    get_report(args.model_tag).log(section="LoRA evaluation", data=[
        vars(args),
        {
            "Base model tag": base_tag,
            "Base model step": base_step,
            "LoRA step": step,
            "LoRA config": lora_config,
        },
        results,
        chatcore_metric_dict,
    ])

    compute_cleanup()
