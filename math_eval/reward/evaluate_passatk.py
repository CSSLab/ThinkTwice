from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DIR = REPO_ROOT / "scratch"

NUM_SAMPLES = 64
PASS_K_VALUES = [1, 2, 4, 8, 16, 32, 64]
RANDOM_SEED = 42

BENCHMARKS = {
    "AIME24": SCRATCH_DIR / "aime2024/test.parquet",
    # "AMC": SCRATCH_DIR / "amc/test.parquet",
    # "MATH500": SCRATCH_DIR / "math500/test.parquet",
    # "Minerva": SCRATCH_DIR / "minerva_math/test.parquet",
    # "OlympiadBench": SCRATCH_DIR / "olympiadbench/test.parquet",
}

DEFAULT_REFLECTION_INSTRUCTION = (
    "Follow this instruction, carefully review your previous solution:\n"
    "1. Go through each calculation step-by-step. Check if there are any errors in calculations, logic, or problem understanding.\n"
    "2. If you find any mistakes, explicitly point out what was wrong and explain the correct approach.\n"
    "3. If the solution is already correct, verify each step and explain it more clearly.\n"
    "4. Finally, after finishing the review, provide your refined solution and answer.\n"
)


def _load_boxed_reward_fn():
    grader_path = REPO_ROOT / "verl" / "verl" / "utils" / "reward_score" / "hendrycks_math_grader.py"
    spec = importlib.util.spec_from_file_location("hendrycks_math_grader", grader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load hendrycks_math_grader from {grader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.boxed_reward_fn


def compute_pass_at_k(correct_counts, total_samples, k_values):
    results = {}
    for k in k_values:
        pass_at_k = np.mean([1 - (1 - c / total_samples) ** k for c in correct_counts])
        results[k] = pass_at_k * 100
    return results


def construct_reflection_prompts(tokenizer, raw_questions, base_responses):
    prompts = []
    for question, answer in zip(raw_questions, base_responses):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": DEFAULT_REFLECTION_INSTRUCTION},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    return prompts


def run_benchmark(llm, tokenizer, boxed_reward_fn, bench_name, bench_path, num_samples, k_values, results):
    df = pd.read_parquet(bench_path)

    print(f"\n{'='*80}")
    print(f"BENCHMARK: {bench_name}")
    print(f"{'='*80}")
    print(f"Total problems: {len(df)}")

    all_prompts = []
    metadata = []

    for idx, row in df.iterrows():
        prompt_messages = row['prompt']
        ground_truth = row['reward_model']['ground_truth']
        raw_question = prompt_messages[0]['content'] if len(prompt_messages) > 0 else ""

        formatted_prompt = tokenizer.apply_chat_template(
            prompt_messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )

        all_prompts.append(formatted_prompt)
        metadata.append({
            'problem_idx': idx,
            'ground_truth': ground_truth,
            'raw_question': raw_question
        })

    sampling_params = SamplingParams(
        n=num_samples,
        temperature=0.7,
        top_p=0.8,
        top_k=20,
        max_tokens=3000,
        seed=RANDOM_SEED,
        repetition_penalty=1.0,
    )

    print(f"Generating {len(all_prompts)} prompts x {num_samples} samples...")
    outputs = llm.generate(all_prompts, sampling_params)

    for idx, output in enumerate(outputs):
        for sample_idx, sample_output in enumerate(output.outputs):
            response = sample_output.text
            ground_truth = metadata[idx]['ground_truth']
            _, score = boxed_reward_fn(response, ground_truth, fast=True)
            metadata[idx][f'score_{sample_idx}'] = score
            metadata[idx][f'response_{sample_idx}'] = response

    correct_counts = []
    for meta in metadata:
        scores = [meta.get(f'score_{i}', 0.0) for i in range(num_samples)]
        correct_count = sum(1 for s in scores if s == 1.0)
        correct_counts.append(correct_count)

    pass_at_k_results = compute_pass_at_k(correct_counts, num_samples, k_values)

    print(f"\n--- Base pass@k ---")
    for k in k_values:
        print(f"  pass@{k:2d} = {pass_at_k_results[k]:.2f}%")

    results[bench_name]['base'] = pass_at_k_results

    reflection_prompts = []
    reflection_base_info = []
    reflection_metadata = []

    for base_idx, m in enumerate(metadata):
        raw_question = m['raw_question']
        for sample_idx in range(num_samples):
            base_response = m.get(f'response_{sample_idx}', '')
            if not base_response:
                continue
            refl_prompts = construct_reflection_prompts(tokenizer, [raw_question], [base_response])
            reflection_prompts.append(refl_prompts[0])
            reflection_base_info.append((base_idx, sample_idx))
            reflection_metadata.append({
                'problem_idx': m['problem_idx'],
                'base_score': m.get(f'score_{sample_idx}', 0.0),
                'ground_truth': m['ground_truth']
            })

    refl_sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=3000,
        seed=RANDOM_SEED,
        repetition_penalty=1.0,
    )

    print(f"\nGenerating {len(reflection_prompts)} reflections...")
    reflection_outputs = llm.generate(reflection_prompts, refl_sampling_params)

    for idx, output in enumerate(reflection_outputs):
        response = output.outputs[0].text
        ground_truth = reflection_metadata[idx]['ground_truth']
        _, score = boxed_reward_fn(response, ground_truth, fast=True)
        reflection_metadata[idx]['reflection_score'] = score

    problem_data = {}
    for rm in reflection_metadata:
        prob_idx = rm['problem_idx']
        if prob_idx not in problem_data:
            problem_data[prob_idx] = []
        problem_data[prob_idx].append(rm)

    refl_correct_counts = []
    for samples in problem_data.values():
        refl_scores = [s['reflection_score'] for s in samples if 'reflection_score' in s]
        correct_count = sum(1 for s in refl_scores if s == 1.0)
        refl_correct_counts.append(correct_count)

    refl_pass_at_k = compute_pass_at_k(refl_correct_counts, num_samples, k_values)

    print(f"\n--- Reflection pass@k ---")
    for k in k_values:
        print(f"  pass@{k:2d} = {refl_pass_at_k[k]:.2f}%")

    results[bench_name]['reflection'] = refl_pass_at_k

    base_scores = []
    refl_scores = []
    for samples in problem_data.values():
        for s in samples:
            base_scores.append(s['base_score'])
            if 'reflection_score' in s:
                refl_scores.append(s['reflection_score'])

    bad_to_good = sum(1 for b, r in zip(base_scores, refl_scores) if b < 1.0 and r >= 1.0)
    total_bad = sum(1 for b in base_scores if b < 1.0)
    bad_to_good_rate = (bad_to_good / total_bad * 100) if total_bad > 0 else 0.0

    print(f"\nBad -> Good rate: {bad_to_good_rate:.2f}%")

    results[bench_name]['bad_to_good'] = bad_to_good_rate

    return results


def main():
    boxed_reward_fn = _load_boxed_reward_fn()

    # Set model_path via command line argument or environment variable
    # Example: CKPT_ROOT=${HOME}/ckpts python evaluate_passatk_posthoc.py
    default_model_path = os.path.expandvars("${CKPT_ROOT}/${MODEL_NAME:-grpo_math_refpo_dapo_Qwen3-4B-Instruct-2507}/best_model")
    model_path = os.sys.argv[1] if len(os.sys.argv) > 1 else default_model_path

    if len(os.sys.argv) > 1:
        model_path = os.sys.argv[1]

    print(f"Model: {model_path}")
    print(f"Num samples: {NUM_SAMPLES}")
    print(f"Pass@k values: {PASS_K_VALUES}")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.8,
        trust_remote_code=True,
        seed=RANDOM_SEED,
        disable_log_stats=True,
        enforce_eager=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    results = {bench: {} for bench in BENCHMARKS.keys()}

    for bench_name, bench_path in BENCHMARKS.items():
        run_benchmark(llm, tokenizer, boxed_reward_fn, bench_name, bench_path, NUM_SAMPLES, PASS_K_VALUES, results)

    output_path = REPO_ROOT / "math_eval" / "reward" / "passatk_posthoc_results_refpo_qwen3.json"
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*80}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*80}")

    print("\nSummary:")
    for bench_name, data in results.items():
        print(f"\n{bench_name}:")
        print("  Base:")
        for k in PASS_K_VALUES:
            print(f"    pass@{k:2d} = {data['base'][k]:6.2f}%", end="")
        print()
        print("  Reflection:")
        for k in PASS_K_VALUES:
            print(f"    pass@{k:2d} = {data['reflection'][k]:6.2f}%", end="")
        print()
        print(f"  Bad->Good: {data.get('bad_to_good', 0):.2f}%")


if __name__ == "__main__":
    main()
