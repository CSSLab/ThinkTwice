from __future__ import annotations

import importlib.util
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DIR = REPO_ROOT / "scratch"


def _load_boxed_reward_fn():
    grader_path = REPO_ROOT / "verl" / "verl" / "utils" / "reward_score" / "hendrycks_math_grader.py"
    spec = importlib.util.spec_from_file_location("hendrycks_math_grader", grader_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load hendrycks_math_grader from {grader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.boxed_reward_fn


def _infer_tensor_parallel_size(default_tp: int = 2) -> int:
    cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cuda_visible_devices:
        return default_tp
    devices = [dev.strip() for dev in cuda_visible_devices.split(",") if dev.strip()]
    if not devices:
        return default_tp
    return min(default_tp, len(devices))

DEFAULT_REFLECTION_INSTRUCTION = (
    "Follow this instruction, carefully review your previous solution:\n"
    "1. Go through each calculation step-by-step. Check if there are any errors in calculations, logic, or problem understanding.\n"
    "2. If you find any mistakes, explicitly point out what was wrong and explain the correct approach.\n"
    "3. If no mistakes are found, organize the steps to make them more concise and keep the correct answer.\n"
    "4. Finally, after finishing the review, you MUST write 'Improved solution:' and then provide your refined solution and answer inside \\boxed{}, like \\boxed{42} or \\boxed{\\frac{1}{2}}.\n"
)

BENCHMARKS = {
    "AIME24": SCRATCH_DIR / "aime2024/test.parquet",
    "AMC": SCRATCH_DIR / "amc/test.parquet",
    "MATH500": SCRATCH_DIR / "math500/test.parquet",
    "Minerva": SCRATCH_DIR / "minerva_math/test.parquet",
    "OlympiadBench": SCRATCH_DIR / "olympiadbench/test.parquet",
}

SAMPLE_SIZES = {
    "AIME24": None,
    "AMC": None,
    "MATH500": 100,
    "Minerva": 100,
    "OlympiadBench": 100,
}

# SAMPLE_SIZES = {
#     "AIME24": 1,
#     "AMC": 1,
#     "MATH500": 1,
#     "Minerva": 1,
#     "OlympiadBench": 1,
# }

N_SAMPLES = 4
RANDOM_SEED = 42
ENABLE_REFLECTION = True

def compute_metrics(scores):
    pass_at_n = 1.0 if any(s == 1.0 for s in scores) else 0.0
    avg_at_n = np.mean(scores)
    maj_at_n = 1.0 if Counter(scores).most_common(1)[0][0] == 1.0 else 0.0
    return pass_at_n, avg_at_n, maj_at_n

def construct_reflection_prompts(tokenizer, raw_questions, base_responses):
    """Construct reflection prompts matching ray_trainer.py format."""
    reflection_prompts = []
    for question, answer in zip(raw_questions, base_responses):
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": DEFAULT_REFLECTION_INSTRUCTION},
        ]
        reflection_prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        reflection_prompts.append(reflection_prompt)
    return reflection_prompts

def main():
    boxed_reward_fn = _load_boxed_reward_fn()
    tensor_parallel_size = _infer_tensor_parallel_size()

    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        # model_path = "/datadrive2/checkpoints/grpo_math_baseline_Qwen3-4B-Instruct-2507/best_model"
        model_path = "Qwen/Qwen3-4B-Instruct-2507"

    print(f"Loading model: {model_path}")
    print(f"vLLM config: tp={tensor_parallel_size}, 0.9 GPU utilization")
    print(f"Sampling: {N_SAMPLES} responses per problem")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Reflection enabled: {ENABLE_REFLECTION}\n")

    llm = LLM(
        model=model_path,
        tensor_parallel_size=tensor_parallel_size,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=RANDOM_SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    np.random.seed(RANDOM_SEED)

    # ============================================================================
    # PHASE 1: BASE GENERATION
    # ============================================================================
    print(f"{'='*80}")
    print("PHASE 1: BASE GENERATION")
    print(f"{'='*80}")

    all_prompts = []
    metadata = []
    raw_questions_per_problem = {}

    for bench_name, bench_path in BENCHMARKS.items():
        df = pd.read_parquet(bench_path)

        sample_size = SAMPLE_SIZES[bench_name]
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

        print(f"Loading {bench_name}: {len(df)} problems x {N_SAMPLES} samples = {len(df) * N_SAMPLES} total")

        for idx, row in df.iterrows():
            prompt_messages = row['prompt']
            ground_truth = row['reward_model']['ground_truth']

            raw_question = prompt_messages[0]['content'] if len(prompt_messages) > 0 else ""
            problem_key = (bench_name, idx)
            raw_questions_per_problem[problem_key] = raw_question

            formatted_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            for sample_idx in range(N_SAMPLES):
                all_prompts.append(formatted_prompt)
                metadata.append({
                    'benchmark': bench_name,
                    'problem_idx': idx,
                    'sample_idx': sample_idx,
                    'ground_truth': ground_truth,
                    'raw_question': raw_question
                })

    print(f"\nTotal base prompts to generate: {len(all_prompts)}")

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=3000,
        top_p=1.0,
    )

    print("Generating all base responses in one batch...")
    outputs = llm.generate(all_prompts, sampling_params)

    print("Grading base responses...")
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        ground_truth = metadata[idx]['ground_truth']
        grading_info, score = boxed_reward_fn(response, ground_truth, fast=True)
        metadata[idx]['score'] = score
        metadata[idx]['response'] = response

    # Compute base metrics
    base_results = {}
    for bench_name in BENCHMARKS.keys():
        bench_metadata = [m for m in metadata if m['benchmark'] == bench_name]
        problem_indices = set(m['problem_idx'] for m in bench_metadata)

        pass_scores = []
        avg_scores = []
        maj_scores = []

        for prob_idx in problem_indices:
            prob_samples = [m for m in bench_metadata if m['problem_idx'] == prob_idx]
            scores = [m['score'] for m in prob_samples]

            pass_at_n, avg_at_n, maj_at_n = compute_metrics(scores)
            pass_scores.append(pass_at_n)
            avg_scores.append(avg_at_n)
            maj_scores.append(maj_at_n)

        base_results[bench_name] = {
            'n_problems': len(problem_indices),
            f'pass@{N_SAMPLES}': np.mean(pass_scores) * 100,
            f'avg@{N_SAMPLES}': np.mean(avg_scores) * 100,
            f'maj@{N_SAMPLES}': np.mean(maj_scores) * 100,
        }

    print(f"\n{'='*80}")
    print("BASE RESULTS")
    print(f"{'='*80}")
    print(f"{'Benchmark':<15} {'N':<6} {'pass@'+str(N_SAMPLES):<12} {'avg@'+str(N_SAMPLES):<12} {'maj@'+str(N_SAMPLES):<12}")
    print(f"{'-'*80}")

    total_pass = 0
    total_avg = 0
    total_maj = 0

    for bench_name, result in base_results.items():
        print(f"{bench_name:<15} {result['n_problems']:<6} "
              f"{result[f'pass@{N_SAMPLES}']:>6.2f}%      "
              f"{result[f'avg@{N_SAMPLES}']:>6.2f}%      "
              f"{result[f'maj@{N_SAMPLES}']:>6.2f}%")
        total_pass += result[f'pass@{N_SAMPLES}']
        total_avg += result[f'avg@{N_SAMPLES}']
        total_maj += result[f'maj@{N_SAMPLES}']

    avg_pass = total_pass / len(base_results)
    avg_avg = total_avg / len(base_results)
    avg_maj = total_maj / len(base_results)

    print(f"{'-'*80}")
    print(f"{'Average':<15} {'':6} {avg_pass:>6.2f}%      {avg_avg:>6.2f}%      {avg_maj:>6.2f}%")

    if not ENABLE_REFLECTION:
        return

    # ============================================================================
    # PHASE 2: REFLECTION GENERATION
    # ============================================================================
    print(f"\n{'='*80}")
    print("PHASE 2: REFLECTION GENERATION")
    print(f"{'='*80}")

    reflection_prompts = []
    reflection_metadata = []

    for m in metadata:
        raw_question = m['raw_question']
        base_response = m['response']

        reflection_prompt_list = construct_reflection_prompts(
            tokenizer, [raw_question], [base_response]
        )
        reflection_prompt = reflection_prompt_list[0]

        reflection_prompts.append(reflection_prompt)
        reflection_metadata.append({
            'benchmark': m['benchmark'],
            'problem_idx': m['problem_idx'],
            'sample_idx': m['sample_idx'],
            'ground_truth': m['ground_truth'],
            'base_score': m['score'],
            'raw_question': raw_question,
            'base_response': base_response
        })

    print(f"Total reflection prompts to generate: {len(reflection_prompts)}")

    print("Generating all reflection responses in one batch...")
    reflection_outputs = llm.generate(reflection_prompts, sampling_params)

    print("Grading reflection responses...")
    for idx, output in enumerate(reflection_outputs):
        response = output.outputs[0].text
        ground_truth = reflection_metadata[idx]['ground_truth']
        grading_info, score = boxed_reward_fn(response, ground_truth, fast=True)
        reflection_metadata[idx]['reflection_score'] = score
        reflection_metadata[idx]['reflection_response'] = response

    # Compute reflection metrics
    reflection_results = {}
    for bench_name in BENCHMARKS.keys():
        bench_metadata = [m for m in reflection_metadata if m['benchmark'] == bench_name]
        problem_indices = set(m['problem_idx'] for m in bench_metadata)

        pass_scores = []
        avg_scores = []
        maj_scores = []

        for prob_idx in problem_indices:
            prob_samples = [m for m in bench_metadata if m['problem_idx'] == prob_idx]
            scores = [m['reflection_score'] for m in prob_samples]

            pass_at_n, avg_at_n, maj_at_n = compute_metrics(scores)
            pass_scores.append(pass_at_n)
            avg_scores.append(avg_at_n)
            maj_scores.append(maj_at_n)

        reflection_results[bench_name] = {
            'n_problems': len(problem_indices),
            f'pass@{N_SAMPLES}': np.mean(pass_scores) * 100,
            f'avg@{N_SAMPLES}': np.mean(avg_scores) * 100,
            f'maj@{N_SAMPLES}': np.mean(maj_scores) * 100,
        }

    print(f"\n{'='*80}")
    print("REFLECTION RESULTS")
    print(f"{'='*80}")
    print(f"{'Benchmark':<15} {'N':<6} {'pass@'+str(N_SAMPLES):<12} {'avg@'+str(N_SAMPLES):<12} {'maj@'+str(N_SAMPLES):<12}")
    print(f"{'-'*80}")

    total_pass = 0
    total_avg = 0
    total_maj = 0

    for bench_name, result in reflection_results.items():
        print(f"{bench_name:<15} {result['n_problems']:<6} "
              f"{result[f'pass@{N_SAMPLES}']:>6.2f}%      "
              f"{result[f'avg@{N_SAMPLES}']:>6.2f}%      "
              f"{result[f'maj@{N_SAMPLES}']:>6.2f}%")
        total_pass += result[f'pass@{N_SAMPLES}']
        total_avg += result[f'avg@{N_SAMPLES}']
        total_maj += result[f'maj@{N_SAMPLES}']

    avg_pass = total_pass / len(reflection_results)
    avg_avg = total_avg / len(reflection_results)
    avg_maj = total_maj / len(reflection_results)

    print(f"{'-'*80}")
    print(f"{'Average':<15} {'':6} {avg_pass:>6.2f}%      {avg_avg:>6.2f}%      {avg_maj:>6.2f}%")

    # ============================================================================
    # PHASE 3: TRANSITION METRICS
    # ============================================================================
    print(f"\n{'='*80}")
    print("TRANSITION METRICS")
    print(f"{'='*80}")

    transition_results = {}

    for bench_name in BENCHMARKS.keys():
        bench_refl_meta = [m for m in reflection_metadata if m['benchmark'] == bench_name]

        if not bench_refl_meta:
            continue

        base_scores = [m['base_score'] for m in bench_refl_meta]
        refl_scores = [m['reflection_score'] for m in bench_refl_meta]

        good_to_good = sum(1 for b, r in zip(base_scores, refl_scores) if b >= 1.0 and r >= 1.0)
        good_to_bad = sum(1 for b, r in zip(base_scores, refl_scores) if b >= 1.0 and r < 1.0)
        bad_to_good = sum(1 for b, r in zip(base_scores, refl_scores) if b < 1.0 and r >= 1.0)
        bad_to_bad = sum(1 for b, r in zip(base_scores, refl_scores) if b < 1.0 and r < 1.0)

        num_good = sum(1 for b in base_scores if b >= 1.0)
        num_bad = sum(1 for b in base_scores if b < 1.0)

        transition_results[bench_name] = {
            'good→good': (good_to_good / num_good * 100) if num_good > 0 else 0.0,
            'good→bad': (good_to_bad / num_good * 100) if num_good > 0 else 0.0,
            'bad→good': (bad_to_good / num_bad * 100) if num_bad > 0 else 0.0,
            'bad→bad': (bad_to_bad / num_bad * 100) if num_bad > 0 else 0.0,
            'num_good': num_good,
            'num_bad': num_bad
        }

    print(f"{'Benchmark':<15} {'good→good':<12} {'good→bad':<12} {'bad→good':<12} {'bad→bad':<12}")
    print(f"{'-'*80}")

    avg_g2g = 0
    avg_g2b = 0
    avg_b2g = 0
    avg_b2b = 0

    for bench_name, trans in transition_results.items():
        print(f"{bench_name:<15} {trans['good→good']:>6.2f}%      "
              f"{trans['good→bad']:>6.2f}%      "
              f"{trans['bad→good']:>6.2f}%      "
              f"{trans['bad→bad']:>6.2f}%")
        avg_g2g += trans['good→good']
        avg_g2b += trans['good→bad']
        avg_b2g += trans['bad→good']
        avg_b2b += trans['bad→bad']

    n_benchmarks = len(transition_results)
    avg_g2g /= n_benchmarks
    avg_g2b /= n_benchmarks
    avg_b2g /= n_benchmarks
    avg_b2b /= n_benchmarks

    print(f"{'-'*80}")
    print(f"{'Average':<15} {avg_g2g:>6.2f}%      {avg_g2b:>6.2f}%      {avg_b2g:>6.2f}%      {avg_b2b:>6.2f}%")

if __name__ == "__main__":
    main()
