from __future__ import annotations

import importlib.util
import json
import os
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

os.environ["VLLM_BATCH_INVARIANT"] = "1"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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


DEFAULT_REFLECTION_INSTRUCTION = (
    "Follow this instruction, carefully review your previous solution:\n"
    "1. Go through each calculation step-by-step. Check if there are any errors in calculations, logic, or problem understanding.\n"
    "2. If you find any mistakes, explicitly point out what was wrong and explain the correct approach.\n"
    "3. If the solution is already correct, verify each step and explain it more clearly.\n"
    "4. Finally, after finishing the review, provide your refined solution and answer.\n"
)


BENCHMARKS = {
    "AIME24": SCRATCH_DIR / "aime2024/test.parquet",
    "AMC": SCRATCH_DIR / "amc/test.parquet",
    "MATH500": SCRATCH_DIR / "math500/test.parquet",
    "Minerva": SCRATCH_DIR / "minerva_math/test.parquet",
    "OlympiadBench": SCRATCH_DIR / "olympiadbench/test.parquet",
}


def get_model_configs():
    """Get model configs from environment variables or use defaults."""
    ckpt_root = os.path.expandvars(os.environ.get("CKPT_ROOT", f"{os.path.expanduser('~')}/ckpts"))
    return {
        "base": "Qwen/Qwen3-4B-Instruct-2507",
        "grpo": f"{ckpt_root}/grpo_math_baseline_Qwen3-4B-Instruct-2507/best_model",
        "drgrpo": f"{ckpt_root}/drgrpo_math_baseline_Qwen3-4B-Instruct-2507/best_model",
        "dapo": f"{ckpt_root}/dapo_math_Qwen3-4B-Instruct-2507-dapo-r8/best_model",
        "refpo": f"{ckpt_root}/grpo_math_refpo_dapo_Qwen3-4B-Instruct-2507/best_model",
    }

MODEL_CONFIGS = get_model_configs()

# Separation: BASE_MODELS generate base responses, REFL_MODELS generate reflections
# For full cross-evaluation, set both to all models. For targeted evaluation, use subsets.
BASE_MODELS = [
    "dapo",
    "refpo",
]

REFL_MODELS = [
    "base",
    "grpo",
    "drgrpo",
    "dapo",
    "refpo",
]


SAMPLE_SIZE = None
RANDOM_SEED = 42
MAX_TOKENS = 3000
N_REFLECTION_SAMPLES = 4
REFLECTION_TEMPERATURE = 0.7
REFLECTION_TOP_P = 0.9


def construct_reflection_prompts(tokenizer, raw_questions, base_responses):
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


def load_benchmark_data():
    all_prompts = []
    metadata = []

    for bench_name, bench_path in BENCHMARKS.items():
        df = pd.read_parquet(bench_path)

        if SAMPLE_SIZE and len(df) > SAMPLE_SIZE:
            df = df.sample(n=SAMPLE_SIZE, random_state=RANDOM_SEED).reset_index(drop=True)

        print(f"Loading {bench_name}: {len(df)} problems")

        for idx, row in df.iterrows():
            prompt_messages = row['prompt']
            ground_truth = row['reward_model']['ground_truth']
            raw_question = prompt_messages[0]['content'] if len(prompt_messages) > 0 else ""

            all_prompts.append(prompt_messages)
            metadata.append({
                'benchmark': bench_name,
                'problem_idx': idx,
                'ground_truth': ground_truth,
                'raw_question': raw_question
            })

    return all_prompts, metadata


def generate_base_responses(model_name: str, model_path: str, prompts: list, metadata: list):
    print(f"\n{'='*80}")
    print(f"GENERATING BASE RESPONSES: {model_name}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        seed=RANDOM_SEED,
        disable_log_stats=True,
        enforce_eager=True,
    )

    formatted_prompts = [
        tokenizer.apply_chat_template(
            p,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        ) for p in prompts
    ]

    sampling_params = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=MAX_TOKENS,
        seed=RANDOM_SEED,
    )

    print(f"Generating {len(formatted_prompts)} base responses...")
    outputs = llm.generate(formatted_prompts, sampling_params)

    base_responses = []
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        base_responses.append(response)

    del llm
    torch.cuda.empty_cache()

    return base_responses


def generate_reflection_responses(
    model_name: str,
    model_path: str,
    raw_questions: list,
    base_responses: list,
    metadata: list
):
    print(f"\n{'='*80}")
    print(f"GENERATING REFLECTION RESPONSES: {model_name}")
    print(f"{'='*80}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.85,
        trust_remote_code=True,
        seed=RANDOM_SEED,
        disable_log_stats=True,
        enforce_eager=True,
    )

    reflection_prompts = construct_reflection_prompts(tokenizer, raw_questions, base_responses)

    sampling_params = SamplingParams(
        n=N_REFLECTION_SAMPLES,
        temperature=REFLECTION_TEMPERATURE,
        top_p=REFLECTION_TOP_P,
        top_k=-1,
        max_tokens=MAX_TOKENS,
        seed=RANDOM_SEED,
    )

    print(f"Generating {len(reflection_prompts)} x {N_REFLECTION_SAMPLES} reflection responses...")
    outputs = llm.generate(reflection_prompts, sampling_params)

    reflection_responses = []
    for output in outputs:
        samples = [sample.text for sample in output.outputs]
        reflection_responses.append(samples)

    del llm
    torch.cuda.empty_cache()

    return reflection_responses


def compute_pass_at_k(scores):
    return 1.0 if any(s == 1.0 for s in scores) else 0.0


def compute_benchmark_metrics(metadata_list, score_key, n_samples=1):
    results = {}

    for bench_name in BENCHMARKS.keys():
        bench_metadata = [m for m in metadata_list if m['benchmark'] == bench_name]
        problem_indices = sorted(set(m['problem_idx'] for m in bench_metadata))

        pass_at_n_scores = []
        pass_at_1_scores = []
        avg_scores = []

        for prob_idx in problem_indices:
            problem_metas = [m for m in bench_metadata if m['problem_idx'] == prob_idx]

            if n_samples == 1:
                scores = [problem_metas[0].get(score_key, 0.0)]
            else:
                scores = [m.get(score_key, 0.0) for m in problem_metas]

            pass_at_n = 1.0 if any(s == 1.0 for s in scores) else 0.0
            pass_at_1 = scores[0] if scores else 0.0

            pass_at_n_scores.append(pass_at_n)
            pass_at_1_scores.append(pass_at_1)
            avg_scores.extend(scores)

        if pass_at_n_scores:
            results[bench_name] = {
                'n_problems': len(problem_indices),
                f'pass@{n_samples}': np.mean(pass_at_n_scores) * 100,
                'pass@1': np.mean(pass_at_1_scores) * 100,
                'avg@1': np.mean(avg_scores) * 100,
            }
        else:
            results[bench_name] = {
                'n_problems': 0,
                f'pass@{n_samples}': 0.0,
                'pass@1': 0.0,
                'avg@1': 0.0,
            }

    return results


def print_results(title: str, results: dict):
    print(f"\n{'='*80}")
    print(title)
    print(f"{'='*80}")
    print(f"{'Benchmark':<15} {'N':<6} {'pass@1':<12} {'avg@1':<12}")
    print(f"{'-'*80}")

    total_pass = 0
    total_avg = 0

    for bench_name, result in results.items():
        print(f"{bench_name:<15} {result['n_problems']:<6} "
              f"{result['pass@1']:>6.2f}%      "
              f"{result['avg@1']:>6.2f}%")
        total_pass += result['pass@1']
        total_avg += result['avg@1']

    avg_pass = total_pass / len(results)
    avg_avg = total_avg / len(results)

    print(f"{'-'*80}")
    print(f"{'Average':<15} {'':6} {avg_pass:>6.2f}%      {avg_avg:>6.2f}%")


def compute_transition_metrics(base_scores, reflection_scores):
    good_to_good = sum(1 for b, r in zip(base_scores, reflection_scores) if b >= 1.0 and r >= 1.0)
    good_to_bad = sum(1 for b, r in zip(base_scores, reflection_scores) if b >= 1.0 and r < 1.0)
    bad_to_good = sum(1 for b, r in zip(base_scores, reflection_scores) if b < 1.0 and r >= 1.0)
    bad_to_bad = sum(1 for b, r in zip(base_scores, reflection_scores) if b < 1.0 and r < 1.0)

    num_good = sum(1 for b in base_scores if b >= 1.0)
    num_bad = sum(1 for b in base_scores if b < 1.0)

    return {
        'good->good': (good_to_good / num_good * 100) if num_good > 0 else 0.0,
        'good->bad': (good_to_bad / num_good * 100) if num_good > 0 else 0.0,
        'bad->good': (bad_to_good / num_bad * 100) if num_bad > 0 else 0.0,
        'bad->bad': (bad_to_bad / num_bad * 100) if num_bad > 0 else 0.0,
    }


def print_matrix(matrix: dict, base_models: list, refl_models: list, n_samples: int = 4):
    print(f"\n{'='*100}")
    print(f"CROSS-REFLECTION MATRIX (pass@{n_samples} %)")
    print(f"{'='*100}")
    header = "Base\\Refl"
    print(f"{header:<15}", end="")
    for refl in refl_models:
        print(f"{refl:<12}", end="")
    print()
    print(f"{'-'*100}")

    for base in base_models:
        print(f"{base:<15}", end="")
        for refl in refl_models:
            key = f"{base}_x_{refl}"
            value = matrix.get(key, {}).get(f'pass@{n_samples}', 0.0)
            print(f"{value:>6.2f}%    ", end="")
        print()

    avg_row = "Avg Base"
    print(f"{'-'*100}")
    print(f"{avg_row:<15}", end="")
    for refl in refl_models:
        values = [matrix.get(f"{b}_x_{refl}", {}).get(f'pass@{n_samples}', 0.0) for b in base_models]
        avg = np.mean(values) if values else 0.0
        print(f"{avg:>6.2f}%    ", end="")
    print()


def main():
    boxed_reward_fn = _load_boxed_reward_fn()

    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    random.seed(RANDOM_SEED)

    prompts, metadata = load_benchmark_data()
    raw_questions = [m['raw_question'] for m in metadata]

    print(f"\nTotal problems: {len(prompts)}")
    print(f"BASE models (generate base responses): {BASE_MODELS}")
    print(f"REFL models (generate reflections): {REFL_MODELS}")

    base_responses = {}
    base_scores = {}

    # Only generate base responses for BASE_MODELS
    for model_name in BASE_MODELS:
        model_path = MODEL_CONFIGS[model_name]
        responses = generate_base_responses(model_name, model_path, prompts, metadata)

        scores = []
        for idx, (response, m) in enumerate(zip(responses, metadata)):
            _, score = boxed_reward_fn(response, m['ground_truth'], fast=True)
            scores.append(score)

        base_responses[model_name] = responses
        base_scores[model_name] = scores

        base_metadata = [dict(m, base_score=s) for m, s in zip(metadata, scores)]
        base_results = compute_benchmark_metrics(base_metadata, 'base_score')
        print_results(f"BASE RESULTS: {model_name}", base_results)

    matrix = {}
    transition_matrix = {}

    # Cross-evaluation: REFL_MODELS reflect on BASE_MODELS' responses
    for refl_model in REFL_MODELS:
        refl_path = MODEL_CONFIGS[refl_model]
        for base_model in BASE_MODELS:
            print(f"\n>>> {base_model} (base) + {refl_model} (reflection)")

            responses = base_responses[base_model]
            refl_responses = generate_reflection_responses(
                refl_model, refl_path, raw_questions, responses, metadata
            )

            refl_metadata = []
            first_sample_scores = []

            for prob_idx, (samples, m) in enumerate(zip(refl_responses, metadata)):
                for sample_idx, sample_text in enumerate(samples):
                    _, score = boxed_reward_fn(sample_text, m['ground_truth'], fast=True)
                    sample_meta = dict(m, reflection_score=score)
                    sample_meta['sample_idx'] = sample_idx
                    sample_meta['problem_idx'] = prob_idx
                    refl_metadata.append(sample_meta)

                if samples:
                    _, first_score = boxed_reward_fn(samples[0], m['ground_truth'], fast=True)
                    first_sample_scores.append(first_score)

            key = f"{base_model}_x_{refl_model}"

            refl_results = compute_benchmark_metrics(refl_metadata, 'reflection_score', n_samples=N_REFLECTION_SAMPLES)

            macro_pass_at_n = sum(r[f'pass@{N_REFLECTION_SAMPLES}'] for r in refl_results.values()) / len(refl_results) if refl_results else 0.0
            macro_pass_at_1 = sum(r['pass@1'] for r in refl_results.values()) / len(refl_results) if refl_results else 0.0
            macro_avg = sum(r['avg@1'] for r in refl_results.values()) / len(refl_results) if refl_results else 0.0

            matrix[key] = {
                'base_model': base_model,
                'refl_model': refl_model,
                f'pass@{N_REFLECTION_SAMPLES}': macro_pass_at_n,
                'pass@1': macro_pass_at_1,
                'avg@1': macro_avg,
                'detailed': refl_results,
            }

            base_sc = base_scores[base_model]
            transition = compute_transition_metrics(base_sc, first_sample_scores)
            transition_matrix[key] = transition

            print(f"\nResults for {base_model} -> {refl_model}:")
            for bench, res in refl_results.items():
                print(f"  {bench}: pass@{N_REFLECTION_SAMPLES}={res[f'pass@{N_REFLECTION_SAMPLES}']:.2f}%, pass@1={res['pass@1']:.2f}%, avg@1={res['avg@1']:.2f}%")
            print(f"  Transition: {transition}")

    print_matrix(matrix, BASE_MODELS, REFL_MODELS, N_REFLECTION_SAMPLES)

    print(f"\n{'='*100}")
    print("TRANSITION METRICS (bad->good % improvement)")
    print(f"{'='*100}")
    header = "Base\\Refl"
    print(f"{header:<15}", end="")
    for refl in REFL_MODELS:
        print(f"{refl:<12}", end="")
    print()
    print(f"{'-'*100}")

    for base in BASE_MODELS:
        print(f"{base:<15}", end="")
        for refl in REFL_MODELS:
            key = f"{base}_x_{refl}"
            value = transition_matrix.get(key, {}).get('bad->good', 0.0)
            print(f"{value:>6.2f}%    ", end="")
        print()

    output_file = REPO_ROOT / "math_eval" / "reward" / "cross_reflection_results.json"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'matrix': matrix,
        'transition_matrix': transition_matrix,
        'base_scores': base_scores,
        'base_models': BASE_MODELS,
        'refl_models': REFL_MODELS,
    }

    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()
