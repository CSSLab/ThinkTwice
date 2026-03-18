import importlib.util
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import json

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_DIR = Path(os.environ.get("SCRATCH_DIR", REPO_ROOT / "scratch"))


def _load_boxed_reward_fn():
    grader_path = REPO_ROOT / "verl" / "verl" / "utils" / "reward_score" / "hendrycks_math_grader.py"
    spec = importlib.util.spec_from_file_location("hendrycks_math_grader", grader_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load grader module from {grader_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.boxed_reward_fn


boxed_reward_fn = _load_boxed_reward_fn()

# DEFAULT_REFLECTION_INSTRUCTION = (
#     "Follow this instruction, carefully review your previous solution:\n"
#     "1. Go through each calculation step-by-step. Check if there are any errors in calculations, logic, or problem understanding?\n"
#     "2. If you find any mistakes, explicitly point out what was wrong and explain the correct approach.\n"
#     "3. If the solution is already correct, verify each step and explain it more clearly.\n"
#     "4. Finally, after finishing the review, YOU MUST write 'Improved solution:' and provide your refined solution and answer.\n"
# )

DEFAULT_REFLECTION_INSTRUCTION = (
    "Review the solution above. Check it step by step for errors in reasoning, calculation, or problem understanding.\n"
    "If you find mistakes, explain what went wrong and provide a corrected solution.\n"
    "If the solution is already correct, refine the solution by explaining it more clearly.\n"
    "Finally, provide your final answer.\n"
)

BENCHMARKS = {
    "AIME24": str(SCRATCH_DIR / "aime2024" / "test.parquet"),
    "AMC": str(SCRATCH_DIR / "amc" / "test.parquet"),
    "MATH500": str(SCRATCH_DIR / "math500" / "test.parquet"),
    "Minerva": str(SCRATCH_DIR / "minerva_math" / "test.parquet"),
    "OlympiadBench": str(SCRATCH_DIR / "olympiadbench" / "test.parquet"),
}

SAMPLE_SIZES = {
    "AIME24": None,
    "AMC": None,
    "MATH500": None,
    "Minerva": None,
    "OlympiadBench": None,
}

RANDOM_SEED = 42

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

def main():
    if len(sys.argv) > 1:
        refiner_model_path = sys.argv[1]
    else:
        # refiner_model_path = "Qwen/Qwen3-4B-Instruct-2507"
        # refiner_model_path = "/data1/qianfeng/ckpts/grpo_math_baseline_Qwen3-4B-Instruct-2507/best_model"
        # refiner_model_path = "/data1/qianfeng/ckpts/drgrpo_math_baseline_Qwen3-4B-Instruct-2507/best_model"
        # refiner_model_path = "/data1/qianfeng/ckpts/dapo_math_Qwen3-4B-Instruct-2507-dapo-r8/best_model"
        refiner_model_path = "/data1/qianfeng/ckpts/grpo_math_refpo_dapo_Qwen3-4B-Instruct-2507/best_model"

    default_traces_dir = REPO_ROOT / "math_eval" / "ood_traces" / "gemma-3-27b-it"
    ood_base_answers_path = sys.argv[2] if len(sys.argv) > 2 else str(default_traces_dir / "base_answers.parquet")
    output_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else (default_traces_dir / "reflection_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Refiner model: {refiner_model_path}")
    print(f"OOD base answers: {ood_base_answers_path}")
    print(f"vLLM config: 2 GPUs, 0.9 GPU utilization")
    print(f"Random seed: {RANDOM_SEED}\n")

    llm = LLM(
        model=refiner_model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=RANDOM_SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(refiner_model_path, fix_mistral_regex=True)
    np.random.seed(RANDOM_SEED)

    print(f"{'='*80}")
    print("LOADING OOD BASE ANSWERS")
    print(f"{'='*80}")

    df_ood = pd.read_parquet(ood_base_answers_path)

    metadata = []
    for bench_name in BENCHMARKS.keys():
        bench_df = df_ood[df_ood['benchmark'] == bench_name].copy()

        sample_size = SAMPLE_SIZES[bench_name]
        if sample_size and len(bench_df) > sample_size:
            bench_df = bench_df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

        print(f"{bench_name}: {len(bench_df)} problems")

        for _, row in bench_df.iterrows():
            metadata.append({
                'benchmark': row['benchmark'],
                'problem_idx': row['problem_idx'],
                'ground_truth': row['ground_truth'],
                'raw_question': row['raw_question'],
                'base_response': row['response'],
                'base_score': row['score']
            })

    print(f"\nTotal problems for reflection: {len(metadata)}")

    base_results = {}
    for bench_name in BENCHMARKS.keys():
        bench_metadata = [m for m in metadata if m['benchmark'] == bench_name]
        if not bench_metadata:
            continue
        scores = [m['base_score'] for m in bench_metadata]
        acc = np.mean(scores) * 100
        base_results[bench_name] = {
            'n_problems': len(bench_metadata),
            'acc@1': acc
        }

    print(f"\n{'='*80}")
    print("BASE RESULTS (OOD)")
    print(f"{'='*80}")
    print(f"{'Benchmark':<15} {'N':<6} {'acc@1':<12}")
    print(f"{'-'*80}")

    total_acc = 0
    for bench_name, result in base_results.items():
        print(f"{bench_name:<15} {result['n_problems']:<6} {result['acc@1']:>6.2f}%")
        total_acc += result['acc@1']

    avg_acc = total_acc / len(base_results)
    print(f"{'-'*80}")
    print(f"{'Average':<15} {'':6} {avg_acc:>6.2f}%")

    print(f"\n{'='*80}")
    print("REFLECTION GENERATION")
    print(f"{'='*80}")

    raw_questions = [m['raw_question'] for m in metadata]
    base_responses = [m['base_response'] for m in metadata]
    reflection_prompts = construct_reflection_prompts(tokenizer, raw_questions, base_responses)

    print(f"Total reflection prompts: {len(reflection_prompts)}")

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=3000,
        top_p=1.0,
    )

    print("Generating reflection responses...")
    reflection_outputs = llm.generate(reflection_prompts, sampling_params)

    print("Grading reflection responses...")
    for idx, output in enumerate(reflection_outputs):
        response = output.outputs[0].text
        ground_truth = metadata[idx]['ground_truth']
        grading_info, score = boxed_reward_fn(response, ground_truth, fast=True)
        metadata[idx]['reflection_response'] = response
        metadata[idx]['reflection_score'] = score

    reflection_results = {}
    for bench_name in BENCHMARKS.keys():
        bench_metadata = [m for m in metadata if m['benchmark'] == bench_name]
        if not bench_metadata:
            continue
        scores = [m['reflection_score'] for m in bench_metadata]
        acc = np.mean(scores) * 100
        reflection_results[bench_name] = {
            'n_problems': len(bench_metadata),
            'acc@1': acc
        }

    print(f"\n{'='*80}")
    print("REFLECTION RESULTS")
    print(f"{'='*80}")
    print(f"{'Benchmark':<15} {'N':<6} {'acc@1':<12}")
    print(f"{'-'*80}")

    total_acc = 0
    for bench_name, result in reflection_results.items():
        print(f"{bench_name:<15} {result['n_problems']:<6} {result['acc@1']:>6.2f}%")
        total_acc += result['acc@1']

    avg_acc = total_acc / len(reflection_results)
    print(f"{'-'*80}")
    print(f"{'Average':<15} {'':6} {avg_acc:>6.2f}%")

    print(f"\n{'='*80}")
    print("TRANSITION METRICS")
    print(f"{'='*80}")

    transition_results = {}
    for bench_name in BENCHMARKS.keys():
        bench_metadata = [m for m in metadata if m['benchmark'] == bench_name]
        if not bench_metadata:
            continue

        base_scores = [m['base_score'] for m in bench_metadata]
        refl_scores = [m['reflection_score'] for m in bench_metadata]

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

    output_jsonl = output_dir / "reflection_results.jsonl"
    with open(output_jsonl, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')

    print(f"\nSaved reflection results to: {output_jsonl}")

    output_parquet = output_dir / "reflection_results.parquet"
    df_output = pd.DataFrame(metadata)
    df_output.to_parquet(output_parquet, index=False)

    print(f"Saved reflection results to: {output_parquet}")

if __name__ == "__main__":
    main()
