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

def main():
    model_path = sys.argv[1] if len(sys.argv) > 1 else "Qwen/Qwen3-30B-A3B-Instruct-2507"
    output_dir = (
        Path(sys.argv[2])
        if len(sys.argv) > 2
        else (REPO_ROOT / "math_eval" / "ood_traces" / "qwen3-30b-a3b")
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    llm = LLM(
        model=model_path,
        tensor_parallel_size=2,
        gpu_memory_utilization=0.9,
        dtype="bfloat16",
        trust_remote_code=True,
        seed=RANDOM_SEED
    )

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    np.random.seed(RANDOM_SEED)

    print(f"{'='*80}")
    print("BASE GENERATION")
    print(f"{'='*80}")

    all_prompts = []
    metadata = []

    for bench_name, bench_path in BENCHMARKS.items():
        df = pd.read_parquet(bench_path)

        sample_size = SAMPLE_SIZES[bench_name]
        if sample_size and len(df) > sample_size:
            df = df.sample(n=sample_size, random_state=RANDOM_SEED).reset_index(drop=True)

        print(f"Loading {bench_name}: {len(df)} problems")

        for idx, row in df.iterrows():
            prompt_messages = row['prompt']
            ground_truth = row['reward_model']['ground_truth']
            raw_question = prompt_messages[0]['content'] if len(prompt_messages) > 0 else ""

            formatted_prompt = tokenizer.apply_chat_template(
                prompt_messages,
                tokenize=False,
                add_generation_prompt=True
            )

            all_prompts.append(formatted_prompt)
            metadata.append({
                'benchmark': bench_name,
                'problem_idx': idx,
                'ground_truth': ground_truth,
                'raw_question': raw_question
            })

    print(f"\nTotal prompts to generate: {len(all_prompts)}")

    sampling_params = SamplingParams(
        temperature=1.0,
        max_tokens=3000,
        top_p=1.0,
    )

    print("Generating base responses...")
    outputs = llm.generate(all_prompts, sampling_params)

    print("Grading base responses...")
    for idx, output in enumerate(outputs):
        response = output.outputs[0].text
        ground_truth = metadata[idx]['ground_truth']
        grading_info, score = boxed_reward_fn(response, ground_truth, fast=True)
        metadata[idx]['score'] = score
        metadata[idx]['response'] = response

    results_by_benchmark = {}
    for bench_name in BENCHMARKS.keys():
        bench_metadata = [m for m in metadata if m['benchmark'] == bench_name]
        if not bench_metadata:
            continue

        scores = [m['score'] for m in bench_metadata]
        acc = np.mean(scores) * 100
        results_by_benchmark[bench_name] = {
            'n_problems': len(bench_metadata),
            'acc@1': acc
        }

    print(f"\n{'='*80}")
    print("BASE RESULTS")
    print(f"{'='*80}")
    print(f"{'Benchmark':<15} {'N':<6} {'acc@1':<12}")
    print(f"{'-'*80}")

    total_acc = 0
    for bench_name, result in results_by_benchmark.items():
        print(f"{bench_name:<15} {result['n_problems']:<6} {result['acc@1']:>6.2f}%")
        total_acc += result['acc@1']

    avg_acc = total_acc / len(results_by_benchmark)
    print(f"{'-'*80}")
    print(f"{'Average':<15} {'':6} {avg_acc:>6.2f}%")

    output_jsonl = output_dir / "base_answers.jsonl"
    with open(output_jsonl, 'w') as f:
        for m in metadata:
            f.write(json.dumps(m, ensure_ascii=False) + '\n')

    print(f"\nSaved base answers to: {output_jsonl}")

    output_parquet = output_dir / "base_answers.parquet"
    df_output = pd.DataFrame(metadata)
    df_output.to_parquet(output_parquet, index=False)

    print(f"Saved base answers to: {output_parquet}")

if __name__ == "__main__":
    main()
