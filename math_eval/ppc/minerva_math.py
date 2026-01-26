from datasets import load_dataset
import pandas as pd
from pathlib import Path

ds = load_dataset("math-ai/minervamath", split="test")

data = []
for idx, item in enumerate(ds):
    problem_text = item['question']
    answer = item['answer']

    prompt_text = f"{problem_text} Solve the following math problem step by step. Put your final answer inside \\boxed{{}}, like \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}."

    data.append({
        "problem": problem_text,
        "level": "Minerva",
        "type": "Math",
        "solution": "",
        "data_source": "math-ai/minervamath",
        "prompt": [{"content": prompt_text, "role": "user"}],
        "ability": "math",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule"
        },
        "extra_info": {
            "index": idx,
            "level": "Minerva",
            "problem": problem_text,
            "solution": "",
            "split": "test",
            "type": "Math"
        }
    })

df = pd.DataFrame(data)
repo_root = Path(__file__).resolve().parents[2]
out_path = repo_root / "scratch" / "minerva_math" / "test.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)
print(f"Created Minerva Math dataset: {len(df)} samples")
print(f"First answer: {df.iloc[0]['reward_model']['ground_truth']}")
print(f"Last answer: {df.iloc[-1]['reward_model']['ground_truth']}")
