from datasets import load_dataset
import pandas as pd
from pathlib import Path

ds = load_dataset("HuggingFaceH4/MATH-500", split="test")

data = []
for idx, item in enumerate(ds):
    problem_text = item['problem']
    answer = item['answer']

    prompt_text = f"{problem_text} Solve the following math problem step by step. Put your final answer inside \\boxed{{}}, like \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}."

    data.append({
        "problem": problem_text,
        "level": f"Level {item['level']}",
        "type": item['subject'],
        "solution": item['solution'],
        "data_source": "HuggingFaceH4/MATH-500",
        "prompt": [{"content": prompt_text, "role": "user"}],
        "ability": "math",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule"
        },
        "extra_info": {
            "index": idx,
            "level": f"Level {item['level']}",
            "problem": problem_text,
            "solution": item['solution'],
            "split": "test",
            "type": item['subject'],
            "unique_id": item['unique_id']
        }
    })

df = pd.DataFrame(data)
repo_root = Path(__file__).resolve().parents[2]
out_path = repo_root / "scratch" / "math500" / "test.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)
print(f"Created MATH500 dataset: {len(df)} samples")
print(df.iloc[0].to_dict())
