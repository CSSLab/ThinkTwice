from datasets import load_dataset
import pandas as pd
import re
from pathlib import Path

ds = load_dataset("AI-MO/aimo-validation-aime", split="train")

def extract_answer(solution):
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    return match.group(1) if match else None

data = []
for idx, item in enumerate(ds):
    problem_text = item['problem']
    answer = item['answer']

    prompt_text = f"{problem_text} Solve the following math problem step by step. Put your final answer inside \\boxed{{}}, like \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}."

    data.append({
        "problem": problem_text,
        "level": "AIME",
        "type": "Competition",
        "solution": item['solution'],
        "data_source": "AI-MO/aimo-validation-aime",
        "prompt": [{"content": prompt_text, "role": "user"}],
        "ability": "math",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule"
        },
        "extra_info": {
            "index": idx,
            "level": "AIME",
            "problem": problem_text,
            "solution": item['solution'],
            "split": "test",
            "type": "Competition",
            "url": item['url']
        }
    })

df = pd.DataFrame(data)
repo_root = Path(__file__).resolve().parents[2]
out_path = repo_root / "scratch" / "aime2024" / "test.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)
print(f"Created AIME 2024 dataset: {len(df)} samples")
print(df.iloc[0].to_dict())
