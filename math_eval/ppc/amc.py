from datasets import load_dataset
import pandas as pd
from pathlib import Path

ds = load_dataset("AI-MO/aimo-validation-amc", split="train")

data = []
for idx, item in enumerate(ds):
    problem_text = item['problem']
    answer = str(int(item['answer'])) if isinstance(item['answer'], float) else str(item['answer'])

    prompt_text = f"{problem_text} Solve the following math problem step by step. Put your final answer inside \\boxed{{}}, like \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}."

    data.append({
        "problem": problem_text,
        "level": "AMC",
        "type": "Competition",
        "solution": "",
        "data_source": "AI-MO/aimo-validation-amc",
        "prompt": [{"content": prompt_text, "role": "user"}],
        "ability": "math",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule"
        },
        "extra_info": {
            "index": idx,
            "level": "AMC",
            "problem": problem_text,
            "solution": "",
            "split": "test",
            "type": "Competition",
            "url": item['url']
        }
    })

df = pd.DataFrame(data)
repo_root = Path(__file__).resolve().parents[2]
out_path = repo_root / "scratch" / "amc" / "test.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)
print(f"Created AMC dataset: {len(df)} samples")
print(df.iloc[0].to_dict())
