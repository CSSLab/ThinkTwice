from datasets import load_dataset
import pandas as pd
from pathlib import Path

ds = load_dataset("afraamn/olympiadbench_math_textonly", split="test_en")

data = []
for idx, item in enumerate(ds):
    problem_text = item['question']
    if item['context']:
        problem_text = f"{item['context']}\n\n{problem_text}"

    answers = item['final_answer']
    if isinstance(answers, list) and len(answers) > 0:
        answer = answers[0]
    else:
        answer = str(answers)

    answer = answer.strip()
    if answer.startswith('$') and answer.endswith('$'):
        answer = answer[1:-1]

    prompt_text = f"{problem_text} Solve the following math problem step by step. Put your final answer inside \\boxed{{}}, like \\boxed{{42}} or \\boxed{{\\frac{{1}}{{2}}}}."

    data.append({
        "problem": problem_text,
        "level": "Olympiad",
        "type": item['subfield'],
        "solution": "",
        "data_source": "afraamn/olympiadbench_math_textonly",
        "prompt": [{"content": prompt_text, "role": "user"}],
        "ability": "math",
        "reward_model": {
            "ground_truth": answer,
            "style": "rule"
        },
        "extra_info": {
            "index": idx,
            "level": "Olympiad",
            "problem": problem_text,
            "solution": "",
            "split": "test",
            "type": item['subfield'],
            "question_id": item['question_id'],
            "answer_type": item['answer_type']
        }
    })

df = pd.DataFrame(data)
repo_root = Path(__file__).resolve().parents[2]
out_path = repo_root / "scratch" / "olympiadbench" / "test.parquet"
out_path.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(out_path, index=False)
print(f"Created OlympiadBench dataset: {len(df)} samples")
print(f"First answer: {df.iloc[0]['reward_model']['ground_truth']}")
print(f"Last answer: {df.iloc[-1]['reward_model']['ground_truth']}")
