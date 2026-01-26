import os
import datasets

data_source = "countdown"
dataset = datasets.load_dataset("parquet", data_files={
    "train": "/datadrive/difan/self-reflection/scratch/countdown/train.parquet",
    "test": "/datadrive/difan/self-reflection/scratch/countdown/test.parquet"
})

train_dataset = dataset["train"]
test_dataset = dataset["test"]

def make_map_fn(split):
    def process_fn(example, idx):
        target = example["target"]
        nums = example["nums"]
        prompt = example["prompt"]

        data = {
            "data_source": "countdown",
            "prompt": prompt,
            "ability": "math",
            "reward_model": {
                "style": "rule",
                "ground_truth": {
                    "numbers": nums if isinstance(nums, list) else nums.tolist(),
                    "target": int(target)
                }
            },
            "extra_info": {
                "split": split,
                "index": idx,
            },
        }
        return data
    return process_fn

train_dataset = train_dataset.map(function=make_map_fn("train"), with_indices=True)
test_dataset = test_dataset.map(function=make_map_fn("test"), with_indices=True)

train_dataset = train_dataset.remove_columns(["target", "nums"])
test_dataset = test_dataset.remove_columns(["target", "nums"])

local_dir = "/datadrive/difan/self-reflection/scratch/countdown"
train_dataset.to_parquet(os.path.join(local_dir, "train.parquet"))
test_dataset.to_parquet(os.path.join(local_dir, "test.parquet"))

print(f"Saved to {local_dir}")
print(f"Train size: {len(train_dataset)}, Test size: {len(test_dataset)}")
print(f"\nColumns: {train_dataset.column_names}")
print(f"Sample: {train_dataset[0]}")
