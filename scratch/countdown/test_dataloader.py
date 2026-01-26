import sys
sys.path.insert(0, '/datadrive/difan/self-reflection/verl')

from transformers import AutoTokenizer
from omegaconf import OmegaConf
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from torch.utils.data import DataLoader

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B-Instruct-2507")

config = OmegaConf.create({
    "cache_dir": "~/.cache/verl/rlhf",
    "prompt_key": "prompt",
    "max_prompt_length": 1500,
    "return_raw_chat": False,
    "return_full_prompt": False,
    "truncation": "error",
    "filter_overlong_prompts": False,
    "apply_chat_template_kwargs": {},
    "use_shm": False,
    "filter_prompts": True,
})

dataset = RLHFDataset(
    data_files="/datadrive/difan/self-reflection/scratch/countdown/train_final.parquet",
    tokenizer=tokenizer,
    config=config,
)

print(f"Dataset size: {len(dataset)}")

sample = dataset[0]
print(f"\nSample keys: {sample.keys()}")
for key, val in sample.items():
    if key not in ['input_ids', 'attention_mask', 'position_ids']:
        print(f"  {key}: {type(val)} = {val}")

dataloader = DataLoader(dataset, batch_size=8, collate_fn=collate_fn, num_workers=0)

batch = next(iter(dataloader))
print(f"\nBatch keys: {batch.keys()}")
print(f"Batch size (from input_ids): {batch['input_ids'].shape[0]}")

for key, val in batch.items():
    if key not in ['input_ids', 'attention_mask', 'position_ids', 'raw_prompt_ids']:
        print(f"  {key}: type={type(val)}, shape={val.shape if hasattr(val, 'shape') else 'N/A'}")
        if hasattr(val, 'shape') and len(val.shape) > 0:
            print(f"    first item: {val[0]}")
