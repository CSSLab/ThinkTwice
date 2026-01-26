from __future__ import annotations

import os
import shutil
from pathlib import Path


def convert_fsdp_checkpoint(
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    *,
    trust_remote_code: bool = True,
) -> None:
    """Convert a VERL FSDP checkpoint folder into a HuggingFace model directory.

    NOTE: VERL's FSDP checkpoints can contain DTensor parameters with different placements
    (e.g., Shard vs Replicate). A naive `torch.cat(..., dim=0)` merge can silently corrupt
    weights (common symptom: vocab embedding has 2x vocab size), and vLLM will fail to load.

    This converter delegates to `verl.model_merger`'s FSDP merger which understands DTensor
    placements and produces a standard HF `save_pretrained` output compatible with vLLM.
    """

    checkpoint_path = Path(checkpoint_dir)
    output_path = Path(output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint dir does not exist: {checkpoint_path}")

    shutil.rmtree(output_path, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)

    hf_cfg_path = checkpoint_path / "huggingface"
    if not hf_cfg_path.exists():
        raise FileNotFoundError(
            f"Missing HuggingFace config dir at {hf_cfg_path}. Expected {checkpoint_path}/huggingface from VERL."
        )

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    cfg = ModelMergerConfig(
        operation="merge",
        backend="fsdp",
        target_dir=str(output_path),
        hf_upload_path=None,
        private=False,
        test_hf_dir=None,
        tie_word_embedding=False,
        trust_remote_code=trust_remote_code,
        is_value_model=False,
        local_dir=str(checkpoint_path),
        hf_model_config_path=str(hf_cfg_path),
        use_cpu_initialization=True,
    )

    merger = FSDPModelMerger(cfg)
    merger.merge_and_save()
    merger.cleanup()


if __name__ == "__main__":  # pragma: no cover
    import argparse

    parser = argparse.ArgumentParser(description="Convert VERL FSDP checkpoint shards to HF-style weights.")
    parser.add_argument(
        "checkpoint_dir", type=str, help="Path to FSDP checkpoint directory (contains model_world_size_* files)."
    )
    parser.add_argument("output_dir", type=str, help="Output directory for consolidated HF checkpoint.")
    args = parser.parse_args()

    convert_fsdp_checkpoint(args.checkpoint_dir, args.output_dir)
