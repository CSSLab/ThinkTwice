# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass, field
from pprint import pprint
from typing import Any, Optional

import numpy as np
import ray
import torch
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.experimental.dataset.sampler import AbstractCurriculumSampler
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import AdvantageEstimator, agg_loss
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reflection_utils import select_reflection_indices
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.trainer.ppo.utils import Role, WorkerType, need_critic, need_reference_policy, need_reward_model
from verl.utils import tensordict_utils as tu
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path, should_save_ckpt_esi
from verl.utils.config import omega_conf_to_dataclass
from verl.utils.debug import marked_timer
from verl.utils.import_utils import load_class_from_fqn
from verl.utils.metric import reduce_metrics
from verl.utils.py_functional import rename_dict
from verl.utils.rollout_skip import RolloutSkip
from verl.utils.seqlen_balancing import calculate_workload, get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.config import FSDPEngineConfig
from verl.workers.utils.padding import left_right_2_no_padding, no_padding_2_padding

# ============================================================================
# REFLECTION PROMPT OPTIONS - Manually uncomment ONE to test
# ============================================================================

# OPTION 1: Original detailed prompt (for non-thinking models)
DEFAULT_REFLECTION_INSTRUCTION = (
    "Follow this instruction, carefully review your previous solution:\n"
    "1. Go through each calculation step-by-step. Check if there are any errors in calculations, logic, or problem understanding.\n"
    "2. If you find any mistakes, explicitly point out what was wrong and explain the correct approach.\n"
    "3. If the solution is already correct, verify each step and explain it more clearly.\n"
    "4. Finally, after finishing the review, provide your refined solution and answer.\n"
)

# # OPTION 2: Verification-focused (for thinking models) - emphasize independent verification
# DEFAULT_REFLECTION_INSTRUCTION = (
#     "You previously solved this math problem. Now independently verify your answer:\n"
#     "1. Re-read the problem and confirm you understood it correctly.\n"
#     "2. Check your final answer by solving the problem again from scratch.\n"
#     "3. If your new answer differs from your previous one, determine which is correct.\n"
#     "4. Provide your refined solution and answer.\n"
# )

# # OPTION 3: Error-hunting focused - focus on finding specific mistakes
# DEFAULT_REFLECTION_INSTRUCTION = (
#     "Review your previous solution and hunt for errors:\n"
#     "1. Did you misunderstand any part of the problem?\n"
#     "2. Are there any calculation mistakes? Check each step carefully.\n"
#     "3. Did you use the wrong formula or approach?\n"
#     "4. If you find an error, correct it and provide the refined answer. If no errors, confirm your answer.\n"
#     "Final answer format: \\boxed{YOUR_ANSWER}\n"
# )

# # OPTION 4: Concise/direct - short and to the point
# DEFAULT_REFLECTION_INSTRUCTION = (
#     "Check your previous answer. If you believe it's wrong, explicitly point out what was wrong and explain the correct approach. "
#     "If you believe it's correct, simply confirm it. Put your final answer in \\boxed{}.\n"
# )

# # OPTION 5: Structured fresh-look - step-by-step with fresh perspective
# DEFAULT_REFLECTION_INSTRUCTION = (
#     "Take a fresh look at this problem and your previous answer:\n"
#     "- Work through the problem methodically in your thinking.\n"
#     "- Identify any mistakes in your previous approach.\n"
#     "- Provide your refined solution, ending with \\boxed{FINAL_ANSWER}.\n"
# )

# OPTION 6
DEFAULT_REFLECTION_INSTRUCTION = (
    "Review your previous solution, including your thinking process:\n"
    "1. Examine your reasoning step-by-step in your thinking. Are there logical gaps, errors, or unclear steps?\n"
    "2. Check your calculations and approach - did you use the right formula or method?\n"
    "3. If you find mistakes in your reasoning or approach, explain what was wrong and provide the correct reasoning.\n"
    "4. If your solution is already correct, verify each step and confirm your approach.\n"
    "5. Finally, after finishing the review, provide your refined solution and answer.\n"
)

VAL_SAMPLE_SIZES: dict[str, int | None] = {
    "AIME24": None,
    "AMC": None,
    "MATH500": None,
    "Minerva": None,
    "OlympiadBench": None,
}

VAL_REFLECTION_SAMPLE_SIZES: dict[str, int] = {
    "AIME24": None,
    "AMC": None,
    "MATH500": None,
    "Minerva": None,
    "OlympiadBench": None,
}

VAL_SAMPLE_RANDOM_SEED = 42
VAL_REFLECTION_RANDOM_SEED = 42

VAL_GENERATION_LOG_SIZE = 128
VAL_GENERATION_LOG_SEED = 42

MATH500_DIFFICULTY_LEVELS: dict[int, int] = {}


def _json_default(obj: Any) -> Any:
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """

    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        """Create Ray resource pools for distributed training.

        Initializes resource pools based on the resource pool specification,
        with each pool managing GPU resources across multiple nodes.
        For FSDP backend, uses max_colocate_count=1 to merge WorkerGroups.
        For Megatron backend, uses max_colocate_count>1 for different models.
        """
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(
                process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name
            )
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray._private.state.available_resources_per_node()
        node_available_gpus = {
            node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0)
            for node, node_info in node_available_resources.items()
        }

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes]
        )
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}"
            )


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl"):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    response_mask = data.batch["response_mask"]
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]

    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(
        data.batch["old_log_probs"], data.batch["ref_log_prob"], kl_penalty=kl_penalty
    )  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards

    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}

    return data, metrics


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(
    data: DataProto,
    adv_estimator: AdvantageEstimator,
    gamma: float = 1.0,
    lam: float = 1.0,
    num_repeat: int = 1,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> DataProto:
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator (AdvantageEstimator): The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in
            GRPO. Defaults to True.
        config (dict, optional): Configuration dictionary for algorithm settings. Defaults to None.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch.keys():
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    if adv_estimator == AdvantageEstimator.GAE:
        # Compute advantages and returns using Generalized Advantage Estimation (GAE)
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if config.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                config.pf_ppo.get("reweight_method"),
                config.pf_ppo.get("weight_pow"),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # Initialize the mask for GRPO calculation
        grpo_calculation_mask = data.batch["response_mask"]

        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    else:
        # handle all other adv estimator type other than GAE and GRPO
        adv_estimator_fn = core_algos.get_adv_estimator_fn(adv_estimator)
        adv_kwargs = {
            "token_level_rewards": data.batch["token_level_rewards"],
            "response_mask": data.batch["response_mask"],
            "config": config,
        }
        if "uid" in data.non_tensor_batch:  # optional
            adv_kwargs["index"] = data.non_tensor_batch["uid"]
        if "reward_baselines" in data.batch:  # optional
            adv_kwargs["reward_baselines"] = data.batch["reward_baselines"]

        # calculate advantage estimator
        advantages, returns = adv_estimator_fn(**adv_kwargs)
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    return data


class RayPPOTrainer:
    """Distributed PPO trainer using Ray for scalable reinforcement learning.

    This trainer orchestrates distributed PPO training across multiple nodes and GPUs,
    managing actor rollouts, critic training, and reward computation with Ray backend.
    Supports various model architectures including FSDP, Megatron, vLLM, and SGLang integration.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: type[RayWorkerGroup] = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name=None,
    ):
        """
        Initialize distributed PPO trainer with Ray backend.
        Note that this trainer runs on the driver process on a single CPU/GPU node.

        Args:
            config: Configuration object containing training parameters.
            tokenizer: Tokenizer used for encoding and decoding text.
            role_worker_mapping (dict[Role, WorkerType]): Mapping from roles to worker classes.
            resource_pool_manager (ResourcePoolManager): Manager for Ray resource pools.
            ray_worker_group_cls (RayWorkerGroup, optional): Class for Ray worker groups. Defaults to RayWorkerGroup.
            processor: Optional data processor, used for multimodal data
            reward_fn: Function for computing rewards during training.
            val_reward_fn: Function for computing rewards during validation.
            train_dataset (Optional[Dataset], optional): Training dataset. Defaults to None.
            val_dataset (Optional[Dataset], optional): Validation dataset. Defaults to None.
            collate_fn: Function to collate data samples into batches.
            train_sampler (Optional[Sampler], optional): Sampler for the training dataset. Defaults to None.
            device_name (str, optional): Device name for training (e.g., "cuda", "cpu"). Defaults to None.
        """

        # Store the tokenizer for text processing
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping or Role.ActorRolloutRef in role_worker_mapping, (
                f"{role_worker_mapping.keys()=}"
            )

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = need_reference_policy(self.role_worker_mapping)
        # legacy reward model implementation
        self.use_rm = need_reward_model(self.role_worker_mapping)
        self.use_reward_loop = self.config.reward_model.use_reward_loop

        self.use_critic = need_critic(self.config)
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name if device_name else self.config.trainer.device
        self.validation_generations_logger = ValidationGenerationsLogger(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
        )

        reflection_cfg = self.config.algorithm.get("reflection", None)
        if reflection_cfg is not None and isinstance(reflection_cfg, DictConfig):
            reflection_cfg = OmegaConf.to_container(reflection_cfg, resolve=True)
        if reflection_cfg is None:
            reflection_cfg = {}
        self._reflection_cfg_present = bool(reflection_cfg)

        self.reflection_steps = int(reflection_cfg.get("steps", 0) or 0)
        # Default: always run reflection in validation (can be disabled explicitly).
        self.reflection_validate = bool(reflection_cfg["validate"]) if "validate" in reflection_cfg else True
        reflection_instruction = reflection_cfg.get("instruction", None) or self.config.algorithm.get(
            "reflection_instruction", None
        )
        self.reflection_instruction = reflection_instruction or DEFAULT_REFLECTION_INSTRUCTION

        use_reflection = self.config.algorithm.get("use_reflection", None)
        if reflection_cfg:
            # Compatibility with the teammate workflow: reflection is enabled via `algorithm.reflection.steps`.
            use_reflection = self.reflection_steps > 0
        elif use_reflection is None:
            use_reflection = False
        self.use_reflection = bool(use_reflection)

        self._val_sampled_indices: Optional[dict[str, list[int]]] = None
        self._val_reflection_sampled_indices: Optional[dict[str, list[int]]] = None
        self._val_generation_log_indices: Optional[list[int]] = None

        self.ref_in_actor = (
            config.actor_rollout_ref.model.get("lora_rank", 0) > 0
            or config.actor_rollout_ref.model.get("lora_adapter_path") is not None
        )

        # define in-reward KL control
        # kl loss control currently not suppoorted
        if self.config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(self.config.algorithm.kl_ctrl)

        self.use_prefix_grouper = self.config.actor_rollout_ref.actor.get("use_prefix_grouper", False)
        self.use_legacy_worker_impl = config.trainer.get("use_legacy_worker_impl", "auto")

        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        filter_groups_cfg = self.config.algorithm.get("filter_groups", None)
        self.dapo_enabled = filter_groups_cfg is not None and filter_groups_cfg.get("enable", False)
        self.dapo_metric = filter_groups_cfg.get("metric", "acc") if self.dapo_enabled else "acc"
        self.dapo_max_gen_batches = filter_groups_cfg.get("max_num_gen_batches", 0) if self.dapo_enabled else 0

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler: Optional[Sampler]):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_ppo import create_rl_dataset, create_rl_sampler

        if train_dataset is None:
            train_dataset = create_rl_dataset(
                self.config.data.train_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("train_max_samples", -1),
            )
        if val_dataset is None:
            val_dataset = create_rl_dataset(
                self.config.data.val_files,
                self.config.data,
                self.tokenizer,
                self.processor,
                max_samples=self.config.data.get("val_max_samples", -1),
            )
        self.train_dataset, self.val_dataset = train_dataset, val_dataset

        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn

            collate_fn = default_collate_fn

        num_workers = self.config.data["dataloader_num_workers"]

        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=num_workers,
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )

        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)

        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=num_workers,
            shuffle=self.config.data.get("validation_shuffle", True),
            drop_last=False,
            collate_fn=collate_fn,
        )

        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"

        print(
            f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: "
            f"{len(self.val_dataloader)}"
        )

        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")

        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")

    def _dump_generations(self, inputs, outputs, gts, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")

        n = len(inputs)
        base_data = {
            "input": inputs,
            "output": outputs,
            "gts": gts,
            "score": scores,
            "step": [self.global_steps] * n,
        }

        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v

        lines = []
        for i in range(n):
            entry = {k: v[i] for k, v in base_data.items()}
            lines.append(json.dumps(entry, ensure_ascii=False, default=_json_default))

        with open(filename, "w") as f:
            f.write("\n".join(lines) + "\n")

        print(f"Dumped generations to {filename}")

    def _log_rollout_data(
        self, batch: DataProto, reward_extra_infos_dict: dict, timing_raw: dict, rollout_data_dir: str
    ):
        """Log rollout data to disk.
        Args:
            batch (DataProto): The batch containing rollout data
            reward_extra_infos_dict (dict): Additional reward information to log
            timing_raw (dict): Timing information for profiling
            rollout_data_dir (str): Directory path to save the rollout data
        """
        with marked_timer("dump_rollout_generations", timing_raw, color="green"):
            inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
            outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
            scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
            sample_gts = [item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in batch]

            reward_extra_infos_to_dump = reward_extra_infos_dict.copy()
            if "request_id" in batch.non_tensor_batch:
                reward_extra_infos_dict.setdefault(
                    "request_id",
                    batch.non_tensor_batch["request_id"].tolist(),
                )

            self._dump_generations(
                inputs=inputs,
                outputs=outputs,
                gts=sample_gts,
                scores=scores,
                reward_extra_infos_dict=reward_extra_infos_to_dump,
                dump_path=rollout_data_dir,
            )

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.log_val_generations

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores, strict=True))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _compute_or_extract_reward(
        self,
        batch: DataProto,
        reward_fn=None,
        return_dict: bool = False,
        sum_reward: bool = False,
    ) -> tuple[torch.Tensor, dict[str, Any]] | torch.Tensor | dict[str, Any]:
        """
        Compute or extract reward from batch.

        When use_reward_loop=True, rewards are already computed during generate_sequences
        and stored in rm_scores. This method directly extracts them instead of calling
        reward functions which would only perform format conversion.

        Args:
            batch: DataProto containing the batch data
            reward_fn: Reward function to use if rm_scores doesn't exist (for training/validation)
            return_dict: Whether to return dict format with reward_extra_info (for validation)
            sum_reward: Whether to sum reward tensor along last dimension (for REMAX baseline)

        Returns:
            If return_dict=True: dict with "reward_tensor" and "reward_extra_info"
            If return_dict=False and sum_reward=True: summed reward_tensor (1D tensor)
            If return_dict=False and sum_reward=False: reward_tensor (2D tensor)
        """
        # When rm_scores already exists, extract it directly (format conversion only)
        if "rm_scores" in batch.batch.keys():
            reward_tensor = batch.batch["rm_scores"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)

            if return_dict:
                # Extract reward_extra_info if available
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_info = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
            else:
                # If sum_reward=True, only return tensor (for REMAX baseline)
                if sum_reward:
                    return reward_tensor
                # Otherwise, return tuple with reward_extra_info (for training loop)
                reward_extra_keys = batch.meta_info.get("reward_extra_keys", [])
                reward_extra_infos_dict = (
                    {key: batch.non_tensor_batch[key] for key in reward_extra_keys} if reward_extra_keys else {}
                )
                return reward_tensor, reward_extra_infos_dict

        # Otherwise, compute reward using reward_fn
        if reward_fn is None:
            raise ValueError("reward_fn must be provided when rm_scores is not available.")

        if return_dict:
            result = reward_fn(batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            reward_extra_info = result.get("reward_extra_info", {})
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            reward_tensor, reward_extra_infos_dict = compute_reward(batch, reward_fn)
            if sum_reward:
                reward_tensor = reward_tensor.sum(dim=-1)
            return reward_tensor, reward_extra_infos_dict

    def _get_gen_batch(self, batch: DataProto) -> DataProto:
        reward_model_keys = set({"data_source", "reward_model", "extra_info", "uid"}) & batch.non_tensor_batch.keys()

        # pop those keys for generation
        batch_keys_to_pop = []
        non_tensor_batch_keys_to_pop = set(batch.non_tensor_batch.keys()) - reward_model_keys
        gen_batch = batch.pop(
            batch_keys=batch_keys_to_pop,
            non_tensor_batch_keys=list(non_tensor_batch_keys_to_pop),
        )

        # For agent loop, we need reward model keys to compute score.
        if self.async_rollout_mode:
            gen_batch.non_tensor_batch.update(batch.non_tensor_batch)

        return gen_batch

    def _extract_last_user_message(self, raw_prompt: Any) -> str:
        if isinstance(raw_prompt, np.ndarray):
            raw_prompt = raw_prompt.tolist()
        if isinstance(raw_prompt, str):
            return raw_prompt
        if not isinstance(raw_prompt, list):
            return ""

        for message in reversed(raw_prompt):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, list):
                parts = []
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item.get("text", ""))
                    elif isinstance(item, str):
                        parts.append(item)
                return "".join(parts)
            if isinstance(content, str):
                return content
            return str(content)
        return ""

    def _mark_extra_info_reflection(self, non_tensor_batch: dict[str, Any], batch_size: int) -> None:
        """Ensure `extra_info.is_reflection=True` exists for reward functions/metrics."""
        extra_info = non_tensor_batch.get("extra_info")
        if extra_info is None:
            non_tensor_batch["extra_info"] = np.array([{"is_reflection": True}] * batch_size, dtype=object)
            return

        if isinstance(extra_info, np.ndarray):
            extra_items = extra_info.tolist()
        elif isinstance(extra_info, list):
            extra_items = extra_info
        elif isinstance(extra_info, dict):
            extra_items = [extra_info]
        else:
            extra_items = [extra_info]

        if len(extra_items) != batch_size:
            if len(extra_items) == 1:
                extra_items = extra_items * batch_size
            else:
                extra_items = (extra_items + [{}] * batch_size)[:batch_size]

        updated = []
        for item in extra_items:
            if isinstance(item, dict):
                new_item = dict(item)
            else:
                new_item = {"value": item}
            new_item["is_reflection"] = True
            updated.append(new_item)
        non_tensor_batch["extra_info"] = np.array(updated, dtype=object)

    def _build_reflection_gen_batch(
        self, gen_batch_output: DataProto, base_non_tensor_batch: Optional[dict[str, Any]] = None
    ) -> Optional[DataProto]:
        raw_prompts = gen_batch_output.non_tensor_batch.get("raw_prompt")
        if raw_prompts is None:
            raw_prompts = gen_batch_output.non_tensor_batch.get("prompt")
        if raw_prompts is None and base_non_tensor_batch is not None:
            raw_prompts = base_non_tensor_batch.get("raw_prompt")
        if raw_prompts is None and base_non_tensor_batch is not None:
            raw_prompts = base_non_tensor_batch.get("prompt")
        if raw_prompts is None or gen_batch_output.batch is None or "responses" not in gen_batch_output.batch:
            return None

        responses = gen_batch_output.batch["responses"]
        response_mask = gen_batch_output.batch["response_mask"] if "response_mask" in gen_batch_output.batch else None
        if response_mask is None and "attention_mask" in gen_batch_output.batch:
            response_mask = compute_response_mask(gen_batch_output)
        reflection_prompts = []

        # Get the thinking markers for Qwen3 (tokens 151667 and 15168)
        thinking_start_marker = self.tokenizer.decode([151667], skip_special_tokens=False)
        thinking_end_marker = self.tokenizer.decode([151668], skip_special_tokens=False)

        for i in range(len(raw_prompts)):
            question = self._extract_last_user_message(raw_prompts[i])
            response_ids = responses[i]
            if response_mask is not None:
                response_ids = response_ids[response_mask[i].bool()]

            # Decode WITHOUT skipping special tokens to preserve thinking markers
            full_response = self.tokenizer.decode(response_ids.tolist(), skip_special_tokens=False)

            # Extract thinking content and final answer
            thinking_content = None
            final_answer = full_response

            if thinking_start_marker in full_response:
                parts = full_response.split(thinking_start_marker, 1)
                if len(parts) > 1:
                    middle_part = parts[1]
                    if thinking_end_marker in middle_part:
                        thinking_part, answer_part = middle_part.split(thinking_end_marker, 1)
                        thinking_content = thinking_part.strip()
                        final_answer = answer_part.strip()
                    else:
                        thinking_content = middle_part.strip()
                        final_answer = ""
            else:
                final_answer = full_response.strip()

            # Build the user's reflection instruction with previous thinking
            reflection_user_content = self.reflection_instruction
            if thinking_content:
                reflection_user_content = (
                    f"Your previous reasoning:\n{thinking_content}\n\n"
                    f"Your previous answer:\n{final_answer}\n\n"
                    f"{self.reflection_instruction}"
                )
            elif final_answer and final_answer != full_response.strip():
                reflection_user_content = (
                    f"Your previous answer:\n{final_answer}\n\n"
                    f"{self.reflection_instruction}"
                )

            messages = [
                {"role": "user", "content": question},
                {"role": "assistant", "content": final_answer},
                {"role": "user", "content": reflection_user_content},
            ]
            reflection_prompts.append(messages)

        if base_non_tensor_batch is not None:
            reflection_non_tensor_batch = dict(base_non_tensor_batch)
        else:
            reflection_non_tensor_batch = dict(gen_batch_output.non_tensor_batch)
        reflection_non_tensor_batch["raw_prompt"] = np.array(reflection_prompts, dtype=object)
        self._mark_extra_info_reflection(reflection_non_tensor_batch, batch_size=len(reflection_prompts))
        return DataProto(batch=None, non_tensor_batch=reflection_non_tensor_batch)

    def _apply_dapo_filter(self, batch: DataProto, metric_name: str = "acc"):
        uids = batch.non_tensor_batch["uid"]
        if metric_name == "seq_final_reward":
            metric_vals = batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
        elif metric_name == "seq_reward":
            metric_vals = batch.batch["token_level_scores"].sum(dim=-1).cpu().numpy()
        elif metric_name in batch.non_tensor_batch:
            metric_vals = np.asarray(batch.non_tensor_batch[metric_name])
        else:
            raise ValueError(f"DAPO filter metric '{metric_name}' not found in batch")
        prompt_uid2vals = defaultdict(list)
        for uid, val in zip(uids, metric_vals):
            prompt_uid2vals[uid].append(val)
        total_prompts = len(prompt_uid2vals)
        kept_prompt_uids = []
        for uid, vals in prompt_uid2vals.items():
            std = np.std(vals)
            if std > 0 or len(vals) == 1:
                kept_prompt_uids.append(uid)
        kept_prompt_uids_set = set(kept_prompt_uids)
        kept_traj_idxs = [i for i, uid in enumerate(uids) if uid in kept_prompt_uids_set]
        filtered_batch = batch[kept_traj_idxs] if kept_traj_idxs else None
        filter_stats = {
            "total_prompts": total_prompts,
            "kept_prompts": len(kept_prompt_uids),
            "filtered_prompts": total_prompts - len(kept_prompt_uids),
            "total_trajectories": len(uids),
            "kept_trajectories": len(kept_traj_idxs),
        }
        print(f"[DAPO Filter] total_prompts={total_prompts}, kept={len(kept_prompt_uids)}, "
              f"filtered_out={total_prompts - len(kept_prompt_uids)}, "
              f"trajectories: {len(uids)} -> {len(kept_traj_idxs)}")
        return filtered_batch, filter_stats

    def _prepare_reflection_selection_data(
        self,
        batch: DataProto,
        reflection_base_batch: DataProto,
        reflection_gen_source: DataProto,
        repeat_times: int,
    ) -> Optional[dict]:
        is_correct = None
        for key in ("acc", "is_correct", "correct"):
            if key in batch.non_tensor_batch:
                is_correct = np.asarray(batch.non_tensor_batch[key], dtype=bool).reshape(-1)
                break
        if is_correct is None:
            reward_scores = batch.batch["token_level_scores"]
            if reward_scores.ndim > 1:
                reward_scores = reward_scores.sum(dim=-1)
            is_correct = (reward_scores.detach().cpu().numpy() > 0).reshape(-1)

        if len(is_correct) == len(batch) and "uid" in batch.non_tensor_batch:
            response_mask = batch.batch["response_mask"] if "response_mask" in batch.batch.keys() else None
            reflection_seed = int(self.config.data.get("seed") or 0)
            selection_mode = str(self.config.algorithm.get("reflection_selection_mode", "wrong-first"))
            num_select = len(batch) // max(1, repeat_times)

            if selection_mode in {"variance-based", "variance_based"}:
                return {
                    "base_batch": reflection_base_batch,
                    "gen_source": reflection_gen_source,
                    "batch_uids": batch.non_tensor_batch["uid"],
                    "batch_responses": batch.batch["responses"],
                    "is_correct": is_correct,
                    "response_mask": response_mask,
                    "num_select": num_select,
                    "repeat_times": repeat_times,
                    "reflection_seed": reflection_seed,
                    "selection_mode": selection_mode,
                }

        return None

    def _select_reflection_candidates(
        self,
        batch: DataProto,
        reflection_base_batch: DataProto,
        reflection_gen_source: DataProto,
        repeat_times: int,
    ) -> tuple[Optional[DataProto], Optional[DataProto]]:
        is_correct = None
        for key in ("acc", "is_correct", "correct"):
            if key in batch.non_tensor_batch:
                is_correct = np.asarray(batch.non_tensor_batch[key], dtype=bool).reshape(-1)
                break
        if is_correct is None:
            reward_scores = batch.batch["token_level_scores"]
            if reward_scores.ndim > 1:
                reward_scores = reward_scores.sum(dim=-1)
            is_correct = (reward_scores.detach().cpu().numpy() > 0).reshape(-1)

        if len(is_correct) == len(batch) and "uid" in batch.non_tensor_batch:
            response_mask = batch.batch["response_mask"] if "response_mask" in batch.batch.keys() else None
            reflection_seed = int(self.config.data.get("seed") or 0)
            selection_mode = str(self.config.algorithm.get("reflection_selection_mode", "wrong-first"))
            num_select = len(batch) // max(1, repeat_times)

            reflection_indices = select_reflection_indices(
                batch.non_tensor_batch["uid"],
                batch.batch["responses"],
                is_correct,
                num_select=num_select,
                response_mask=response_mask,
                seed=reflection_seed,
                step=self.global_steps,
                selection_mode=selection_mode,
            )
            if reflection_indices:
                selected_base = reflection_base_batch.select_idxs(reflection_indices)
                selected_gen = reflection_gen_source.select_idxs(reflection_indices)
                self._mark_extra_info_reflection(
                    selected_base.non_tensor_batch, batch_size=len(selected_base)
                )
                reflection_gen_batch = self._build_reflection_gen_batch(
                    selected_gen,
                    base_non_tensor_batch=selected_base.non_tensor_batch,
                )
                if reflection_gen_batch is not None:
                    return selected_base, reflection_gen_batch

        return None, None

    def _generate_variance_based_reflection(
        self,
        stored_data: dict,
        gen_batch_output: DataProto,
    ) -> Optional[tuple[DataProto, DataProto]]:
        reflection_base_batch = stored_data["base_batch"]
        reflection_gen_source = stored_data["gen_source"]
        uids = stored_data["batch_uids"]
        is_correct = stored_data["is_correct"]
        response_mask = stored_data["response_mask"]
        num_select = stored_data["num_select"]
        repeat_times = stored_data["repeat_times"]
        reflection_seed = stored_data["reflection_seed"]
        selection_mode = stored_data["selection_mode"]

        reflection_gen_batch = self._build_reflection_gen_batch(
            reflection_gen_source,
            base_non_tensor_batch=reflection_base_batch.non_tensor_batch,
        )

        if reflection_gen_batch is None:
            return None

        reflection_gen_batch.meta_info = dict(gen_batch_output.meta_info)
        reflection_gen_batch = reflection_gen_batch.repeat(repeat_times=repeat_times, interleave=True)

        size_divisor = (
            self.actor_rollout_wg.world_size
            if not self.async_rollout_mode
            else self.config.actor_rollout_ref.rollout.agent.num_workers
        )

        reflection_gen_batch_padded, pad_size = pad_dataproto_to_divisor(
            reflection_gen_batch, size_divisor
        )

        if not self.async_rollout_mode:
            reflection_output = self.actor_rollout_wg.generate_sequences(reflection_gen_batch_padded)
        else:
            reflection_output = self.async_rollout_manager.generate_sequences(reflection_gen_batch_padded)

        reflection_output = unpad_dataproto(reflection_output, pad_size=pad_size)
        reflection_output.meta_info["validate"] = False

        computed_correctness = None
        try:
            reward_extra_keys = set(reflection_output.meta_info.get("reward_extra_keys", []))
            merged_non_tensor = dict(reflection_output.non_tensor_batch)
            for key, value in reflection_gen_batch.non_tensor_batch.items():
                if key not in reward_extra_keys and key not in merged_non_tensor:
                    merged_non_tensor[key] = value

            temp_batch = DataProto(
                batch=reflection_output.batch,
                non_tensor_batch=merged_non_tensor,
                meta_info=reflection_output.meta_info
            )

            reward_result = self.reward_fn(temp_batch, return_dict=True)
            reward_extra_info = reward_result.get("reward_extra_info", {})

            if "acc" in reward_extra_info:
                computed_correctness = np.asarray(reward_extra_info["acc"], dtype=bool).reshape(-1)
        except Exception:
            pass

        reward_extra_keys = set(reflection_output.meta_info.get("reward_extra_keys", []))
        merged_non_tensor = dict(reflection_output.non_tensor_batch)
        for key, value in reflection_gen_batch.non_tensor_batch.items():
            if key not in reward_extra_keys and key not in merged_non_tensor:
                merged_non_tensor[key] = value
        reflection_output.non_tensor_batch = merged_non_tensor

        if computed_correctness is not None:
            reflection_is_correct = computed_correctness
        else:
            reflection_is_correct = None
            for key in ("acc", "is_correct", "correct"):
                if key in reflection_output.non_tensor_batch:
                    reflection_is_correct = np.asarray(reflection_output.non_tensor_batch[key], dtype=bool).reshape(-1)
                    break

        if reflection_is_correct is None:
            reflection_reward_scores = reflection_output.batch.get("token_level_scores")
            if reflection_reward_scores is not None:
                if reflection_reward_scores.ndim > 1:
                    reflection_reward_scores = reflection_reward_scores.sum(dim=-1)
                reflection_is_correct = (reflection_reward_scores.detach().cpu().numpy() > 0).reshape(-1)

        if reflection_is_correct is not None:
            reflection_indices = select_reflection_indices(
                uids,
                reflection_gen_source.batch["responses"],
                is_correct,
                num_select=num_select,
                response_mask=response_mask,
                seed=reflection_seed,
                step=self.global_steps,
                selection_mode=selection_mode,
                reflection_is_correct=reflection_is_correct,
                repeat_times=repeat_times,
            )

            if reflection_indices:
                selected_base_batch = reflection_base_batch.select_idxs(reflection_indices)
                selected_gen_source = reflection_gen_source.select_idxs(reflection_indices)

                self._mark_extra_info_reflection(
                    selected_base_batch.non_tensor_batch, batch_size=len(selected_base_batch)
                )

                selected_reflection_gen_batch = self._build_reflection_gen_batch(
                    selected_gen_source,
                    base_non_tensor_batch=selected_base_batch.non_tensor_batch,
                )
                if selected_reflection_gen_batch is not None:
                    selected_reflection_gen_batch.meta_info = dict(gen_batch_output.meta_info)
                    return selected_base_batch, selected_reflection_gen_batch

        return None, None

    def _validate(self):
        data_source_lst: list[np.ndarray] = []
        reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        reflection_reward_extra_infos_dict: dict[str, list] = defaultdict(list)

        sample_inputs = []
        sample_outputs = []
        sample_gts = []
        sample_scores = []
        sample_turns = []
        sample_difficulty_levels = []
        base_correct_list = []
        reflection_correct_list = []
        reflection_scores = []
        reflection_inputs = []
        reflection_outputs = []
        reflection_gts = []
        reflection_transitions = []
        reflection_data_source_lst = []
        reflection_score_by_idx: dict[int, float] = {}
        reflection_input_by_idx: dict[int, str] = {}
        reflection_output_by_idx: dict[int, str] = {}
        reflection_correct_by_idx: dict[int, int] = {}
        data_source_by_idx: dict[int, str] = {}

        def _extract_correct(reward_extra_info: dict[str, list], scores: list[float]) -> np.ndarray:
            for key in ("acc", "is_correct", "correct"):
                if key in reward_extra_info:
                    return np.asarray(reward_extra_info[key], dtype=bool).reshape(-1)
            return (np.asarray(scores) > 0).reshape(-1)

        enable_reflection_val = self.reflection_validate
        cfg_enable_reflection_val = self.config.algorithm.get("use_reflection_in_validation", None)
        if cfg_enable_reflection_val is not None:
            enable_reflection_val = bool(cfg_enable_reflection_val)
        enable_reflection_val = bool(enable_reflection_val)

        if self._val_sampled_indices is None:
            self._val_sampled_indices = {}
            dataset_by_source: dict[str, list[int]] = defaultdict(list)
            for idx, sample in enumerate(self.val_dataset):
                data_source = "unknown"
                if isinstance(sample, dict):
                    data_source = sample.get("data_source", "unknown")
                else:
                    data_source = getattr(sample, "data_source", "unknown")
                dataset_by_source[data_source].append(idx)

            rng = np.random.RandomState(VAL_SAMPLE_RANDOM_SEED)
            for data_source, indices in dataset_by_source.items():
                max_samples = VAL_SAMPLE_SIZES.get(data_source, None)
                if max_samples is not None and len(indices) > max_samples:
                    sampled = rng.choice(indices, size=max_samples, replace=False).tolist()
                    self._val_sampled_indices[data_source] = sampled
                else:
                    self._val_sampled_indices[data_source] = indices

        if self._val_reflection_sampled_indices is None:
            self._val_reflection_sampled_indices = {}
            dataset_by_source: dict[str, list[int]] = defaultdict(list)
            for idx, sample in enumerate(self.val_dataset):
                data_source = "unknown"
                if isinstance(sample, dict):
                    data_source = sample.get("data_source", "unknown")
                else:
                    data_source = getattr(sample, "data_source", "unknown")
                dataset_by_source[data_source].append(idx)

            rng = np.random.RandomState(VAL_REFLECTION_RANDOM_SEED)
            for data_source, indices in dataset_by_source.items():
                max_samples = VAL_REFLECTION_SAMPLE_SIZES.get(data_source, None)
                if max_samples is not None and len(indices) > max_samples:
                    sampled = rng.choice(indices, size=max_samples, replace=False).tolist()
                    self._val_reflection_sampled_indices[data_source] = sampled
                else:
                    self._val_reflection_sampled_indices[data_source] = indices

        base_indices_set: set[int] = set()
        for indices in (self._val_sampled_indices or {}).values():
            base_indices_set.update(indices)

        reflection_indices_set: set[int] = set()
        for indices in (self._val_reflection_sampled_indices or {}).values():
            reflection_indices_set.update(indices)

        current_idx = 0
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            batch_size = len(test_batch)
            batch_indices = range(current_idx, current_idx + batch_size)
            current_idx += batch_size

            if base_indices_set:
                keep_indices = [i for i, idx in enumerate(batch_indices) if idx in base_indices_set]
                if not keep_indices:
                    continue
                test_batch = test_batch[keep_indices]
                # Update batch_indices to reflect the kept samples (needed for reflection indexing)
                batch_indices = [batch_indices[i] for i in keep_indices]

            if "uid" not in test_batch.non_tensor_batch:
                test_batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(test_batch))], dtype=object
                )

            n_val = self.config.actor_rollout_ref.rollout.val_kwargs.n
            test_batch = test_batch.repeat(repeat_times=n_val, interleave=True)
            # Repeat batch_indices to match the repeated batch (interleaved)
            if n_val > 1:
                batch_indices = [idx for idx in batch_indices for _ in range(n_val)]

            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}

            ground_truths = [
                item.non_tensor_batch.get("reward_model", {}).get("ground_truth", None) for item in test_batch
            ]
            sample_gts.extend(ground_truths)

            for item in test_batch:
                level_str = item.non_tensor_batch.get("extra_info", {}).get("level", None)
                if level_str is None:
                    level_str = item.non_tensor_batch.get("level", "Level 3")
                if isinstance(level_str, str) and level_str.startswith("Level "):
                    try:
                        level = int(level_str.split()[-1])
                    except ValueError:
                        level = 3
                else:
                    level = 3
                sample_difficulty_levels.append(level)

            test_gen_batch = self._get_gen_batch(test_batch)
            test_gen_batch.meta_info = {
                "eos_token_id": self.tokenizer.eos_token_id,
                "pad_token_id": self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample": False,
                "temperature": 0.0,
                "validate": True,
                "global_steps": self.global_steps,
            }

            # pad to be divisible by dp_size
            size_divisor = (
                self.actor_rollout_wg.world_size
                if not self.async_rollout_mode
                else self.config.actor_rollout_ref.rollout.agent.num_workers
            )
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, size_divisor)
            if not self.async_rollout_mode:
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            else:
                test_output_gen_batch_padded = self.async_rollout_manager.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)

            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)
            test_batch.meta_info["validate"] = True

            # Store original inputs
            input_ids = test_batch.batch["prompts"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            # evaluate using reward_function
            if self.val_reward_fn is None:
                raise ValueError("val_reward_fn must be provided for validation.")
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_extra_infos_dict["reward"].extend(scores)
            reward_extra_info = result.get("reward_extra_info", {})
            for key, values in reward_extra_info.items():
                if key not in reward_extra_infos_dict:
                    reward_extra_infos_dict[key] = []
                if isinstance(values, np.ndarray):
                    reward_extra_infos_dict[key].extend(values.tolist())
                else:
                    reward_extra_infos_dict[key].extend(values if isinstance(values, list) else [values])

            base_correct = _extract_correct(reward_extra_info, scores)
            base_correct_list.append(base_correct)

            # collect num_turns of each prompt
            if "__num_turns__" in test_batch.non_tensor_batch:
                sample_turns.append(test_batch.non_tensor_batch["__num_turns__"])

            base_data_sources = test_batch.non_tensor_batch.get("data_source", ["unknown"] * reward_tensor.shape[0])
            data_source_lst.append(np.array([f"{ds}_id_base" for ds in base_data_sources], dtype=object))
            for idx, ds in zip(batch_indices, base_data_sources):
                data_source_by_idx[idx] = ds

            if enable_reflection_val:
                in_reflection_set = [idx in reflection_indices_set for idx in batch_indices]
                if any(in_reflection_set):
                    refl_test_batch = test_batch[in_reflection_set]
                    refl_test_output_gen_batch = test_output_gen_batch[in_reflection_set]
                    refl_ground_truths = [ground_truths[i] for i, val in enumerate(in_reflection_set) if val]

                    reflection_gen_batch = self._build_reflection_gen_batch(
                        refl_test_output_gen_batch, base_non_tensor_batch=refl_test_batch.non_tensor_batch
                    )
                    if reflection_gen_batch is not None:
                        reflection_gen_batch.meta_info = dict(test_gen_batch.meta_info)
                        reflection_gen_batch_padded, reflection_pad_size = pad_dataproto_to_divisor(
                            reflection_gen_batch, size_divisor
                        )
                        if not self.async_rollout_mode:
                            reflection_output_padded = self.actor_rollout_wg.generate_sequences(reflection_gen_batch_padded)
                        else:
                            reflection_output_padded = self.async_rollout_manager.generate_sequences(reflection_gen_batch_padded)
                        reflection_output = unpad_dataproto(reflection_output_padded, pad_size=reflection_pad_size)
                        reflection_gen_batch = unpad_dataproto(reflection_gen_batch_padded, pad_size=reflection_pad_size)
                        reflection_output.meta_info["validate"] = True

                        reward_extra_keys = set(reflection_output.meta_info.get("reward_extra_keys", []))
                        merged_non_tensor = dict(reflection_output.non_tensor_batch)
                        for key, value in reflection_gen_batch.non_tensor_batch.items():
                            if key not in reward_extra_keys and key not in merged_non_tensor:
                                merged_non_tensor[key] = value
                        reflection_output.non_tensor_batch = merged_non_tensor
                        reflection_output_ids = reflection_output.batch["responses"]
                        reflection_output_texts = [
                            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in reflection_output_ids
                        ]
                        reflection_outputs.extend(reflection_output_texts)
                        refl_idx_counter = 0
                        for orig_idx, is_reflected in zip(batch_indices, in_reflection_set):
                            if is_reflected and refl_idx_counter < len(reflection_output_texts):
                                reflection_output_by_idx[orig_idx] = reflection_output_texts[refl_idx_counter]
                                refl_idx_counter += 1
                        reflection_input_ids = reflection_output.batch["prompts"]
                        reflection_input_texts = [
                            self.tokenizer.decode(ids, skip_special_tokens=True) for ids in reflection_input_ids
                        ]
                        reflection_inputs.extend(reflection_input_texts)
                        reflection_gts.extend(refl_ground_truths)
                        refl_idx_counter = 0
                        for orig_idx, is_reflected in zip(batch_indices, in_reflection_set):
                            if is_reflected and refl_idx_counter < len(reflection_input_texts):
                                reflection_input_by_idx[orig_idx] = reflection_input_texts[refl_idx_counter]
                                refl_idx_counter += 1

                        reflection_result = self.val_reward_fn(reflection_output, return_dict=True)
                        reflection_reward_tensor = reflection_result["reward_tensor"]
                        reflection_score_list = reflection_reward_tensor.sum(-1).cpu().tolist()
                        reflection_scores.extend(reflection_score_list)

                        reflection_reward_extra_info = reflection_result.get("reward_extra_info", {})
                        reflection_reward_extra_infos_dict["reward"].extend(reflection_score_list)
                        for key, values in reflection_reward_extra_info.items():
                            if key not in reflection_reward_extra_infos_dict:
                                reflection_reward_extra_infos_dict[key] = []
                            if isinstance(values, np.ndarray):
                                reflection_reward_extra_infos_dict[key].extend(values.tolist())
                            else:
                                reflection_reward_extra_infos_dict[key].extend(
                                    values if isinstance(values, list) else [values]
                                )

                        reflection_correct = _extract_correct(reflection_reward_extra_info, reflection_score_list)
                        reflection_correct_list.append(reflection_correct)
                        refl_idx_counter = 0
                        for orig_idx, is_reflected in zip(batch_indices, in_reflection_set):
                            if is_reflected and refl_idx_counter < len(reflection_score_list):
                                reflection_score_by_idx[orig_idx] = reflection_score_list[refl_idx_counter]
                                if refl_idx_counter < len(reflection_correct):
                                    reflection_correct_by_idx[orig_idx] = int(reflection_correct[refl_idx_counter])
                                refl_idx_counter += 1
                        refl_base_correct = [base_correct[i] for i, val in enumerate(in_reflection_set) if val]
                        if refl_base_correct:
                            transitions = []
                            for b, r in zip(refl_base_correct, reflection_correct.tolist(), strict=True):
                                if b and r:
                                    transitions.append("good_to_good")
                                elif b and not r:
                                    transitions.append("good_to_bad")
                                elif (not b) and r:
                                    transitions.append("bad_to_good")
                                else:
                                    transitions.append("bad_to_bad")
                            reflection_transitions.extend(transitions)
                        reflection_data_sources = reflection_output.non_tensor_batch.get(
                            "data_source", ["unknown"] * len(reflection_score_list)
                        )
                        reflection_data_source_lst.append(
                            np.array([f"{ds}_id_reflection" for ds in reflection_data_sources], dtype=object)
                        )

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        if enable_reflection_val and reflection_inputs:
            self._maybe_log_val_generations(
                inputs=reflection_inputs,
                outputs=reflection_outputs,
                scores=reflection_scores,
            )

        # dump generations
        val_data_dir = self.config.trainer.get("validation_data_dir", None)
        if val_data_dir:
            self._dump_generations(
                inputs=sample_inputs,
                outputs=sample_outputs,
                gts=sample_gts,
                scores=sample_scores,
                reward_extra_infos_dict=reward_extra_infos_dict,
                dump_path=val_data_dir,
            )
            if reflection_inputs:
                reflection_dump_dir = os.path.join(val_data_dir, "reflection")
                reflection_dump_extras = dict(reflection_reward_extra_infos_dict)
                if len(reflection_transitions) == len(reflection_scores):
                    reflection_dump_extras["transition"] = reflection_transitions
                self._dump_generations(
                    inputs=reflection_inputs,
                    outputs=reflection_outputs,
                    gts=reflection_gts,
                    scores=reflection_scores,
                    reward_extra_infos_dict=reflection_dump_extras,
                    dump_path=reflection_dump_dir,
                )

        # === EVAL MODE: Save metadata and extended generations ===
        # Support explicit eval_outputs_dir config, otherwise use default_local_dir/eval_outputs
        eval_save_dir = self.config.trainer.get("eval_outputs_dir", None)
        if eval_save_dir is None:
            eval_save_dir = self.config.trainer.default_local_dir
            if eval_save_dir:
                eval_save_dir = os.path.join(eval_save_dir, "eval_outputs")

        if eval_save_dir:
            os.makedirs(eval_save_dir, exist_ok=True)

            # 1. Save metadata: correct/incorrect (0/1) + difficulty level for all MATH500 problems
            base_correct_flat = np.concatenate(base_correct_list, axis=0) if base_correct_list else np.array([], dtype=bool)
            reflection_correct_flat = np.concatenate(reflection_correct_list, axis=0) if reflection_correct_list else np.array([], dtype=bool)
            metadata_records = []
            for i in range(len(sample_scores)):
                is_correct = int(base_correct_flat[i]) if i < len(base_correct_flat) else int(sample_scores[i] >= 1.0)
                difficulty = sample_difficulty_levels[i] if i < len(sample_difficulty_levels) else 3
                data_source = data_source_by_idx.get(i, "unknown")
                record = {
                    "problem_idx": i,
                    "data_source": data_source,
                    "is_correct": is_correct,
                    "difficulty_level": difficulty,
                    "base_score": sample_scores[i] if i < len(sample_scores) else 0.0,
                }
                if i in reflection_score_by_idx:
                    record["reflection_score"] = reflection_score_by_idx[i]
                metadata_records.append(record)
            metadata_path = os.path.join(eval_save_dir, f"metadata_step{self.global_steps}.jsonl")
            with open(metadata_path, "w") as f:
                for record in metadata_records:
                    f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            with_reflection = sum(1 for r in metadata_records if "reflection_score" in r)
            print(f"Saved metadata to {metadata_path}: {len(metadata_records)} samples, {with_reflection} with reflection")

            # 2. Initialize fixed 128 sample indices (same every val step)
            if self._val_generation_log_indices is None:
                rng = np.random.RandomState(VAL_GENERATION_LOG_SEED)
                n_samples = min(VAL_GENERATION_LOG_SIZE, len(sample_inputs))
                self._val_generation_log_indices = sorted(rng.choice(len(sample_inputs), size=n_samples, replace=False).tolist())
                print(f"Initialized {n_samples} fixed sample indices for detailed generation logging")

            extended_gen_records = []
            for idx in self._val_generation_log_indices:
                if idx >= len(sample_inputs):
                    continue
                record = {
                    "sample_idx": idx,
                    "data_source": data_source_by_idx.get(idx, "unknown"),
                    "difficulty_level": sample_difficulty_levels[idx] if idx < len(sample_difficulty_levels) else 3,
                    "ground_truth": sample_gts[idx] if idx < len(sample_gts) else None,
                    "base_input": sample_inputs[idx] if idx < len(sample_inputs) else "",
                    "base_output": sample_outputs[idx] if idx < len(sample_outputs) else "",
                    "base_score": sample_scores[idx] if idx < len(sample_scores) else 0.0,
                    "base_correct": int(base_correct_flat[idx]) if idx < len(base_correct_flat) else 0,
                }
                if idx in reflection_score_by_idx:
                    record["id_reflection_score"] = reflection_score_by_idx[idx]
                if idx in reflection_input_by_idx:
                    record["id_reflection_input"] = reflection_input_by_idx[idx]
                if idx in reflection_output_by_idx:
                    record["id_reflection_output"] = reflection_output_by_idx[idx]
                if idx in reflection_correct_by_idx:
                    record["id_reflection_correct"] = reflection_correct_by_idx[idx]
                extended_gen_records.append(record)

            extended_gen_path = os.path.join(eval_save_dir, f"extended_generations_step{self.global_steps}.jsonl")
            with open(extended_gen_path, "w") as f:
                for record in extended_gen_records:
                    f.write(json.dumps(record, ensure_ascii=False, default=_json_default) + "\n")
            with_reflection = sum(1 for r in extended_gen_records if "id_reflection_score" in r)
            print(f"Saved {len(extended_gen_records)} extended generations to {extended_gen_path}: {with_reflection} with reflection")

        for key_info, lst in reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(sample_scores), f"{key_info}: {len(lst)=}, {len(sample_scores)=}"

        for key_info, lst in reflection_reward_extra_infos_dict.items():
            assert len(lst) == 0 or len(lst) == len(reflection_scores), (
                f"{key_info}: {len(lst)=}, {len(reflection_scores)=}"
            )

        transition_metrics: dict[str, float] = {}
        if enable_reflection_val and reflection_transitions and reflection_scores:
            # reflection_transitions was computed correctly per-batch during validation loop
            # Use it directly instead of recomputing from mismatched aggregated lists
            good_to_good = sum(1 for t in reflection_transitions if t == "good_to_good")
            good_to_bad = sum(1 for t in reflection_transitions if t == "good_to_bad")
            bad_to_good = sum(1 for t in reflection_transitions if t == "bad_to_good")
            bad_to_bad = sum(1 for t in reflection_transitions if t == "bad_to_bad")

            num_good = good_to_good + good_to_bad
            num_bad = bad_to_good + bad_to_bad

            transition_metrics = {
                "val-aux/id_transition_good_to_good": good_to_good / num_good if num_good > 0 else 0.0,
                "val-aux/id_transition_good_to_bad": good_to_bad / num_good if num_good > 0 else 0.0,
                "val-aux/id_transition_bad_to_good": bad_to_good / num_bad if num_bad > 0 else 0.0,
                "val-aux/id_transition_bad_to_bad": bad_to_bad / num_bad if num_bad > 0 else 0.0,
            }

        all_data_source_lists = data_source_lst + reflection_data_source_lst
        all_data_sources = np.concatenate(all_data_source_lists, axis=0) if all_data_source_lists else np.array([], dtype=object)
        all_sample_uids = sample_inputs + reflection_inputs

        all_reward_extra_infos_dict: dict[str, list] = defaultdict(list)
        for key, values in reward_extra_infos_dict.items():
            all_reward_extra_infos_dict[key].extend(values)
        for key, values in reflection_reward_extra_infos_dict.items():
            all_reward_extra_infos_dict[key].extend(values)

        all_reward_extra_infos_dict.pop("score_base", None)
        all_reward_extra_infos_dict.pop("score_reflection", None)

        expected_len = len(all_data_sources)
        keys_to_remove = [k for k, v in all_reward_extra_infos_dict.items() if len(v) != expected_len]
        for k in keys_to_remove:
            all_reward_extra_infos_dict.pop(k, None)

        data_src2var2metric2val = process_validation_metrics(all_data_sources, all_sample_uids, all_reward_extra_infos_dict)
        metric_dict: dict[str, float] = {}
        for data_source, var2metric2val in data_src2var2metric2val.items():
            core_var = "acc" if "acc" in var2metric2val else "reward"
            for var_name, metric2val in var2metric2val.items():
                n_max = max(int(name.split("@")[-1].split("/")[0]) for name in metric2val.keys())
                for metric_name, metric_val in metric2val.items():
                    if (
                        (var_name == core_var)
                        and any(metric_name.startswith(pfx) for pfx in ["mean", "maj", "best"])
                        and (f"@{n_max}" in metric_name)
                    ):
                        metric_sec = "val-core"
                    else:
                        if var_name not in ["score"]:
                            continue
                        metric_sec = "val-aux"
                    metric_dict[f"{metric_sec}/{data_source}/{var_name}/{metric_name}"] = float(metric_val)

        metric_dict.update(transition_metrics)

        if len(sample_turns) > 0:
            sample_turns_concat = np.concatenate(sample_turns)
            metric_dict["val-aux/num_turns/min"] = float(sample_turns_concat.min())
            metric_dict["val-aux/num_turns/max"] = float(sample_turns_concat.max())
            metric_dict["val-aux/num_turns/mean"] = float(sample_turns_concat.mean())

        print(f"\n{'='*80}")
        print(f"VALIDATION RESULTS @ STEP {self.global_steps}")
        print(f"{'='*80}")

        id_base_acc_per_dataset: dict[str, float] = {}
        id_reflection_acc_per_dataset: dict[str, float] = {}
        for key, val in metric_dict.items():
            if not key.startswith("val-core/") or not key.endswith("/mean@1"):
                continue

            try:
                rest = key[len("val-core/") :]
                data_source, _var_name, _metric_name = rest.rsplit("/", 2)
            except ValueError:
                continue

            if data_source.endswith("_id_base"):
                id_base_acc_per_dataset[data_source[: -len("_id_base")]] = float(val) * 100
            elif data_source.endswith("_id_reflection"):
                id_reflection_acc_per_dataset[data_source[: -len("_id_reflection")]] = float(val) * 100
            else:
                id_base_acc_per_dataset[data_source] = float(val) * 100

        if id_base_acc_per_dataset:
            print("\nI.D. Base acc@1 per dataset:")
            for ds, acc in sorted(id_base_acc_per_dataset.items()):
                print(f"  {ds:<30} {acc:>6.2f}%")
            avg_id_base = sum(id_base_acc_per_dataset.values()) / len(id_base_acc_per_dataset)
            print(f"  {'Average':<30} {avg_id_base:>6.2f}%")
            metric_dict["val-core/avg_id_base_acc"] = avg_id_base / 100.0
            metric_dict["val-core/avg_base_acc"] = avg_id_base / 100.0

        if id_reflection_acc_per_dataset:
            print("\nI.D. Reflection acc@1 per dataset:")
            for ds, acc in sorted(id_reflection_acc_per_dataset.items()):
                print(f"  {ds:<30} {acc:>6.2f}%")
            avg_id_refl = sum(id_reflection_acc_per_dataset.values()) / len(id_reflection_acc_per_dataset)
            print(f"  {'Average':<30} {avg_id_refl:>6.2f}%")
            metric_dict["val-core/avg_id_reflection_acc"] = avg_id_refl / 100.0
            metric_dict["val-core/avg_reflection_acc"] = avg_id_refl / 100.0

        if transition_metrics:
            print("\nTransition metrics:")
            id_transitions = {k: v for k, v in transition_metrics.items() if "id_transition" in k}
            if id_transitions:
                print("  I.D. Reflection:")
                for tkey, tval in sorted(id_transitions.items()):
                    metric_name = tkey.split("/")[-1].replace("id_transition_", "")
                    print(f"    {metric_name:<28} {tval*100:>6.2f}%")

        print(f"{'='*80}\n")

        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        actor_role = Role.ActorRolloutRef if Role.ActorRolloutRef in self.role_worker_mapping else Role.ActorRollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(actor_role)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[actor_role],
                config=self.config.actor_rollout_ref,
                role=str(actor_role),
            )
            self.resource_pool_to_cls[resource_pool][str(actor_role)] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)

            from verl.workers.config import CriticConfig

            critic_cfg: CriticConfig = omega_conf_to_dataclass(self.config.critic)

            if self.use_legacy_worker_impl == "disable":
                # convert critic_cfg into TrainingWorkerConfig
                from verl.workers.engine_workers import TrainingWorkerConfig

                orig_critic_cfg = critic_cfg
                if orig_critic_cfg.strategy == "fsdp":
                    engine_config: FSDPEngineConfig = orig_critic_cfg.model.fsdp_config
                    engine_config.infer_max_token_len_per_gpu = critic_cfg.ppo_infer_max_token_len_per_gpu
                    engine_config.max_token_len_per_gpu = critic_cfg.ppo_max_token_len_per_gpu
                else:
                    raise NotImplementedError(f"Unknown strategy {orig_critic_cfg.strategy=}")

                critic_cfg = TrainingWorkerConfig(
                    model_type="value_model",
                    model_config=orig_critic_cfg.model_config,
                    engine_config=engine_config,
                    optimizer_config=orig_critic_cfg.optim,
                    checkpoint_config=orig_critic_cfg.checkpoint,
                )

            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=critic_cfg)
            self.resource_pool_to_cls[resource_pool][str(Role.Critic)] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy and Role.RefPolicy in self.role_worker_mapping:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role=str(Role.RefPolicy),
            )
            self.resource_pool_to_cls[resource_pool][str(Role.RefPolicy)] = ref_policy_cls

        # create a reward model if reward_fn is None
        # for legacy discriminative reward model, we create a reward model worker here
        # for reward loop discriminative reward model, we create a reward loop manager here
        if not self.use_reward_loop:
            # legacy reward model only handle reward-model based scenario
            if self.use_rm:
                # we create a RM here
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                rm_cls = RayClassWithInitArgs(
                    self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
                )
                self.resource_pool_to_cls[resource_pool][str(Role.RewardModel)] = rm_cls
        else:
            # reward loop handle hybrid reward scenario (rule, disrm, genrm, ...)
            # Note: mode is always "async" since sync mode is deprecated
            can_reward_loop_parallelize = not self.use_rm or self.config.reward_model.enable_resource_pool
            # judge if we can asynchronously parallelize reward model with actor rollout
            # two condition that we can parallelize reward model with actor rollout:
            # 1. reward model is not enabled (rule-based reward can parallelize)
            # 2. reward model is enabled but extra resource pool is enabled
            # If we cannot parallelize, we should enable synchronous mode here, and launch a reward loop manager here
            # else for parallelize mode, we launch a reward worker for each rollout worker (in agent loop, not here)
            if not can_reward_loop_parallelize:
                from verl.experimental.reward_loop import RewardLoopManager

                self.config.reward_model.n_gpus_per_node = self.config.trainer.n_gpus_per_node
                resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
                self.reward_loop_manager = RewardLoopManager(
                    config=self.config,
                    rm_resource_pool=resource_pool,
                )

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        if OmegaConf.select(self.config.global_profiler, "steps") is not None:
            wg_kwargs["profile_steps"] = OmegaConf.select(self.config.global_profiler, "steps")
            # Only require nsight worker options when tool is nsys
            if OmegaConf.select(self.config.global_profiler, "tool") == "nsys":
                assert (
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                    is not None
                ), "worker_nsight_options must be set when using nsys with profile_steps"
                wg_kwargs["worker_nsight_options"] = OmegaConf.to_container(
                    OmegaConf.select(self.config.global_profiler.global_tool_config.nsys, "worker_nsight_options")
                )
        wg_kwargs["device_name"] = self.device_name

        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg[str(Role.Critic)]
            if self.use_legacy_worker_impl == "disable":
                self.critic_wg.reset()
                # assign critic loss
                from functools import partial

                from verl.workers.utils.losses import value_loss

                value_loss_ = partial(value_loss, config=orig_critic_cfg)
                self.critic_wg.set_loss_fn(value_loss_)
            else:
                self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            if str(Role.RefPolicy) in all_wg:
                self.ref_policy_wg = all_wg[str(Role.RefPolicy)]
                self.ref_policy_wg.init_model()
            else:
                # Model engine: ActorRolloutRefWorker
                assert str(Role.ActorRolloutRef) in all_wg, f"{all_wg.keys()=}"
                self.ref_policy_wg = all_wg[str(Role.ActorRolloutRef)]

        self.rm_wg = None
        # initalization of rm_wg will be deprecated in the future
        if self.use_rm and not self.use_reward_loop:
            self.rm_wg = all_wg[str(Role.RewardModel)]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg[str(actor_role)]
        self.actor_rollout_wg.init_model()

        if self.ref_in_actor:
            self.ref_policy_wg = self.actor_rollout_wg

        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        self.async_rollout_manager = None
        if self.config.actor_rollout_ref.rollout.mode == "async":
            # Support custom AgentLoopManager via config
            manager_class_fqn = self.config.actor_rollout_ref.rollout.get("agent", {}).get("agent_loop_manager_class")
            if manager_class_fqn:
                AgentLoopManager = load_class_from_fqn(manager_class_fqn, "AgentLoopManager")
            else:
                from verl.experimental.agent_loop import AgentLoopManager

            if self.config.reward_model.enable and self.config.reward_model.enable_resource_pool:
                rm_resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            else:
                rm_resource_pool = None

            self.async_rollout_mode = True
            self.async_rollout_manager = AgentLoopManager(
                config=self.config,
                worker_group=self.actor_rollout_wg,
                rm_resource_pool=rm_resource_pool,
            )

    def _save_checkpoint(self):
        from verl.utils.fs import local_mkdir_safe

        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")

        actor_remote_path = (
            None
            if self.config.trainer.default_hdfs_dir is None
            else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        )

        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print(
                "Warning: remove_previous_ckpt_in_save is deprecated,"
                + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead"
            )
        max_actor_ckpt_to_keep = (
            self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )
        max_critic_ckpt_to_keep = (
            self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        )

        self.actor_rollout_wg.save_checkpoint(
            actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep
        )

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, str(Role.Critic))
            critic_remote_path = (
                None
                if self.config.trainer.default_hdfs_dir is None
                else os.path.join(
                    self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", str(Role.Critic)
                )
            )
            self.critic_wg.save_checkpoint(
                critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep
            )

        # save dataloader
        local_mkdir_safe(local_global_step_folder)
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        if (
            hasattr(self.config.actor_rollout_ref.actor.checkpoint, "async_save")
            and self.config.actor_rollout_ref.actor.checkpoint.async_save
        ) or (
            "async_save" in self.config.actor_rollout_ref.actor.checkpoint
            and self.config.actor_rollout_ref.actor.checkpoint["async_save"]
        ):
            print("skip write latest_checkpointed_iteration.txt when async_save is True")
            return
        local_latest_checkpointed_iteration = os.path.join(
            self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt"
        )
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))

    def _save_best_model(self, step: int, metric_key: str, metric_value: float) -> None:
        actor_ckpt_dir = os.path.join(self.config.trainer.default_local_dir, f"global_step_{step}", "actor")
        best_hf_dir = self._get_best_model_dir()

        if not os.path.exists(actor_ckpt_dir):
            print(f"Best model checkpoint not found at {actor_ckpt_dir}, skipping conversion")
            return

        try:
            import shutil
            import importlib.util

            convert_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../convert_checkpoint.py"))
            spec = importlib.util.spec_from_file_location("convert_checkpoint", convert_path)
            if spec is None or spec.loader is None:
                raise ImportError(f"Failed to load convert_checkpoint from {convert_path}")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            convert_fsdp_checkpoint = module.convert_fsdp_checkpoint

            best_hf_parent_dir = os.path.dirname(best_hf_dir)
            if best_hf_parent_dir:
                os.makedirs(best_hf_parent_dir, exist_ok=True)

            tmp_best_hf_dir = f"{best_hf_dir}.tmp_{uuid.uuid4().hex}"
            shutil.rmtree(tmp_best_hf_dir, ignore_errors=True)
            os.makedirs(tmp_best_hf_dir, exist_ok=True)

            print("Converting best checkpoint to HF format...")
            convert_fsdp_checkpoint(actor_ckpt_dir, tmp_best_hf_dir)

            with open(os.path.join(tmp_best_hf_dir, "best_model_info.txt"), "w") as f:
                f.write(f"step: {step}\n")
                f.write(f"metric: {metric_key}\n")
                f.write(f"value: {metric_value}\n")

            old_best_hf_dir = None
            if os.path.exists(best_hf_dir):
                old_best_hf_dir = f"{best_hf_dir}.bak_{uuid.uuid4().hex}"
                os.rename(best_hf_dir, old_best_hf_dir)
            os.rename(tmp_best_hf_dir, best_hf_dir)
            if old_best_hf_dir is not None:
                shutil.rmtree(old_best_hf_dir, ignore_errors=True)

            self._write_best_model_state(step=step, metric_key=metric_key, metric_value=metric_value)
            print(f"Best model saved to {best_hf_dir}")
        except Exception as e:
            print(f"Failed to convert best checkpoint to HF format: {e}")

    def _get_best_model_dir(self) -> str:
        best_model_dir = self.config.trainer.get("best_model_dir", None)
        if best_model_dir:
            return best_model_dir
        return os.path.join(self.config.trainer.default_local_dir, "best_model")

    def _best_model_state_path(self) -> str:
        return os.path.join(self.config.trainer.default_local_dir, "best_model_state.json")

    def _write_best_model_state(self, step: int, metric_key: str, metric_value: float) -> None:
        state_path = self._best_model_state_path()
        tmp_path = f"{state_path}.tmp"
        payload = {"step": step, "metric_key": metric_key, "metric_value": metric_value}
        try:
            state_dir = os.path.dirname(state_path)
            if state_dir:
                os.makedirs(state_dir, exist_ok=True)
            with open(tmp_path, "w") as f:
                json.dump(payload, f)
            os.replace(tmp_path, state_path)
        except Exception as e:
            print(f"Warning: failed to write best model state to {state_path}: {e}")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

    def _load_best_model_state(self) -> None:
        if bool(self.config.trainer.get("reset_best_model_on_resume", False)):
            return
        if self.global_steps <= 0:
            return

        state: dict[str, Any] | None = None
        state_path = self._best_model_state_path()
        if os.path.exists(state_path):
            try:
                with open(state_path) as f:
                    state = json.load(f)
            except Exception as e:
                print(f"Warning: failed to load best model state from {state_path}: {e}")

        if state is None:
            info_path = os.path.join(self._get_best_model_dir(), "best_model_info.txt")
            if os.path.exists(info_path):
                try:
                    parsed: dict[str, str] = {}
                    with open(info_path) as f:
                        for line in f:
                            if ":" not in line:
                                continue
                            key, value = line.split(":", 1)
                            parsed[key.strip().lower()] = value.strip()
                    state = {
                        "step": int(parsed["step"]),
                        "metric_key": parsed.get("metric"),
                        "metric_value": float(parsed["value"]),
                    }
                except Exception as e:
                    print(f"Warning: failed to parse best model info from {info_path}: {e}")

        if not state:
            return

        try:
            best_step = int(state["step"])
            best_value = float(state["metric_value"])
            best_key = state.get("metric_key")
        except Exception as e:
            print(f"Warning: invalid best model state at {state_path}: {e}")
            return

        if best_step > self.global_steps:
            print(
                "Warning: ignoring best model state "
                f"(best_step={best_step}) since current checkpoint is step {self.global_steps}"
            )
            return

        self.best_val_step = best_step
        self.best_val_metric = best_value
        if isinstance(best_key, str) and best_key:
            if best_key != self.best_model_metric_key:
                print(
                    "Warning: best_model_metric_key mismatch on resume "
                    f"(config={self.best_model_metric_key}, saved={best_key}); using saved key for consistency."
                )
            self.best_model_metric_key = best_key

        print(
            "Restored best model state: "
            f"{self.best_model_metric_key}={self.best_val_metric:.6f} at step {self.best_val_step}"
        )

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, (
                    "resume ckpt must specify the global_steps"
                )
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])

        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, str(Role.Critic))
        # load actor
        self.actor_rollout_wg.load_checkpoint(
            actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
        )
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(
                critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load
            )

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _start_profiling(self, do_profile: bool) -> None:
        """Start profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.start_profile(role="e2e", profile_step=self.global_steps)
            if self.use_reference_policy:
                self.ref_policy_wg.start_profile(profile_step=self.global_steps)
            if self.use_critic:
                self.critic_wg.start_profile(profile_step=self.global_steps)
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.start_profile(profile_step=self.global_steps)

    def _stop_profiling(self, do_profile: bool) -> None:
        """Stop profiling for all worker groups if profiling is enabled."""
        if do_profile:
            self.actor_rollout_wg.stop_profile()
            if self.use_reference_policy:
                self.ref_policy_wg.stop_profile()
            if self.use_critic:
                self.critic_wg.stop_profile()
            if self.use_rm and not self.use_reward_loop:
                self.rm_wg.stop_profile()

    def _get_dp_size(self, worker_group, role: str) -> int:
        """Get data parallel size from worker group dispatch info.

        This method retrieves the data parallel size by querying the dispatch info
        for the specified role. The dispatch info is cached for subsequent calls.

        Args:
            worker_group: The worker group to query dispatch info from.
            role: The role name (e.g., "actor", "critic") to get DP size for.

        Returns:
            The data parallel size (number of DP ranks).
        """
        if role not in worker_group._dispatch_info:
            dp_rank_mapping = worker_group._query_dispatch_info(role)
            worker_group._dispatch_info[role] = dp_rank_mapping
        else:
            dp_rank_mapping = worker_group._dispatch_info[role]
        return max(dp_rank_mapping) + 1

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen", keep_minibatch=False):
        """Reorder the data on single controller such that each dp rank gets similar total tokens.

        When use_prefix_grouper is enabled, uses group-level balancing to keep samples with
        the same uid together on the same rank for prefix sharing optimization.
        """
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1)  # (train_batch_size,)
        workload_lst = calculate_workload(global_seqlen_lst)
        # Get dp_size from dispatch info to correctly balance across data parallel ranks
        # Note: world_size may include tensor/pipeline parallel dimensions, but we only want DP
        dp_size = self._get_dp_size(self.actor_rollout_wg, "actor")

        # Use group-level balancing for PrefixGrouper to keep same-uid samples together
        if getattr(self, "use_prefix_grouper", False) and "uid" in batch.non_tensor_batch:
            from verl.utils.seqlen_balancing import get_group_balanced_partitions

            uid_list = list(batch.non_tensor_batch["uid"])
            seqlen_list = global_seqlen_lst.tolist()

            # Count number of uid groups
            num_groups = len(set(uid_list))

            if num_groups % dp_size != 0:
                raise ValueError(
                    f"PrefixGrouper with balance_batch requires num_uid_groups ({num_groups}) "
                    f"% dp_size ({dp_size}) == 0. "
                    f"This ensures each rank gets equal number of groups. "
                    f"Current batch_size={batch_size}, adjust batch_size to be a multiple of "
                    f"dp_size * rollout.n."
                )

            global_partition_lst = get_group_balanced_partitions(
                seqlen_list=seqlen_list,
                uid_list=uid_list,
                k_partitions=dp_size,
            )

        elif keep_minibatch:
            # Decouple the DP balancing and mini-batching.
            minibatch_size = self.config.actor_rollout_ref.actor.get("ppo_mini_batch_size")
            minibatch_num = len(workload_lst) // minibatch_size
            global_partition_lst = [[] for _ in range(dp_size)]
            for i in range(minibatch_num):
                rearrange_minibatch_lst = get_seqlen_balanced_partitions(
                    workload_lst[i * minibatch_size : (i + 1) * minibatch_size],
                    k_partitions=dp_size,
                    equal_size=True,
                )
                for j, part in enumerate(rearrange_minibatch_lst):
                    global_partition_lst[j].extend([x + minibatch_size * i for x in part])
        else:
            global_partition_lst = get_seqlen_balanced_partitions(workload_lst, k_partitions=dp_size, equal_size=True)
        # Place smaller micro-batches at both ends to reduce the bubbles in pipeline parallel.
        # Skip reordering within partitions for PrefixGrouper to maintain uid grouping
        if not getattr(self, "use_prefix_grouper", False):
            for idx, partition in enumerate(global_partition_lst):
                partition.sort(key=lambda x: (workload_lst[x], x))
                ordered_partition = partition[::2] + partition[1::2][::-1]
                global_partition_lst[idx] = ordered_partition

        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(
            seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix
        )
        metrics.update(global_balance_stats)
        return global_idx

    def _compute_values(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, compute_loss=False)
            output = self.critic_wg.infer_batch(batch_td)
            output = output.get()
            values = tu.get(output, "values")
            values = no_padding_2_padding(values, batch_td)
            values = tu.get_tensordict({"values": values.float()})
            values = DataProto.from_tensordict(values)
        else:
            values = self.critic_wg.compute_values(batch)
        return values

    def _compute_ref_log_prob(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=False, compute_loss=False)
            output = self.ref_policy_wg.compute_ref_log_prob(batch_td)
            # gather output
            log_probs = tu.get(output, "log_probs")
            # step 4. No padding to padding
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            ref_log_prob = tu.get_tensordict({"ref_log_prob": log_probs.float()})
            ref_log_prob = DataProto.from_tensordict(ref_log_prob)
        else:
            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)

        return ref_log_prob

    def _compute_old_log_prob(self, batch: DataProto):
        if self.use_legacy_worker_impl == "disable":
            # TODO: remove step 1, 2, 4 after we make the whole training tensordict and padding free
            # step 1: convert dataproto to tensordict.
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to nopadding
            batch_td = left_right_2_no_padding(batch_td)
            # step 3: add meta info
            tu.assign_non_tensor(batch_td, calculate_entropy=True, compute_loss=False)
            output = self.actor_rollout_wg.compute_log_prob(batch_td)
            # gather output
            entropy = tu.get(output, "entropy")
            log_probs = tu.get(output, "log_probs")
            old_log_prob_mfu = tu.get(output, "metrics")["mfu"]
            # step 4. No padding to padding
            entropy = no_padding_2_padding(entropy, batch_td)
            log_probs = no_padding_2_padding(log_probs, batch_td)
            # step 5: rebuild a tensordict and convert to dataproto
            old_log_prob = tu.get_tensordict({"old_log_probs": log_probs.float(), "entropys": entropy.float()})
            old_log_prob = DataProto.from_tensordict(old_log_prob)
        else:
            old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
            old_log_prob_mfu = 0
        return old_log_prob, old_log_prob_mfu

    def _update_actor(self, batch: DataProto) -> DataProto:
        rollout_config = self.config.actor_rollout_ref.rollout
        batch.meta_info["multi_turn"] = rollout_config.multi_turn.enable
        # TODO: Make "temperature" single source of truth from generation.
        batch.meta_info["temperature"] = rollout_config.temperature
        # update actor
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            calculate_entropy = self.config.actor_rollout_ref.actor.entropy_coeff != 0.0
            ppo_mini_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.actor_rollout_ref.actor.ppo_epochs
            seed = self.config.actor_rollout_ref.actor.data_loader_seed
            shuffle = self.config.actor_rollout_ref.actor.shuffle
            tu.assign_non_tensor(
                batch_td,
                calculate_entropy=calculate_entropy,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            actor_output = self.actor_rollout_wg.update_actor(batch_td)
            actor_output = tu.get(actor_output, "metrics")
            actor_output = rename_dict(actor_output, "actor/")
            # modify key name
            actor_output["perf/mfu/actor"] = actor_output.pop("actor/mfu")
            actor_output = DataProto.from_single_dict(data={}, meta_info={"metrics": actor_output})
        else:
            actor_output = self.actor_rollout_wg.update_actor(batch)
        return actor_output

    def _update_critic(self, batch: DataProto) -> DataProto:
        if self.use_legacy_worker_impl == "disable":
            batch_td = batch.to_tensordict()
            # step 2: convert from padding to no-padding
            batch_td = left_right_2_no_padding(batch_td)
            ppo_mini_batch_size = self.config.critic.ppo_mini_batch_size
            ppo_mini_batch_size = ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
            ppo_epochs = self.config.critic.ppo_epochs
            seed = self.config.critic.data_loader_seed
            shuffle = self.config.critic.shuffle
            tu.assign_non_tensor(
                batch_td,
                global_batch_size=ppo_mini_batch_size,
                mini_batch_size=ppo_mini_batch_size,
                epochs=ppo_epochs,
                seed=seed,
                dataloader_kwargs={"shuffle": shuffle},
            )

            output = self.critic_wg.train_mini_batch(batch_td)
            output = output.get()
            output = tu.get(output, "metrics")
            output = rename_dict(output, "critic/")
            # modify key name
            output["perf/mfu/critic"] = output.pop("critic/mfu")
            critic_output = DataProto.from_single_dict(data={}, meta_info={"metrics": output})
        else:
            critic_output = self.critic_wg.update_critic(batch)
        return critic_output

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf

        from verl.utils.tracking import Tracking

        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )

        self.global_steps = 0
        self.best_val_metric = -float("inf")
        self.best_val_step = -1
        self.best_model_metric_key = self.config.trainer.get(
            "best_model_metric_key", "val-core/avg_id_base_acc"
        )

        # load checkpoint before doing anything
        self._load_checkpoint()
        self._load_best_model_state()

        current_epoch = self.global_steps // len(self.train_dataloader)

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                logger.finish()
                return

        if self.config.actor_rollout_ref.rollout.get("skip_rollout", False):
            rollout_skip = RolloutSkip(self.config, self.actor_rollout_wg)
            rollout_skip.wrap_generate_sequences()

        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        self.max_steps_duration = 0

        prev_step_profile = False
        curr_step_profile = (
            self.global_steps in self.config.global_profiler.steps
            if self.config.global_profiler.steps is not None
            else False
        )
        next_step_profile = False

        def run_step(batch: DataProto, gen_batch: DataProto, *, repeat_times: int, allow_reflection: bool):
            nonlocal prev_step_profile, curr_step_profile, next_step_profile, last_val_metrics
            nonlocal dapo_accumulated_batch, dapo_num_prompt_in_batch, dapo_num_gen_batches
            metrics = {}
            timing_raw = {}

            with marked_timer("start_profile", timing_raw):
                self._start_profiling(
                    not prev_step_profile and curr_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            # pass global_steps to trace
            gen_batch.meta_info["global_steps"] = self.global_steps
            gen_batch_output = gen_batch.repeat(repeat_times=repeat_times, interleave=True)

            pending_reflection = None
            reflection_base_batch = None
            reflection_gen_source = None
            is_last_step = self.global_steps >= self.total_training_steps
            with marked_timer("step", timing_raw):
                # generate a batch
                with marked_timer("gen", timing_raw, color="red"):
                    if not self.async_rollout_mode:
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch_output)
                    else:
                        gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch_output)

                    timing_raw.update(gen_batch_output.meta_info["timing"])
                    gen_batch_output.meta_info.pop("timing", None)

                # Prepare reflection data if reflection is enabled
                # Reflection can work with any advantage estimator, not just GRPO
                if allow_reflection and self.use_reflection:
                    num_prompts = len(np.unique(batch.non_tensor_batch["uid"])) if "uid" in batch.non_tensor_batch else len(batch)
                    dapo_status = "DAPO" if self.dapo_enabled else "GRPO"
                    print(f"[{dapo_status} Base Step] Generated {len(batch)} base rollouts (prompts={num_prompts}, repeat_times={repeat_times})")
                    reflection_base_batch = batch.repeat(repeat_times=repeat_times, interleave=True)
                    reflection_gen_source = gen_batch_output

                if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                    if self.reward_fn is None:
                        raise ValueError("A reward_fn is required for REMAX advantage estimation.")

                    with marked_timer("gen_max", timing_raw, color="purple"):
                        gen_baseline_batch = deepcopy(gen_batch)
                        gen_baseline_batch.meta_info["do_sample"] = False
                        if not self.async_rollout_mode:
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                        else:
                            gen_baseline_output = self.async_rollout_manager.generate_sequences(gen_baseline_batch)
                        batch = batch.union(gen_baseline_output)
                        # compute reward model score on batch
                        rm_scores = None
                        if self.use_rm and "rm_scores" not in batch.batch.keys():
                            if not self.use_reward_loop:
                                rm_scores = self.rm_wg.compute_rm_score(batch)
                            else:
                                assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                                rm_scores = self.reward_loop_manager.compute_rm_score(batch)
                            batch = batch.union(rm_scores)

                        # Compute or extract reward for REMAX baseline
                        reward_baseline_tensor = self._compute_or_extract_reward(
                            batch, reward_fn=self.reward_fn, sum_reward=True
                        )

                        keys_to_pop = set(gen_baseline_output.batch.keys())
                        if rm_scores is not None:
                            keys_to_pop.update(rm_scores.batch.keys())
                        batch.pop(batch_keys=list(keys_to_pop))

                        batch.batch["reward_baselines"] = reward_baseline_tensor

                        del rm_scores, gen_baseline_batch, gen_baseline_output
                # repeat to align with repeated responses in rollout
                batch = batch.repeat(repeat_times=repeat_times, interleave=True)
                batch = batch.union(gen_batch_output)

                if "response_mask" not in batch.batch.keys():
                    batch.batch["response_mask"] = compute_response_mask(batch)
                # Balance the number of valid tokens across DP ranks.
                # NOTE: This usually changes the order of data in the `batch`,
                # which won't affect the advantage calculation (since it's based on uid),
                # but might affect the loss calculation (due to the change of mini-batching).
                balance_idx = None
                if self.config.trainer.balance_batch:
                    balance_idx = self._balance_batch(batch, metrics=metrics)
                if balance_idx is not None and reflection_base_batch is not None and reflection_gen_source is not None:
                    reflection_base_batch.reorder(balance_idx)
                    reflection_gen_source.reorder(balance_idx)

                # compute global_valid tokens
                batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                with marked_timer("reward", timing_raw, color="yellow"):
                    # compute reward model score
                    if self.use_rm and "rm_scores" not in batch.batch.keys():
                        if not self.use_reward_loop:
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                        else:
                            assert self.reward_loop_manager is not None, "RewardLoopManager is None"
                            reward_tensor = self.reward_loop_manager.compute_rm_score(batch)
                        batch = batch.union(reward_tensor)

                    # Compute or extract reward for training
                    if self.config.reward_model.launch_reward_fn_async:
                        future_reward = compute_reward_async.remote(
                            data=batch, config=self.config, tokenizer=self.tokenizer
                        )
                    else:
                        reward_tensor, reward_extra_infos_dict = self._compute_or_extract_reward(
                            batch, reward_fn=self.reward_fn, return_dict=False
                        )

                # Operating Mode Selection:
                # - Bypass mode: Sets old_log_probs = rollout_log_probs (2 policies: π_rollout, π_θ)
                # - Decoupled mode: Recomputes old_log_probs as proximal anchor (3 policies: π_rollout, π_old, π_θ)
                #   Note: π_old computed once per data batch, serves as stable reference during mini-batch updates
                rollout_corr_config = self.config.algorithm.get("rollout_correction", None)
                bypass_recomputing_logprobs = rollout_corr_config and rollout_corr_config.get("bypass_mode", False)
                if bypass_recomputing_logprobs:  # Use `rollout_log_probs`
                    from verl.trainer.ppo.rollout_corr_helper import apply_bypass_mode

                    apply_bypass_mode(
                        batch=batch,
                        rollout_corr_config=rollout_corr_config,
                        policy_loss_config=self.config.actor_rollout_ref.actor.policy_loss,
                    )
                else:  # Recompute old_log_probs
                    with marked_timer("old_log_prob", timing_raw, color="blue"):
                        old_log_prob, old_log_prob_mfu = self._compute_old_log_prob(batch)
                        entropys = old_log_prob.batch["entropys"]
                        response_masks = batch.batch["response_mask"]
                        actor_config = self.config.actor_rollout_ref.actor
                        entropy_agg = agg_loss(
                            loss_mat=entropys,
                            loss_mask=response_masks,
                            loss_agg_mode=actor_config.loss_agg_mode,
                            loss_scale_factor=actor_config.loss_scale_factor,
                        )
                        old_log_prob_metrics = {
                            "actor/entropy": entropy_agg.detach().item(),
                            "perf/mfu/actor_infer": old_log_prob_mfu,
                        }
                        metrics.update(old_log_prob_metrics)
                        old_log_prob.batch.pop("entropys")
                        batch = batch.union(old_log_prob)
                        if "rollout_log_probs" in batch.batch.keys():
                            # TODO: we may want to add diff of probs too.
                            from verl.utils.debug.metrics import calculate_debug_metrics

                            metrics.update(calculate_debug_metrics(batch))

                assert "old_log_probs" in batch.batch, f'"old_log_prob" not in {batch.batch.keys()=}'

                if self.use_reference_policy:
                    # compute reference log_prob
                    with marked_timer(str(Role.RefPolicy), timing_raw, color="olive"):
                        ref_log_prob = self._compute_ref_log_prob(batch)
                        batch = batch.union(ref_log_prob)

                # compute values
                if self.use_critic:
                    with marked_timer("values", timing_raw, color="cyan"):
                        values = self._compute_values(batch)
                        batch = batch.union(values)

                with marked_timer("adv", timing_raw, color="brown"):
                    # we combine with rule-based rm
                    reward_extra_infos_dict: dict[str, list]
                    if self.config.reward_model.launch_reward_fn_async:
                        reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                    batch.batch["token_level_scores"] = reward_tensor

                    if reward_extra_infos_dict:
                        batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})

                    # Select reflection prompts once correctness is available.
                    stored_reflection_data = None  # Initialize for variance-based selection after actor update
                    if reflection_base_batch is not None and reflection_gen_source is not None:
                        # Use helper method to prepare reflection data
                        stored_reflection_data = self._prepare_reflection_selection_data(
                            batch, reflection_base_batch, reflection_gen_source, repeat_times
                        )

                        # If not variance-based, do immediate selection
                        if stored_reflection_data is None:
                            selected_base, selected_gen = self._select_reflection_candidates(
                                batch, reflection_base_batch, reflection_gen_source, repeat_times
                            )
                            if selected_base is not None and selected_gen is not None:
                                pending_reflection = (selected_base, selected_gen)

                    # compute rewards. apply_kl_penalty if available
                    if self.config.algorithm.use_kl_in_reward:
                        batch, kl_metrics = apply_kl_penalty(
                            batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty
                        )
                        metrics.update(kl_metrics)
                    else:
                        batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]

                    step_type = "Reflection" if not allow_reflection else "Base"

                    # DAPO filtering: Decoupled from reflection
                    # DAPO filtering can work independently of reflection (vanilla DAPO)
                    # DAPO filtering can also work with reflection (DAPO-based RefPo)
                    if self.dapo_enabled:
                        num_prompts = len(np.unique(batch.non_tensor_batch["uid"])) if "uid" in batch.non_tensor_batch else len(batch)
                        print(f"[DAPO Filter {step_type}] BEFORE filter: batch={len(batch)} prompts={num_prompts}")
                        filtered_batch, filter_stats = self._apply_dapo_filter(batch, metric_name=self.dapo_metric)
                        print(f"[DAPO Filter {step_type}] AFTER filter: kept={filter_stats['kept_prompts']}/{filter_stats['total_prompts']} prompts, "
                              f"filtered={filter_stats['filtered_prompts']}, traj={filter_stats['kept_trajectories']}/{filter_stats['total_trajectories']}")
                        dapo_num_gen_batches += 1
                        if filtered_batch is not None and len(filtered_batch) > 0:
                            if dapo_accumulated_batch is None:
                                dapo_accumulated_batch = filtered_batch
                            else:
                                for key in ["global_token_num"]:
                                    dapo_accumulated_batch.meta_info.pop(key, None)
                                    filtered_batch.meta_info.pop(key, None)
                                dapo_accumulated_batch = DataProto.concat([dapo_accumulated_batch, filtered_batch])
                            dapo_num_prompt_in_batch += filter_stats["kept_prompts"]
                        prompt_bsz = self.config.data.train_batch_size
                        traj_bsz = prompt_bsz * repeat_times
                        metrics["dapo/num_gen_batches"] = dapo_num_gen_batches
                        metrics["dapo/accumulated_prompts"] = dapo_num_prompt_in_batch
                        if dapo_num_prompt_in_batch < prompt_bsz:
                            max_gen_batches = self.dapo_max_gen_batches
                            print(f"[DAPO Filter {step_type}] Need more: {dapo_num_prompt_in_batch}/{prompt_bsz} prompts, gen_batches={dapo_num_gen_batches}")
                            if max_gen_batches <= 0 or dapo_num_gen_batches < max_gen_batches:
                                return None, False
                            else:
                                print(f"[DAPO Filter {step_type}] ERROR: Reached max_gen_batches={max_gen_batches} but only "
                                              f"{dapo_num_prompt_in_batch}/{prompt_bsz} prompts.")
                                dapo_accumulated_batch = None
                                dapo_num_prompt_in_batch = 0
                                dapo_num_gen_batches = 0
                                pending_reflection = None
                                return None, True
                        print(f"[DAPO Filter {step_type}] Training with {traj_bsz} trajectories ({prompt_bsz} prompts)")
                        batch = dapo_accumulated_batch[:traj_bsz]
                        batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()
                        dapo_accumulated_batch = None
                        dapo_num_prompt_in_batch = 0
                        dapo_num_gen_batches = 0
                    else:
                        mode_str = f"GRPO+Reflection" if self.use_reflection else "Vanilla GRPO"
                        print(f"[{mode_str} {step_type} Step] Training with {len(batch)} trajectories")

                    # Compute rollout correction: IS weights, rejection sampling, and metrics
                    # Only runs in decoupled mode (computes once per batch using stable π_old)
                    # In bypass mode, this is skipped - actor computes metrics from evolving π_θ vs π_rollout
                    if (
                        rollout_corr_config is not None
                        and "rollout_log_probs" in batch.batch
                        and not bypass_recomputing_logprobs  # Only in decoupled mode
                    ):
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_add_to_batch

                        # Compute IS weights, apply rejection sampling, compute metrics
                        batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
                        # IS and off-policy metrics already have rollout_corr/ prefix
                        metrics.update(is_metrics)

                    # compute advantages, executed on the driver process
                    norm_adv_by_std_in_grpo = self.config.algorithm.get(
                        "norm_adv_by_std_in_grpo", True
                    )  # GRPO adv normalization factor

                    batch = compute_advantage(
                        batch,
                        adv_estimator=self.config.algorithm.adv_estimator,
                        gamma=self.config.algorithm.gamma,
                        lam=self.config.algorithm.lam,
                        num_repeat=self.config.actor_rollout_ref.rollout.n,
                        norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                        config=self.config.algorithm,
                    )

                # update critic
                if self.use_critic:
                    with marked_timer("update_critic", timing_raw, color="pink"):
                        critic_output = self._update_critic(batch)
                    critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                    metrics.update(critic_output_metrics)

                # implement critic warmup
                if self.config.trainer.critic_warmup <= self.global_steps:
                    # update actor
                    with marked_timer("update_actor", timing_raw, color="red"):
                        actor_output = self._update_actor(batch)
                    actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                    metrics.update(actor_output_metrics)

                # Generate reflections using the UPDATED model (after actor update)
                if "stored_reflection_data" in locals() and stored_reflection_data is not None:
                    reflection_result = self._generate_variance_based_reflection(stored_reflection_data, gen_batch_output)
                    if reflection_result is not None:
                        pending_reflection = reflection_result

                # Log rollout generations if enabled
                rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                if rollout_data_dir:
                    self._log_rollout_data(batch, reward_extra_infos_dict, timing_raw, rollout_data_dir)

            new_best_model = False
            best_metric_key = self.best_model_metric_key
            best_metric_value: Optional[float] = None

            # validate
            if (
                self.val_reward_fn is not None
                and self.config.trainer.test_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0)
            ):
                with marked_timer("testing", timing_raw, color="green"):
                    val_metrics: dict = self._validate()
                    if is_last_step:
                        last_val_metrics = val_metrics

                    metric_key = best_metric_key
                    current_metric = val_metrics.get(metric_key)
                    if current_metric is None:
                        for fallback_key in (
                            "val-core/avg_id_reflection_acc",
                            "val-core/avg_reflection_acc",
                            "val-core/avg_id_base_acc",
                            "val-core/avg_base_acc",
                        ):
                            if fallback_key in val_metrics:
                                metric_key = fallback_key
                                current_metric = val_metrics[fallback_key]
                                break
                    current_metric_f = float(current_metric) if current_metric is not None else -float("inf")

                    if current_metric_f > self.best_val_metric:
                        self.best_val_metric = current_metric_f
                        self.best_val_step = self.global_steps
                        best_metric_key = metric_key
                        best_metric_value = current_metric_f
                        new_best_model = True
                        print(f"\n{'='*80}")
                        print(f"NEW BEST MODEL: {best_metric_key}={current_metric_f:.4f} at step {self.global_steps}")
                        print(f"{'='*80}")
                metrics.update(val_metrics)

            # Check if the ESI (Elastic Server Instance)/training plan is close to expiration.
            esi_close_to_expiration = should_save_ckpt_esi(
                max_steps_duration=self.max_steps_duration,
                redundant_time=self.config.trainer.esi_redundant_time,
            )
            # Check if the conditions for saving a checkpoint are met.
            # The conditions include a mandatory condition (1) and
            # one of the following optional conditions (2/3/4):
            # 1. The save frequency is set to a positive value.
            # 2. It's the last training step.
            # 3. The current step number is a multiple of the save frequency.
            # 4. The ESI(Elastic Server Instance)/training plan is close to expiration.
            save_best_model = bool(self.config.trainer.get("save_best_model", True))
            should_save_ckpt = (
                self.config.trainer.save_freq > 0
                and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0 or esi_close_to_expiration)
            ) or (save_best_model and new_best_model)
            if should_save_ckpt:
                if esi_close_to_expiration:
                    print("Force saving checkpoint: ESI instance expiration approaching.")
                with marked_timer("save_checkpoint", timing_raw, color="green"):
                    self._save_checkpoint()
                if save_best_model and new_best_model and best_metric_value is not None:
                    self._save_best_model(step=self.global_steps, metric_key=best_metric_key, metric_value=best_metric_value)

            with marked_timer("stop_profile", timing_raw):
                next_step_profile = (
                    self.global_steps + 1 in self.config.global_profiler.steps
                    if self.config.global_profiler.steps is not None
                    else False
                )
                self._stop_profiling(
                    curr_step_profile and not next_step_profile
                    if self.config.global_profiler.profile_continuous_steps
                    else curr_step_profile
                )
                prev_step_profile = curr_step_profile
                curr_step_profile = next_step_profile

            steps_duration = timing_raw["step"]
            self.max_steps_duration = max(self.max_steps_duration, steps_duration)

            # training metrics
            metrics.update(
                {
                    "training/global_step": self.global_steps,
                    "training/epoch": epoch,
                }
            )
            # collect metrics
            metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
            metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
            # TODO: implement actual tflpo and theoretical tflpo
            n_gpus = self.resource_pool_manager.get_n_gpus()
            metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
            # Note: mismatch metrics (KL, PPL, etc.) are collected at line 1179 after advantage computation

            # this is experimental and may be changed/removed in the future in favor of a general-purpose one
            if isinstance(self.train_dataloader.sampler, AbstractCurriculumSampler):
                self.train_dataloader.sampler.update(batch=batch)

            # TODO: make a canonical logger that supports various backend
            logger.log(data=metrics, step=self.global_steps)

            progress_bar.update(1)
            self.global_steps += 1

            if (
                hasattr(self.config.actor_rollout_ref.actor, "profiler")
                and self.config.actor_rollout_ref.actor.profiler.tool == "torch_memory"
            ):
                self.actor_rollout_wg.dump_memory_snapshot(
                    tag=f"post_update_step{self.global_steps}", sub_dir=f"step{self.global_steps}"
                )

            if is_last_step:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=True)
                pprint(f"Final validation metrics: {last_val_metrics}")
                progress_bar.close()
                return pending_reflection, True

            # this is experimental and may be changed/removed in the future
            # in favor of a general-purpose data buffer pool
            if hasattr(self.train_dataset, "on_batch_end"):
                # The dataset may be changed after each training batch
                self.train_dataset.on_batch_end(batch=batch)

            return pending_reflection, False

        for epoch in range(current_epoch, self.config.trainer.total_epochs):
            dapo_accumulated_batch = None
            dapo_num_prompt_in_batch = 0
            dapo_num_gen_batches = 0

            for batch_dict in self.train_dataloader:
                if hasattr(self.actor_rollout_wg, "async_calls_finalize_fn_exec"):
                    self.actor_rollout_wg.async_calls_finalize_fn_exec(blocking=False)
                batch: DataProto = DataProto.from_single_dict(batch_dict)

                # add uid to batch
                batch.non_tensor_batch["uid"] = np.array(
                    [str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object
                )

                gen_batch = self._get_gen_batch(batch)
                repeat_times = self.config.actor_rollout_ref.rollout.n
                pending_reflection, stop_training = run_step(
                    batch,
                    gen_batch,
                    repeat_times=repeat_times,
                    allow_reflection=True,
                )
                if stop_training:
                    logger.finish()
                    return
                if pending_reflection is not None:
                    reflection_batch, reflection_gen_batch = pending_reflection
                    dapo_status = "DAPO-based" if self.dapo_enabled else "GRPO-based"
                    print(f"[{dapo_status} Reflection] Starting REFLECTION STEP training with {len(reflection_batch)} base prompts")
                    _, stop_training = run_step(
                        reflection_batch,
                        reflection_gen_batch,
                        repeat_times=repeat_times,
                        allow_reflection=False,
                    )
                    if stop_training:
                        logger.finish()
                    dapo_accumulated_batch = None
                    dapo_num_prompt_in_batch = 0
                    dapo_num_gen_batches = 0
                    pending_reflection = None
                    if stop_training:
                        return
