#!/bin/bash

source $(conda info --base)/etc/profile.d/conda.sh
conda activate verl

export WANDB_ENTITY=difanjiao
export WANDB_PROJECT=drgrpo-math
# offline mode
export WANDB_MODE=offline
export WANDB_API_KEY=f510b3737ade928e3e94556e9fae86fcbd716dc2
## For online logging, set `WANDB_API_KEY` in your environment.

set -x

# Make sure we're not attaching to an old Ray cluster (which can ignore new _temp_dir/object_store_memory settings).
# Default to an isolated local Ray instance so multiple runs can coexist on the same node.
# If you want to attach to an existing cluster, set: RAY_INIT_ADDRESS=auto (or a concrete address).
RAY_INIT_ADDRESS=${RAY_INIT_ADDRESS:-local}
RAY_STOP_BEFORE_RUN=${RAY_STOP_BEFORE_RUN:-0}
if [[ "${RAY_STOP_BEFORE_RUN}" == "1" ]]; then
    ray stop -f || true
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRATCH_DIR="${REPO_ROOT}/scratch"
B=32
VAL_B=512
N=8
L=3000
NAME=drgrpo_math_baseline_Qwen3-4B-Instruct-2507
DEFAULT_CKPT_ROOT="/data1/qianfeng/ckpts"
DEFAULT_RAY_TMPDIR="/data1/qianfeng/ray"
DEFAULT_TMPDIR="/data1/qianfeng/tmp"
CKPT_ROOT=${CKPT_ROOT:-"${DEFAULT_CKPT_ROOT}"}
RAY_TMPDIR=${RAY_TMPDIR:-"${DEFAULT_RAY_TMPDIR}"}
TMPDIR=${TMPDIR:-"${DEFAULT_TMPDIR}"}
# make directories if not exist
mkdir -p "${CKPT_ROOT}" "${RAY_TMPDIR}" "${TMPDIR}"
export RAY_TMPDIR
export TMPDIR
# Ray memory / spilling controls (host RAM + object store).
RAY_OBJECT_STORE_GB=${RAY_OBJECT_STORE_GB:-64}
RAY_OBJECT_STORE_BYTES=$((RAY_OBJECT_STORE_GB * 1024 * 1024 * 1024))
RAY_MEMORY_USAGE_THRESHOLD=${RAY_MEMORY_USAGE_THRESHOLD:-0.99}
export RAY_memory_usage_threshold="${RAY_MEMORY_USAGE_THRESHOLD}"
TRAINER_DEFAULT_LOCAL_DIR="${CKPT_ROOT}/${NAME}"
OOD_BASE_ANSWERS_PATH="${REPO_ROOT}/math_eval/ood_traces/qwen3-30b-a3b/base_answers.parquet"
VAL_FILE=${VAL_FILE:-"$SCRATCH_DIR/math_combined/test.parquet"}
mkdir -p "${TRAINER_DEFAULT_LOCAL_DIR}" "${RAY_TMPDIR}"
REFLECTION_STEPS=${REFLECTION_STEPS:-0}
# vLLM KV cache sizing. Higher values can OOM when combined with large max_num_batched_tokens / long context.
GPU_UTIL=${GPU_UTIL:-0.5}
# vLLM uses this to size/compile some buffers; too large can trigger CUDA OOM (e.g. 40000 -> ~1.45GiB temp buf).
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-20000}
VISIBLE_DEVICES=${VISIBLE_DEVICES:-4,5}
SAVE_FREQ=${SAVE_FREQ:-10}
TEST_FREQ=${TEST_FREQ:-10}
RESUME_MODE=${RESUME_MODE:-auto}
ACTOR_PARAM_OFFLOAD=${ACTOR_PARAM_OFFLOAD:-False}
ACTOR_OPTIMIZER_OFFLOAD=${ACTOR_OPTIMIZER_OFFLOAD:-False}
ENABLE_ACTIVATION_OFFLOAD=${ENABLE_ACTIVATION_OFFLOAD:-False}
AGENT_NUM_WORKERS=${AGENT_NUM_WORKERS:-2}
REWARD_USE_REWARD_LOOP=${REWARD_USE_REWARD_LOOP:-False}

CUDA_VISIBLE_DEVICES=${VISIBLE_DEVICES} PYTHONUNBUFFERED=1 python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.use_kl_in_reward=False \
    +algorithm.reflection.steps=${REFLECTION_STEPS} \
    +algorithm.reflection.validate=True \
    +algorithm.reflection.ood_base_answers_path="${OOD_BASE_ANSWERS_PATH}" \
    data.train_files=$SCRATCH_DIR/hendrycks_math/train.parquet \
    data.val_files=$VAL_FILE \
    data.validation_shuffle=False \
    data.train_batch_size=$B \
    data.val_batch_size=$VAL_B \
    data.max_prompt_length=5120 \
    data.max_response_length=$L \
    data.filter_overlong_prompts=False \
    data.truncation='error' \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.model.path=/data1/models/Qwen/Qwen3-4B-Instruct-2507 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-sum-norm \
    actor_rollout_ref.actor.loss_scale_factor=$L \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=4 \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=40000 \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=40000 \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=40000 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.enable_activation_offload=${ENABLE_ACTIVATION_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${ACTOR_PARAM_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${ACTOR_OPTIMIZER_OFFLOAD} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.top_p=1.0 \
    actor_rollout_ref.rollout.top_k=-1 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_UTIL} \
    actor_rollout_ref.rollout.n=$N \
    actor_rollout_ref.rollout.max_model_len=8192 \
    actor_rollout_ref.rollout.max_num_batched_tokens=${MAX_NUM_BATCHED_TOKENS} \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.6 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.95 \
    actor_rollout_ref.rollout.agent.num_workers=${AGENT_NUM_WORKERS} \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.use_reward_loop=${REWARD_USE_REWARD_LOOP} \
    reward_model.reward_manager=naive \
    custom_reward_function.path="${REPO_ROOT}/verl/verl/utils/reward_score/math_dataset.py" \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.logger=[console,wandb] \
    trainer.project_name=reflection-grpo-math \
    trainer.experiment_name=${NAME} \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.default_local_dir="${TRAINER_DEFAULT_LOCAL_DIR}" \
    +trainer.save_best_model=True \
    +trainer.best_model_metric_key=val-core/avg_combined_reflection_acc \
    trainer.save_freq=${SAVE_FREQ} \
    trainer.test_freq=${TEST_FREQ} \
    trainer.total_epochs=5 \
    ray_kwargs.ray_init.num_cpus=32 \
    +ray_kwargs.ray_init.address="${RAY_INIT_ADDRESS}" \
    +ray_kwargs.ray_init._temp_dir="${RAY_TMPDIR}" \
    +ray_kwargs.ray_init.object_store_memory=${RAY_OBJECT_STORE_BYTES} \
    trainer.val_before_train=True \
    trainer.log_val_generations=32 \
    trainer.resume_mode=${RESUME_MODE} \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.max_critic_ckpt_to_keep=1 \
    +trainer.start_save_step=10 \
    $@
