#!/usr/bin/env bash
set -euo pipefail
set -x

# Minimal DAPO-style GRPO run on the built-in Countdown parquet dataset.
#
# Prereqs:
# - Qwen model downloaded locally (set MODEL_PATH)
# - vLLM installed (pip install -e ".[vllm]" or similar)
#
# Optional (offline logging):
#   export WANDB_MODE=offline

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

DEFAULT_DATA_DIR="$REPO_ROOT/data/countdown"
if [[ -d "$REPO_ROOT/verl/data/countdown" && ! -d "$DEFAULT_DATA_DIR" ]]; then
  DEFAULT_DATA_DIR="$REPO_ROOT/verl/data/countdown"
fi
DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test.parquet}"

MODEL_PATH="${MODEL_PATH:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}"
MODEL_NAME="${MODEL_NAME:-qwen3-4b}"

DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$HOME/ckpts/countdown_dapo}"
export TMPDIR="${TMPDIR:-$DEFAULT_LOCAL_DIR/tmp}"
export RAY_TMPDIR="${RAY_TMPDIR:-$DEFAULT_LOCAL_DIR/ray_tmp}"
mkdir -p "$DEFAULT_LOCAL_DIR" "$TMPDIR" "$RAY_TMPDIR"

# Detect GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _DEVICES <<<"$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#_DEVICES[@]}
else
  NUM_GPUS=${NUM_GPUS:-8}
fi

RUN_ID="$(date +'%Y%m%d_%H%M%S')"

ROLLOUT_N="${ROLLOUT_N:-4}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"
TP_SIZE="${TP_SIZE:-1}"

PPO_MINI_BATCH="${PPO_MINI_BATCH:-8}"
PPO_MICRO_BATCH="${PPO_MICRO_BATCH:-16}"
LOGPROB_MICRO="${LOGPROB_MICRO:-16}"

ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-False}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 4))}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

EXPERIMENT_NAME="${MODEL_NAME}_countdown_dapo_${RUN_ID}_n${ROLLOUT_N}"
ROLLOUT_DIR="${DEFAULT_LOCAL_DIR}/rollouts/${EXPERIMENT_NAME}"
mkdir -p "$ROLLOUT_DIR"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILE" \
  data.val_files="$TEST_FILE" \
  data.train_batch_size="${TRAIN_BATCH_SIZE}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="$MODEL_PATH" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=1e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TP_SIZE}" \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.7 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MICRO}" \
  actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
  trainer.critic_warmup=0 \
  trainer.logger='["console","wandb"]' \
  trainer.project_name='verl_dapo_countdown' \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.resume_mode=disable \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
  trainer.rollout_data_dir="${ROLLOUT_DIR}" \
  reward_model.reward_manager=dapo \
  +reward_model.reward_kwargs.max_resp_len="${MAX_RESPONSE_LENGTH}" \
  +reward_model.reward_kwargs.overlong_buffer_cfg.enable="${ENABLE_OVERLONG_BUFFER}" \
  +reward_model.reward_kwargs.overlong_buffer_cfg.len="${OVERLONG_BUFFER_LEN}" \
  +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor="${OVERLONG_PENALTY_FACTOR}" \
  +reward_model.reward_kwargs.overlong_buffer_cfg.log=False \
  "$@"
