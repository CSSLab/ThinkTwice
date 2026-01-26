#!/usr/bin/env bash
set -euo pipefail
set -x

# DAPO baseline (dynamic sampling) using the `recipe` submodule trainer.
#
# Prereq: `git submodule update --init --recursive recipe`
#
# Example:
#   MODEL_PATH=/data1/models/Qwen/Qwen3-4B-Instruct-2507 \
#   CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
#   bash examples/dapo_trainer/run_qwen3_4b_countdown_dapo.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

if [[ ! -f "$REPO_ROOT/recipe/dapo/main_dapo.py" ]]; then
  echo "Missing recipe submodule. Run: git submodule update --init --recursive recipe" >&2
  exit 1
fi

DEFAULT_DATA_DIR="$REPO_ROOT/data/countdown"
if [[ -d "$REPO_ROOT/verl/data/countdown" && ! -d "$DEFAULT_DATA_DIR" ]]; then
  DEFAULT_DATA_DIR="$REPO_ROOT/verl/data/countdown"
fi

DATA_DIR="${DATA_DIR:-$DEFAULT_DATA_DIR}"
TRAIN_FILE="${TRAIN_FILE:-$DATA_DIR/train.parquet}"
TEST_FILE="${TEST_FILE:-$DATA_DIR/test.parquet}"

MODEL_PATH="${MODEL_PATH:-/data1/models/Qwen/Qwen3-4B-Instruct-2507}"
REWARD_FN="${REWARD_FN:-$REPO_ROOT/verl/reward_functions/countdown.py}"

DEFAULT_LOCAL_DIR="${DEFAULT_LOCAL_DIR:-$HOME/ckpts/dapo_countdown}"
mkdir -p "$DEFAULT_LOCAL_DIR"

# Detect GPUs
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -ra _DEVICES <<<"$CUDA_VISIBLE_DEVICES"
  NUM_GPUS=${#_DEVICES[@]}
else
  NUM_GPUS=${NUM_GPUS:-8}
fi

RUN_ID="$(date +'%Y%m%d_%H%M%S')"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_4b_countdown_dapo_${RUN_ID}}"

MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-1024}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-1024}"

TRAIN_PROMPT_BSZ="${TRAIN_PROMPT_BSZ:-32}"
N_RESP_PER_PROMPT="${N_RESP_PER_PROMPT:-4}"
GEN_PROMPT_BSZ="${GEN_PROMPT_BSZ:-$((TRAIN_PROMPT_BSZ * 3))}"

ENABLE_FILTER_GROUPS="${ENABLE_FILTER_GROUPS:-true}"
FILTER_GROUPS_METRIC="${FILTER_GROUPS_METRIC:-acc}"
MAX_NUM_GEN_BATCHES="${MAX_NUM_GEN_BATCHES:-10}"

ENABLE_OVERLONG_BUFFER="${ENABLE_OVERLONG_BUFFER:-false}"
OVERLONG_BUFFER_LEN="${OVERLONG_BUFFER_LEN:-$((MAX_RESPONSE_LENGTH / 4))}"
OVERLONG_PENALTY_FACTOR="${OVERLONG_PENALTY_FACTOR:-1.0}"

PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 -m recipe.dapo.main_dapo \
  data.train_files="${TRAIN_FILE}" \
  data.val_files="${TEST_FILE}" \
  data.prompt_key=prompt \
  data.truncation='error' \
  data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH}" \
  data.gen_batch_size="${GEN_PROMPT_BSZ}" \
  data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
  +data.apply_chat_template_kwargs.enable_thinking=False \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.filter_groups.enable="${ENABLE_FILTER_GROUPS}" \
  algorithm.filter_groups.metric="${FILTER_GROUPS_METRIC}" \
  algorithm.filter_groups.max_num_gen_batches="${MAX_NUM_GEN_BATCHES}" \
  reward_model.reward_manager=dapo \
  reward_model.overlong_buffer.enable="${ENABLE_OVERLONG_BUFFER}" \
  reward_model.overlong_buffer.len="${OVERLONG_BUFFER_LEN}" \
  reward_model.overlong_buffer.penalty_factor="${OVERLONG_PENALTY_FACTOR}" \
  custom_reward_function.path="${REWARD_FN}" \
  custom_reward_function.name=compute_countdown_score \
  trainer.project_name="verl_dapo_countdown" \
  trainer.experiment_name="${EXPERIMENT_NAME}" \
  trainer.n_gpus_per_node="${NUM_GPUS}" \
  trainer.nnodes=1 \
  trainer.default_local_dir="${DEFAULT_LOCAL_DIR}" \
  "$@"
