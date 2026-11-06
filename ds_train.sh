#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${BASE_MODEL:?Set BASE_MODEL to the pretrained model path}"
: "${VISUAL_DATA:?Set VISUAL_DATA to the visual hidden-state directory}"
: "${TEXT_DATA:?Set TEXT_DATA to the text hidden-state directory}"
: "${MODEL_CONFIG:?Set MODEL_CONFIG to the draft model config}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${repo_root}/EAGLE/eagle/train"
deepspeed --master_port "${MASTER_PORT:-29500}"     --include "${GPU_INCLUDE:-localhost:0}" main_deepspeed.py     --deepspeed_config "${DEEPSPEED_CONFIG:-ds_config.json}"     --tmpdir_v "${VISUAL_DATA}" --tmpdir_t "${TEXT_DATA}"     --basepath "${BASE_MODEL}" --cpdir "${CKPT_DIR:-${repo_root}/ckpt}"     --config "${MODEL_CONFIG}" "$@"
