#!/usr/bin/env bash
set -euo pipefail
: "${BASE_MODEL:?Set BASE_MODEL to the pretrained model path}"
: "${IMAGE_DATA:?Set IMAGE_DATA to the image directory}"
: "${JSON_DATA:?Set JSON_DATA to the annotations file}"
: "${DATA_OUTPUT:?Set DATA_OUTPUT to the output directory}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${repo_root}/EAGLE/eagle/ge_data"
"${PYTHON_BIN:-python}" -m eagle.ge_data.allocation     --model_type "qwen2_vl_v" --model "${BASE_MODEL}"     --image_data_path "${IMAGE_DATA}" --json_data_path "${JSON_DATA}"     --outdir "${DATA_OUTPUT}" "$@"
