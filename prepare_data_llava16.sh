#!/usr/bin/env bash
set -euo pipefail
: "${BASE_MODEL:?Set BASE_MODEL to the pretrained model path}"
: "${IMAGE_DATA:?Set IMAGE_DATA to the image directory}"
: "${JSON_DATA:?Set JSON_DATA to the annotations file}"
: "${DATA_OUTPUT:?Set DATA_OUTPUT to the output directory}"
"${PYTHON_BIN:-python}" -m eagle.ge_data.prepare_llava16_hidden_states     --model "${BASE_MODEL}" --image-root "${IMAGE_DATA}"     --annotation-path "${JSON_DATA}" --output-dir "${DATA_OUTPUT}" "$@"
