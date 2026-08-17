#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python_bin="${PYTHON_BIN:-/mnt/bn/bytenn-yg2/llx/venvs/msd/bin/python}"
torchrun_bin="$(dirname "${python_bin}")/torchrun"
ssh_key="/opt/tiger/rh2/rh2/init/id_rsa_private"
master_addr="${MASTER_ADDR:-10.122.137.87}"
master_port="${MASTER_PORT:-40041}"
gpus_per_node="${GPUS_PER_NODE:-8}"
job_name="${JOB_NAME:-llava16_7b_decouple_kmeans32_minimal_full}"
cpdir="${CKPT_DIR:-${repo_root}/ckpt_a800_decouple_kmeans32}"
log_dir="${LOG_DIR:-${repo_root}/logs/train_llava16_7b_msd_a800/${job_name}}"

remote_hosts=(
    "2605:340:cd51:603:7931:e14f:127e:141c"
    "2605:340:cd51:601:8766:550f:5b26:7ea8"
    "2605:340:cd51:602:9084:5144:e8ef:546d"
)
remote_ports=(10353 10353 10353)
node_names=(worker workera workerb workerc)

if [[ ! "${gpus_per_node}" =~ ^[1-8]$ ]]; then
    echo "GPUS_PER_NODE must be in [1, 8]: ${gpus_per_node}" >&2
    exit 2
fi
if [[ ! "${master_port}" =~ ^[0-9]+$ ]]; then
    echo "MASTER_PORT must be numeric: ${master_port}" >&2
    exit 2
fi
if [[ ! -x "${python_bin}" || ! -x "${torchrun_bin}" ]]; then
    echo "Training environment is incomplete: ${python_bin}" >&2
    exit 1
fi

mkdir -p "${cpdir}" "${log_dir}/pids"
if find "${cpdir}" -mindepth 1 -maxdepth 1 -type d -name 'llava_v16_7b_msd_decouple_kmeans32_[0-9]*' -print -quit | grep -q .; then
    echo "Refusing to overwrite an existing reproduction checkpoint in ${cpdir}" >&2
    exit 1
fi

cuda_devices=""
for ((gpu=0; gpu<gpus_per_node; gpu++)); do
    [[ -n "${cuda_devices}" ]] && cuda_devices+=","
    cuda_devices+="${gpu}"
done

train_args=(
    "${repo_root}/EAGLE/eagle/train/main_deepspeed_llava16_decouple_kmeans_repro.py"
    --deepspeed_config "${repo_root}/EAGLE/eagle/train/ds_config_llava16_repro.json"
    --basepath /mnt/bn/bytenn-yg2/llx/models/llava-v1.6-vicuna-7b-hf
    --tmpdir_v /mnt/bn/bytenn-yg2/llx/hidden_states/llava16_7b_lcs_genlong_original_teacher
    --tmpdir_t /mnt/bn/bytenn-yg2/llx/hidden_states/llava16_7b_sharegpt_vllm_teacher
    --config "${repo_root}/EAGLE/eagle/train/llava_hf_v16_7B_config.json"
    --cpdir "${cpdir}"
)

quoted_train_args=""
printf -v quoted_train_args '%q ' "${train_args[@]}"

node_command() {
    local node_rank="$1"
    local pid_file="${log_dir}/pids/${node_names[$node_rank]}.pid"
    local command
    printf -v command \
        "cd %q && export PATH=%q:\$PATH PYTHONPATH=%q TORCH_EXTENSIONS_DIR=%q MAX_JOBS=8 CUDA_VISIBLE_DEVICES=%q OMP_NUM_THREADS=4 PYTHONUNBUFFERED=1 WANDB_MODE=offline WANDB_SILENT=true NCCL_SOCKET_IFNAME=eth0 GLOO_SOCKET_IFNAME=eth0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && printf '%%s\\n' \$\$ > %q && exec %q --nnodes=4 --nproc-per-node=%q --node-rank=%q --master-addr=%q --master-port=%q %s" \
        "${repo_root}" \
        "$(dirname "${python_bin}")" \
        "${repo_root}/EAGLE" \
        "/mnt/bn/bytenn-yg2/llx/venvs/msd/torch_extensions" \
        "${cuda_devices}" \
        "${pid_file}" \
        "${torchrun_bin}" \
        "${gpus_per_node}" \
        "${node_rank}" \
        "${master_addr}" \
        "${master_port}" \
        "${quoted_train_args}"
    printf '%s' "${command}"
}

declare -a launcher_pids=()
cleaned_up=0

terminate_from_pid_file() {
    local pid_file="$1"
    [[ -f "${pid_file}" ]] || return 0
    local process_id
    process_id="$(<"${pid_file}")"
    if [[ "${process_id}" =~ ^[0-9]+$ ]]; then
        pkill -TERM -P "${process_id}" 2>/dev/null || true
        kill -TERM "${process_id}" 2>/dev/null || true
    fi
}

cleanup() {
    if (( cleaned_up )); then
        return
    fi
    cleaned_up=1
    for pid in "${launcher_pids[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    terminate_from_pid_file "${log_dir}/pids/worker.pid"
    for remote_index in 0 1 2; do
        remote_pid_file="${log_dir}/pids/${node_names[$((remote_index + 1))]}.pid"
        remote_cleanup=""
        printf -v remote_cleanup \
            "if [[ -f %q ]]; then p=\$(<%q); [[ \$p =~ ^[0-9]+$ ]] && { pkill -TERM -P \$p 2>/dev/null || true; kill -TERM \$p 2>/dev/null || true; }; fi" \
            "${remote_pid_file}" "${remote_pid_file}"
        timeout 10s ssh -6 \
            -o BatchMode=yes \
            -o StrictHostKeyChecking=no \
            -o UserKnownHostsFile=/dev/null \
            -o ConnectTimeout=5 \
            -i "${ssh_key}" \
            -p "${remote_ports[$remote_index]}" \
            "tiger@${remote_hosts[$remote_index]}" \
            "${remote_cleanup}" >/dev/null 2>&1 || true
    done
}
trap cleanup EXIT INT TERM

for remote_index in 0 1 2; do
    node_rank=$((remote_index + 1))
    remote_command="$(node_command "${node_rank}")"
    ssh -6 \
        -o BatchMode=yes \
        -o StrictHostKeyChecking=no \
        -o UserKnownHostsFile=/dev/null \
        -o ServerAliveInterval=30 \
        -o ServerAliveCountMax=20 \
        -i "${ssh_key}" \
        -p "${remote_ports[$remote_index]}" \
        "tiger@${remote_hosts[$remote_index]}" \
        "${remote_command}" \
        >"${log_dir}/${node_names[$node_rank]}.log" 2>&1 &
    launcher_pids+=("$!")
done

local_command="$(node_command 0)"
bash -lc "${local_command}" >"${log_dir}/worker.log" 2>&1 &
launcher_pids+=("$!")

set +e
return_code=0
while (( ${#launcher_pids[@]} > 0 )); do
    finished_pid=""
    wait -n -p finished_pid "${launcher_pids[@]}"
    node_rc=$?
    if (( node_rc != 0 )); then
        return_code=${node_rc}
        cleanup
        break
    fi
    remaining_pids=()
    for pid in "${launcher_pids[@]}"; do
        [[ "${pid}" == "${finished_pid}" ]] || remaining_pids+=("${pid}")
    done
    launcher_pids=("${remaining_pids[@]}")
done
for pid in "${launcher_pids[@]}"; do
    wait "${pid}" 2>/dev/null || true
done
set -e

if (( return_code != 0 )); then
    echo "Training job ${job_name} failed with rc=${return_code}; logs: ${log_dir}" >&2
    exit "${return_code}"
fi
echo "Training job ${job_name} completed; logs: ${log_dir}"
