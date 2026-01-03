#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
wait_for_gpu.sh: GPUの空きVRAMが指定量以上になるまで待ってからコマンドを実行します。

Usage:
  scripts/wait_for_gpu.sh [--gpu N] [--min-free-mib MIB] [--interval SEC] [--timeout SEC] -- <command...>

Options:
  --gpu N            GPU index (default: 0)
  --min-free-mib MIB 空きVRAMの下限 (MiB) (default: 2000)
  --interval SEC     ポーリング間隔 (秒) (default: 30)
  --timeout SEC      タイムアウト (秒). 0は無制限 (default: 0)

Examples:
  scripts/wait_for_gpu.sh --gpu 0 --min-free-mib 8000 -- python -m my_method.train.train_multitask --config configs_my_method/multitask_receiver_shot_d2_consistency.yaml

  tmux new-session -d -s wait_train_d2 \
    "bash -lc 'scripts/wait_for_gpu.sh --gpu 0 --min-free-mib 8000 -- python -m my_method.train.train_multitask --config configs_my_method/multitask_receiver_shot_d2_consistency.yaml'"
EOF
}

GPU=0
MIN_FREE_MIB=2000
INTERVAL=30
TIMEOUT=0

ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --gpu)
      GPU="${2:-}"; shift 2;;
    --min-free-mib)
      MIN_FREE_MIB="${2:-}"; shift 2;;
    --interval)
      INTERVAL="${2:-}"; shift 2;;
    --timeout)
      TIMEOUT="${2:-}"; shift 2;;
    -h|--help)
      usage; exit 0;;
    --)
      shift
      ARGS=("$@")
      break;;
    *)
      echo "[wait_for_gpu] unknown arg: $1" >&2
      usage
      exit 2;;
  esac
done

if [[ ${#ARGS[@]} -eq 0 ]]; then
  echo "[wait_for_gpu] missing command after --" >&2
  usage
  exit 2
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "[wait_for_gpu] nvidia-smi not found. Are you on a GPU node/container?" >&2
  exit 127
fi

start_ts="$(date +%s)"
echo "[wait_for_gpu] waiting: gpu=${GPU}, min_free_mib=${MIN_FREE_MIB}, interval=${INTERVAL}s, timeout=${TIMEOUT}s"

while true; do
  # memory.free はドライバ/環境により揺れるので total-used でも計算できるように両方取る
  # CSV: "free, total, used, util"
  line="$(
    nvidia-smi -i "${GPU}" \
      --query-gpu=memory.free,memory.total,memory.used,utilization.gpu \
      --format=csv,noheader,nounits 2>/dev/null | head -n 1 || true
  )"

  if [[ -z "${line}" ]]; then
    echo "[wait_for_gpu] failed to query GPU via nvidia-smi (will retry)" >&2
    sleep "${INTERVAL}"
    continue
  fi

  free_mib="$(echo "${line}" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
  total_mib="$(echo "${line}" | awk -F',' '{gsub(/ /,"",$2); print $2}')"
  used_mib="$(echo "${line}" | awk -F',' '{gsub(/ /,"",$3); print $3}')"
  util="$(echo "${line}" | awk -F',' '{gsub(/ /,"",$4); print $4}')"

  # free_mib が取れない環境用フォールバック: total-used
  if [[ -z "${free_mib}" || "${free_mib}" == "N/A" ]]; then
    if [[ -n "${total_mib}" && -n "${used_mib}" ]]; then
      free_mib="$(( total_mib - used_mib ))"
    else
      free_mib="0"
    fi
  fi

  now_ts="$(date +%s)"
  elapsed="$(( now_ts - start_ts ))"

  echo "[wait_for_gpu] elapsed=${elapsed}s free=${free_mib}MiB total=${total_mib}MiB used=${used_mib}MiB util=${util}%"

  if [[ "${free_mib}" -ge "${MIN_FREE_MIB}" ]]; then
    echo "[wait_for_gpu] condition met (free ${free_mib}MiB >= ${MIN_FREE_MIB}MiB). running command:"
    echo "  ${ARGS[*]}"
    exec "${ARGS[@]}"
  fi

  if [[ "${TIMEOUT}" -gt 0 && "${elapsed}" -ge "${TIMEOUT}" ]]; then
    echo "[wait_for_gpu] timeout reached (${TIMEOUT}s). exiting." >&2
    exit 124
  fi

  sleep "${INTERVAL}"
done


