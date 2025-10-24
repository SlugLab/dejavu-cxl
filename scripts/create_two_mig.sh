#!/usr/bin/env bash
set -euo pipefail

# Create two MIG instances on a target GPU and print their UUIDs.
#
# Usage:
#   scripts/create_two_mig.sh [-g <gpu_index>] [-p <gi_profile_id>]
#
# Examples:
#   scripts/create_two_mig.sh                 # prompts for a profile id
#   scripts/create_two_mig.sh -g 0 -p 14      # creates two GI profile 14 instances on GPU 0
#
# Notes:
# - Enabling MIG and creating/destroying instances usually requires root.
# - Use `nvidia-smi mig -i <GPU> -lgip` to list GPU Instance profile IDs.
# - The script attempts with sudo first and falls back to non-sudo.

GPU_IDX=0
GI_PROFILE_ID=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    -g|--gpu)
      GPU_IDX="$2"; shift 2 ;;
    -p|--profile)
      GI_PROFILE_ID="$2"; shift 2 ;;
    -h|--help)
      grep -E '^# ' "$0" | sed 's/^# //'; exit 0 ;;
    *)
      echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || { echo "Error: required command '$1' not found" >&2; exit 1; }
}

run_root() {
  if command -v sudo >/dev/null 2>&1; then
    sudo "$@" || "$@"
  else
    "$@"
  fi
}

need_cmd nvidia-smi

echo "==> Checking GPU $GPU_IDX and MIG capability"
nvidia-smi -i "$GPU_IDX" -L || true

# Detect MIG mode
MIG_CURRENT=$(nvidia-smi -i "$GPU_IDX" -q 2>/dev/null | awk '/MIG Mode/{f=1} f && /Current/{print $4; exit}')
MIG_PENDING=$(nvidia-smi -i "$GPU_IDX" -q 2>/dev/null | awk '/MIG Mode/{f=1} f && /Pending/{print $4; exit}')

if [[ "${MIG_CURRENT:-Disabled}" != "Enabled" ]]; then
  echo "==> Enabling MIG mode on GPU $GPU_IDX (requires root)"
  run_root nvidia-smi -i "$GPU_IDX" -mig 1
  echo "MIG mode enabled. If this is a multi-tenant system, ensure no processes are using the GPU."
fi

if [[ "${MIG_PENDING:-Disabled}" == "Enabled" ]]; then
  echo "==> MIG pending state detected; a GPU reset may be required"
  echo "Attempting a GPU reset on GPU $GPU_IDX (may fail if busy)"
  run_root nvidia-smi -i "$GPU_IDX" --gpu-reset || true
fi

echo "==> Clearing any existing MIG instances on GPU $GPU_IDX"
run_root nvidia-smi mig -i "$GPU_IDX" -dci || true
run_root nvidia-smi mig -i "$GPU_IDX" -dgi || true

if [[ -z "$GI_PROFILE_ID" ]]; then
  echo "==> Available GPU Instance profiles on GPU $GPU_IDX:"
  nvidia-smi mig -i "$GPU_IDX" -lgip || true
  echo
  read -rp "Enter GI profile ID to create (will create two of this): " GI_PROFILE_ID
fi

if [[ -z "$GI_PROFILE_ID" ]]; then
  echo "Error: GI profile ID is required" >&2
  exit 1
fi

echo "==> Creating two MIG instances of GI profile $GI_PROFILE_ID on GPU $GPU_IDX"
run_root nvidia-smi mig -i "$GPU_IDX" -cgi "$GI_PROFILE_ID,$GI_PROFILE_ID" -C

echo "==> MIG devices present (system-wide):"
nvidia-smi -L | sed 's/^/  /'

echo
echo "Done. The run script will automatically pick the first two MIG UUIDs for CUDA_VISIBLE_DEVICES."
