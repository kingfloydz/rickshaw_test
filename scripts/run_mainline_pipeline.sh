#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON="${PYTHON:-python}"
export ISAACLAB_PATH="${ISAACLAB_PATH:-$ROOT_DIR/../IsaacLab}"

FINAL_THRESHOLDS="${FINAL_THRESHOLDS:-$ROOT_DIR/config/final_thresholds.yaml}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/outputs/ablation_pipeline}"
SELECTED_RUN_ID="${SELECTED_RUN_ID:-fat2_weight_0.1}"
VIDEO_LENGTH="${VIDEO_LENGTH:-1000}"
VIDEO_NUM_ENVS="${VIDEO_NUM_ENVS:-1}"
GPU_LIST="${GPUS:-0}"
read -r -a GPU_ARGS <<< "$GPU_LIST"
RUN_LIST="${RUNS:-}"
MODE="${MODE:-all}"

pipeline_args=(
    scripts/run_ablation_pipeline.py
    --final-thresholds "$FINAL_THRESHOLDS"
    --output-dir "$OUTPUT_DIR"
    --gpus "${GPU_ARGS[@]}"
    --selected-run-id "$SELECTED_RUN_ID"
    --video-length "$VIDEO_LENGTH"
    --video-num-envs "$VIDEO_NUM_ENVS"
)

case "$MODE" in
    all) ;;
    worker)
        pipeline_args+=(--worker-only)
        ;;
    finalize)
        pipeline_args+=(--finalize-only)
        ;;
    *)
        echo "MODE must be one of: all, worker, finalize" >&2
        exit 2
        ;;
esac

if [[ "${RESUME:-1}" == "1" ]]; then
    pipeline_args+=(--resume)
fi
if [[ "${SKIP_VIDEO:-0}" == "1" ]]; then
    pipeline_args+=(--skip-video)
fi
if [[ "${SKIP_POSTPROCESS:-0}" == "1" ]]; then
    pipeline_args+=(--skip-postprocess)
fi
if [[ -n "$RUN_LIST" ]]; then
    read -r -a RUN_ARGS <<< "$RUN_LIST"
    pipeline_args+=(--runs "${RUN_ARGS[@]}")
fi
pipeline_args+=("$@")

exec "$PYTHON" "${pipeline_args[@]}"
