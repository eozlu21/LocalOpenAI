#!/bin/bash
set -euo pipefail

MODEL=""
GPUS=""
PORT=8000

while [[ $# -gt 0 ]]; do
	case "$1" in
		--model)
			MODEL="$2"; shift 2 ;;
		--gpus|--gpu-count|--tp)
			GPUS="$2"; shift 2 ;;
		--port)
			PORT="$2"; shift 2 ;;
		-h|--help)
			echo "Usage: $0 [--model <hf_model_name>] [--gpus <tensor_parallel_size>] [--port <port>]";
			exit 0 ;;
		*)
			echo "Unknown arg: $1" >&2; exit 1 ;;
	esac
done

export PYTHONPATH=.
if [[ -n "$MODEL" ]]; then
	export LOCAL_OPENAI_MODEL="$MODEL"
	echo "[start] Model: $MODEL"
fi
if [[ -n "$GPUS" ]]; then
	export LOCAL_OPENAI_TP="$GPUS"
	echo "[start] Tensor parallel size: $GPUS"
fi
echo "Starting server on port $PORT"
uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --workers 1
