#!/bin/bash
#SBATCH --job-name=localopenai
#SBATCH --output=logs/localopenai-%j.out
#SBATCH --error=logs/localopenai-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=4              # matches: -n 4
#SBATCH --cpus-per-task=8        # adjust if you want more CPU per task
#SBATCH --gres=gpu:ampere_a40:4  # matches: --gres=gpu:ampere_a40:4
#SBATCH --mem=128G               # matches: --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=ai           # matches: -p ai
# ^ Adjust time/CPU if needed. All other specs mirror the provided srun command.

set -euo pipefail

# Create logs dir in case it does not exist
mkdir -p logs

echo "[batch] Job $SLURM_JOB_ID starting on $(hostname) at $(date)"
echo "[batch] Working directory: $(pwd)"

# Activate micromamba environment
if command -v micromamba >/dev/null 2>&1; then
  echo "[batch] Found micromamba: $(command -v micromamba)"
  eval "$(micromamba shell hook -s bash)"
  micromamba activate localopenai || { echo "[batch][error] micromamba env 'localopenai' not found"; exit 1; }
else
  echo "[batch][error] micromamba not found in PATH" >&2
  exit 1
fi

echo "[batch] Python: $(which python)"
python -V || true

# IMPORTANT: Under sbatch, BASH_SOURCE points to the temporary copy in /var/spool/slurm/...,
# which does NOT contain the repository files like start_server.sh, causing
# "./start_server.sh: No such file or directory". Use SLURM_SUBMIT_DIR (the dir you ran sbatch from)
# so we stay in the project root that has the script.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
echo "[batch] Submission directory: $SUBMIT_DIR"
cd "$SUBMIT_DIR"

# If you intend to use all 4 GPUs via tensor parallel, you can either pass --gpus 4 when submitting or hard-set below.
TP_OVERRIDE=""
for a in "$@"; do
  if [[ "$a" == "--gpus" || "$a" == "--gpu-count" || "$a" == "--tp" ]]; then
    TP_OVERRIDE="yes"; break
  fi
done

# Forward all args. If user didn't specify a TP size, default to 4 (all GPUs).
if [[ -z "$TP_OVERRIDE" ]]; then
  echo "[batch] No tensor parallel arg provided; defaulting to --gpus 4"
  ./start_server.sh --gpus 4 "$@"
else
  ./start_server.sh "$@"
fi

# If the server exits, keep container alive briefly for log inspection (optional)
echo "[batch] Server process ended at $(date)"
