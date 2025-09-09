#!/bin/bash
#SBATCH -J qwen32b-dl
#SBATCH -p ai                      # partition
#SBATCH -N 1                       # nodes
#SBATCH -n 4                       # total tasks (per your srun example)
#SBATCH --gres=gpu:ampere_a40:1    # one A40 GPU
#SBATCH --mem=128G                 # total memory
#SBATCH --cpus-per-task=1          # 1 CPU per task (adjust if needed)
#SBATCH -t 24:00:00
#SBATCH -o logs/qwen32b-dl-%j.out
#SBATCH -e logs/qwen32b-dl-%j.err

set -euo pipefail

echo "Starting Qwen 32B Instruct model snapshot download using micromamba env 'localopenai'..."

micromamba run -n localopenai python - <<'PYCODE'
from huggingface_hub import snapshot_download

# Download to default cache (~/.cache/huggingface/hub). Set cache_dir if you want a specific path.
print('Downloading Qwen/Qwen2.5-32B-Instruct ...')
local_path = snapshot_download(repo_id='Qwen/Qwen2.5-32B-Instruct', revision='main', cache_dir=None)
print('Completed. Local path:', local_path)
PYCODE

echo "Download finished."
