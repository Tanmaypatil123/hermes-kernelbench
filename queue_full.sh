#!/bin/zsh
# Full KernelBench-Hard (all 6 RTX PRO 6000 problems) with plain Hermes - NO skill preloaded,
# and gpu-kernel-optimization moved out of ~/.hermes/skills so Hermes can't find it on its own.
# One lane per model; lanes run in parallel (every GPU call gets its own Modal container).
# Usage: zsh queue_full.sh <model id> <short name> [provider, default openrouter]
cd "$(dirname "$0")" && source env.sh
m=$1 short=$2 provider=${3:-openrouter}
[[ -d ~/.hermes/skills/mlops/gpu-kernel-optimization ]] && { echo "skill is installed - move it aside first"; exit 1; }
echo "$(date +%H:%M) START ${short}_full ($m)"
MODEL=$m python -u hermes_kernel_eval.py --tier hard --runs 1 --max-practice 6 --timeout 2700 \
  --run-name ${short}_full --provider $provider > ${short}_full.log 2>&1
echo "$(date +%H:%M) DONE  ${short}_full (exit $?)"
