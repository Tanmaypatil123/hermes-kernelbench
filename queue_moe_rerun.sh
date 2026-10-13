#!/bin/zsh
# Re-run moe for the two lanes whose practice tool hit the 150 s redirect bug (fixed: curl -L).
cd "$(dirname "$0")" && source env.sh
m=$1 short=$2
[[ -d ~/.hermes/skills/mlops/gpu-kernel-optimization ]] && { echo "skill is installed - move it aside first"; exit 1; }
echo "$(date +%H:%M) START ${short}_moe_rerun ($m)"
MODEL=$m python -u hermes_kernel_eval.py --tier hard --tasks moe --runs 1 --max-practice 6 --timeout 2700 \
  --run-name ${short}_moe_rerun > ${short}_moe_rerun.log 2>&1
echo "$(date +%H:%M) DONE  ${short}_moe_rerun (exit $?)"
