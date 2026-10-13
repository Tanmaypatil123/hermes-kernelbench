#!/bin/zsh
# A/B arm B: the same runs as queue.sh's Hermes jobs, with the gpu-kernel-optimization skill preloaded.
# Usage: zsh queue_skill.sh easy|hard
cd "$(dirname "$0")" && source env.sh
lane=$1
[[ $lane == hard ]] && while pgrep -f "run-name kimi_hard" > /dev/null; do sleep 20; done

run() {  # run <model> <log> <command...>
  local model=$1 log=$2; shift 2
  echo "$(date +%H:%M) START $log ($model)"
  MODEL=$model "$@" > "$log" 2>&1
  echo "$(date +%H:%M) DONE  $log (exit $?)"
}
SK=(--hermes-skills gpu-kernel-optimization)
for m in deepseek/deepseek-v4.1-flash z-ai/glm-5.3-flash moonshotai/kimi-k3; do
  short=${${m#*/}%%-*}   # deepseek, glm, kimi
  if [[ $lane == easy ]]; then
    run $m ${short}_easy_skill.log python -u hermes_kernel_eval.py --runs 3 --max-practice 3 --run-name ${short}_easy_skill $SK
  else
    run $m ${short}_hard_skill.log python -u hermes_kernel_eval.py --tier hard --runs 1 --max-practice 4 --run-name ${short}_hard_skill $SK
  fi
done
echo "$(date +%H:%M) LANE $lane FINISHED"
