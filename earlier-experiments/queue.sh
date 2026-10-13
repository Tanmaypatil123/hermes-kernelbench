#!/bin/zsh
# Runs one lane of the benchmark plan, one job at a time, so two jobs never share a GPU type
# (sharing would distort timings). Usage: zsh queue.sh easy|hard
# Every job pins its model with MODEL=..., so Hermes' configured default does not matter.
cd "$(dirname "$0")" && source env.sh
lane=$1
# wait for jobs started before this queue existed
if [[ $lane == easy ]]; then wait_for="kernel_eval.py --seeds 3"; else wait_for="run-name deepseek_hard"; fi
while pgrep -f "$wait_for" > /dev/null; do sleep 20; done

run() {  # run <model> <log> <command...>
  local model=$1 log=$2; shift 2
  echo "$(date +%H:%M) START $log ($model)"
  MODEL=$model "$@" > "$log" 2>&1
  echo "$(date +%H:%M) DONE  $log (exit $?)"
}

if [[ $lane == easy ]]; then
  run z-ai/glm-5.3-flash  glm_easy_hermes.log  python -u hermes_kernel_eval.py --runs 3 --max-practice 3 --run-name glm_easy
  run moonshotai/kimi-k3  kimi_easy_alone.log  python -u kernel_eval.py --seeds 3
  run moonshotai/kimi-k3  kimi_easy_hermes.log python -u hermes_kernel_eval.py --runs 3 --max-practice 3 --run-name kimi_easy
else
  run z-ai/glm-5.3-flash  glm_hard_alone.log   python -u hard_eval.py --alone --seeds 1 --run-name glm_hard_alone
  run z-ai/glm-5.3-flash  glm_hard_hermes.log  python -u hermes_kernel_eval.py --tier hard --runs 1 --max-practice 4 --run-name glm_hard
  run moonshotai/kimi-k3  kimi_hard_alone.log  python -u hard_eval.py --alone --seeds 1 --run-name kimi_hard_alone
  run moonshotai/kimi-k3  kimi_hard_hermes.log python -u hermes_kernel_eval.py --tier hard --runs 1 --max-practice 4 --run-name kimi_hard
fi
echo "$(date +%H:%M) LANE $lane FINISHED"
