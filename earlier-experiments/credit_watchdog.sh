#!/bin/zsh
# Stops every benchmark job if Modal (KernelIDE) looks out of credit or unreachable.
# Signals: /health (no GPU, free) failing twice in a row, or an infra/billing error in a practice log.
cd "$(dirname "$0")"
URL="https://<your-workspace>--kernelide-executor-api.modal.run/health"
fails=0
stop_all() {
  echo "$(date +%H:%M:%S) STOPPING: $1"
  pkill -f queue_skill.sh; pkill -f queue.sh; pkill -f hermes_kernel_eval.py; pkill -f hard_eval.py
  pkill -f regrade_hard.py; pkill -f "hermes -z"
  echo "$(date +%H:%M:%S) STOPPED all benchmark jobs"
  exit 0
}
while pgrep -f "hermes_kernel_eval.py|hard_eval.py|regrade_hard.py|queue_skill.sh" > /dev/null; do
  code=$(curl -s -o /tmp/claude-501/kide_health.txt -w "%{http_code}" -m 30 "$URL")
  if [[ $code == 200 ]]; then fails=0; else fails=$((fails+1)); echo "$(date +%H:%M:%S) health check HTTP $code (fail $fails)"; fi
  (( fails >= 2 )) && stop_all "KernelIDE /health failed twice (HTTP $code): $(head -c 200 /tmp/claude-501/kide_health.txt)"
  if grep -h -i -E '"status": "infra_error".*(billing|credit|quota|limit|payment|insufficient|disabled|suspended|402|429)' \
       ~/kernel_runs/kimi_hard_skill/*/.practice_log.jsonl 2>/dev/null | head -1 | grep -q .; then
    stop_all "billing/credit error in a practice run"
  fi
  if grep -h -i -E 'billing|out of credit|insufficient|payment required|spend limit' kimi_hard_skill.log regrade.log 2>/dev/null | head -1 | grep -q .; then
    stop_all "billing/credit error in a run log"
  fi
  sleep 60
done
echo "$(date +%H:%M:%S) no benchmark jobs left; watchdog exiting"
