#!/bin/zsh
# The runs already in progress hold the old practice tool in memory. Patch each task folder's
# test_kernel.py (curl -L, and no -X POST so the redirect is fetched with GET) as soon as it exists.
while pgrep -f "hermes_kernel_eval.py" > /dev/null; do
  for f in ~/kernel_runs/*_(full|moe_rerun)/*/test_kernel.py(N); do
    if grep -q '"-X", "POST", "-H"' $f || grep -q '"curl", "-s", "-m"' $f; then
      sed -i '' -e 's/"curl", "-s", "-m"/"curl", "-s", "-L", "-m"/' -e 's/, "-X", "POST", "-H"/, "-H"/' $f && echo "$(date +%H:%M:%S) patched $f"
    fi
  done
  sleep 0.5
done
