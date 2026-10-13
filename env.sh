# Source this once per terminal: source env.sh
# The OpenRouter key is read from Hermes' own config, so it never lives in this folder.
# Your KernelIDE executor URL goes in env.local (git-ignored), e.g.
#   export KERNELIDE_URL="https://<your-workspace>--kernelide-executor-api.modal.run"
[[ -f "$(dirname "${BASH_SOURCE[0]:-$0}")/env.local" ]] && source "$(dirname "${BASH_SOURCE[0]:-$0}")/env.local"
: "${KERNELIDE_URL:?set KERNELIDE_URL in env.local}"
export KERNELIDE_GPU="T4"
export OPENAI_BASE_URL="https://openrouter.ai/api/v1"
export OPENAI_API_KEY="$(grep -m1 '^OPENROUTER_API_KEY=' ~/.hermes/.env | cut -d= -f2- | tr -d "\"'")"
export MODEL="z-ai/glm-5.3-flash"
source "$(dirname "${BASH_SOURCE[0]:-$0}")/.venv/bin/activate"
