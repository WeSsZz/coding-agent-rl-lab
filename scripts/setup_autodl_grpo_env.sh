#!/usr/bin/env bash
set -euo pipefail

BASE_PYTHON=${BASE_PYTHON:-/root/miniconda3/bin/python}
GRPO_ENV=${GRPO_ENV:-/root/autodl-tmp/grpo-env}
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)

if [[ ! -x "$GRPO_ENV/bin/python" ]]; then
  "$BASE_PYTHON" -m venv --system-site-packages "$GRPO_ENV"
fi

"$GRPO_ENV/bin/python" -m pip install --upgrade \
  'trl==1.12.0' \
  'transformers>=5.2.0' \
  'datasets>=4.7.0' \
  jmespath

SITE_PACKAGES=$(
  "$GRPO_ENV/bin/python" -c 'import site; print(site.getsitepackages()[0])'
)
install -m 0644 "$SCRIPT_DIR/grpo_sitecustomize.py" "$SITE_PACKAGES/sitecustomize.py"

"$GRPO_ENV/bin/python" -c \
  'import inspect, torch, transformers, trl; from trl import GRPOTrainer; assert "environment_factory" in inspect.signature(GRPOTrainer.__init__).parameters; print(torch.__version__, transformers.__version__, trl.__version__)'
