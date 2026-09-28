"""No-model check of the pinned SWE-Gym OpenHands Docker interaction path."""

import os
from pathlib import Path
from types import SimpleNamespace

import tenacity
from datasets import load_dataset

os.environ['USE_INSTANCE_IMAGE'] = 'true'
os.environ['RUNTIME'] = 'eventstream'
os.environ['EVAL_DOCKER_IMAGE_PREFIX'] = 'docker.io/xingyaoww'

from evaluation.swe_bench.run_infer import get_config, initialize_runtime  # noqa: E402
from openhands.core.config import LLMConfig  # noqa: E402
from openhands.core.main import create_runtime  # noqa: E402
from openhands.events.action import CmdRunAction  # noqa: E402
from openhands.utils.async_utils import call_async_from_sync  # noqa: E402


task_id = os.environ.get('SWEGYM_TASK_ID', 'getmoto__moto-7646')
dataset = load_dataset('SWE-Gym/SWE-Gym', split='train')
rows = dataset.filter(lambda row: row['instance_id'] == task_id)
assert len(rows) == 1, (task_id, len(rows))
instance = rows.to_pandas().iloc[0]
# Preserve the fork's pandas.Series access while making Arrow-backed arrays
# JSON serializable in initialize_runtime's instance metadata file.
for key, value in instance.items():
    if hasattr(value, 'tolist'):
        instance[key] = value.tolist()
output_dir = Path('/home/wesz/swegym-runtime-probe')
output_dir.mkdir(exist_ok=True)
metadata = SimpleNamespace(
    agent_class='CodeActAgent',
    max_iterations=50,
    llm_config=LLMConfig(model='openai/no-model-probe', api_key='EMPTY'),
    eval_output_dir=str(output_dir),
)
config = get_config(instance, metadata)
# Docker bridge port-forwarding on this VM does not let the host reach a
# container's /alive endpoint. Host networking is the VM-specific workaround.
config.sandbox.use_host_network = True
runtime = create_runtime(config)
# The first cold runtime start on this low-CPU VM took longer than the fork's
# default 120-second readiness retry. Keep the upstream fork unchanged while
# allowing this no-model probe enough time to validate the startup path.
runtime._wait_until_alive.retry.stop = tenacity.stop_after_delay(300)
try:
    call_async_from_sync(runtime.connect)
    initialize_runtime(runtime, instance)
    obs = runtime.run_action(CmdRunAction(command='pwd && git rev-parse HEAD'))
    assert obs.exit_code == 0, str(obs)
    assert str(instance['base_commit']) in obs.content, str(obs)
    print(f'OpenHands runtime passed: {task_id} @ {instance["base_commit"]}')
finally:
    runtime.close()
