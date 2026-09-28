"""No-GPU integration check against the frozen SWE-Gym Docker image."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import subprocess


root = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "evaluate_openhands_candidate", root / "scripts/evaluate_openhands_candidate.py"
)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)

image = "xingyaoww/sweb.eval.x86_64.getmoto_s_moto-7646:latest"
workspace = "/testbed"
container = shell = None
try:
    container = module.start_container(
        image, "98ad497eb4de25943ceadabe7c2c5c73cc810e27", "shell-smoke", workspace
    )
    shell = module.PersistentDockerShell(container, workspace, soft_timeout=1)
    code, output = shell.execute({"command": "pwd"})
    print("initial:", code, repr(output))
    assert code == 0 and "/testbed" in output
    code, output = shell.execute({
        "command": "test -x /opt/miniconda3/envs/testbed/bin/pytest && "
                   "test -d tests/test_dynamodb && test ! -e test.py && "
                   "test \"$(command -v pytest)\" = /opt/miniconda3/envs/testbed/bin/pytest && "
                   "test \"$(command -v python)\" = /opt/miniconda3/envs/testbed/bin/python"
    })
    print("environment contract:", code, repr(output))
    assert code == 0

    code, output = shell.execute({
        "command": "find /testbed -type f -name '*.py' -exec grep -l 'update_item' {} \\;",
        "timeout": 30,
    })
    assert code == 0 and "moto/dynamodb" in output, (code, output)

    editor_history = {}
    task_list = []
    edit_path = workspace + "/route_v2_eval_smoke.txt"
    observation, _ = module.tool_result(
        container, "str_replace_editor",
        {"command": "create", "path": edit_path, "file_text": "before\n"},
        workspace, shell, editor_history, task_list,
    )
    assert observation.startswith("exit_code=0"), observation
    observation, _ = module.tool_result(
        container, "str_replace_editor",
        {"command": "str_replace", "path": edit_path,
         "old_str": "before", "new_str": "after"},
        workspace, shell, editor_history, task_list,
    )
    assert observation.startswith("exit_code=0"), observation
    assert "after" in shell.execute({"command": "cat route_v2_eval_smoke.txt"})[1]
    observation, _ = module.tool_result(
        container, "str_replace_editor", {"command": "undo_edit", "path": edit_path},
        workspace, shell, editor_history, task_list,
    )
    assert observation.startswith("exit_code=0"), observation
    assert "before" in shell.execute({"command": "cat route_v2_eval_smoke.txt"})[1]
    observation, _ = module.tool_result(
        container, "str_replace_editor", {"command": "undo_edit", "path": edit_path},
        workspace, shell, editor_history, task_list,
    )
    assert observation.startswith("exit_code=0"), observation
    assert shell.execute({"command": "test ! -e route_v2_eval_smoke.txt"})[0] == 0
    print("editor create/replace/undo: passed")

    code, output = shell.execute({"command": "cd /tmp; export ROUTE_SHELL_SMOKE=kept"})
    assert code == 0, (code, output)
    code, output = shell.execute({"command": "pwd; printf '%s' \"$ROUTE_SHELL_SMOKE\""})
    print("persistent:", code, repr(output))
    assert code == 0 and "/tmp" in output and "kept" in output

    code, output = shell.execute({"command": "sleep 3; echo completed"})
    print("soft timeout:", code, repr(output))
    assert code == -1 and "still running" in output
    assert shell.execute({"command": "pwd"})[0] == -1
    code, output = shell.execute({"command": "", "is_input": "true"})
    if code == -1:
        code, output = shell.execute({"command": "", "is_input": "true"})
    print("poll:", code, repr(output))
    assert code == 0 and "completed" in output

    code, output = shell.execute({"command": "sleep 30", "timeout": 1})
    print("hard timeout:", code, repr(output))
    assert code == 124 and "hard timeout" in output
    code, output = shell.execute({"command": "pwd"})
    print("after interrupt:", code, repr(output))
    assert code == 0 and "/tmp" in output

    code, output = shell.execute({"command": "sleep 30"})
    assert code == -1
    code, output = shell.execute({"command": "C-c", "is_input": "true"})
    print("manual Ctrl+C:", code, repr(output))
    assert code != -1
finally:
    if shell:
        shell.close()
    if container:
        subprocess.run(["docker", "rm", "-f", container], capture_output=True, check=False)
