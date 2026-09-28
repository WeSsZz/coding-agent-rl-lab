"""Run one OpenHands-style SWE-Gym rollout and verify its patch in a clean container.

The policy container never receives the verifier test patch.  Its candidate patch
also excludes every path owned by that patch.  Verification starts from a second
container at the frozen base commit, applies the candidate and private test patch,
then runs only the audited FAIL_TO_PASS/PASS_TO_PASS nodes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
import urllib.request
import urllib.error
import uuid


TOOLS = json.loads((Path(__file__).resolve().parents[1] / "envs/openhands-v0.54-tools.json").read_text(encoding="utf-8"))


def run(command: list[str], *, input_text: str | None = None, timeout: int = 120, check: bool = True):
    return subprocess.run(command, input=input_text, text=True, capture_output=True,
                          timeout=timeout, check=check)


def clip_output(value: str, limit: int) -> str:
    if len(value) <= limit:
        return value
    front = limit // 2
    back = limit - front
    return (value[:front] + f"\n[... {len(value) - limit} characters clipped ...]\n"
            + value[-back:])


def docker_exec(name: str, command: str, timeout: int = 120,
                workdir: str = "/testbed", limit_chars: int = 5000) -> tuple[int, str]:
    # Timeout must run inside the container so it also terminates a child pytest
    # process.  A host-side subprocess timeout only kills the docker CLI.
    result = run(["docker", "exec", "-w", workdir, name, "timeout",
                  "--signal=TERM", "--kill-after=5s", str(timeout) + "s",
                  "bash", "-lc", command], timeout=timeout + 15, check=False)
    text = clip_output(result.stdout + result.stderr, limit_chars)
    return result.returncode, text


def api_chat(endpoint: str, model: str, messages: list[dict], *, non_thinking: bool = False,
             failed_request_path: Path | None = None, max_output_tokens: int = 2048,
             tool_choice: str = "required") -> dict:
    payload = {"model": model, "messages": messages, "tools": TOOLS,
               "tool_choice": tool_choice, "temperature": 0.0,
               "max_tokens": max_output_tokens}
    if non_thinking:
        payload["chat_template_kwargs"] = {"enable_thinking": False}
    body = json.dumps(payload).encode()
    request = urllib.request.Request(endpoint.rstrip("/") + "/v1/chat/completions", body,
                                     {"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            payload = json.load(response)
    except urllib.error.HTTPError as exc:
        if failed_request_path is not None:
            failed_request_path.parent.mkdir(parents=True, exist_ok=True)
            failed_request_path.write_bytes(body)
        raise RuntimeError(f"inference HTTP {exc.code}: {exc.read(2000).decode(errors='replace')}") from exc
    return payload["choices"][0]["message"]


def patch_paths(patch: str) -> list[str]:
    return sorted(set(re.findall(r"^diff --git a/(.+?) b/", patch, flags=re.MULTILINE)))


class PersistentDockerShell:
    """One terminal for the policy's entire rollout, as advertised by execute_bash."""

    def __init__(self, container: str, workspace: str, soft_timeout: int = 10):
        try:
            import pexpect
        except ImportError as exc:
            raise RuntimeError("pexpect is required for the persistent evaluation shell") from exc
        self.pexpect = pexpect
        self.container = container
        self.soft_timeout = soft_timeout
        self.child = pexpect.spawn(
            "docker", ["exec", "-it", "-w", workspace, container,
                       "bash", "--noprofile", "--norc", "-i"],
            encoding="utf-8", codec_errors="replace", echo=False,
            timeout=soft_timeout,
        )
        self.marker: str | None = None
        self.output = ""
        self.child.sendline(
            "stty -echo; bind 'set enable-bracketed-paste off'; "
            "export PS1=''; export PROMPT_COMMAND=''; "
            "export PATH=/opt/miniconda3/envs/testbed/bin:$PATH; "
            "export CONDA_DEFAULT_ENV=testbed"
        )
        ready = "__SHELL_READY_" + uuid.uuid4().hex + "__"
        self.child.sendline(f"printf '\\n{ready}:%s\\n' \"$$\"")
        self.child.expect(re.escape(ready) + r":(\d+)", timeout=15)
        self.shell_pid = int(self.child.match.group(1))

    def _interrupt(self) -> None:
        # Docker's attached TTY does not reliably relay Ctrl+C to the foreground
        # process. Signal the child of this exact shell from inside the container.
        result = run(["docker", "exec", self.container, "pkill", "-INT", "-P",
                      str(self.shell_pid)], timeout=10, check=False)
        if result.returncode:
            self.child.sendcontrol("c")

    def _wait(self, seconds: int) -> tuple[int, str]:
        assert self.marker is not None
        try:
            self.child.expect(re.escape(self.marker) + r":(-?\d+)", timeout=seconds)
        except self.pexpect.TIMEOUT:
            self.output = clip_output(self.output + self.child.before, 3000)
            return -1, self.output + "\n[command is still running; use is_input=true to poll or interrupt]"
        except self.pexpect.EOF:
            self.output = clip_output(self.output + self.child.before, 3000)
            self.marker = None
            return -1, self.output + "\n[shell exited]"
        self.output = clip_output(self.output + self.child.before, 3000)
        code, output = int(self.child.match.group(1)), self.output.strip("\r\n")
        self.marker = None
        self.output = ""
        return code, output

    def execute(self, arguments: dict) -> tuple[int, str]:
        command = str(arguments.get("command", ""))
        is_input = str(arguments.get("is_input", "false")).lower() == "true"
        if self.marker is not None:
            if not is_input:
                return -1, "Previous command is still running; use is_input=true to poll or interrupt it."
            if command == "C-c":
                self._interrupt()
            elif command == "C-d":
                self.child.sendcontrol("d")
            elif command == "C-z":
                self.child.sendcontrol("z")
            elif command:
                self.child.sendline(command)
            return self._wait(self.soft_timeout)
        if is_input:
            return 1, "No command is running; issue a new execute_bash command."
        if not command.strip():
            return 1, "Empty command; issue a bash command."
        self.marker = "__COMMAND_DONE_" + uuid.uuid4().hex + "__"
        self.child.sendline(command)
        self.child.sendline(f"printf '\\n{self.marker}:%s\\n' \"$?\"")
        requested = arguments.get("timeout")
        seconds = self.soft_timeout if requested is None else min(max(int(requested), 1), 120)
        code, output = self._wait(seconds)
        if code == -1 and requested is not None and self.marker is not None:
            self._interrupt()
            interrupted, tail = self._wait(5)
            return 124 if interrupted != -1 else -1, tail + f"\n[hard timeout after {seconds}s]"
        return code, output

    def close(self) -> None:
        if self.child.isalive():
            self.child.close(force=True)


def repeated_command(previous: str | None, count: int,
                     tool: str, arguments: dict, observation: str) -> tuple[str | None, int]:
    if tool == "task_tracker":
        if arguments.get("command") == "view":
            return "task_tracker:view", count + 1 if previous == "task_tracker:view" else 1
        return None, 0
    if tool == "str_replace_editor" and str(arguments.get("command")) in {
            "create", "str_replace", "insert", "undo_edit"} and observation.startswith("exit_code=0"):
        return None, 0
    if tool != "execute_bash" or str(arguments.get("is_input", "false")).lower() == "true":
        return previous, count
    match = re.match(r"exit_code=(-?\d+)", observation)
    if not match or int(match.group(1)) == -1:
        return previous, count
    command = str(arguments.get("command", ""))
    return command, count + 1 if command == previous else 1


def search_feedback(observation: str, previous: str | None,
                    count: int) -> tuple[str, str | None, int, bool]:
    body = re.sub(r"^exit_code=-?\d+\n", "", observation).strip()
    if not body or body == previous:
        count = count + 1 if body == previous else 1
    else:
        count = 1
    previous = body
    paths = [line.strip() for line in body.splitlines() if line.strip()]
    if paths and all(line.startswith("/testbed/") for line in paths):
        observation += "\n[Search returned file paths. Read a relevant source file before searching again.]"
    if count >= 2:
        observation += "\n[This action produced no new information. Read source contents or change strategy.]"
    if count >= 4:
        observation += "\n[Stopped after four actions with no new information.]"
    return observation, previous, count, count >= 4


def compact_messages(messages: list[dict], keep_turns: int = 6) -> list[dict]:
    groups: list[list[dict]] = []
    for message in messages[2:]:
        if message["role"] == "assistant" or not groups:
            groups.append([message])
        else:
            groups[-1].append(message)
    return messages[:2] + [message for group in groups[-keep_turns:] for message in group]


def view_feedback(name: str, arguments: dict, observation: str,
                  previous_path: str | None, count: int) -> tuple[str, str | None, int, bool]:
    if name != "str_replace_editor" or arguments.get("command") != "view":
        return observation, None, 0, False
    path = str(arguments.get("path", ""))
    count = count + 1 if path == previous_path else 1
    if count >= 2:
        observation += ("\n[You have already read this file. Do not view it again. "
                        "Use the source context to edit, run a focused test, or inspect a different file.]")
    if count >= 4:
        observation += "\n[Stopped after four consecutive views of the same file.]"
    return observation, path, count, count >= 4


def tool_result(container: str, name: str, arguments: dict,
                workspace: str, shell: PersistentDockerShell,
                editor_history: dict[str, list[tuple[str, bool]]],
                task_list: list[dict]) -> tuple[str, bool]:
    if name == "think":
        return "Thought recorded.", False
    if name == "task_tracker":
        if arguments.get("command") == "plan":
            proposed = arguments.get("task_list")
            if not isinstance(proposed, list):
                return "Error: plan requires task_list; no changes recorded.", False
            task_list[:] = proposed
        elif arguments.get("command") != "view":
            return "Error: task_tracker command must be plan or view.", False
        return json.dumps({"task_list": task_list}, ensure_ascii=False), False
    if name == "finish":
        return arguments.get("message", "Finished."), True
    if name == "execute_bash":
        code, output = shell.execute(arguments)
        return f"exit_code={code}\n{output}", False
    if name != "str_replace_editor":
        return "Unknown tool.", False
    command, path = arguments.get("command"), arguments.get("path", "")
    if not isinstance(path, str):
        return f"Error: path must be under {workspace}", False
    if path == workspace or path.startswith(workspace + "/"):
        relative = os.path.relpath(path, workspace)
    elif path == "/testbed" or path.startswith("/testbed/"):
        relative = os.path.relpath(path, "/testbed")
    else:
        return f"Error: path must be under {workspace}", False
    if relative == ".." or relative.startswith("../"):
        return f"Error: path must be under {workspace}", False
    if command == "undo_edit":
        history = editor_history.get(relative, [])
        if not history:
            return "Error: no editor change to undo for this path", False
        backup, existed = history[-1]
        script = """import json, pathlib, shutil, sys
a=json.loads(sys.stdin.read()); p=pathlib.Path(a['path']); b=pathlib.Path(a['backup'])
if a['existed']: shutil.copyfile(b,p)
else: p.unlink()
b.unlink(missing_ok=True)
print('Done')
"""
        result = run(["docker", "exec", "-i", container, "python3", "-c", script],
                     input_text=json.dumps({"path": "/testbed/" + relative,
                                            "backup": backup, "existed": existed}),
                     check=False)
        if result.returncode == 0:
            history.pop()
        return f"exit_code={result.returncode}\n{result.stdout}{result.stderr}"[-5000:], False
    if command == "view":
        # Keep broad reads focused; returning 400 lines caused the output clipper
        # to discard useful context and encourage repeated whole-file reads.
        start, end = (arguments.get("view_range") or [1, 120])[:2]
        end_expr = "$" if end == -1 else str(end)
        target = shlex.quote(relative)
        shell = (f"set -o pipefail; if [ -d {target} ]; then find {target} -mindepth 1 -maxdepth 2 "
                 f"-not -path '*/.*' | sort | head -400; "
                 f"else nl -ba -- {target} | sed -n '{int(start)},{end_expr}p'; fi")
        code, output = docker_exec(container, shell)
        return f"exit_code={code}\n{output}", False
    if command in {"create", "str_replace", "insert"}:
        backup = "/tmp/route-v2-editor-undo-" + uuid.uuid4().hex
        snapshot = """import json, pathlib, shutil, sys
a=json.loads(sys.stdin.read()); p=pathlib.Path(a['path']); b=pathlib.Path(a['backup'])
if p.exists(): shutil.copyfile(p,b); print('1')
else: print('0')
"""
        payload = {"path": "/testbed/" + relative, "backup": backup}
        before = run(["docker", "exec", "-i", container, "python3", "-c", snapshot],
                     input_text=json.dumps(payload), check=False)
        if before.returncode:
            return f"exit_code={before.returncode}\n{before.stdout}{before.stderr}"[-5000:], False
        existed = before.stdout.strip() == "1"
        script = """import json, pathlib, sys
a=json.loads(sys.stdin.read()); p=pathlib.Path(a['path'])
cmd=a['command']
if cmd=='create':
    if p.exists(): raise SystemExit('file already exists')
    p.parent.mkdir(parents=True, exist_ok=True); p.write_text(a['file_text'])
elif cmd=='str_replace':
    s=p.read_text(); old=a['old_str']
    if s.count(old)!=1: raise SystemExit('old_str must match exactly once')
    p.write_text(s.replace(old,a.get('new_str',''),1))
else:
    lines=p.read_text().splitlines(keepends=True); n=int(a['insert_line'])
    lines[n:n]=[a['new_str'] + ('' if a['new_str'].endswith('\\n') else '\\n')]
    p.write_text(''.join(lines))
print('Done')
"""
        payload = dict(arguments, path="/testbed/" + relative)
        result = run(["docker", "exec", "-i", container, "python3", "-c", script],
                     input_text=json.dumps(payload), check=False)
        if result.returncode == 0:
            editor_history.setdefault(relative, []).append((backup, existed))
        return f"exit_code={result.returncode}\n{result.stdout}{result.stderr}"[-5000:], False
    return "Error: unsupported editor command", False


def start_container(image: str, base_commit: str, label: str,
                    workspace: str) -> str:
    name = f"route-v2-{label}-{uuid.uuid4().hex[:8]}"
    run(["docker", "run", "-d", "--rm", "--network", "none", "--name", name,
         image, "sleep", "infinity"])
    code, output = docker_exec(name, f"git reset --hard {base_commit} && git clean -fdx")
    if code:
        raise RuntimeError(output)
    if workspace != "/testbed":
        code, output = docker_exec(name, f"mkdir -p /workspace && ln -sfn /testbed {shlex.quote(workspace)}")
        if code:
            raise RuntimeError(output)
    return name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", type=Path, required=True)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt-template", type=Path, required=True)
    parser.add_argument("--non-thinking", action="store_true",
                        help="render Qwen3 with enable_thinking=false, matching its SFT data")
    parser.add_argument("--max-steps", type=int, default=24)
    parser.add_argument("--max-wall-seconds", type=int, default=600)
    parser.add_argument("--max-output-tokens", type=int, default=2048)
    parser.add_argument("--tool-choice", choices=("auto", "required"), default="required")
    args = parser.parse_args()
    row = json.loads(args.task.read_text(encoding="utf-8"))
    template = json.loads(args.prompt_template.read_text(encoding="utf-8"))
    repo_dir = row["repo"].replace("/", "__") + "__" + row["version"]
    workspace = "/testbed"
    user_prompt = (template["prefix"] + row["problem_statement"] + template["suffix"])
    user_prompt = user_prompt.replace(template["source_workspace"], workspace)
    user_prompt = user_prompt.replace(template["source_repo_dir"], repo_dir)
    user_prompt = user_prompt.replace(template["source_base_commit"], row["base_commit"])
    image = "xingyaoww/sweb.eval.x86_64." + row["instance_id"].replace("__", "_s_").lower() + ":latest"
    forbidden = patch_paths(row["test_patch"])
    policy = verifier = shell = None
    messages = [{"role": "system", "content": template["system"]},
                {"role": "user", "content": user_prompt}]
    trace, started = [], time.time()
    editor_history: dict[str, list[tuple[str, bool]]] = {}
    task_list: list[dict] = []
    previous_failed, repeat_count, stop_reason = None, 0, "step_budget"
    previous_observation, same_observation_count = None, 0
    previous_view_path, same_view_count = None, 0
    try:
        policy = start_container(image, row["base_commit"], "policy", workspace)
        shell = PersistentDockerShell(policy, workspace)
        for step in range(args.max_steps):
            if time.time() - started >= args.max_wall_seconds:
                trace.append({"step": step, "error": "wall_time_budget_exceeded"})
                stop_reason = "wall_time_budget_exceeded"
                break
            try:
                messages = compact_messages(messages)
                message = api_chat(args.endpoint, args.model, messages,
                                   non_thinking=args.non_thinking,
                                   failed_request_path=args.output.with_suffix(".failed-request.json"),
                                   max_output_tokens=args.max_output_tokens,
                                   tool_choice=args.tool_choice)
            except Exception as exc:
                trace.append({"step": step, "error": str(exc)[:3000]})
                stop_reason = "inference_error"
                break
            assistant = {k: v for k, v in message.items() if k in {"role", "content", "tool_calls"}}
            # Preserve a valid tool-call history even when a model emits malformed arguments.
            invalid_calls = {}
            for call in assistant.get("tool_calls") or []:
                raw_arguments = call["function"].get("arguments") or "{}"
                try:
                    json.loads(raw_arguments)
                except (TypeError, ValueError):
                    invalid_calls[call["id"]] = str(raw_arguments)[:1000]
                    call["function"]["arguments"] = "{}"
            messages.append(assistant)
            calls = message.get("tool_calls") or []
            if not calls:
                trace.append({"step": step, "assistant": assistant, "error": "no_tool_call"})
                stop_reason = "no_tool_call"
                break
            finished = repeat_stop = False
            for call in calls:
                fn = call["function"]
                arguments = {}
                try:
                    arguments = json.loads(fn.get("arguments") or "{}")
                    if call["id"] in invalid_calls:
                        raise ValueError("model emitted malformed tool arguments")
                    observation, done = tool_result(policy, fn["name"], arguments,
                                                    workspace, shell, editor_history, task_list)
                except Exception as exc:
                    observation, done = f"Tool error: {type(exc).__name__}: {exc}", False
                previous_failed, repeat_count = repeated_command(
                    previous_failed, repeat_count, fn["name"], arguments, observation)
                if fn["name"] == "execute_bash":
                    observation, previous_observation, same_observation_count, no_progress_stop = search_feedback(
                        observation, previous_observation, same_observation_count)
                else:
                    previous_observation, same_observation_count, no_progress_stop = None, 0, False
                observation, previous_view_path, same_view_count, view_stop = view_feedback(
                    fn["name"], arguments, observation, previous_view_path, same_view_count)
                if repeat_count == 2:
                    observation += ("\n[The same action produced no new progress twice. "
                                    "Inspect the repository or change approach.]")
                if repeat_count >= 3:
                    observation += "\n[Stopped after three identical actions.]"
                    repeat_stop = True
                repeat_stop = repeat_stop or no_progress_stop or view_stop
                messages.append({"role": "tool", "tool_call_id": call["id"],
                                 "name": fn["name"], "content": observation})
                trace.append({"step": step, "tool": fn["name"], "arguments": arguments,
                              "invalid_arguments": invalid_calls.get(call["id"]),
                              "observation": observation, "finished": done})
                finished = finished or done
                if repeat_stop:
                    break
            if finished:
                stop_reason = "finish"
                break
            if repeat_stop:
                stop_reason = "repeated_or_uninformative_action"
                break
        # Include new source files in the patch.  Intent-to-add leaves the worktree
        # untouched while making untracked files visible to git diff.
        run(["docker", "exec", "-w", "/testbed", policy, "git", "add", "-N", "--all"])
        excludes = [f":(exclude){path}" for path in forbidden]
        excludes.extend([":(exclude)tests/**", ":(exclude)**/tests/**"])
        result = run(["docker", "exec", "-w", "/testbed", policy, "git", "diff", "--binary",
                      "--", ".", *excludes], check=False)
        candidate = result.stdout
        verifier = start_container(image, row["base_commit"], "verify", workspace)
        with tempfile.TemporaryDirectory() as temp:
            candidate_path = Path(temp) / "candidate.patch"
            tests_path = Path(temp) / "tests.patch"
            candidate_path.write_text(candidate, encoding="utf-8")
            tests_path.write_text(row["test_patch"], encoding="utf-8")
            run(["docker", "cp", str(candidate_path), f"{verifier}:/tmp/candidate.patch"])
            run(["docker", "cp", str(tests_path), f"{verifier}:/tmp/tests.patch"])
        apply_candidate = docker_exec(verifier, "git apply --index /tmp/candidate.patch" if candidate else "true")
        apply_tests = docker_exec(verifier, "git apply /tmp/tests.patch")
        targets = list(dict.fromkeys(row["FAIL_TO_PASS"] + row["PASS_TO_PASS"]))
        quoted = " ".join(json.dumps(value) for value in targets)
        test_command = f"/opt/miniconda3/envs/testbed/bin/pytest -n0 -rA -- {quoted}"
        test_result = docker_exec(verifier, test_command, timeout=300, limit_chars=16000) if apply_candidate[0] == apply_tests[0] == 0 else (-1, "patch apply failed")
        report = {
            "schema": "openhands-clean-verifier-v1", "instance_id": row["instance_id"],
            "model": args.model, "base_commit": row["base_commit"], "steps": len(trace),
            "non_thinking": args.non_thinking,
            "max_output_tokens": args.max_output_tokens,
            "tool_choice": args.tool_choice,
            "history_turns_kept": 6,
            "tool_runtime": "persistent-docker-pty-v3-testbed-clipped", "stop_reason": stop_reason,
            "workspace": workspace,
            "prompt_template_sha256": hashlib.sha256(args.prompt_template.read_bytes()).hexdigest(),
            "candidate_patch_sha256": hashlib.sha256(candidate.encode()).hexdigest(),
            "candidate_patch_bytes": len(candidate.encode()), "excluded_test_paths": forbidden,
            "candidate_applied": apply_candidate[0] == 0, "test_patch_applied": apply_tests[0] == 0,
            "test_exit_code": test_result[0], "resolved": test_result[0] == 0,
            "test_output": test_result[1], "duration_seconds": round(time.time() - started, 3),
        }
        args.output.mkdir(parents=True, exist_ok=False)
        (args.output / "candidate.patch").write_text(candidate, encoding="utf-8")
        (args.output / "trace.json").write_text(json.dumps(trace, indent=2) + "\n", encoding="utf-8")
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(report, indent=2))
    finally:
        if shell:
            shell.close()
        for container in (policy, verifier):
            if container:
                run(["docker", "rm", "-f", container], check=False)


if __name__ == "__main__":
    main()
