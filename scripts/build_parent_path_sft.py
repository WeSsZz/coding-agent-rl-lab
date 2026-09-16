from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from coding_agent_rl_lab.grpo_remote import add_parent_path_evidence
from coding_agent_rl_lab.grpo_train import bare_json_system_prompt


_PATH_PATTERN = re.compile(r"(?m)^PARENT_IMPLEMENTATION_PATH:(.+)$")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--input-report", required=True)
    parser.add_argument("--trace", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", required=True)
    args = parser.parse_args()

    rows = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    source_by_task = {}
    for row in rows:
        source_by_task.setdefault(row["task_id"], row)
    sessions = defaultdict(list)
    for trace in (json.loads(line) for line in Path(args.trace).read_text().splitlines() if line):
        action = trace.get("request", {}).get("action")
        if action and "/sessions/" in trace.get("path", ""):
            session_id = trace["path"].split("/sessions/", 1)[1].split("/", 1)[0]
            sessions[(trace["task_id"], session_id)].append(trace)

    selected = []
    seen = set()
    for (task_id, _), steps in sessions.items():
        for current, following in zip(steps, steps[1:]):
            action = current["request"]["action"]
            target = following["request"]["action"]
            if action["kind"] != "read_file" or target["kind"] != "read_file":
                continue
            observation = add_parent_path_evidence(
                current["response"]["observation"], action["arguments"]["path"]
            )
            if target["arguments"]["path"] not in _PATH_PATTERN.findall(observation):
                continue
            source = source_by_task[task_id]
            messages = [dict(source["messages"][0]), dict(source["messages"][1])]
            messages[0]["content"] = bare_json_system_prompt(
                messages[0]["content"], navigation_first=True
            )
            messages.extend([
                {"role": "assistant", "content": json.dumps(
                    {"name": action["kind"], "arguments": action["arguments"]},
                    separators=(",", ":"),
                )},
                {"role": "tool", "name": action["kind"], "content": observation},
                {"role": "assistant", "content": json.dumps(
                    {"name": target["kind"], "arguments": target["arguments"]},
                    separators=(",", ":"),
                )},
            ])
            row = dict(source)
            row.update({
                "messages": messages,
                "source_path": target["arguments"]["path"],
                "target_action": target,
                "target_tool_call": {"name": target["kind"], "arguments": target["arguments"]},
                "navigation_source": "parent-path",
                "prefix_actions": 1,
            })
            identity = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
            if identity in seen:
                continue
            seen.add(identity)
            row["example_id"] = "parent-path-sft-" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            selected.append(row)

    report = json.loads(Path(args.input_report).read_text())
    report.update({
        "example_count": len(selected),
        "task_ids": sorted({row["task_id"] for row in selected}),
        "task_count": len({row["task_id"] for row in selected}),
        "source_counts": {"parent-path": len(selected)},
        "derivation": "real imported-parent path followed by the official train-only read action",
    })
    Path(args.output).write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected), encoding="utf-8"
    )
    Path(args.report).write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
