"""Render the pinned Qwen chat template and assert the invariants this project depends on.

This runs without a GPU and without `transformers`: the template is Jinja, so rendering it
directly shows exactly what text the model will be asked to continue. That matters because the
SFT rendering and the RL rollout must produce the same token stream; if the template is only
exercised on the GPU box, a mismatch is discovered after the money is spent.

What is checked, and why each one is load-bearing:

1. With `tools`, the system turn is synthesized by the template and embeds the tool schemas
   verbatim. If a schema is dropped or renamed, the model is being asked for tools it was not
   told about.
2. `enable_thinking=False` appends an **empty** `<think></think>` block to the generation prompt.
   That is the mechanical meaning of "non-thinking mode". The flag has to be passed identically
   in SFT rendering and in RL rollout, or the two token streams differ.
3. An assistant turn with `tool_calls` renders as `<tool_call>{"name": ..., "arguments": ...}
   </tool_call>`, and a `tool` turn renders as a `user` turn wrapping `<tool_response>`.
   Consecutive tool turns are merged into one user turn.
4. The observation text survives byte-for-byte inside `<tool_response>`. The environment's
   output is the training signal; silently rewriting it would train on a different world.

Usage:
    python scripts/preflight_chat_template.py \
        --template envs/qwen3-8b-chat-template.jinja \
        --tools envs/openhands-v0.54-tools.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class PreflightError(RuntimeError):
    pass


def build_environment() -> "object":
    try:
        from jinja2 import Environment
        from jinja2 import StrictUndefined
    except ImportError as exc:  # pragma: no cover - depends on the host
        raise PreflightError("jinja2 is required to render the pinned template") from exc

    environment = Environment(undefined=StrictUndefined, trim_blocks=False, lstrip_blocks=False)
    environment.filters["tojson"] = lambda value, **_: json.dumps(value, ensure_ascii=False)
    environment.globals["raise_exception"] = lambda message: (_ for _ in ()).throw(
        PreflightError(str(message))
    )
    return environment


def render(template_source: str, messages: list[dict], tools: list[dict] | None, *, enable_thinking: bool) -> str:
    environment = build_environment()
    template = environment.from_string(template_source)
    try:
        return template.render(
            messages=messages,
            tools=tools,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except PreflightError:
        raise
    except Exception as exc:  # jinja2 raises its own types; callers get one stable one
        raise PreflightError(f"template rendering failed: {type(exc).__name__}: {exc}") from exc


def check(name: str, condition: bool, detail: str, failures: list[str]) -> None:
    if condition:
        print(f"  ok    {name}")
    else:
        print(f"  FAIL  {name}: {detail}")
        failures.append(f"{name}: {detail}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, default=REPOSITORY_ROOT / "envs/qwen3-8b-chat-template.jinja")
    parser.add_argument("--tools", type=Path, default=REPOSITORY_ROOT / "envs/openhands-v0.54-tools.json")
    parser.add_argument("--show", action="store_true", help="print each rendered prompt")
    arguments = parser.parse_args(argv)

    template_source = arguments.template.read_text(encoding="utf-8")
    tools = json.loads(arguments.tools.read_text(encoding="utf-8"))
    tool_names = [entry["function"]["name"] for entry in tools]
    failures: list[str] = []

    system = {
        "role": "system",
        "content": "You are a software engineer. Fix the issue in the repository.",
    }

    print("scenario 1: first turn, tools present, non-thinking")
    first = render(
        template_source,
        [system, {"role": "user", "content": "The tests fail with a KeyError in parser.py."}],
        tools,
        enable_thinking=False,
    )
    check("system turn is synthesized with a tools block", "# Tools" in first, first[:200], failures)
    check("tool schemas are embedded", "<tools>" in first and "</tools>" in first, "", failures)
    for name in tool_names:
        check(f"schema for {name} is present", f'"name": "{name}"' in first, "", failures)
    check(
        "generation prompt opens an empty thinking block",
        first.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        repr(first[-60:]),
        failures,
    )
    if arguments.show:
        print(first)

    print("scenario 2: thinking enabled differs only by the block")
    thinking = render(
        template_source,
        [system, {"role": "user", "content": "The tests fail with a KeyError in parser.py."}],
        tools,
        enable_thinking=True,
    )
    check(
        "thinking prompt ends at the assistant header",
        thinking.endswith("<|im_start|>assistant\n"),
        repr(thinking[-40:]),
        failures,
    )
    check(
        "the two prompts differ exactly by the empty thinking block",
        thinking + "<think>\n\n</think>\n\n" == first,
        "the non-thinking prompt is not the thinking prompt plus an empty block",
        failures,
    )

    print("scenario 3: multi-turn with a tool call and an observation")
    observation = "Traceback (most recent call last):\n  File \"parser.py\", line 12\nKeyError: 'name'"
    conversation = [
        system,
        {"role": "user", "content": "The tests fail with a KeyError in parser.py."},
        {
            "role": "assistant",
            "content": "Let me look at the failing test.",
            "tool_calls": [
                {
                    "type": "function",
                    "function": {"name": "execute_bash", "arguments": {"command": "pytest -x -q"}},
                }
            ],
        },
        {"role": "tool", "name": "execute_bash", "tool_call_id": "call_1", "content": observation},
    ]
    multi = render(template_source, conversation, tools, enable_thinking=False)
    check("<tool_call> is emitted", "<tool_call>" in multi, "", failures)
    check("the tool name reaches the call", '"name": "execute_bash"' in multi, "", failures)
    check(
        "arguments are serialized as a JSON object, not a string",
        '"arguments": {"command": "pytest -x -q"}' in multi,
        "arguments were stringified, which changes the token stream",
        failures,
    )
    check("the observation is wrapped in <tool_response>", "<tool_response>" in multi, "", failures)
    check(
        "the observation survives byte-for-byte",
        observation in multi,
        "the rendered prompt does not contain the observation verbatim",
        failures,
    )
    if arguments.show:
        print(multi)

    print("scenario 4: consecutive tool turns merge into one user turn")
    merged = render(
        template_source,
        conversation
        + [
            {"role": "tool", "name": "execute_bash", "tool_call_id": "call_2", "content": "second"},
        ],
        tools,
        enable_thinking=False,
    )
    check(
        "both responses are inside one user turn",
        merged.count("<|im_start|>user") == 2,
        f"expected 2 user turns (the issue and the merged tool block), got {merged.count('<|im_start|>user')}",
        failures,
    )
    check(
        "the merged turn closes once",
        merged.endswith("<|im_start|>assistant\n<think>\n\n</think>\n\n"),
        repr(merged[-60:]),
        failures,
    )

    print("scenario 5: without tools the template does not invent a tools block")
    plain = render(template_source, [system, {"role": "user", "content": "hello"}], None, enable_thinking=False)
    check("no tools block", "# Tools" not in plain, "", failures)
    check("plain system turn is used", plain.startswith("<|im_start|>system\n"), repr(plain[:40]), failures)

    print()
    if failures:
        print(f"RESULT: FAILED ({len(failures)} check(s))")
        return 1
    print("RESULT: OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
