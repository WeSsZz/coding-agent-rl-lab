from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from .contracts import ActionKind, AgentAction, CodingTask, DatasetSplit, TrajectoryStep
from .environment import is_test_path, render_numbered_window
from .model_policy import PROMPT_VERSION, build_action_messages
from .swe_gym_smoke import _download_pinned_rows, pinned_rows_for_task_set


class SFTDatasetError(ValueError):
    pass


@dataclass(frozen=True)
class PatchHunk:
    old_path: str
    new_path: str
    old_start: int
    old_text: str
    new_text: str


_HUNK_HEADER = re.compile(r"^@@ -(?P<old_start>\d+)(?:,\d+)? \+\d+(?:,\d+)? @@")
MAX_TARGET_ACTION_CHARS = 4096


def parse_unified_diff(patch: str) -> tuple[PatchHunk, ...]:
    lines = patch.splitlines(keepends=True)
    hunks: list[PatchHunk] = []
    old_path: str | None = None
    new_path: str | None = None
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("--- "):
            old_path = _patch_header_path(line[4:])
            index += 1
            if index >= len(lines) or not lines[index].startswith("+++ "):
                raise SFTDatasetError("unified diff --- header is not followed by +++")
            new_path = _patch_header_path(lines[index][4:])
            index += 1
            continue
        match = _HUNK_HEADER.match(line)
        if match:
            if old_path is None or new_path is None:
                raise SFTDatasetError("unified diff hunk appears before file headers")
            old_lines: list[str] = []
            new_lines: list[str] = []
            previous_marker: str | None = None
            index += 1
            while index < len(lines):
                hunk_line = lines[index]
                if hunk_line.startswith(("diff --git ", "--- ", "@@ ")):
                    break
                if hunk_line.startswith("\\ No newline at end of file"):
                    if previous_marker in {" ", "-"} and old_lines:
                        old_lines[-1] = old_lines[-1].rstrip("\r\n")
                    if previous_marker in {" ", "+"} and new_lines:
                        new_lines[-1] = new_lines[-1].rstrip("\r\n")
                    index += 1
                    continue
                if not hunk_line or hunk_line[0] not in {" ", "+", "-"}:
                    break
                previous_marker = hunk_line[0]
                content = hunk_line[1:]
                if previous_marker in {" ", "-"}:
                    old_lines.append(content)
                if previous_marker in {" ", "+"}:
                    new_lines.append(content)
                index += 1
            hunks.append(
                PatchHunk(
                    old_path=old_path,
                    new_path=new_path,
                    old_start=int(match.group("old_start")),
                    old_text="".join(old_lines),
                    new_text="".join(new_lines),
                )
            )
            continue
        index += 1
    return tuple(hunks)


def build_train_gold_sft_dataset(
    rows: Iterable[dict[str, Any]],
    *,
    harvested_failures: dict[str, tuple[str, ...]] | None = None,
    task_sources: dict[str, dict[str, str]] | None = None,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    items = tuple(rows)
    if not items:
        raise SFTDatasetError("at least one train row is required")
    harvested = harvested_failures or {}
    sources = task_sources or {}
    allowed = {item.instance_id: item for item in pinned_rows_for_task_set("train")}
    seen: set[str] = set()
    examples: list[dict[str, Any]] = []
    skipped: Counter[str] = Counter()
    example_counts: Counter[str] = Counter()
    task_ids: list[str] = []

    for row in items:
        task_id = row.get("instance_id")
        if not isinstance(task_id, str) or task_id not in allowed:
            raise SFTDatasetError(f"row is outside the pinned train split: {task_id!r}")
        if task_id in seen:
            raise SFTDatasetError(f"duplicate train row: {task_id}")
        seen.add(task_id)
        pinned = allowed[task_id]
        if row.get("base_commit") != pinned.base_commit:
            raise SFTDatasetError(f"base commit mismatch for {task_id}")
        if row.get("repo") != "getmoto/moto" or row.get("version") != "5.0":
            raise SFTDatasetError(f"repository metadata mismatch for {task_id}")
        issue = row.get("problem_statement")
        patch = row.get("patch")
        if not isinstance(issue, str) or not issue.strip():
            raise SFTDatasetError(f"train row {task_id} has no problem statement")
        if not isinstance(patch, str) or not patch.strip():
            raise SFTDatasetError(f"train row {task_id} has no gold patch")
        task = CodingTask(
            task_id=task_id,
            issue=issue,
            fixture_path=None,
            base_commit=pinned.base_commit,
            test_command=("verifier",),
            split=DatasetSplit.DEVELOPMENT,
            provenance="swe-gym:train-gold-supervision:v1",
            max_steps=12,
            metadata={"repo": "getmoto/moto", "version": "5.0"},
        )
        initial_observation = _training_initial_observation(
            row,
            harvested.get(task_id, ()),
        )
        usable_hunks = 0
        for hunk_index, hunk in enumerate(parse_unified_diff(patch), start=1):
            reason = _unsupported_hunk_reason(hunk)
            if reason is not None:
                skipped[reason] += 1
                continue
            usable_hunks += 1
            for example in _hunk_examples(
                task,
                hunk,
                hunk_index=hunk_index,
                initial_observation=initial_observation,
                sources=sources.get(task_id, {}),
            ):
                examples.append(example)
                example_counts[example["stage"]] += 1
        if usable_hunks == 0:
            raise SFTDatasetError(f"train row {task_id} has no supported source-edit hunks")
        _require_no_gold_leak(task_id, initial_observation, patch)
        task_ids.append(task_id)

    report = {
        "schema_version": 1,
        "dataset_schema": "coding-agent-gold-sft-v1",
        "task_set": "train",
        "task_ids": task_ids,
        "task_count": len(task_ids),
        "example_count": len(examples),
        "stage_counts": dict(sorted(example_counts.items())),
        "skipped_hunk_counts": dict(sorted(skipped.items())),
        # Every example should be True. A False here means the read observation numbered the patch
        # fragment, so the data is internally inconsistent and should not be trained on.
        "examples_with_real_read_window": sum(
            1 for example in examples if example["read_window_is_real_source"]
        ),
        "contains_answers": True,
        "answer_source": "official_swe_gym_gold_patch",
        "prompt_version": PROMPT_VERSION,
        "max_target_action_chars": MAX_TARGET_ACTION_CHARS,
        "training_performed": False,
        "intended_use": "train-split-only-supervised-tool-warm-start",
    }
    return tuple(examples), report


def _require_no_gold_leak(task_id: str, observation: str, patch: str) -> None:
    """Refuse a row whose observation quotes the fix.

    The observation is built from the test the verifier runs and from archived failure output, so it
    should never contain a line the gold patch adds. The dataset is answer-bearing by design, but the
    *prompt* must not be: the model has to find the file, and a prompt that already prints the fix
    would make every arm above meaningless.
    """

    for line in patch.splitlines():
        if not line.startswith("+") or line.startswith("+++"):
            continue
        statement = line[1:].strip()
        if len(statement) < 12:
            # Short lines - a brace, `pass`, a bare return, an import - appear in any failure
            # output, and a fix is rarely one of them alone.
            continue
        if statement in observation:
            raise SFTDatasetError(
                f"train row {task_id} leaks a gold patch line into the initial observation: "
                f"{statement[:80]!r}"
            )


def write_sft_dataset(
    examples: tuple[dict[str, Any], ...],
    report: dict[str, Any],
    *,
    output_path: str | Path,
    report_path: str | Path,
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "".join(json.dumps(example, ensure_ascii=False) + "\n" for example in examples),
        encoding="utf-8",
    )
    report_target = Path(report_path)
    report_target.parent.mkdir(parents=True, exist_ok=True)
    report_target.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build an explicitly answer-containing SFT warm-start dataset from train-only gold patches"
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", help="JSONL containing full official train rows with gold patches")
    source.add_argument(
        "--download-pinned-train",
        action="store_true",
        help="Download only the six pinned train rows in memory; do not write a raw row cache.",
    )
    parser.add_argument(
        "--output",
        default="work/private/swe-gym-train-gold-sft-v1.jsonl",
    )
    parser.add_argument(
        "--report",
        default="work/private/swe-gym-train-gold-sft-v1-report.json",
    )
    parser.add_argument(
        "--failures",
        default="work/private/swe-gym-train-failure-lines.json",
        help=(
            "Verifier failure lines harvested from archived train-task rollouts by "
            "work/harvest_train_failures.py; missing means the rows keep only what the "
            "test patch states"
        ),
    )
    parser.add_argument(
        "--source-root",
        default="work/private/swe-gym-source-cache",
        help=(
            "Per-task real source at the base commit, filled by work/fetch_train_source.py; "
            "without it a read_file observation numbers the patch fragment from 1 and "
            "contradicts the window its own action requests"
        ),
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parents[2]
    if args.download_pinned_train:
        rows = _download_pinned_rows(pinned_rows_for_task_set("train"))
    else:
        rows = _load_rows(project_root / args.input)
    examples, report = build_train_gold_sft_dataset(
        rows,
        harvested_failures=load_harvested_failures(project_root / args.failures),
        task_sources=load_task_sources(project_root / args.source_root),
    )
    write_sft_dataset(
        examples,
        report,
        output_path=project_root / args.output,
        report_path=project_root / args.report,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def _load_rows(path: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SFTDatasetError(f"invalid JSON on row {line_number}") from exc
        if not isinstance(row, dict):
            raise SFTDatasetError(f"row {line_number} must be an object")
        rows.append(row)
    return tuple(rows)


_PYTHON_KEYWORDS = frozenset(
    """False None True and as assert async await break class continue def del elif else except
    finally for from global if import in is lambda nonlocal not or pass raise return try while with
    yield self len print""".split()
)

#: Modules a failure names because they raised, not because the fix lives in them.
_LIBRARY_MODULES = frozenset(
    """requests botocore boto3 urllib http json decimal pytest unittest os sys re typing
    pathlib datetime collections""".split()
)

#: Quoted values a traceback carries that name nothing in the repository.
_NON_IDENTIFIER_VALUES = frozenset(
    """utf-8 utf8 ascii latin-1 true false none""".split()
)

#: Words a traceback sentence is built from, which name nothing to search for.
_PROSE_WORDS = frozenset(
    """Expecting value line column char invalid syntax error occurred calling operation the
    provided key element does not match schema should have failed already list index out of
    range decode codec byte position continuation with exit tests failed baseline verifier
    result requests exceptions during handling above another""".split()
)


def _observations_assertion(initial_observation: str) -> str:
    """The `[failing statement]` line's assertion text, and nothing else.

    Parsing the whole observation instead would offer the header - `Baseline`, `Tests` - as the
    searchable literal, which is a word every row shares and no search can act on.
    """

    for line in (initial_observation or "").splitlines():
        if line.startswith("[failing statement] "):
            body = line[len("[failing statement] ") :]
            _, _, statement = body.partition(": ")
            return statement or body
    return ""


def _is_ephemeral_value(value: str) -> bool:
    """Whether a quoted failure value is something a mock made up for this one run.

    `t911877`, a generated UUID, a random bucket name: the traceback reports them because the test
    created them, and no source file contains them, so teaching one as the query to try spends a
    step on nothing. A value that carries a space, a dot or a hyphen is a name someone wrote.
    """

    if re.fullmatch(r"[0-9._+-]+", value):
        # A numeric literal is the compared value, not a name - `11.700000000000003` is what the
        # arithmetic produced and searching it finds nothing.
        return True
    if any(character in value for character in " .-:/_"):
        return False
    if not re.fullmatch(r"[A-Za-z0-9]+", value):
        return False
    return bool(re.search(r"\d", value)) and bool(re.search(r"[A-Za-z]", value))


def _search_literal(statement: str) -> str | None:
    """A literal from a failure that a search can actually match.

    The locate stage used to teach `search_text` with the gold file's path, which is a query no
    policy can derive and which tells the search nothing it did not already know. What the live
    prompt asks for is the opposite: search an identifier or a literal the failure names.

    Two shapes are worth teaching, and both come from the failure rather than the fix. A quoted
    value is what a failure reports or compares against - `'Not yet implemented'`, `'t911877'`,
    `select_query` - so it wins outright: it is a string the source has to contain somewhere. After
    that, an identifier carrying an underscore or an inner capital is a name a human chose, as
    against the English words and library classes a traceback is mostly made of.
    """

    text = statement or ""
    candidates: list[tuple[int, str]] = []
    for match in re.finditer(r"""['"]([^'"]{4,})['"]""", text):
        value = match.group(1).strip()
        if not value or value.endswith(".py") or "/" in value:
            continue
        if any(bad in value for bad in (",", "=", "(", ")", "{", "}")):
            # A slice of a failure sentence rather than a value: `', select_query ='`.
            continue
        if value.casefold() in _NON_IDENTIFIER_VALUES:
            # `utf-8`: a codec the traceback configured, not a name the repository contains.
            continue
        if _is_ephemeral_value(value):
            # `t911877`: a table name the mock generated for this run. A policy that learns to
            # search it spends a step on a string no source file contains. It is still shown in the
            # observation, where it belongs; it is just not what the stage teaches.
            continue
        candidates.append((2, value))
    for fragment in re.findall(r"[A-Za-z_][A-Za-z0-9_]{3,}", text):
        candidate = fragment.rstrip("_")
        if len(candidate) < 4 or candidate in _PYTHON_KEYWORDS:
            continue
        if candidate.casefold() in _PROSE_WORDS:
            continue
        if candidate.startswith(("test_", "Test")):
            continue
        if candidate.endswith(("Error", "Exception")):
            # `IndexError`, `ClientError`, `UnicodeDecodeError`: what raised, not where the fix is.
            continue
        if "_" not in candidate and not re.search(r"[a-z][A-Z]", candidate):
            # Nothing marks it as a name: it is an English word or a class from a library.
            continue
        if candidate.split(".")[0].casefold() in _LIBRARY_MODULES:
            continue
        candidates.append((1, candidate))
    if not candidates:
        return None
    best_rank, best_candidate = max(
        candidates,
        key=lambda item: (item[0], -abs(len(item[1]) - 12), item[1]),
    )
    return best_candidate


def _hunk_examples(
    task: CodingTask,
    hunk: PatchHunk,
    *,
    hunk_index: int,
    initial_observation: str,
    sources: dict[str, str] | None = None,
) -> tuple[dict[str, Any], ...]:
    path = hunk.new_path
    source_line = next(
        (line.strip() for line in hunk.old_text.splitlines() if line.strip()),
        PurePosixPath(path).name,
    )
    # The locate target is the stage that teaches how to find a file, and the live prompt asks for
    # a literal the failure names rather than a path the policy cannot know yet. Teaching it with
    # the gold path contradicted the observation right above it, and the action wins: the arm that
    # was trained this way never searched a failure literal once, in any configuration.
    #
    # The assertion is the only failure evidence a training row honestly has: `[last error]` and
    # `[string values in the failing frame]` are produced by running the verifier against the base
    # commit, which the builder does not do, and inventing them would train the policy on text no
    # run ever produced. What that leaves uncovered is the next thing to look at: the arm trained
    # this way does search a failure literal, but it searches the exception class from
    # `[last error]` rather than a value the failing frame named.
    locate_query = (
        _search_literal(_harvested_literal_source(initial_observation))
        or _search_literal(_observations_assertion(initial_observation))
        or path
    )
    locate_action = AgentAction(ActionKind.SEARCH_TEXT, {"query": locate_query})
    search_observation = f"{path}:{hunk.old_start}:{source_line[:300]}"
    range_start = max(1, hunk.old_start - 10)
    range_end = max(range_start + 19, hunk.old_start + len(hunk.old_text.splitlines()) + 9)
    range_end = min(range_start + 399, range_end)
    read_action = AgentAction(
        ActionKind.READ_FILE,
        {"path": path, "start_line": range_start, "end_line": range_end},
    )
    replace_action = AgentAction(
        ActionKind.REPLACE_TEXT,
        {"path": path, "old": hunk.old_text, "new": hunk.new_text},
    )
    run_tests_action = AgentAction(ActionKind.RUN_TESTS)

    search_step = TrajectoryStep(1, locate_action, search_observation, False)
    # The observation has to be rendered the way the live environment renders it: the same window,
    # the same absolute line numbers, and the same closing footer. Rendering the patch fragment
    # instead numbered the hunk from 1 while the read action asked for lines 399-433, so the
    # observation contradicted the command that produced it and `replace_lines` was taught against
    # numbering that no file has. A missing source file falls back to the fragment and is counted,
    # because a row that cannot show the real window should not be mistaken for one that does.
    real_source = sources.get(path)
    if real_source is not None:
        total = len(real_source.splitlines())
        read_observation = render_numbered_window(
            real_source,
            (range_start, min(range_end, total)),
            max_lines=max(range_end - range_start + 1, 1),
        )
        faithful_read = True
    else:
        read_observation = render_numbered_window(
            hunk.old_text,
            None,
            max_lines=max(range_end - range_start + 1, len(hunk.old_text.splitlines())),
        )
        faithful_read = False
    read_step = TrajectoryStep(2, read_action, read_observation, False)
    replace_step = TrajectoryStep(3, replace_action, f"Updated {path}.", False)
    stages = (
        ("locate", (), locate_action),
        ("inspect", (search_step,), read_action),
        ("edit", (search_step, read_step), replace_action),
        ("verify", (search_step, read_step, replace_step), run_tests_action),
    )
    return tuple(
        _sft_example(
            task,
            history,
            target,
            stage=stage,
            hunk_index=hunk_index,
            path=path,
            initial_observation=initial_observation,
            read_window_is_real_source=faithful_read,
        )
        for stage, history, target in stages
    )


def _sft_example(
    task: CodingTask,
    history: tuple[TrajectoryStep, ...],
    target: AgentAction,
    *,
    stage: str,
    hunk_index: int,
    path: str,
    initial_observation: str,
    read_window_is_real_source: bool,
) -> dict[str, Any]:
    target_text = json.dumps(target.to_dict(), ensure_ascii=False, separators=(",", ":"))
    identity = f"{task.task_id}\0{path}\0{hunk_index}\0{stage}\0{target_text}"
    messages = [
        *build_action_messages(task, history, initial_observation),
        {"role": "assistant", "content": target_text},
    ]
    return {
        "schema_version": 1,
        "example_id": "sft-" + hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20],
        "task_id": task.task_id,
        "task_set": "train",
        "stage": stage,
        "source_path": path,
        "hunk_index": hunk_index,
        "prompt_version": PROMPT_VERSION,
        "messages": messages,
        "target_action": target.to_dict(),
        "contains_answers": True,
        "answer_source": "official_swe_gym_gold_patch",
        # False means the read observation numbered the patch fragment instead of the file, so its
        # line numbers are relative to the hunk and do not match the window the action requests.
        "read_window_is_real_source": read_window_is_real_source,
    }


def _unsupported_hunk_reason(hunk: PatchHunk) -> str | None:
    if hunk.old_path == "/dev/null" or hunk.new_path == "/dev/null":
        return "file_creation_or_deletion"
    if hunk.old_path != hunk.new_path:
        return "file_rename"
    path = PurePosixPath(hunk.new_path)
    if path.is_absolute() or ".." in path.parts:
        return "unsafe_path"
    folded_parts = {part.casefold() for part in path.parts}
    filename = path.name.casefold()
    if (
        "tests" in folded_parts
        or "test" in folded_parts
        or filename.startswith("test_")
        or filename.endswith("_test.py")
    ):
        return "test_file"
    if not hunk.old_text:
        return "empty_old_text"
    if hunk.old_text == hunk.new_text:
        return "no_op"
    action = AgentAction(
        ActionKind.REPLACE_TEXT,
        {"path": hunk.new_path, "old": hunk.old_text, "new": hunk.new_text},
    )
    target_text = json.dumps(action.to_dict(), ensure_ascii=False, separators=(",", ":"))
    if len(target_text) > MAX_TARGET_ACTION_CHARS:
        return "oversized_target_action"
    return None


def _test_patch_assertion(test_patch: str) -> tuple[str, int, str] | None:
    """The assertion a failing test states, as `(path, line, statement)`.

    The live initial observation carries `[failing statement] <path>:<line>: <assert ...>` lifted
    from the verifier's traceback, and that line is what tells a policy which literals to search.
    A training row has no verifier run, so the honest source for the same thing is the test the
    verifier will run: assertions are taken from added test lines, which is where a test states
    what it expects. The line number is the added line's position in the patched file and is
    therefore exact for a new test file; for a modified one it is close but not guaranteed, so it
    is a position to read from rather than a promise.
    """

    if not isinstance(test_patch, str) or not test_patch.strip():
        return None
    current_path: str | None = None
    current_line = 0
    requested = False
    response_asserts: list[tuple[str, int, str]] = []
    other_asserts: list[tuple[str, int, str]] = []
    for line in test_patch.splitlines():
        if line.startswith("diff --git"):
            # Each file section restarts the flag, and `new file mode` precedes the `+++` header.
            current_path = None
            requested = False
            continue
        if line.startswith("+++ "):
            value = _patch_header_path(line[4:])
            current_path = None if value == "/dev/null" else value
            continue
        if line.startswith("@@"):
            match = re.match(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)", line)
            current_line = int(match.group("start")) if match else 0
            continue
        if line.startswith(("--- ", "diff --git", "index ", "new file mode")):
            continue
        if line.startswith("-"):
            # A deletion consumes an old-side line only. Counting it against the new side pushes
            # every following assertion one line past where the verifier will report it.
            continue
        if not line.startswith("+"):
            current_line += 1
            continue
        stripped = line[1:].strip()
        if "requests.get" in stripped or "requests.post" in stripped:
            requested = True
        if current_path and is_test_path(current_path) and stripped.startswith("assert "):
            entry = (current_path, current_line, stripped)
            # A route that does not exist fails on the first check of what it returned, so an
            # assertion about a response is the shape to show, and the first one after a request
            # is the line the verifier actually reports. A test that calls no endpoint - most of
            # them - still asserts something the failure names, so any assertion is a fallback
            # rather than nothing.
            if requested and re.search(
                r"\b(resp|response|r)\.(json|text|status_code|content)\b", stripped.casefold()
            ):
                response_asserts.append(entry)
            else:
                other_asserts.append(entry)
        current_line += 1
    for group in (response_asserts, other_asserts):
        if not group:
            continue
        path, line, statement = min(group, key=lambda item: item[1])
        return path, line, statement
    return None


def load_task_sources(root: Path) -> dict[str, dict[str, str]]:
    """The real source of every file a task's patch touches, keyed by task then path.

    `work/fetch_train_source.py` fills this from the repository at each task's base commit. Without
    it the builder can only render the patch fragment, which numbers from 1 while the read action
    asks for an absolute window - so a row built from the fragment teaches a line numbering that
    contradicts the command beside it.
    """

    sources: dict[str, dict[str, str]] = {}
    if not root.is_dir():
        return sources
    for task_dir in sorted(entry for entry in root.iterdir() if entry.is_dir()):
        files: dict[str, str] = {}
        for path in sorted(task_dir.rglob("*.py")):
            files[path.relative_to(task_dir).as_posix()] = path.read_text(encoding="utf-8")
        if files:
            sources[task_dir.name] = files
    return sources


def load_harvested_failures(path: Path) -> dict[str, tuple[str, ...]]:
    """The verifier's own failure lines for the pinned train tasks, keyed by task id.

    `work/harvest_train_failures.py` reads them out of the archived train-task rollouts, so they
    are what a real run printed rather than something the builder composed. A missing file is not
    an error: the dataset is still useful without them, it just loses the runtime values.
    """

    if not path.is_file():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SFTDatasetError(f"harvested failure file is not valid JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise SFTDatasetError(f"harvested failure file must map task ids to lines: {path}")
    harvested: dict[str, tuple[str, ...]] = {}
    for task_id, lines in payload.items():
        if isinstance(lines, list) and all(isinstance(line, str) for line in lines):
            harvested[str(task_id)] = tuple(lines)
    return harvested


def _harvested_literal_source(initial_observation: str) -> str:
    """The evidence lines a real run produced, which are the failure's strongest literals.

    `[last error]` names the exception and the values it carried, and
    `[string values in the failing frame]` names the values the failing frame held - an operation
    name, a bucket, a route. Those are what the live prompt offers and what a policy has to search;
    the assertion below them is the fallback, not the first choice.
    """

    wanted = (
        "[last error] ",
        "[logged errors] ",
        "[string values in the failing frame] ",
    )
    lines = []
    for line in (initial_observation or "").splitlines():
        for marker in wanted:
            if line.startswith(marker):
                lines.append(line[len(marker) :])
                break
    return " ".join(lines)


def _training_initial_observation(
    row: dict[str, Any],
    harvested: tuple[str, ...] = (),
) -> str:
    raw_tests = row.get("FAIL_TO_PASS", ())
    if isinstance(raw_tests, str):
        try:
            decoded = json.loads(raw_tests)
        except json.JSONDecodeError:
            decoded = ()
        tests = decoded if isinstance(decoded, list) else ()
    elif isinstance(raw_tests, list):
        tests = raw_tests
    else:
        tests = ()
    rendered = "\n".join(str(test) for test in tests[:20])
    parts = ["Baseline verifier result:", "Tests failed (exit=1)."]
    if rendered:
        parts.append("Failing tests:")
        parts.append(rendered)
    # The live observation continues with `[failing statement]`, so a training row that stops at
    # the test name teaches the policy to answer a failure by naming a file. The assertion is the
    # part of the failure that names something searchable, and it comes from the test patch rather
    # than from the gold patch, so nothing about the fix leaks into it.
    assertion = _test_patch_assertion(row.get("test_patch"))
    if assertion is not None:
        path, line, statement = assertion
        parts.append(f"[failing statement] {path}:{line}: {statement[:200]}")
    # Then the lines only a run produces. A row cannot synthesise them, and `[string values in the
    # failing frame]` is where a failure names the value that points at the module answering it,
    # so they are carried in from the archived verifier output when it exists.
    parts.extend(harvested)
    return "\n".join(parts)


def _patch_header_path(value: str) -> str:
    path = value.rstrip("\r\n").split("\t", 1)[0]
    if path in {"/dev/null", "dev/null"}:
        return "/dev/null"
    if path.startswith(("a/", "b/")):
        path = path[2:]
    return path


if __name__ == "__main__":
    main()
