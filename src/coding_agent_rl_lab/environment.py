from __future__ import annotations

import ast
import difflib
import hashlib
import re
import shutil
import tempfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Protocol, Sequence, runtime_checkable

READ_FILE_DEFAULT_LINES = 200
READ_FILE_MAX_CHARS = 8_000
SEARCH_TOTAL_LIMIT = 100
SEARCH_PER_FILE_LIMIT = 5
SEARCH_FALLBACK_LIMIT = 20
IMPLEMENTATION_CANDIDATE_LIMIT = 6
RELATED_QUERY_MATCH_LIMIT = 8
RELATED_QUERY_FILE_LIMIT = 12
RELATED_QUERY_MAX_CHARS = 2_400
_PATH_MATCH_PREFIX = "PATH_MATCH:"
_EVIDENCE_PATH = re.compile(
    r"^(?:IMPLEMENTATION_CANDIDATE|SUGGESTED_PATH|PATH_MATCH):(\S+)$",
    re.MULTILINE,
)
_EVIDENCE_MATCH = re.compile(
    r"^([\w./+-]+\.(?:py|pyi|cfg|ini|json|toml|ya?ml)):\d+:",
    re.MULTILINE,
)
_IMPORT_STATEMENT = re.compile(
    r"^[ \t]*(?:from[ \t]+([A-Za-z_][\w.]*)[ \t]+import|import[ \t]+([A-Za-z_][\w.]*))",
    re.MULTILINE,
)
_FAILED_NODE = re.compile(r"^(?:FAILED|ERROR)[ \t]+(\S+)", re.MULTILINE)
_RAISED_LINE = re.compile(r"^E[ \t]+(\S.*)$", re.MULTILINE)
_LOGGED_ERROR = re.compile(r"^ERROR[ \t]+(\S+)[ \t]+(\S.*)$", re.MULTILINE)
_LOGGED_ERROR_MAX_CHARS = 240
_FRAME_STRING = re.compile(r"^[ \t]*(\w+)[ \t]*=[ \t]*'([^'\n]{1,80})'", re.MULTILINE)
_FRAME_MARK = re.compile(
    r"^>[ \t]+(?P<statement>.*)\n\s*\n(?P<path>[^\s:]+):(?P<line>\d+):[ \t]*$",
    re.MULTILINE,
)
_DOCUMENTATION_DIRECTORIES = frozenset({"docs", "doc", "examples", "example"})
_DOCUMENTATION_NAMES = (
    "changelog",
    "implementation_coverage",
    "contributing",
    "readme",
    "notice",
    "license",
    "authors",
    "news",
)

from .contracts import ActionKind, AgentAction, CodingTask, StepResult, TestResult
from .verifier import LocalPythonVerifier


class EnvironmentError(RuntimeError):
    pass


class ToolError(RuntimeError):
    """A recoverable tool failure that should be shown to the policy."""


def action_violation_code(action: AgentAction, error: EnvironmentError) -> str:
    """Return a stable answer-free category for a terminating action violation."""

    message = str(error)
    if message.startswith("cannot modify verifier-owned test file:"):
        category = "protected_test_file"
    elif message == "path escapes repository workspace":
        category = "path_escape"
    elif message.startswith("unsupported action:"):
        category = "unsupported_action"
    else:
        category = "environment_state"
    return f"invalid_action:{action.kind.value}:{category}"


def read_line_range(arguments: dict[str, Any]) -> tuple[int, int] | None:
    has_start = "start_line" in arguments
    has_end = "end_line" in arguments
    if not has_start and not has_end:
        return None
    if not has_start or not has_end:
        raise ToolError("read_file line ranges require both start_line and end_line")
    start_line = arguments["start_line"]
    end_line = arguments["end_line"]
    if (
        isinstance(start_line, bool)
        or isinstance(end_line, bool)
        or not isinstance(start_line, int)
        or not isinstance(end_line, int)
    ):
        raise ToolError("read_file line ranges must be integers")
    if start_line < 1 or end_line < start_line:
        raise ToolError("read_file requires 1 <= start_line <= end_line")
    line_count = end_line - start_line + 1
    if line_count < 20:
        raise ToolError("read_file line range must include at least 20 lines of context")
    if line_count > 400:
        raise ToolError("read_file line range cannot exceed 400 lines")
    return start_line, end_line


def render_numbered_window(
    content: str,
    line_range: tuple[int, int] | None = None,
    *,
    max_lines: int = READ_FILE_DEFAULT_LINES,
    max_chars: int = READ_FILE_MAX_CHARS,
) -> str:
    """Render a line-numbered, self-describing window over one file.

    Line numbers are what let a policy derive `replace_text` and `replace_lines`
    arguments from an observation, and the closing footer is what stops a truncated
    read from looking like the whole file.
    """

    lines = content.splitlines()
    total = len(lines)
    if total == 0:
        return "[file is empty]"
    if line_range is None:
        first, last = 1, min(total, max_lines)
    else:
        first, last = max(1, line_range[0]), min(total, line_range[1])
    if last < first:
        return f"[no lines in range: the file has {total} lines]"
    rendered: list[str] = []
    used_chars = 0
    shown_last = first - 1
    for number in range(first, last + 1):
        line = f"{number}: {lines[number - 1]}"
        if rendered and used_chars + len(line) + 1 > max_chars:
            break
        rendered.append(line)
        used_chars += len(line) + 1
        shown_last = number
    if shown_last >= total:
        return "\n".join(rendered)
    notes = ["character budget reached"] if shown_last < last else []
    notes.append(f"file has {total} lines")
    notes.append(
        "continue with read_file start_line="
        f"{shown_last + 1} end_line={min(total, shown_last + READ_FILE_DEFAULT_LINES)}"
    )
    return "\n".join((*rendered, f"[read_file lines {first}-{shown_last}: {'; '.join(notes)}]"))


def shown_line_span(rendered: str) -> tuple[int, int] | None:
    """The lines a `read_file` observation actually put in front of the policy.

    The policy can only reason about line numbers it has seen, so this - not the requested range -
    is what an edit range has to be checked against. A read clipped by the character budget shows
    less than it asked for, and the footer says so; trusting the request instead of the output is
    how a policy ends up rewriting lines it never saw.
    """

    numbers = [int(match) for match in re.findall(r"^(\d+): ", rendered, re.MULTILINE)]
    if not numbers:
        return None
    return min(numbers), max(numbers)


def describe_spans(spans: Sequence[tuple[int, int]]) -> str:
    return ", ".join(f"{first}-{last}" for first, last in sorted(spans))


def read_range_requirement_message(relative: str, start: int, end: int, spans: Sequence[tuple[int, int]]) -> str:
    return (
        f"replace_lines {start}-{end} is outside every range you have read of {relative}. "
        f"You have read lines {describe_spans(spans)}. Read the lines you want to change first "
        f"(read_file start_line={max(1, start - 20)} end_line={end + 20}), then edit them: an edit "
        "whose indentation you have not seen is a guess, and guessing here is recorded as a failed "
        "edit rather than a wrong one."
    )


def search_location_rank(parts: Sequence[str]) -> int:
    """Rank implementation files ahead of tests and prose; lower is better."""

    if any(part in _DOCUMENTATION_DIRECTORIES for part in parts):
        return 2
    name = parts[-1] if parts else ""
    if name.endswith((".md", ".rst", ".txt")) or name.startswith(_DOCUMENTATION_NAMES):
        return 2
    if any(part == "tests" or part.startswith("test") for part in parts):
        return 1
    return 0


def repository_file_listing(repository: Path) -> tuple[str, ...]:
    """Deterministic, workspace-relative listing of searchable repository files."""

    return tuple(
        sorted(
            path.relative_to(repository).as_posix()
            for path in repository.rglob("*")
            if path.is_file()
            and not any(
                part in {".git", "__pycache__", ".pytest_cache"} or part.endswith(".egg-info")
                for part in path.relative_to(repository).parts
            )
        )
    )


def is_test_path(relative_path: str) -> bool:
    """Report whether a workspace path belongs to the verifier-owned test tree."""

    return any(
        part == "tests" or part.startswith("test")
        for part in (piece.casefold() for piece in PurePosixPath(relative_path).parts)
    )


def imported_module_paths(text: str, *, exists: Callable[[str], bool]) -> list[str]:
    """Resolve the repository files for the modules `text` imports, deepest module first.

    A policy that searches an issue-specific literal usually lands on the verifier-owned
    test that asserts the behavior, and that test's imports are the only pointer it has to
    the implementation. Resolving them costs one pass over the text and turns "every match
    is a test" into a concrete file to read.
    """

    resolved: list[str] = []
    for match in _IMPORT_STATEMENT.finditer(text):
        module = match.group(1) or match.group(2)
        if not module:
            continue
        relative = module.replace(".", "/")
        for candidate in (f"{relative}.py", f"{relative}/__init__.py"):
            if exists(candidate) and candidate not in resolved:
                resolved.append(candidate)
                break
    return sorted(
        resolved,
        key=lambda path: (-path.count("/"), path.endswith("/__init__.py")),
    )


def implementation_candidate_lines(
    sources: Sequence[str],
    *,
    exists: Callable[[str], bool],
    limit: int = IMPLEMENTATION_CANDIDATE_LIMIT,
) -> list[str]:
    """Name the implementation modules a test-only search result actually exercises."""

    resolved: list[str] = []
    for source in sources:
        for path in imported_module_paths(source, exists=exists):
            if path not in resolved:
                resolved.append(path)
    ranked = sorted(
        resolved,
        key=lambda path: (-path.count("/"), path.endswith("/__init__.py")),
    )
    return [f"IMPLEMENTATION_CANDIDATE:{path}" for path in ranked[:limit]]


def _collect_search_matches(
    repository: Path,
    query: str,
    *,
    listing: Sequence[str],
    per_file_limit: int,
) -> list[tuple[int, str, str]]:
    folded = query.casefold()
    ranked: list[tuple[int, int, str, int, str]] = []
    for relative in listing:
        parts = tuple(part.casefold() for part in PurePosixPath(relative).parts)
        rank = search_location_rank(parts)
        if folded in relative.casefold():
            ranked.append((rank, 0, relative, 0, f"{_PATH_MATCH_PREFIX}{relative}"))
        path = repository / relative
        try:
            if path.stat().st_size > 1_000_000:
                continue
            lines = path.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeError):
            continue
        matched_in_file = 0
        for number, line in enumerate(lines, start=1):
            if folded not in line.casefold():
                continue
            ranked.append((rank, 1, relative, number, f"{relative}:{number}:{line[:300]}"))
            matched_in_file += 1
            if matched_in_file >= per_file_limit:
                break
    ranked.sort(key=lambda item: item[:4])
    return [(item[0], item[2], item[4]) for item in ranked]


def _implementation_candidate_lines(
    repository: Path,
    matches: Sequence[tuple[int, str, str]],
    *,
    listing: Sequence[str],
) -> list[str]:
    """Add the test-only navigation hint, or nothing when an implementation already matched."""

    matched_files: list[str] = []
    for rank, relative, _ in matches:
        if rank == 0:
            return []
        if relative not in matched_files:
            matched_files.append(relative)
    if not matched_files:
        return []
    available = set(listing)
    sources: list[str] = []
    for relative in matched_files[:3]:
        try:
            sources.append((repository / relative).read_text(encoding="utf-8"))
        except (OSError, UnicodeError):
            continue
    return implementation_candidate_lines(
        sources,
        exists=lambda path: path in available,
    )


def _related_query_matches(
    repository: Path,
    segment: str,
    *,
    listing: Sequence[str],
    per_file_limit: int,
) -> dict[str, list[str]]:
    """Group a segment's implementation matches by file, most relevant file first.

    Path order puts `moto/core/...` ahead of `moto/moto_api/...`, and the busiest file is
    the one that registers the path, so files whose path carries the segment come first and
    ties break towards the file with the most matches.
    """

    grouped: dict[str, list[str]] = {}
    for rank, relative, line in _collect_search_matches(
        repository,
        segment,
        listing=listing,
        per_file_limit=per_file_limit,
    ):
        if rank == 0 and not line.startswith(_PATH_MATCH_PREFIX):
            grouped.setdefault(relative, []).append(line)
    folded = segment.casefold().replace("-", "_")
    ranked = sorted(
        grouped.items(),
        key=lambda item: (
            folded not in item[0].casefold(),
            -len(item[1]),
            item[0],
        ),
    )
    return dict(ranked)


def related_query_lines(
    repository: Path,
    query: str,
    *,
    listing: Sequence[str],
    per_file_limit: int = SEARCH_PER_FILE_LIMIT,
    match_limit: int = RELATED_QUERY_MATCH_LIMIT,
    file_limit: int = RELATED_QUERY_FILE_LIMIT,
    max_chars: int = RELATED_QUERY_MAX_CHARS,
) -> list[str]:
    """Re-query the longest piece of a string that only tests spell out in full.

    A request path such as `/moto-api/config` exists verbatim in the test that calls it and
    in nothing else: the implementation registers a pattern for it, so an exact search can
    never reach the module that serves it. Retrying the longest path segment is the one
    search that separates "the module with the matching name" from the module that answers
    the request, and the segment has to be specific to be offered, so a generic word such
    as `config` never displaces the file the policy needs.
    """

    segments = {
        segment
        for segment in (piece.strip("`'\" ") for piece in query.split("/"))
        if len(segment) >= 3
    }
    for segment in sorted(segments, key=len, reverse=True):
        if segment.casefold() == query.casefold():
            continue
        grouped = _related_query_matches(
            repository,
            segment,
            listing=listing,
            per_file_limit=per_file_limit,
        )
        if not grouped or len(grouped) > file_limit:
            continue
        shown: list[str] = []
        for index, lines in enumerate(grouped.values()):
            if index >= 3:
                break
            shown.extend(lines[: 4 if index == 0 else 2])
        return [
            f'No implementation file contains "{query}". Shorter query "{segment}" matches '
            "implementation files:",
            *_bounded_search_lines(shown[:match_limit], max_chars=max_chars),
        ]
    return []


def _navigation_lines(
    repository: Path,
    query: str,
    matches: Sequence[tuple[int, str, str]],
    *,
    listing: Sequence[str],
    per_file_limit: int,
) -> list[str]:
    """Append the hints that only apply while no implementation file has matched.

    The related-query lines are evidence read from the repository, so they come first; the
    import-derived candidates are a guess read from the matched test, so they follow it.
    """

    lines: list[str] = []
    if not any(rank == 0 for rank, _, _ in matches):
        lines.extend(
            related_query_lines(
                repository,
                query,
                listing=listing,
                per_file_limit=per_file_limit,
            )
        )
    lines.extend(_implementation_candidate_lines(repository, matches, listing=listing))
    return lines


def _remaining_search_chars(hints: Sequence[str], max_chars: int) -> int:
    """Budget the appended hints inside the same limit as the match lines they follow."""

    reserved = sum(len(line) + 1 for line in hints)
    return max(max_chars // 2, max_chars - reserved)


def fallback_query_segments(query: str, *, minimum: int = 4) -> tuple[str, ...]:
    """The parts of a missed path-like query that are worth searching on their own.

    A policy that misses with `moto/config/server.py` has named a path that does not exist but
    whose basename does, and splitting on whitespace alone leaves it holding one token - the whole
    path - so the single retry it is promised repeats the miss. The basename comes first because a
    filename is the part of a mixed-up path most likely to be right, then the remaining segments
    longest-first.
    """

    stripped = query.strip().strip("\"'`")
    if not stripped or any(character.isspace() for character in stripped):
        return ()
    name = PurePosixPath(stripped).name
    if "." not in name:
        return ()
    ordered = [name, *reversed(stripped.split("/")[:-1]), stripped]
    segments: list[str] = []
    for candidate in ordered:
        if len(candidate) >= minimum and candidate not in segments:
            segments.append(candidate)
    return tuple(segments[:4])


def _render_query_matches(
    repository: Path,
    query: str,
    *,
    listing: Sequence[str],
    total_limit: int,
    per_file_limit: int,
    max_chars: int,
) -> str | None:
    """Render the matches for one query, or `None` when it has none.

    This is the body of a search without its fallback, so a fallback retry can ask it for a second
    query without re-entering the fallback and recursing.
    """

    matches = _collect_search_matches(
        repository,
        query,
        listing=listing,
        per_file_limit=per_file_limit,
    )
    if not matches:
        return None
    shown = matches[:total_limit]
    hints = _navigation_lines(
        repository,
        query,
        shown,
        listing=listing,
        per_file_limit=per_file_limit,
    )
    rendered = _bounded_search_lines(
        [line for _, _, line in shown],
        max_chars=_remaining_search_chars(hints, max_chars),
    )
    rendered.extend(hints)
    return "\n".join(rendered)


def search_repository(
    repository: Path,
    query: str,
    *,
    total_limit: int = SEARCH_TOTAL_LIMIT,
    per_file_limit: int = SEARCH_PER_FILE_LIMIT,
    fallback_limit: int = SEARCH_FALLBACK_LIMIT,
    max_chars: int = READ_FILE_MAX_CHARS,
) -> str:
    """Return a ranked, per-file-capped literal search with one token fallback.

    An issue-derived sentence almost never matches source text, and an uncapped match
    list lets one large file consume the entire result budget, so a failed query is
    retried with its longest token instead of spending another agent step on nothing.
    The result is trimmed to `max_chars` from the top so the highest-ranked matches
    survive the prompt budget instead of the tail the model happens to receive. The
    navigation hints are appended after that trim, so they are budgeted first: a caller
    that truncates from the front would otherwise drop the highest-ranked matches.
    """

    listing = repository_file_listing(repository)
    direct = _render_query_matches(
        repository,
        query,
        listing=listing,
        total_limit=total_limit,
        per_file_limit=per_file_limit,
        max_chars=max_chars,
    )
    if direct is not None:
        return direct
    rendered = [f"No exact matches for: {query}"]
    # A phrase query is retried on its longest word, which is what reaches an identifier. A
    # path-like query needs the other treatment: the whole path is one word, so the retry would
    # repeat the miss, while its basename is a filename the listing branch can actually answer -
    # `moto/config/server.py` does not exist and `server.py` does.
    if "/" in query:
        attempts = [
            (
                segment,
                f"Basename of the query: {segment}"
                if not index
                else f"Path segment of the query: {segment}",
            )
            for index, segment in enumerate(fallback_query_segments(query))
        ]
    else:
        attempts = [
            (token, f"Longest token in the query: {token}")
            for token in sorted(
                {word for word in query.split() if len(word) >= 4},
                key=len,
                reverse=True,
            )[:1]
        ]
    for token, label in attempts:
        body = _render_query_matches(
            repository,
            token,
            listing=listing,
            total_limit=fallback_limit,
            per_file_limit=per_file_limit,
            max_chars=_remaining_search_chars([label], max_chars),
        )
        if body is None:
            continue
        rendered.append(label)
        rendered.append(body)
        break
    else:
        candidates = {path.casefold(): path for path in listing}
        rendered.extend(
            f"SUGGESTED_PATH:{candidates[item]}"
            for item in difflib.get_close_matches(query.casefold(), candidates, n=8, cutoff=0.45)
        )
    return "\n".join(_bounded_search_lines(rendered, max_chars=max_chars))


def _bounded_search_lines(lines: Sequence[str], *, max_chars: int) -> list[str]:
    """Keep the best-ranked whole match lines, and say so when the budget cut them."""

    rendered: list[str] = []
    used_chars = 0
    for line in lines:
        if rendered and used_chars + len(line) + 1 > max_chars:
            break
        rendered.append(line)
        used_chars += len(line) + 1
    if len(rendered) < len(lines):
        rendered.append(
            f"[search_text: showing {len(rendered)} of {len(lines)} matches; "
            "narrow the query or read the first file]"
        )
    return rendered


def replace_line_range_bounds(start_line: Any, end_line: Any) -> tuple[int, int]:
    """Validate a `replace_lines` range on its own, so it can be checked before the file is read."""

    if isinstance(start_line, bool) or isinstance(end_line, bool):
        raise ToolError("replace_lines line ranges must be integers")
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        raise ToolError("replace_lines line ranges must be integers")
    if start_line < 1 or end_line < start_line:
        raise ToolError("replace_lines requires 1 <= start_line <= end_line")
    if end_line - start_line + 1 > 80:
        raise ToolError("replace_lines cannot replace more than 80 lines")
    return start_line, end_line


def replace_line_range(
    content: str,
    *,
    start_line: int,
    end_line: int,
    new: str,
) -> str:
    """Replace a small inclusive one-based line range while preserving its final newline."""

    start_line, end_line = replace_line_range_bounds(start_line, end_line)
    if not isinstance(new, str):
        raise ToolError("new must be a string")

    lines = content.splitlines(keepends=True)
    if end_line > len(lines):
        raise ToolError(
            f"replace_lines end_line {end_line} exceeds file length {len(lines)}"
        )
    selected = "".join(lines[start_line - 1 : end_line])
    replacement = new
    if replacement and selected.endswith("\n") and not replacement.endswith(("\n", "\r")):
        replacement += "\n"
    return "".join(lines[: start_line - 1]) + replacement + "".join(lines[end_line:])


def _occurrences(content: str, old: str) -> list[int]:
    positions: list[int] = []
    start = content.find(old)
    while start >= 0:
        positions.append(start)
        start = content.find(old, start + 1)
    return positions


def enclosed_statement_span(
    original: str | None,
    replaced_lines: tuple[int, int] | None,
) -> tuple[int, int] | None:
    """The innermost statement that contains the replaced lines, when it is wider than them.

    A `v21` trial replaced lines 9-14 of a `url_paths` dictionary that runs to line 31, which
    left an indented block with no opening line and put the syntax error on line 26 - outside
    the range the policy had replaced, and therefore invisible to it. Naming the span is what
    turns that refusal into the repair.
    """

    if original is None or replaced_lines is None:
        return None
    start_line, end_line = replaced_lines
    if not isinstance(start_line, int) or not isinstance(end_line, int):
        return None
    try:
        module = ast.parse(original)
    except SyntaxError:
        return None
    span: tuple[int, int] | None = None
    for node in ast.walk(module):
        if not isinstance(node, ast.stmt):
            continue
        first, last = node.lineno, node.end_lineno
        if first is None or last is None:
            continue
        if (first, last) == (start_line, end_line):
            # The range is exactly a statement, so the range is not what is wrong with the
            # edit: the text it was replaced with is, and pointing at the enclosing block
            # would send the policy off to rewrite code that was never the problem.
            return None
        if first <= start_line and last >= end_line:
            if span is None or (last - first) < (span[1] - span[0]):
                span = (first, last)
    return span


def replaced_line_span(content: str, old: str) -> tuple[int, int] | None:
    """The inclusive range of lines a matched `old` block occupies in `content`."""

    offset = content.find(old)
    if offset < 0:
        return None
    first = content.count("\n", 0, offset) + 1
    return first, first + old.count("\n")


def python_edit_syntax_error(
    relative_path: str,
    updated: str,
    *,
    original: str | None = None,
    replaced_lines: tuple[int, int] | None = None,
) -> str | None:
    """Refuse an edit that would leave an edited Python file unparseable.

    A replacement that spans the wrong lines, or that keeps the indentation of the
    text it copied from another line, still matches and still writes. The container
    then imports the broken module, pytest reports a collection error instead of the
    assertion under test, and the policy has no way to see what it broke. Checking the
    edit before it is written keeps that failure recoverable and names the line.
    """

    if not relative_path.endswith(".py"):
        return None
    try:
        compile(updated, relative_path, "exec")
    except SyntaxError as exc:
        message = (
            f"edit not applied: it would leave {relative_path} unparseable "
            f"({type(exc).__name__}: {exc.msg} at line {exc.lineno}). Replace the whole "
            "statement, including its indentation, in one edit."
        )
        span = enclosed_statement_span(original, replaced_lines)
        if span is not None:
            message += (
                f" The statement you replaced lines {replaced_lines[0]}-{replaced_lines[1]} of "
                f"spans lines {span[0]}-{span[1]}: give replace_lines that whole range."
            )
        return message
    return None


def verifier_output_streams(result: TestResult) -> str:
    """Return the most informative captured stream, plus the other when it adds content.

    A failing graded command can report the same failure in `stdout` (the pytest session
    and the import traceback) and in `stderr` (a two-line collection error). Preferring
    `stderr` outright hid the traceback the policy needed, so the longer stream wins and
    the shorter one is appended only when the longer one does not already contain it.
    """

    streams = [text.strip() for text in (result.stdout, result.stderr) if text.strip()]
    if not streams:
        return ""
    detail = max(streams, key=len)
    for other in streams:
        if other != detail and other not in detail:
            detail = f"{detail}\n{other}"
    return detail


def verifier_output_detail(result: TestResult, *, max_chars: int = 4_000) -> str:
    """Return the tail of the most informative captured stream."""

    return verifier_output_streams(result)[-max_chars:]


def failing_nodes(text: str) -> list[str]:
    """The nodes pytest reported as failing, in the order it printed them.

    pytest writes one `FAILED <node>` line per failure into its short summary, but a
    captured-log line also begins with `ERROR` and carries a logger name such as
    `moto.core.responses:responses.py:180`. Only tokens that name a path or a node id are
    failures, so a logged error cannot displace the node the run actually failed on.
    """

    nodes: list[str] = []
    for match in _FAILED_NODE.finditer(text):
        node = match.group(1)
        if ("/" in node or "::" in node) and node not in nodes:
            nodes.append(node)
    return nodes


def raised_line(text: str) -> str | None:
    """The exception pytest reported, without the assertion diff printed under it.

    pytest writes `E   AssertionError: assert ...` and then an indented difference whose
    lines start with `+`, `-` or `?`, so taking the last `E` line reported `+ States.Runtime`
    instead of the error that ended the run.
    """

    lines = [
        line.strip()
        for line in _RAISED_LINE.findall(text)
        if line.strip()[:1] not in {"+", "-", "?"}
    ]
    return lines[-1] if lines else None


def logged_error_line(text: str) -> str | None:
    """The errors pytest captured from the code under test's own logging.

    A run can end on an assertion about a wrong value while the reason sits only in a
    logged exception, and that line reaches the observation but not the head that a weak
    policy reads. The logger token carries the module and line the exception came from.
    """

    entries: list[str] = []
    for match in _LOGGED_ERROR.finditer(text):
        token = match.group(1)
        if "/" in token or "::" in token:
            continue
        entry = f"{token} {' '.join(match.group(2).split())}"
        if entry not in entries:
            entries.append(entry)
    if not entries:
        return None
    return "[logged errors] " + ", ".join(entries[:2])[:_LOGGED_ERROR_MAX_CHARS]


def failure_summary(result: TestResult, *, max_chars: int = 700) -> str:
    """Lift the actionable head out of a failed verifier run.

    pytest renders a failure as a screen of framework source around one assertion, so a
    weak policy reads the wrong lines, or none of them. Naming the failing node, the
    exception that ended the run, and the short string values in the failing frame is a
    deterministic extraction of the same evidence, and the prompt already asks the policy
    to search the literals a failure contains.
    """

    if result.passed:
        return ""
    text = verifier_output_streams(result)
    failures = failing_nodes(text)
    statement: str | None = None
    for match in _FRAME_MARK.finditer(text):
        path = match.group("path")
        if path.startswith("/") or "site-packages" in path:
            continue
        statement = (
            f"{path}:{match.group('line')}: "
            f"{' '.join(match.group('statement').split())}"
        )
    raised = raised_line(text)
    literals: list[str] = []
    for name, value in _FRAME_STRING.findall(text):
        entry = f"{name} = {value!r}"
        if entry not in literals:
            literals.append(entry)
    parts: list[str] = []
    if failures:
        parts.append("[failing tests] " + ", ".join(failures[:4]))
    if statement:
        parts.append("[failing statement] " + statement[:200])
    if raised:
        parts.append("[last error] " + raised[:200])
    logged = logged_error_line(text)
    if logged:
        parts.append(logged)
    if literals:
        parts.append("[string values in the failing frame] " + ", ".join(literals[:4]))
    return "\n".join(parts)[:max_chars]


def _collapse_whitespace(content: str) -> tuple[str, list[int]]:
    """Collapse whitespace runs to a single space, keeping each character's source offset."""

    collapsed: list[str] = []
    offsets: list[int] = []
    after_whitespace = False
    for index, character in enumerate(content):
        if character.isspace():
            after_whitespace = True
            continue
        if collapsed and after_whitespace:
            collapsed.append(" ")
            offsets.append(index - 1)
        collapsed.append(character)
        offsets.append(index)
        after_whitespace = False
    return "".join(collapsed), offsets


def replace_text_mismatch_message(content: str, old: str) -> str:
    """Explain a failed `replace_text` match, naming the text that would have matched.

    Weak policies rebuild `old` from search output, which prints one matching line at a
    time, so a value that spans a line break never matches the file and the policy retypes
    it until the step budget runs out. Returning the offending region's exact text turns
    that dead end into a one-step repair.
    """

    occurrences = content.count(old)
    if occurrences > 1:
        lines = [content.count("\n", 0, index) + 1 for index in _occurrences(content, old)]
        listed = ", ".join(str(line) for line in lines[:8])
        return (
            f"replace_text requires exactly one match, found {occurrences}: lines {listed}. "
            "Include more surrounding context, such as a whole line or two, so that `old` "
            "matches exactly once."
        )
    collapsed_content, offsets = _collapse_whitespace(content)
    collapsed_old = " ".join(old.split())
    if collapsed_old:
        index = collapsed_content.find(collapsed_old)
        if index >= 0 and collapsed_content.find(collapsed_old, index + 1) < 0:
            start = offsets[index]
            end = offsets[index + len(collapsed_old) - 1] + 1
            first_line = content.count("\n", 0, start) + 1
            last_line = content.count("\n", 0, end) + 1
            exact = content[start:end]
            shown = exact if len(exact) <= 300 else f"{exact[:300]}..."
            return (
                "replace_text requires exactly one match, found 0. The same text appears at "
                f"lines {first_line}-{last_line} with different whitespace, so `old` has to "
                f"repeat the file byte for byte: {shown!r}. Use exactly that value, or "
                f"replace_lines start_line={first_line} end_line={last_line}."
            )
    return (
        "replace_text requires exactly one match, found 0. Search results print one matching "
        "line at a time, so a value joined from two result lines never matches the file. Read "
        "the file and copy `old` from the numbered read_file output, or use replace_lines on "
        "the range that read shows. If this text came from a verifier failure or the issue, run "
        "search_text with it: the file that produces it is not necessarily the file you have "
        "open."
    )


def premature_finish_refusal(
    patched: bool,
    verifier_result: TestResult | None,
    *,
    steps_remaining: int | None = None,
) -> str | None:
    """Explain why `finish` cannot end the episode yet, or return None to allow it.

    A policy that has not applied a source edit cannot have fixed a failing verifier, so
    `finish` at that point is a give-up. Accepting it ends the episode with a zero reward and
    hides how far the policy actually got, which is the signal a weak policy has to be measured
    on. Callers pass the verifier result they want the refusal to quote: the fresh run for a
    patched attempt, and the still-valid failure for an unrepaired one. A verifier timeout is
    allowed through, exactly as `run_tests` terminates on it, because another attempt cannot
    change the outcome.

    The same give-up re-appears one step later once the policy can edit: every `v23` held-out
    trial applied one edit, read the failure the already-refused `finish` had handed it, and
    called `finish` again at step 7 or 8 of 24 with the error still unrepaired. A patched
    failure is therefore refused too, while at least two steps remain - two, because a repair
    costs an edit and the test run that confirms it. With one step left the refusal would only
    replace the action the policy chose with an observation it has no budget to act on, so the
    episode is allowed to end - patched or not, because the refusal that lands on the last step
    ends the episode either way, one observation later. A caller that does not know the
    remaining budget still refuses, and leaves the count out of the message.
    """

    if verifier_result is None or verifier_result.passed or verifier_result.timed_out:
        return None
    if steps_remaining is not None and steps_remaining < 2:
        return None
    if not patched:
        return (
            "finish refused: the verifier still fails and no source file has been edited. Read "
            "the failing assertion above, find the implementation it calls, and change that "
            "file with replace_text or replace_lines, then run the tests again. Do not finish "
            "while the verifier fails."
        )
    budget = (
        f" You have {steps_remaining} tool steps left."
        if steps_remaining is not None
        else ""
    )
    return (
        "finish refused: a source file has been edited and the verifier still fails, so the "
        "patch is not finished. The failure above names what is still wrong: read the file it "
        "names, fix that cause with replace_text or replace_lines, and run the tests again."
        f"{budget} `finish` is accepted once the verifier passes."
    )


def unread_evidence_paths(
    observations: Sequence[str],
    *,
    read: Sequence[str],
    is_read_only: Callable[[str], bool],
    limit: int = 4,
) -> list[str]:
    """Implementation files this episode's own results named and the policy never read.

    A policy stalled on refusals is usually holding its next move in an observation it already
    has: an `IMPLEMENTATION_CANDIDATE` line, or the file of a content match. Repeating those
    back is the difference between "try something else" and naming what is left to try, which
    is what the four no-edit episodes of the `v20` development sweep needed.
    """

    known = set(read)
    found: list[str] = []
    for observation in observations:
        names = list(_EVIDENCE_PATH.findall(observation))
        names.extend(match.group(1) for match in _EVIDENCE_MATCH.finditer(observation))
        for name in names:
            if name in known or name in found or is_read_only(name):
                continue
            found.append(name)
    return found[:limit]


class ActionLoopGuard:
    """Reject unproductive repeated tool calls as recoverable observations.

    A weak policy often answers a rejection by repeating itself and then burns the whole
    step budget on refusals. Rejections therefore escalate: the first one is the plain
    message, and later ones add the state the policy is failing to track.
    """

    def __init__(
        self,
        is_read_only: Callable[[str], bool] = is_test_path,
    ) -> None:
        self._records: list[tuple[AgentAction, str]] = []
        self._is_read_only = is_read_only
        self.rejection_count = 0
        self.consecutive_rejections = 0

    def reset(self) -> None:
        self._records = []
        self.rejection_count = 0
        self.consecutive_rejections = 0

    @property
    def patched(self) -> bool:
        return any(observation.startswith("Updated ") for _, observation in self._records)

    def rejection_for(self, action: AgentAction) -> str | None:
        base = self._base_rejection(action)
        if base is None:
            return None
        self.rejection_count += 1
        self.consecutive_rejections += 1
        if self.consecutive_rejections == 1:
            return base
        return f"{base}. {self._recovery_directive()}"

    def record(self, action: AgentAction, observation: str) -> None:
        if not observation.startswith("Tool error:"):
            self.consecutive_rejections = 0
        self._records.append((action, observation))

    def _base_rejection(self, action: AgentAction) -> str | None:
        if self._records:
            previous_action, previous_observation = self._records[-1]
            if previous_action == action and previous_observation.startswith("Tool error:"):
                return "do not repeat the same action after a tool error; choose another path or tool"
        if action.kind is ActionKind.SEARCH_TEXT and any(
            previous == action and not observation.startswith("Tool error:")
            for previous, observation in self._records
        ):
            return "do not repeat a search_text query that already returned a result"
        if action.kind is ActionKind.LIST_FILES and any(
            previous.kind is ActionKind.LIST_FILES for previous, _ in self._records
        ):
            return "do not repeat list_files"
        if action.kind is ActionKind.READ_FILE:
            # Only a read that *returned content* makes a repeat pointless. Counting a failed read
            # here told the policy it already had a file it never received: one `FileNotFoundError`
            # made that exact action permanently refusable, answered with "do not reread an
            # unchanged file" - a statement about content the policy never got. The retry after a
            # failure is still bounded by the branch above, which refuses an immediate repeat.
            previous_reads = [
                index
                for index, (previous, observation) in enumerate(self._records)
                if previous == action and not observation.startswith("Tool error:")
            ]
            if previous_reads:
                path = action.arguments.get("path")
                changed_after_read = any(
                    previous.kind in {ActionKind.REPLACE_TEXT, ActionKind.REPLACE_LINES}
                    and previous.arguments.get("path") == path
                    and observation.startswith("Updated ")
                    for previous, observation in self._records[previous_reads[-1] + 1 :]
                )
                if not changed_after_read:
                    return "do not reread an unchanged file; use search_text or inspect another file"
        return None

    def _recovery_directive(self) -> str:
        parts = [f"Repeated rejection {self.consecutive_rejections} times in a row."]
        if not self.patched:
            parts.append("No patch has been applied yet.")
        read_files: list[str] = []
        editable_files: list[str] = []
        queries: list[str] = []
        for previous, _ in self._records:
            if previous.kind is ActionKind.READ_FILE:
                path = previous.arguments.get("path")
                if isinstance(path, str) and path not in read_files:
                    read_files.append(path)
                    if not self._is_read_only(path):
                        editable_files.append(path)
            elif previous.kind is ActionKind.SEARCH_TEXT:
                query = previous.arguments.get("query")
                if isinstance(query, str) and query not in queries:
                    queries.append(query)
        if editable_files:
            parts.append(
                "Implementation files already read: " + ", ".join(editable_files[:5]) + "."
            )
        elif read_files:
            parts.append(
                "Only verifier-owned tests have been read: "
                + ", ".join(read_files[:5])
                + ". The edit belongs in the source file that produces the failing value, "
                "which is not necessarily the module those tests import, and never in the "
                "test file."
            )
        if queries:
            parts.append("Queries already used: " + "; ".join(queries[:5]) + ".")
        unread = unread_evidence_paths(
            [observation for _, observation in self._records],
            read=read_files,
            is_read_only=self._is_read_only,
        )
        if unread:
            parts.append(
                "Files your own results named and you have not read: "
                + ", ".join(unread)
                + ". Read one and make the edit there."
            )
        parts.append(
            "Do not issue it again. Search for the exact value the failure quotes, read the "
            "file that produces it, and edit that file with replace_lines on a small range; "
            "finish is refused while no source edit has been applied."
        )
        return " ".join(parts)


@runtime_checkable
class CodingEnvironment(Protocol):
    """Runtime contract used by rollout collection, independent of sandbox type."""

    baseline_result: TestResult | None
    tool_calls: int
    violations: list[str]

    def reset(self, task: CodingTask) -> str: ...

    def step(self, action: AgentAction) -> StepResult: ...

    def finalize(self) -> TestResult: ...

    def changed_files(self) -> tuple[str, ...]: ...

    def patch_is_valid(self) -> bool: ...

    def graded_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]: ...

    @property
    def loop_rejections(self) -> int: ...

    def close(self) -> None: ...


class LocalFixtureEnvironment:
    """A deliberately narrow environment for infrastructure tests, not untrusted code."""

    def __init__(self, project_root: str | Path, verifier: LocalPythonVerifier | None = None) -> None:
        self.project_root = Path(project_root).resolve()
        self.verifier = verifier or LocalPythonVerifier()
        self.task: CodingTask | None = None
        self.workspace: Path | None = None
        self.repository: Path | None = None
        self.baseline_result: TestResult | None = None
        self.last_test_result: TestResult | None = None
        self.steps = 0
        self.tool_calls = 0
        self.violations: list[str] = []
        self._initial_hashes: dict[str, str] = {}
        self._read_spans: dict[str, list[tuple[int, int]]] = {}
        self._edited_since_verification = False
        self._action_loop_guard = ActionLoopGuard()

    def reset(self, task: CodingTask) -> str:
        self.close()
        if task.fixture_path is None:
            raise EnvironmentError(f"task {task.task_id} does not declare a local fixture path")
        source = self._resolve_project_path(task.fixture_path) / "repo"
        if not source.is_dir():
            raise EnvironmentError(f"fixture repository does not exist: {task.fixture_path}")
        self.workspace = Path(tempfile.mkdtemp(prefix=f"coding-agent-{task.task_id}-")).resolve()
        self.repository = self.workspace / "repo"
        shutil.copytree(source, self.repository)
        self.repository = self.repository.resolve()
        self.task = task
        self.steps = 0
        self.tool_calls = 0
        self.violations = []
        self._action_loop_guard.reset()
        self._read_spans = {}
        self._initial_hashes = self._file_hashes()
        self.baseline_result = self.verifier.run(self.repository, task.test_command)
        self.last_test_result = self.baseline_result
        self._edited_since_verification = False
        if self.baseline_result.passed:
            self.close()
            raise EnvironmentError(f"task {task.task_id} is invalid: baseline tests already pass")
        return f"Baseline verifier result:\n{self._test_observation(self.baseline_result)}"

    def step(self, action: AgentAction) -> StepResult:
        task, repository = self._require_active()
        if self.steps >= task.max_steps:
            return StepResult("Step budget exhausted.", True, self.last_test_result)
        self.steps += 1
        self.tool_calls += 1
        rejection = self._action_loop_guard.rejection_for(action)
        if rejection:
            result = StepResult(f"Tool error: {rejection}", False, self.last_test_result)
            self._action_loop_guard.record(action, result.observation)
            return result
        try:
            if action.kind is ActionKind.LIST_FILES:
                files = sorted(
                    str(path.relative_to(repository))
                    for path in repository.rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts
                )
                result = StepResult(self._bounded_listing("\n".join(files)), False)
            elif action.kind is ActionKind.SEARCH_TEXT:
                query = self._required_string(action.arguments, "query")
                if len(query) > 200:
                    raise ToolError("search_text query must be at most 200 characters")
                result = StepResult(search_repository(repository, query), False)
            elif action.kind is ActionKind.READ_FILE:
                path = self._resolve_repository_path(action.arguments.get("path"))
                content = path.read_text(encoding="utf-8")
                relative = path.relative_to(repository).as_posix()
                rendered = render_numbered_window(content, read_line_range(action.arguments))
                span = shown_line_span(rendered)
                if span is not None:
                    self._read_spans.setdefault(relative, []).append(span)
                result = StepResult(rendered, False)
            elif action.kind is ActionKind.REPLACE_TEXT:
                path = self._resolve_repository_path(action.arguments.get("path"))
                relative = path.relative_to(repository).as_posix()
                old = self._required_string(action.arguments, "old")
                new = self._required_string(action.arguments, "new", allow_empty=True)
                if old == new:
                    raise ToolError(
                        f"replace_text would not change {relative}: `new` is identical to `old`. "
                        "This edit is refused instead of reported as applied, so pick the "
                        "statement that produces the failing value and replace it with corrected "
                        "code, or use replace_lines on the range read_file showed."
                    )
                content = path.read_text(encoding="utf-8")
                occurrences = content.count(old)
                if occurrences != 1:
                    raise ToolError(replace_text_mismatch_message(content, old))
                updated = content.replace(old, new, 1)
                unparseable = python_edit_syntax_error(
                    relative,
                    updated,
                    original=content,
                    replaced_lines=replaced_line_span(content, old),
                )
                if unparseable is not None:
                    raise ToolError(unparseable)
                path.write_text(updated, encoding="utf-8")
                self._read_spans.pop(relative, None)
                self._edited_since_verification = True
                result = StepResult(f"Updated {relative}.", False)
            elif action.kind is ActionKind.REPLACE_LINES:
                path = self._resolve_repository_path(action.arguments.get("path"))
                relative = path.relative_to(repository).as_posix()
                spans = self._read_spans.get(relative)
                if not spans:
                    raise ToolError("replace_lines requires reading the target file first")
                start_line, end_line = replace_line_range_bounds(
                    action.arguments.get("start_line"), action.arguments.get("end_line")
                )
                if not any(first <= start_line and end_line <= last for first, last in spans):
                    raise ToolError(
                        read_range_requirement_message(relative, start_line, end_line, spans)
                    )
                new = self._required_string(action.arguments, "new", allow_empty=True)
                content = path.read_text(encoding="utf-8")
                updated = replace_line_range(
                    content,
                    start_line=action.arguments.get("start_line"),
                    end_line=action.arguments.get("end_line"),
                    new=new,
                )
                if updated == content:
                    raise ToolError(
                        f"replace_lines would not change {relative}: the replacement is identical "
                        "to the lines it replaces. This edit is refused instead of reported as "
                        "applied, so change the code those lines contain."
                    )
                unparseable = python_edit_syntax_error(
                    relative,
                    updated,
                    original=content,
                    replaced_lines=(
                        action.arguments.get("start_line"),
                        action.arguments.get("end_line"),
                    ),
                )
                if unparseable is not None:
                    raise ToolError(unparseable)
                path.write_text(updated, encoding="utf-8")
                self._read_spans.pop(relative, None)
                self._edited_since_verification = True
                result = StepResult(f"Updated {relative}.", False)
            elif action.kind is ActionKind.RUN_TESTS:
                result = self.verifier.run(repository, task.test_command)
                self.last_test_result = result
                self._edited_since_verification = False
                result = StepResult(self._test_observation(result), result.passed, result)
            elif action.kind is ActionKind.FINISH:
                # `finish` re-runs the verifier, so a finish with no edit behind it can be
                # refused with the failure the policy needs instead of ending the episode.
                # An unrepaired failure stays valid evidence, so it is reused rather than
                # paid for twice. A finish that follows an edit is the one case that has to
                # run: nothing else can change a failure, and a policy that probes with
                # `finish` between reads otherwise pays for a container run per probe.
                verified = self.last_test_result
                if verified is None or verified.passed or self._edited_since_verification:
                    verified = self.verifier.run(repository, task.test_command)
                    self.last_test_result = verified
                    self._edited_since_verification = False
                refusal = premature_finish_refusal(
                    self._action_loop_guard.patched,
                    verified,
                    steps_remaining=task.max_steps - self.steps,
                )
                if refusal is not None:
                    result = StepResult(
                        f"Tool error: {refusal}\n{self._test_observation(verified)}",
                        False,
                        verified,
                    )
                else:
                    result = StepResult(self._test_observation(verified), True, verified)
            else:
                raise EnvironmentError(f"unsupported action: {action.kind.value}")
        except (ToolError, OSError, UnicodeError) as exc:
            result = StepResult(f"Tool error: {exc}", False, self.last_test_result)
        except EnvironmentError as exc:
            violation = action_violation_code(action, exc)
            self.violations.append(violation)
            result = StepResult(str(exc), True, self.last_test_result, violation)
        self._action_loop_guard.record(action, result.observation)
        return result

    def finalize(self) -> TestResult:
        task, repository = self._require_active()
        result = self.verifier.run(repository, task.test_command)
        self.last_test_result = result
        self._edited_since_verification = False
        return result

    def changed_files(self) -> tuple[str, ...]:
        if self.repository is None:
            return ()
        current = self._file_hashes()
        names = set(self._initial_hashes) | set(current)
        return tuple(sorted(name for name in names if self._initial_hashes.get(name) != current.get(name)))

    def patch_is_valid(self) -> bool:
        if self.repository is None:
            return False
        changed = self.changed_files()
        if not changed:
            return False
        try:
            for relative in changed:
                path = self.repository / relative
                if not path.is_file():
                    return False
                if path.suffix == ".py":
                    compile(path.read_text(encoding="utf-8"), relative, "exec")
        except (OSError, SyntaxError, UnicodeError):
            return False
        return True

    def graded_targets(self) -> tuple[tuple[str, ...], tuple[str, ...]]:
        """Fixtures grade whatever their own trusted test command covers."""

        return (), ()

    @property
    def loop_rejections(self) -> int:
        """Count of policy actions the loop guard refused as unproductive repeats."""

        return self._action_loop_guard.rejection_count

    def close(self) -> None:
        if self.workspace is not None and self.workspace.exists():
            shutil.rmtree(self.workspace)
        self.task = None
        self.workspace = None
        self.repository = None
        self.baseline_result = None
        self.last_test_result = None
        self._initial_hashes = {}
        self._read_spans = {}
        self._edited_since_verification = False
        self._action_loop_guard.reset()

    def __enter__(self) -> LocalFixtureEnvironment:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _resolve_project_path(self, relative: str) -> Path:
        candidate = (self.project_root / relative).resolve()
        if not candidate.is_relative_to(self.project_root):
            raise EnvironmentError("fixture path escapes project root")
        return candidate

    def _resolve_repository_path(self, raw: Any) -> Path:
        if not isinstance(raw, str) or not raw:
            raise ToolError("path must be a non-empty string")
        _, repository = self._require_active()
        candidate = (repository / raw).resolve()
        if not candidate.is_relative_to(repository):
            raise EnvironmentError("path escapes repository workspace")
        if not candidate.is_file():
            raise ToolError(f"file does not exist: {raw}")
        return candidate

    def _file_hashes(self) -> dict[str, str]:
        _, repository = self._require_active()
        result: dict[str, str] = {}
        for path in repository.rglob("*"):
            if not path.is_file() or "__pycache__" in path.parts:
                continue
            result[str(path.relative_to(repository))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return result

    @staticmethod
    def _bounded_listing(content: str, limit: int = 20_000) -> str:
        if len(content) <= limit:
            return content
        half = (limit - 100) // 2
        return (
            content[:half]
            + "\n...[file list truncated; use search_text or run_tests to locate relevant code]...\n"
            + content[-half:]
        )

    def _require_active(self) -> tuple[CodingTask, Path]:
        if self.task is None or self.repository is None:
            raise EnvironmentError("environment is not active")
        return self.task, self.repository

    @staticmethod
    def _required_string(arguments: dict[str, Any], name: str, *, allow_empty: bool = False) -> str:
        value = arguments.get(name)
        if not isinstance(value, str) or (not allow_empty and not value):
            raise ToolError(f"{name} must be a {'string' if allow_empty else 'non-empty string'}")
        return value

    @staticmethod
    def _test_observation(result: TestResult) -> str:
        status = "passed" if result.passed else "failed"
        summary = failure_summary(result)
        detail = verifier_output_detail(result)
        if summary:
            return f"Tests {status} (exit={result.exit_code}).\n{summary}\n{detail}"
        return f"Tests {status} (exit={result.exit_code}).\n{detail}"
