"""Validate one traced OpenHands rollout and attach its clean verifier reward."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read_one(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    if len(rows) != 1:
        raise ValueError(f'Expected one row in {path}, got {len(rows)}')
    return rows[0]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--verifier', type=Path, required=True)
    parser.add_argument('--completions-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()

    trace = [json.loads(line) for line in args.trace.read_text(encoding='utf-8').splitlines() if line]
    if not trace:
        raise ValueError('Trace is empty')
    rollout = read_one(args.rollout)
    verifier = read_one(args.verifier)
    run_ids = {entry['run_id'] for entry in trace}
    instance_ids = {entry['instance_id'] for entry in trace}
    policy_revisions = {entry.get('policy_revision') for entry in trace}
    model_ids = {entry['response'].get('model') for entry in trace}
    if len(run_ids) != 1 or instance_ids != {rollout['instance_id']}:
        raise ValueError('Trace run or instance identity is inconsistent')
    if len(policy_revisions) != 1 or not next(iter(policy_revisions)):
        raise ValueError('Trace needs one nonempty policy revision')
    if len(model_ids) != 1 or not next(iter(model_ids)):
        raise ValueError('Trace needs one served model ID')
    if rollout['instance_id'] != verifier['instance_id']:
        raise ValueError('Verifier instance does not match rollout')
    if (args.completions_dir.name != rollout['instance_id'] or
            args.completions_dir.parent.parent.resolve() != args.rollout.parent.resolve()):
        raise ValueError('Completion directory does not belong to this rollout')
    patch = rollout['test_result']['git_patch']
    if patch != verifier['test_result']['git_patch']:
        raise ValueError('Verifier patch does not match rollout patch')
    if [entry['sequence'] for entry in trace] != list(range(1, len(trace) + 1)):
        raise ValueError('Trace sequence is incomplete or out of order')

    response_ids = []
    token_counts = []
    observations = 0
    for entry in trace:
        if entry['status'] != 200:
            raise ValueError(f"Completion {entry['sequence']} failed: HTTP {entry['status']}")
        request = entry['request']
        if (request.get('temperature') != 1.0 or request.get('top_p', 1.0) != 1.0 or
                request.get('n', 1) != 1 or request.get('stream', False) or
                any(request.get(key, 0) != 0 for key in (
                    'frequency_penalty', 'presence_penalty'
                )) or request.get('repetition_penalty', 1) != 1 or
                request.get('min_p', 0) != 0 or request.get('top_k', -1) != -1 or
                any(request.get(key) for key in ('logit_bias', 'guided_decoding', 'bad_words'))):
            raise ValueError('RL replay requires temperature=1, top_p=1 and no sampling penalties')
        response = entry['response']
        choice = response['choices'][0]
        token_ids = choice['token_ids']
        prompt_token_ids = response.get('prompt_token_ids')
        logprobs = choice['logprobs']['content']
        if not isinstance(prompt_token_ids, list) or not prompt_token_ids or not all(
            type(token_id) is int and token_id >= 0 for token_id in prompt_token_ids
        ):
            raise ValueError(f"Completion {entry['sequence']} lacks replayable prompt token IDs")
        if not token_ids or len(token_ids) != len(logprobs):
            raise ValueError(f"Completion {entry['sequence']} has misaligned token logprobs")
        if not all(type(token_id) is int and token_id >= 0 for token_id in token_ids):
            raise ValueError(f"Completion {entry['sequence']} has invalid generated token IDs")
        if not all(isinstance(item['logprob'], (float, int)) for item in logprobs):
            raise ValueError(f"Completion {entry['sequence']} has missing sampled logprobs")
        if any(item.get('token') != f'token_id:{token_id}'
               for item, token_id in zip(logprobs, token_ids)):
            raise ValueError(f"Completion {entry['sequence']} logprobs do not identify sampled tokens")
        response_ids.append(response['id'])
        token_counts.append(len(token_ids))
        messages = entry['request']['messages']
        if entry['sequence'] > 1 and messages[-1]['role'] == 'user':
            observations += 1
    if len(set(response_ids)) != len(response_ids):
        raise ValueError('Duplicate model response ID in trace')
    completion_files = list(args.completions_dir.glob('*.json'))
    completions = {
        row['response']['id']: row
        for path in completion_files
        for row in [json.loads(path.read_text(encoding='utf-8'))]
    }
    completion_ids = set(completions)
    if completion_ids != set(response_ids) or len(completion_files) != len(trace):
        raise ValueError('Trace responses do not match the official rollout completions')

    report = verifier['test_result']['report']
    if report.get('error_eval') or report.get('test_timeout'):
        raise ValueError('Verifier infrastructure error cannot be used as a reward')
    result = {
        'run_id': next(iter(run_ids)),
        'instance_id': rollout['instance_id'],
        'policy_revision': next(iter(policy_revisions)),
        'model_id': next(iter(model_ids)),
        'trace_sha256': sha256(args.trace),
        'rollout_sha256': sha256(args.rollout),
        'verifier_sha256': sha256(args.verifier),
        'git_patch_sha256': hashlib.sha256(patch.encode('utf-8')).hexdigest(),
        'response_ids': response_ids,
        'tool_boundaries': [
            {
                'response_id': response_id,
                'tool_calls': (
                    (completions[response_id].get('fncall_response') or {})
                    .get('choices', [{}])[0].get('message', {}).get('tool_calls') or []
                ),
            }
            for response_id in response_ids
        ],
        'generation_count': len(trace),
        'sampled_token_count': sum(token_counts),
        'model_visible_observation_turns': observations,
        'agent_action_events_in_final_history': sum(
            event.get('source') == 'agent' and event.get('action') is not None
            for event in rollout['history']
        ),
        'reward': int(report['resolved']),
        'verifier_report': report,
        'rollout_error': rollout.get('error'),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps({key: result[key] for key in (
        'run_id', 'instance_id', 'generation_count', 'sampled_token_count',
        'model_visible_observation_turns', 'reward',
    )}))


if __name__ == '__main__':
    main()
