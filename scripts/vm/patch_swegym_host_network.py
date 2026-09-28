#!/usr/bin/env python3
"""Apply the minimal Docker-network adaptation needed by this VM."""

from __future__ import annotations

import sys
from pathlib import Path


fork_dir = Path(sys.argv[1]).resolve()
upstream = 'use_host_network=False,'
patched = (
    'use_host_network=(\n'
    '                os.environ.get(\n'
    "                    'SWEGYM_USE_HOST_NETWORK', 'false'\n"
    "                ).lower() == 'true'\n"
    '            ),'
)
previous_patch = (
    "use_host_network=os.environ.get('SWEGYM_USE_HOST_NETWORK', 'false').lower() "
    '== "true",'
)

for name in ('run_infer.py', 'eval_infer.py'):
    path = fork_dir / 'evaluation/swe_bench' / name
    source = path.read_text(encoding='utf-8')
    if patched in source:
        print(f'Host-network compatibility patch already present: {path}')
    elif previous_patch in source:
        path.write_text(source.replace(previous_patch, patched, 1), encoding='utf-8')
        print(f'Formatted VM host-network compatibility patch: {path}')
    elif source.count(upstream) == 1:
        path.write_text(source.replace(upstream, patched, 1), encoding='utf-8')
        print(f'Applied VM host-network compatibility patch: {path}')
    else:
        raise SystemExit(
            f'Expected exactly one upstream use_host_network=False setting in {path}; '
            'refusing to patch an unknown source revision.'
        )
