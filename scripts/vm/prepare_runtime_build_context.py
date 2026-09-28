"""Prepare an OpenHands-generated runtime context for this VM's package mirrors."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path


context = Path(sys.argv[1]).resolve()
code_dir = context / 'code'
wheel_source = Path(
    '/tmp/swegym-litellm-src/litellm-33ceb7ca1ffab1401decca773253b58d5fc461a2'
) / 'dist/litellm-1.52.5-py3-none-any.whl'
wheel_rel = Path('wheels') / wheel_source.name
(code_dir / 'wheels').mkdir(exist_ok=True)
(code_dir / wheel_rel).write_bytes(wheel_source.read_bytes())

pyproject = code_dir / 'pyproject.toml'
contents = pyproject.read_text()
old = 'litellm = "^1.52.3"'
new = f'litellm = {{ path = "{wheel_rel.as_posix()}" }}'
if old in contents:
    pyproject.write_text(contents.replace(old, new, 1))
elif new not in contents:
    raise RuntimeError('Unexpected LiteLLM declaration in generated runtime context')

dockerfile = context / 'Dockerfile'
contents = dockerfile.read_text()
copy_lock = 'COPY ./code/pyproject.toml ./code/poetry.lock /openhands/code/'
copy_wheel = 'COPY ./code/wheels/ /openhands/code/wheels/'
if copy_wheel not in contents:
    if copy_lock not in contents:
        raise RuntimeError('Could not find lockfile COPY in generated Dockerfile')
    contents = contents.replace(copy_lock, copy_lock + '\n' + copy_wheel, 1)
contents = contents.replace(
    '-c conda-forge',
    '-c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge',
)
contents = contents.replace(
    'python=3.12 -y',
    'python=3.12 --override-channels -y',
    1,
)
contents = contents.replace(
    '/openhands/micromamba/bin/micromamba run -n openhands poetry cache clear --all . && \\\n',
    'echo "Keep Poetry cache for non-interactive runtime build" && \\\n',
)
# The browser binary comes from the local overlay; keep its OS dependencies.
contents = contents.replace(
    'poetry run playwright install --with-deps chromium',
    'poetry run playwright install-deps chromium',
)
dockerfile.write_text(contents)

lock_template = Path.home() / 'swegym-runtime-poetry.lock'
if lock_template.exists():
    shutil.copyfile(lock_template, code_dir / 'poetry.lock')
else:
    env = os.environ.copy()
    env['PATH'] = '/home/wesz/.local/bin:' + env.get('PATH', '')
    subprocess.run(
        ['poetry', 'lock', '--no-update'],
        cwd=code_dir,
        env=env,
        check=True,
    )
    shutil.copyfile(code_dir / 'poetry.lock', lock_template)
print(f'Prepared runtime context: {context}')
