#!/usr/bin/env python3
# Copyright 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Print the test files of the Stage A smoke and Stage B required partitions from pr-test-hcu.yml.

Used by k8s-stage-a-test.yml to run the same files file-by-file on the K8s pool and on the
legacy runners (run_suite.py on main currently stops at its registry sanity check).
"""
import re
import sys
from pathlib import Path

import yaml

wf = yaml.safe_load(Path('.github/workflows/pr-test-hcu.yml').read_text(encoding='utf-8'))
files = []
for job in ('stage-a-test-1-hcu-small', 'stage-b-test-1-hcu-small'):
    for step in wf['jobs'][job]['steps']:
        files += re.findall(r'--include-file\s+(\S+)', str(step.get('run', '')))
seen = []
for f in files:
    if f not in seen:
        seen.append(f)
print('\n'.join(seen))
sys.exit(0 if seen else 1)
