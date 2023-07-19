#!/usr/bin/env python3

import evals
import json

registry = evals.registry.Registry()

all_base_evals = registry.get_base_evals()

items = {}
for base_eval in all_base_evals:
    items[base_eval.id] = base_eval.description

with open('data/openai-evals/evals-descriptions.json', 'w') as f:
    json.dump(items, f, indent=4)
