# This file will be executed inside a separate virtual environment specifially for DS-1000.
# See evaluation/benchmarks/ds_1000.py where this file will be called.

import json

import ds1000

data = ds1000.DS1000Dataset('../ds1000_data/ds1000_data').data
output = {}
for k, v in data.items():
    if k == 'Matplotlib':
        continue # does not support insertion
    output[k] = []
    for item in v:
        output[k].append({ 'prompt': item['prompt'], 'reference': item['reference_code'] })
print(json.dumps(output))
