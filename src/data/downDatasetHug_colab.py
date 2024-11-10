import json
from datasets import load_dataset
data = load_dataset('deepmind/narrativeqa', split = 'train[:29]')
data.to_json('narrativeqa_train.json')