# pip install datasets
# cd vào data
# python downDatasetHug.py  
from datasets import load_dataset

# Tải một phần nhỏ của tập train (100 mẫu)
dataset = load_dataset("deepmind/narrativeqa", split='train[:100]')
dataset.save_to_disk("narrativeqa_sample")
print("Sample dataset saved to 'narrativeqa_sample'")