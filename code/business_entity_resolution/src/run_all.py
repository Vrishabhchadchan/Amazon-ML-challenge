"""End to end: raw TSVs -> output/matching_results.tsv + output/candidate_pairs.tsv"""
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
STEPS = [
    ["translit.py"],
    ["prep.py", "--split", "train"],
    ["prep.py", "--split", "test"],
    ["candidates.py", "--split", "train"],
    ["candidates.py", "--split", "test"],
    ["train.py"],
    ["predict.py"],
]

for step in STEPS:
    print(">>", " ".join(step), flush=True)
    subprocess.run([sys.executable, os.path.join(HERE, step[0])] + step[1:], check=True, cwd=HERE)
