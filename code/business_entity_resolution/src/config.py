import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# default layout: code/business_entity_resolution/ sits inside student_resource/
DATA = os.environ.get("ER_DATA", os.path.join(ROOT, "..", "..", "dataset"))
CACHE = os.environ.get("ER_CACHE", os.path.join(ROOT, "cache"))
OUTPUT = os.environ.get("ER_OUTPUT", os.path.join(ROOT, "..", "..", "output"))

SEED = 42

# blocking: how many neighbours to keep per S1 record from each source
TOPK_PER_SOURCE = 20
# tokens more frequent than this (within a country) are ignored for blocking
MAX_DF = 25000

# pruning before the matcher (see pipeline.prune)
KEEP_RANK_BOTH = 10
KEEP_RANK_ONE = 3
