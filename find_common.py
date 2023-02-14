import numpy as np
from glob import glob
import os, sys

results_dir = "/nfs/projects/healthcare/nasir/Fast-MoCo/ssl_exp/fast_moco/fastmoco_e100/experiments_ssl/100/weighted_loss_10_sum_2/results"
all_paths = glob(os.path.join(results_dir, "dropped_*.txt"))
all_paths.sort()

all_sets = []

results = None
final_results = None
prev_set = None

for fpath in all_paths:
    temp = np.genfromtxt(fpath, dtype='str')
    if prev_set is None:
        prev_set = set(temp)
        final_results = set(temp)
        prev_fname = os.path.basename(fpath)
    else:
        curr_fname = os.path.basename(fpath)
        curr_set = set(temp)

        final_results = final_results.intersection(curr_set)
        print("="*70)
        print(f"Intersection b/w {curr_fname} and {prev_fname}: {len(prev_set.intersection(curr_set))} | Overall intersection: {len(final_results)}")
        print("="*70)
        prev_fname = curr_fname
        prev_set = curr_set

    # else:
    #     results 
    #     all_sets.append(set(temp))

# print(fpaths)