
import pickle
import os, sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

# fpath = "/nfs/projects/healthcare/nasir/Fast-MoCo/pred_dict.pickle"
fpath = "/nfs/projects/healthcare/nasir/Fast-MoCo/ssl_exp/fast_moco/fastmoco_e100/experiments_ssl/100/online_pruning_forget_k_1/results/pred_dict_1_50.pickle"

preds = None
with open(fpath, 'rb') as fp:
    preds = pickle.load(fp)

all_keys = list(preds.keys())

def count_forgetevent(x:np.ndarray, n_epochs:None):
    result = []
    
    if sum(x) == 0:
        return -1

    # left - right elem wise; forget event = switch from 1->0
    all_switches = x[:-1] - x[1:]
    if n_epochs:
        all_switches = x[:n_epochs][:-1] - x[:n_epochs][1:]

    if len(np.where(all_switches == 1)[0]) > 0:
        result = np.where(all_switches == 1)[0] + 2

    return len(result) 

forget_counts = {}
all_forget_events = []

pkl_savepath = "/nfs/projects/healthcare/nasir/imagenet1k/prune_strategies/forget_score_epoch50/pkl"
fig_savepath = "/nfs/projects/healthcare/nasir/imagenet1k/prune_strategies/forget_score_epoch50/figs"

os.makedirs(pkl_savepath, exist_ok=True)
os.makedirs(fig_savepath, exist_ok=True)


for temp_key in tqdm(all_keys):
    fgtevent = count_forgetevent(preds[temp_key], n_epochs=None)
    forget_counts[temp_key] = fgtevent
    all_forget_events.append(fgtevent)



with open(os.path.join(pkl_savepath, f"forget_dict.pickle"), "wb") as fp:
    pickle.dump(forget_counts, fp)

# print("Forget counts stored")
plt.clf()
plt.title("Forget Score")
unique, counts = np.unique(np.array(all_forget_events), return_counts=True)
print(np.asarray((unique, counts)).T)
plt.bar(unique, counts)
plt.savefig(os.path.join(fig_savepath, f"forget_events.png"))

