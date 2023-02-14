
import pickle
import os, sys
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import seaborn as sns

fpath = "/nfs/projects/healthcare/nasir/Fast-MoCo/pred_dict.pickle"

preds = None
with open(fpath, 'rb') as fp:
    preds = pickle.load(fp)

all_keys = list(preds.keys())

def count_forgetevent(x:np.ndarray, n_epochs:None):
    result = []
    
    # left - right elem wise; forget event = switch from 1->0

    all_switches = x[:-1] - x[1:]
    if n_epochs:
        all_switches = x[:n_epochs][:-1] - x[:n_epochs][1:]

    if len(np.where(all_switches == 1)[0]) > 0:
        result = np.where(all_switches == 1)[0] + 2

    return len(result) 

forget_counts = {}
all_forget_events = []

pkl_savepath = "/nfs/projects/healthcare/nasir/Fast-MoCo/forget_score_summary/pkl"
fig_savepath = "/nfs/projects/healthcare/nasir/Fast-MoCo/forget_score_summary/figs"

os.makedirs(pkl_savepath, exist_ok=True)
os.makedirs(fig_savepath, exist_ok=True)

epoch_range = [i for i in range(1,100)]

for cur_epoch in tqdm(epoch_range):

    forget_counts = {}
    all_forget_events = []
    n_zeros_lasttime = None

    for temp_key in tqdm(all_keys):
        fgtevent = count_forgetevent(preds[temp_key], cur_epoch)
        assert fgtevent <= cur_epoch
        forget_counts[temp_key] = fgtevent
        all_forget_events.append(fgtevent)

    curr_zeros = sum([1 if x==0 else 0 for x in all_forget_events])
    print("="*50)
    print(f"Upto {cur_epoch}: #0s: ", curr_zeros)
    print("="*50)
    n_zeros_lasttime = curr_zeros

    with open(os.path.join(pkl_savepath, f"forget_dict_upto_{cur_epoch}.pickle"), "wb") as fp:
        pickle.dump(forget_counts, fp)

    # print("Forget counts stored")
    plt.clf()
    plt.title(f"Forget Score (Epoch={cur_epoch})")
    unique, counts = np.unique(np.array(all_forget_events), return_counts=True)
    plt.bar(unique, counts)
    # plt.hist(all_forget_events, rwidth=0.7, align='left')
    plt.savefig(os.path.join(fig_savepath, f"forget_events_upto{cur_epoch}.png"))


    # dis_plot = sns.distplot(all_forget_events, kde=False, bins=15, hist_kws={"rwidth":0.75})
    # dis_plot = sns.distplot(all_forget_events, kde=False, hist_kws={"rwidth":0.75})
    # mids = [rect.get_x() + rect.get_width() / 2 for rect in dis_plot.patches]
    # dis_plot.set_xticks([int(x) for x in mids])
    # dis_fig = dis_plot.get_figure()
    # dis_fig.savefig(os.path.join(fig_savepath, f"forget_events_upto{cur_epoch}.png"))


    plt.clf()

