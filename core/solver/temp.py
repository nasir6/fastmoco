import pickle
import sys
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

def count_forgetevent(x:np.ndarray)->int:
    """Return forget counts given a binary array where 
    array[i] = 1 denotes the on ith epoch the given sample was
    correctly classified as positive pair.

    Args:
        x (np.ndarray): predictions (clfs) per epoch

    Returns:
        int: forgot count i.e the number of times it was forgotten (#1->0 transitiions)
    """
    result = []
    
    # left - right elem wise; forget event = switch from 1->0
    all_switches = x[:-1] - x[1:]

    if len(np.where(all_switches == 1)[0]) > 0:
        result = np.where(all_switches == 1)[0] + 2

    return len(result) 

prefix = "/nfs/projects/healthcare/nasir/Fast-MoCo/ssl_exp/fast_moco/fastmoco_e100/results"
start_epoch = 1
end_epoch = 100
forget_score_threshold = 2

pred_dict = None
# fpath = f'{prefix}/pred_dict_{start_epoch}_{end_epoch}.pickle'
fpath = "/nfs/projects/healthcare/nasir/Fast-MoCo/pred_dict.pickle"
try:
    with open(fpath, 'rb') as handle: 
        pred_dict = pickle.load(handle)
except OSError:
    print("="*50)
    print("Could not open/read file:", fpath)
    print("="*50)
    sys.exit()

all_forget_events = []
all_keys = list(pred_dict.keys())

for temp_key in tqdm(all_keys, desc="Calculating forgetting scores"):
    prediction_subset = pred_dict[temp_key][start_epoch: start_epoch+end_epoch]
    fgtevent = count_forgetevent(prediction_subset)
    # forget_counts[temp_key] = fgtevent
    all_forget_events.append(fgtevent)

all_forget_events = np.array(all_forget_events)
unique, counts = np.unique(all_forget_events, return_counts=True)
result = np.column_stack((unique, counts))
plt.clf()
plt.bar(unique, counts)
# plt.savefig(f"{prefix}/forgot_plot_{start_epoch}_{end_epoch}.png")

plt.savefig("temp.png")

keep_datapoints = []
drop_datapoints = []
for cur_forgot_score, cur_datapoint in tqdm(zip(all_forget_events, all_keys), desc=f"Pruning based on min forget_score={forget_score_threshold}"):
    if cur_forgot_score <= forget_score_threshold:
        # assumption: data does not contribute contr. loss
        drop_datapoints.append(cur_datapoint)
    else:
        keep_datapoints.append(cur_datapoint)

# Sanity check for confirming no data leak and uniq datapoints overall.
assert len(all_keys) == len(set(drop_datapoints)) + len(set(keep_datapoints))


np.savetxt(f'{prefix}/pruned_keys_{start_epoch}_{end_epoch}_{k}.txt', keep_datapoints, fmt="%s" ) # used for further training
np.savetxt(f'{prefix}/dropped_keys_{start_epoch}_{end_epoch}_{k}.txt', drop_datapoints, fmt="%s" )
print(f'dropped keys {len(drop_datapoints)}')
# np.savetxt(f'{prefix}/dropped_keys_{start_epoch}_{end_epoch}_{k}.txt',drop_datapoints,fmt="%s")
# np.savetxt(f'{prefix}/dropped_keys_{start_epoch}_{end_epoch}_{k}.txt',drop_datapoints,fmt="%s")
print()





unique = np.sort(unique)

# import pdb; pdb.set_trace()

# k=min_true_pred
# pruned_keys = []; 
# dropped_keys = []; 


# for count in unique:
#     if count >= k:
#         dropped_keys+=selected_keys[count]
#     else:
#         pruned_keys+=selected_keys[count]

# # for count in unique[-k:]: dropped_keys+=selected_keys[count]
# np.savetxt(f'{prefix}/pruned_keys_{start_epoch}_{end_epoch}_{k}.txt', pruned_keys, fmt="%s" ) # used for further training
# np.savetxt(f'{prefix}/dropped_keys_{start_epoch}_{end_epoch}_{k}.txt', dropped_keys, fmt="%s" )
# print(f'dropped keys {len(dropped_keys)}')


