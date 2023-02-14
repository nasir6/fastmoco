import os
import timeit
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle
from tqdm import tqdm
import sys

def merge_ranks(prefix='ssl_exp/fast_moco/fastmoco_e100/results', world_size=24, start_epoch=1, end_epoch=100):
    print('start merging')
    prefix = f'{prefix}/train_results.txt.rank'
    for epoch in tqdm(range(start_epoch, end_epoch+1)):
        merged_fd = open(f'{prefix}_all_{epoch}', 'w')
        count = 0
        starttime = timeit.default_timer()
        for rank in range(world_size):
            res_file = f'{prefix}_{rank}_{epoch}'
            assert os.path.exists(res_file), f'No such file or directory: {res_file}'
            with open(res_file, 'r') as fin:
                for line_idx, line in enumerate(fin):
                    merged_fd.write(line)
                    count+=1
        print(f'count for epoch {epoch} {count}')
        print(f"The time difference for epoch {epoch} : {timeit.default_timer() - starttime}")
    merged_fd.close()

def merge_epochs(prefix='ssl_exp/fast_moco/fastmoco_e100/results', start_epoch=1, end_epoch=100):
    # prefix = 'ssl_exp/fast_moco/fastmoco_e100/results/train_results.txt.rank'
    # prefix = f'{prefix}/results/train_results.txt.rank'

    scores_dict = {}
    pred_dict = {}
    tp = []
    for epoch in range(start_epoch, end_epoch+1):
        merged_fd = open(f'{prefix}/train_results.txt.rank_all_{epoch}', 'r')
        starttime = timeit.default_timer()

        for _, line in enumerate(merged_fd):
            info = json.loads(line)
            key = "/".join(info['filename'].split('/')[-2:])
            if key in scores_dict:
                scores_dict[key][epoch-1] = info['score']
                pred_dict[key][epoch-1] = info['prediction']
            else:
                scores_dict[key] = np.zeros(end_epoch)
                pred_dict[key]= np.zeros(end_epoch)
                scores_dict[key][epoch-1] = info['score']
                pred_dict[key][epoch-1] = info['prediction']

            if epoch == end_epoch:
               tp.append(pred_dict[key].sum()) 
        print(f"The time difference for epoch {epoch} : {timeit.default_timer() - starttime}")



    tp = np.array(tp)
    np.save('tp.npy', tp)
    
    with open(f'{prefix}/scores_dict_{start_epoch}_{end_epoch}.pickle', 'wb') as handle: pickle.dump(scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    with open(f'{prefix}/pred_dict_{start_epoch}_{end_epoch}.pickle', 'wb') as handle: pickle.dump(pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    # import pdb; pdb.set_trace()

    # with open('scores_dict.json', 'w') as fp: json.dump(scores_dict, fp)
    # fp.close()
    # with open('pred_dict.json', 'w') as fp: json.dump(pred_dict, fp)
    # fp.close()

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
    if sum(x) == 0:
        return -1

    all_switches = x[:-1] - x[1:]

    if len(np.where(all_switches == 1)[0]) > 0:
        result = np.where(all_switches == 1)[0] + 2

    return len(result) 

def prune_forget_scores(prefix='ssl_exp/fast_moco/fastmoco_e100/results', start_epoch=1, end_epoch=100, min_k=2):
# def prune_forget_scores(*args):
    # _,prefix, start_epoch, end_epoch, min_k = args
    # pred pickle path
    fpath = f'{prefix}/pred_dict_{start_epoch}_{end_epoch}.pickle'
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
    # result = np.column_stack((unique, counts))
    plt.clf()
    plt.bar(unique, counts)
    plt.savefig(f"{prefix}/forgot_plot_{start_epoch}_{end_epoch}.png")
    k=min_k
    keep_datapoints = []
    drop_datapoints = []
    for cur_forgot_score, cur_datapoint in tqdm(zip(all_forget_events, all_keys), desc=f"Pruning based on min forget_score={min_k}"):
        if cur_forgot_score != -1 and cur_forgot_score <= min_k:
            # assumption: data does not contribute contr. loss
            drop_datapoints.append(cur_datapoint)
        else:
            keep_datapoints.append(cur_datapoint)

    # Sanity check for confirming no data leak and uniq datapoints overall.
    assert len(all_keys) == len(set(drop_datapoints)) + len(set(keep_datapoints))


    np.savetxt(f'{prefix}/selected_keys_{start_epoch}_{end_epoch}_{k}.txt', keep_datapoints, fmt="%s" ) # used for further training
    np.savetxt(f'{prefix}/dropped_keys_{start_epoch}_{end_epoch}_{k}.txt', drop_datapoints, fmt="%s" )
    print(f'dropped keys {len(drop_datapoints)}')

# def prune_keys(strategy='sum', *args): #start_epoch=1, end_epoch=100, min_true_pred=2, strategy='sum'): 
def prune_keys(strategy, **kwargs):
    # strategy = args[0]
         
    # import pdb; pdb.set_trace()
    if strategy == 'forget':
        return prune_forget_scores(**kwargs)
    elif strategy == 'sum':
        return prune_sum_scores(**kwargs)

def prune_sum_scores(prefix='ssl_exp/fast_moco/fastmoco_e100/results', start_epoch=1, end_epoch=100, min_k=2):
# def prune_sum_scores(*args):
    # _,prefix, start_epoch, end_epoch, min_k = args
    # import pdb; pdb.set_trace()
    


    # with open('/nfs/projects/healthcare/nasir/Fast-MoCo/pred_dict.pickle', 'rb') as handle: pred_dict = pickle.load(handle)
    with open(f'{prefix}/pred_dict_{start_epoch}_{end_epoch}.pickle', 'rb') as handle: pred_dict = pickle.load(handle)
    # with open(f'{prefix}/scores_dict_{start_epoch}_{end_epoch}.pickle', 'rb') as handle: scores_dict = pickle.load(handle)

    
    tp_range = []
    selected_keys = {}
    for key in tqdm(pred_dict):
        count = pred_dict[key][start_epoch:start_epoch+end_epoch].sum()
        tp_range.append( count)
        if count in selected_keys:
            selected_keys[count].append(key)
        else:
            selected_keys[count] = []
            selected_keys[count].append(key)

            # selected_keys.append(key)
    
    tp_range = np.array(tp_range)
    unique, counts = np.unique(tp_range, return_counts=True)
    result = np.column_stack((unique, counts))
    plt.bar(unique, counts)
    plt.savefig(f"{prefix}/sum_plot_{start_epoch}_{end_epoch}.png")
    plt.clf()
    unique = np.sort(unique)
    
    # import pdb; pdb.set_trace()

    k=min_k
    pruned_keys = []; 
    dropped_keys = []; 
    
    for count in unique:
        if ((end_epoch - start_epoch + 1) - count) <= k:
            dropped_keys += selected_keys[count]
        else:
            pruned_keys += selected_keys[count]

    # for count in unique:
    #     if count >= (max(unique) - k):
    #         dropped_keys+=selected_keys[count]
    #     else:
    #         pruned_keys+=selected_keys[count]

    # for count in unique[-k:]: dropped_keys+=selected_keys[count]
    np.savetxt(f'{prefix}/selected_keys_{start_epoch}_{end_epoch}_{k}.txt', pruned_keys, fmt="%s" ) # used for further training
    np.savetxt(f'{prefix}/dropped_keys_{start_epoch}_{end_epoch}_{k}.txt', dropped_keys, fmt="%s" )
    print(f'dropped keys {len(dropped_keys)}')


    

    # plt.hist(tp_range); plt.grid(True); plt.savefig(f"plots/plot_{epoch}_{epoch+step}.png"); plt.clf()
    # print(f'plotted {epoch} {epoch+step}')
# merge_ranks()
# merge_epochs()
def modify_train_txt(prefix, start_epoch=1, end_epoch=100, min_k=2):

    filetrain = open('/nfs/projects/healthcare/nasir/imagenet1k/train.txt', 'r')
    lines = filetrain.readlines()
    file_map = {line.split(' ')[0]:line  for line in lines}
    filetrain.close()
    
    # lines = [line.rstrip() for line in lines]
    k = min_k
    # for k in range(min_k, drop_easy_k):
    filepruned = open(f'{prefix}/selected_keys_{start_epoch}_{end_epoch}_{k}.txt', 'r')
    keys_prune = filepruned.readlines()
    keys_prune = [line.rstrip() for line in keys_prune]
    subset_lines = [file_map[key] for key in keys_prune ]
    print(f'len of subset {len(subset_lines)} for k is {k}')

    path_ = f'{prefix}/train_pruned_{start_epoch}_{end_epoch}_{k}.txt'
    filetrainpruned = open(path_, 'w')
    filetrainpruned.writelines(subset_lines)
    filetrainpruned.close()
    filepruned.close()

    return path_

# result_path = '/nfs/projects/healthcare/nasir/Fast-MoCo/ssl_exp/fast_moco/fastmoco_e100/experiments_ssl/100/rand_forget_2_71p/results'
# result_path = '/nfs/projects/healthcare/nasir/Fast-MoCo/ssl_exp/fast_moco/fastmoco_e100/experiments_ssl/100/online_pruning_k_8/results/'

# curr_epoch=1
# end_epoch=7
# world_size=16
# merge_ranks(prefix=result_path, world_size=world_size, start_epoch=curr_epoch, end_epoch=end_epoch)
# min_k = 1
# merge_epochs(prefix=result_path, start_epoch=curr_epoch, end_epoch=end_epoch)

# prune_keys(strategy='sum', prefix=result_path, start_epoch=curr_epoch, end_epoch=end_epoch, min_k=min_k)
# prune_keys(strategy='forget', prefix=result_path, start_epoch=curr_epoch, end_epoch=end_epoch, min_k=min_k)
# modify_train_txt(prefix=result_path, start_epoch=curr_epoch, end_epoch=end_epoch, min_k=min_k)
                

# merge_ranks()
# merge_epochs()
# selects_ranges()
# prune_examples()



        

