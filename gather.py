import os
import timeit
import json
import numpy as np
import matplotlib.pyplot as plt
import pickle

def merge_ranks(prefix='ssl_exp/fast_moco/fastmoco_e100/results', world_size=24, start_epoch=1, end_epoch=100):
    prefix = f'{prefix}/train_results.txt.rank'
    for epoch in range(start_epoch, end_epoch+1):
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
    plt.grid(True)
    plt.savefig(f"{prefix}/hist_plot.png")
    with open(f'{prefix}/scores_dict.pickle', 'wb') as handle: pickle.dump(scores_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    with open(f'{prefix}/pred_dict.pickle', 'wb') as handle: pickle.dump(pred_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)
    handle.close()
    # import pdb; pdb.set_trace()

    # with open('scores_dict.json', 'w') as fp: json.dump(scores_dict, fp)
    # fp.close()
    # with open('pred_dict.json', 'w') as fp: json.dump(pred_dict, fp)
    # fp.close()

    
def selects_ranges(prefix='ssl_exp/fast_moco/fastmoco_e100/results', start_epoch=1, end_epoch=100):

    with open(f'{prefix}/pred_dict.pickle', 'rb') as handle: pred_dict = pickle.load(handle)
    with open(f'{prefix}/scores_dict.pickle', 'rb') as handle: scores_dict = pickle.load(handle)

    # step = 100
    # epoch = 0
    tp_range = []
    selected_keys = {}
    # pruned_keys = []
    for key in pred_dict:
        # import pdb; pdb.set_trace()

        count = pred_dict[key][start_epoch:start_epoch+end_epoch].sum()
        tp_range.append( count)

        # if tp_range[key] == step:
        if count in selected_keys:
            selected_keys[count].append(key)
        else:
            selected_keys[count] = []
            selected_keys[count].append(key)

            # selected_keys.append(key)
    
    tp_range = np.array(tp_range)
    unique, counts = np.unique(tp_range, return_counts=True)
    result = np.column_stack((unique, counts))
    # print(result)

       
    drop_easy_k = 20
    min_k = 1
    for k in range(min_k, drop_easy_k):
        pruned_keys = []; 
        dropped_keys = []; 
        for count in unique[:-k]: pruned_keys+=selected_keys[count]
        for count in unique[-k:]: dropped_keys+=selected_keys[count]
        np.savetxt(f'{prefix}/pruned_keys_{k}.txt', pruned_keys, fmt="%s" )
        np.savetxt(f'{prefix}/dropped_keys_{k}.txt', dropped_keys, fmt="%s" )
        print(f'dropped keys {len(dropped_keys)}')

    # import pdb; pdb.set_trace()

    

    # plt.hist(tp_range); plt.grid(True); plt.savefig(f"plots/plot_{epoch}_{epoch+step}.png"); plt.clf()
    # print(f'plotted {epoch} {epoch+step}')
# merge_ranks()
# merge_epochs()
def prune_examples():

    filetrain = open('/nfs/projects/healthcare/nasir/imagenet1k/train.txt', 'r')
    lines = filetrain.readlines()
    file_map = {line.split(' ')[0]:line  for line in lines}
    filetrain.close()
    
    # lines = [line.rstrip() for line in lines]
    drop_easy_k = 20
    min_k = 2

    for k in range(min_k, drop_easy_k):
        filepruned = open(f'/nfs/projects/healthcare/nasir/Fast-MoCo/pruned_keys_{k}.txt', 'r')
        keys_prune = filepruned.readlines()
        keys_prune = [line.rstrip() for line in keys_prune]
        subset_lines = [file_map[key] for key in keys_prune ]
        print(f'len of subset {len(subset_lines)} for k is {k}')
        filetrainpruned = open(f'/nfs/projects/healthcare/nasir/imagenet1k/train_pruned_{k}.txt', 'w')
        filetrainpruned.writelines(subset_lines)
        filetrainpruned.close()
        filepruned.close()

# merge_ranks()
# merge_epochs()
selects_ranges()
# prune_examples()



        

