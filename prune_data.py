from copyreg import pickle
import os, sys
import numpy as np
from tqdm import tqdm
import pickle

# Load training labels
imagenet_source = "/nfs/projects/healthcare/nasir/imagenet1k"

train_labels = np.load(os.path.join(imagenet_source, "train_labels.npy"))
train_images  = np.load(os.path.join(imagenet_source, "train_images.npy"))

print('train_images.shape: ',train_images.shape, ' train_labels.shape: ',train_labels.shape)

class_to_labels = {}
n_classes = 0
for img_path,img_label in tqdm(zip(train_images, train_labels), total=len(train_images)):
    if n_classes == 1000:
        break
    fname = os.path.basename(img_path)
    temp_class = fname.split("_")[0]
    if temp_class not in class_to_labels:
        class_to_labels[temp_class] = img_label
        n_classes += 1

assert len(list(class_to_labels.keys())) == 1000

# print(class_to_labels['n02492035']) #883
# print(class_to_labels['n01950731']) #112
# print(class_to_labels['n02128925']) #813
# print(class_to_labels['n09193705']) #42
# print(class_to_labels['n02489166']) #308

def prune_examples(scores, score_thresh):
    total_data = len(scores)
    sorted_items = reversed(sorted(scores.items(), key=lambda x: x[1]))
    
    result = []
    for x,y in tqdm(sorted_items,desc="Pruning the data", total=total_data):
        if y>score_thresh:
            result.append(x)
        else:
            break
    selected_data = len(result)
    print("Pruned % = ", (total_data-selected_data)/total_data*100.)
    assert len(result) == len(set(result)) # sanity check
    return result


# def get_prune_scores(prune_strategy, prune_filename, score_thresh):
#     assert prune_filename.endswith(".pickle"), " Only pickle format supported"
#     assert prune_strategy.lower() in ['forget_events']

#     scores = None
#     with open(os.path.join("prune_strategies", prune_strategy, prune_filename),"rb") as fp:
#         scores = pickle.load(fp)
    
#     keep_images = prune_examples(scores, score_thresh)
#     return keep_images

def get_prune_scores(prune_fpath, score_thresh):
    assert os.path.basename(prune_fpath).endswith(".pickle"), " Only pickle format supported"

    scores = None
    with open(prune_fpath,"rb") as fp:
        scores = pickle.load(fp)
    
    keep_images = prune_examples(scores, score_thresh)
    return keep_images

def write_imagepaths(imagelist, class_to_labels, savepath):

    fp = open(savepath, "w")
    for imgpath in tqdm(imagelist, desc="Writing selected imagepaths after pruning", total=len(imagelist)):
        imgclass = imgpath.split("/")[0]
        imglabel = class_to_labels[imgclass]
        fp.write(f"{imgpath} {imglabel}\n")
    fp.close()


# prune_strategy = "forget_events"
# prune_file = "forget_dict.pickle"
prune_fpath = "/nfs/projects/healthcare/nasir/imagenet1k/prune_strategies/forget_score_epoch50/pkl/forget_dict.pickle"
score_thresh = 10 #2
# save_dir = os.path.join(imagenet_source, "prune_strategies",prune_strategy)
# save_dir = os.path.join(imagenet_source, "forget_score_epoch50",prune_strategy)
save_dir = "/nfs/projects/healthcare/nasir/imagenet1k/prune_strategies/forget_score_epoch50"
os.makedirs(save_dir, exist_ok=True)


# selected_images = get_prune_scores(prune_strategy, prune_file, score_thresh)
selected_images = get_prune_scores(prune_fpath, score_thresh)

total_images = train_images.shape[0]
ratio = int(len(selected_images)/total_images * 100.)
write_imagepaths(selected_images, class_to_labels, os.path.join(save_dir, f"drop_atleast_{score_thresh}_{ratio}p.txt"))
