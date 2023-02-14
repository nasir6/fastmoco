import numpy as np

train_images = np.load('/nfs/users/ext_prateek.munjal/data/imagenet1k/train_images.npy')
train_labels = np.load('/nfs/users/ext_prateek.munjal/data/imagenet1k/train_labels.npy')
lines = []
for i in range(len(train_labels)):
    line = train_images[i].split('train/')[-1]
    line = f"{line} {train_labels[i]} \n"
    lines.append(line)

filetrain = open('/nfs/projects/healthcare/nasir/imagenet1k/train.txt', 'w')
filetrain.writelines(lines)
filetrain.close()
val_images = np.load('/nfs/users/ext_prateek.munjal/data/imagenet1k/val_images.npy')
val_labels = np.load('/nfs/users/ext_prateek.munjal/data/imagenet1k/val_labels.npy')
lines = []
for i in range(len(val_labels)):
    line = val_images[i].split('val/')[-1]
    line = f"{line} {val_labels[i]} \n"
    lines.append(line)

fileval = open('/nfs/projects/healthcare/nasir/imagenet1k/val.txt', 'w')
fileval.writelines(lines)
fileval.close()