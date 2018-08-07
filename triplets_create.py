import os
import random

""" triplets_file: 
A text file with each line containing three integers, 
where integer i refers to the i-th image in the filenames file. 
For a line of intergers 'a b c', a triplet is defined such that image a is more 
similar to image c than it is to image b, e.g., 
0 2017 42 """

if __name__ == '__main__':
    root_dir = "../auto-encoder/Caltech256/backup/"

    train_root_dir = os.path.join(root_dir, "images", "train")
    train_triplet_path_file = "triplet_paths_train.txt"
    train_triplet_idx_file = "triplet_index_train.txt"
    train_tot_items = len(os.listdir(train_root_dir))
    ttif = open(train_triplet_idx_file, "a")
    ttpf = open(train_triplet_path_file, "a")

    val_root_dir = os.path.join(root_dir, "images", "val")
    val_triplet_path_file = "triplet_paths_val.txt"
    val_triplet_idx_file = "triplet_index_val.txt"
    val_tot_items = len(os.listdir(val_root_dir))
    vtif = open(val_triplet_idx_file, "a")
    vtpf = open(val_triplet_path_file, "a")



    fid = 0  # anchor id of triplet
    rid = 0  # random id for second part of triplet
    for item in os.listdir(train_root_dir):
        if item.endswith(".jpg"):
            im_item = os.path.join(train_root_dir, item)
            sk_item = im_item.replace("images", "sketch")
            ttpf.write("{}\n".format(im_item))
            ttpf.write("{}\n".format(sk_item))

            rid = random.randint(0, train_tot_items - 1)
            while rid == fid or rid == fid + 1:
                rid = random.randint(0, train_tot_items - 1)

            a = fid
            b = rid
            c = fid+1

            # update
            fid = fid+2

            ttif.write("{}\t{}\t{}\n".format(a, b, c))



    fid = 0  # anchor id of triplet
    rid = 0  # random id for second part of triplet
    for item in os.listdir(val_root_dir):
        if item.endswith(".jpg"):
            im_item = os.path.join(val_root_dir, item)
            sk_item = im_item.replace("images", "sketch")
            vtpf.write("{}\n".format(im_item))
            vtpf.write("{}\n".format(sk_item))

            rid = random.randint(0, val_tot_items - 1)
            while rid == fid or rid == fid + 1:
                rid = random.randint(0, val_tot_items - 1)

            a = fid
            b = rid
            c = fid+1

            # update
            fid = fid+2

            vtif.write("{}\t{}\t{}\n".format(a, b, c))
