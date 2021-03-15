import os
import torch
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from config import args
from tqdm import tqdm

def load_quadruples(inPath, fileName, fileName2=None):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        quadrupleList = []
        times = set()
        for line in fr:
            line_split = line.split()
            head = int(line_split[0])
            tail = int(line_split[2])
            rel = int(line_split[1])
            time = int(line_split[3])
            quadrupleList.append([head, rel, tail, time])
            times.add(time)
        # times = list(times)
        # times.sort()
    if fileName2 is not None:
        with open(os.path.join(inPath, fileName2), 'r') as fr:
            for line in fr:
                line_split = line.split()
                head = int(line_split[0])
                tail = int(line_split[2])
                rel = int(line_split[1])
                time = int(line_split[3])
                quadrupleList.append([head, rel, tail, time])
                times.add(time)
    times = list(times)
    times.sort()

    return np.asarray(quadrupleList), np.asarray(times)

def get_total_number(inPath, fileName):
    with open(os.path.join(inPath, fileName), 'r') as fr:
        for line in fr:
            line_split = line.split()
            return int(line_split[0]), int(line_split[1])

def get_data_with_t(data, tim):
    triples = [[quad[0], quad[1], quad[2]] for quad in data if quad[3] == tim]
    return np.array(triples)

train_data, train_times = load_quadruples('./data/{}'.format(args.dataset), 'train.txt')
num_e, num_r = get_total_number('./data/{}'.format(args.dataset), 'stat.txt')

save_dir_obj = './data/{}/copy_seq/'.format(args.dataset)
save_dir_sub = './data/{}/copy_seq_sub/'.format(args.dataset)

def mkdirs(path):
	if not os.path.exists(path):
		os.makedirs(path)

mkdirs(save_dir_obj)
mkdirs(save_dir_sub)

for tim in tqdm(train_times):
    train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in train_data if quad[3] == tim])
    # get object entities
    row = train_new_data[:, 0] * num_r + train_new_data[:, 1]
    col = train_new_data[:, 2]
    d = np.ones(len(row))
    tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset, tim), tail_seq)
    # get subject_entities
    row1 = train_new_data[:, 2] * num_r + train_new_data[:, 1]
    col1 = train_new_data[:, 0]
    d1 = np.ones(len(row1))
    tail_seq_sub = sp.csr_matrix((d1, (row1, col1)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq_sub/train_h_r_copy_seq_{}.npz'.format(args.dataset, tim), tail_seq_sub)
