import os
import torch
import numpy as np
from scipy import sparse
import scipy.sparse as sp
from config import args

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

for tim in train_times:
    train_new_data = np.array([[quad[0], quad[1], quad[2], quad[3]] for quad in train_data if quad[3] == tim])
    print(train_new_data)
    row = []
    col = []
    for i in tqdm(range(num_e)):
        for j in range(num_r):
            for k in range(len(train_new_data)):
                if train_new_data[k, 0] == i and train_new_data[k, 1] == j:
                    row.append(i * num_r + j)
                    col.append(train_new_data[k, 2])

    d = np.ones(len(row))
    tail_seq = sp.csr_matrix((d, (row, col)), shape=(num_e * num_r, num_e))
    sp.save_npz('./data/{}/copy_seq/train_h_r_copy_seq_{}.npz'.format(args.dataset,tim), tail_seq)