import numpy as np
import random
import scipy.sparse as sp
import torch
import codecs
import json
import copy

def normalize(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


class GraphMaker(object):
    def __init__(self, opt, filename):
        self.opt = opt
        self.user = set()
        self.item = set()
        data=[]
        with codecs.open(filename) as infile:
            for line in infile:
                line = line.strip().split("\t")
                data.append([int(line[0]), int(line[1])])
                self.user.add(int(line[0]))
                self.item.add(int(line[1]))

        self.number_user = max(self.user) + 1
        self.number_item = max(self.item) + 1

        print("number_user", max(self.user) + 1)
        print("number_item", max(self.item) + 1)
        
        self.raw_data = data
        self.UV, self.VU, self.adj = self.preprocess(data, opt)

    def preprocess(self,data,opt):
        UV_edges = []
        VU_edges = []
        all_edges = []
        # real_adj = {}

        user_real_dict = {}
        item_real_dict = {}
        for edge in data:
            UV_edges.append([edge[0],edge[1]])
            if edge[0] not in user_real_dict.keys():
                user_real_dict[edge[0]] = set()
            user_real_dict[edge[0]].add(edge[1])

            VU_edges.append([edge[1], edge[0]])
            if edge[1] not in item_real_dict.keys():
                item_real_dict[edge[1]] = set()
            item_real_dict[edge[1]].add(edge[0])

            all_edges.append([edge[0],edge[1] + self.number_user])
            all_edges.append([edge[1] + self.number_user, edge[0]])
            # if edge[0] not in real_adj :
            #     real_adj[edge[0]] = {}
            # real_adj[edge[0]][edge[1]] = 1

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        all_edges = np.array(all_edges)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(self.number_user, self.number_item),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(self.number_item, self.number_user),
                               dtype=np.float32)
        all_adj = sp.coo_matrix((np.ones(all_edges.shape[0]), (all_edges[:, 0], all_edges[:, 1])),shape=(self.number_item+self.number_user, self.number_item+self.number_user),dtype=np.float32)
        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)
        all_adj = normalize(all_adj)
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)
        all_adj = sparse_mx_to_torch_sparse_tensor(all_adj)

        print("real graph loaded!")
        return UV_adj, VU_adj, all_adj

    def random_sample(self):
        data = self.raw_data
        opt = self.opt

        UV_edges = []
        VU_edges = []

        user_real_dict = {}
        item_real_dict = {}
        for edge in data:
            if edge[0] not in user_real_dict.keys():
                user_real_dict[edge[0]] = set()
            user_real_dict[edge[0]].add(edge[1])

            if edge[1] not in item_real_dict.keys():
                item_real_dict[edge[1]] = set()
            item_real_dict[edge[1]].add(edge[0])

        # Random remove edges
        for user, items in user_real_dict.items():
            rm_num = int(len(items) * opt["remove_rate"])
            rm_num = max(1, rm_num)
            if len(items) == 1:
                rm_num = 0
            tgts = random.sample(items, rm_num)
            user_real_dict[user] -= set(tgts)
            for tgt in tgts:
                item_real_dict[tgt] -= set([user])
        
        for user, items in user_real_dict.items():
            for item in items:
                UV_edges.append([user, item])
        for item, users in item_real_dict.items():
            for user in users:
                VU_edges.append([item, user])

        UV_edges = np.array(UV_edges)
        VU_edges = np.array(VU_edges)
        UV_adj = sp.coo_matrix((np.ones(UV_edges.shape[0]), (UV_edges[:, 0], UV_edges[:, 1])),
                               shape=(self.number_user, self.number_item),
                               dtype=np.float32)
        VU_adj = sp.coo_matrix((np.ones(VU_edges.shape[0]), (VU_edges[:, 0], VU_edges[:, 1])),
                               shape=(self.number_item, self.number_user),
                               dtype=np.float32)
        UV_adj = normalize(UV_adj)
        VU_adj = normalize(VU_adj)
        UV_adj = sparse_mx_to_torch_sparse_tensor(UV_adj)
        VU_adj = sparse_mx_to_torch_sparse_tensor(VU_adj)

        print("real graph loaded!")
        self.UV = UV_adj
        self.VU = VU_adj
        self.adj = None

