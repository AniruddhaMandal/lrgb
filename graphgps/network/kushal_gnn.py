import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random
import math
import os 
import numpy as np
import dgl
import networkx as nx
import sys

from torch_geometric.nn import GCNConv, GINConv, global_mean_pool
# from ginconv import GINConv
from torch_geometric.utils import degree, to_networkx
# from torch.nn.parameter import Parameter
import torch_geometric.transforms as T 
from torch_geometric.data import Data
# for the registeration of the custom model
from torch_geometric.graphgym.register import register_network
from torch_geometric.graphgym.config import cfg

import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings('ignore')

import scipy.sparse as sp

# torch.autograd.set_detect_anomaly(True)

class MLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, num_layers, dropout):
        super(MLP, self).__init__()
        
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, hidden_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, hidden_channels))

        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x):
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x
    

# defining gnn models
class MolGNN(nn.Module):

    def __init__(self, dim_in, dim_out):
        super(MolGNN, self).__init__()

        self.input_dim = dim_in
        self.output_dim = dim_out
        self.num_layers = cfg.gnn.layers_mp
        self.hidden_dim = cfg.gnn.dim_inner
        self.dropout = cfg.gnn.dropout
        self.device = cfg.device
        self.gnn_convs = nn.ModuleList()
        self.lns = nn.ModuleList()
        self.batch_size = cfg.train.batch_size
        self.batch_frac = 0.6#cfg.train.batch_frac
        self.name = 'gcn'
        # classifier layer
        self.cls = nn.Linear(self.hidden_dim, self.output_dim)

        # init layer
        self.init = nn.Linear(self.input_dim, self.hidden_dim)
        
        if self.name == 'gcn':
            print("Using GCN model...")
            for i in range(self.num_layers):
                if i == self.num_layers - 1:
                    self.gnn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                else:
                    self.gnn_convs.append(GCNConv(self.hidden_dim, self.hidden_dim).to(self.device))
                    self.lns.append(nn.LayerNorm(self.hidden_dim))
        elif self.name == 'gin':
            print("Using GIN model...")
            for i in range(self.num_layers):
                nn_obj = nn.Sequential(nn.Linear(self.hidden_dim, self.hidden_dim),
                                    nn.BatchNorm1d(self.hidden_dim),   
                                    nn.ReLU(),
                                    nn.Linear(self.hidden_dim, self.hidden_dim))
                self.gnn_convs.append(GINConv(nn_obj, eps=0.1, train_eps=False).to(self.device))
                self.lns.append(nn.LayerNorm(self.hidden_dim))
        else:
            print("Invalid model name...")

        
    def forward(self, batch_loader):

        x = batch_loader.x
        edge_index = batch_loader.edge_index
        batch = batch_loader.batch
        ptr = batch_loader.ptr
        y = batch_loader.y 

        x = x.to(self.device,dtype=torch.float32)
        x = self.init(x)
        num_nodes = x.shape[0]
        edge_index = edge_index.to(self.device)
        batch = batch.to(self.device)
        edge_batch = batch[edge_index[0]]
        # print(batch)
        _, node_counts = torch.unique(batch, return_counts = True)
        # print(node_counts)
        # print(x.shape, "   ", edge_index.shape)

        # geenration of batch of masks for the graphs in the batch 
        # batch_mask = self.mask_generation(node_counts, self.batch_frac)
        # print("before ", batch_mask)

        prev_embed = x
        for i in range(self.num_layers):
            x = self.gnn_convs[i](x, edge_index)
            if i != self.num_layers - 1:
                x = F.relu(x)
                x = self.lns[i](x)
                x = F.dropout(x, p=self.dropout, training=True)

            if self.batch_frac != 0:
                batch_mask = self.mask_generation_centrality(node_counts, x, edge_index, batch, edge_batch, ptr, i)
                x, prev_embed = self.async_update(batch_mask, x, prev_embed)

        # applying mean pooling
        x = global_mean_pool(x, batch)

        x = F.dropout(x, p=self.dropout, training=True)
        x = self.cls(x)

        embedding = x
        x = F.log_softmax(x, dim = 1)
        return x,y


    # Randomized Asynchronous Message Passing
    def rand_async_msg_passing(self, x, prev_embed, batch_mask):
        update_mask = torch.from_numpy(batch_mask).to(self.device)
        update_mask = update_mask.unsqueeze(1)
        x = torch.where(update_mask == 1, x, prev_embed)
        prev_embed = x
        return x, prev_embed

    # shuffling node indices
    def batch_shuffle(self, batch_mask):
        if isinstance(batch_mask, np.ndarray):
            num_graphs = batch_mask.shape[0]
        else:
            num_graphs = len(batch_mask)
        for g in range(num_graphs):
            np.random.shuffle(batch_mask[g])
        return batch_mask

    # generating mask for the selecting nodes
    def mask_generation(self, node_counts, batch_frac):
        num_graphs = node_counts.shape[0]
        batch_mask = []
        for g in range(num_graphs):
            num_target_nodes = math.floor(node_counts[g] * batch_frac)
            # print(num_target_nodes)
            ones_ = np.ones(num_target_nodes)
            zeros_ = np.zeros(node_counts[g] - num_target_nodes)
            node_idx_g = np.concatenate([ones_, zeros_], axis = 0)
            # np.random.shuffle(node_idx_g)
            batch_mask.append(node_idx_g)
        return batch_mask


    def async_update(self, batch_mask, x, prev_embed):
        # batch_mask = self.batch_shuffle(batch_mask)
        # print("before ", batch_mask)
        index_mask = np.hstack(batch_mask)
        # print("after ", index_mask)
        x, prev_embed = self.rand_async_msg_passing(x, prev_embed, index_mask)
        return x, prev_embed
    
    # generate mask based on centrality for a single graph 
    def mask_based_on_centrality(self, x, edge_index, layer):
        num_nodes = x.shape[0]
        sorted_indices = self.centrality_measure(edge_index, num_nodes, centrality = 'degree') 
        chunk_size = math.floor(num_nodes / self.num_layers)
        if layer == self.num_layers - 1:
            candidate_nodes = sorted_indices[layer*chunk_size:num_nodes]
        else:
            candidate_nodes = sorted_indices[layer*chunk_size:(layer+1)*chunk_size]
        # print("candidates  ", candidate_nodes.shape)
        sample_size = min(math.floor(self.batch_frac*num_nodes), len(candidate_nodes))
        # print("size ", chunk_size, "    ", sample_size)
        rand_indices = torch.randperm(candidate_nodes.shape[0])
        # print("rand ", rand_indices.shape)
        selected_nodes = candidate_nodes[rand_indices[:sample_size]]
        # print("selected ", selected_nodes.shape)
        node_idx = torch.zeros(num_nodes).to(self.device)
        node_idx[selected_nodes] = 1
        node_idx = node_idx.detach().cpu().numpy()
        # node_idx = node_idx.unsqueeze(1)
        # print(node_idx.shape)
        return node_idx

    def mask_generation_centrality(self, node_counts, x, edge_index, batch, edge_batch, ptr, layer):
        batch_mask = []
        num_graphs = node_counts.shape[0]
        # print("counts ", node_counts)
        for g in range(num_graphs):
            n_index_selector = torch.where(batch == g)
            feats = x[n_index_selector]
            # print(feats.shape)
            e_index_selector = torch.where(edge_batch == g)
            if g == 0:
                edge_indices = torch.stack([edge_index[0][e_index_selector], edge_index[1][e_index_selector]], dim = 0)
            else:
                edge_indices = torch.stack([edge_index[0][e_index_selector] - ptr[g], edge_index[1][e_index_selector] - ptr[g]], dim = 0)
            graph_wise_mask = self.mask_based_on_centrality(feats, edge_indices, layer)
            batch_mask.append(graph_wise_mask)
        return batch_mask


    def centrality_measure(self, edge_index, num_nodes, centrality):
        pyg_g = Data(edge_index = edge_index, num_nodes = num_nodes)
        nx_G = to_networkx(pyg_g)
        if centrality == 'degree':
            node_degrees = degree(edge_index[0], num_nodes = num_nodes)
            sorted_node_degree, sorted_indices = torch.sort(node_degrees, descending = True)
            return sorted_indices
        elif centrality == 'betweeneness':
            centrality = nx.betweenness_centrality(nx_G)
            centrality = torch.tensor(list(centrality.values()))
            return centrality
        elif centrality == 'pagerank':
            centrality = nx.pagerank(nx_G)
            centrality = torch.tensor(list(centrality.values()))
            return centrality
        elif centrality == 'load':
            entrality = nx.load_centrality(nx_G)
            centrality = torch.tensor(list(centrality.values()))
            return centrality
        elif centrality == 'closeness':
            centrality = nx.closeness_centrality(nx_G)
            centrality = torch.tensor(list(centrality.values()))
            return centrality
        else:
            print("Incorrect centrality...")
            
        
register_network("MolGNN",MolGNN)


'''
centrality --- node degree / betweeness / closeness / pagrank / load
nodes with higher degree are highly likely to cause oversquashing. 
nodes with higher centrality should communicate in the earlier layers than the nodes with lower centrality 
'''
