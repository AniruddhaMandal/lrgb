import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn


from torch_geometric.graphgym.config import cfg
import torch_geometric.graphgym.models.head  # noqa, register module
import torch_geometric.graphgym.register as register
from torch_geometric.graphgym.register import register_network


class MPEncoder(torch.nn.Module):
    """ Message Passing Encoder"""
    def __init__(self, dim_in, dim_latent, depth=5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_in = dim_in
        self.dim_latent = dim_latent
        self.depth = depth

        self.gcn = pyg_nn.GCNConv(self.dim_latent,self.dim_latent)

        self.encoder_layers = torch.nn.ModuleList()
        for i in range(self.depth):
            self.encoder_layers.append(torch.nn.Linear(self.dim_latent,self.dim_latent))
        self.mean_layer = torch.nn.Linear(self.dim_latent,self.dim_latent)
        self.std_layer = torch.nn.Linear(self.dim_latent,self.dim_latent)

    def forward(self, batch):
        X, edge_index = batch.x, batch.edge_index
        # Need to cast input tensor from Long to float
        X = self.gcn(X,edge_index)
        for i in range(self.depth):
            X = self.encoder_layers[i](X)
            X = F.sigmoid(X)
        mu = self.mean_layer(X)
        std = self.std_layer(X)
        std = F.softplus(std-5,beta=1)
        batch.x = self.reparametrization(mu,std)
        return batch, mu,std

    def reparametrization(self, mu, std):
        """ Latent sampling"""
        eps = torch.rand_like(std)
        return (mu+std*eps)

class Decoder(torch.nn.Module):
    def __init__(self, dim_latent, dim_out, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dim_latent = dim_latent
        self.dim_out = dim_out
        self.decoder_layers = torch.nn.ModuleList()
        self.decoder_layers.append(torch.nn.Linear(dim_latent, 256))
        self.decoder_layers.append(torch.nn.Linear(256,128))
        self.decoder_layers.append(torch.nn.Linear(128,64))
        self.decoder_layers.append(torch.nn.Linear(64,32))
        self.decoder_layers.append(torch.nn.Linear(32,16))
        self.output_layer = torch.nn.Linear(16,dim_out)

    def forward(self, X):
        for layer in self.decoder_layers:
            X = layer(X)
            X = F.relu(X)
        X = self.output_layer(X)
        y_pred = F.sigmoid(X)
        return y_pred

class IBGCN(torch.nn.Module):
    def __init__(self, dim_in, dim_out, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.dim_in = dim_in
        self.dim_out = dim_out
        self.max_hops = cfg.gnn.layers_mp 
        self.dim_latent = cfg.gnn.dim_inner

        self.input_layer = torch.nn.Linear(self.dim_in, self.dim_latent)
        self.mp_encoder = MPEncoder(self.dim_in,self.dim_latent)
        self.decoder = Decoder(self.dim_latent, self.dim_out)
        
    def forward(self, batch, k):
        """ k: Number of message passing hops"""
        X = batch.x
        X = X.to(torch.float32)
        X = self.input_layer(X)
        batch.x = F.sigmoid(X)
        for i in range(k):
            batch, mu, std = self.mp_encoder(batch)

        X = batch.x
        y_true = batch.y
        y_pred = pyg_nn.global_mean_pool(self.decoder(X), batch.batch)
        
        output = (y_pred, mu, std)
        return output, y_true
        
register_network("ibgcn", IBGCN)
