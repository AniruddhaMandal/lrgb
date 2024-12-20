import torch
import torch.nn as nn
from torch.nn import functional as F
from torch_geometric.graphgym.config import cfg
from torch_geometric.graphgym.register import register_loss


def vib_loss_func(output, true):
    """ Variational information bottleneck loss function
        output: (y_pred, mu, std)
    """
    if cfg.model.loss_fun == 'vib_loss':
        batch_shape = (len(output)-len(true))/2
        y_pred = output[:len(true)]
        mu = output[len(true): batch_shape]
        std = output[len(true)+batch_shape:]
        return vib_loss(y_pred, true, mu, std), y_pred


register_loss('vib_loss', vib_loss_func)

def vib_loss(y_pred, y, mean, std):
    cross_entropy = F.cross_entropy(y_pred, y,reduction='sum')
    kl = 0.5*torch.mean(mean.pow(2)+std.pow(2)- 2*std.log()-1)
    beta = 1e-3
    return beta*kl+cross_entropy
