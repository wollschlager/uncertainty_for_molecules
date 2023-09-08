import torch
from torch.distributions import MultivariateNormal

def rmse(targets, pred):
    """
    Mean L2 Error
    """
    return torch.mean(torch.norm((pred - targets), p=2, dim=1))

def mae(pred, targets):
    """
    Mean Absolute Error
    """
    if isinstance(pred, MultivariateNormal):
        pred = pred.mean
    return torch.nn.functional.l1_loss(pred, targets, reduction="mean")

def map_state_dict(sd):
    new_dict = {}
    for k in sd:
        if k[:16] == "model.model.out_module":
            new_dict[k[12:]] = sd[k]
        else:
            new_dict[k[18:]] = sd[k]
    return new_dict