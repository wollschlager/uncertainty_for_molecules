import numpy as np
from torch.utils.data import Dataset
import torch
from torch_geometric.data import InMemoryDataset, Data
from torch.utils.data._utils.collate import default_collate

class Batch:
    def __init__(self, inputs: torch.Tensor, targets: torch.Tensor) -> None:
        self.x = inputs
        self.y = targets
        
    def cuda(self):
        return Batch(self.x.cuda(), self.y.cuda())
    
    def cpu(self):
        return Batch(self.x.cpu(), self.y.cpu())

    def to(self, device):
        if device.type == 'cuda':
            return self.cuda()
        elif device.type == 'cpu':
            return self.cpu()
        else:
            raise NotImplementedError


class H2ODataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        raw_data = np.load(root + 'h2o.npz')
        data_list = []
        for i in range(raw_data['R'].shape[0]):
            data_list.append(
                Data(
                    Z=torch.tensor(raw_data['Z'][i]).to(torch.long),
                    energy=torch.tensor(raw_data['E'][i]).to(torch.float),
                    force=torch.tensor(raw_data['F'][i]).to(torch.float),
                    pos=torch.tensor(raw_data['R'][i]).to(torch.float)
                )
            )
        self.data_list = data_list
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_file_names(self):
        return 'h2o.pt'

    def process(self):
        torch.save(self.collate(self.data_list), self.processed_paths[0])


class DimeNetQM9(Dataset):
    def __init__(self, inputs, targets):
        super(DimeNetQM9, self).__init__()
        self.inputs = inputs
        self.targets = targets
    
    def index_select(self, indices):
        return DimeNetQM9(self.inputs[indices], self.targets[indices])

    def index_select_(self, indices):
        self.inputs = self.inputs[indices]
        self.targets = self.targets[indices]
        
    def __len__(self):
        return len(self.targets)
    
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
    
def collate_batching(data):
    inputs, targets = default_collate(data)
    batch = Batch(inputs, targets)
    if torch.cuda.is_available():
        batch = batch.cuda()
    return batch

def collate_targets(data):
    # make sure that the target is always called energy and forces
    pass