from os.path import join
import torch
from torch.nn import Identity
from torch_geometric.nn import radius_graph

from ..interfaces import BaseModel

from nequip.model import model_from_config
from nequip.utils import Config
from nequip.data import dataset_from_config
from nequip.train import Trainer


# we just use the normal code from nequip (see https://github.com/mir-group/nequip)
# only build wrapper around the batch for simple interface
def nequip_state_dict_map(state_dict):
    new_state_dict = {}
    for k in state_dict:
        if k[:11] == "model.func.":
            new_state_dict['model.'+k[11:]] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    return new_state_dict


def create_missing_keys(batch, type_mapper):
    device = batch.pos.device
    r_max = 4.
    batch_size = batch.batch.max() + 1
    edge_index = radius_graph(batch.pos, r_max, batch=batch.batch).to(device)
    batch.edge_index = edge_index.to(device)
    batch.r_max = torch.ones((batch_size), device=device) * r_max
    batch.z = type_mapper.transform(batch.z).to(device) 
    return batch


def to_atomic_data(data, exclude_keys=tuple()):
    keys = data.keys
    return {
            keyword_map(k): data[k]
            for k in keys
            if (
                keyword_map(k) not in exclude_keys
                and data[k] is not None
                and isinstance(data[k], torch.Tensor)
            )
        }
    
def keyword_map(k):
    if k in mapping:
        return mapping[k]
    return k


mapping = {
    "z": "atom_types",
    "energy": "total_energy"
}


model_builders = [
    'SimpleIrrepsConfig',
    'EnergyModel',
    'PerSpeciesRescale',
    'RescaleEnergyEtc'
]



class NequIP(BaseModel):
    def __init__(self, path, **kwargs):
        super().__init__()
        #path = path['path']
        print(path)
        _, model_config = Trainer.load_model_from_training_session(
            traindir=path
        )
        model_r_max = model_config["r_max"]
        dataset_config = Config.from_file(
                join(path, "config.yaml"), defaults={"r_max": model_r_max}
            )
        dataset = dataset_from_config(dataset_config)
        model_config.model_builders = model_builders
        self.model = model_from_config(
            config=model_config,
            initialize=True,
            dataset=dataset
        )
        self.model_config = model_config
        self.type_mapper = dataset.type_mapper
        
    def forward(self, batch, pure_energy=True):
        input_data = to_atomic_data(create_missing_keys(batch, self.type_mapper))
        our_input_data = {k: v for k, v in input_data.items()}
        if pure_energy:
            return self.model(our_input_data)['total_energy']
        else: 
            return self.model(our_input_data)
    
    def load(self, file):
        state_dict = torch.load(file)
        if 'ema_state_dict' in state_dict:
            print('loading ema state dict')
            state_dict = nequip_state_dict_map(state_dict['ema_state_dict'])
        elif 'state_dict' in state_dict:
            state_dict = nequip_state_dict_map(state_dict['state_dict'])
        else:
            state_dict = nequip_state_dict_map(state_dict)
        state_dict['scale_by'].unsqueeze_(0)
            
        print(self.model.load_state_dict(state_dict))
        
    def to_encoder(self, output_dim=None):
        if output_dim is None or output_dim == 16:
            self.model.model.output_hidden_to_scalar.linear = Identity()
            self.model.model.per_species_rescale = Identity()
        elif output_dim == 256:
            self.model.model.output_hidden_to_scalar.linear = Identity()
            self.model.model.conv_to_output_hidden.linear = Identity()
            self.model.model.per_species_rescale = Identity()
        elif output_dim == 1:
            pass
        else:
            raise NotImplementedError(f"for this output dim {output_dim} encoder is not implemented")
        self.is_encoder = True
    
    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False

    
class NequIPMulti(NequIP):
    def __init__(self, params):
        super(NequIPMulti, self).__init__(params)
        
    def forward(self, batch):        
        return self.model(batch)
   
    def _forward(self, batch):
        input_data = to_atomic_data(create_missing_keys(batch, self.type_mapper))
        our_input_data = {k: v for k, v in input_data.items()}
        return self.model(our_input_data)['atomic_energy']
        
    def to_encoder(self, output_dim=None):
        if output_dim is None or output_dim == 16:
            self.model.model.model.output_hidden_to_scalar.linear = Identity()
            self.model.model.model.per_species_rescale = Identity()
            self.model.model.model.total_energy_sum = Identity()

        elif output_dim == 256:
            self.model.model.model.output_hidden_to_scalar.linear = Identity()
            self.model.model.model.conv_to_output_hidden.linear = Identity()
            self.model.model.model.per_species_rescale = Identity()
            self.model.model.model.total_energy_sum = Identity()

        elif output_dim == 1:
            pass
        else:
            raise NotImplementedError(f"for this output dim {output_dim} encoder is not implemented")
        self.is_encoder = True
                


