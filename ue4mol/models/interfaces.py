from torch.distributions import MultivariateNormal
import torch.nn as nn
from torch.nn import Linear, Identity
from torch.autograd import grad
import torch
from typing import Tuple
import abc
from ue4mol.utils import *

class LightningInterface(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        raise NotImplementedError
    

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def to_encoder(self):
        raise NotImplementedError

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False
    
    def unfreeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = True
            elif name in param_list:
                param.requires_grad = True
    
    def load(self, file):
        state_dict = torch.load(file)
        if 'ema_state_dict' in state_dict:
            print('loading ema state dict')
            state_dict = map_state_dict(state_dict['ema_state_dict'])
        elif 'state_dict' in state_dict:
            state_dict = map_state_dict(state_dict['state_dict'])
        print(self.load_state_dict(state_dict))



class BaseForceModel(LightningInterface):
    def __init__(self, model, rho_force=0.99):
        super().__init__()
        self.model = model
        self.rho_force = rho_force

    def forward(self, batch):
        batch.pos.requires_grad = True
        torch.set_grad_enabled(True)
        energy = self.model(batch)
        forces = -grad(
                    outputs=energy.sum(), 
                    inputs=batch.pos, 
                    create_graph=True,
                    retain_graph=True
                )[0]
        batch.pos.requires_grad = False
        return energy, forces
    
    def loss_fn(self, energy, forces, targets_energy, targets_forces):
        energy_loss = mae(energy, targets_energy)
        force_loss = mae(forces, targets_forces)
        return (1 - self.rho_force) * energy_loss + self.rho_force * force_loss
        
    def execution_step(self, batch: torch.Tensor) -> Tuple[torch.Tensor, dict]:
        energy, forces = self.forward(batch)
        energy = energy.reshape(-1)
        loss = self.loss_fn(energy, forces, batch.energy, batch.force)
        energy_mae = mae(energy, batch.energy)
        force_mae = mae(forces, batch.force)
        
        return loss, {
            'energy_mae': energy_mae, 
            'force_mae': force_mae
        }
        
        
class DimeNetBackbone(LightningInterface, BaseModel):
    def __init__(self):
        super().__init__()
        self._model = None
        
    @property
    def model(self):
        if self._model is None:
            raise ValueError('Model is not initialized')
        return self._model
    
    def initialize_model(self, dimenet_pp_params):
        pass

    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)

    def to_encoder(self, output_dim=None):
        for output_block in self.model.output_blocks:
            if output_dim is None or output_dim == 256:
                output_block.lin = Identity()
                output_block.to_encoder()
            elif output_dim == 1:
                continue
            else:
                output_block.lin = Linear(256, output_dim)
        self.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                if self.is_encoder:
                    if name in re.findall('output_blocks\..\.lin\..*', name):
                        print('skipping freezing of parameter ', name)
                        continue
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False


class SchNetBackbone(LightningInterface, BaseModel):
    def __init__(self, params):
        super().__init__()
        self._model = None
        
    @property
    def model(self):
        if self._model is None:
            raise ValueError('Model is not initialized')
        return self._model
    
    def initialize_model(self, params):
        pass
    
    def forward(self, batch):
        z, pos, b = batch.z, batch.pos, batch.batch
        return self.model(z, pos, b)
 
    def to_encoder(self, output_dim=None):
        self.model.lin2 = Identity()
        self.is_encoder = True
        self.model.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False
                
                
class NequIPBackbone(torch.nn.Module):
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

def nequip_state_dict_map(state_dict):
    new_state_dict = {}
    for k in state_dict:
        if k[:11] == "model.func.":
            new_state_dict['model.'+k[11:]] = state_dict[k]
        else:
            new_state_dict[k] = state_dict[k]
    return new_state_dict   
    
