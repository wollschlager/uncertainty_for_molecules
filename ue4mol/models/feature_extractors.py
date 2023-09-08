from ue4mol.models.backbones import *
from ue4mol.models.interfaces import DimeNetBackbone, SchNetBackbone
from ue4mol.utils import *
import re
from typing import Tuple
import torch
from torch.nn import Linear, Identity


class DimeNetPPDropout(DimeNetBackbone):
    def __init__(self, dimenet_pp_params: dict):
        super(DimeNetPPDropout, self).__init__()
        self.is_encoder = False

    def initialize_model(self, dimenet_pp_params):
        self.model = DimenetDropoutBase(**dimenet_pp_params)

                

class DimeNetPPMultiDropout(DimeNetBackbone):
    def __init__(self, dimenet_pp_params):
        super(DimeNetPPMultiDropout, self).__init__()
        self.is_encoder = False
        self.initialize_model
    
    def initialize_model(self, dimenet_pp_params):
        self.model = DimeNetPPDropoutMultiBase(dimenet_pp_params)
                
                
class SchNetDropout(SchNetBackbone):
    def __init__(self, params):
        super(SchNetDropout, self).__init__()
        assert "drop_prob" in params, "You did not specify a dropout probability for SchNet Dropout"
        self.model = SchNetDropoutBase(**params)
        self.is_encoder = False
        

class SchNetMultDropout(SchNetBackbone):
    def __init__(self, params):
        super(SchNetMultDropout, self).__init__()
        self.is_encoder = False
        
    def initialize_model(self, params):
        self.model = SchNetDropoutBase(**params, localized=True)
        
                
class NequIPMultiBase(NequIP):

    def __init__(self, params):
        super(NequIPMultiBase, self).__init__(**params)
        
    def forward(self, batch):
        input_data = to_atomic_data(create_missing_keys(batch, self.type_mapper))
        our_input_data = {k: v for k, v in input_data.items()}
        return self.model(our_input_data)['atomic_energy']
    
class NequIPMulti(LightningInterface, BaseModel):

    def __init__(self, params):
        super(NequIPMulti, self).__init__()
        self.model = NequIPMultiBase(params)
        self.is_encoder = False
        
    def forward(self, batch):        
        return self.model(batch)
   
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
            
        print(self.model.model.load_state_dict(state_dict))
        
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
    
    def freeze(self, param_list=None):
        for name, param in self.model.named_parameters():
            if param_list is None:
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False