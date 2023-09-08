from math import pi as PI
from math import sqrt
from typing import Callable

import torch
from torch.nn import Embedding, Linear, Dropout, Identity
from torch_scatter import scatter
from torch_sparse import SparseTensor

from torch_geometric.nn import radius_graph

from torch_geometric.nn.acts import swish
from torch_geometric.nn.inits import glorot_orthogonal
import re
from ue4mol.models.interfaces import BaseModel

qm9_target_dict = {
    0: 'mu',
    1: 'alpha',
    2: 'homo',
    3: 'lumo',
    5: 'r2',
    6: 'zpve',
    7: 'U0',
    8: 'U',
    9: 'H',
    10: 'G',
    11: 'Cv',
}


class Envelope(torch.nn.Module):
    def __init__(self, exponent):
        super().__init__()
        self.p = exponent + 1
        self.a = -(self.p + 1) * (self.p + 2) / 2
        self.b = self.p * (self.p + 2)
        self.c = -self.p * (self.p + 1) / 2

    def forward(self, x):
        p, a, b, c = self.p, self.a, self.b, self.c
        x_pow_p0 = x.pow(p - 1)
        x_pow_p1 = x_pow_p0 * x
        x_pow_p2 = x_pow_p1 * x
        return (1. / x + a * x_pow_p0 + b * x_pow_p1 + c * x_pow_p2) * (x < 1.0).to(x.dtype)


class BesselBasisLayer(torch.nn.Module):
    def __init__(self, num_radial, cutoff=5.0, envelope_exponent=5):
        super().__init__()
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        self.freq = torch.nn.Parameter(torch.Tensor(num_radial))

        self.reset_parameters()

    def reset_parameters(self):
        with torch.no_grad():
            torch.arange(1, self.freq.numel() + 1, out=self.freq).mul_(PI)
        self.freq.requires_grad_()

    def forward(self, dist):
        dist = (dist.unsqueeze(-1) / self.cutoff)
        return self.envelope(dist) * (self.freq * dist).sin()


class SphericalBasisLayer(torch.nn.Module):
    def __init__(self, num_spherical, num_radial, cutoff=5.0,
                 envelope_exponent=5):
        super().__init__()
        import sympy as sym

        from torch_geometric.nn.models.dimenet_utils import (bessel_basis,
                                                             real_sph_harm)

        assert num_radial <= 64
        self.num_spherical = num_spherical
        self.num_radial = num_radial
        self.cutoff = cutoff
        self.envelope = Envelope(envelope_exponent)

        bessel_forms = bessel_basis(num_spherical, num_radial)
        sph_harm_forms = real_sph_harm(num_spherical)
        self.sph_funcs = []
        self.bessel_funcs = []

        x, theta = sym.symbols('x theta')
        modules = {'sin': torch.sin, 'cos': torch.cos}
        for i in range(num_spherical):
            if i == 0:
                sph1 = sym.lambdify([theta], sph_harm_forms[i][0], modules)(0)
                self.sph_funcs.append(lambda x: torch.zeros_like(x) + sph1)
            else:
                sph = sym.lambdify([theta], sph_harm_forms[i][0], modules)
                self.sph_funcs.append(sph)
            for j in range(num_radial):
                bessel = sym.lambdify([x], bessel_forms[i][j], modules)
                self.bessel_funcs.append(bessel)

    def forward(self, dist, angle, idx_kj):
        dist = dist / self.cutoff
        rbf = torch.stack([f(dist) for f in self.bessel_funcs], dim=1)
        rbf = self.envelope(dist).unsqueeze(-1) * rbf

        cbf = torch.stack([f(angle) for f in self.sph_funcs], dim=1)

        n, k = self.num_spherical, self.num_radial
        out = (rbf[idx_kj].view(-1, n, k) * cbf.view(-1, n, 1)).view(-1, n * k)
        return out


class EmbeddingBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, act=swish, drop_prob=0.2):
        super().__init__()
        self.act = act

        self.emb = Embedding(95, hidden_channels)
        self.lin_rbf = Linear(num_radial, hidden_channels)
        self.lin = Linear(3 * hidden_channels, hidden_channels)
        
        self.dropout = Dropout(p=drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        self.emb.weight.data.uniform_(-sqrt(3), sqrt(3))
        
        # add glorot init
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        self.lin_rbf.bias.data.fill_(0)
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        
        #self.lin_rbf.reset_parameters()
        #self.lin.reset_parameters()

    def forward(self, x, rbf, i, j):
        x = self.emb(x)
        rbf = self.act(self.lin_rbf(rbf))
        x = self.act(self.lin(torch.cat([x[i], x[j], rbf], dim=-1)))
        x = self.dropout(x) # DROPOUT
        return x


class ResidualLayer(torch.nn.Module):
    def __init__(self, hidden_channels, act=swish, drop_prob=0.2):
        super().__init__()
        self.act = act
        self.lin1 = Linear(hidden_channels, hidden_channels)
        self.lin2 = Linear(hidden_channels, hidden_channels)
        
        self.dropout = Dropout(p=drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin1.weight, scale=2.0)
        self.lin1.bias.data.fill_(0)
        glorot_orthogonal(self.lin2.weight, scale=2.0)
        self.lin2.bias.data.fill_(0)

    def forward(self, x):
        x_trans = self.act(self.lin1(x))
        x_trans = self.dropout(x_trans) # DROPOUT
        x_trans = self.act(self.lin2(x_trans))
        x_trans = self.dropout(x_trans) # DROPOUT
        return x + x_trans


class InteractionBlock(torch.nn.Module):
    def __init__(self, hidden_channels, int_emb_size, basis_emb_size, num_spherical,
                 num_radial, num_before_skip, num_after_skip, act=swish, drop_prob=0.2):
        super().__init__()
        self.act = act

        self.lin_rbf1 = Linear(num_radial, basis_emb_size, bias=False)
        self.lin_rbf2 = Linear(basis_emb_size, hidden_channels, bias=False)
        self.lin_sbf1 = Linear(num_spherical * num_radial, basis_emb_size, bias=False)
        self.lin_sbf2 = Linear(basis_emb_size, int_emb_size, bias=False)

        # Dense transformations of input messages.
        self.lin_kj = Linear(hidden_channels, hidden_channels)
        self.lin_ji = Linear(hidden_channels, hidden_channels)
        
        # Embedding projections for interaction triplets
        self.down_projection = Linear(hidden_channels, int_emb_size, bias=False)
        self.up_projection = Linear(int_emb_size, hidden_channels, bias=False)

        self.layers_before_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act, drop_prob=drop_prob) for _ in range(num_before_skip)
        ])
        self.lin = Linear(hidden_channels, hidden_channels)
        self.layers_after_skip = torch.nn.ModuleList([
            ResidualLayer(hidden_channels, act, drop_prob=drop_prob) for _ in range(num_after_skip)
        ])
        
        self.dropout = Dropout(p=drop_prob)

        self.reset_parameters()

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_rbf2.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf1.weight, scale=2.0)
        glorot_orthogonal(self.lin_sbf2.weight, scale=2.0)
        
        glorot_orthogonal(self.lin_kj.weight, scale=2.0)
        self.lin_kj.bias.data.fill_(0)
        glorot_orthogonal(self.lin_ji.weight, scale=2.0)
        self.lin_ji.bias.data.fill_(0)
        
        glorot_orthogonal(self.down_projection.weight, scale=2.0)
        glorot_orthogonal(self.up_projection.weight, scale=2.0)
        
        for res_layer in self.layers_before_skip:
            res_layer.reset_parameters()
        glorot_orthogonal(self.lin.weight, scale=2.0)
        self.lin.bias.data.fill_(0)
        for res_layer in self.layers_after_skip:
            res_layer.reset_parameters()

    def forward(self, x, rbf, sbf, idx_kj, idx_ji):
        rbf = self.lin_rbf1(rbf)
        rbf = self.lin_rbf2(rbf)
        sbf = self.lin_sbf1(sbf)
        sbf = self.lin_sbf2(sbf)

        x_ji = self.act(self.lin_ji(x))
        x_ji = self.dropout(x_ji) # DROPOUT
        
        x_kj = self.act(self.lin_kj(x))
        x_kj = x_kj * rbf
        
        x_kj = self.dropout(x_kj) # DROPOUT
        
        x_kj = self.act(self.down_projection(x_kj))
        x_kj = x_kj[idx_kj] * sbf
        
        x_kj = self.dropout(x_kj) # DROPOUT
        
        x_kj = scatter(x_kj, idx_ji, dim=0, dim_size=x.size(0))
        x_kj = self.act(self.up_projection(x_kj))
        
        x_kj = self.dropout(x_kj) # DROPOUT

        h = x_ji + x_kj
        for layer in self.layers_before_skip:
            h = layer(h)
        h = self.act(self.lin(h))
        
        h = self.dropout(h) # DROPOUT
        
        h = h + x
        for layer in self.layers_after_skip:
            h = layer(h)

        return h


class ScaleShift(torch.nn.Module):
    r"""Scale and shift layer for standardization.
    .. math::
       y = x \times \sigma + \mu
    Args:
        mean (torch.Tensor): mean value :math:`\mu`.
        stddev (torch.Tensor): standard deviation value :math:`\sigma`.
    """

    def __init__(self, mean, stddev):
        super(ScaleShift, self).__init__()
        self.mean = mean
        self.stddev = stddev

    def forward(self, input):
        """Compute layer output.
        Args:
            input (torch.Tensor): input data.
        Returns:
            torch.Tensor: layer output.
        """
        y = input * self.stddev + self.mean
        return y

    
class OutputBlock(torch.nn.Module):
    def __init__(self, num_radial, hidden_channels, out_emb_size, out_channels, num_layers,
                 act=swish, drop_prob=0.2):
        super().__init__()
        self.act = act

        self.lin_rbf = Linear(num_radial, hidden_channels, bias=False)
        
        self.up_projection = Linear(hidden_channels, out_emb_size, bias=False)
        
        self.lins = torch.nn.ModuleList()
        for _ in range(num_layers):
            self.lins.append(Linear(out_emb_size, out_emb_size))
        self.lin = Linear(out_emb_size, out_channels, bias=False)

        self.reset_parameters()
        self.encoder = False
        self.dropout = Dropout(p=drop_prob)

    def reset_parameters(self):
        glorot_orthogonal(self.lin_rbf.weight, scale=2.0)
        glorot_orthogonal(self.up_projection.weight, scale=2.0)
        for lin in self.lins:
            glorot_orthogonal(lin.weight, scale=2.0)
            lin.bias.data.fill_(0)
        self.lin.weight.data.fill_(0)

    def to_encoder(self):
        self.encoder = True

    def forward(self, x, rbf, i, num_nodes=None):
        g = self.lin_rbf(rbf)
        x = g * x
        x = scatter(x, i, dim=0, dim_size=num_nodes)
        
        x = self.up_projection(x)
        
        num_lins = len(self.lins) - 1
        for idx, lin in enumerate(self.lins):
            if self.encoder:
                x = lin(x)
                if idx < num_lins:
                    x = self.act(x)
            else:
                x = self.act(lin(x))
                x = self.dropout(x) # DROPOUT
        return self.lin(x)


class DimeNetPPDropout(BaseModel):
    r"""
    
    Args:
        hidden_channels (int): Hidden embedding size.
        out_channels (int): Size of each output sample.
        num_blocks (int): Number of building blocks.
        num_bilinear (int): Size of the bilinear layer tensor.
        num_spherical (int): Number of spherical harmonics.
        num_radial (int): Number of radial basis functions.
        ------ Dimenet++ specific parameters -----
        int_emb_size (int): Embedding size used for interaction triplets.
        basis_emb_size (int): Embedding size used inside the basis transformation.
        out_emb_size (int): Embedding size used for atoms in the output block
        ------------------------------------------
        cutoff: (float, optional): Cutoff distance for interatomic
            interactions. (default: :obj:`5.0`)
        max_num_neighbors (int, optional): The maximum number of neighbors to
            collect for each node within the :attr:`cutoff` distance.
            (default: :obj:`32`)
        envelope_exponent (int, optional): Shape of the smooth cutoff.
            (default: :obj:`5`)
        num_before_skip: (int, optional): Number of residual layers in the
            interaction blocks before the skip connection. (default: :obj:`1`)
        num_after_skip: (int, optional): Number of residual layers in the
            interaction blocks after the skip connection. (default: :obj:`2`)
        num_output_layers: (int, optional): Number of linear layers for the
            output blocks. (default: :obj:`3`)
        act: (Callable, optional): The activation funtion.
            (default: :obj:`swish`)
    """

    url = ('https://github.com/klicperajo/dimenet/raw/master/pretrained/'
           'dimenet')

    def __init__(self, hidden_channels: int, out_channels: int,
                 num_blocks: int, num_bilinear: int, num_spherical: int,
                 num_radial, int_emb_size: int = 64, basis_emb_size: int = 8,
                 out_emb_size: int = 256, cutoff: float = 5.0, 
                 max_num_neighbors: int = 32,
                 envelope_exponent: int = 5, num_before_skip: int = 1,
                 num_after_skip: int = 2, num_output_layers: int = 3,
                 act: Callable = swish, predict_forces: bool = False,
                 mean=None, stddev=None, drop_prob: float = 0.0, is_inference: bool = False, **kwargs):
        super().__init__()

        self.cutoff = cutoff
        self.max_num_neighbors = max_num_neighbors
        self.num_blocks = num_blocks
        
        self.predict_forces = predict_forces
        self.drop_prob = drop_prob
        self.is_inference = is_inference
        
        #mean = torch.FloatTensor([0.0]) if mean is None else mean
        #stddev = torch.FloatTensor([1.0]) if stddev is None else stddev

        self.rbf = BesselBasisLayer(num_radial, cutoff, envelope_exponent)
        self.sbf = SphericalBasisLayer(num_spherical, num_radial, cutoff,
                                       envelope_exponent)

        self.emb = EmbeddingBlock(num_radial, hidden_channels, act, drop_prob=self.drop_prob)

        self.output_blocks = torch.nn.ModuleList([
            OutputBlock(num_radial, hidden_channels, out_emb_size, out_channels,
                        num_output_layers, act, drop_prob=self.drop_prob) for _ in range(num_blocks + 1)
        ])

        self.interaction_blocks = torch.nn.ModuleList([
            InteractionBlock(hidden_channels, int_emb_size, basis_emb_size, num_spherical,
                             num_radial, num_before_skip, num_after_skip, act, drop_prob=self.drop_prob)
            for _ in range(num_blocks)
        ])

        # build standardization layer
        #self.standardize = ScaleShift(mean, stddev)
        
        self.reset_parameters()

    def reset_parameters(self):
        self.rbf.reset_parameters()
        self.emb.reset_parameters()
        for out in self.output_blocks:
            out.reset_parameters()
        for interaction in self.interaction_blocks:
            interaction.reset_parameters()

            
    def triplets(self, edge_index, num_nodes):
        row, col = edge_index  # j->i

        value = torch.arange(row.size(0), device=row.device)
        adj_t = SparseTensor(row=col, col=row, value=value,
                             sparse_sizes=(num_nodes, num_nodes))
        adj_t_row = adj_t[row]
        num_triplets = adj_t_row.set_value(None).sum(dim=1).to(torch.long)

        # Node indices (k->j->i) for triplets.
        idx_i = col.repeat_interleave(num_triplets)
        idx_j = row.repeat_interleave(num_triplets)
        idx_k = adj_t_row.storage.col()
        mask = idx_i != idx_k  # Remove i == k triplets.
        idx_i, idx_j, idx_k = idx_i[mask], idx_j[mask], idx_k[mask]

        # Edge indices (k-j, j->i) for triplets.
        idx_kj = adj_t_row.storage.value()[mask]
        idx_ji = adj_t_row.storage.row()[mask]

        return col, row, idx_i, idx_j, idx_k, idx_kj, idx_ji

    def forward(self, batch):
        return self._forward(batch.z, batch.pos, batch.batch)
    
    def _forward(self, z, pos, batch=None):
        """"""
        if self.predict_forces:
            torch.set_grad_enabled(True)
            pos.requires_grad = True
            
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_kj = pos[idx_j] - pos_i, pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)
            
        if self.predict_forces:
            create_graph = not self.is_inference
            F = -torch.autograd.grad(P, pos, grad_outputs=torch.ones_like(P), create_graph=create_graph, retain_graph=True)[0]
            pos.requires_grad = False
            P = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
            return (P, F)

        P = P.sum(dim=0) if batch is None else scatter(P, batch, dim=0)
        return P
    
    def to_encoder(self, output_dim=None):
        for output_block in self.output_blocks:
            if output_dim is None or output_dim == 256:
                output_block.lin = Identity()
                output_block.to_encoder()
            elif output_dim == 1:
                continue
            else:
                output_block.lin = Linear(256, output_dim)
        self.is_encoder = True

    def freeze(self, param_list=None):
        for name, param in self.named_parameters():
            if param_list is None:
                if self.is_encoder:
                    if name in re.findall('output_blocks\..\.lin\..*', name):
                        print('skipping freezing of parameter ', name)
                        continue
                param.requires_grad = False
            elif name in param_list:
                param.requires_grad = False


class DimeNetPPDropoutMulti(DimeNetPPDropout):
    def __init__(self, dimenet_pp_params: dict):
        super(DimeNetPPDropoutMulti, self).__init__(**dimenet_pp_params)

    def _forward(self, z, pos, batch=None):
        edge_index = radius_graph(pos, r=self.cutoff, batch=batch,
                                  max_num_neighbors=self.max_num_neighbors)

        i, j, idx_i, idx_j, idx_k, idx_kj, idx_ji = self.triplets(
            edge_index, num_nodes=z.size(0))

        # Calculate distances.
        dist = (pos[i] - pos[j]).pow(2).sum(dim=-1).sqrt()

        # Calculate angles.
        pos_i = pos[idx_i]
        pos_ji, pos_kj = pos[idx_j] - pos_i, pos[idx_k] - pos[idx_j]
        a = (pos_ji * pos_kj).sum(dim=-1)
        b = torch.cross(pos_ji, pos_kj).norm(dim=-1)
        angle = torch.atan2(b, a)

        rbf = self.rbf(dist)
        sbf = self.sbf(dist, angle, idx_kj)

        # Embedding block.
        x = self.emb(z, rbf, i, j)
        P = self.output_blocks[0](x, rbf, i, num_nodes=pos.size(0))

        # Interaction blocks.
        for interaction_block, output_block in zip(self.interaction_blocks,
                                                   self.output_blocks[1:]):
            x = interaction_block(x, rbf, sbf, idx_kj, idx_ji)
            P += output_block(x, rbf, i)

        return P