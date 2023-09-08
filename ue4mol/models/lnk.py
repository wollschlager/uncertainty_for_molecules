import gpytorch
import torch
from torch.autograd import grad
from .utils import load_localized_backbone, calc_inducing_points, init_kernel
from torch_scatter import scatter
from gpytorch.distributions import MultivariateNormal

from .variational_strategy import FixedIndPoint_VariationalStrategy
from .gp_base import Standard_GPModel

class LNK(torch.nn.Module):
    def __init__(self, gp_params, backbone_params, train_loader):
        super(LNK, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = load_local_backbone(backbone_params)
        encoder.to_encoder()
        
        if gp_params['freeze_encoder']:
            encoder.freeze()
            encoder.eval()
        else:
            encoder.train()

        inducing_points, _, initial_lengthscale = calc_inducing_points(
            model=encoder, 
            train_loader=train_loader, 
            type=gp_params['inducing_point_initialization'], 
            n_inducing_points=gp_params['n_inducing_points']
        )
        
        kernel = init_kernel(gp_params['kernel_name'], gp_params['kernel_params'])
        kernel.lengthscale = initial_lengthscale.to(device) * torch.ones_like(kernel.lengthscale, device=device)
        covar_module = gpytorch.kernels.ScaleKernel(kernel)
        
        if gp_params['fixed_ind_point']:
            single_gp = Fixed_IndPoint_GPModel(inducing_points, covar_module).to(device)
        else:
            single_gp = Standard_GPModel(inducing_points, covar_module).to(device)
        
        gp = MultiGPModel(single_gp, **gp_params)
        likelihood = gpytorch.likelihoods.GaussianLikelihood().to(device)

        self.encoder = encoder
        self.gp = gp
        self.likelihood = likelihood
        
    def forward(self, x):
        x.pos.requires_grad = True
        embedding = self.encoder(x)
        energy = self.gp(embedding)
        forces = - grad(
                    outputs=energy.mean.sum(),
                    inputs=x.pos,
                    create_graph=True,
                    retain_graph=True
                )[0]
        x.pos.requires_grad = False
        return energy, forces


class MultiGPModel(torch.nn.Module):
    def __init__(self, gp, reduce_mean="sum", reduce_covar="sum", full_covar=False, **kwargs):
        super().__init__()
        self.gp = gp
        self.reduce_mean = reduce_mean
        self.reduce_covar = reduce_covar
        self.full_covar = full_covar
        self.full_operator = None
        self.batch_max = 0
        self.batch_length = 0
        
    def forward(self, embeddings, batch=None):
        pred = self.gp(embeddings)
        if batch is not None: #if batch is none we are in the evaluation mode for the forces
            if self.full_covar:
                combined_mean, combined_covar = self.aggregate_covar(pred, batch)
            else:
                mean, covar = pred.mean, pred.covariance_matrix
                combined_mean = scatter(mean, batch, 0, reduce=self.reduce_mean)
                combined_covar = torch.eye(max(batch) + 1).to(covar) * scatter(covar.diag(), batch, 0, reduce=self.reduce_covar)
            pred = MultivariateNormal(combined_mean, combined_covar)
        return pred
        
    def aggregate_covar(self, energy_dist, batch):
        if self.batch_max == batch.max() and self.batch_length == len(batch) and self.full_operator is not None:
            operator = self.full_operator
        else:
            self.batch_max = batch.max()
            self.batch_length = len(batch)
            operator = torch.zeros(batch.max() + 1, len(batch))
            for i in range(batch.max() + 1):
                operator[i, batch == i] += 1
            operator = operator.to(energy_dist.loc)
            self.full_operator = operator
        mean = operator @ energy_dist.loc
        covar = operator @ energy_dist.covariance_matrix @ operator.transpose(1, 0)
        return mean, covar
    
    
class Fixed_IndPoint_GPModel(gpytorch.models.ApproximateGP):
    def __init__(self, inducing_points, covar_module, device=torch.device("cuda")):
        variational_distribution = gpytorch.variational.CholeskyVariationalDistribution(inducing_points.size(0) + 1)
        variational_strategy = FixedIndPoint_VariationalStrategy(
            self, inducing_points, variational_distribution, learn_inducing_locations=True, device=device
        )
        super(Fixed_IndPoint_GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = covar_module

    def forward(self, x, batch=None):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)