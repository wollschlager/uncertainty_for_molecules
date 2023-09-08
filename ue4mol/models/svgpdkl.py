import gpytorch
import torch
from torch.autograd import grad
from utils import load_global_backbone, calc_inducing_points, init_kernel

from models.gp_base import Standard_GPModel

class SVGPDKL(torch.nn.Module):
    def __init__(self, gp_params, backbone_params, train_loader):
        super(SVGPDKL, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        encoder = load_global_backbone(backbone_params)
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
        gp = Standard_GPModel(inducing_points, covar_module).to(device)
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